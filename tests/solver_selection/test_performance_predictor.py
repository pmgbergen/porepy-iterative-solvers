import pytest
import numpy as np
from pp_solvers.solver_selection.performance_predictor import (
    assemble_default_performance_predictor,
    EPSGREEDY_EXPECTATION,
    FAIL_REWARD,
)


_SOLVER_FEATURES = np.eye(5)
_N_SOLVERS = _SOLVER_FEATURES.shape[0]


def _partial_fit_n(predictor, n, start_i=0):
    """Feed n results to the predictor, mixing successes and failures."""
    for i in range(n):
        predictor.partial_fit(
            features=_SOLVER_FEATURES[(start_i + i) % _N_SOLVERS],
            solve_time=1.0,
            construct_time=0.5,
            success=((start_i + i) % 3 != 0),
        )


@pytest.mark.parametrize(
    "num_initial_exploration",
    [
        pytest.param(5, id="num_initial_exploration=5"),
        pytest.param(10, id="num_initial_exploration=10"),
        pytest.param(20, id="num_initial_exploration=20"),
    ],
)
def test_num_initial_exploration(num_initial_exploration):
    """Tests that the predictor explores randomly for exactly `num_initial_exploration`
    solves before switching to ML-guided selection.
    """
    predictor = assemble_default_performance_predictor(
        num_initial_exploration=num_initial_exploration
    )

    assert predictor.num_initial_exploration == num_initial_exploration

    _partial_fit_n(predictor, num_initial_exploration - 1)
    _, _, is_model_prediction = predictor.select_solver(_SOLVER_FEATURES)
    assert not is_model_prediction

    _partial_fit_n(predictor, 1, start_i=num_initial_exploration - 1)
    _, _, is_model_prediction = predictor.select_solver(_SOLVER_FEATURES)
    assert is_model_prediction


@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(5, id="batch_size=5"),
        pytest.param(10, id="batch_size=10"),
        pytest.param(20, id="batch_size=20"),
    ],
)
def test_batch_size(batch_size):
    """Tests that after the initial exploration phase, the model updates in batches of
    `batch_size` results.
    """
    num_initial = 6
    predictor = assemble_default_performance_predictor(
        num_initial_exploration=num_initial,
        batch_size=batch_size,
    )

    assert predictor.batch_size == batch_size

    _partial_fit_n(predictor, num_initial)
    assert predictor.is_ready_to_predict

    _partial_fit_n(predictor, batch_size - 1, start_i=num_initial)
    assert len(predictor.y_history) == batch_size - 1

    _partial_fit_n(predictor, 1, start_i=num_initial + batch_size - 1)
    assert len(predictor.y_history) == 0


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"eps": 0.0, "explores": False}, id="eps=0, no exploration"),
        pytest.param({"eps": 1.0, "explores": True}, id="eps=1, always explore"),
    ],
)
def test_eps(params):
    """Tests that `eps` controls whether epsilon-greedy exploration happens after the
    initial phase.
    """
    eps = params["eps"]
    explores = params["explores"]
    num_initial = 6

    predictor = assemble_default_performance_predictor(
        num_initial_exploration=num_initial, eps=eps
    )

    assert predictor.model.eps == eps

    _partial_fit_n(predictor, num_initial)
    _, expected_reward, _ = predictor.select_solver(_SOLVER_FEATURES)

    if explores:
        assert expected_reward == EPSGREEDY_EXPECTATION
    else:
        assert expected_reward != EPSGREEDY_EXPECTATION


@pytest.mark.parametrize(
    "eps1",
    [
        pytest.param(0.9, id="eps1=0.9"),
        pytest.param(0.5, id="eps1=0.5"),
    ],
)
def test_eps1(eps1):
    """Tests that `eps1` is the factor by which `eps` decays after each exploration
    step.
    """
    num_initial = 6
    eps_initial = 1.0

    predictor = assemble_default_performance_predictor(
        num_initial_exploration=num_initial, eps=eps_initial, eps1=eps1
    )

    assert predictor.model.eps1 == eps1

    _partial_fit_n(predictor, num_initial)

    # eps=1.0 guarantees exploration; verify eps decays by eps1
    predictor.select_solver(_SOLVER_FEATURES)
    assert predictor.model.eps == pytest.approx(eps_initial * eps1)


def test_ml_model_predicts_failure_and_success():
    """Tests that with eps=0 the ML model learns to distinguish an always-failing solver
    from an always-succeeding one: it should select the successful solver and predict a
    reward clearly above FAIL_REWARD for it while predicting FAIL_REWARD - 1 for the
    failing solver.
    """
    num_initial = 20
    predictor = assemble_default_performance_predictor(
        num_initial_exploration=num_initial,
        batch_size=num_initial,
        eps=0,
    )

    success_features = _SOLVER_FEATURES[0]
    failure_features = _SOLVER_FEATURES[1]

    for _ in range(num_initial // 2):
        predictor.partial_fit(
            features=success_features, solve_time=0.5, construct_time=0.1, success=True
        )
        predictor.partial_fit(
            features=failure_features, solve_time=1.0, construct_time=0.5, success=False
        )

    assert predictor.is_ready_to_predict

    two_solvers = np.array([success_features, failure_features])
    chosen_idx, _, is_model_prediction = predictor.select_solver(two_solvers)

    assert is_model_prediction
    assert chosen_idx == 0  # success solver preferred

    rewards = predictor.model.predict(two_solvers)
    assert rewards[0] > FAIL_REWARD   # success solver: meaningful positive-ish reward
    assert rewards[1] < FAIL_REWARD   # failure solver: below the fail threshold
