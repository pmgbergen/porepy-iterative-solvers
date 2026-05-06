import pytest
import numpy as np
from pp_solvers.solver_selection.selector import (
    SolverSelector,
    SolverSelectorHistory,
    concatenate_characteristics_solvers,
)
from pp_solvers.solver_selection.solver_space import SolverSpace, CategoricalChoices
from pp_solvers.solver_selection.performance_predictor import (
    assemble_default_performance_predictor,
)


_SPACE = SolverSpace({"pc_type": CategoricalChoices(["ilu", "gamg", "hypre"])})
_ALL_CONFIGS = [_SPACE.config_from_decision(d) for d in _SPACE.all_decisions_encoding]
_NUM_SOLVERS = len(_SPACE.all_decisions_encoding)


def _make_selector(num_initial_exploration=4):
    predictor = assemble_default_performance_predictor(
        num_initial_exploration=num_initial_exploration
    )
    return SolverSelector(solver_space=_SPACE, performance_predictor=predictor)


def _run_cycles(selector, n, solve_time=1.0, construct_time=0.5, success=True):
    """Run n select+feedback cycles, return the list of returned decision indices."""
    indices = []
    for _ in range(n):
        _, idx = selector.select_linear_solver_scheme(
            characteristics=np.array([0.5]),
            active_solver_idx=None,
        )
        indices.append(idx)
        selector.provide_performance_feedback(
            solve_time=solve_time,
            construct_time=construct_time,
            success=success,
        )
    return indices


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {
                "num_characteristics": 2,
                "num_solvers": 4,
                "num_features": 3,
                "solver_in_use_idx": None,
            },
            id="no active solver",
        ),
        pytest.param(
            {
                "num_characteristics": 2,
                "num_solvers": 4,
                "num_features": 3,
                "solver_in_use_idx": 0,
            },
            id="active solver at first index",
        ),
        pytest.param(
            {
                "num_characteristics": 2,
                "num_solvers": 4,
                "num_features": 3,
                "solver_in_use_idx": 3,
            },
            id="active solver at last index",
        ),
    ],
)
def test_concatenate_characteristics_solvers(params):
    """Tests the shape, characteristics broadcast, and solver-in-use flag of the
    concatenated feature matrix.
    """
    num_characteristics = params["num_characteristics"]
    num_solvers = params["num_solvers"]
    num_features = params["num_features"]
    solver_in_use_idx = params["solver_in_use_idx"]

    characteristics = np.arange(num_characteristics, dtype=float)
    solvers = np.ones((num_solvers, num_features))

    result = concatenate_characteristics_solvers(
        characteristics=characteristics,
        solvers=solvers,
        solver_in_use_idx=solver_in_use_idx,
    )

    assert result.shape == (num_solvers, num_characteristics + 1 + num_features)

    # Characteristics are broadcast identically to every row
    for row in result[:, :num_characteristics]:
        np.testing.assert_array_equal(row, characteristics)

    # Flag column is zero everywhere unless solver_in_use_idx is given
    flag_col = result[:, num_characteristics]
    if solver_in_use_idx is None:
        assert np.all(flag_col == 0)
    else:
        assert flag_col[solver_in_use_idx] == 1
        assert np.sum(flag_col) == 1

    # Solver encodings are preserved unchanged
    np.testing.assert_array_equal(result[:, num_characteristics + 1 :], solvers)


def test_select_linear_solver_scheme_returns_valid_config():
    """Tests that select_linear_solver_scheme returns a config dict in the solver space
    and a valid decision index.
    """
    selector = _make_selector()
    config, decision_idx = selector.select_linear_solver_scheme(
        characteristics=np.array([0.5]),
        active_solver_idx=None,
    )

    assert isinstance(config, dict)
    assert config in _ALL_CONFIGS
    assert 0 <= decision_idx < _NUM_SOLVERS


@pytest.mark.parametrize(
    "n_cycles",
    [
        pytest.param(1, id="1 cycle"),
        pytest.param(3, id="3 cycles"),
        pytest.param(5, id="5 cycles"),
    ],
)
def test_provide_performance_feedback_updates_history(n_cycles):
    """Tests that each select+feedback cycle appends one entry to every history list,
    and that the recorded decision indices match those returned by select.
    """
    selector = _make_selector()
    returned_indices = _run_cycles(selector, n_cycles)

    history = selector.history
    assert len(history.decision_idx) == n_cycles
    assert len(history.reward) == n_cycles
    assert len(history.greedy) == n_cycles
    assert len(history.features) == n_cycles
    assert len(history.expectation) == n_cycles
    assert len(history.predict_time) == n_cycles
    assert len(history.fit_time) == n_cycles
    assert history.decision_idx == returned_indices


def test_history_save_load(tmp_path):
    """Tests that the selector history can be saved to disk and loaded back intact."""
    selector = _make_selector()
    _run_cycles(selector, 3)

    path = str(tmp_path / "history.pkl")
    selector.history.save(path)

    loaded = SolverSelectorHistory()
    loaded.load(path)

    assert loaded.decision_idx == selector.history.decision_idx
    assert loaded.reward == selector.history.reward
    assert loaded.greedy == selector.history.greedy
    assert loaded.expectation == selector.history.expectation
    for orig, restored in zip(selector.history.features, loaded.features):
        np.testing.assert_array_equal(orig, restored)
