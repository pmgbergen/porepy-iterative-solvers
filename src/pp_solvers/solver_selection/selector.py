"""Entry point for solver selection via the SolverSelector class."""

from pickle import dump, load
from time import time

import numpy as np

from pp_solvers.solver_selection.performance_predictor import (
    InitialExplorationEstimator,
)
from pp_solvers.solver_selection.solver_space import SolverSpace


def concatenate_characteristics_solvers(
    characteristics: np.ndarray,
    solvers: np.ndarray,
    solver_in_use_idx: int | None = None,
) -> np.ndarray:
    """Concatenates the arrays of characteristics (e.g. CFL number) and encoded solver
    configurations, so the resulting array is arranged:
    `[characteristics, solvers, flag]`
    where `flag` denotes if the solver configuration is the one that has been used in
    the previous linear system.

    The `solver_in_use_idx` flag can be utilized to cut costs of re-construction a
    linear solver by using the previously constructed one. This optimization is however
    not implemented in PorePy, so you can ignore it by passing `None`.

    Parameters:
        characteristics: `shape=(num_char,)` one vector of characteristics for the
            current state of simulation.
        solvers: `shape=(num_solvers, num_features)` all the encoded solver
            configurations in the solver space.
        solver_in_use_idx: index in [0, num_solvers) of the configuration in use for
            the previous linear system.

    Returns: a 2D array of `shape=(num_solvers, num_char + num_features + 1)`.

    """
    solver_reused_flag = np.zeros((solvers.shape[0], 1))
    if solver_in_use_idx is not None:
        solver_reused_flag[solver_in_use_idx] = 1

    characteristics = np.broadcast_to(
        characteristics, (solvers.shape[0], characteristics.size)
    )
    return np.concatenate([characteristics, solver_reused_flag, solvers], axis=1)


class SolverSelectorHistory:
    """Stores the history of solver selection decisions. Used for statistics."""

    def __init__(self):
        self.features: list[np.ndarray] = []
        self.reward: list[float] = []
        self.decision_idx: list[int] = []
        self.greedy: list[bool] = []
        self.expectation: list[float] = []
        self.predict_time: list[float] = []
        self.fit_time: list[float] = []

    def save(self, path: str):
        with open(path, "wb") as f:
            dump(
                (
                    self.features,
                    self.reward,
                    self.decision_idx,
                    self.greedy,
                    self.expectation,
                    self.predict_time,
                    self.fit_time,
                ),
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            data = load(f)
            self.features = data[0]
            self.reward = data[1]
            self.decision_idx = data[2]
            self.greedy = data[3]
            self.expectation = data[4]
            try:
                self.predict_time = data[5]
                self.fit_time = data[6]
            except IndexError:
                pass


class SolverSelector:
    """The ML solver selector. Pass it characteristics of the current simulation state
    (e.g. CFL number), and it will return you the configuration to assemble a linear
    solver. Then provide feedback about the solver performance to improve its decision
    making process.

    Parameters:
        solver_space: Describes the avaliable options to choose from.
        performance_predictor: The underlying ML model for selection.

    """

    def __init__(
        self,
        solver_space: SolverSpace,
        performance_predictor: InitialExplorationEstimator,
    ):
        self.solver_space: SolverSpace = solver_space
        """Describes the avaliable options to choose from."""
        self.performance_predictor: InitialExplorationEstimator = performance_predictor
        """The underlying ML model for selection."""
        self.history = SolverSelectorHistory()
        """Struct to save the decision history."""

    def select_linear_solver_scheme(
        self, characteristics: np.ndarray, active_solver_idx: int | None
    ) -> tuple[dict, int]:
        """Pass the characteristics of the current simulation state (e.g. CFL number),
        and it will return you a configuration to assemble linear solver.

        Parameters:
            characteristics: `shape=(num_char,)` one vector of characteristics for the
                current state of simulation, e.g. CFL number, Peclet number.
            active_solver_idx: index of the solver configuration used for the previous
                linear system. This value is returned by a previous invocation of this
                function, see `decision_idx`. Enables an optimization to avoid
                re-building the same solver twice. Not supported in PorePy (yet?).

        Returns:
            - config: A solver configuration used to construct a linear solver.
            - decision_idx: An internal identifier of the chosen solver configuration.

        """
        t0 = time()
        # Features is a 2D array shape=(num_solvers, num_features) for each solver
        # configuration in the solver space. Second axis is concatenated
        # [characteristics, encoded_solver, solver_in_use_flag].
        features = concatenate_characteristics_solvers(
            characteristics=characteristics,
            solvers=self.solver_space.all_decisions_encoding,
            solver_in_use_idx=active_solver_idx,
        )
        # Making the ML decision here.
        decision_idx, expectation, greedy = self.performance_predictor.select_solver(
            features=features
        )
        decision = self.solver_space.all_decisions_encoding[decision_idx]

        # Storing the decision internally to fetch it when feedback is provided.
        self.__decision_idx = decision_idx
        self.__expectation = expectation
        self.__greedy = greedy
        self.__features = features[decision_idx].copy()
        self.__predict_time = time() - t0

        # Build the human-readible config from the internal decision representation.
        config = self.solver_space.config_from_decision(decision=decision)
        assert isinstance(config, dict), "At this point, should be a dictionary."
        return config, decision_idx

    def provide_performance_feedback(
        self, solve_time: float, construct_time: float, success: bool
    ) -> None:
        """Give feedback regarding the previously taken ML decision to improve further
        decisions.

        Parameters:
            solve_time: Time to solve the linear system, seconds.
            construct_time: Time to construct the linear solver, seconds.
            success: Whether the linear solve was successful.

        """
        t0 = time()
        # Storing statistics.
        reward = self.performance_predictor.rewarder.estimate_reward(
            solve_time=solve_time, construct_time=construct_time, success=success
        )
        self.history.decision_idx.append(self.__decision_idx)
        self.history.expectation.append(self.__expectation)
        self.history.greedy.append(self.__greedy)
        self.history.features.append(self.__features)
        self.history.reward.append(reward)
        self.history.predict_time.append(self.__predict_time)

        # Giving feedback.
        self.performance_predictor.partial_fit(
            features=self.history.features[-1],
            solve_time=solve_time,
            construct_time=construct_time,
            success=success,
        )
        self.history.fit_time.append(time() - t0)
