"""Entry point for solver selection via the SolverSelector class."""

from __future__ import annotations

from pickle import dump, load
from time import time
from typing import Optional

import numpy as np

from pp_solvers.solver_selection.performance_predictor import (
    FAIL_REWARD,
    BaseIncrementalMLModel,
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

    def __init__(self) -> None:
        self.features: list[np.ndarray] = []
        self.reward: list[float] = []
        self.decision_idx: list[int] = []
        self.expectation: list[float] = []
        self.predict_time: list[float] = []
        self.fit_time: list[float] = []

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            dump(
                (
                    self.features,
                    self.reward,
                    self.decision_idx,
                    self.expectation,
                    self.predict_time,
                    self.fit_time,
                ),
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = load(f)
            self.features = data[0]
            self.reward = data[1]
            self.decision_idx = data[2]
            self.expectation = data[3]
            try:
                self.predict_time = data[4]
                self.fit_time = data[5]
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
        rewarder: Formula to compute reward.

    """

    def __init__(
        self,
        solver_space: SolverSpace,
        performance_predictor: BaseIncrementalMLModel,
        rewarder: Optional[Rewarder] = None,
    ) -> None:
        self.solver_space: SolverSpace = solver_space
        """Describes the avaliable options to choose from."""
        self.performance_predictor: BaseIncrementalMLModel = performance_predictor
        """The underlying ML model for selection."""
        if rewarder is None:
            rewarder = Rewarder()
        self.rewarder: Rewarder = rewarder
        """Formula to compute reward."""
        self.history: SolverSelectorHistory = SolverSelectorHistory()
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
        expectations = self.performance_predictor.predict(X=features)
        decision_idx = int(np.argmax(expectations))
        decision = self.solver_space.all_decisions_encoding[decision_idx]

        # Storing the decision internally to fetch it when feedback is provided.
        self.__decision_idx = decision_idx
        self.__expectation = expectations[decision_idx]
        self.__features = features[decision_idx].copy()

        # Build the human-readible config from the internal decision representation.
        config = self.solver_space.config_from_decision(decision=decision)
        assert isinstance(config, dict), "At this point, should be a dictionary."

        self.__predict_time = time() - t0
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
        # Storing statistics. It is done here and not in select_linear_solver_scheme, to
        # prevent a situation where select_linear_solver_scheme is called more than
        # once, and feedback for each decision is not provided.
        reward = self.rewarder.estimate_reward(
            solve_time=solve_time, construct_time=construct_time, success=success
        )
        self.history.decision_idx.append(self.__decision_idx)
        self.history.expectation.append(self.__expectation)
        self.history.features.append(self.__features)
        self.history.reward.append(reward)
        self.history.predict_time.append(self.__predict_time)

        # Giving feedback.
        self.performance_predictor.partial_fit(
            X=self.history.features[-1], y=np.array(reward, dtype=float)
        )
        self.history.fit_time.append(time() - t0)


class Rewarder:
    """Transforms `solve_time` and `construct_time` into the reward for the ML
    algorithm.

    """

    def __init__(self) -> None:
        # FAIL_REWARD - 1 here to not think about "less or equal" edge case.
        self.worst_known_reward: float = FAIL_REWARD - 1

    def estimate_reward(
        self, solve_time: float, construct_time: float, success: bool
    ) -> float:
        if success:
            reward = -np.log(construct_time + solve_time)
        else:
            reward = -2 * abs(self.worst_known_reward)
        return reward
