from pickle import dump, load
from time import time

import numpy as np

from pp_solvers.solver_selection.performance_predictor import (
    InitialExplorationEstimator,
)
from pp_solvers.solver_selection.solver_space import SolverSpace


def concatenate_characteristics_solvers(
    characteristics: np.ndarray, solvers: np.ndarray, solver_in_use_idx: int | None
) -> np.ndarray:
    solver_reused_flag = np.zeros((solvers.shape[0], 1))
    if solver_in_use_idx is not None:
        solver_reused_flag[solver_in_use_idx] = 1

    characteristics = np.broadcast_to(
        characteristics, (solvers.shape[0], characteristics.size)
    )
    return np.concatenate([characteristics, solver_reused_flag, solvers], axis=1)


class SolverSelectorHistory:
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
    def __init__(
        self,
        solver_space: SolverSpace,
        performance_predictor: InitialExplorationEstimator,
    ):
        self.solver_space: SolverSpace = solver_space
        self.performance_predictor: InitialExplorationEstimator
        self.performance_predictor = performance_predictor
        self.history = SolverSelectorHistory()

    def select_linear_solver_scheme(
        self, characteristics: np.ndarray, active_solver_idx: int | None
    ) -> tuple[dict, int]:
        t0 = time()
        features = concatenate_characteristics_solvers(
            characteristics=characteristics,
            solvers=self.solver_space.all_decisions_encoding,
            solver_in_use_idx=active_solver_idx,
        )
        decision_idx, expectation, greedy = self.performance_predictor.select_solver(
            features=features
        )
        decision = self.solver_space.all_decisions_encoding[decision_idx]

        # We 
        self.__decision_idx = decision_idx
        self.__expectation = expectation
        self.__greedy = greedy
        self.__features = features[decision_idx].copy()
        self.__predict_time = time() - t0

        config = self.solver_space.config_from_decision(decision=decision)
        assert isinstance(config, dict), "At this point, should be a dictionary."
        return config, decision_idx

    def provide_performance_feedback(
        self, solve_time: float, construct_time: float, success: bool
    ) -> None:
        reward = self.performance_predictor.reward_maker.estimate_reward(
            solve_time=solve_time, construct_time=construct_time, success=success
        )

        t0 = time()
        self.history.decision_idx.append(self.__decision_idx)
        self.history.expectation.append(self.__expectation)
        self.history.greedy.append(self.__greedy)
        self.history.features.append(self.__features)
        self.history.reward.append(reward)
        self.history.predict_time.append(self.__predict_time)
        self.performance_predictor.partial_fit(
            features=self.history.features[-1],
            solve_time=solve_time,
            construct_time=construct_time,
            success=success,
        )
        self.history.fit_time.append(time() - t0)
