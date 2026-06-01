"""Machine learning model components for solver performance prediction.

For most use cases, call `assemble_default_performance_predictor` to get a ready-to-use
ML predictor.

The default predictor uses a two-phase strategy, introduced in:
https://arxiv.org/abs/2510.04920

1. **Initial exploration** - randomly samples solvers until enough data is collected
    (`num_initial_exploration` samples), then trains an initial model.
2. **Incremental (or online) learning** - updates the model in batches as new solve
    results arrive, optionally injecting epsilon-greedy exploration to avoid local
    optima.

Internally the predictor composes several wrappers, each conforming to
a BaseIncrementalMLModel protocol:

- `IncrementalRefitModel` - adapts scikit-learn estimators (which lack ``partial_fit``)
    to incremental updates by accumulating data and refitting from scratch each batch.
- `TwoEstimators` - separates success/failure classification from reward regression so
    that failed solves do not distort the regressor.
- `EpsGreedyExplorationModel` - wraps a model with decaying epsilon-greedy exploration.
- `InitialExplorationEstimator` - applies random initial exploration, then runs ML
    inference for selection.

"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FAIL_REWARD = -99
"""A large negative constant reward for failed solution attempts."""
EPSGREEDY_EXPECTATION = 100
"""A large positive constant reward to encourage exploration.
It will be much larger than any predicted reward so the selector will choose this
option.

"""


class BaseIncrementalMLModel(ABC):
    """This base class defines an sklearn-like interface for a machine learning model
    with methods `fit`, `partial_fit`, and `predict`.

    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def save_history(self) -> None:
        pass


class IncrementalRefitModel(BaseIncrementalMLModel):
    """Wraps an ML model and provides a `partial_fit` method, so that:
    - all the passed for training data is cached.
    - `partial_fit` refits the model using all the previously passed data.

    """

    def __init__(self, model) -> None:
        self.model = model
        """A machine learning model with methods `fit` and `predict`."""
        self.X: list[np.ndarray] = []
        self.y: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initially fit the model. Drops all the cached data if present."""
        self.X = X.tolist()
        self.y = y.tolist()
        self.model.fit(X, y)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the cached data with the new X and y and refit the model using all the
        cached data.

        """
        self.X.extend(X.tolist())
        self.y.extend(y.tolist())
        self.model.fit(self.X, self.y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class TwoEstimators(BaseIncrementalMLModel):
    """Wraps two ML models, which denote two steps:
    1. A success/failure classifier which predicts whether the reward will be above the
        fixed threshold `FAIL_REWARD`.
    2. A regressor which predicts the reward, only for the samples marked as "success"
        by the classifier in step 1.

    For the samples marked as "failure" by step 1, the predicted reward is a large
    negative constant.

    """

    def __init__(
        self, classifier: BaseIncrementalMLModel, regressor: BaseIncrementalMLModel
    ):
        self.classifier: BaseIncrementalMLModel = classifier
        """A classification ML model with methods `fit`, `partial_fit` and `predict`."""
        self.regressor: BaseIncrementalMLModel = regressor
        """A regression ML model with methods `fit`, `partial_fit` and `predict`."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        success = y >= FAIL_REWARD
        self.classifier.fit(X, success)
        self.regressor.fit(X[success], y[success])

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X.shape) == 1:
            X = np.array(X).reshape(1, -1)
        y = np.atleast_1d(y)
        success = y >= FAIL_REWARD
        self.classifier.partial_fit(X, success)
        if np.any(success):
            self.regressor.partial_fit(X[success], y[success])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the reward. If a sample is marked as "failure" in step 1, the
        predicted reward is a large negative constant.

        """
        # FAIL_REWARD - 1 here to not think about "less or equal" edge case.
        reward_estimate = np.full(X.shape[0], FAIL_REWARD - 1, dtype=float)
        success_estimate = self.classifier.predict(X)
        if not np.any(success_estimate):
            return reward_estimate

        reward_estimate[success_estimate] = self.regressor.predict(X[success_estimate])
        return reward_estimate


class EpsGreedyExplorationModel(BaseIncrementalMLModel):
    """Wraps an ML model with decaying epsilon-greedy exploration.

    With probability `eps`, ignores the model and returns a large positive reward for a
    randomly chosen option to encourage exploration. After each exploration step, `eps`
    is multiplied by `eps1` so that exploration decays over time.

    You can disable exploration by passing `eps=0`.

    """

    def __init__(self, model: BaseIncrementalMLModel, eps: float, eps1: float) -> None:
        self.model: BaseIncrementalMLModel = model
        """A machine learning model with methods `fit`, `partial_fit` and `predict`."""
        self.eps: float = eps
        """Current exploration probability; decays after each exploration step."""
        self.eps1: float = eps1
        """Decay factor applied to `eps` after each exploration step."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if np.random.random() < self.eps:
            self.eps *= self.eps1
            result = np.zeros(X.shape[0])
            result[np.random.randint(result.size)] = EPSGREEDY_EXPECTATION
            return result
        return self.model.predict(X)


class InitialExplorationEstimator(BaseIncrementalMLModel):
    """Orchestrates two-phase solver selection: random exploration followed by ML-guided
    selection.

    During the initial phase, solvers are chosen at random until
    `num_initial_exploration` results are collected, at which point the model is fitted.
    Afterwards, the model is updated in batches of `batch_size` as new results arrive.

    Parameters:
        model: A machine learning model with methods `fit`, `partial_fit` and `predict`.
        num_initial_exploration: Number of random solve results to collect before
            fitting the initial model.
        batch_size: Number of new results to accumulate before updating the model via
            `partial_fit`.

    """

    def __init__(
        self,
        model: BaseIncrementalMLModel,
        num_initial_exploration: int,
        batch_size: int,
    ) -> None:
        self.model: BaseIncrementalMLModel = model
        """A machine learning model with methods `fit`, `partial_fit` and `predict`."""
        self.num_initial_exploration: int = num_initial_exploration
        """Number of random solve results to collect before fitting the initial model.
        """
        self.batch_size: int = batch_size
        """Number of new results to accumulate before updating the model via 
        `partial_fit`."""

        # Buffers for the current batch; cleared after each model update.
        self.X_history: list[np.ndarray] = []
        self.y_history: list[float] = []
        self.is_ready_to_predict = False
        """Permanently switches to True when the initial exploration phase is complete.
        """
        self.exploration_expectation = EPSGREEDY_EXPECTATION

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Select a solver given a feature matrix of candidate solvers.

        Parameters:
            X: `shape=(num_solver_configurations, num_encoded_data_in_conf)`, feature
                matrix.

        Returns:
            `shape=(num_solver_configurations,)`, predicted reward for each solver
            configuration. During the initial exploration phase, all values are zero
            except one randomly chosen entry, which is set to `EPSGREEDY_EXPECTATION`
            to force the selector to pick it. After training, returns the model's
            predicted rewards directly.

        """
        if not self.is_ready_to_predict:
            expectations = np.zeros_like(X[:, 0])
            random_selected = np.random.randint(X.shape[0])
            expectations[random_selected] = self.exploration_expectation
            return expectations

        return self.model.predict(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """This does the same thing as partial_fit, implemented for completeness."""
        self.partial_fit(X, y)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Record the result of a solve and update the model if a batch is ready.

        Triggers an initial `fit` once `num_initial_exploration` results are collected,
        and subsequent `partial_fit` calls every `batch_size` results thereafter.

        Parameters:
            X: `shape=(num_encoded_data_in_conf,)`, feature vector corresponding
                to the chosen solver configuration.
            y: `shape=(1,)` reward.

        """
        assert y.size == 1, "Only single feedback at a time is allowed."
        self.X_history.append(X)
        self.y_history.append(y.item())
        if (
            not self.is_ready_to_predict
            and len(self.y_history) >= self.num_initial_exploration
        ):
            self.model.fit(np.array(self.X_history), np.array(self.y_history))
            self.is_ready_to_predict = True
            self.X_history.clear()
            self.y_history.clear()

        if self.is_ready_to_predict and len(self.y_history) >= self.batch_size:
            self.model.partial_fit(np.array(self.X_history), np.array(self.y_history))
            self.X_history.clear()
            self.y_history.clear()


def assemble_default_performance_predictor(
    random_state: int = 42,
    num_initial_exploration: int = 64,
    batch_size: int = 64,
    eps: float = 0,
    eps1: float = 0.9,
):
    """Use this function to get a ready-to-use ML predictor for the solver selector.

    The algorithm:
    - Explores randomly for the first `num_initial_exploration` solves, then switches to
        ML selection.
    - Optionally injects epsilon-greedy exploration, controlled by parameters `eps` and
        `eps1`. Off by default, since `eps=0` disables it.
    - Separates success/failure prediction (RidgeClassifier) from reward regression
      (GradientBoostingRegressor), so failed solves do not distort the regressor.
    - Adapts both models incrementally by refitting on all accumulated data each
        `batch_size` linear systems.

    """
    return InitialExplorationEstimator(
        num_initial_exploration=num_initial_exploration,
        batch_size=batch_size,
        model=EpsGreedyExplorationModel(
            eps=eps,
            eps1=eps1,
            model=TwoEstimators(
                classifier=IncrementalRefitModel(
                    model=make_pipeline(StandardScaler(), RidgeClassifier())
                ),
                regressor=IncrementalRefitModel(
                    model=GradientBoostingRegressor(random_state=random_state)
                ),
            ),
        ),
    )
