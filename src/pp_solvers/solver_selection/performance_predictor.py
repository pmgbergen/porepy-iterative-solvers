"""Machine learning model components for solver performance prediction.

For most use cases, call `assemble_default_performance_predictor` to get a ready-to-use
ML predictor.

The default predictor uses a two-phase strategy, introduced in:
https://arxiv.org/abs/2510.04920

1. **Initial exploration** - randomly samples solvers until enough data is collected
    (`num_initial_exploration` samples), then trains an initial model.
2. **Incremental (or online) learning** - updates the model in batches as new solve
    results arrive, optionally injecting epsilon-greedy exploration to avoid local optima.

The reward signal is ``-log(construct_time + solve_time)`` for successful solves and a
large negative constant for failures, computed by `RewardEstimator`.

Internally the predictor composes several wrappers:

- `InitialExplorationEstimator` - orchestrates the two phases above.
- `EpsGreedyExplorationModel` - wraps a model with decaying epsilon-greedy exploration.
- `TwoEstimators` - separates success/failure classification from reward regression so
    that failed solves do not distort the regressor.
- `IncrementalRefitModel` - adapts scikit-learn estimators (which lack ``partial_fit``)
    to incremental updates by accumulating data and refitting from scratch each batch.

"""

import numpy as np

FAIL_REWARD = -99
"""A large negative constant reward for failed solution attempts."""
EPSGREEDY_EXPECTATION = 100
"""A large positive constant reward to encourage exploration.
It will be much larger than any predicted reward so the selector will choose this
option.

"""


class IncrementalRefitModel:
    """Wraps an ML model and provides a `partial_fit` method, so that:
    - all the passed for training data is cached.
    - `partial_fit` refits the model using all the previously passed data.

    """

    def __init__(self, model):
        self.model = model
        """A machine learning model with methods `fit` and `predict`."""
        self.X = []
        self.y = []

    def fit(self, X, y):
        """Initially fit the model. Drops all the cached data if present."""
        self.X = X.tolist()
        self.y = y.tolist()
        self.model.fit(X, y)

    def partial_fit(self, X, y):
        """Update the cached data with the new X and y and refit the model using all the
        cached data.

        """
        self.X.extend(X.tolist())
        self.y.extend(y.tolist())
        self.model.fit(self.X, self.y)

    def predict(self, X):
        return self.model.predict(X)


class TwoEstimators:
    """Wraps two ML models, which denote two steps:
    1. A success/failure classifier which predicts whether the reward will be above the
        fixed threshold `FAIL_REWARD`.
    2. A regressor which predicts the reward, only for the samples marked as "success"
        by the classifier in step 1.

    For the samples marked as "failure" by step 1, the predicted reward is a large
    negative constant.

    """

    def __init__(self, classifier, regressor):
        self.classifier = classifier
        """A classification ML model with methods `fit`, `partial_fit` and `predict`."""
        self.regressor = regressor
        """A regression ML model with methods `fit`, `partial_fit` and `predict`."""

    def fit(self, X, y):
        success = y >= FAIL_REWARD
        self.classifier.fit(X, success)
        self.regressor.fit(X[success], y[success])

    def partial_fit(self, X, y):
        if len(X.shape) == 1:
            X = np.array(X).reshape(1, -1)
        y = np.atleast_1d(y)
        success = y >= FAIL_REWARD
        self.classifier.partial_fit(X, success)
        if np.any(success):
            self.regressor.partial_fit(X[success], y[success])

    def predict(self, X):
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


class EpsGreedyExplorationModel:
    """Wraps an ML model with decaying epsilon-greedy exploration.

    With probability `eps`, ignores the model and returns a large positive reward for a
    randomly chosen option to encourage exploration. After each exploration step, `eps`
    is multiplied by `eps1` so that exploration decays over time.

    You can disable exploration by passing `eps=0`.

    """

    def __init__(self, model, eps: float, eps1: float) -> None:
        self.model = model
        """A machine learning model with methods `fit`, `partial_fit` and `predict`."""
        self.eps: float = eps
        """Current exploration probability; decays after each exploration step."""
        self.eps1: float = eps1
        """Decay factor applied to `eps` after each exploration step."""

    def fit(self, X, y):
        self.model.fit(X, y)

    def partial_fit(self, X, y):
        self.model.partial_fit(X, y)

    def predict(self, X):
        if np.random.random() < self.eps:
            self.eps *= self.eps1
            result = np.zeros(X.shape[0])
            result[np.random.randint(result.size)] = EPSGREEDY_EXPECTATION
            return result
        return self.model.predict(X)


class InitialExplorationEstimator:
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

    def __init__(self, model, num_initial_exploration: int, batch_size: int):
        self.model = model
        """A machine learning model with methods `fit`, `partial_fit` and `predict`."""
        self.num_initial_exploration: int = num_initial_exploration
        """Number of random solve results to collect before fitting the initial model.
        """
        self.batch_size: int = batch_size
        """Number of new results to accumulate before updating the model via 
        `partial_fit`."""

        # Buffers for the current batch; cleared after each model update.
        self.X_history = []
        self.y_history = []
        self.is_ready_to_predict = False
        """Permanently switches to True when the initial exploration phase is complete.
        """
        self.exploration_expectation = EPSGREEDY_EXPECTATION

        # Hard-coded it here with possibility to extend if a different reward formula is
        # ever needed (e.g. penalize very large construct_time for some reason).
        self.rewarder = Rewarder()

    def select_solver(self, features: np.ndarray) -> tuple[int, float, bool]:
        """Select a solver given a feature matrix of candidate solvers.

        Parameters:
            features: `shape=(num_solver_configurations, num_encoded_data_in_conf)`,
                feature matrix.

        Returns:
            A tuple of `(index, expected_reward, is_model_prediction)`, where the last
            element is `False` during the initial random exploration phase.

        """
        if not self.is_ready_to_predict:
            return (
                np.random.randint(features.shape[0]),
                self.exploration_expectation,
                False,
            )

        expectations = self.model.predict(features)
        argmax = int(np.argmax(expectations))
        expectation = float(expectations[argmax])
        return argmax, expectation, True

    def partial_fit(
        self,
        features: np.ndarray,
        solve_time: float,
        construct_time: float,
        success: bool,
    ):
        """Record the result of a solve and update the model if a batch is ready.

        Converts the solve outcome into a reward and buffers it. Triggers an initial
        `fit` once `num_initial_exploration` results are collected, and subsequent
        `partial_fit` calls every `batch_size` results thereafter.

        Parameters:
            features: `shape=(num_encoded_data_in_conf,)`, feature vector corresponding
                to the chosen solver configuration.
            solve_time: Time it took to solve the linear system, s.
            construct_time: Time it took to construct the linear solver, s.
            success: Whether the solution is successful.

        """
        reward = self.rewarder.estimate_reward(solve_time, construct_time, success)
        self.X_history.append(features)
        self.y_history.append(reward)
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


class Rewarder:
    """Transforms `solve_time` and `construct_time` into the reward for the ML
    algorithm.

    """

    def __init__(self):
        # FAIL_REWARD - 1 here to not think about "less or equal" edge case.
        self.worst_known_reward: float = FAIL_REWARD - 1

    def estimate_reward(self, solve_time: float, construct_time: float, success: bool):
        if success:
            reward = -np.log(construct_time + solve_time)
        else:
            reward = -2 * abs(self.worst_known_reward)
        return reward


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
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeClassifier
        from sklearn.ensemble import GradientBoostingRegressor
    except:
        raise ImportError("Sklearn not installed, try `pip install scikit-learn`.")

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
