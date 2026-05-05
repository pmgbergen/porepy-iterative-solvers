"""An optional machine-learning solver selection sub-module."""

from .solver_space import CategoricalChoices, NumericalChoices, SolverSpace
from .selector import SolverSelector
from .performance_predictor import assemble_default_performance_predictor

__all__ = [
    "CategoricalChoices",
    "NumericalChoices",
    "SolverSpace",
    "SolverSelector",
    "assemble_default_performance_predictor",
]
