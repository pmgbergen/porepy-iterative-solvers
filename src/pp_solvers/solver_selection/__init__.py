"""An optional machine-learning solver selection sub-module."""

from .performance_predictor import assemble_default_performance_predictor
from .selector import SolverSelector
from .solver_space import CategoricalChoices, NumericalChoices, SolverSpace

__all__ = [
    "CategoricalChoices",
    "NumericalChoices",
    "SolverSpace",
    "SolverSelector",
    "assemble_default_performance_predictor",
]
