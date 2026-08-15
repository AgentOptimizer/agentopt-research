"""Model selection algorithms."""

from .base import BaseModelSelector, DatapointResult, ModelResult, SelectionResults
from .brute_force import BruteForceModelSelector
from .matrix_ucb import MatrixUCBModelSelector
from .random_search import RandomSearchModelSelector

# Bayesian is optional (requires torch/botorch)
try:
    from .bayesian_optimization import BayesianOptimizationModelSelector
except ImportError:
    pass

__all__ = [
    "BaseModelSelector",
    "BruteForceModelSelector",
    "RandomSearchModelSelector",
    "MatrixUCBModelSelector",
    "BayesianOptimizationModelSelector",
    "DatapointResult",
    "ModelResult",
    "SelectionResults",
]
