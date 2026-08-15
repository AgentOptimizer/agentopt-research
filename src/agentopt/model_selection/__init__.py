"""Model selection algorithms."""

from .base import BaseModelSelector, DatapointResult, ModelResult, SelectionResults
from .brute_force import BruteForceModelSelector
from .matrix_ucb import MatrixUCBModelSelector
from .random_search import RandomSearchModelSelector

try:
    from .gittins import GittinsModelSelector
except ImportError:
    GittinsModelSelector = None  # type: ignore[misc, assignment]

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
    "GittinsModelSelector",
    "BayesianOptimizationModelSelector",
    "DatapointResult",
    "ModelResult",
    "SelectionResults",
]
