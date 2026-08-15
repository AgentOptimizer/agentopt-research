"""
agentopt (research slice) — model selection algorithms + supporting types.

Full product surface (MITM proxy, daemon, routing) is omitted; see the main
``agentopt`` package for live evaluation. Offline selector sims under
``experiments/`` replay frozen benchmark pickles without API calls.
"""

__version__ = "0.1.0-research"

from .base_models import AgentFn, Dataset, EvalFn, ModelsConfig
from .model_selection import (
    BaseModelSelector,
    BruteForceModelSelector,
    DatapointResult,
    MatrixUCBModelSelector,
    ModelResult,
    RandomSearchModelSelector,
    SelectionResults,
)
from .proxy import CallRecord, LLMTracker, SessionInfo

try:
    from .model_selection import BayesianOptimizationModelSelector
except ImportError:
    BayesianOptimizationModelSelector = None  # type: ignore[misc, assignment]

_METHODS = {
    "auto": MatrixUCBModelSelector,
    "brute_force": BruteForceModelSelector,
    "random": RandomSearchModelSelector,
    "matrix_ucb": MatrixUCBModelSelector,
    "bayesian": BayesianOptimizationModelSelector,
}


def ModelSelector(
    agent=None, models=None, eval_fn=None, dataset=None, method="auto", **kwargs,
):
    """Create a model selector by ``method`` name."""
    cls = _METHODS.get(method)
    if cls is None:
        if method == "bayesian":
            raise ImportError(
                "Bayesian optimization requires optional deps: "
                "numpy, torch, botorch, gpytorch"
            )
        raise ValueError(
            f"Unknown method {method!r}. Choose from: {', '.join(_METHODS)}"
        )
    return cls(agent=agent, models=models, eval_fn=eval_fn, dataset=dataset, **kwargs)


__all__ = [
    "__version__",
    "ModelSelector",
    "BaseModelSelector",
    "LLMTracker",
    "CallRecord",
    "SessionInfo",
    "BruteForceModelSelector",
    "RandomSearchModelSelector",
    "MatrixUCBModelSelector",
    "BayesianOptimizationModelSelector",
    "DatapointResult",
    "ModelResult",
    "SelectionResults",
    "AgentFn",
    "Dataset",
    "EvalFn",
    "ModelsConfig",
]
