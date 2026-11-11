"""Mean-cost normalization for nonnegative, linear-scale acquisitions.

Import this module only when using the optional ``bayesian`` dependencies.
"""

from typing import Callable

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor


class CostNormalizedAcquisition(AcquisitionFunction):
    """Divide an acquisition by the predicted cost of the next evaluation.

    ``cost_estimator(X)`` returns strictly positive, finite total costs with
    the same shape as ``acquisition(X)``. Each cost includes all evaluations
    in the corresponding candidate batch. These are point estimates, without
    integration over cost uncertainty. Use nonnegative, linear-scale scores
    such as qNEHVI; log acquisitions require a different transformation.
    """

    def __init__(
        self,
        acquisition: AcquisitionFunction,
        cost_estimator: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__(model=acquisition.model)
        self.acquisition = acquisition
        self.cost_estimator = cost_estimator

    def forward(self, X: Tensor) -> Tensor:
        values = self.acquisition(X)
        costs = self.cost_estimator(X).to(values)
        if costs.shape != values.shape:
            raise ValueError("predicted costs must match the acquisition score shape")
        if not torch.all(torch.isfinite(costs) & (costs > 0)):
            raise ValueError("predicted costs must be positive and finite")
        return values / costs

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.acquisition.set_X_pending(X_pending)
