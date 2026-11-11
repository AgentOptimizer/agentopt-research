"""Cost normalization preserves rankings, batch boundaries, and gradients."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.acquisition.acquisition import AcquisitionFunction

from agentopt.model_selection.cost_normalized_acquisition import CostNormalizedAcquisition


class SumAcquisition(AcquisitionFunction):
    def __init__(self):
        super().__init__(model=None)

    def forward(self, X):
        return X.sum(dim=(-1, -2))


def test_cost_normalization_changes_ranking_and_preserves_torch_gradients():
    X = torch.tensor([[[0.4]], [[0.9]]], dtype=torch.float64, requires_grad=True)
    costs = torch.tensor([1.0, 3.0], dtype=torch.float64)
    base = SumAcquisition()
    acquisition = CostNormalizedAcquisition(base, lambda X: costs)
    assert base(X).argmax().item() == 1
    scores = acquisition(X)
    assert scores.argmax().item() == 0
    assert scores.dtype == X.dtype
    assert scores.device == X.device
    scores.sum().backward()
    torch.testing.assert_close(X.grad.flatten(), torch.tensor([1.0, 1 / 3], dtype=X.dtype))


def test_cost_estimator_receives_entire_candidate_batch():
    X = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=torch.float64)
    acquisition = CostNormalizedAcquisition(
        SumAcquisition(), lambda X: (X + 1).sum(dim=(-1, -2))
    )
    torch.testing.assert_close(acquisition(X), torch.tensor([3 / 5, 7 / 9], dtype=X.dtype))


@pytest.mark.parametrize("cost", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_costs(cost):
    acquisition = CostNormalizedAcquisition(
        SumAcquisition(), lambda X: torch.tensor([cost])
    )
    with pytest.raises(ValueError, match="positive and finite"):
        acquisition(torch.ones(1, 1, 1))


def test_rejects_broadcasting_costs_across_distinct_candidates():
    acquisition = CostNormalizedAcquisition(
        SumAcquisition(), lambda X: torch.tensor(1.0)
    )
    with pytest.raises(ValueError, match="score shape"):
        acquisition(torch.ones(2, 1, 1))


def test_pending_points_are_forwarded_to_base_acquisition():
    base = SumAcquisition()
    acquisition = CostNormalizedAcquisition(base, lambda X: torch.ones(X.shape[:-2]))
    pending = torch.ones(1, 1)
    acquisition.set_X_pending(pending)
    torch.testing.assert_close(base.X_pending, pending)
    acquisition.set_X_pending(None)
    assert base.X_pending is None
