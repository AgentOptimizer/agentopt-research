"""Shared offline metrics for the canonical two-axis ablation figures.

Preserves the evaluation space used by the paper figures on origin/ablation.
Full-data truth is used only for diagnostics, never acquisition.
"""
from __future__ import annotations

import math

import numpy as np


def nondominated_indices(points: np.ndarray) -> np.ndarray:
    """Indices not dominated when both coordinates are maximized."""
    array = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(array), dtype=bool)
    for index, point in enumerate(array):
        dominates = np.all(array >= point, axis=1) & np.any(array > point, axis=1)
        dominates[index] = False
        keep[index] = not np.any(dominates)
    return np.flatnonzero(keep)


def hypervolume_2d(points: np.ndarray) -> float:
    array = np.asarray(points, dtype=np.float64)
    eligible = array[np.all(array >= 0.0, axis=1)]
    if not len(eligible):
        return 0.0
    front = eligible[nondominated_indices(eligible)]
    front = front[np.argsort(front[:, 0])]
    suffix_y = np.maximum.accumulate(front[::-1, 1])[::-1]
    area = 0.0
    previous_x = 0.0
    for point, max_y in zip(front, suffix_y):
        x = max(float(point[0]), previous_x)
        area += (x - previous_x) * max(0.0, float(max_y))
        previous_x = x
    return float(area)


def front_distance(obtained: np.ndarray, reference: np.ndarray) -> float:
    """Mean nearest-point distance over the complete supplied sets.

    GD supplies returned points first; IGD supplies them second. The caller
    constructs the true reference front separately.
    """
    if not len(obtained):
        return math.inf if len(reference) else 0.0
    if not len(reference):
        return math.inf
    delta = obtained[:, None, :] - reference[None, :, :]
    return float(np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=1).mean())


def evaluation_space(raw_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    positive_costs = raw_truth[:, 1][raw_truth[:, 1] > 0.0]
    if not len(positive_costs):
        raise ValueError("truth vectors contain no positive deployment costs")
    cost_reference = float(np.median(positive_costs))
    metric_truth = np.column_stack(
        (raw_truth[:, 0], cost_reference / (cost_reference + raw_truth[:, 1]))
    )
    truth_front = metric_truth[nondominated_indices(metric_truth)]
    return metric_truth, truth_front, cost_reference, hypervolume_2d(truth_front)


def score_selection(
    selected: tuple[int, ...], metric_truth: np.ndarray, truth_front: np.ndarray,
    truth_hv: float,
) -> dict[str, float]:
    obtained = metric_truth[np.asarray(selected, dtype=int)]
    return {
        "hv_regret": max(0.0, truth_hv - hypervolume_2d(obtained)),
        "generational_distance": front_distance(obtained, truth_front),
        "inverted_generational_distance": front_distance(truth_front, obtained),
    }
