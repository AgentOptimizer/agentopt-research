"""Deterministic uncertainty and price schedules at actual batch boundaries."""

from __future__ import annotations

import numpy as np


def validate_dp_schedule(variances, costs, horizon: int, dimensions: int):
    """Validate posterior variances and strictly positive stage prices."""
    variance = np.asarray(variances, dtype=np.float64)
    price = np.asarray(costs, dtype=np.float64)
    expected = (horizon + 1,) if dimensions == 1 else (horizon + 1, dimensions)
    if variance.shape != expected or not np.all(np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError(f"variance_schedule must be finite, nonnegative, and have shape {expected}")
    if np.any(np.diff(variance, axis=0) > 0):
        raise ValueError("variance_schedule must be nonincreasing")
    if price.shape != (horizon,) or not np.all(np.isfinite(price)) or np.any(price <= 0):
        raise ValueError("stage_costs must contain one finite positive price per transition")
    return variance.copy(), price.copy()


def finite_benchmark_schedule(initial_var, question_noise_var, *, n_total, n_observed, batch_size):
    """Return full-benchmark variances and actual subsequent batch sizes.

    ``initial_var`` is the latent variance after the observed warm batch.
    The last row is zero: completing the fixed benchmark reveals its payoff.
    """
    for name, value, minimum in (("n_total", n_total, 1), ("n_observed", n_observed, 0),
                                 ("batch_size", batch_size, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if n_observed > n_total:
        raise ValueError("n_observed cannot exceed n_total")
    initial = np.asarray(initial_var, dtype=np.float64)
    noise = np.asarray(question_noise_var, dtype=np.float64)
    if initial.ndim != 1 or initial.shape != noise.shape or initial.size not in (1, 2):
        raise ValueError("initial_var and question_noise_var must be matching one- or two-coordinate vectors")
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(noise)) or np.any(initial <= 0) or np.any(noise <= 0):
        raise ValueError("initial variances and per-question noise must be finite and positive")
    horizon = (n_total - n_observed + batch_size - 1) // batch_size
    counts = np.minimum(n_observed + np.arange(horizon + 1) * batch_size, n_total)
    latent = 1.0 / (1.0 / initial + (counts - n_observed)[:, None] / noise)
    remaining = (n_total - counts)[:, None]
    variance = (remaining * remaining * latent + remaining * noise) / n_total**2
    return variance, np.diff(counts)
