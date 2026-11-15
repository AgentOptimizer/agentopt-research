"""Gittins boundaries targeting the realized mean of a fixed benchmark.

The cache is separate from latent-mean disk tables. Scalar and radial indices
use the same full-benchmark uncertainty schedule, including zero terminal
variance and the actual size and proportional price of a partial final batch.
"""

from __future__ import annotations

import numpy as np

from .axis_gittins_dp import build_axis_gittins_boundary_table
from .gittins_schedules import finite_benchmark_schedule
from .radial_gittins_dp import RadialGittinsGrid, build_radial_gittins_boundary_table


ACQUISITION_TARGETS = ("latent_mean", "finite_benchmark_mean")


def validate_acquisition_target(target):
    if target not in ACQUISITION_TARGETS:
        raise ValueError(f"acquisition_target must be one of {ACQUISITION_TARGETS}")
    return target


def deterministic_direction_utility(mean, direction, reference=(0.0, 0.0)):
    """Exact terminal utility after all benchmark outcomes have been observed."""
    mean, direction, reference = (np.asarray(x, dtype=np.float64) for x in (mean, direction, reference))
    active = direction > 0
    if mean.shape != (2,) or direction.shape != (2,) or reference.shape != (2,):
        raise ValueError("mean, direction, and reference must have two coordinates")
    if not all(np.all(np.isfinite(x)) for x in (mean, direction, reference)) or not np.any(active) or np.any(direction < 0) or not np.isclose(direction.sum(), 1.0):
        raise ValueError("invalid direction or nonfinite terminal state")
    return float(np.min(direction.max() / direction[active] * (mean - reference)[active]))


class FiniteBenchmarkGittinsCache:
    """Memory cache keyed by target horizon, warm count, noise, price, and grid."""

    def __init__(self):
        self.tables = {}
        self.hits = 0

    def get(self, *, direction, effective_pull_cost, initial_var, question_noise_var,
            n_total, n_observed, batch_size, grid=RadialGittinsGrid()):
        direction = tuple(map(float, direction))
        deterministic_direction_utility((0.0, 0.0), direction)
        initial = np.asarray(initial_var, dtype=np.float64)
        noise = np.asarray(question_noise_var, dtype=np.float64)
        if initial.shape != (2,) or noise.shape != (2,):
            raise ValueError("initial_var and question_noise_var must have two coordinates")
        variances, batches = finite_benchmark_schedule(
            initial, noise, n_total=n_total, n_observed=n_observed, batch_size=batch_size,
        )
        cost = float(effective_pull_cost)
        if not np.isfinite(cost) or cost <= 0:
            raise ValueError("effective_pull_cost must be finite and positive")
        if not len(batches):
            raise ValueError("completed arms use their exact terminal utility, without a boundary table")
        key = (direction, cost, tuple(initial), tuple(noise), n_total, n_observed, batch_size, grid)
        if key in self.tables:
            self.hits += 1
            return self.tables[key]
        costs = cost * batches / batch_size
        active = np.flatnonzero(direction)
        if len(active) == 1:
            axis = int(active[0])
            result = build_axis_gittins_boundary_table(
                effective_pull_cost=cost, initial_var=float(initial[axis]),
                obs_noise_var=float(noise[axis] / batch_size), horizon=len(batches),
                grid_size=grid.state_size, variance_schedule=variances[:, axis], stage_costs=costs,
            )
        else:
            result = build_radial_gittins_boundary_table(
                direction=direction, effective_pull_cost=cost, initial_var=initial,
                obs_noise_var=noise / batch_size, horizon=len(batches), grid=grid,
                variance_schedule=variances, stage_costs=costs,
            )
        self.tables[key] = result
        return result

    def stats_snapshot(self):
        return {"builds": len(self.tables), "memory_hits": self.hits}
