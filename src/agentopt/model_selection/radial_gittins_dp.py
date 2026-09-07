"""Deterministic boundary tables for the two-objective radial-Gittins policy.

The retirement dynamic program is solved in centered direction-scaled
coordinates ``x1 = m1 - alpha`` and ``x2 = m2 - alpha``. In these coordinates
the two posterior-mean increments are independent, so each backward step is
two deterministic one-dimensional Gaussian expectations of the linearly
interpolated value function. The resulting value function is converted to the
online ``b_n(delta)`` boundary table.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union
from zipfile import BadZipFile

import numpy as np

from .radial_gittins import expected_min_of_two_normals


VectorLike = Union[float, Sequence[float], np.ndarray]


# These versions deliberately do not follow the package version.  Bump the
# schema version for an incompatible file-layout change, and the solver version
# whenever the numerical recurrence or its interpretation changes.
RADIAL_BOUNDARY_CACHE_SCHEMA_VERSION = 1
RADIAL_BOUNDARY_SOLVER_VERSION = 1
_CACHE_ARRAY_DTYPE = np.dtype("<f8")


class BoundaryGridError(RuntimeError):
    """Raised when a numerical grid cannot contain a reliable DP boundary."""


def _two_vector(value: VectorLike, name: str, *, positive: bool) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array, 2)
    if array.shape != (2,):
        raise ValueError(f"{name} must be a scalar or a length-2 vector")
    # Scalar checks rather than numpy reductions: this is the innermost helper
    # of the replay loop, where a two-element ``np.all`` costs more than the
    # closed-form utility it is guarding.
    first = float(array[0])
    second = float(array[1])
    if not (math.isfinite(first) and math.isfinite(second)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and (first <= 0.0 or second <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array.copy()


def _direction(value: VectorLike) -> np.ndarray:
    result = _two_vector(value, "direction", positive=True)
    total = float(result[0]) + float(result[1])
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("direction components must sum to one")
    return result


def _scaling_factors(direction_array: np.ndarray) -> np.ndarray:
    """Return the direction-scaling vector ``max(direction) / direction``."""
    largest = max(float(direction_array[0]), float(direction_array[1]))
    return largest / direction_array


@lru_cache(maxsize=512)
def _regular_grid(lower: float, upper: float, size: int) -> np.ndarray:
    """Return a shared read-only ``linspace``.

    Grid axes are pure functions of the frozen grid settings and are re-read
    on every boundary lookup, so rebuilding them each time is wasted work.
    """
    grid = np.linspace(lower, upper, size, dtype=np.float64)
    grid.setflags(write=False)
    return grid


@dataclass(frozen=True)
class RadialGittinsGrid:
    """Numerical state, root, and interpolation grids."""

    # For the default directions, normalized objectives in [0, 1] imply
    # |delta| <= 9 and the relevant stopping roots are nonnegative. These
    # asymmetric/narrower defaults devote substantially more resolution to
    # that reachable region while retaining guard bands on every side.
    z_min: float = -2.0
    z_max: float = 12.0
    z_size: int = 257
    delta_min: float = -12.0
    delta_max: float = 12.0
    delta_size: int = 257
    state_size: int = 257
    # Minimum state-grid guard band. The builder expands it per objective when
    # cumulative posterior movement requires more room.
    state_halo: float = 1.0
    kernel_stddevs: float = 6.0
    boundary_margin_cells: int = 3
    monotonicity_tolerance: float = 2e-7

    def __post_init__(self) -> None:
        numeric = (
            self.z_min,
            self.z_max,
            self.delta_min,
            self.delta_max,
            self.state_halo,
            self.kernel_stddevs,
            self.monotonicity_tolerance,
        )
        if not all(math.isfinite(float(x)) for x in numeric):
            raise ValueError("grid settings must be finite")
        if self.z_max <= self.z_min or self.delta_max <= self.delta_min:
            raise ValueError("grid maxima must exceed grid minima")
        for name in ("z_size", "delta_size", "state_size"):
            value = getattr(self, name)
            if int(value) != value or value < 17:
                raise ValueError(f"{name} must be an integer of at least 17")
        if self.state_halo <= 0.0 or self.kernel_stddevs <= 0.0:
            raise ValueError("state_halo and kernel_stddevs must be positive")
        if int(self.boundary_margin_cells) != self.boundary_margin_cells:
            raise ValueError("boundary_margin_cells must be an integer")
        if not 1 <= self.boundary_margin_cells < self.z_size // 2:
            raise ValueError("boundary_margin_cells is incompatible with z_size")
        if self.monotonicity_tolerance < 0.0:
            raise ValueError("monotonicity_tolerance must be nonnegative")

    @property
    def z_grid(self) -> np.ndarray:
        return _regular_grid(self.z_min, self.z_max, self.z_size)

    @property
    def delta_grid(self) -> np.ndarray:
        return _regular_grid(self.delta_min, self.delta_max, self.delta_size)

    @property
    def x1_grid(self) -> np.ndarray:
        lower = self.z_min + self.delta_min / 2.0 - self.state_halo
        upper = self.z_max + self.delta_max / 2.0 + self.state_halo
        return _regular_grid(lower, upper, self.state_size)

    @property
    def x2_grid(self) -> np.ndarray:
        lower = self.z_min - self.delta_max / 2.0 - self.state_halo
        upper = self.z_max - self.delta_min / 2.0 + self.state_halo
        return _regular_grid(lower, upper, self.state_size)


def direction_aware_grid(
    direction: VectorLike,
    *,
    base_grid: RadialGittinsGrid = RadialGittinsGrid(),
    objective_lower: VectorLike = (0.0, 0.0),
    objective_upper: VectorLike = (1.0, 1.0),
    reference: VectorLike = (0.0, 0.0),
    delta_padding: float = 0.2,
    z_padding: float = 1.0,
) -> RadialGittinsGrid:
    """Tighten a grid to the direction's reachable objective box.

    The defaults target the selector's normalized objective vectors in
    ``[0, 1]^2``. For example, the balanced direction gets approximately
    ``delta in [-1.2, 1.2]``, while direction ``(0.1, 0.9)`` gets
    ``delta in [-1.2, 9.2]``. Bounds are intersected with ``base_grid`` so it
    remains an explicit safety envelope. A new grid is returned; the input is
    never mutated.
    """
    direction_array = _direction(direction)
    lower = _two_vector(objective_lower, "objective_lower", positive=False)
    upper = _two_vector(objective_upper, "objective_upper", positive=False)
    reference_array = _two_vector(reference, "reference", positive=False)
    if np.any(upper <= lower):
        raise ValueError(
            "objective_upper must exceed objective_lower componentwise"
        )
    if not math.isfinite(delta_padding) or delta_padding <= 0.0:
        raise ValueError("delta_padding must be finite and positive")
    if not math.isfinite(z_padding) or z_padding <= 0.0:
        raise ValueError("z_padding must be finite and positive")

    factors = _scaling_factors(direction_array)
    scaled_lower = factors * (lower - reference_array)
    scaled_upper = factors * (upper - reference_array)
    reachable_delta_min = float(scaled_lower[0] - scaled_upper[1])
    reachable_delta_max = float(scaled_upper[0] - scaled_lower[1])
    reachable_u_min = float((scaled_lower[0] + scaled_lower[1]) / 2.0)
    reachable_u_max = float((scaled_upper[0] + scaled_upper[1]) / 2.0)

    z_min = min(0.0, reachable_u_min) - z_padding
    z_max = max(0.0, reachable_u_max) + z_padding
    delta_min = reachable_delta_min - delta_padding
    delta_max = reachable_delta_max + delta_padding
    if (
        z_min < base_grid.z_min
        or z_max > base_grid.z_max
        or delta_min < base_grid.delta_min
        or delta_max > base_grid.delta_max
    ):
        raise BoundaryGridError(
            "base_grid does not contain the reachable direction-scaled domain"
        )
    return replace(
        base_grid,
        z_min=z_min,
        z_max=z_max,
        delta_min=delta_min,
        delta_max=delta_max,
    )


@dataclass(frozen=True)
class RadialGittinsBoundaryTable:
    """Stopping boundaries at stages ``0..H``; row ``H`` is terminal."""

    direction: Tuple[float, float]
    effective_pull_cost: float
    initial_var: Tuple[float, float]
    obs_noise_var: Tuple[float, float]
    horizon: int
    grid: RadialGittinsGrid
    boundaries: np.ndarray
    max_monotonicity_violation: float = 0.0
    # Objective exchange sends ``delta`` to ``-delta``.  A cache alias keeps
    # the caller-facing metadata and sampled boundary row in that orientation,
    # while delegating scalar queries to the table that was actually solved.
    # Excluding this implementation detail from repr/equality preserves the
    # public value semantics of a boundary table.
    _mirrored_source: "RadialGittinsBoundaryTable | None" = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        violation = float(self.max_monotonicity_violation)
        if not math.isfinite(violation) or violation < 0.0:
            raise ValueError(
                "max_monotonicity_violation must be finite and nonnegative"
            )
        array = np.asarray(self.boundaries, dtype=np.float64).copy()
        expected_shape = (self.horizon + 1, self.grid.delta_size)
        if array.shape != expected_shape:
            raise ValueError(
                f"boundaries must have shape {expected_shape}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("boundaries must be finite")
        array.setflags(write=False)
        object.__setattr__(self, "max_monotonicity_violation", violation)
        object.__setattr__(self, "boundaries", array)

    @property
    def delta_grid(self) -> np.ndarray:
        return self.grid.delta_grid

    def boundary(self, stage: int, delta: float) -> float:
        if int(stage) != stage or not 0 <= int(stage) <= self.horizon:
            raise IndexError(f"stage must lie in [0, {self.horizon}]")
        delta = float(delta)
        if not math.isfinite(delta):
            raise ValueError("delta must be finite")
        if delta < self.grid.delta_min or delta > self.grid.delta_max:
            raise BoundaryGridError(
                f"delta={delta:.6g} lies outside configured grid "
                f"[{self.grid.delta_min}, {self.grid.delta_max}]"
            )
        if self._mirrored_source is not None:
            return self._mirrored_source.boundary(stage, -delta)
        return float(
            np.interp(delta, self.delta_grid, self.boundaries[int(stage)])
        )



def posterior_variance_schedule(
    initial_var: VectorLike,
    obs_noise_var: VectorLike,
    horizon: int,
) -> np.ndarray:
    """Return posterior variances for stages ``0..horizon``."""
    v0 = _two_vector(initial_var, "initial_var", positive=True)
    tau_sq = _two_vector(obs_noise_var, "obs_noise_var", positive=True)
    if int(horizon) != horizon or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    stages = np.arange(int(horizon) + 1, dtype=np.float64)[:, None]
    return 1.0 / (1.0 / v0[None, :] + stages / tau_sq[None, :])


def radial_posterior_coordinates(
    mean: VectorLike,
    var: VectorLike,
    direction: VectorLike,
    reference: VectorLike = (0.0, 0.0),
) -> Tuple[float, float, np.ndarray]:
    """Return ``(u, delta, scaled_variances)`` for one arm and direction."""
    mean_array = _two_vector(mean, "mean", positive=False)
    var_array = _two_vector(var, "var", positive=True)
    direction_array = _direction(direction)
    reference_array = _two_vector(reference, "reference", positive=False)
    factors = _scaling_factors(direction_array)
    scaled_means = factors * (mean_array - reference_array)
    scaled_var = factors * factors * var_array
    u = float((scaled_means[0] + scaled_means[1]) / 2.0)
    delta = float(scaled_means[0] - scaled_means[1])
    return u, delta, scaled_var


def terminal_expected_radial_utility(
    mean: VectorLike,
    var: VectorLike,
    direction: VectorLike,
    reference: VectorLike = (0.0, 0.0),
) -> float:
    """Posterior expectation of the direction-scaled minimum utility."""
    u, delta, scaled_var = radial_posterior_coordinates(
        mean,
        var,
        direction,
        reference,
    )
    return expected_min_of_two_normals(
        u + delta / 2.0,
        float(scaled_var[0]),
        u - delta / 2.0,
        float(scaled_var[1]),
    )


def radial_transition_covariance(
    current_var: VectorLike,
    next_var: VectorLike,
    direction: VectorLike,
) -> np.ndarray:
    """Covariance of the next posterior-mean increment in ``(u, delta)``."""
    current = _two_vector(current_var, "current_var", positive=True)
    following = _two_vector(next_var, "next_var", positive=True)
    q = current - following
    tolerance = 1e-13 * np.maximum(current, 1.0)
    if np.any(q < -tolerance):
        raise ValueError("next_var cannot exceed current_var")
    q = np.maximum(q, 0.0)
    direction_array = _direction(direction)
    factors = _scaling_factors(direction_array)
    scaled_q = factors * factors * q
    return np.array(
        [
            [
                (scaled_q[0] + scaled_q[1]) / 4.0,
                (scaled_q[0] - scaled_q[1]) / 2.0,
            ],
            [
                (scaled_q[0] - scaled_q[1]) / 2.0,
                scaled_q[0] + scaled_q[1],
            ],
        ],
        dtype=np.float64,
    )


def _terminal_imbalance(
    delta_grid: np.ndarray,
    scaled_var: np.ndarray,
) -> np.ndarray:
    difference_var = float(scaled_var.sum())
    if difference_var <= np.finfo(np.float64).tiny:
        return -np.abs(delta_grid) / 2.0
    s = math.sqrt(difference_var)
    d = delta_grid / s
    from scipy.special import ndtr

    phi = np.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
    cdf = ndtr(d)
    m1 = delta_grid / 2.0
    m2 = -delta_grid / 2.0
    return m1 * (1.0 - cdf) + m2 * cdf - s * phi


def _terminal_value(
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    scaled_var: np.ndarray,
) -> np.ndarray:
    x1 = x1_grid[:, None]
    x2 = x2_grid[None, :]
    difference_var = float(scaled_var.sum())
    if difference_var <= np.finfo(np.float64).tiny:
        return np.maximum(0.0, np.minimum(x1, x2))
    s = math.sqrt(difference_var)
    d = (x1 - x2) / s
    from scipy.special import ndtr

    phi = np.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
    expected_min = x1 * (1.0 - ndtr(d)) + x2 * ndtr(d) - s * phi
    return np.maximum(0.0, expected_min)


def _linear_gaussian_kernel_1d(
    variance: float,
    step: float,
    kernel_stddevs: float,
) -> Tuple[np.ndarray, int]:
    """Return exact Gaussian weights for a piecewise-linear grid function.

    A sampled Gaussian kernel aliases to a point mass when its standard
    deviation is smaller than ``step``. The weights here instead integrate
    every triangular linear-interpolation basis function against the
    continuous Gaussian law. Neighboring weights therefore remain of order
    ``std / step`` in the sub-grid regime rather than underflowing to zero.
    """
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("transition variance must be finite and positive")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("grid step must be finite and positive")

    scaled_std = math.sqrt(variance) / step
    # A hat centered at k has support [k - 1, k + 1], hence the extra cell
    # beyond the requested Gaussian-tail cutoff.
    radius = max(1, int(math.ceil(kernel_stddevs * scaled_std)) + 1)
    centers = np.arange(-radius, radius + 1, dtype=np.float64)
    left = centers - 1.0
    right = centers + 1.0

    from scipy.special import ndtr

    inv_std = 1.0 / scaled_std
    phi_scale = 1.0 / math.sqrt(2.0 * math.pi)

    def probability(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ndtr(b * inv_std) - ndtr(a * inv_std)

    def first_moment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        phi_a = np.exp(-0.5 * (a * inv_std) ** 2) * phi_scale
        phi_b = np.exp(-0.5 * (b * inv_std) ** 2) * phi_scale
        return scaled_std * (phi_a - phi_b)

    left_probability = probability(left, centers)
    right_probability = probability(centers, right)
    kernel = (
        first_moment(left, centers)
        - left * left_probability
        + right * right_probability
        - first_moment(centers, right)
    )
    # Roundoff in far tails can create tiny negative values or asymmetry.
    kernel = np.maximum(0.0, 0.5 * (kernel + kernel[::-1]))
    mass = float(kernel.sum())
    if not math.isfinite(mass) or mass <= 0.0:
        raise BoundaryGridError(
            "failed to construct a Gaussian transition kernel"
        )
    kernel /= mass
    return kernel, radius


def _convolve_axis(
    value: np.ndarray,
    variance: float,
    grid: np.ndarray,
    *,
    axis: int,
    kernel_stddevs: float,
) -> np.ndarray:
    step = float(grid[1] - grid[0])
    kernel, radius = _linear_gaussian_kernel_1d(
        variance,
        step,
        kernel_stddevs,
    )
    if radius >= grid.size // 2:
        raise BoundaryGridError(
            "Gaussian transition kernel is too wide for the state grid"
        )
    pad_width = [(0, 0)] * value.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(value, pad_width, mode="edge")
    kernel_shape = [1] * value.ndim
    kernel_shape[axis] = kernel.size
    from scipy.signal import fftconvolve

    convolved = fftconvolve(
        padded,
        kernel.reshape(kernel_shape),
        mode="same",
        axes=(axis,),
    )
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(radius, radius + grid.size)
    return convolved[tuple(slices)]


def gaussian_expectation_separable(
    value: np.ndarray,
    scaled_transition_var: VectorLike,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    *,
    kernel_stddevs: float = 6.0,
) -> np.ndarray:
    """Take a Gaussian expectation of the bilinear grid interpolant.

    The operation is separable and deterministic. Each one-dimensional kernel
    exactly integrates the piecewise-linear interpolant against its continuous
    Gaussian increment, including when the transition standard deviation is
    much smaller than a state-grid cell.
    """
    value = np.asarray(value, dtype=np.float64)
    scaled_var = _two_vector(
        scaled_transition_var,
        "scaled_transition_var",
        positive=True,
    )
    x1_grid = np.asarray(x1_grid, dtype=np.float64)
    x2_grid = np.asarray(x2_grid, dtype=np.float64)
    if value.shape != (x1_grid.size, x2_grid.size):
        raise ValueError("value shape must match x1_grid by x2_grid")
    if x1_grid.size < 2 or x2_grid.size < 2:
        raise ValueError("state grids need at least two points")
    if not np.allclose(np.diff(x1_grid), x1_grid[1] - x1_grid[0]) or not np.allclose(
        np.diff(x2_grid), x2_grid[1] - x2_grid[0]
    ):
        raise ValueError("state grids must be regular")
    first = _convolve_axis(
        value,
        float(scaled_var[0]),
        x1_grid,
        axis=0,
        kernel_stddevs=kernel_stddevs,
    )
    return _convolve_axis(
        first,
        float(scaled_var[1]),
        x2_grid,
        axis=1,
        kernel_stddevs=kernel_stddevs,
    )


def _direction_aware_state_grids(
    grid: RadialGittinsGrid,
    factors: np.ndarray,
    variance_schedule: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build state grids with a halo for cumulative posterior movement.

    Backward convolutions can propagate edge padding over many stages. The
    required guard band is therefore based on the total posterior-mean
    variance from stage zero to the horizon, separately for each scaled
    objective. The configured halo remains a lower bound.
    """
    total_base_variance = np.maximum(
        variance_schedule[0] - variance_schedule[-1],
        0.0,
    )
    cumulative_std = factors * np.sqrt(total_base_variance)
    halo = np.maximum(grid.state_halo, grid.kernel_stddevs * cumulative_std)

    core_span = (grid.z_max - grid.z_min) + (
        grid.delta_max - grid.delta_min
    ) / 2.0
    nominal_step = (core_span + 2.0 * grid.state_halo) / (
        grid.state_size - 1
    )
    x1_lower = grid.z_min + grid.delta_min / 2.0 - float(halo[0])
    x1_upper = grid.z_max + grid.delta_max / 2.0 + float(halo[0])
    x2_lower = grid.z_min - grid.delta_max / 2.0 - float(halo[1])
    x2_upper = grid.z_max - grid.delta_min / 2.0 + float(halo[1])
    x1_size = max(
        grid.state_size,
        int(math.ceil((x1_upper - x1_lower) / nominal_step)) + 1,
    )
    x2_size = max(
        grid.state_size,
        int(math.ceil((x2_upper - x2_lower) / nominal_step)) + 1,
    )
    return (
        np.linspace(x1_lower, x1_upper, x1_size, dtype=np.float64),
        np.linspace(x2_lower, x2_upper, x2_size, dtype=np.float64),
    )


def _boundary_from_q(
    q_values: np.ndarray,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    z_grid: np.ndarray,
    delta_grid: np.ndarray,
    *,
    margin_cells: int,
    monotonicity_tolerance: float,
) -> Tuple[np.ndarray, float]:
    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator(
        (x1_grid, x2_grid),
        q_values,
        method="linear",
        bounds_error=True,
    )
    # One batched interpolation over the whole (delta, z) sheet. The per-delta
    # call overhead of RegularGridInterpolator otherwise dominates the backward
    # recursion by an order of magnitude over the Gaussian expectations that do
    # the actual work.
    half_delta = delta_grid[:, None] / 2.0
    points = np.empty((delta_grid.size, z_grid.size, 2), dtype=np.float64)
    points[:, :, 0] = z_grid[None, :] + half_delta
    points[:, :, 1] = z_grid[None, :] - half_delta
    lines = np.asarray(
        interpolator(points.reshape(-1, 2)),
        dtype=np.float64,
    ).reshape(delta_grid.size, z_grid.size)

    row_violations = np.maximum(0.0, -np.min(np.diff(lines, axis=1), axis=1))
    maximum_violation = float(np.max(row_violations))
    nonnegative = lines >= 0.0
    has_root = nonnegative.any(axis=1)
    upper = np.argmax(nonnegative, axis=1)

    _reject_unusable_boundary_rows(
        delta_grid,
        z_grid,
        row_violations=row_violations,
        has_root=has_root,
        upper=upper,
        margin_cells=margin_cells,
        monotonicity_tolerance=monotonicity_tolerance,
    )

    rows = np.arange(delta_grid.size)
    lower = upper - 1
    line_lower = lines[rows, lower]
    denominator = lines[rows, upper] - line_lower
    # A flat bracket puts the root at the upper node, which is weight one.
    weight = np.divide(
        -line_lower,
        denominator,
        out=np.ones_like(denominator),
        where=denominator != 0.0,
    )
    boundaries = z_grid[lower] + weight * (z_grid[upper] - z_grid[lower])
    return boundaries, maximum_violation


def _reject_unusable_boundary_rows(
    delta_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    row_violations: np.ndarray,
    has_root: np.ndarray,
    upper: np.ndarray,
    margin_cells: int,
    monotonicity_tolerance: float,
) -> None:
    """Raise for the first delta whose root search cannot be trusted."""
    not_monotone = row_violations > monotonicity_tolerance
    at_or_below_min = has_root & (upper == 0)
    near_edge = (
        has_root
        & (upper != 0)
        & ((upper < margin_cells) | (upper >= z_grid.size - margin_cells))
    )
    unusable = not_monotone | ~has_root | at_or_below_min | near_edge
    if not unusable.any():
        return
    row = int(np.argmax(unusable))
    delta = float(delta_grid[row])
    if not_monotone[row]:
        raise BoundaryGridError(
            "q(z, delta) is not numerically nondecreasing in z; "
            f"violation {float(row_violations[row]):.3g} exceeds "
            f"{monotonicity_tolerance:.3g}"
        )
    if not has_root[row]:
        raise BoundaryGridError(
            f"no boundary root before z_max for delta={delta:.6g}; "
            "widen the z grid"
        )
    if at_or_below_min[row]:
        raise BoundaryGridError(
            f"boundary root is at or below z_min for delta={delta:.6g}; "
            "widen the z grid"
        )
    raise BoundaryGridError(
        "boundary root is too close to a z-grid edge for "
        f"delta={delta:.6g}: crossing_index={int(upper[row])}, "
        f"required=[{margin_cells}, {z_grid.size - margin_cells}), "
        f"z_range=[{z_grid[0]:.6g}, {z_grid[-1]:.6g}]"
    )


def build_radial_gittins_boundary_table(
    *,
    direction: VectorLike,
    effective_pull_cost: float,
    initial_var: VectorLike,
    obs_noise_var: VectorLike,
    horizon: int,
    grid: RadialGittinsGrid = RadialGittinsGrid(),
) -> RadialGittinsBoundaryTable:
    """Solve the finite-horizon radial retirement DP."""
    direction_array = _direction(direction)
    initial = _two_vector(initial_var, "initial_var", positive=True)
    noise = _two_vector(obs_noise_var, "obs_noise_var", positive=True)
    cost = float(effective_pull_cost)
    if not math.isfinite(cost) or cost <= 0.0:
        raise ValueError("effective_pull_cost must be finite and strictly positive")
    if int(horizon) != horizon or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    horizon = int(horizon)

    z_grid = grid.z_grid
    delta_grid = grid.delta_grid
    variances = posterior_variance_schedule(initial, noise, horizon)
    factors = _scaling_factors(direction_array)
    x1_grid, x2_grid = _direction_aware_state_grids(
        grid,
        factors,
        variances,
    )
    terminal_scaled_var = factors * factors * variances[horizon]
    value = _terminal_value(x1_grid, x2_grid, terminal_scaled_var)
    boundaries = np.empty((horizon + 1, delta_grid.size), dtype=np.float64)
    boundaries[horizon] = -_terminal_imbalance(delta_grid, terminal_scaled_var)
    maximum_violation = 0.0

    for stage in range(horizon - 1, -1, -1):
        q_base = np.maximum(variances[stage] - variances[stage + 1], 0.0)
        scaled_q = factors * factors * q_base
        expectation = gaussian_expectation_separable(
            value,
            scaled_q,
            x1_grid,
            x2_grid,
            kernel_stddevs=grid.kernel_stddevs,
        )
        q_values = expectation - cost
        boundaries[stage], violation = _boundary_from_q(
            q_values,
            x1_grid,
            x2_grid,
            z_grid,
            delta_grid,
            margin_cells=grid.boundary_margin_cells,
            monotonicity_tolerance=grid.monotonicity_tolerance,
        )
        maximum_violation = max(maximum_violation, violation)
        value = np.maximum(0.0, q_values)

    return RadialGittinsBoundaryTable(
        direction=tuple(float(x) for x in direction_array),
        effective_pull_cost=cost,
        initial_var=tuple(float(x) for x in initial),
        obs_noise_var=tuple(float(x) for x in noise),
        horizon=horizon,
        grid=grid,
        boundaries=boundaries,
        max_monotonicity_violation=maximum_violation,
    )


@dataclass(frozen=True)
class BoundaryCacheStats:
    """Immutable counters for one boundary-cache instance."""

    builds: int = 0
    memory_hits: int = 0
    disk_hits: int = 0
    disk_misses: int = 0
    corruptions: int = 0
    read_failures: int = 0
    write_failures: int = 0


class _BoundaryCacheDataError(ValueError):
    """An on-disk entry failed validation and may be safely regenerated."""


def _cache_float(value: float) -> str:
    """Return an exact, JSON-stable representation of a finite float."""
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("cache-key floats must be finite")
    if resolved == 0.0:
        resolved = 0.0  # Treat -0.0 like the existing tuple cache does.
    return resolved.hex()


def _grid_cache_payload(grid: RadialGittinsGrid) -> Dict[str, object]:
    return {
        "z_min": _cache_float(grid.z_min),
        "z_max": _cache_float(grid.z_max),
        "z_size": int(grid.z_size),
        "delta_min": _cache_float(grid.delta_min),
        "delta_max": _cache_float(grid.delta_max),
        "delta_size": int(grid.delta_size),
        "state_size": int(grid.state_size),
        "state_halo": _cache_float(grid.state_halo),
        "kernel_stddevs": _cache_float(grid.kernel_stddevs),
        "boundary_margin_cells": int(grid.boundary_margin_cells),
        "monotonicity_tolerance": _cache_float(
            grid.monotonicity_tolerance
        ),
    }


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _cache_array(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=_CACHE_ARRAY_DTYPE)


def _array_manifest(value: np.ndarray) -> Dict[str, object]:
    array = _cache_array(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


class RadialGittinsBoundaryCache:
    """Memory cache with optional validated, versioned disk persistence.

    Disk entries contain only the compact online boundary schedules, never the
    two-dimensional DP work arrays.  Publishing is atomic, and disk failures
    are treated as cache misses so caching cannot make a valid solve fail.

    When both objective posteriors have identical variance schedules, swapping
    the objectives is an exact symmetry of the retirement DP.  A table for
    ``(d1, d2)`` can then serve ``(d2, d1)`` on the reflected delta grid via
    ``b_swapped(delta) = b_original(-delta)``.  Mirrored caller-facing tables
    are aliases and do not count as additional DP builds in :meth:`__len__`.

    Counters describe logical ``get`` requests: a reflected-file load is one
    disk hit, and a reflected in-memory alias is one memory hit. Instances are
    intended for the single-threaded replay scripts; separate threads should
    use separate cache instances that share the same atomic disk cache.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        disk_read: bool = True,
        disk_write: bool = True,
    ) -> None:
        self.cache_dir = (
            None if cache_dir is None else Path(cache_dir).expanduser()
        )
        self.disk_read = bool(disk_read and self.cache_dir is not None)
        self.disk_write = bool(disk_write and self.cache_dir is not None)
        self._tables: Dict[Tuple[object, ...], RadialGittinsBoundaryTable] = {}
        self._mirrored_tables: Dict[
            Tuple[object, ...], RadialGittinsBoundaryTable
        ] = {}
        self._stats: Dict[str, int] = {
            name: 0 for name in BoundaryCacheStats.__dataclass_fields__
        }
        self._warned: set[str] = set()

    @property
    def stats(self) -> BoundaryCacheStats:
        """Return a point-in-time copy of the cache counters."""
        return BoundaryCacheStats(**self._stats)

    def stats_snapshot(self) -> BoundaryCacheStats:
        """Method-form alias useful when taking before/after snapshots."""
        return self.stats

    def _increment(self, name: str) -> None:
        self._stats[name] += 1

    def _warn_once(self, category: str, message: str) -> None:
        if category not in self._warned:
            self._warned.add(category)
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    @staticmethod
    def _radial_key(
        direction: Tuple[float, float],
        effective_pull_cost: float,
        initial_var: Tuple[float, float],
        obs_noise_var: Tuple[float, float],
        horizon: int,
        grid: RadialGittinsGrid,
    ) -> Tuple[object, ...]:
        return (
            direction,
            effective_pull_cost,
            initial_var,
            obs_noise_var,
            horizon,
            grid,
        )


    @staticmethod
    def _radial_disk_key(key: Tuple[object, ...]) -> Dict[str, object]:
        direction, cost, initial, noise, horizon, grid = key
        assert isinstance(grid, RadialGittinsGrid)
        return {
            "format": "agentopt.radial-gittins-boundary",
            "schema_version": RADIAL_BOUNDARY_CACHE_SCHEMA_VERSION,
            "solver_version": RADIAL_BOUNDARY_SOLVER_VERSION,
            "kind": "radial",
            "direction": [_cache_float(x) for x in direction],
            "effective_pull_cost": _cache_float(cost),
            "initial_var": [_cache_float(x) for x in initial],
            "obs_noise_var": [_cache_float(x) for x in noise],
            "horizon": int(horizon),
            "grid": _grid_cache_payload(grid),
            "output_dtype": _CACHE_ARRAY_DTYPE.str,
        }


    def _entry_path(
        self,
        kind: str,
        disk_key: Mapping[str, object],
    ) -> Tuple[Path, str]:
        if self.cache_dir is None:
            raise RuntimeError("a disk-cache path requires cache_dir")
        digest = hashlib.sha256(
            b"agentopt-radial-boundary-key\0"
            + _canonical_json_bytes(disk_key)
        ).hexdigest()
        path = (
            self.cache_dir
            / f"schema-{RADIAL_BOUNDARY_CACHE_SCHEMA_VERSION}"
            / f"solver-{RADIAL_BOUNDARY_SOLVER_VERSION}"
            / kind
            / digest[:2]
            / f"{digest}.npz"
        )
        return path, digest

    def _record_corruption(self, path: Path, error: Exception) -> None:
        self._increment("corruptions")
        self._warn_once(
            "corruption",
            f"Ignoring invalid radial-Gittins cache entry {path}: {error}",
        )

    def _read_entry(
        self,
        *,
        kind: str,
        disk_key: Mapping[str, object],
        expected_boundary_shape: Tuple[int, ...],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]] | None:
        if not self.disk_read:
            return None
        path, digest = self._entry_path(kind, disk_key)
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {"manifest", "boundaries"}
                if set(archive.files) != required:
                    raise _BoundaryCacheDataError(
                        f"expected members {sorted(required)}, got "
                        f"{sorted(archive.files)}"
                    )
                encoded_manifest = np.asarray(archive["manifest"])
                if encoded_manifest.dtype != np.dtype(np.uint8) or (
                    encoded_manifest.ndim != 1
                ):
                    raise _BoundaryCacheDataError(
                        "manifest must be a one-dimensional uint8 array"
                    )
                manifest = json.loads(
                    encoded_manifest.tobytes().decode("utf-8")
                )
                if not isinstance(manifest, dict):
                    raise _BoundaryCacheDataError("manifest must be an object")
                if manifest.get("key") != disk_key:
                    raise _BoundaryCacheDataError("cache key mismatch")
                if manifest.get("key_sha256") != digest:
                    raise _BoundaryCacheDataError("cache digest mismatch")
                array_specs = manifest.get("arrays")
                if not isinstance(array_specs, dict):
                    raise _BoundaryCacheDataError("missing array manifest")

                arrays: Dict[str, np.ndarray] = {}
                for name in required - {"manifest"}:
                    array = np.asarray(archive[name])
                    if array.dtype.str != _CACHE_ARRAY_DTYPE.str:
                        raise _BoundaryCacheDataError(
                            f"{name} has unexpected dtype {array.dtype.str}"
                        )
                    expected_spec = _array_manifest(array)
                    if array_specs.get(name) != expected_spec:
                        raise _BoundaryCacheDataError(
                            f"{name} failed shape or checksum validation"
                        )
                    if not np.all(np.isfinite(array)):
                        raise _BoundaryCacheDataError(
                            f"{name} contains non-finite values"
                        )
                    arrays[name] = np.array(array, dtype=np.float64, copy=True)

            if arrays["boundaries"].shape != expected_boundary_shape:
                raise _BoundaryCacheDataError(
                    "boundaries have the wrong shape for the cache key"
                )
            return arrays, manifest
        except FileNotFoundError:
            return None
        except (OSError, PermissionError) as error:
            self._increment("read_failures")
            self._warn_once(
                "read",
                f"Cannot read radial-Gittins disk cache {path}: {error}",
            )
            return None
        except (
            BadZipFile,
            EOFError,
            KeyError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as error:
            self._record_corruption(path, error)
            return None

    def _write_entry(
        self,
        *,
        kind: str,
        disk_key: Mapping[str, object],
        arrays: Mapping[str, np.ndarray],
        extra_manifest: Mapping[str, object] | None = None,
    ) -> None:
        if not self.disk_write:
            return
        path, digest = self._entry_path(kind, disk_key)
        normalized_arrays = {
            name: _cache_array(value) for name, value in arrays.items()
        }
        manifest: Dict[str, object] = {
            "key": dict(disk_key),
            "key_sha256": digest,
            "arrays": {
                name: _array_manifest(value)
                for name, value in normalized_arrays.items()
            },
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        encoded_manifest = np.frombuffer(
            _canonical_json_bytes(manifest), dtype=np.uint8
        )
        temp_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=path.parent,
                prefix=f".{digest}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temp_path = Path(output.name)
                # Boundary schedules are already compact. Avoid compression
                # and durability fsyncs here: on cloud-synced filesystems they
                # can cost more than the DP solve itself. Closing before the
                # atomic replace keeps concurrent readers safe; a machine
                # crash can at worst leave a missing/corrupt cache entry,
                # which the validated read path rebuilds.
                np.savez(
                    output,
                    manifest=encoded_manifest,
                    **normalized_arrays,
                )
            os.replace(temp_path, path)
            temp_path = None
        except Exception as error:
            self._increment("write_failures")
            self._warn_once(
                "write",
                f"Cannot write radial-Gittins disk cache {path}: {error}",
            )
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _reflected_grid(grid: RadialGittinsGrid) -> RadialGittinsGrid:
        """Return the grid induced by exchanging objective coordinates."""
        return replace(
            grid,
            delta_min=-grid.delta_max,
            delta_max=-grid.delta_min,
        )

    @staticmethod
    def _mirrored_table(
        source: RadialGittinsBoundaryTable,
        *,
        direction: Tuple[float, float],
        initial_var: Tuple[float, float],
        obs_noise_var: Tuple[float, float],
        grid: RadialGittinsGrid,
    ) -> RadialGittinsBoundaryTable:
        """Return a metadata-correct objective-swapped view of ``source``."""
        return RadialGittinsBoundaryTable(
            direction=direction,
            effective_pull_cost=source.effective_pull_cost,
            initial_var=initial_var,
            obs_noise_var=obs_noise_var,
            horizon=source.horizon,
            grid=grid,
            boundaries=source.boundaries[:, ::-1],
            max_monotonicity_violation=source.max_monotonicity_violation,
            _mirrored_source=source,
        )

    def _load_radial(
        self,
        key: Tuple[object, ...],
    ) -> RadialGittinsBoundaryTable | None:
        direction, cost, initial, noise, horizon, grid = key
        assert isinstance(grid, RadialGittinsGrid)
        disk_key = self._radial_disk_key(key)
        loaded = self._read_entry(
            kind="radial",
            disk_key=disk_key,
            expected_boundary_shape=(int(horizon) + 1, grid.delta_size),
        )
        if loaded is None:
            return None
        arrays, manifest = loaded
        try:
            violation = float.fromhex(str(manifest["max_monotonicity_violation"]))
            return RadialGittinsBoundaryTable(
                direction=direction,
                effective_pull_cost=float(cost),
                initial_var=initial,
                obs_noise_var=noise,
                horizon=int(horizon),
                grid=grid,
                boundaries=arrays["boundaries"],
                max_monotonicity_violation=violation,
            )
        except (KeyError, TypeError, ValueError) as error:
            path, _ = self._entry_path("radial", disk_key)
            self._record_corruption(path, error)
            return None


    def get(
        self,
        *,
        direction: VectorLike,
        effective_pull_cost: float,
        initial_var: VectorLike,
        obs_noise_var: VectorLike,
        horizon: int,
        grid: RadialGittinsGrid,
    ) -> RadialGittinsBoundaryTable:
        direction_array = _direction(direction)
        initial = _two_vector(initial_var, "initial_var", positive=True)
        noise = _two_vector(obs_noise_var, "obs_noise_var", positive=True)
        direction_key = tuple(float(x) for x in direction_array)
        initial_key = tuple(float(x) for x in initial)
        noise_key = tuple(float(x) for x in noise)
        cost_key = float(effective_pull_cost)
        if not math.isfinite(cost_key) or cost_key <= 0.0:
            raise ValueError("effective_pull_cost must be finite and strictly positive")
        if int(horizon) != horizon or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        horizon_key = int(horizon)
        key = self._radial_key(
            direction_key,
            cost_key,
            initial_key,
            noise_key,
            horizon_key,
            grid,
        )
        table = self._tables.get(key)
        if table is not None:
            self._increment("memory_hits")
            return table
        table = self._mirrored_tables.get(key)
        if table is not None:
            self._increment("memory_hits")
            return table

        # Exact equality is intentional: treating merely close posterior
        # variances as exchangeable would make this optimization approximate.
        exchangeable = (
            initial_key[0] == initial_key[1]
            and noise_key[0] == noise_key[1]
            and direction_key[0] != direction_key[1]
        )
        reflected_key: Tuple[object, ...] | None = None
        if exchangeable:
            reflected_key = self._radial_key(
                (direction_key[1], direction_key[0]),
                cost_key,
                initial_key,
                noise_key,
                horizon_key,
                self._reflected_grid(grid),
            )
            source = self._tables.get(reflected_key)
            if source is not None:
                table = self._mirrored_table(
                    source,
                    direction=direction_key,
                    initial_var=initial_key,
                    obs_noise_var=noise_key,
                    grid=grid,
                )
                self._mirrored_tables[key] = table
                self._increment("memory_hits")
                return table

        table = self._load_radial(key)
        if table is not None:
            self._tables[key] = table
            self._increment("disk_hits")
            return table
        if reflected_key is not None:
            source = self._load_radial(reflected_key)
            if source is not None:
                self._tables[reflected_key] = source
                table = self._mirrored_table(
                    source,
                    direction=direction_key,
                    initial_var=initial_key,
                    obs_noise_var=noise_key,
                    grid=grid,
                )
                self._mirrored_tables[key] = table
                self._increment("disk_hits")
                return table
        if self.disk_read:
            self._increment("disk_misses")

        table = build_radial_gittins_boundary_table(
            direction=direction_array,
            effective_pull_cost=cost_key,
            initial_var=initial,
            obs_noise_var=noise,
            horizon=horizon_key,
            grid=grid,
        )
        self._tables[key] = table
        self._increment("builds")
        self._write_entry(
            kind="radial",
            disk_key=self._radial_disk_key(key),
            arrays={"boundaries": table.boundaries},
            extra_manifest={
                "max_monotonicity_violation": _cache_float(
                    table.max_monotonicity_violation
                )
            },
        )
        return table


    def __len__(self) -> int:
        return len(self._tables)

__all__ = [
    "BoundaryCacheStats",
    "BoundaryGridError",
    "RADIAL_BOUNDARY_CACHE_SCHEMA_VERSION",
    "RADIAL_BOUNDARY_SOLVER_VERSION",
    "RadialGittinsBoundaryCache",
    "RadialGittinsBoundaryTable",
    "RadialGittinsGrid",
    "build_radial_gittins_boundary_table",
    "direction_aware_grid",
    "gaussian_expectation_separable",
    "posterior_variance_schedule",
    "radial_posterior_coordinates",
    "radial_transition_covariance",
    "terminal_expected_radial_utility",
]
