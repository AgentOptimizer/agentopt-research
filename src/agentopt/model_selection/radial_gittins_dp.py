"""Deterministic boundary tables for the two-objective radial-Gittins policy.

The retirement dynamic program is solved in centered direction-scaled
coordinates ``x1 = m1 - alpha`` and ``x2 = m2 - alpha``. In these coordinates
the two posterior-mean increments are independent, so each backward step is
two deterministic one-dimensional Gaussian expectations of the linearly
interpolated value function. The resulting value function is converted to the
online ``b_n(delta)`` boundary table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Sequence, Tuple, Union

import numpy as np

from .radial_gittins import expected_min_of_two_normals


VectorLike = Union[float, Sequence[float], np.ndarray]


class BoundaryGridError(RuntimeError):
    """Raised when a numerical grid cannot contain a reliable DP boundary."""


def _two_vector(value: VectorLike, name: str, *, positive: bool) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array, 2)
    if array.shape != (2,):
        raise ValueError(f"{name} must be a scalar or a length-2 vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array.copy()


def _direction(value: VectorLike) -> np.ndarray:
    result = _two_vector(value, "direction", positive=True)
    if not math.isclose(float(result.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("direction components must sum to one")
    return result


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
        return np.linspace(self.z_min, self.z_max, self.z_size, dtype=np.float64)

    @property
    def delta_grid(self) -> np.ndarray:
        return np.linspace(
            self.delta_min,
            self.delta_max,
            self.delta_size,
            dtype=np.float64,
        )

    @property
    def x1_grid(self) -> np.ndarray:
        lower = self.z_min + self.delta_min / 2.0 - self.state_halo
        upper = self.z_max + self.delta_max / 2.0 + self.state_halo
        return np.linspace(lower, upper, self.state_size, dtype=np.float64)

    @property
    def x2_grid(self) -> np.ndarray:
        lower = self.z_min - self.delta_max / 2.0 - self.state_halo
        upper = self.z_max - self.delta_min / 2.0 + self.state_halo
        return np.linspace(lower, upper, self.state_size, dtype=np.float64)


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

    factors = float(np.max(direction_array)) / direction_array
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

    def __post_init__(self) -> None:
        array = np.asarray(self.boundaries, dtype=np.float64).copy()
        expected_shape = (self.horizon + 1, self.grid.delta_size)
        if array.shape != expected_shape:
            raise ValueError(
                f"boundaries must have shape {expected_shape}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("boundaries must be finite")
        array.setflags(write=False)
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
    factors = float(np.max(direction_array)) / direction_array
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
    factors = float(np.max(direction_array)) / direction_array
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
    boundaries = np.empty(delta_grid.size, dtype=np.float64)
    maximum_violation = 0.0
    for delta_index, delta in enumerate(delta_grid):
        points = np.column_stack(
            (z_grid + delta / 2.0, z_grid - delta / 2.0)
        )
        line = np.asarray(interpolator(points), dtype=np.float64)
        violation = max(0.0, -float(np.min(np.diff(line))))
        maximum_violation = max(maximum_violation, violation)
        if violation > monotonicity_tolerance:
            raise BoundaryGridError(
                "q(z, delta) is not numerically nondecreasing in z; "
                f"violation {violation:.3g} exceeds {monotonicity_tolerance:.3g}"
            )
        nonnegative = np.flatnonzero(line >= 0.0)
        if nonnegative.size == 0:
            raise BoundaryGridError(
                "no boundary root before z_max for "
                f"delta={delta:.6g}; widen the z grid"
            )
        upper = int(nonnegative[0])
        if upper == 0:
            raise BoundaryGridError(
                "boundary root is at or below z_min for "
                f"delta={delta:.6g}; widen the z grid"
            )
        if upper < margin_cells or upper >= z_grid.size - margin_cells:
            raise BoundaryGridError(
                "boundary root is too close to a z-grid edge for "
                f"delta={delta:.6g}: crossing_index={upper}, "
                f"required=[{margin_cells}, {z_grid.size - margin_cells}), "
                f"z_range=[{z_grid[0]:.6g}, {z_grid[-1]:.6g}]"
            )
        lower = upper - 1
        denominator = float(line[upper] - line[lower])
        if denominator == 0.0:
            boundaries[delta_index] = float(z_grid[upper])
        else:
            weight = -float(line[lower]) / denominator
            boundaries[delta_index] = float(
                z_grid[lower] + weight * (z_grid[upper] - z_grid[lower])
            )
    return boundaries, maximum_violation


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
    factors = float(np.max(direction_array)) / direction_array
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


class RadialGittinsBoundaryCache:
    """In-process cache keyed by every setting that changes a DP table."""

    def __init__(self) -> None:
        self._tables: Dict[Tuple[object, ...], RadialGittinsBoundaryTable] = {}

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
        key = (
            tuple(float(x) for x in direction_array),
            float(effective_pull_cost),
            tuple(float(x) for x in initial),
            tuple(float(x) for x in noise),
            int(horizon),
            grid,
        )
        table = self._tables.get(key)
        if table is None:
            table = build_radial_gittins_boundary_table(
                direction=direction_array,
                effective_pull_cost=effective_pull_cost,
                initial_var=initial,
                obs_noise_var=noise,
                horizon=horizon,
                grid=grid,
            )
            self._tables[key] = table
        return table

    def __len__(self) -> int:
        return len(self._tables)


__all__ = [
    "BoundaryGridError",
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
