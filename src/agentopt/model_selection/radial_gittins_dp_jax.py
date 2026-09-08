"""Optional batched JAX backend for the two-objective radial-Gittins DP.

JAX is imported only when :func:`build_radial_gittins_boundary_tables_jax`
is called.  The public table and grid types remain the NumPy types from
``radial_gittins_dp`` so callers do not acquire a JAX dependency unless they
explicitly select this backend.

The batch shares one direction and horizon, but every member may have its own
cost, posterior variance schedule, and numerical grid.  Active state arrays
are stored in the upper-left corner of a common array.  Inactive cells are an
exact edge extension of the active array, while each Gaussian kernel is
zero-padded around a common center.  Consequently the active part of every
convolution is the same discrete operation it would receive in an unpadded
solve; padding is used only to give JAX one static batch shape.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator, Sequence, Tuple

import numpy as np

from .radial_gittins_dp import (
    BoundaryGridError,
    RadialGittinsBoundaryTable,
    RadialGittinsGrid,
    VectorLike,
    _direction,
    _direction_aware_state_grids,
    _linear_gaussian_kernel_1d,
    _reject_unusable_boundary_rows,
    _scaling_factors,
    _terminal_imbalance,
    _terminal_value,
    _two_vector,
    posterior_variance_schedule,
)


@dataclass(frozen=True)
class RadialGittinsJaxTableSpec:
    """Parameters that may differ between members of one JAX batch."""

    effective_pull_cost: float
    initial_var: VectorLike
    obs_noise_var: VectorLike
    grid: RadialGittinsGrid = RadialGittinsGrid()


@dataclass(frozen=True)
class _PreparedSpec:
    cost: float
    initial_var: np.ndarray
    obs_noise_var: np.ndarray
    grid: RadialGittinsGrid
    variances: np.ndarray
    x1_grid: np.ndarray
    x2_grid: np.ndarray
    terminal_value: np.ndarray
    terminal_boundary: np.ndarray
    kernels1: Tuple[np.ndarray, ...]
    kernels2: Tuple[np.ndarray, ...]
    radii1: Tuple[int, ...]
    radii2: Tuple[int, ...]
    interpolation: Tuple[np.ndarray, ...]


def _jax_modules() -> tuple[Any, Any]:
    """Import JAX lazily and return ``(jax, jax.numpy)``."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - depends on optional env
        raise ImportError(
            "the JAX radial-Gittins backend requires the 'radial-jax' "
            "optional dependencies (install agentopt-research[radial-jax])"
        ) from error
    return jax, jnp


@contextlib.contextmanager
def _jax_float64_context(jax: Any, jnp: Any) -> Iterator[None]:
    """Enable float64 locally and fail rather than silently downcast."""

    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is not None:
        context = enable_x64()
    else:  # Compatibility with older supported JAX releases.
        previous = bool(jax.config.read("jax_enable_x64"))

        @contextlib.contextmanager
        def temporary_global_setting() -> Iterator[None]:
            jax.config.update("jax_enable_x64", True)
            try:
                yield
            finally:
                jax.config.update("jax_enable_x64", previous)

        context = temporary_global_setting()

    with context:
        probe = jnp.asarray(0.0, dtype=jnp.float64)
        if np.dtype(probe.dtype) != np.dtype(np.float64):
            raise RuntimeError(
                "JAX float64 support could not be enabled; refusing to run "
                "the radial-Gittins DP in reduced precision"
            )
        yield


def _interpolation_coordinates(
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    z_grid: np.ndarray,
    delta_grid: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Precompute bilinear indices and weights for the ``(delta, z)`` sheet."""

    half_delta = delta_grid[:, None] / 2.0
    x1 = z_grid[None, :] + half_delta
    x2 = z_grid[None, :] - half_delta

    def axis_coordinates(
        grid: np.ndarray, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.any(points < grid[0]) or np.any(points > grid[-1]):
            raise ValueError(
                "boundary interpolation points lie outside the state grid"
            )
        upper = np.searchsorted(grid, points, side="right")
        upper = np.clip(upper, 1, grid.size - 1)
        lower = upper - 1
        denominator = grid[upper] - grid[lower]
        weight = (points - grid[lower]) / denominator
        return lower.astype(np.int32), weight.astype(np.float64)

    x1_lower, x1_weight = axis_coordinates(x1_grid, x1)
    x2_lower, x2_weight = axis_coordinates(x2_grid, x2)
    return x1_lower, x2_lower, x1_weight, x2_weight


def _prepare_spec(
    spec: RadialGittinsJaxTableSpec,
    *,
    factors: np.ndarray,
    horizon: int,
) -> _PreparedSpec:
    if not isinstance(spec, RadialGittinsJaxTableSpec):
        raise TypeError("each spec must be a RadialGittinsJaxTableSpec")
    if not isinstance(spec.grid, RadialGittinsGrid):
        raise TypeError("spec.grid must be a RadialGittinsGrid")

    cost = float(spec.effective_pull_cost)
    if not math.isfinite(cost) or cost <= 0.0:
        raise ValueError("effective_pull_cost must be finite and strictly positive")
    initial = _two_vector(spec.initial_var, "initial_var", positive=True)
    noise = _two_vector(spec.obs_noise_var, "obs_noise_var", positive=True)
    variances = posterior_variance_schedule(initial, noise, horizon)
    x1_grid, x2_grid = _direction_aware_state_grids(
        spec.grid,
        factors,
        variances,
    )

    kernels1 = []
    kernels2 = []
    radii1 = []
    radii2 = []
    steps = (float(x1_grid[1] - x1_grid[0]), float(x2_grid[1] - x2_grid[0]))
    for stage in range(horizon):
        base_variance = np.maximum(
            variances[stage] - variances[stage + 1],
            0.0,
        )
        scaled_variance = factors * factors * base_variance
        kernel1, radius1 = _linear_gaussian_kernel_1d(
            float(scaled_variance[0]),
            steps[0],
            spec.grid.kernel_stddevs,
        )
        kernel2, radius2 = _linear_gaussian_kernel_1d(
            float(scaled_variance[1]),
            steps[1],
            spec.grid.kernel_stddevs,
        )
        if radius1 >= x1_grid.size // 2 or radius2 >= x2_grid.size // 2:
            raise BoundaryGridError(
                "Gaussian transition kernel is too wide for the state grid"
            )
        kernels1.append(kernel1)
        kernels2.append(kernel2)
        radii1.append(radius1)
        radii2.append(radius2)

    terminal_scaled_var = factors * factors * variances[horizon]
    terminal_value = _terminal_value(x1_grid, x2_grid, terminal_scaled_var)
    terminal_boundary = -_terminal_imbalance(
        spec.grid.delta_grid,
        terminal_scaled_var,
    )
    interpolation = _interpolation_coordinates(
        x1_grid,
        x2_grid,
        spec.grid.z_grid,
        spec.grid.delta_grid,
    )
    return _PreparedSpec(
        cost=cost,
        initial_var=initial,
        obs_noise_var=noise,
        grid=spec.grid,
        variances=variances,
        x1_grid=x1_grid,
        x2_grid=x2_grid,
        terminal_value=terminal_value,
        terminal_boundary=terminal_boundary,
        kernels1=tuple(kernels1),
        kernels2=tuple(kernels2),
        radii1=tuple(radii1),
        radii2=tuple(radii2),
        interpolation=interpolation,
    )


def _edge_pad_state(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Embed an active state array and exactly edge-extend inactive cells."""

    n1, n2 = value.shape
    maximum1, maximum2 = shape
    return np.pad(
        value,
        ((0, maximum1 - n1), (0, maximum2 - n2)),
        mode="edge",
    )


def _center_pad_kernel(
    kernel: np.ndarray,
    radius: int,
    maximum_radius: int,
) -> np.ndarray:
    result = np.zeros(2 * maximum_radius + 1, dtype=np.float64)
    start = maximum_radius - radius
    result[start : start + kernel.size] = kernel
    return result


@lru_cache(maxsize=32)
def _compiled_solver(
    signature: tuple[int, int, int, int, int, int, int, int],
) -> Any:
    """Return a compiled solver specialized only by padded array shapes."""

    (
        batch_size,
        horizon,
        maximum1,
        maximum2,
        maximum_z,
        maximum_delta,
        kernel_size1,
        kernel_size2,
    ) = signature
    del batch_size, horizon  # Encoded by argument shapes; retained in the key.
    radius1 = (kernel_size1 - 1) // 2
    radius2 = (kernel_size2 - 1) // 2
    jax, jnp = _jax_modules()

    state_rows = jnp.arange(maximum1, dtype=jnp.int32)
    state_columns = jnp.arange(maximum2, dtype=jnp.int32)
    z_columns = jnp.arange(maximum_z, dtype=jnp.int32)
    delta_rows = jnp.arange(maximum_delta, dtype=jnp.int32)

    def edge_extend_one(value: Any, n1: Any, n2: Any) -> Any:
        row_indices = jnp.minimum(state_rows, n1 - 1)
        column_indices = jnp.minimum(state_columns, n2 - 1)
        return value[row_indices[:, None], column_indices[None, :]]

    edge_extend_batch = jax.vmap(edge_extend_one)

    from scipy.fft import next_fast_len

    fft_size1 = int(
        next_fast_len(maximum1 + 4 * radius1, real=True)
    )
    fft_size2 = int(
        next_fast_len(maximum2 + 4 * radius2, real=True)
    )

    def convolve_batch(value: Any, kernel1: Any, kernel2: Any) -> Any:
        padded1 = jnp.pad(
            value,
            ((0, 0), (radius1, radius1), (0, 0)),
            mode="edge",
        )
        value_spectrum1 = jnp.fft.rfft(padded1, n=fft_size1, axis=1)
        kernel_spectrum1 = jnp.fft.rfft(kernel1, n=fft_size1, axis=1)
        full1 = jnp.fft.irfft(
            value_spectrum1 * kernel_spectrum1[:, :, None],
            n=fft_size1,
            axis=1,
        )
        first = full1[
            :, 2 * radius1 : 2 * radius1 + maximum1, :
        ]
        padded2 = jnp.pad(
            first,
            ((0, 0), (0, 0), (radius2, radius2)),
            mode="edge",
        )
        value_spectrum2 = jnp.fft.rfft(padded2, n=fft_size2, axis=2)
        kernel_spectrum2 = jnp.fft.rfft(kernel2, n=fft_size2, axis=1)
        full2 = jnp.fft.irfft(
            value_spectrum2 * kernel_spectrum2[:, None, :],
            n=fft_size2,
            axis=2,
        )
        return full2[
            :, :, 2 * radius2 : 2 * radius2 + maximum2
        ]

    def boundary_one(
        q_values: Any,
        x1_lower: Any,
        x2_lower: Any,
        x1_weight: Any,
        x2_weight: Any,
        z_values: Any,
        z_count: Any,
        delta_count: Any,
    ) -> tuple[Any, Any, Any, Any]:
        x1_upper = x1_lower + 1
        x2_upper = x2_lower + 1
        value00 = q_values[x1_lower, x2_lower]
        value10 = q_values[x1_upper, x2_lower]
        value01 = q_values[x1_lower, x2_upper]
        value11 = q_values[x1_upper, x2_upper]
        one_minus_x1 = 1.0 - x1_weight
        one_minus_x2 = 1.0 - x2_weight
        lines = (
            one_minus_x1 * one_minus_x2 * value00
            + x1_weight * one_minus_x2 * value10
            + one_minus_x1 * x2_weight * value01
            + x1_weight * x2_weight * value11
        )

        valid_z = z_columns < z_count
        valid_delta = delta_rows < delta_count
        nonnegative = (lines >= 0.0) & valid_z[None, :]
        has_root = jnp.any(nonnegative, axis=1)
        upper = jnp.argmax(nonnegative, axis=1).astype(jnp.int32)

        differences = jnp.diff(lines, axis=1)
        valid_differences = z_columns[:-1] < (z_count - 1)
        minimum_difference = jnp.min(
            jnp.where(valid_differences[None, :], differences, jnp.inf),
            axis=1,
        )
        row_violation = jnp.maximum(0.0, -minimum_difference)

        lower = jnp.maximum(upper - 1, 0)
        line_lower = lines[delta_rows, lower]
        line_upper = lines[delta_rows, upper]
        denominator = line_upper - line_lower
        safe_denominator = jnp.where(denominator == 0.0, 1.0, denominator)
        weight = jnp.where(
            denominator == 0.0,
            1.0,
            -line_lower / safe_denominator,
        )
        boundary = z_values[lower] + weight * (
            z_values[upper] - z_values[lower]
        )

        # Inactive rows have fixed benign diagnostics. The host slices them off,
        # but masking here also prevents padded rows from affecting reductions.
        boundary = jnp.where(valid_delta, boundary, 0.0)
        row_violation = jnp.where(valid_delta, row_violation, 0.0)
        has_root = jnp.where(valid_delta, has_root, True)
        upper = jnp.where(valid_delta, upper, 1)
        return boundary, row_violation, has_root, upper

    boundary_batch = jax.vmap(boundary_one)

    @jax.jit
    def solve(
        terminal_values: Any,
        costs: Any,
        kernels1: Any,
        kernels2: Any,
        n1_counts: Any,
        n2_counts: Any,
        x1_lower: Any,
        x2_lower: Any,
        x1_weight: Any,
        x2_weight: Any,
        z_values: Any,
        z_counts: Any,
        delta_counts: Any,
    ) -> tuple[Any, Any, Any, Any]:
        values = edge_extend_batch(terminal_values, n1_counts, n2_counts)

        def backward_step(value: Any, stage_kernels: tuple[Any, Any]):
            stage_kernel1, stage_kernel2 = stage_kernels
            expectation = convolve_batch(value, stage_kernel1, stage_kernel2)
            q_values = expectation - costs[:, None, None]
            diagnostics = boundary_batch(
                q_values,
                x1_lower,
                x2_lower,
                x1_weight,
                x2_weight,
                z_values,
                z_counts,
                delta_counts,
            )
            next_value = edge_extend_batch(
                jnp.maximum(0.0, q_values),
                n1_counts,
                n2_counts,
            )
            return next_value, diagnostics

        _, diagnostics = jax.lax.scan(
            backward_step,
            values,
            (kernels1, kernels2),
            reverse=True,
        )
        return diagnostics

    return solve


def build_radial_gittins_boundary_tables_jax(
    *,
    direction: VectorLike,
    horizon: int,
    specs: Sequence[RadialGittinsJaxTableSpec],
) -> tuple[RadialGittinsBoundaryTable, ...]:
    """Solve a heterogeneous batch of radial-Gittins boundary tables with JAX.

    ``direction`` and ``horizon`` are shared compile-group parameters. Callers
    that exploit objective-swap symmetry should pass the already-canonicalized
    direction. Costs, prior/noise variances, and grids may differ by spec.

    Only the current padded two-dimensional value array is carried through the
    backward ``lax.scan``. Device outputs contain boundary rows and the compact
    root-validity diagnostics needed for host-side validation, never a history
    of two-dimensional value functions.
    """

    direction_array = _direction(direction)
    if int(horizon) != horizon or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    horizon = int(horizon)
    specs = tuple(specs)
    if not specs:
        raise ValueError("specs must contain at least one table specification")

    factors = _scaling_factors(direction_array)
    prepared = tuple(
        _prepare_spec(spec, factors=factors, horizon=horizon) for spec in specs
    )

    batch_size = len(prepared)
    maximum1 = max(item.x1_grid.size for item in prepared)
    maximum2 = max(item.x2_grid.size for item in prepared)
    maximum_z = max(item.grid.z_size for item in prepared)
    maximum_delta = max(item.grid.delta_size for item in prepared)
    maximum_radius1 = max(max(item.radii1) for item in prepared)
    maximum_radius2 = max(max(item.radii2) for item in prepared)
    kernel_size1 = 2 * maximum_radius1 + 1
    kernel_size2 = 2 * maximum_radius2 + 1

    terminal_values = np.stack(
        [
            _edge_pad_state(item.terminal_value, (maximum1, maximum2))
            for item in prepared
        ]
    )
    costs = np.asarray([item.cost for item in prepared], dtype=np.float64)
    n1_counts = np.asarray(
        [item.x1_grid.size for item in prepared], dtype=np.int32
    )
    n2_counts = np.asarray(
        [item.x2_grid.size for item in prepared], dtype=np.int32
    )
    z_counts = np.asarray([item.grid.z_size for item in prepared], dtype=np.int32)
    delta_counts = np.asarray(
        [item.grid.delta_size for item in prepared], dtype=np.int32
    )

    kernels1 = np.empty(
        (horizon, batch_size, kernel_size1), dtype=np.float64
    )
    kernels2 = np.empty(
        (horizon, batch_size, kernel_size2), dtype=np.float64
    )
    for stage in range(horizon):
        for member, item in enumerate(prepared):
            kernels1[stage, member] = _center_pad_kernel(
                item.kernels1[stage],
                item.radii1[stage],
                maximum_radius1,
            )
            kernels2[stage, member] = _center_pad_kernel(
                item.kernels2[stage],
                item.radii2[stage],
                maximum_radius2,
            )

    x1_lower = np.zeros(
        (batch_size, maximum_delta, maximum_z), dtype=np.int32
    )
    x2_lower = np.zeros_like(x1_lower)
    x1_weight = np.zeros(
        (batch_size, maximum_delta, maximum_z), dtype=np.float64
    )
    x2_weight = np.zeros_like(x1_weight)
    z_values = np.zeros((batch_size, maximum_z), dtype=np.float64)
    for member, item in enumerate(prepared):
        delta_size = item.grid.delta_size
        z_size = item.grid.z_size
        interpolation = item.interpolation
        x1_lower[member, :delta_size, :z_size] = interpolation[0]
        x2_lower[member, :delta_size, :z_size] = interpolation[1]
        x1_weight[member, :delta_size, :z_size] = interpolation[2]
        x2_weight[member, :delta_size, :z_size] = interpolation[3]
        z_values[member, :z_size] = item.grid.z_grid
        z_values[member, z_size:] = item.grid.z_grid[-1]

    signature = (
        batch_size,
        horizon,
        maximum1,
        maximum2,
        maximum_z,
        maximum_delta,
        kernel_size1,
        kernel_size2,
    )
    jax, jnp = _jax_modules()
    with _jax_float64_context(jax, jnp):
        solve = _compiled_solver(signature)
        device_outputs = solve(
            jnp.asarray(terminal_values, dtype=jnp.float64),
            jnp.asarray(costs, dtype=jnp.float64),
            jnp.asarray(kernels1, dtype=jnp.float64),
            jnp.asarray(kernels2, dtype=jnp.float64),
            jnp.asarray(n1_counts, dtype=jnp.int32),
            jnp.asarray(n2_counts, dtype=jnp.int32),
            jnp.asarray(x1_lower, dtype=jnp.int32),
            jnp.asarray(x2_lower, dtype=jnp.int32),
            jnp.asarray(x1_weight, dtype=jnp.float64),
            jnp.asarray(x2_weight, dtype=jnp.float64),
            jnp.asarray(z_values, dtype=jnp.float64),
            jnp.asarray(z_counts, dtype=jnp.int32),
            jnp.asarray(delta_counts, dtype=jnp.int32),
        )
        stage_boundaries, row_violations, has_roots, upper_indices = (
            np.asarray(value)
            for value in jax.device_get(device_outputs)
        )

    tables = []
    for member, item in enumerate(prepared):
        delta_size = item.grid.delta_size
        boundaries = np.empty((horizon + 1, delta_size), dtype=np.float64)
        boundaries[:horizon] = stage_boundaries[:, member, :delta_size]
        boundaries[horizon] = item.terminal_boundary

        maximum_violation = 0.0
        # Match the reference solver's validation order: the last transition is
        # processed first by the backward recursion.
        for stage in range(horizon - 1, -1, -1):
            stage_violations = row_violations[stage, member, :delta_size]
            maximum_violation = max(
                maximum_violation,
                float(np.max(stage_violations)),
            )
            _reject_unusable_boundary_rows(
                item.grid.delta_grid,
                item.grid.z_grid,
                row_violations=stage_violations,
                has_root=has_roots[stage, member, :delta_size],
                upper=upper_indices[stage, member, :delta_size],
                margin_cells=item.grid.boundary_margin_cells,
                monotonicity_tolerance=item.grid.monotonicity_tolerance,
            )

        tables.append(
            RadialGittinsBoundaryTable(
                direction=tuple(float(value) for value in direction_array),
                effective_pull_cost=item.cost,
                initial_var=tuple(float(value) for value in item.initial_var),
                obs_noise_var=tuple(float(value) for value in item.obs_noise_var),
                horizon=horizon,
                grid=item.grid,
                boundaries=boundaries,
                max_monotonicity_violation=maximum_violation,
            )
        )
    return tuple(tables)


__all__ = [
    "RadialGittinsJaxTableSpec",
    "build_radial_gittins_boundary_tables_jax",
]
