"""Coordinate lazy, batched prewarming of radial-Gittins boundaries.

This module deliberately sits above the numerical solvers and boundary cache.
It can therefore route sufficiently large cold groups to an optional batched
backend without making JAX a dependency of the ordinary SciPy/cache path.
"""

from __future__ import annotations

import importlib.util
import math
import warnings
from dataclasses import dataclass, replace
from typing import Callable, Hashable, Iterable, Sequence, Tuple

import numpy as np

from .radial_gittins_dp import (
    RadialGittinsBoundaryCache,
    RadialGittinsBoundaryTable,
    RadialGittinsGrid,
    build_radial_gittins_boundary_table,
)


VectorPair = Tuple[float, float]


def _pair(value: object, name: str, *, positive: bool) -> VectorPair:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array, 2)
    if array.shape != (2,):
        raise ValueError(f"{name} must be a scalar or a length-2 vector")
    pair = (float(array[0]), float(array[1]))
    if not all(math.isfinite(component) for component in pair):
        raise ValueError(f"{name} must contain only finite values")
    if positive and any(component <= 0.0 for component in pair):
        raise ValueError(f"{name} must be strictly positive")
    return pair


@dataclass(frozen=True)
class RadialGittinsPrewarmRequest:
    """One caller-facing radial boundary-table request.

    Scalar posterior variances/noise values are normalized to equal pairs so
    equality and hashing have the same semantics as
    :class:`RadialGittinsBoundaryCache`.
    """

    direction: VectorPair
    effective_pull_cost: float
    initial_var: VectorPair
    obs_noise_var: VectorPair
    horizon: int
    grid: RadialGittinsGrid

    def __post_init__(self) -> None:
        direction = _pair(self.direction, "direction", positive=True)
        if not math.isclose(
            direction[0] + direction[1],
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("direction components must sum to one")
        cost = float(self.effective_pull_cost)
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError(
                "effective_pull_cost must be finite and strictly positive"
            )
        if int(self.horizon) != self.horizon or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(self.grid, RadialGittinsGrid):
            raise TypeError("grid must be a RadialGittinsGrid")

        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "effective_pull_cost", cost)
        object.__setattr__(
            self,
            "initial_var",
            _pair(self.initial_var, "initial_var", positive=True),
        )
        object.__setattr__(
            self,
            "obs_noise_var",
            _pair(self.obs_noise_var, "obs_noise_var", positive=True),
        )
        object.__setattr__(self, "horizon", int(self.horizon))

    def cache_kwargs(self) -> dict[str, object]:
        """Return keyword arguments accepted by the existing cache."""
        return {
            "direction": self.direction,
            "effective_pull_cost": self.effective_pull_cost,
            "initial_var": self.initial_var,
            "obs_noise_var": self.obs_noise_var,
            "horizon": self.horizon,
            "grid": self.grid,
        }


def _reflected_grid(grid: RadialGittinsGrid) -> RadialGittinsGrid:
    return replace(
        grid,
        delta_min=-grid.delta_max,
        delta_max=-grid.delta_min,
    )


def _is_exchangeable(request: RadialGittinsPrewarmRequest) -> bool:
    return (
        request.initial_var[0] == request.initial_var[1]
        and request.obs_noise_var[0] == request.obs_noise_var[1]
        and request.direction[0] != request.direction[1]
    )


def canonicalize_radial_gittins_request(
    request: RadialGittinsPrewarmRequest,
) -> RadialGittinsPrewarmRequest:
    """Choose a deterministic physical solve for an exact mirror pair."""
    if not _is_exchangeable(request):
        return request
    if request.direction[0] <= request.direction[1]:
        return request
    return replace(
        request,
        direction=(request.direction[1], request.direction[0]),
        grid=_reflected_grid(request.grid),
    )


@dataclass(frozen=True)
class RadialGittinsStaticGroup:
    """Default compile-compatible grid family for a padded JAX batch."""

    z_size: int
    delta_size: int
    state_size: int
    kernel_stddevs: float
    boundary_margin_cells: int
    monotonicity_tolerance: float


def default_static_group_key(
    request: RadialGittinsPrewarmRequest,
) -> RadialGittinsStaticGroup:
    """Group grids whose cost-dependent bounds can be padded together.

    Bounds are intentionally absent: they change across effective-cost bins.
    The batched solver owns padding of the resulting state axes and kernels.
    """
    grid = request.grid
    return RadialGittinsStaticGroup(
        z_size=grid.z_size,
        delta_size=grid.delta_size,
        state_size=grid.state_size,
        kernel_stddevs=grid.kernel_stddevs,
        boundary_margin_cells=grid.boundary_margin_cells,
        monotonicity_tolerance=grid.monotonicity_tolerance,
    )


BatchBuilder = Callable[
    [Tuple[RadialGittinsPrewarmRequest, ...]],
    Sequence[RadialGittinsBoundaryTable],
]
StaticGroupKey = Callable[[RadialGittinsPrewarmRequest], Hashable]
ScalarBuilder = Callable[..., RadialGittinsBoundaryTable]


@dataclass(frozen=True)
class RadialGittinsPrewarmStats:
    requested: int = 0
    unique_requests: int = 0
    canonical_requests: int = 0
    memory_hits: int = 0
    disk_hits: int = 0
    cold_misses: int = 0
    jax_groups: int = 0
    jax_tables: int = 0
    scipy_tables: int = 0
    jax_fallback_groups: int = 0


@dataclass(frozen=True)
class RadialGittinsPrewarmResult:
    """Tables in input order and coordinator-local routing statistics."""

    tables: Tuple[RadialGittinsBoundaryTable, ...]
    stats: RadialGittinsPrewarmStats


def _cache_key(
    cache: RadialGittinsBoundaryCache,
    request: RadialGittinsPrewarmRequest,
) -> Tuple[object, ...]:
    return cache._radial_key(  # noqa: SLF001 - integration adapter by design
        request.direction,
        request.effective_pull_cost,
        request.initial_var,
        request.obs_noise_var,
        request.horizon,
        request.grid,
    )


def _mirror_request(
    request: RadialGittinsPrewarmRequest,
) -> RadialGittinsPrewarmRequest:
    return replace(
        request,
        direction=(request.direction[1], request.direction[0]),
        grid=_reflected_grid(request.grid),
    )


def _mirrored_table(
    cache: RadialGittinsBoundaryCache,
    source: RadialGittinsBoundaryTable,
    request: RadialGittinsPrewarmRequest,
) -> RadialGittinsBoundaryTable:
    return cache._mirrored_table(  # noqa: SLF001 - integration adapter
        source,
        direction=request.direction,
        initial_var=request.initial_var,
        obs_noise_var=request.obs_noise_var,
        grid=request.grid,
    )


def _probe_cache(
    cache: RadialGittinsBoundaryCache,
    request: RadialGittinsPrewarmRequest,
) -> tuple[RadialGittinsBoundaryTable | None, str | None]:
    """Perform the read-only portion of ``cache.get`` without a cold solve."""
    key = _cache_key(cache, request)
    table = cache._tables.get(key)  # noqa: SLF001
    if table is None:
        table = cache._mirrored_tables.get(key)  # noqa: SLF001
    if table is not None:
        cache._increment("memory_hits")  # noqa: SLF001
        return table, "memory"

    reflected_request: RadialGittinsPrewarmRequest | None = None
    reflected_key: Tuple[object, ...] | None = None
    if _is_exchangeable(request):
        reflected_request = _mirror_request(request)
        reflected_key = _cache_key(cache, reflected_request)
        source = cache._tables.get(reflected_key)  # noqa: SLF001
        if source is not None:
            table = _mirrored_table(cache, source, request)
            cache._mirrored_tables[key] = table  # noqa: SLF001
            cache._increment("memory_hits")  # noqa: SLF001
            return table, "memory"

    table = cache._load_radial(key)  # noqa: SLF001
    if table is not None:
        cache._tables[key] = table  # noqa: SLF001
        cache._increment("disk_hits")  # noqa: SLF001
        return table, "disk"

    if reflected_key is not None and reflected_request is not None:
        source = cache._load_radial(reflected_key)  # noqa: SLF001
        if source is not None:
            cache._tables[reflected_key] = source  # noqa: SLF001
            table = _mirrored_table(cache, source, request)
            cache._mirrored_tables[key] = table  # noqa: SLF001
            cache._increment("disk_hits")  # noqa: SLF001
            return table, "disk"

    if cache.disk_read:
        cache._increment("disk_misses")  # noqa: SLF001
    return None, None


def _validate_table(
    request: RadialGittinsPrewarmRequest,
    table: RadialGittinsBoundaryTable,
) -> None:
    if not isinstance(table, RadialGittinsBoundaryTable):
        raise TypeError("a boundary builder returned a non-table result")
    expected = (
        request.direction,
        request.effective_pull_cost,
        request.initial_var,
        request.obs_noise_var,
        request.horizon,
        request.grid,
    )
    observed = (
        tuple(table.direction),
        table.effective_pull_cost,
        tuple(table.initial_var),
        tuple(table.obs_noise_var),
        table.horizon,
        table.grid,
    )
    if observed != expected:
        raise ValueError(
            "a boundary builder returned a table for a different request"
        )


def _publish_table(
    cache: RadialGittinsBoundaryCache,
    request: RadialGittinsPrewarmRequest,
    table: RadialGittinsBoundaryTable,
    *,
    builder_backend: str,
) -> None:
    _validate_table(request, table)
    key = _cache_key(cache, request)
    cache._tables[key] = table  # noqa: SLF001
    cache._increment("builds")  # noqa: SLF001
    cache._write_entry(  # noqa: SLF001
        kind="radial",
        disk_key=cache._radial_disk_key(key),  # noqa: SLF001
        arrays={"boundaries": table.boundaries},
        extra_manifest={
            "builder_backend": builder_backend,
            "max_monotonicity_violation": float(
                table.max_monotonicity_violation
            ).hex()
        },
    )


def _default_jax_available() -> bool:
    return (
        importlib.util.find_spec("jax") is not None
        and importlib.util.find_spec(
            "agentopt.model_selection.radial_gittins_dp_jax"
        )
        is not None
    )


def _default_jax_batch_builder(
    requests: Tuple[RadialGittinsPrewarmRequest, ...],
) -> Sequence[RadialGittinsBoundaryTable]:
    from .radial_gittins_dp_jax import (  # type: ignore[import-not-found]
        RadialGittinsJaxTableSpec,
        build_radial_gittins_boundary_tables_jax,
    )

    first = requests[0]
    specs = tuple(
        RadialGittinsJaxTableSpec(
            effective_pull_cost=request.effective_pull_cost,
            initial_var=request.initial_var,
            obs_noise_var=request.obs_noise_var,
            grid=request.grid,
        )
        for request in requests
    )
    return build_radial_gittins_boundary_tables_jax(
        direction=first.direction,
        horizon=first.horizon,
        specs=specs,
    )


def _normalize_requests(
    requests: Iterable[RadialGittinsPrewarmRequest],
) -> Tuple[RadialGittinsPrewarmRequest, ...]:
    normalized = tuple(requests)
    for request in normalized:
        if not isinstance(request, RadialGittinsPrewarmRequest):
            raise TypeError(
                "requests must contain RadialGittinsPrewarmRequest values"
            )
    return normalized


def prewarm_radial_gittins_boundaries(
    requests: Iterable[RadialGittinsPrewarmRequest],
    *,
    cache: RadialGittinsBoundaryCache,
    jax_min_batch_size: int = 4,
    jax_batch_builder: BatchBuilder | None = None,
    jax_available: Callable[[], bool] | bool | None = None,
    static_group_key: StaticGroupKey = default_static_group_key,
    fallback_on_jax_error: bool = True,
    scipy_builder: ScalarBuilder = build_radial_gittins_boundary_table,
) -> RadialGittinsPrewarmResult:
    """Populate ``cache`` using batched JAX only for sufficiently large misses.

    Requests are deduplicated before cache I/O and exact mirror pairs share one
    physical solve. A failing optional JAX group is rebuilt completely through
    the scalar SciPy builder unless ``fallback_on_jax_error`` is false. Backend
    selection applies to cold misses: cached tables are backend-neutral because
    both builders implement the same discretized DP within numerical tolerance.

    The existing cache currently has no public probe/insert API, so this
    adapter intentionally confines its use of cache-private methods here. It
    preserves the cache's counters, validated disk format, and mirror aliases.
    Like the cache itself, one coordinator call is intended to be single-threaded.
    """
    if type(cache) is not RadialGittinsBoundaryCache:
        raise TypeError("cache must be a concrete RadialGittinsBoundaryCache")
    if int(jax_min_batch_size) != jax_min_batch_size:
        raise ValueError("jax_min_batch_size must be an integer")
    jax_min_batch_size = int(jax_min_batch_size)
    if jax_min_batch_size <= 0:
        raise ValueError("jax_min_batch_size must be positive")

    original = _normalize_requests(requests)
    unique_original = tuple(dict.fromkeys(original))
    canonical_by_original = {
        request: canonicalize_radial_gittins_request(request)
        for request in unique_original
    }
    canonical = tuple(dict.fromkeys(canonical_by_original.values()))

    resolved: dict[
        RadialGittinsPrewarmRequest, RadialGittinsBoundaryTable
    ] = {}
    memory_hits = 0
    disk_hits = 0
    cold: list[RadialGittinsPrewarmRequest] = []
    for request in canonical:
        table, source = _probe_cache(cache, request)
        if table is None:
            cold.append(request)
        else:
            resolved[request] = table
            memory_hits += int(source == "memory")
            disk_hits += int(source == "disk")

    groups: dict[
        tuple[object, ...], list[RadialGittinsPrewarmRequest]
    ] = {}
    for request in cold:
        extra_key = static_group_key(request)
        try:
            hash(extra_key)
        except TypeError as error:
            raise TypeError("static_group_key must return a hashable value") from error
        group_key = (
            request.direction,
            request.horizon,
            request.initial_var,
            request.obs_noise_var,
            extra_key,
        )
        groups.setdefault(group_key, []).append(request)

    if jax_batch_builder is None:
        jax_batch_builder = _default_jax_batch_builder
    if jax_available is None:
        use_jax = _default_jax_available()
    elif callable(jax_available):
        use_jax = bool(jax_available())
    else:
        use_jax = bool(jax_available)

    jax_groups = 0
    jax_tables = 0
    scipy_tables = 0
    jax_fallback_groups = 0
    for group in groups.values():
        group_requests = tuple(group)
        tables: Sequence[RadialGittinsBoundaryTable] | None = None
        builder_backend = "scipy"
        if use_jax and len(group_requests) >= jax_min_batch_size:
            try:
                candidate = tuple(jax_batch_builder(group_requests))
                if len(candidate) != len(group_requests):
                    raise ValueError(
                        "JAX batch builder returned the wrong table count"
                    )
                for request, table in zip(group_requests, candidate):
                    _validate_table(request, table)
                tables = candidate
                builder_backend = "jax"
                jax_groups += 1
                jax_tables += len(candidate)
            except Exception as error:
                if not fallback_on_jax_error:
                    raise
                jax_fallback_groups += 1
                # Import/device/compiler failures are normally persistent for
                # this process. Do not retry every remaining group in this call.
                use_jax = False
                warnings.warn(
                    "JAX radial-Gittins prewarm failed; rebuilding this "
                    f"group with SciPy: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if tables is None:
            candidate = tuple(
                scipy_builder(**request.cache_kwargs())
                for request in group_requests
            )
            # Validate the whole group before mutating memory or disk cache.
            # This gives bad custom builders all-or-nothing publication.
            for request, table in zip(group_requests, candidate):
                _validate_table(request, table)
            tables = candidate
            scipy_tables += len(tables)

        for request, table in zip(group_requests, tables):
            _publish_table(
                cache,
                request,
                table,
                builder_backend=builder_backend,
            )
            resolved[request] = table

    caller_tables: dict[
        RadialGittinsPrewarmRequest, RadialGittinsBoundaryTable
    ] = {}
    for request in unique_original:
        canonical_request = canonical_by_original[request]
        source = resolved[canonical_request]
        if request == canonical_request:
            caller_tables[request] = source
            continue
        table = _mirrored_table(cache, source, request)
        cache._mirrored_tables[_cache_key(cache, request)] = table  # noqa: SLF001
        caller_tables[request] = table

    return RadialGittinsPrewarmResult(
        tables=tuple(caller_tables[request] for request in original),
        stats=RadialGittinsPrewarmStats(
            requested=len(original),
            unique_requests=len(unique_original),
            canonical_requests=len(canonical),
            memory_hits=memory_hits,
            disk_hits=disk_hits,
            cold_misses=len(cold),
            jax_groups=jax_groups,
            jax_tables=jax_tables,
            scipy_tables=scipy_tables,
            jax_fallback_groups=jax_fallback_groups,
        ),
    )


__all__ = [
    "BatchBuilder",
    "RadialGittinsPrewarmRequest",
    "RadialGittinsPrewarmResult",
    "RadialGittinsPrewarmStats",
    "RadialGittinsStaticGroup",
    "canonicalize_radial_gittins_request",
    "default_static_group_key",
    "prewarm_radial_gittins_boundaries",
]
