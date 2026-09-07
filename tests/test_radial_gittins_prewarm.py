import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agentopt.model_selection.radial_gittins_dp import (
    RadialGittinsBoundaryCache,
    RadialGittinsBoundaryTable,
    RadialGittinsGrid,
)
from agentopt.model_selection.radial_gittins_prewarm import (
    RadialGittinsPrewarmRequest,
    canonicalize_radial_gittins_request,
    prewarm_radial_gittins_boundaries,
)


TEST_GRID = RadialGittinsGrid(
    z_min=-3.0,
    z_max=4.0,
    z_size=17,
    delta_min=-3.0,
    delta_max=2.0,
    delta_size=17,
    state_size=17,
    state_halo=2.0,
    kernel_stddevs=4.0,
    boundary_margin_cells=1,
    monotonicity_tolerance=1e-6,
)


def _request(
    *,
    direction=(0.3, 0.7),
    cost=0.05,
    initial_var=0.04,
    obs_noise_var=0.0625,
    horizon=2,
    grid=TEST_GRID,
):
    return RadialGittinsPrewarmRequest(
        direction=direction,
        effective_pull_cost=cost,
        initial_var=initial_var,
        obs_noise_var=obs_noise_var,
        horizon=horizon,
        grid=grid,
    )


def _fake_table(request):
    values = np.arange(
        (request.horizon + 1) * request.grid.delta_size,
        dtype=np.float64,
    ).reshape(request.horizon + 1, request.grid.delta_size)
    values = values + request.effective_pull_cost
    return RadialGittinsBoundaryTable(
        direction=request.direction,
        effective_pull_cost=request.effective_pull_cost,
        initial_var=request.initial_var,
        obs_noise_var=request.obs_noise_var,
        horizon=request.horizon,
        grid=request.grid,
        boundaries=values,
        max_monotonicity_violation=0.0,
    )


def _fake_scalar_builder(**kwargs):
    return _fake_table(RadialGittinsPrewarmRequest(**kwargs))


class RadialGittinsPrewarmTests(unittest.TestCase):
    def test_request_validation_and_exact_mirror_canonicalization(self):
        reflected_grid = replace(
            TEST_GRID,
            delta_min=-TEST_GRID.delta_max,
            delta_max=-TEST_GRID.delta_min,
        )
        request = _request(direction=(0.7, 0.3), grid=reflected_grid)

        canonical = canonicalize_radial_gittins_request(request)

        self.assertEqual(canonical.direction, (0.3, 0.7))
        self.assertEqual(canonical.grid, TEST_GRID)
        self.assertEqual(canonical.initial_var, (0.04, 0.04))
        self.assertEqual(canonical.obs_noise_var, (0.0625, 0.0625))

        asymmetric = _request(
            direction=(0.7, 0.3),
            initial_var=(0.04, 0.05),
            grid=reflected_grid,
        )
        self.assertIs(
            canonicalize_radial_gittins_request(asymmetric),
            asymmetric,
        )

    def test_deduplicates_mirrors_batches_cold_costs_and_persists(self):
        reflected_grid = replace(
            TEST_GRID,
            delta_min=-TEST_GRID.delta_max,
            delta_max=-TEST_GRID.delta_min,
        )
        canonical = _request(cost=0.05)
        mirror = _request(
            direction=(0.7, 0.3),
            cost=0.05,
            grid=reflected_grid,
        )
        other_cost = _request(cost=0.1)
        calls = []

        def batch_builder(requests):
            calls.append(requests)
            return tuple(_fake_table(request) for request in requests)

        with tempfile.TemporaryDirectory() as directory:
            cache = RadialGittinsBoundaryCache(cache_dir=directory)
            result = prewarm_radial_gittins_boundaries(
                (mirror, canonical, mirror, other_cost),
                cache=cache,
                jax_min_batch_size=2,
                jax_batch_builder=batch_builder,
                jax_available=True,
                scipy_builder=lambda **kwargs: self.fail(
                    "the eligible cold group must use the batch builder"
                ),
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(
                [request.effective_pull_cost for request in calls[0]],
                [0.05, 0.1],
            )
            self.assertTrue(
                all(request.direction == (0.3, 0.7) for request in calls[0])
            )
            self.assertEqual(result.stats.requested, 4)
            self.assertEqual(result.stats.unique_requests, 3)
            self.assertEqual(result.stats.canonical_requests, 2)
            self.assertEqual(result.stats.cold_misses, 2)
            self.assertEqual(result.stats.jax_groups, 1)
            self.assertEqual(result.stats.jax_tables, 2)
            self.assertEqual(result.stats.scipy_tables, 0)
            self.assertEqual(cache.stats.builds, 2)
            self.assertEqual(cache.stats.disk_misses, 2)
            self.assertEqual(len(cache), 2)
            self.assertEqual(len(list(Path(directory).rglob("*.npz"))), 2)
            self.assertIs(result.tables[0], result.tables[2])
            np.testing.assert_array_equal(
                result.tables[0].boundaries,
                result.tables[1].boundaries[:, ::-1],
            )
            self.assertEqual(result.tables[0].direction, (0.7, 0.3))

            fresh = RadialGittinsBoundaryCache(cache_dir=directory)
            with patch(
                "agentopt.model_selection.radial_gittins_dp."
                "build_radial_gittins_boundary_table",
                side_effect=AssertionError("prewarm must populate disk cache"),
            ):
                loaded = fresh.get(**mirror.cache_kwargs())
            np.testing.assert_array_equal(
                loaded.boundaries,
                result.tables[0].boundaries,
            )
            self.assertEqual(fresh.stats.disk_hits, 1)
            self.assertEqual(fresh.stats.builds, 0)

            # Backend selection governs only cold construction. A fresh
            # SciPy-routed coordinator accepts the valid JAX-built disk table.
            fresh_scipy = RadialGittinsBoundaryCache(cache_dir=directory)
            reused = prewarm_radial_gittins_boundaries(
                (canonical,),
                cache=fresh_scipy,
                jax_available=False,
                scipy_builder=lambda **kwargs: self.fail(
                    "a backend-neutral disk hit must not rebuild"
                ),
            )
            self.assertEqual(reused.stats.disk_hits, 1)
            self.assertEqual(reused.stats.scipy_tables, 0)

    def test_groups_by_horizon_posterior_and_static_grid_family(self):
        different_grid = replace(TEST_GRID, state_size=19)
        requests = (
            _request(cost=0.04),
            _request(cost=0.08),
            _request(cost=0.12, horizon=3),
            _request(cost=0.16, grid=different_grid),
            _request(cost=0.2, initial_var=0.05),
        )
        batch_calls = []
        scalar_calls = []

        def batch_builder(group):
            batch_calls.append(group)
            return tuple(_fake_table(request) for request in group)

        def scalar_builder(**kwargs):
            request = RadialGittinsPrewarmRequest(**kwargs)
            scalar_calls.append(request)
            return _fake_table(request)

        result = prewarm_radial_gittins_boundaries(
            requests,
            cache=RadialGittinsBoundaryCache(),
            jax_min_batch_size=2,
            jax_batch_builder=batch_builder,
            jax_available=True,
            scipy_builder=scalar_builder,
        )

        self.assertEqual(len(batch_calls), 1)
        self.assertEqual(len(batch_calls[0]), 2)
        self.assertEqual(len(scalar_calls), 3)
        self.assertEqual(result.stats.jax_tables, 2)
        self.assertEqual(result.stats.scipy_tables, 3)

    def test_unavailable_jax_uses_scalar_builder(self):
        calls = []

        def scalar_builder(**kwargs):
            request = RadialGittinsPrewarmRequest(**kwargs)
            calls.append(request)
            return _fake_table(request)

        result = prewarm_radial_gittins_boundaries(
            (_request(cost=0.04), _request(cost=0.08)),
            cache=RadialGittinsBoundaryCache(),
            jax_min_batch_size=2,
            jax_batch_builder=lambda requests: self.fail(
                "unavailable JAX must not be called"
            ),
            jax_available=False,
            scipy_builder=scalar_builder,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(result.stats.jax_groups, 0)
        self.assertEqual(result.stats.scipy_tables, 2)

    def test_jax_failure_warns_and_falls_back_as_one_group(self):
        requests = (_request(cost=0.04), _request(cost=0.08))
        scalar_calls = []

        def scalar_builder(**kwargs):
            request = RadialGittinsPrewarmRequest(**kwargs)
            scalar_calls.append(request)
            return _fake_table(request)

        with self.assertWarnsRegex(RuntimeWarning, "rebuilding.*SciPy"):
            result = prewarm_radial_gittins_boundaries(
                requests,
                cache=RadialGittinsBoundaryCache(),
                jax_min_batch_size=2,
                jax_batch_builder=lambda group: (_ for _ in ()).throw(
                    RuntimeError("simulated accelerator failure")
                ),
                jax_available=True,
                scipy_builder=scalar_builder,
            )

        self.assertEqual(len(scalar_calls), 2)
        self.assertEqual(result.stats.jax_fallback_groups, 1)
        self.assertEqual(result.stats.jax_tables, 0)
        self.assertEqual(result.stats.scipy_tables, 2)

    def test_jax_failure_disables_retries_for_later_groups(self):
        requests = (
            _request(cost=0.04, horizon=2),
            _request(cost=0.08, horizon=2),
            _request(cost=0.12, horizon=3),
            _request(cost=0.16, horizon=3),
        )
        batch_calls = []

        def failing_batch(group):
            batch_calls.append(group)
            raise RuntimeError("persistent compiler failure")

        with self.assertWarnsRegex(RuntimeWarning, "rebuilding.*SciPy"):
            result = prewarm_radial_gittins_boundaries(
                requests,
                cache=RadialGittinsBoundaryCache(),
                jax_min_batch_size=2,
                jax_batch_builder=failing_batch,
                jax_available=True,
                scipy_builder=_fake_scalar_builder,
            )

        self.assertEqual(len(batch_calls), 1)
        self.assertEqual(result.stats.jax_fallback_groups, 1)
        self.assertEqual(result.stats.scipy_tables, 4)

    def test_invalid_scalar_group_is_not_partially_published(self):
        requests = (_request(cost=0.04), _request(cost=0.08))
        cache = RadialGittinsBoundaryCache()
        calls = 0

        def invalid_second(**kwargs):
            nonlocal calls
            calls += 1
            request = RadialGittinsPrewarmRequest(**kwargs)
            if calls == 2:
                request = replace(request, effective_pull_cost=0.09)
            return _fake_table(request)

        with self.assertRaisesRegex(ValueError, "different request"):
            prewarm_radial_gittins_boundaries(
                requests,
                cache=cache,
                jax_available=False,
                scipy_builder=invalid_second,
            )

        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.stats.builds, 0)

    def test_existing_memory_and_disk_entries_are_not_rebuilt(self):
        request = _request()
        with tempfile.TemporaryDirectory() as directory:
            cache = RadialGittinsBoundaryCache(cache_dir=directory)
            first = prewarm_radial_gittins_boundaries(
                (request,),
                cache=cache,
                jax_available=False,
                scipy_builder=_fake_scalar_builder,
            )
            memory = prewarm_radial_gittins_boundaries(
                (request, request),
                cache=cache,
                jax_available=False,
                scipy_builder=lambda **kwargs: self.fail(
                    "memory hit must not rebuild"
                ),
            )
            self.assertEqual(memory.stats.memory_hits, 1)
            self.assertEqual(memory.stats.cold_misses, 0)
            self.assertIs(memory.tables[0], memory.tables[1])

            fresh = RadialGittinsBoundaryCache(cache_dir=directory)
            disk = prewarm_radial_gittins_boundaries(
                (request,),
                cache=fresh,
                jax_available=False,
                scipy_builder=lambda **kwargs: self.fail(
                    "disk hit must not rebuild"
                ),
            )
            self.assertEqual(disk.stats.disk_hits, 1)
            self.assertEqual(disk.stats.cold_misses, 0)
            np.testing.assert_array_equal(
                disk.tables[0].boundaries,
                first.tables[0].boundaries,
            )


if __name__ == "__main__":
    unittest.main()
