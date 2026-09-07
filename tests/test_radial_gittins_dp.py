import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agentopt.model_selection.radial_gittins import DEFAULT_DIRECTIONS
from agentopt.model_selection.radial_gittins_dp import (
    BoundaryGridError,
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
    build_radial_gittins_boundary_table,
    direction_aware_grid,
    gaussian_expectation_separable,
    posterior_variance_schedule,
    radial_transition_covariance,
    terminal_expected_radial_utility,
)


TEST_GRID = RadialGittinsGrid(
    z_min=-4.0,
    z_max=4.0,
    z_size=81,
    delta_min=-3.0,
    delta_max=3.0,
    delta_size=61,
    state_size=81,
    state_halo=5.0,
    boundary_margin_cells=2,
    monotonicity_tolerance=1e-6,
)

MIRROR_TEST_GRID = RadialGittinsGrid(
    z_min=-4.0,
    z_max=4.0,
    z_size=41,
    delta_min=-3.0,
    delta_max=3.0,
    delta_size=41,
    state_size=41,
    state_halo=5.0,
    boundary_margin_cells=2,
    monotonicity_tolerance=1e-5,
)


class RadialTransitionTests(unittest.TestCase):
    def test_variance_schedule_and_balanced_transition_covariance(self):
        schedule = posterior_variance_schedule(0.04, 0.0625, horizon=1)
        expected_next = 1.0 / (1.0 / 0.04 + 1.0 / 0.0625)
        np.testing.assert_allclose(schedule[0], [0.04, 0.04])
        np.testing.assert_allclose(schedule[1], [expected_next, expected_next])

        q = 0.04 - expected_next
        covariance = radial_transition_covariance(
            schedule[0], schedule[1], (0.5, 0.5)
        )
        np.testing.assert_allclose(
            covariance,
            np.array([[q / 2.0, 0.0], [0.0, 2.0 * q]]),
        )

    def test_unbalanced_transition_covariance_matches_formula(self):
        current = np.array([0.04, 0.04])
        following = np.array([0.024390243902439025] * 2)
        q = current[0] - following[0]

        covariance = radial_transition_covariance(
            current,
            following,
            (0.25, 0.75),
        )

        np.testing.assert_allclose(
            covariance,
            np.array([[2.5 * q, 4.0 * q], [4.0 * q, 10.0 * q]]),
        )
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) >= 0.0))


class DeterministicConvolutionTests(unittest.TestCase):
    def test_sub_grid_standard_deviation_has_continuum_relu_expectation(self):
        grid = np.linspace(-1.0, 1.0, 41)
        x1, x2 = np.meshgrid(grid, grid, indexing="ij")
        value = np.maximum(0.0, x1) + 0.0 * x2
        step = float(grid[1] - grid[0])
        standard_deviation = step / 20.0

        convolved = gaussian_expectation_separable(
            value,
            (standard_deviation**2, standard_deviation**2),
            grid,
            grid,
        )

        expected = standard_deviation / math.sqrt(2.0 * math.pi)
        center = len(grid) // 2
        self.assertAlmostEqual(
            convolved[center, center],
            expected,
            places=12,
        )


class DirectionAwareGridTests(unittest.TestCase):
    def test_unit_objective_ranges_follow_direction_scaling(self):
        balanced = direction_aware_grid((0.5, 0.5))
        extreme = direction_aware_grid((0.1, 0.9))

        self.assertEqual(
            (balanced.z_min, balanced.z_max),
            (-1.0, 2.0),
        )
        self.assertEqual(
            (balanced.delta_min, balanced.delta_max),
            (-1.2, 1.2),
        )
        self.assertEqual((extreme.z_min, extreme.z_max), (-1.0, 6.0))
        self.assertEqual(
            (extreme.delta_min, extreme.delta_max),
            (-1.2, 9.2),
        )

        with self.assertRaises(BoundaryGridError):
            direction_aware_grid(
                (0.1, 0.9),
                base_grid=RadialGittinsGrid(
                    delta_min=-2.0,
                    delta_max=8.0,
                ),
            )

    def test_b4_h49_boundaries_converge_under_state_refinement(self):
        initial_var = 1.0 / 41.0
        reference_values = {
            (0.5, 0.5): np.array([0.23363]),
            (0.1, 0.9): np.array([0.20065, 1.11017, 2.59282]),
        }
        query_deltas = {
            (0.5, 0.5): (0.0,),
            (0.1, 0.9): (0.0, 2.0, 5.0),
        }

        for direction in reference_values:
            results = []
            for state_size in (257, 513):
                base = RadialGittinsGrid(
                    z_size=129,
                    delta_size=129,
                    state_size=state_size,
                )
                grid = direction_aware_grid(direction, base_grid=base)
                table = build_radial_gittins_boundary_table(
                    direction=direction,
                    effective_pull_cost=0.005,
                    initial_var=initial_var,
                    obs_noise_var=0.0625,
                    horizon=49,
                    grid=grid,
                )
                results.append(
                    np.array(
                        [
                            table.boundary(0, delta)
                            for delta in query_deltas[direction]
                        ]
                    )
                )

            coarse_error = float(
                np.max(np.abs(results[0] - reference_values[direction]))
            )
            fine_error = float(
                np.max(np.abs(results[1] - reference_values[direction]))
            )
            self.assertLess(fine_error, coarse_error)
            self.assertLess(fine_error, 0.009)


class BoundaryTableTests(unittest.TestCase):
    def test_terminal_boundary_is_the_analytic_expected_min_root(self):
        table = build_radial_gittins_boundary_table(
            direction=(0.5, 0.5),
            effective_pull_cost=0.05,
            initial_var=(0.04, 0.04),
            obs_noise_var=(0.0625, 0.0625),
            horizon=2,
            grid=TEST_GRID,
        )
        terminal_var = posterior_variance_schedule(0.04, 0.0625, 2)[-1, 0]

        self.assertEqual(table.boundaries.shape, (3, TEST_GRID.delta_size))
        self.assertAlmostEqual(
            table.boundary(2, 0.0),
            math.sqrt(terminal_var / math.pi),
            places=10,
        )

    def test_balanced_boundary_is_symmetric_and_cost_raises_it(self):
        low_cost = build_radial_gittins_boundary_table(
            direction=(0.5, 0.5),
            effective_pull_cost=0.02,
            initial_var=0.04,
            obs_noise_var=0.0625,
            horizon=2,
            grid=TEST_GRID,
        )
        high_cost = build_radial_gittins_boundary_table(
            direction=(0.5, 0.5),
            effective_pull_cost=0.08,
            initial_var=0.04,
            obs_noise_var=0.0625,
            horizon=2,
            grid=TEST_GRID,
        )

        np.testing.assert_allclose(
            low_cost.boundaries,
            low_cost.boundaries[:, ::-1],
            atol=2e-10,
        )
        self.assertGreater(high_cost.boundary(0, 0.0), low_cost.boundary(0, 0.0))
        self.assertLessEqual(low_cost.max_monotonicity_violation, 1e-6)

    def test_cache_key_includes_cost_horizon_and_grid(self):
        cache = RadialGittinsBoundaryCache()
        kwargs = dict(
            direction=(0.5, 0.5),
            effective_pull_cost=0.05,
            initial_var=(0.04, 0.04),
            obs_noise_var=(0.0625, 0.0625),
            horizon=1,
            grid=TEST_GRID,
        )

        first = cache.get(**kwargs)
        second = cache.get(**kwargs)
        self.assertIs(first, second)
        self.assertEqual(len(cache), 1)

        cache.get(**{**kwargs, "effective_pull_cost": 0.06})
        cache.get(**{**kwargs, "horizon": 2})
        self.assertEqual(len(cache), 3)

    def test_radial_disk_cache_round_trip_avoids_rebuild(self):
        kwargs = dict(
            direction=(0.5, 0.5),
            effective_pull_cost=0.05,
            initial_var=(0.04, 0.04),
            obs_noise_var=(0.0625, 0.0625),
            horizon=1,
            grid=MIRROR_TEST_GRID,
        )
        with tempfile.TemporaryDirectory() as directory:
            first_cache = RadialGittinsBoundaryCache(cache_dir=directory)
            first = first_cache.get(**kwargs)
            self.assertEqual(first_cache.stats.builds, 1)
            self.assertEqual(first_cache.stats.disk_misses, 1)
            self.assertEqual(len(list(Path(directory).rglob("*.npz"))), 1)

            second_cache = RadialGittinsBoundaryCache(cache_dir=directory)
            with patch(
                "agentopt.model_selection.radial_gittins_dp."
                "build_radial_gittins_boundary_table",
                side_effect=AssertionError("disk hit must not rebuild"),
            ):
                second = second_cache.get(**kwargs)

            np.testing.assert_array_equal(second.boundaries, first.boundaries)
            self.assertEqual(second.direction, first.direction)
            self.assertEqual(second.grid, first.grid)
            self.assertFalse(second.boundaries.flags.writeable)
            self.assertEqual(second_cache.stats.disk_hits, 1)
            self.assertEqual(second_cache.stats.builds, 0)

    def test_mirrored_direction_can_reuse_disk_source(self):
        direction = (0.3, 0.7)
        swapped = (0.7, 0.3)
        grid = direction_aware_grid(direction, base_grid=TEST_GRID)
        swapped_grid = direction_aware_grid(swapped, base_grid=TEST_GRID)
        shared = dict(
            effective_pull_cost=0.05,
            initial_var=0.04,
            obs_noise_var=0.0625,
            horizon=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = RadialGittinsBoundaryCache(cache_dir=directory).get(
                direction=direction,
                grid=grid,
                **shared,
            )
            cache = RadialGittinsBoundaryCache(cache_dir=directory)
            with patch(
                "agentopt.model_selection.radial_gittins_dp."
                "build_radial_gittins_boundary_table",
                side_effect=AssertionError("mirror disk hit must not rebuild"),
            ):
                mirrored = cache.get(
                    direction=swapped,
                    grid=swapped_grid,
                    **shared,
                )

            np.testing.assert_array_equal(
                mirrored.boundaries,
                source.boundaries[:, ::-1],
            )
            self.assertEqual(cache.stats.disk_hits, 1)
            self.assertEqual(cache.stats.builds, 0)
            self.assertEqual(len(cache), 1)

    def test_corrupt_radial_disk_entry_is_rebuilt_and_replaced(self):
        kwargs = dict(
            direction=(0.5, 0.5),
            effective_pull_cost=0.05,
            initial_var=0.04,
            obs_noise_var=0.0625,
            horizon=1,
            grid=MIRROR_TEST_GRID,
        )
        with tempfile.TemporaryDirectory() as directory:
            RadialGittinsBoundaryCache(cache_dir=directory).get(**kwargs)
            path = next(Path(directory).rglob("*.npz"))
            path.write_bytes(b"truncated")

            cache = RadialGittinsBoundaryCache(cache_dir=directory)
            with self.assertWarns(RuntimeWarning):
                rebuilt = cache.get(**kwargs)
            self.assertEqual(cache.stats.corruptions, 1)
            self.assertEqual(cache.stats.builds, 1)

            fresh = RadialGittinsBoundaryCache(cache_dir=directory)
            loaded = fresh.get(**kwargs)
            np.testing.assert_array_equal(loaded.boundaries, rebuilt.boundaries)
            self.assertEqual(fresh.stats.disk_hits, 1)

    def test_disk_write_failure_is_best_effort_and_cleans_temp_file(self):
        kwargs = dict(
            direction=(0.5, 0.5),
            effective_pull_cost=0.05,
            initial_var=0.04,
            obs_noise_var=0.0625,
            horizon=1,
            grid=MIRROR_TEST_GRID,
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = RadialGittinsBoundaryCache(cache_dir=directory)
            with patch(
                "agentopt.model_selection.radial_gittins_dp.os.replace",
                side_effect=OSError("simulated publish failure"),
            ), self.assertWarns(RuntimeWarning):
                table = cache.get(**kwargs)

            self.assertEqual(table.boundaries.shape, (2, 41))
            self.assertEqual(cache.stats.builds, 1)
            self.assertEqual(cache.stats.write_failures, 1)
            self.assertEqual(list(Path(directory).rglob("*.tmp")), [])
            self.assertEqual(list(Path(directory).rglob("*.npz")), [])

    def test_cache_builds_only_five_tables_for_nine_symmetric_directions(self):
        cache = RadialGittinsBoundaryCache()

        with patch(
            "agentopt.model_selection.radial_gittins_dp."
            "build_radial_gittins_boundary_table",
            wraps=build_radial_gittins_boundary_table,
        ) as builder:
            tables = [
                cache.get(
                    direction=direction,
                    effective_pull_cost=0.05,
                    initial_var=0.04,
                    obs_noise_var=0.0625,
                    horizon=1,
                    grid=MIRROR_TEST_GRID,
                )
                for direction in DEFAULT_DIRECTIONS
            ]

        self.assertEqual(builder.call_count, 5)
        self.assertEqual(len(cache), 5)
        self.assertEqual(
            [table.direction for table in tables],
            list(DEFAULT_DIRECTIONS),
        )

    def test_mirrored_cache_view_matches_a_direct_swapped_build(self):
        direction = (0.3, 0.7)
        swapped_direction = (0.7, 0.3)
        grid = direction_aware_grid(direction, base_grid=TEST_GRID)
        swapped_grid = direction_aware_grid(
            swapped_direction,
            base_grid=TEST_GRID,
        )
        kwargs = dict(
            effective_pull_cost=0.05,
            initial_var=(0.04, 0.04),
            obs_noise_var=(0.0625, 0.0625),
            horizon=2,
        )
        cache = RadialGittinsBoundaryCache()
        original = cache.get(direction=direction, grid=grid, **kwargs)
        mirrored = cache.get(
            direction=swapped_direction,
            grid=swapped_grid,
            **kwargs,
        )
        direct = build_radial_gittins_boundary_table(
            direction=swapped_direction,
            grid=swapped_grid,
            **kwargs,
        )

        self.assertEqual(len(cache), 1)
        self.assertIs(
            mirrored,
            cache.get(
                direction=swapped_direction,
                grid=swapped_grid,
                **kwargs,
            ),
        )
        self.assertEqual(mirrored.direction, swapped_direction)
        self.assertEqual(mirrored.initial_var, kwargs["initial_var"])
        self.assertEqual(mirrored.obs_noise_var, kwargs["obs_noise_var"])
        self.assertEqual(mirrored.grid, swapped_grid)
        np.testing.assert_array_equal(
            mirrored.boundaries,
            original.boundaries[:, ::-1],
        )
        np.testing.assert_allclose(
            mirrored.boundaries,
            direct.boundaries,
            atol=2e-12,
            rtol=0.0,
        )
        for stage in range(kwargs["horizon"] + 1):
            for delta in (-2.0, -1.0, 0.0, 0.5, 1.0):
                self.assertAlmostEqual(
                    mirrored.boundary(stage, delta),
                    original.boundary(stage, -delta),
                    places=14,
                )

    def test_mirror_reuse_falls_back_for_asymmetric_posterior_parameters(self):
        cases = (
            ((0.04, 0.05), (0.0625, 0.0625)),
            ((0.04, 0.04), (0.05, 0.07)),
        )
        for initial_var, obs_noise_var in cases:
            with self.subTest(
                initial_var=initial_var,
                obs_noise_var=obs_noise_var,
            ):
                cache = RadialGittinsBoundaryCache()
                with patch(
                    "agentopt.model_selection.radial_gittins_dp."
                    "build_radial_gittins_boundary_table",
                    wraps=build_radial_gittins_boundary_table,
                ) as builder:
                    for direction in ((0.3, 0.7), (0.7, 0.3)):
                        cache.get(
                            direction=direction,
                            effective_pull_cost=0.05,
                            initial_var=initial_var,
                            obs_noise_var=obs_noise_var,
                            horizon=1,
                            grid=MIRROR_TEST_GRID,
                        )

                self.assertEqual(builder.call_count, 2)
                self.assertEqual(len(cache), 2)

    def test_mirror_reuse_falls_back_for_asymmetric_reference_grids(self):
        cache = RadialGittinsBoundaryCache()
        directions = ((0.3, 0.7), (0.7, 0.3))
        grids = tuple(
            direction_aware_grid(
                direction,
                base_grid=TEST_GRID,
                reference=(0.1, 0.2),
            )
            for direction in directions
        )

        with patch(
            "agentopt.model_selection.radial_gittins_dp."
            "build_radial_gittins_boundary_table",
            wraps=build_radial_gittins_boundary_table,
        ) as builder:
            for direction, grid in zip(directions, grids):
                cache.get(
                    direction=direction,
                    effective_pull_cost=0.05,
                    initial_var=0.04,
                    obs_noise_var=0.0625,
                    horizon=1,
                    grid=grid,
                )

        self.assertEqual(builder.call_count, 2)
        self.assertEqual(len(cache), 2)

    def test_boundary_interpolation_rejects_out_of_grid_delta(self):
        table = build_radial_gittins_boundary_table(
            direction=(0.5, 0.5),
            effective_pull_cost=0.05,
            initial_var=0.04,
            obs_noise_var=0.0625,
            horizon=1,
            grid=TEST_GRID,
        )
        with self.assertRaises(BoundaryGridError):
            table.boundary(0, TEST_GRID.delta_max + 0.1)
        with self.assertRaises(IndexError):
            table.boundary(2, 0.0)

    def test_completed_index_uses_posterior_uncertainty(self):
        observed = terminal_expected_radial_utility(
            mean=(0.5, 0.5),
            var=(0.04, 0.04),
            direction=(0.5, 0.5),
        )
        self.assertAlmostEqual(observed, 0.5 - 0.2 / math.sqrt(math.pi), places=12)
        self.assertLess(observed, 0.5)


if __name__ == "__main__":
    unittest.main()
