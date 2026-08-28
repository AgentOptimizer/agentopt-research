import math
import unittest

import numpy as np

from agentopt.model_selection.radial_gittins_dp import (
    BoundaryGridError,
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
    axis_aware_grid,
    build_radial_gittins_boundary_table,
    build_scalar_gittins_boundary_table,
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



class ExactAxisGittinsTests(unittest.TestCase):
    def test_axis_grid_is_finite_without_near_zero_direction_scaling(self):
        grid = axis_aware_grid(reference=0.0)

        self.assertEqual((grid.z_min, grid.z_max), (-2.0, 2.0))
        self.assertEqual((grid.delta_min, grid.delta_max), (-12.0, 12.0))

    def test_scalar_table_has_terminal_mean_index_and_cost_ordering(self):
        grid = RadialGittinsGrid(
            z_min=-2.0,
            z_max=2.0,
            z_size=81,
            delta_min=-3.0,
            delta_max=3.0,
            delta_size=61,
            state_size=81,
            state_halo=3.0,
            boundary_margin_cells=2,
        )
        low_cost = build_scalar_gittins_boundary_table(
            objective_index=0,
            effective_pull_cost=0.001,
            initial_var=0.04,
            obs_noise_var=0.0625,
            horizon=3,
            grid=grid,
        )
        high_cost = build_scalar_gittins_boundary_table(
            objective_index=0,
            effective_pull_cost=0.05,
            initial_var=0.04,
            obs_noise_var=0.0625,
            horizon=3,
            grid=grid,
        )

        self.assertEqual(low_cost.boundary(3), 0.0)
        self.assertEqual(high_cost.boundary(3), 0.0)
        self.assertGreater(high_cost.boundary(0), low_cost.boundary(0))

    def test_scalar_cache_reuses_matching_axis_tables(self):
        cache = RadialGittinsBoundaryCache()
        kwargs = dict(
            objective_index=1,
            effective_pull_cost=0.05,
            initial_var=0.04,
            obs_noise_var=0.0625,
            horizon=3,
            grid=axis_aware_grid(),
        )

        first = cache.get_axis(**kwargs)
        second = cache.get_axis(**kwargs)

        self.assertIs(first, second)
        self.assertEqual(len(cache), 1)


if __name__ == "__main__":
    unittest.main()
