import importlib.util
import unittest

import numpy as np

from agentopt.model_selection.radial_gittins_dp import (
    RadialGittinsBoundaryCache,
    RadialGittinsGrid,
    build_radial_gittins_boundary_table,
)
from agentopt.model_selection.radial_gittins_dp_jax import (
    RadialGittinsJaxTableSpec,
    build_radial_gittins_boundary_tables_jax,
)
from experiments.combined_objective.offline_radial_gittins import (
    simulate_radial_gittins,
)
from experiments.single_objective.offline_selector_sim import SampleResult


HAS_JAX = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(HAS_JAX, "JAX is an optional dependency")
class RadialGittinsJaxBatchTests(unittest.TestCase):
    def test_heterogeneous_padded_batch_matches_scipy_tables(self):
        first_grid = RadialGittinsGrid(
            z_min=-4.0,
            z_max=4.0,
            z_size=41,
            delta_min=-3.0,
            delta_max=3.0,
            delta_size=31,
            state_size=41,
            state_halo=5.0,
            boundary_margin_cells=2,
            monotonicity_tolerance=1e-5,
        )
        second_grid = RadialGittinsGrid(
            z_min=-4.5,
            z_max=4.25,
            z_size=47,
            delta_min=-3.4,
            delta_max=2.8,
            delta_size=37,
            state_size=53,
            state_halo=4.5,
            kernel_stddevs=5.5,
            boundary_margin_cells=2,
            monotonicity_tolerance=1e-5,
        )
        direction = (0.35, 0.65)
        horizon = 3
        specifications = (
            RadialGittinsJaxTableSpec(
                effective_pull_cost=0.03,
                initial_var=(0.04, 0.025),
                obs_noise_var=(0.0625, 0.05),
                grid=first_grid,
            ),
            RadialGittinsJaxTableSpec(
                effective_pull_cost=0.075,
                initial_var=(0.05, 0.035),
                obs_noise_var=(0.08, 0.045),
                grid=second_grid,
            ),
        )

        expected = tuple(
            build_radial_gittins_boundary_table(
                direction=direction,
                effective_pull_cost=spec.effective_pull_cost,
                initial_var=spec.initial_var,
                obs_noise_var=spec.obs_noise_var,
                horizon=horizon,
                grid=spec.grid,
            )
            for spec in specifications
        )
        actual = build_radial_gittins_boundary_tables_jax(
            direction=direction,
            horizon=horizon,
            specs=specifications,
        )

        self.assertEqual(len(actual), 2)
        for reference, table, spec in zip(expected, actual, specifications):
            self.assertEqual(table.direction, direction)
            self.assertEqual(table.effective_pull_cost, spec.effective_pull_cost)
            self.assertIs(table.grid, spec.grid)
            self.assertEqual(table.boundaries.dtype, np.dtype(np.float64))
            np.testing.assert_allclose(
                table.boundaries,
                reference.boundaries,
                rtol=2e-11,
                atol=2e-11,
            )
            self.assertAlmostEqual(
                table.max_monotonicity_violation,
                reference.max_monotonicity_violation,
                places=12,
            )

    def test_float64_enablement_is_local_to_solver_call(self):
        import jax

        grid = RadialGittinsGrid(
            z_min=-4.0,
            z_max=4.0,
            z_size=41,
            delta_min=-3.0,
            delta_max=3.0,
            delta_size=31,
            state_size=41,
            state_halo=5.0,
            boundary_margin_cells=2,
            monotonicity_tolerance=1e-5,
        )
        was_enabled = bool(jax.config.read("jax_enable_x64"))
        table = build_radial_gittins_boundary_tables_jax(
            direction=(0.5, 0.5),
            horizon=1,
            specs=(
                RadialGittinsJaxTableSpec(
                    effective_pull_cost=0.05,
                    initial_var=0.04,
                    obs_noise_var=0.0625,
                    grid=grid,
                ),
            ),
        )[0]

        self.assertEqual(table.boundaries.dtype, np.dtype(np.float64))
        self.assertEqual(
            bool(jax.config.read("jax_enable_x64")),
            was_enabled,
        )

    def test_direction_lazy_replay_matches_scipy_policy(self):
        models = [f"M{index}" for index in range(4)]
        datapoints = list(range(5))
        scores = (
            (0.52, 0.60, 0.55, 0.58, 0.61),
            (0.49, 0.54, 0.57, 0.59, 0.56),
            (0.45, 0.51, 0.62, 0.53, 0.60),
            (0.50, 0.48, 0.56, 0.63, 0.58),
        )
        deployment_costs = (0.01, 0.02, 0.03, 0.04)
        table = {
            model: {
                question: SampleResult(
                    score=scores[arm][question],
                    latency_seconds=0.1,
                    input_tokens={},
                    output_tokens={},
                    cost=deployment_costs[arm],
                )
                for question in datapoints
            }
            for arm, model in enumerate(models)
        }
        grid = RadialGittinsGrid(
            z_min=-5.0,
            z_max=5.0,
            z_size=41,
            delta_min=-4.0,
            delta_max=4.0,
            delta_size=41,
            state_size=41,
            state_halo=5.0,
            boundary_margin_cells=1,
            monotonicity_tolerance=1e-5,
        )
        shared = {
            "batch_size": 1,
            "directions": ((0.5, 0.5),),
            "expected_batch_cost_usd": (0.012, 0.021, 0.033, 0.047),
            "effective_cost_bin_ratio": None,
            "max_total_question_evaluations": 12,
            "boundary_grid": grid,
            "halt_on_gittins_stop": False,
            "seed": 7,
        }

        results = {
            backend: simulate_radial_gittins(
                models,
                datapoints,
                table,
                boundary_cache=RadialGittinsBoundaryCache(),
                boundary_build_backend=backend,
                **shared,
            )
            for backend in ("scipy", "jax")
        }

        def selected_arms(result):
            return [
                event["selected_arm"]
                for event in result.trace
                if event["event"] == "direction_visit"
                and event["selected_arm"] is not None
            ]

        scipy_result = results["scipy"]
        jax_result = results["jax"]
        self.assertEqual(selected_arms(jax_result), selected_arms(scipy_result))
        self.assertEqual(jax_result.stop_reason, scipy_result.stop_reason)
        self.assertEqual(jax_result.selected_models, scipy_result.selected_models)
        self.assertEqual(
            jax_result.total_evaluations,
            scipy_result.total_evaluations,
        )
        self.assertEqual(jax_result.total_cost, scipy_result.total_cost)
        self.assertEqual(jax_result.hypervolume, scipy_result.hypervolume)
        self.assertEqual(
            jax_result.params["boundary_table_build"]["stats"]["jax_groups"],
            1,
        )
        self.assertEqual(
            scipy_result.params["boundary_table_build"]["stats"][
                "scipy_tables"
            ],
            4,
        )


if __name__ == "__main__":
    unittest.main()
