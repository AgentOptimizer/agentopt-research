"""Distance metrics score every returned configuration in true coordinates."""

import math

import numpy as np
import pytest

from experiments.combined_objective import ablation_metrics as ablation
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.combined_objective import three_objective_metrics as three


@pytest.fixture(params=("replay", "ablation", "three_objective"))
def metric_case(request):
    # Arm 2 is dominated by arm 0, but is the nearest returned configuration
    # to missing true-front arm 1. A constant third objective keeps the same
    # analytic distances while exercising the three-objective path.
    truth = np.array([[0.2, 1.0], [1.0, 0.2], [0.2, 0.2]])
    if request.param == "three_objective":
        truth = np.column_stack((truth, np.full(3, 0.5)))
    return request.param, truth


def score(case, selected):
    name, truth = case
    front = truth[:2]
    if name == "replay":
        result = replay.front_quality_metrics(
            truth[list(selected)], front, (0.0, 0.0), replay.hypervolume_2d(front),
        )
        return (result.generational_distance, result.inverted_generational_distance,
                result.hypervolume_regret)
    if name == "ablation":
        result = ablation.score_selection(selected, truth, front, ablation.hypervolume_2d(front))
    else:
        result = three.score_selection(selected, truth, front, three.hypervolume_3d(front))
    return (result["generational_distance"], result["inverted_generational_distance"],
            result["hv_regret"])


def test_all_returned_points_affect_distances_but_dominated_points_do_not_change_hv(metric_case):
    gd, igd, hv_regret = score(metric_case, (0, 2))
    subset_gd, subset_igd, subset_hv_regret = score(metric_case, (0,))
    assert gd == pytest.approx(0.4)
    assert igd == pytest.approx(0.4)
    assert subset_gd == 0.0
    assert subset_igd == pytest.approx(math.sqrt(1.28) / 2.0)
    assert hv_regret == pytest.approx(subset_hv_regret)
    assert hv_regret == pytest.approx(0.08 if metric_case[0] == "three_objective" else 0.16)


def test_distinct_returned_configurations_with_equal_coordinates_all_count(metric_case):
    name, truth = metric_case
    duplicated = np.vstack((truth, truth[2]))
    gd, igd, _ = score((name, duplicated), (0, 2, 3))
    assert gd == pytest.approx(1.6 / 3.0)
    assert igd == pytest.approx(0.4)


def test_empty_recommendation_has_infinite_distances_and_full_hv_regret(metric_case):
    gd, igd, hv_regret = score(metric_case, ())
    assert math.isinf(gd)
    assert math.isinf(igd)
    assert hv_regret == pytest.approx(0.18 if metric_case[0] == "three_objective" else 0.36)


@pytest.mark.parametrize("distance,dimensions", ((ablation.front_distance, 2), (three.front_distance, 3)))
def test_empty_distance_inputs(distance, dimensions):
    empty = np.empty((0, dimensions))
    point = np.ones((1, dimensions))
    assert distance(empty, empty) == 0.0
    assert math.isinf(distance(empty, point))
    assert math.isinf(distance(point, empty))


def test_three_objective_enrichment_keeps_membership_and_false_positive_diagnostics():
    run = {
        "raw_truth_vectors": [[0.2, 0.0, 1.0], [1.0, 4.0, 1.0], [0.2, 4.0, 1.0]],
        "bruteforce_search_cost_usd": 3.0,
        "total_cost": 1.0,
        "points": [{"selected_arm_indices": [0, 2], "cumulative_search_cost_usd": 1.0}],
    }
    three.enrich_run(run)
    point = run["points"][0]
    assert point["selected_arm_indices"] == [0, 2]
    assert point["generational_distance"] == pytest.approx(0.25)
    assert point["inverted_generational_distance"] == pytest.approx(0.4)
    assert point["pareto_false_positive_count"] == 1
    assert point["pareto_false_negative_count"] == 1
    assert point["pareto_precision"] == 0.5
    assert point["pareto_recall"] == 0.5
