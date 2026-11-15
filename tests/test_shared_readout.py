import json

import pytest

from experiments.combined_objective import cost_aware_replay as replay
from experiments.combined_objective import run_shared_readout as shared
from experiments.single_objective.offline_selector_sim import SampleResult


@pytest.fixture
def benchmark(monkeypatch):
    models = ("cheap", "middle", "expensive")
    questions = list(range(31))
    table = {
        model: {
            q: SampleResult(
                score=quality + .02 * (q % 3), cost=cost * (1 + .03 * (q % 5)),
                latency_seconds=0, input_tokens={}, output_tokens={},
            ) for q in questions
        } for model, quality, cost in zip(models, (.2, .5, .8), (.01, .02, .03))
    }
    monkeypatch.setattr(shared, "load_benchmark", lambda _: (models, questions, table, {}))


def test_shared_readout_uses_finite_benchmark_acquisition(benchmark, monkeypatch, tmp_path):
    simulator = replay.simulate_radial_gittins
    targets = []

    def simulate(*args, **kwargs):
        targets.append(kwargs["acquisition_target"])
        return simulator(*args, **kwargs)

    monkeypatch.setattr(replay, "simulate_radial_gittins", simulate)
    shared.run_benchmark("mathqa", (42,), (replay.CC_GITTINS,), tmp_path)
    result = json.loads((tmp_path / "mathqa-seed42.json").read_text())
    assert targets == ["finite_benchmark_mean"]
    assert result["cc_gittins_acquisition_target"] == "finite_benchmark_mean"
    assert {row["readout"] for row in result["rows"]} == {"finite_mean_beta0", replay.READOUT}
    assert all(row["cost_fraction"] <= row["target_cost_fraction"] + 1e-10
               for row in result["rows"])


def test_readouts_share_observations_and_spend_for_each_policy(benchmark, tmp_path):
    shared.run_benchmark("mathqa", (42,), replay.DEFAULT_METHODS, tmp_path)
    result = json.loads((tmp_path / "mathqa-seed42.json").read_text())
    grouped = {}
    for row in result["rows"]:
        grouped.setdefault((row["method"], row["target_cost_fraction"]), []).append(row)
    expected = {
        replay.CC_GITTINS: {"finite_mean_beta0", "finite_lcb_beta1"},
        replay.RANDOM_QUESTIONS: {"empirical_means", "finite_mean_beta0", "finite_lcb_beta1"},
    }
    assert {method for method, _ in grouped} == set(expected)
    for (method, _), rows in grouped.items():
        assert {row["readout"] for row in rows} == expected[method]
        assert len({(row["actual_usd"], row["evaluations"]) for row in rows}) == 1


@pytest.mark.parametrize("saved_target", (None, "latent_mean"))
def test_shared_readout_rejects_old_acquisition_results(benchmark, tmp_path, saved_target):
    shared.run_benchmark("mathqa", (42,), (replay.CC_GITTINS,), tmp_path)
    path = tmp_path / "mathqa-seed42.json"
    result = json.loads(path.read_text())
    if saved_target is None:
        result.pop("cc_gittins_acquisition_target")
    else:
        result["cc_gittins_acquisition_target"] = saved_target
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="different protocol"):
        shared.run_benchmark("mathqa", (42,), (replay.CC_GITTINS,), tmp_path)
