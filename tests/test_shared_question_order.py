"""Paired question prefixes share an order, not an observation cursor."""

import json
import unittest

import numpy as np

from agentopt.model_selection.radial_gittins import PerArmQuestionSchedule
from experiments.combined_objective import offline_radial_gittins as replay
from experiments.single_objective.offline_selector_sim import SampleResult


_ARM_IDS = ("a", "b", "c")
# Captured from the historical implementation before adding shared tails.
# The two factories historically permuted different tail candidate orders.
_CREATE_GOLDEN = {
    "a": (7, 4, 9, 2, 5, 0, 11, 1, 10, 8, 3, 6),
    "b": (7, 4, 9, 2, 10, 5, 0, 3, 1, 11, 6, 8),
    "c": (7, 4, 9, 2, 11, 8, 3, 10, 6, 5, 0, 1),
}
_AVAILABLE_GOLDEN = {
    "a": (7, 4, 9, 2, 1, 3, 6, 0, 8, 10, 5, 11),
    "b": (7, 4, 9, 2, 8, 1, 3, 5, 0, 6, 11, 10),
    "c": (7, 4, 9, 2, 6, 10, 5, 8, 11, 1, 3, 0),
}


def _schedule(mode="shared", **overrides):
    options = dict(n_questions=12, warm_start_batch_size=4, seed=42,
                   question_order=mode)
    options.update(overrides)
    return PerArmQuestionSchedule.create(_ARM_IDS, **options)


def _problem(available=None):
    if available is None:
        available = {arm: tuple(range(12)) for arm in _ARM_IDS}
    questions = sorted(set().union(*(set(q) for q in available.values())))
    table = {
        model: {
            q: SampleResult(score=.2 + .15 * arm + .06 * (q % 5),
                            cost=.001 * (arm + 1) + .00001 * q,
                            latency_seconds=.1, input_tokens={}, output_tokens={})
            for q in ids
        }
        for arm, (model, ids) in enumerate(available.items())
    }
    return list(available), questions, table


def _run(problem=None, **overrides):
    options = dict(
        seed=42, batch_size=4, anytime=True,
        eta_decay_schedule="direction_stop", directions=((.5, .5), (1., 0.)),
        cost_model="raw_mean", recommendation_rule="finite_mean",
        recommendation_min_samples=4, effective_cost_bin_ratio=None,
        record_trace=True, record_recommendation_trajectory=True,
        recommendation_changes_only=True,
        index_provider=lambda context, arm: (
            100. - arm if context.current_lambda <= .5 else -100.
        ),
    )
    options.update(overrides)
    return replay.simulate_radial_gittins(*(_problem() if problem is None else problem), **options)


def _physical_events(result):
    return [event for event in result.trace
            if event["event"] == "warm_start" or event.get("selected_arm") is not None]


def _questions_by_arm(result):
    observed = {arm: [] for arm in range(len(result.model_results))}
    for event in _physical_events(result):
        arm = event["arm_index"] if event["event"] == "warm_start" else event["selected_arm"]
        observed[arm].extend(event["question_ids"])
    return observed


class SharedScheduleTests(unittest.TestCase):
    def test_independent_default_preserves_each_factory_seed42_golden_order(self):
        for explicit in (False, True):
            mode = {"question_order": "independent"} if explicit else {}
            with self.subTest(explicit=explicit):
                regular = PerArmQuestionSchedule.create(
                    _ARM_IDS, n_questions=12, warm_start_batch_size=4, seed=42, **mode)
                available = PerArmQuestionSchedule.create_from_available(
                    {arm: range(12) for arm in _ARM_IDS},
                    warm_start_batch_size=4, seed=42, **mode)
                self.assertEqual(regular.orders, _CREATE_GOLDEN)
                self.assertEqual(available.orders, _AVAILABLE_GOLDEN)
                self.assertEqual(regular.question_order, "independent")
                self.assertEqual(available.question_order, "independent")

    def test_shared_changes_only_tail_and_both_complete_universe_factories_agree(self):
        regular = _schedule()
        available = PerArmQuestionSchedule.create_from_available(
            {arm: range(12) for arm in _ARM_IDS},
            warm_start_batch_size=4, seed=42, question_order="shared")
        self.assertEqual(regular.warm_start_question_ids, (7, 4, 9, 2))
        self.assertEqual(regular.warm_start_question_ids,
                         _schedule("independent").warm_start_question_ids)
        self.assertEqual(available.orders, regular.orders)
        self.assertEqual(len(set(regular.orders.values())), 1)
        self.assertEqual(regular.question_order, "shared")
        self.assertEqual(set(regular.orders["a"]), set(range(12)))

    def test_arm_cursors_advance_independently_through_same_question_prefix(self):
        schedule = _schedule()
        warm = schedule.take_uniform_warm_start()
        order = schedule.orders["a"]
        self.assertEqual(set(warm.values()), {order[:4]})
        self.assertEqual(schedule.next_batch("a", 5), order[4:9])
        self.assertEqual(schedule.attempted_question_ids("b"), order[:4])
        self.assertEqual(schedule.next_batch("b", 2), order[4:6])
        self.assertEqual(schedule.next_batch("c", 3), order[4:7])
        self.assertEqual(schedule.next_batch("a", 99), order[9:])
        self.assertEqual(schedule.next_batch("a", 1), ())
        self.assertEqual(schedule.next_batch("b", 99), order[6:])
        self.assertEqual(schedule.next_batch("c", 99), order[7:])
        for arm in _ARM_IDS:
            self.assertEqual(schedule.attempted_question_ids(arm), order)
            self.assertEqual(schedule.remaining(arm), 0)

    def test_shared_order_is_reproducible_and_not_arm_insertion_order_dependent(self):
        first = _schedule()
        repeated = _schedule()
        reversed_arms = PerArmQuestionSchedule.create(
            reversed(_ARM_IDS), n_questions=12, warm_start_batch_size=4,
            seed=42, question_order="shared")
        self.assertEqual(first.orders, repeated.orders)
        self.assertEqual(first.orders, reversed_arms.orders)
        self.assertNotEqual(first.orders["a"], _schedule(seed=43).orders["a"])

    def test_independent_warm_start_uses_each_arms_own_seeded_prefix(self):
        first = _schedule(
            "independent", warm_start_question_order="independent"
        )
        repeated = _schedule(
            "independent", warm_start_question_order="independent"
        )
        batches = first.take_uniform_warm_start()

        self.assertEqual(first.warm_start_question_ids, ())
        self.assertEqual(first.warm_start_batch_size, 4)
        self.assertEqual(first.orders, repeated.orders)
        self.assertEqual(
            first.warm_start_question_ids_by_arm,
            repeated.warm_start_question_ids_by_arm,
        )
        self.assertGreater(len(set(batches.values())), 1)
        for arm in _ARM_IDS:
            self.assertEqual(batches[arm], first.orders[arm][:4])
            self.assertEqual(
                batches[arm], first.warm_start_question_ids_by_arm[arm]
            )

    def test_independent_warm_start_does_not_require_a_common_question(self):
        available = {
            "a": (0, 1, 2, 3),
            "b": (4, 5, 6, 7),
            "c": (8, 9, 10, 11),
        }
        schedule = PerArmQuestionSchedule.create_from_available(
            available,
            warm_start_batch_size=2,
            seed=42,
            question_order="independent",
            warm_start_question_order="independent",
        )
        batches = schedule.take_uniform_warm_start()
        for arm, ids in available.items():
            self.assertEqual(len(batches[arm]), 2)
            self.assertTrue(set(batches[arm]).issubset(ids))
            self.assertEqual(set(schedule.orders[arm]), set(ids))

    def test_ragged_orders_filter_one_global_order_without_missing_or_repeating_cells(self):
        available = {
            "anchor": (1, 3, 5, 7, 9, 11, 13, 15, 20, 22, 30, 32),
            "a": (1, 3, 5, 7, 9, 11, 13, 15),
            "b": (1, 3, 5, 7, 9, 20, 22),
            "c": (1, 3, 5, 7, 9, 30, 32),
        }
        shared = PerArmQuestionSchedule.create_from_available(
            available, warm_start_batch_size=4, seed=42, question_order="shared")
        independent = PerArmQuestionSchedule.create_from_available(
            available, warm_start_batch_size=4, seed=42, question_order="independent")
        self.assertEqual(shared.warm_start_question_ids, (7, 3, 5, 9))
        self.assertEqual(shared.warm_start_question_ids, independent.warm_start_question_ids)
        shared.take_uniform_warm_start()
        global_order = shared.orders["anchor"]
        for arm, ids in available.items():
            expected = tuple(q for q in global_order if q in set(ids))
            self.assertEqual(shared.orders[arm], expected)
            self.assertEqual(shared.next_batch(arm, 100), expected[4:])
            seen = shared.attempted_question_ids(arm)
            self.assertEqual(set(seen), set(ids))
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(shared.next_batch(arm, 1), ())
        reordered = PerArmQuestionSchedule.create_from_available(
            {arm: tuple(reversed(ids)) for arm, ids in reversed(tuple(available.items()))},
            warm_start_batch_size=4, seed=42, question_order="shared")
        self.assertEqual(shared.orders, reordered.orders)

    def test_no_tail_remains_when_warm_batch_covers_all_available_rows(self):
        for mode in ("shared", "independent"):
            with self.subTest(mode=mode):
                schedule = _schedule(mode, n_questions=4)
                schedule.take_uniform_warm_start()
                for arm in _ARM_IDS:
                    self.assertEqual(schedule.next_batch(arm, 4), ())
                    self.assertEqual(set(schedule.attempted_question_ids(arm)), set(range(4)))

    def test_invalid_modes_fail_in_both_factories_and_direct_schedule(self):
        for mode in ("paired", "", None, True, 3):
            for factory in ("regular", "available", "direct"):
                with self.subTest(mode=mode, factory=factory), self.assertRaisesRegex(ValueError, "question_order"):
                    if factory == "regular":
                        _schedule(mode)
                    elif factory == "available":
                        PerArmQuestionSchedule.create_from_available(
                            {"a": range(4)}, warm_start_batch_size=4, question_order=mode)
                    else:
                        PerArmQuestionSchedule({"a": (0,)}, question_order=mode)

    def test_invalid_warm_start_modes_fail_in_both_factories(self):
        for mode in ("paired", "", None, True, 3):
            for factory in ("regular", "available"):
                with self.subTest(mode=mode, factory=factory), self.assertRaisesRegex(
                    ValueError, "warm_start_question_order"
                ):
                    if factory == "regular":
                        _schedule(warm_start_question_order=mode)
                    else:
                        PerArmQuestionSchedule.create_from_available(
                            {"a": range(4)},
                            warm_start_batch_size=2,
                            warm_start_question_order=mode,
                        )

    def test_shared_rejects_empty_duplicate_and_insufficient_common_questions(self):
        invalid = ({}, {"a": ()}, {"a": (1, 1, 2, 3)},
                   {"a": (1, 2, 3, 4), "b": (5, 6, 7, 8)},
                   {"a": (1, 2, 3, 4), "b": (1, 2, 3, 8)})
        for available in invalid:
            with self.subTest(available=available), self.assertRaises(ValueError):
                PerArmQuestionSchedule.create_from_available(
                    available, warm_start_batch_size=4, seed=42, question_order="shared")


class SharedReplayTests(unittest.TestCase):
    def test_shared_default_and_explicit_mode_have_identical_physical_replay(self):
        default, explicit = _run(), _run(question_order="shared")
        self.assertEqual(_physical_events(default), _physical_events(explicit))
        self.assertEqual(default.observed_cells, explicit.observed_cells)
        self.assertEqual(default.selected_models, explicit.selected_models)
        self.assertEqual(default.params["question_order"], "shared")

    def test_warm_observations_prior_and_noise_are_unchanged_by_tail_mode(self):
        shared, independent = _run(question_order="shared"), _run(question_order="independent")
        self.assertEqual([e for e in shared.trace if e["event"] == "warm_start"],
                         [e for e in independent.trace if e["event"] == "warm_start"])
        self.assertEqual(shared.prior_mean, independent.prior_mean)
        self.assertEqual(shared.prior_variance, independent.prior_variance)
        self.assertEqual(shared.cost_reference_usd, independent.cost_reference_usd)
        for key in ("obs_noise_variance", "reward_prior_mean", "reward_prior_variance",
                    "raw_cost_question_noise_variance_usd2", "finite_target_question_noise_variance"):
            self.assertEqual(shared.params[key], independent.params[key], key)
        shared_questions, independent_questions = _questions_by_arm(shared), _questions_by_arm(independent)
        self.assertEqual(len(set(map(tuple, shared_questions.values()))), 1)
        self.assertGreater(len(set(map(tuple, independent_questions.values()))), 1)
        for arm, name in enumerate(_ARM_IDS):
            self.assertEqual(tuple(independent_questions[arm]), _AVAILABLE_GOLDEN[name])

    def test_independent_warm_replay_uses_different_seeded_questions_per_arm(self):
        result = _run(
            question_order="independent",
            warm_start_question_order="independent",
        )
        warm = [
            tuple(event["question_ids"])
            for event in result.trace
            if event["event"] == "warm_start"
        ]
        self.assertGreater(len(set(warm)), 1)
        self.assertTrue(all(len(batch) == 4 for batch in warm))
        self.assertEqual(
            result.params["warm_start_question_order"], "independent"
        )
        self.assertEqual(
            result.params["question_order_rng_scheme"],
            "spawned_per_arm_full_orders",
        )
        self.assertIn(
            "independent_warm", result.params["question_order_semantics"]
        )
        self.assertEqual(result.params["warm_start_question_ids"], [])
        self.assertEqual(
            result.params["warm_start_question_ids_by_arm"],
            [list(batch) for batch in warm],
        )

    def test_every_shared_replay_batch_uses_its_own_prefix_and_reports_actual_observations(self):
        problem = _problem()
        result = _run(problem)
        schedule = _schedule()
        observed = {arm: [] for arm in range(3)}
        unequal_after_adaptive_pull = False
        for event in _physical_events(result):
            arm = event["arm_index"] if event["event"] == "warm_start" else event["selected_arm"]
            ids = event["question_ids"]
            previous = len(observed[arm])
            expected = schedule.orders[_ARM_IDS[arm]][previous:previous + len(ids)]
            self.assertEqual(tuple(ids), expected)
            observed[arm].extend(ids)
            samples = [problem[2][_ARM_IDS[arm]][q] for q in ids]
            self.assertAlmostEqual(event["batch_score_mean"], np.mean([s.score for s in samples]))
            self.assertAlmostEqual(event["actual_batch_search_cost_usd"], sum(s.cost for s in samples))
            if event["event"] != "warm_start":
                unequal_after_adaptive_pull |= len(set(map(len, observed.values()))) > 1
                self.assertEqual(event["cumulative_evaluations"], sum(map(len, observed.values())))
        self.assertTrue(unequal_after_adaptive_pull)
        expected_cells = {(arm, q) for arm in range(3) for q in range(12)}
        self.assertEqual(set(result.observed_cells), expected_cells)
        self.assertEqual(result.total_evaluations, len(expected_cells))
        self.assertAlmostEqual(result.hypervolume_regret, 0.)
        self.assertEqual(result.recommendation_final_snapshot.selected_arm_indices,
                         result.recommendation_final_event.selected_arm_indices)

    def test_ragged_shared_replay_covers_only_available_cells_and_keeps_common_relative_order(self):
        available = {"a": tuple(range(12)), "b": (0, 1, 2, 3, 4, 6, 8),
                     "c": (0, 1, 2, 3, 4, 5, 7, 9, 11)}
        result = _run(_problem(available), question_universe="per_arm")
        observed = _questions_by_arm(result)
        anchor = observed[0]
        for arm, ids in enumerate(available.values()):
            self.assertEqual(observed[arm], [q for q in anchor if q in set(ids)])
        expected = {(arm, q) for arm, ids in enumerate(available.values()) for q in ids}
        self.assertEqual(set(result.observed_cells), expected)
        self.assertEqual(result.total_evaluations, len(expected))
        self.assertEqual(result.params["available_cells_in_universe"], len(expected))
        self.assertEqual(result.params["ragged_tail_cells_excluded"], 0)
        self.assertEqual(result.params["unobserved_cells_at_stop"], 0)
        self.assertEqual(result.params["finite_target_question_counts"], [12, 7, 9])

    def test_serialized_question_order_metadata_distinguishes_sampling_modes(self):
        for mode, rng in (("shared", "shared_warm_rng_continuation"),
                          ("independent", "spawned_per_arm_tails")):
            with self.subTest(mode=mode):
                result = _run(question_order=mode)
                saved = json.loads(json.dumps(replay._jsonable_result(result)))
                self.assertEqual(saved["params"]["question_order"], mode)
                self.assertEqual(saved["params"]["question_order_rng_scheme"], rng)
                self.assertIn("shared_warm", saved["params"]["question_order_semantics"])
                self.assertEqual(saved["params"]["common_question_count"], 12)
                self.assertEqual(saved["params"]["eta_decay_schedule"], "direction_stop")
                self.assertEqual(saved["params"]["recommendation_min_samples"], 4)
                self.assertTrue(saved["params"]["recommendation_changes_only"])

    def test_replay_rejects_invalid_question_order(self):
        for mode in ("paired", "", None, True):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "question_order"):
                _run(question_order=mode)


if __name__ == "__main__":
    unittest.main()
