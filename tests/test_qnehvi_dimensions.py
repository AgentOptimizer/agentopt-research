"""Objective-dimension checks for the discrete qNEHVI wrapper."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from agentopt.model_selection.qnehvi import select_qnehvi_index


class QNEHVIDimensionTests(unittest.TestCase):
    def setUp(self):
        self.train_features = np.array([[0], [1], [2], [0], [1], [2]], dtype=float)
        self.candidate_features = np.array([[2], [0]], dtype=float)

    @unittest.skipUnless(importlib.util.find_spec("botorch"), "botorch is not installed")
    def test_one_gp_per_objective_and_matching_reference_point(self):
        import torch

        for n_objectives in (2, 3):
            with self.subTest(n_objectives=n_objectives):
                objectives = np.arange(6 * n_objectives, dtype=float).reshape(6, n_objectives) / 20
                reference = (0.0,) * n_objectives if n_objectives == 3 else (0.0, 0.0)
                with (
                    patch("botorch.fit.fit_gpytorch_mll"),
                    patch(
                        "botorch.acquisition.multi_objective.monte_carlo."
                        "qNoisyExpectedHypervolumeImprovement"
                    ) as acquisition_cls,
                ):
                    acquisition_cls.return_value.return_value = torch.tensor([0.1, 0.9])
                    kwargs = {} if n_objectives == 2 else {"reference_point": reference}
                    selected = select_qnehvi_index(
                        self.train_features,
                        objectives,
                        self.candidate_features,
                        categorical_dims=[0],
                        mc_samples=8,
                        **kwargs,
                    )

                self.assertEqual(selected, 1)
                acq_kwargs = acquisition_cls.call_args.kwargs
                self.assertEqual(len(acq_kwargs["model"].models), n_objectives)
                self.assertEqual(tuple(acq_kwargs["ref_point"].tolist()), reference)
                self.assertEqual(tuple(acquisition_cls.return_value.call_args.args[0].shape), (2, 1, 1))

    @unittest.skipUnless(importlib.util.find_spec("botorch"), "botorch is not installed")
    def test_rejects_one_objective_and_wrong_reference_length(self):
        with self.assertRaisesRegex(ValueError, "at least two columns"):
            select_qnehvi_index(
                self.train_features,
                np.ones((6, 1)),
                self.candidate_features,
                categorical_dims=[0],
            )
        with self.assertRaisesRegex(ValueError, "length-3"):
            select_qnehvi_index(
                self.train_features,
                np.ones((6, 3)),
                self.candidate_features,
                categorical_dims=[0],
                reference_point=(0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
