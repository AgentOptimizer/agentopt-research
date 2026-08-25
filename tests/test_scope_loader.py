import csv
import tempfile
import unittest
from pathlib import Path

from experiments.single_objective.offline_selector_sim import load_scope


class ScopeLoaderTests(unittest.TestCase):
    def _write_matrix(self, directory, filename, rows):
        path = Path(directory) / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model_name", "question_2", "question_7"])
            writer.writerows(rows)

    def test_loads_paired_matrices_as_lookup_table(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_matrix(
                directory,
                "accuracy_matrix.csv",
                [["config-a", 1, 0], ["config-b", 0.5, 1]],
            )
            self._write_matrix(
                directory,
                "cost_matrix_usd.csv",
                [["config-a", 0.1, 0.2], ["config-b", 0.01, 0.02]],
            )

            models, datapoints, table = load_scope(directory)

            self.assertEqual(models, ["config-a", "config-b"])
            self.assertEqual(datapoints, [2, 7])
            self.assertEqual(table["config-a"][2].score, 1.0)
            self.assertEqual(table["config-b"][7].cost, 0.02)

    def test_rejects_misaligned_matrix_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_matrix(
                directory, "accuracy_matrix.csv", [["config-a", 1, 0]]
            )
            self._write_matrix(
                directory, "cost_matrix_usd.csv", [["config-b", 0.1, 0.2]]
            )
            with self.assertRaisesRegex(ValueError, "different model rows"):
                load_scope(directory)


if __name__ == "__main__":
    unittest.main()
