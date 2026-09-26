"""Submission metadata must remain portable without exposing local identity."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from experiments.combined_objective import run_gittins_ablation_v2_slurm_task as task
from experiments.combined_objective.anonymous_metadata import artifact_reference
from experiments.combined_objective.three_objective_metrics import (
    MATRIX_FILENAMES,
    benchmark_input_hashes,
)


class AnonymousExperimentMetadataTests(unittest.TestCase):
    def _run_task(self, directory, *, configuration=task.G0_CONFIGURATION, extra_patches=()):
        pair = task.CONFIGURATIONS[configuration]["pair_name"]
        result = Path(directory) / configuration / f"seed-{task.SEEDS[0]}" / pair / task.BENCHMARKS[0] / "result.json"

        def run_command(command, **kwargs):
            # Paths used for execution still resolve to the requested location.
            self.assertEqual(Path(command[command.index("--outdir") + 1]), result.parents[2])
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(json.dumps({"config": {"wall_time_seconds": 0.25}, "run": {}}))
            return SimpleNamespace(returncode=0)

        output = io.StringIO()
        argv = ["task", "--configuration", configuration, "--task-id", "0", "--output-root", directory]
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch("sys.argv", argv))
            stack.enter_context(mock.patch.object(task.subprocess, "run", side_effect=run_command))
            stack.enter_context(mock.patch.object(task, "resource", None))
            stack.enter_context(mock.patch.dict(task.os.environ, {
                "SLURM_JOB_ACCOUNT": "private-account", "SLURM_JOB_PARTITION": "private-cluster",
                "SLURM_JOB_NAME": "private-person", "SLURM_SUBMIT_DIR": directory,
                "SLURM_JOB_NODELIST": "private-host", "SLURM_JOB_ID": "1234567",
                "SLURM_CPUS_PER_TASK": "2", "SLURM_MEM_PER_NODE": "4096",
            }, clear=True))
            for patch in extra_patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(output))
            with self.assertRaises(SystemExit) as stopped:
                task.main()
        metadata = json.loads(result.with_name("slurm_task_metadata.json").read_text())
        self.assertEqual(json.loads(output.getvalue()), metadata)
        return metadata, stopped.exception.code

    def test_successful_run_without_unix_resource_keeps_reproducibility_not_identity(self):
        with tempfile.TemporaryDirectory(prefix="private-person-") as directory:
            metadata, code = self._run_task(directory)
            serialized = json.dumps(metadata)
            self.assertNotIn(directory, serialized)
            self.assertNotIn("private-", serialized)
            self.assertNotIn(task.sys.executable, serialized)
            self.assertEqual(metadata["result_sha256"], task.sha256_file(
                Path(directory) / Path(metadata["result_path"]).relative_to("results")
            ))
        self.assertEqual(code, 0)
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["slurm"], {"SLURM_CPUS_PER_TASK": "2", "SLURM_MEM_PER_NODE": "4096"})
        self.assertEqual(metadata["simulation_wall_time_seconds"], 0.25)
        for field in ("child_user_cpu_seconds", "child_system_cpu_seconds", "child_max_rss_kib"):
            self.assertIsNone(metadata[field])
        for field in ("hostname", "git_commit", "git_status_porcelain"):
            self.assertNotIn(field, metadata)
        self.assertTrue(all(len(value) == 64 for value in metadata["source_sha256"].values()))
        self.assertEqual(metadata["command"][0], "python")
        self.assertEqual(metadata["python"], task.platform.python_version())

    def test_missing_baseline_does_not_copy_private_path_from_os_error(self):
        configuration = next(name for name in task.RUN_CONFIGURATIONS if name != task.G0_CONFIGURATION)
        with tempfile.TemporaryDirectory(prefix="private-person-") as directory:
            missing = Path(directory) / "private-account" / "not-present.json"
            metadata, code = self._run_task(directory, configuration=configuration, extra_patches=(
                mock.patch.object(task, "protocol_result_path", return_value=missing),
            ))
            # The exception description must not include the full OS error text.
            failure = metadata["g0_truth_validation_error"]
            self.assertEqual(failure["error"], "failed to load G0 truth reference (FileNotFoundError)")
            self.assertNotIn(directory, json.dumps(metadata))
        self.assertEqual(code, 2)

    def test_dry_run_records_relocatable_command_for_external_output(self):
        with tempfile.TemporaryDirectory(prefix="private-person-") as directory:
            output = io.StringIO()
            argv = ["task", "--configuration", task.G0_CONFIGURATION, "--task-id", "0",
                    "--output-root", directory, "--dry-run"]
            with mock.patch("sys.argv", argv), contextlib.redirect_stdout(output):
                task.main()
            mapping = json.loads(output.getvalue())
        self.assertNotIn("private-person", output.getvalue())
        self.assertEqual(mapping["command"][0], "python")
        self.assertEqual(mapping["path_base"], "results")
        self.assertFalse(Path(mapping["command"][2]).is_absolute())
        self.assertTrue(mapping["result_path"].startswith("results/"))

    def test_external_path_references_are_platform_independent(self):
        root = task.ROOT
        self.assertEqual(artifact_reference(root / "data/example.csv", root=root), "data/example.csv")
        self.assertEqual(artifact_reference("C:\\Users\\private-person\\source.json", root=root), "external/source.json")
        self.assertEqual(artifact_reference("/home/private-person/source.json", root=root), "external/source.json")

    def test_external_matrix_hashes_are_stable_after_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            hashes = []
            for person in ("first-person", "second-person"):
                source = Path(directory) / person
                source.mkdir()
                for filename in (*MATRIX_FILENAMES, "metadata.json"):
                    (source / filename).write_bytes(filename.encode())
                hashes.append(benchmark_input_hashes(source))
            self.assertEqual(hashes[0], hashes[1])
            self.assertEqual(hashes[0]["external/latency_matrix_seconds.csv"],
                             hashlib.sha256(b"latency_matrix_seconds.csv").hexdigest())
            self.assertNotIn(directory, json.dumps(hashes))

    def test_compaction_sanitizes_old_provenance_without_mutating_the_source(self):
        from experiments.combined_objective.run_three_objective_gittins_multiseed import compact_payload
        provenance = {"source_result": "/home/private-person/result.json", "hostname": "private-host",
                      "git_commit": "original-history", "source_result_sha256": "a" * 64,
                      "launcher_source_sha256": "b" * 64}
        original = {"run": {"points": []}, "provenance": provenance.copy()}
        compact = compact_payload(original)
        self.assertEqual(original["provenance"], provenance)
        self.assertEqual(compact["provenance"], {
            "source_result": "external/result.json", "source_result_sha256": "a" * 64,
            "launcher_source_sha256": "b" * 64,
        })


if __name__ == "__main__":
    unittest.main()
