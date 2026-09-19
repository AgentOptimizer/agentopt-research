#!/usr/bin/env python3
"""Replace the too-small qNEHVI chunk run with a vectorized 512-candidate run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "analysis/complete_20seed_runs"
LOG_ROOT = RUN_ROOT / "logs"
STATE_PATH = RUN_ROOT / "submitted_jobs.json"
REBALANCE_PATH = RUN_ROOT / "rebalanced_jobs.json"
RESTART_PATH = RUN_ROOT / "qnehvi_restart.json"
MANIFEST = RUN_ROOT / "manifests/qnehvi.tsv"
SBATCH = ROOT / "experiments/combined_objective/run_20seed_manifest.sbatch"
FINALIZE = ROOT / "experiments/combined_objective/finalize_20seed_results.sbatch"


def run(command: list[str], *, capture: bool = False) -> str:
    completed = subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def submit(command: list[str]) -> str:
    output = run(command, capture=True)
    job_id = output.split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job id from {output!r}")
    return job_id


def save(state: dict[str, object]) -> None:
    temporary = RESTART_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESTART_PATH)


def main() -> None:
    if RESTART_PATH.exists():
        raise SystemExit(f"Refusing duplicate qNEHVI restart: {RESTART_PATH}")
    original = json.loads(STATE_PATH.read_text())
    rebalanced = json.loads(REBALANCE_PATH.read_text())
    old_q_jobs = [
        str(original["arrays"]["qnehvi"]["job_id"]),
        str(rebalanced["replacements"]["qnehvi"]["job_id"]),
    ]
    old_finalizer = str(rebalanced["finalizer_job_id"])
    run(["scancel", *old_q_jobs, old_finalizer])

    new_q_job = submit(
        [
            "sbatch",
            "--parsable",
            "--job-name=complete-qnehvi-batch512",
            "--array=0-58",
            "--cpus-per-task=4",
            "--mem=128G",
            "--time=48:00:00",
            f"--output={LOG_ROOT}/%A_qnehvi512_%a.out",
            f"--error={LOG_ROOT}/%A_qnehvi512_%a.err",
            str(SBATCH),
            str(MANIFEST),
        ]
    )

    dependencies = [
        str(details["job_id"])
        for group, details in original["arrays"].items()
        if group != "qnehvi"
    ]
    dependencies.extend(
        str(details["job_id"])
        for group, details in rebalanced["replacements"].items()
        if group != "qnehvi"
    )
    dependencies.append(new_q_job)
    finalizer = submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{':'.join(dependencies)}",
            str(FINALIZE),
        ]
    )
    state = {
        "cancelled_qnehvi_job_ids": old_q_jobs,
        "cancelled_finalizer_job_id": old_finalizer,
        "candidate_batch_size": 512,
        "qnehvi_job_id": new_q_job,
        "finalizer_job_id": finalizer,
    }
    save(state)
    print(f"submitted qNEHVI batch-512 array: {new_q_job}", flush=True)
    print(f"submitted replacement finalizer: {finalizer}", flush=True)


if __name__ == "__main__":
    main()
