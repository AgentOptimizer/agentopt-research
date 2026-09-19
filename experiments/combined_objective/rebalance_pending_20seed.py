#!/usr/bin/env python3
"""Requeue only pending 20-seed array elements with measured memory requests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "analysis/complete_20seed_runs"
MANIFEST_ROOT = RUN_ROOT / "manifests"
LOG_ROOT = RUN_ROOT / "logs"
STATE_PATH = RUN_ROOT / "submitted_jobs.json"
REBALANCE_PATH = RUN_ROOT / "rebalanced_jobs.json"
SBATCH = ROOT / "experiments/combined_objective/run_20seed_manifest.sbatch"
FINALIZE = ROOT / "experiments/combined_objective/finalize_20seed_results.sbatch"
RESOURCES = {
    "ege_sr": {"cpus": 4, "mem": "8G", "time": "48:00:00"},
    "ape_k": {"cpus": 4, "mem": "8G", "time": "48:00:00"},
    "qnehvi": {"cpus": 4, "mem": "8G", "time": "48:00:00"},
    "random": {"cpus": 2, "mem": "4G", "time": "04:00:00"},
}


def run(command: list[str], *, capture: bool = False) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def submit(command: list[str]) -> str:
    output = run(command, capture=True)
    job_id = output.split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job id from {output!r}")
    return job_id


def pending_indices(job_id: str) -> list[int]:
    completed = subprocess.run(
        ["squeue", "-j", job_id, "-r", "-h", "-t", "PENDING", "-o", "%K"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return []
    output = completed.stdout.strip()
    return sorted({int(value) for value in output.splitlines() if value.strip().isdigit()})


def save(state: dict[str, object]) -> None:
    temporary = REBALANCE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(REBALANCE_PATH)


def main() -> None:
    state: dict[str, object] = {"replacements": {}}
    if REBALANCE_PATH.exists():
        state = json.loads(REBALANCE_PATH.read_text())
        if state.get("finalizer_job_id"):
            raise SystemExit(f"Refusing duplicate rebalance: {REBALANCE_PATH}")
    original = json.loads(STATE_PATH.read_text())
    save(state)

    for group in ("ege_sr", "ape_k", "qnehvi", "random"):
        if group in state["replacements"]:
            continue
        old_job = str(original["arrays"][group]["job_id"])
        indices = pending_indices(old_job)
        if not indices:
            continue
        run(["scancel", *[f"{old_job}_{index}" for index in indices]])
        resources = RESOURCES[group]
        array_spec = ",".join(str(index) for index in indices)
        new_job = submit(
            [
                "sbatch",
                "--parsable",
                f"--job-name=complete-{group}-lowmem",
                f"--array={array_spec}",
                f"--cpus-per-task={resources['cpus']}",
                f"--mem={resources['mem']}",
                f"--time={resources['time']}",
                f"--output={LOG_ROOT}/%A_{group}_%a.out",
                f"--error={LOG_ROOT}/%A_{group}_%a.err",
                str(SBATCH),
                str(MANIFEST_ROOT / f"{group}.tsv"),
            ]
        )
        state["replacements"][group] = {
            "cancelled_pending_from": old_job,
            "job_id": new_job,
            "task_count": len(indices),
            "task_indices": indices,
            "memory": resources["mem"],
        }
        save(state)
        print(f"rebalanced {group}: {new_job} ({len(indices)} tasks)", flush=True)

    old_finalizers = [str(original["finalizer_job_id"])]
    queued_finalizers = run(
        ["squeue", "-h", "-n", "complete-20seed-finalize", "-o", "%A"],
        capture=True,
    )
    old_finalizers.extend(
        value for value in queued_finalizers.splitlines() if value.isdigit()
    )
    old_finalizers = sorted(set(old_finalizers))
    run(["scancel", *old_finalizers])
    dependencies = [
        str(details["job_id"]) for details in original["arrays"].values()
    ]
    dependencies.extend(
        str(details["job_id"]) for details in state["replacements"].values()
    )
    finalizer = submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{':'.join(dependencies)}",
            str(FINALIZE),
        ]
    )
    state["cancelled_finalizer_job_ids"] = old_finalizers
    state["finalizer_job_id"] = finalizer
    save(state)
    print(f"submitted replacement finalizer: {finalizer}", flush=True)


if __name__ == "__main__":
    main()
