#!/usr/bin/env python3
"""Submit all missing 20-seed cells as parallel Slurm arrays."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "analysis/complete_20seed_runs"
MANIFEST_ROOT = RUN_ROOT / "manifests"
LOG_ROOT = RUN_ROOT / "logs"
STATE_PATH = RUN_ROOT / "submitted_jobs.json"
SBATCH = ROOT / "experiments/combined_objective/run_20seed_manifest.sbatch"
FINALIZE = ROOT / "experiments/combined_objective/finalize_20seed_results.sbatch"
RESOURCES = {
    "ege_sr": {"cpus": 4, "mem": "8G", "time": "48:00:00"},
    "ape_k": {"cpus": 4, "mem": "8G", "time": "48:00:00"},
    "qnehvi": {"cpus": 4, "mem": "32G", "time": "48:00:00"},
    "random": {"cpus": 2, "mem": "4G", "time": "04:00:00"},
}


def submit(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    job_id = completed.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job id from {completed.stdout!r}")
    return job_id


def save(state: dict[str, object]) -> None:
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE_PATH)


def main() -> None:
    if STATE_PATH.exists():
        raise SystemExit(f"Refusing duplicate submission; state already exists: {STATE_PATH}")
    counts = json.loads((MANIFEST_ROOT / "counts.json").read_text())
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {"arrays": {}}
    save(state)
    try:
        for group in ("ege_sr", "ape_k", "qnehvi", "random"):
            count = int(counts[group])
            if count == 0:
                continue
            resources = RESOURCES[group]
            job_id = submit(
                [
                    "sbatch",
                    "--parsable",
                    f"--job-name=complete-{group}",
                    f"--array=0-{count - 1}",
                    f"--cpus-per-task={resources['cpus']}",
                    f"--mem={resources['mem']}",
                    f"--time={resources['time']}",
                    f"--output={LOG_ROOT}/%A_{group}_%a.out",
                    f"--error={LOG_ROOT}/%A_{group}_%a.err",
                    str(SBATCH),
                    str(MANIFEST_ROOT / f"{group}.tsv"),
                ]
            )
            state["arrays"][group] = {"job_id": job_id, "task_count": count}
            save(state)
            print(f"submitted {group}: {job_id} ({count} tasks)", flush=True)

        dependencies = ":".join(
            str(details["job_id"])
            for details in state["arrays"].values()
        )
        finalizer_id = submit(
            [
                "sbatch",
                "--parsable",
                f"--dependency=afterany:{dependencies}",
                str(FINALIZE),
            ]
        )
        state["finalizer_job_id"] = finalizer_id
        save(state)
        print(f"submitted finalizer: {finalizer_id}", flush=True)
    except Exception:
        save(state)
        raise


if __name__ == "__main__":
    main()
