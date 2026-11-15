"""Render paper figures from the bundled frozen plotting data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
GROUPS = {
    "hero": "Figure 1: BIRD Dev cost and frontier",
    "system_overview": "Figure 2: system diagram (XeLaTeX)",
    "main_curves": "Figures 3-5: eight-method main curves",
    "stackoverflow_frontier": "Figure 6: Stack Overflow frontier",
    "three_main": "Figure 7: three-objective main curves",
    "job_runtime": "Figure 8: Slurm job elapsed runtime",
    "frequencies": "Figures 9-10: recommendation frequencies",
    "estimated_actual": "Figures 11-12: actual/estimated frontiers",
    "two_ablations": "Figures 13,14,17,18: two-objective ablations",
    "quality_frontier": "Figure 15: CC-Gittins / quality-only frontiers",
    "two_direction_runtime": "Figure 16: direction policy runtime",
    "two_prior": "Figure 19: two-objective prior variance",
    "recommendation": "Figures 20,26: recommendation rules",
    "three_families": "Figures 21,24: three-objective cost and continuation",
    "three_directions": "Figure 22: three-objective directions",
    "three_direction_runtime": "Figure 23: three-objective direction runtime",
    "three_prior": "Figure 25: three-objective prior variance",
    "three_frontiers": "Supplement: complete 20-seed 3D frontiers (30%)",
}


def verify_data():
    manifest = json.loads((ROOT / "data_checksums.json").read_text(encoding="utf-8"))
    failures = [name for name, digest in manifest.items()
                if not (ROOT / name).is_file()
                or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    if failures:
        raise ValueError("Missing or changed plotting inputs: " + ", ".join(failures))
    print(f"Verified {len(manifest)} plotting inputs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--only", nargs="+", choices=GROUPS)
    parser.add_argument("--skip-diagram", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "rendered")
    parser.add_argument("--style", type=Path, default=ROOT / "style.json")
    args = parser.parse_args()
    if args.list:
        for key, label in GROUPS.items():
            print(f"{key:26} {label}")
        return
    verify_data()
    if args.check_data:
        return
    output = args.output.resolve()
    if output == ROOT or output.is_relative_to(ROOT / "data") or output.is_relative_to(ROOT / "source"):
        parser.error("Choose an output directory separate from the package inputs")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PAPER_FIGURE_OUTPUT=str(output),
               PAPER_FIGURE_STYLE=str(args.style.resolve()), PYTHONIOENCODING="utf-8")
    selected = args.only or list(GROUPS)
    failures = []
    for group in selected:
        if group == "system_overview" and args.skip_diagram:
            continue
        print(f"Rendering {group}", flush=True)
        if group == "system_overview":
            engine = shutil.which("xelatex")
            if not engine:
                failures.append("system_overview: install XeLaTeX or use --skip-diagram")
                continue
            proc = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error",
                                   "-output-directory", str(output),
                                   str(ROOT / "source/system_overview.tex")],
                                  cwd=ROOT / "source", env=env, capture_output=True)
            (output / "system_overview_build.log").write_bytes(proc.stdout + proc.stderr)
        else:
            proc = subprocess.run([sys.executable, str(ROOT / "renderers" / f"{group}.py")],
                                  cwd=output, env=env)
        if proc.returncode:
            failures.append(group)
    if failures:
        raise SystemExit("Failed groups: " + ", ".join(failures))
    print(f"Figures saved to {output}")


if __name__ == "__main__":
    main()
