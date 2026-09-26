#!/usr/bin/env python3
"""Render PNG-only Pareto checkpoint pages for two-direction ablations."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentopt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.plot.plot_lcb_recommendations import (  # noqa: E402
    COLORS,
    GRAY,
    _draw_panel,
    checkpoint_sequence,
)


from experiments.combined_objective.run_two_direction_ablation import (  # noqa: E402
    DEFAULT_OUTDIR as DEFAULT_RESULTS,
    PRIMARY_PAIR as DEFAULT_PAIR,
)


DEFAULT_BENCHMARKS = ("hotpotqa", "mathqa")


def render_one(result_path: Path, *, panels_per_page: int = 9) -> dict:
    payload = json.loads(result_path.read_text())
    run = payload["run"]
    config = payload["config"]
    points = checkpoint_sequence(run, terminal_label="Final")
    outdir = result_path.parent
    columns = 3
    color = COLORS["finite_lcb"]
    pages = []

    for start in range(0, len(points), panels_per_page):
        page = points[start : start + panels_per_page]
        rows = math.ceil(len(page) / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(14.4, 3.9 * rows + 1.25),
            squeeze=False,
        )
        for axis, point in zip(axes.flat, page):
            _draw_panel(axis, point, run, "finite_lcb")
        for axis in list(axes.flat)[len(page) :]:
            axis.set_visible(False)

        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=6,
                label="Recommended · completed",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                markersize=6,
                label="Recommended · partial",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=GRAY,
                markersize=4,
                label="All configurations · full-data reference",
            ),
            Line2D(
                [],
                [],
                linestyle=":",
                color="#717b84",
                label="Full-data Pareto frontier",
            ),
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.035),
            ncol=2,
            fontsize=9,
            frameon=False,
        )
        page_number = start // panels_per_page + 1
        page_count = math.ceil(len(points) / panels_per_page)
        directions = " + ".join(f"({a:g},{b:g})" for a, b in config["directions"])
        figure.suptitle(
            f"Two-direction Radial-Gittins · {config['benchmark']} · {directions}\n"
            f"finite-LCB beta=1 · seed {config['seed']} · independent questions · "
            f"page {page_number}/{page_count} ({page[0]['checkpoint']}–{page[-1]['checkpoint']})",
            fontsize=13,
            y=0.985,
        )
        figure.text(
            0.5,
            0.008,
            "Offline diagnostic coordinates use the complete common-question matrix. "
            "Labels show configuration/arm and sample count.",
            ha="center",
            va="bottom",
            fontsize=8.2,
        )
        figure.tight_layout(
            rect=(0, 0.13 if rows == 1 else 0.105 if rows == 2 else 0.085, 1, 0.92)
        )
        suffix = "" if page_number == 1 else f"_page{page_number:03d}"
        output = outdir / f"pareto_key_checkpoints{suffix}.png"
        figure.savefig(output, dpi=170, facecolor="white")
        plt.close(figure)
        pages.append(
            {
                "page": page_number,
                "file": output.name,
                "checkpoints": [point["checkpoint"] for point in page],
            }
        )

    manifest = {
        "benchmark": config["benchmark"],
        "pair_name": config["pair_name"],
        "directions": config["directions"],
        "membership_change_count": sum(
            point.get("snapshot_role") == "membership_change" for point in run["points"]
        ),
        "panel_count": len(points),
        "page_count": len(pages),
        "pages": pages,
    }
    (outdir / "pareto_key_checkpoints_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Rendered {config['benchmark']}: {len(points)} panels on {len(pages)} PNG pages",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument(
        "--benchmarks", nargs="+", choices=DEFAULT_BENCHMARKS, default=list(DEFAULT_BENCHMARKS)
    )
    args = parser.parse_args()
    manifests = []
    for benchmark in args.benchmarks:
        result_path = args.results / args.pair / benchmark / "result.json"
        manifests.append(render_one(result_path))
    (args.results / args.pair / "pareto_plot_manifest.json").write_text(
        json.dumps(manifests, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
