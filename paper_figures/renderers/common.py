"""Shared paths, typography, and export validation for the paper figures."""
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(os.environ.get("PAPER_FIGURE_OUTPUT", ROOT / "rendered")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
STYLE = json.loads(Path(os.environ.get("PAPER_FIGURE_STYLE", ROOT / "style.json")).read_text(encoding="utf-8"))
for key in ("font_scale", "line_width_scale"):
    if not isinstance(STYLE[key], (int, float)) or not math.isfinite(STYLE[key]) or STYLE[key] <= 0:
        raise ValueError(f"{key} must be a finite positive number")
if STYLE.get("dpi") is not None and (not isinstance(STYLE["dpi"], (int, float))
                                    or not math.isfinite(STYLE["dpi"]) or STYLE["dpi"] <= 0):
    raise ValueError("dpi must be null or a finite positive number")
FONT = STYLE["font_family"]
FONT_PATH = findfont(FontProperties(family=FONT), fallback_to_default=False)
matplotlib.rcParams.update({"font.family": FONT, "mathtext.fontset": STYLE["math_fontset"],
                            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def data_signature(fig):
    """Record numeric artists and axis limits; styling must not change them."""
    digest = hashlib.sha256()
    for ax in fig.axes:
        digest.update(repr((ax.get_xlim(), ax.get_ylim())).encode())
        for line in ax.lines:
            digest.update(np.asarray(line.get_xydata()).tobytes())
        for collection in ax.collections:
            digest.update(np.asarray(collection.get_offsets()).tobytes())
            for path in collection.get_paths():
                digest.update(path.vertices.tobytes())
    return digest.hexdigest()


def apply_style(fig):
    if getattr(fig, "_paper_style_applied", False):
        return
    before = data_signature(fig)
    fig.canvas.draw()  # Materialize automatically generated tick labels.
    for text in fig.findobj(match=Text):
        prop = text.get_fontproperties().copy()
        prop.set_file(None)
        prop.set_family(FONT)
        prop.set_size(prop.get_size_in_points() * STYLE["font_scale"])
        text.set_fontproperties(prop)
        text.set_math_fontfamily(STYLE["math_fontset"])
    for ax in fig.axes:
        for line in ax.lines:
            line.set_linewidth(line.get_linewidth() * STYLE["line_width_scale"])
    assert before == data_signature(fig), "Typography changed plotted values or axis limits"
    fig._paper_style_applied = True


_savefig = Figure.savefig


def savefig(fig, filename, *args, **kwargs):
    apply_style(fig)
    if STYLE.get("dpi") is not None:
        kwargs["dpi"] = STYLE["dpi"]
    if isinstance(filename, (str, Path)):
        path = Path(filename).resolve()
        if not path.is_relative_to(OUT):
            raise ValueError(f"Figure output must be inside {OUT}: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    result = _savefig(fig, filename, *args, **kwargs)
    if isinstance(filename, (str, Path)):
        entry = {"file": str(Path(filename).relative_to(OUT)), "font": FONT,
                 "font_path": FONT_PATH, "data_signature": data_signature(fig)}
        with (OUT / "font_audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
    return result


Figure.savefig = savefig
