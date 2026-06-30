"""Publication-quality matplotlib defaults and small helpers.

Call set_paper_style() at the top of each notebook. savefig() writes a
vector PDF (for LaTeX) and a PNG (for quick viewing) into figures/.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "figures"

# colour-blind-safe, consistent model colours
PALETTE = {"CBM": "#0072B2", "MCBM": "#D55E00", "GT": "#444444",
           "visible": "#009E73", "occluded": "#CC79A7"}


def set_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.figsize": (6.0, 4.0),
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def savefig(name: str, fig=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = fig or plt.gcf()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    return FIG_DIR / f"{name}.pdf"
