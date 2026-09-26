"""Shared chart styling so every figure uses the same restrained palette.

Colour roles
------------
CHURN_COLOR     the one accent colour, used for churned customers / emphasis
RETAINED_COLOR  neutral grey for retained customers / context
ORDINAL_BLUES   light-to-dark blue steps for ordered categories (e.g. contract length)
SEQUENTIAL_CMAP single-hue blue ramp for heatmaps
"""
import sys

import matplotlib

if "ipykernel" not in sys.modules:  # running as a plain script: no GUI windows
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from config import FIGURE_DIR, PROJECT_ROOT

CHURN_COLOR = "#1c5cab"
RETAINED_COLOR = "#b9bcc1"
ORDINAL_BLUES = ["#86b6ef", "#2a78d6", "#104281"]
INK = "#222222"
MUTED_INK = "#5f5e5a"
GRID_COLOR = "#e4e3dd"
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)


def apply_style():
    """Set matplotlib defaults: light grid, no top/right spines, readable text."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "text.parse_math": False,   # treat "$" as a literal dollar sign, not maths markup
    })


def in_notebook():
    return "ipykernel" in sys.modules


def save_figure(fig, filename):
    """Save a figure to outputs/figures, then show it (notebook) or close it (script)."""
    path = FIGURE_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {path.relative_to(PROJECT_ROOT)}")
    if in_notebook():
        plt.show()
    else:
        plt.close(fig)
