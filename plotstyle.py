#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from cycler import cycler

# Ink and chrome
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

# Series colours: teal for models and fits, plum for SPmini haloes, purple and grey for galaxies
TEAL, PLUM = "#88BBDA", "#F1749C"
GAS_RICH, GAS_POOR = "#8E70BA", "#978575"
SERIES = [TEAL, PLUM, "#8695E3", "#C08A1E", "#4E7D5B", "#A8443A", GAS_RICH, GAS_POOR]

# Blue <-> red around a neutral grey, for signed quantities
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#184f95", "#2a78d6", "#b3b1aa", "#e34948", "#a8302f"])

def ramp(color, n=3):
    # n steps of one hue, from the colour itself towards white
    rgb = to_rgb(color)
    return [tuple(c + (1 - c) * t for c in rgb) for t in [0.8 * i / max(n - 1, 1) for i in range(n)]]

def use():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 11,
        "text.color": INK,

        # Opaque background, so the figures read on a dark editor theme too
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "figure.constrained_layout.use": True,

        # Recessive axes: no box, hairline grid behind the data
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.prop_cycle": cycler(color=SERIES),
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,

        # Marks
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "scatter.edgecolors": "none",
        "errorbar.capsize": 0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.labelcolor": INK_2,
        "image.cmap": "Blues",
    })

    # Sharp figures on high-resolution screens (Jupyter kernels only)
    try:
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats("retina")
    except ImportError:
        pass
