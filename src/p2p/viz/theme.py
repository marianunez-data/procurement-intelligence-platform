"""One visual system for notebooks (matplotlib) and the app (Plotly).

Colours come from a validated colour-blind-safe categorical palette, used in a fixed order.
Status colours are reserved for KPI states and always appear with a text label.
"""
from __future__ import annotations

BRAND = {"navy": "#0d366b", "navy_light": "#184f95", "ink": "#0b0b0b", "ink_2": "#52514e", "muted": "#8a8984",
         "grid": "#e6e5e0", "surface": "#fcfcfb", "panel": "#f5f4f0"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"On target": "#0ca30c", "At risk": "#fab219", "Off target": "#d03b3b", "Monitor": "#8a8984"}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORIGIN_COLORS = {"Foreign": SERIES[0], "Local": SERIES[1], "Not Sourced": BRAND["muted"], "Unknown": "#b8b7b1"}
SHOCK_FILL = "#f0efec"
FONT = "Inter, IBM Plex Sans, Helvetica Neue, Arial, sans-serif"


def use_matplotlib():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.figsize": (11, 4.2), "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": BRAND["grid"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": BRAND["grid"], "grid.linewidth": 0.8, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10, "axes.labelcolor": BRAND["ink_2"],
        "xtick.color": BRAND["ink_2"], "ytick.color": BRAND["ink_2"], "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0, "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "IBM Plex Sans", "DejaVu Sans", "Arial"], "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def shade_shocks(ax, shocks=None, label=True, alpha=1.0):
    """Shade the dated events from the configuration on a time axis."""
    import pandas as pd

    from p2p.config import CFG
    shocks = shocks or [s for s in CFG.shocks if s.name not in ("Approval policy change", "New marine contract")]
    for s in shocks:
        ax.axvspan(pd.Timestamp(s.start), pd.Timestamp(s.end), color=SHOCK_FILL, zorder=0, alpha=alpha)
        if label:
            ax.text(pd.Timestamp(s.start), 1.0, " " + s.name, transform=ax.get_xaxis_transform(), fontsize=8,
                    color=BRAND["ink_2"], va="top", ha="left")
    for s in CFG.shocks:
        if s.name in ("Approval policy change", "New marine contract"):
            ax.axvline(pd.Timestamp(s.start), color=BRAND["muted"], lw=1, ls="--", zorder=0)
            if label:
                ax.text(pd.Timestamp(s.start), 0.92, " " + s.name, transform=ax.get_xaxis_transform(), fontsize=8,
                        color=BRAND["ink_2"], va="top", ha="left")


def plotly_template():
    import plotly.graph_objects as go
    import plotly.io as pio

    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=13, color=BRAND["ink"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
        margin=dict(l=8, r=8, t=48, b=8),
        xaxis=dict(showgrid=False, linecolor=BRAND["grid"], ticks="outside", tickcolor=BRAND["grid"], tickfont=dict(color=BRAND["ink_2"], size=12)),
        yaxis=dict(gridcolor=BRAND["grid"], zeroline=False, tickfont=dict(color=BRAND["ink_2"], size=12)),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, font=dict(size=12, color=BRAND["ink_2"])),
        hoverlabel=dict(bgcolor="white", bordercolor=BRAND["grid"], font=dict(family=FONT, size=12, color=BRAND["ink"])),
        hovermode="x unified",
        title=dict(font=dict(size=15, color=BRAND["ink"]), x=0, xanchor="left"),
    )
    pio.templates["p2p"] = t
    pio.templates.default = "p2p"
    return t
