"""Unified style for paper figures: x ticks sit on the data points (every tick is drawn)
and are rotated, with no font-size reduction.

- On log axes avoid "non-uniform" tick sets such as [1, 5, 20, 100] and use the data
  points themselves instead; the spacing then matches the sampling (1-2-5 ladder) and
  ticks never stop at 100 while the data reach 200.
- Discrete categories (network hidden-layer names) are all listed, rotated by 60° to
  prevent overlap.
"""
import numpy as np
from matplotlib.ticker import NullLocator

FS_TICK = 7.0
X_ROT = 60.0


def xticks_data(ax, xs, fmt="%g", labels=None, rot=X_ROT, fs=FS_TICK):
    """Place the x ticks on the data points (works for linear and log axes)."""
    vals = np.unique(np.asarray(xs, dtype=float))
    ax.set_xticks(vals)
    if labels is None:
        labels = [fmt % v for v in vals]
    ax.set_xticklabels(list(labels), rotation=rot, ha="right",
                       rotation_mode="anchor", fontsize=fs)
    ax.xaxis.set_minor_locator(NullLocator())


def xticks_names(ax, names, rot=X_ROT, fs=FS_TICK):
    """Discrete categories (hidden-layer names): label every tick and rotate."""
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(list(names), rotation=rot, ha="right",
                       rotation_mode="anchor", fontsize=fs)


def xticks_fixed(ax, vals, labels=None, rot=0.0, fs=FS_TICK):
    """Numeric axis: draw only a few "tidy" ticks (unrotated) instead of one per data point."""
    vals = [float(v) for v in vals]
    ax.set_xticks(vals)
    if labels is None:
        labels = [f"{v:g}" for v in vals]
    ax.set_xticklabels(list(labels), rotation=rot, ha="center" if not rot else "right",
                       rotation_mode="anchor" if rot else "default", fontsize=fs)
    ax.xaxis.set_minor_locator(NullLocator())
