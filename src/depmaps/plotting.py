"""Plotting for dependence maps (matplotlib)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Diverging blue <-> neutral grey <-> red: negative quadrant dependence is
# blue, positive is red, independence is the neutral midpoint.
DEPMAP_CMAP = LinearSegmentedColormap.from_list(
    "depmap",
    ["#123a6b", "#2a78d6", "#9fc2ec", "#f0efec", "#f1a3a2", "#e34948", "#7d1716"],
)


def _edges(p: np.ndarray) -> np.ndarray:
    """Cell edges for non-uniform levels: midpoints, clipped to [0, 1]."""
    mid = 0.5 * (p[1:] + p[:-1])
    lo = max(0.0, p[0] - (mid[0] - p[0])) if p.size > 1 else max(0.0, p[0] - 0.05)
    hi = min(1.0, p[-1] + (p[-1] - mid[-1])) if p.size > 1 else min(1.0, p[0] + 0.05)
    return np.concatenate([[lo], mid, [hi]])


def plot_map(
    dm,
    ax: Optional[plt.Axes] = None,
    *,
    which: str = "z",
    absolute: bool = False,
    style: str = "cells",
    vmax: Optional[float] = None,
    show_critical: bool = True,
    pointwise_level: Optional[float] = None,
    colorbar: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    cmap=None,
):
    """Draw a dependence map.

    X levels run along the horizontal axis and Y levels along the vertical
    axis, origin at bottom-left (lower tails meet at the origin).

    Parameters
    ----------
    which : "z" (default, standardized), "phi" (effect size, Z / sqrt(n)
        under i.i.d. scaling), "phi_rel" (phi relative to the largest value
        the margins allow, see ``DependenceMap.phi_max``) or "raw" (C_n - ab).
    absolute : plot |Z| with a sequential scale instead of the signed map.
    style : "cells" (one flat cell per grid point; no interpolation) or
        "contour" (filled contours; smoother, reads like the talk figures).
    show_critical : mark simultaneous significance (requires
        ``dm.simultaneous_band()`` to have been called). With style="cells"
        a small dot is placed on every cell whose |Z| does not exceed the
        critical value, so the marking follows the grid exactly; with
        style="contour" the |Z| = c contour is drawn instead.
    pointwise_level : optionally also outline pointwise significance at this
        level, e.g. 0.05 -> |Z| > 1.96.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    maps = {"z": dm.z, "raw": dm.raw, "phi": dm.phi, "phi_rel": dm.phi_rel}
    if which not in maps:
        raise ValueError(f"which must be one of {sorted(maps)}")
    if style == "contour" and (dm.a.size < 2 or dm.b.size < 2):
        raise ValueError("style='contour' needs at least two grid levels on each axis")
    M = maps[which].T  # rows -> Y
    a, b = dm.a, dm.b

    if absolute:
        M = np.abs(M)
        vm = vmax or float(np.nanmax(M))
        cm = cmap or LinearSegmentedColormap.from_list("depmap_abs", ["#f0efec", "#e34948", "#7d1716"])
        norm = plt.Normalize(0, vm)
    else:
        vm = vmax or float(np.nanmax(np.abs(M)))
        if which == "z" and dm.critical_value is not None:
            vm = max(vm, 1.25 * dm.critical_value)
        vm = vm if vm > 0 else 1.0
        cm = cmap or DEPMAP_CMAP
        norm = TwoSlopeNorm(vmin=-vm, vcenter=0.0, vmax=vm)

    if style == "cells":
        mesh = ax.pcolormesh(_edges(a), _edges(b), M, cmap=cm, norm=norm, shading="flat")
    elif style == "contour":
        levels = np.linspace(norm.vmin, norm.vmax, 21)
        mesh = ax.contourf(a, b, M, levels=levels, cmap=cm, norm=norm, extend="both")
    else:
        raise ValueError("style must be 'cells' or 'contour'")

    if show_critical and dm.critical_value is not None:
        Zabs = np.abs(dm.z.T)
        if style == "cells":
            jj, ii = np.nonzero(Zabs <= dm.critical_value)
            ax.scatter(a[ii], b[jj], s=3, color="#1a1a19", linewidths=0, zorder=3)
        else:
            ax.contour(a, b, Zabs, levels=[dm.critical_value], colors="#1a1a19",
                       linewidths=1.2)
    if which == "z":
        Zabs = np.abs(dm.z.T)
        if pointwise_level is not None:
            from scipy.stats import norm as _n

            ax.contour(a, b, Zabs, levels=[_n.isf(pointwise_level / 2)], colors="#1a1a19",
                       linewidths=0.8, linestyles=":")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(length=2, labelsize=8, colors="#555")
    for s in ax.spines.values():
        s.set_color("#bbb")
    ax.set_xlabel(xlabel or f"{dm.xlabel} (quantile level u)", fontsize=9)
    ax.set_ylabel(ylabel or f"{dm.ylabel} (quantile level v)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left")

    if colorbar:
        cb = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        lab = {"z": "Z(u, v)", "phi": "φ(u, v)", "phi_rel": "φ / φ_max",
               "raw": "Cₙ(u, v) − uv"}[which]
        if absolute:
            lab = f"|{lab}|"
        cb.set_label(lab, fontsize=9)
        cb.ax.tick_params(labelsize=8, length=2)
        cb.outline.set_edgecolor("#bbb")
        if which == "z" and dm.critical_value is not None:
            c = dm.critical_value
            ticks = [-c, c] if not absolute else [c]
            for t in ticks:
                cb.ax.axhline(t, color="#1a1a19", lw=1.0)
    return ax


def rank_scatter(x, y, ax=None, *, s=2, alpha=0.25, xlabel="u", ylabel="v", max_points=20000, rng=0):
    """Scatter of pseudo-observations (F_n(X), F_n(Y)): the empirical copula sample."""
    from .core import pseudo_obs

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    U, V = pseudo_obs(x[ok]), pseudo_obs(y[ok])
    if U.size > max_points:
        idx = np.random.default_rng(rng).choice(U.size, max_points, replace=False)
        U, V = U[idx], V[idx]
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.scatter(U, V, s=s, alpha=alpha, color="#2a78d6", linewidths=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(length=2, labelsize=8, colors="#555")
    for sp in ax.spines.values():
        sp.set_color("#bbb")
    return ax
