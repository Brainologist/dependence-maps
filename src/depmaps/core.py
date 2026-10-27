"""Core computation of dependence maps.

A dependence map evaluates, on a grid of quantile levels (u, v), the signed
standardized deviation of the empirical copula from the independence copula

    Z(u, v) = sqrt(n) * (C_n(a, b) - a b) / sqrt(a (1 - a) b (1 - b)),

where a = a(u) and b = b(v) are the marginal proportions actually attained
when X is cut at its empirical u-quantile and Y at its empirical v-quantile.
Without ties a = floor(n u) / n -> u; with ties a is the largest attainable
level not exceeding u, which keeps the statistic exact. The Gaussian limit
below does not require continuous margins.

Equivalently, Z(u, v) = sqrt(n) * phi(u, v), where phi is the phi coefficient
of the 2x2 contingency table obtained by dichotomizing X at its u-quantile and
Y at its v-quantile, and Z(u, v)^2 is that table's Pearson chi-square
statistic. Under independence (i.i.d. sampling) Z(u, v) is asymptotically
N(0, 1) pointwise, and the field {Z(u, v)} converges to a centred Gaussian
field with covariance

    rho_a(u, u') * rho_b(v, v'),   rho(s, t) = (s ^ t - s t) / sqrt(s(1-s) t(1-t)),

i.e. a normalized tied-down Brownian pillow.

For serially dependent observations the pointwise null variance of
sqrt(n)(C_n - ab) is sum_h gamma_A(h) gamma_B(h), where A_t, B_t are the
centred indicator series. ``serial="hac"`` estimates the ratio of this
long-run variance to the i.i.d. variance with Bartlett weights and rescales Z.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np

GridSpec = Union[int, Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def pseudo_obs(x: np.ndarray) -> np.ndarray:
    """Empirical CDF values F_n(x_i) = #{j : x_j <= x_i} / n ("max" ranks / n).

    Ties receive the same value, the largest rank in their group, so that
    {F_n(X_i) <= u} is exactly the event {X_i <= empirical u-quantile}.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    order = np.sort(x)
    return np.searchsorted(order, x, side="right") / n


def make_grid(grid: GridSpec) -> np.ndarray:
    """Quantile levels strictly inside (0, 1).

    An integer g gives the equispaced interior grid (1, ..., g) / (g + 1).
    """
    if np.isscalar(grid):
        if isinstance(grid, (bool, np.bool_)) or not float(grid).is_integer():
            raise ValueError(
                f"grid={grid!r}: a scalar grid must be a whole number of levels; "
                "pass quantile levels as a list or array, e.g. grid=[0.25, 0.75]"
            )
        g = int(grid)
        if g < 1:
            raise ValueError("grid size must be >= 1")
        return np.arange(1, g + 1) / (g + 1)
    levels = np.sort(np.asarray(grid, dtype=float).ravel())
    if np.any(levels <= 0) or np.any(levels >= 1):
        raise ValueError("grid levels must lie strictly inside (0, 1)")
    return levels


def _indicators(F: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """n x g matrix of 1{F_n(X_i) <= level_k} as float64."""
    return (F[:, None] <= levels[None, :] + 1e-12).astype(float)


def _acf_columns(M: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelations of each (centred) column of M for lags 1..max_lag.

    Returns an array of shape (max_lag, g). Uses FFT; columns with zero
    variance get zero autocorrelation.
    """
    n, g = M.shape
    Mc = M - M.mean(axis=0, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    f = np.fft.rfft(Mc, n=nfft, axis=0)
    ac = np.fft.irfft(f * np.conj(f), n=nfft, axis=0)[: max_lag + 1]
    var = ac[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(var > 0, ac[1:] / var, 0.0)
    return r


def default_max_lag(n: int) -> int:
    """Newey-West (1994) style rule of thumb, floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def andrews_max_lag(A: np.ndarray, B: np.ndarray) -> int:
    """Andrews (1991) AR(1) plug-in truncation lag for the Bartlett kernel.

    The long-run variance of interest is that of the products A_t B_t, whose
    first-order autocorrelation under independence is rho_A * rho_B. The
    plug-in uses the median of these products over the grid:
    L = 1.1447 (alpha n)^(1/3), alpha = 4 rho^2 / (1 - rho^2)^2.
    """
    n = A.shape[0]
    rA = _acf_columns(A, 1)[0]
    rB = _acf_columns(B, 1)[0]
    rho = float(np.median(np.abs(np.outer(rA, rB))))
    rho = min(rho, 0.99)
    alpha = 4.0 * rho**2 / (1.0 - rho**2) ** 2
    L = int(np.ceil(1.1447 * (alpha * n) ** (1.0 / 3.0)))
    return max(1, min(L, n // 4))


def bartlett_weights(max_lag: int) -> np.ndarray:
    h = np.arange(1, max_lag + 1)
    return 1.0 - h / (max_lag + 1.0)


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------

@dataclass
class DependenceMap:
    """Result of :func:`dependence_map`.

    Attributes
    ----------
    u, v : nominal quantile levels kept on each axis (after dropping levels
        that collapse onto the same attainable proportion because of ties).
    a, b : attained marginal proportions at those levels (equal u, v up to
        O(1/n) without ties). Maps are plotted against ``a`` and ``b``.
    n : number of complete observations used.
    raw : signed raw deviation C_n(a, b) - a b, shape (len(a), len(b)).
        Rows index X levels, columns index Y levels.
    z_iid : sqrt(n) * raw / sqrt(a(1-a) b(1-b)), the i.i.d. standardization.
    kappa : long-run variance inflation factor (None unless serial="hac").
    z : the map to read; z_iid / sqrt(kappa) if serial correction is on.
    """

    u: np.ndarray
    v: np.ndarray
    a: np.ndarray
    b: np.ndarray
    n: int
    raw: np.ndarray
    z_iid: np.ndarray
    kappa: Optional[np.ndarray] = None
    max_lag: Optional[int] = None
    xlabel: str = "X"
    ylabel: str = "Y"
    # kept for inference (indicator matrices); not part of the public repr
    _A: Optional[np.ndarray] = field(default=None, repr=False)
    _B: Optional[np.ndarray] = field(default=None, repr=False)
    critical_value: Optional[float] = None
    critical_method: Optional[str] = None
    alpha: Optional[float] = None
    global_pvalue: Optional[float] = None

    @property
    def z(self) -> np.ndarray:
        if self.kappa is None:
            return self.z_iid
        return self.z_iid / np.sqrt(self.kappa)

    @property
    def phi(self) -> np.ndarray:
        """Effect-size map: the phi coefficient of each dichotomized 2x2 table.

        phi = raw / sqrt(a(1-a) b(1-b)) = z_iid / sqrt(n). Comparable across
        samples of different size, in [-1, 1].
        """
        return self.raw / np.sqrt(np.outer(self.a * (1 - self.a), self.b * (1 - self.b)))

    @property
    def phi_max(self) -> np.ndarray:
        """Largest |phi| attainable given the margins, for the sign of each cell.

        For phi >= 0 the bound is sqrt(min(a,b)(1-max(a,b)) / (max(a,b)(1-min(a,b))));
        for phi < 0 it is sqrt(ab / ((1-a)(1-b))) when a + b <= 1 and
        sqrt((1-a)(1-b) / (ab)) otherwise. Off the diagonal the bound is well
        below one, so phi maps fade away from the diagonal even under perfect
        dependence.
        """
        A, B = np.meshgrid(self.a, self.b, indexing="ij")
        lo, hi = np.minimum(A, B), np.maximum(A, B)
        pos = np.sqrt(lo * (1 - hi) / (hi * (1 - lo)))
        neg = np.where(A + B <= 1, np.sqrt(A * B / ((1 - A) * (1 - B))),
                       np.sqrt((1 - A) * (1 - B) / (A * B)))
        return np.where(self.raw >= 0, pos, neg)

    @property
    def phi_rel(self) -> np.ndarray:
        """phi / phi_max: phi relative to its attainable bound, in [-1, 1].

        Equals +1 everywhere under comonotonicity and -1 under
        countermonotonicity, regardless of (u, v).
        """
        return self.phi / self.phi_max

    @property
    def pvalues(self) -> np.ndarray:
        """Pointwise two-sided p-values from the N(0, 1) approximation."""
        from scipy.stats import norm

        return 2.0 * norm.sf(np.abs(self.z))

    @property
    def max_abs_z(self) -> float:
        return float(np.nanmax(np.abs(self.z)))

    def significant(self) -> np.ndarray:
        """Boolean mask of cells exceeding the simultaneous critical value."""
        if self.critical_value is None:
            raise ValueError("call simultaneous_band(...) first")
        return np.abs(self.z) > self.critical_value

    def to_frame(self, which: str = "z"):
        """Return the map as a pandas DataFrame (rows: X levels, cols: Y levels)."""
        import pandas as pd

        M = {"z": self.z, "z_iid": self.z_iid, "raw": self.raw, "phi": self.phi,
             "phi_rel": self.phi_rel}[which]
        return pd.DataFrame(
            M,
            index=pd.Index(np.round(self.a, 4), name=f"{self.xlabel} level"),
            columns=pd.Index(np.round(self.b, 4), name=f"{self.ylabel} level"),
        )

    def plot(self, **kwargs):
        from .plotting import plot_map

        return plot_map(self, **kwargs)

    def simultaneous_band(self, **kwargs) -> float:
        from .inference import simultaneous_band

        return simultaneous_band(self, **kwargs)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def _as_array_and_label(obj, default):
    name = getattr(obj, "name", None)
    return np.asarray(obj, dtype=float).ravel(), (str(name) if name is not None else default)


def dependence_map(
    x,
    y=None,
    grid: Union[GridSpec, Tuple[GridSpec, GridSpec]] = 19,
    *,
    data=None,
    serial: Optional[str] = None,
    max_lag: Union[int, str, None] = "auto",
    keep_indicators: bool = True,
) -> DependenceMap:
    """Compute a signed, standardized dependence map.

    Parameters
    ----------
    x, y : array-like of equal length, or column names when ``data`` is given.
        Pairs with a missing value in either coordinate are dropped.
    grid : int, list/array of levels, or a tuple ``(x_grid, y_grid)`` of these.
        Default 19 gives levels 0.05, 0.10, ..., 0.95.
    data : optional DataFrame from which ``x`` and ``y`` are taken.
    serial : None (i.i.d. standardization) or "hac" (Bartlett long-run
        variance correction for serially dependent data).
    max_lag : truncation lag for ``serial="hac"``. "auto" (default) uses the
        Andrews (1991) AR(1) plug-in rule (:func:`andrews_max_lag`); "nw"
        uses the Newey-West rule of thumb (:func:`default_max_lag`), which is
        far too short for persistent data such as hourly prices; an integer
        sets the lag directly. Because the long-run variance of seasonal
        series may not settle as the lag grows, results should be reported
        for several lags.
    keep_indicators : store indicator matrices, needed by
        :func:`depmaps.simultaneous_band` and the permutation/shift methods.
    """
    if data is not None:
        xlab, ylab = str(x), str(y)
        x, y = data[x], data[y]
    if y is None:
        raise ValueError("y is required")
    ix, iy = getattr(x, "index", None), getattr(y, "index", None)
    if hasattr(ix, "equals") and hasattr(iy, "equals") and not ix.equals(iy):
        raise ValueError("x and y have different indexes; align them before calling")
    xv, xlab_ = _as_array_and_label(x, "X")
    yv, ylab_ = _as_array_and_label(y, "Y")
    if data is None:
        xlab, ylab = xlab_, ylab_
    if xv.shape != yv.shape:
        raise ValueError("x and y must have the same length")

    ok = np.isfinite(xv) & np.isfinite(yv)
    n_dropped = int((~ok).sum())
    if serial is not None and n_dropped:
        import warnings

        warnings.warn(
            f"{n_dropped} incomplete observations dropped; the remaining series is "
            "treated as contiguous when estimating serial dependence",
            stacklevel=2,
        )
    xv, yv = xv[ok], yv[ok]
    n = xv.shape[0]
    if n < 10:
        raise ValueError("need at least 10 complete observations")

    if isinstance(grid, tuple):
        if len(grid) != 2:
            raise ValueError("a tuple grid must be (x_grid, y_grid); pass levels as a list or array")
        gx, gy = make_grid(grid[0]), make_grid(grid[1])
    else:
        gx = gy = make_grid(grid)

    U, V = pseudo_obs(xv), pseudo_obs(yv)
    A, B = _indicators(U, gx), _indicators(V, gy)
    a, b = A.mean(axis=0), B.mean(axis=0)

    # drop levels that are degenerate (a in {0, 1}) or duplicate because of ties
    def _keep(p):
        valid = (p > 0) & (p < 1)
        idx = np.flatnonzero(valid)
        _, first = np.unique(np.round(p[idx], 12), return_index=True)
        return np.sort(idx[first])

    kx, ky = _keep(a), _keep(b)
    if kx.size == 0 or ky.size == 0:
        raise ValueError("no non-degenerate grid levels (variable is constant?)")
    gx, gy, A, B, a, b = gx[kx], gy[ky], A[:, kx], B[:, ky], a[kx], b[ky]

    p11 = (A.T @ B) / n
    raw = p11 - np.outer(a, b)
    scale = np.sqrt(np.outer(a * (1 - a), b * (1 - b)))
    z_iid = np.sqrt(n) * raw / scale

    kappa = None
    L = None
    if serial is None and max_lag not in (None, "auto"):
        import warnings

        warnings.warn("max_lag is ignored unless serial='hac'", stacklevel=2)
    if serial is not None:
        if serial != "hac":
            raise ValueError("serial must be None or 'hac'")
        if max_lag is None or max_lag == "auto":
            L = andrews_max_lag(A, B)
        elif max_lag == "nw":
            L = default_max_lag(n)
        elif isinstance(max_lag, (int, np.integer)) and not isinstance(max_lag, bool):
            L = int(max_lag)
            if L < 0:
                raise ValueError("max_lag must be non-negative")
        else:
            raise ValueError("max_lag must be 'auto', 'nw' or a non-negative integer")
        if L >= n:
            raise ValueError("max_lag must be smaller than the number of observations")
        if L >= 1:
            w = bartlett_weights(L)
            RA = _acf_columns(A, L)  # (L, gx)
            RB = _acf_columns(B, L)  # (L, gy)
            kappa = 1.0 + 2.0 * (RA * w[:, None]).T @ RB
            kappa = np.maximum(kappa, 1e-3)
        else:
            kappa = np.ones_like(z_iid)

    return DependenceMap(
        u=gx, v=gy, a=a, b=b, n=n, raw=raw, z_iid=z_iid, kappa=kappa,
        max_lag=L, xlabel=xlab, ylabel=ylab,
        _A=A if keep_indicators else None, _B=B if keep_indicators else None,
    )
