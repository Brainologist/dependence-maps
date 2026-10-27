"""Simultaneous inference for dependence maps.

All methods calibrate the distribution of max_{(u,v) in grid} |Z(u, v)| under
the null of independence between X and Y, giving a critical value c_alpha
such that, under the null, P(max |Z| > c_alpha) ~= alpha. Cells with
|Z| > c_alpha are significant with family-wise error rate alpha.

Methods
-------
gaussian     Simulate the limiting Gaussian field. For i.i.d. data its
             covariance is the Kronecker product of normalized Brownian-bridge
             kernels in the attained levels (no unknowns). With a HAC-corrected
             map (serial="hac") the covariance is
             sum_h w_h Gamma_A(h) (x) Gamma_B(h), estimated from the
             indicator series, then normalized to unit diagonal.
permutation  Randomly permute Y relative to X. Exact for i.i.d. data and
             handles ties; invalid for serially dependent data.
shift        Random circular shifts of Y relative to X (toroidal shifts).
             Preserves each series' own serial dependence and breaks the
             cross-dependence; the natural resampling null for stationary
             time series.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .core import DependenceMap, bartlett_weights


MAX_HAC_CELLS = 2500


def _bridge_corr(p: np.ndarray) -> np.ndarray:
    s = np.sqrt(p * (1 - p))
    return (np.minimum.outer(p, p) - np.outer(p, p)) / np.outer(s, s)


def _psd_factor(S: np.ndarray) -> np.ndarray:
    """Symmetric square-root factor of a (nearly) PSD matrix."""
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 0.0, None)
    return V * np.sqrt(w)


def _lagged_cov(Mc: np.ndarray, h: int) -> np.ndarray:
    n = Mc.shape[0]
    if h == 0:
        return Mc.T @ Mc / n
    return Mc[:-h].T @ Mc[h:] / n


def null_covariance(dm: DependenceMap) -> np.ndarray:
    """Null correlation matrix of vec(Z) (row-major over the grid)."""
    if dm.kappa is None:
        return np.kron(_bridge_corr(dm.a), _bridge_corr(dm.b))
    cells = dm.a.size * dm.b.size
    if cells > MAX_HAC_CELLS:
        raise ValueError(
            f"the HAC Gaussian null needs a dense {cells}x{cells} covariance matrix; "
            f"use a grid with at most {MAX_HAC_CELLS} cells (e.g. 49x49) or method='shift'"
        )
    if dm._A is None or dm._B is None:
        raise ValueError("map was computed with keep_indicators=False")
    Ac = dm._A - dm.a
    Bc = dm._B - dm.b
    S = np.kron(_lagged_cov(Ac, 0), _lagged_cov(Bc, 0))
    for h, w in zip(range(1, dm.max_lag + 1), bartlett_weights(dm.max_lag)):
        GA, GB = _lagged_cov(Ac, h), _lagged_cov(Bc, h)
        S += w * (np.kron(GA, GB) + np.kron(GA.T, GB.T))
    d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
    return S / np.outer(d, d)


def _max_abs_z_from_B(dm: DependenceMap, Bp: np.ndarray) -> float:
    n = dm.n
    raw = (dm._A.T @ Bp) / n - np.outer(dm.a, dm.b)
    z = np.sqrt(n) * raw / np.sqrt(np.outer(dm.a * (1 - dm.a), dm.b * (1 - dm.b)))
    if dm.kappa is not None:
        z = z / np.sqrt(dm.kappa)
    return float(np.max(np.abs(z)))


def null_max_distribution(
    dm: DependenceMap,
    method: str = "gaussian",
    n_sim: int = 2000,
    rng: Optional[np.random.Generator] = None,
    min_shift: Optional[int] = None,
) -> np.ndarray:
    """Draws of max |Z| over the grid under the null of independence."""
    rng = np.random.default_rng(rng)
    ga, gb = dm.a.size, dm.b.size

    if method == "gaussian":
        if dm.kappa is None:
            La, Lb = _psd_factor(_bridge_corr(dm.a)), _psd_factor(_bridge_corr(dm.b))
            out = np.empty(n_sim)
            chunk = 500
            for s in range(0, n_sim, chunk):
                k = min(chunk, n_sim - s)
                E = rng.standard_normal((k, La.shape[1], Lb.shape[1]))
                Z = La @ E @ Lb.T
                out[s : s + k] = np.abs(Z).reshape(k, -1).max(axis=1)
            return out
        F = _psd_factor(null_covariance(dm))
        out = np.empty(n_sim)
        chunk = 500
        for s in range(0, n_sim, chunk):
            k = min(chunk, n_sim - s)
            Z = rng.standard_normal((k, F.shape[1])) @ F.T
            out[s : s + k] = np.abs(Z).max(axis=1)
        return out

    if dm._A is None or dm._B is None:
        raise ValueError("map was computed with keep_indicators=False")

    if method == "permutation":
        if dm.kappa is not None:
            warnings.warn(
                "permutation destroys serial dependence in Y; use method='shift' "
                "or 'gaussian' for time series",
                stacklevel=2,
            )
        return np.array(
            [_max_abs_z_from_B(dm, dm._B[rng.permutation(dm.n)]) for _ in range(n_sim)]
        )

    if method == "shift":
        n = dm.n
        m = n // 10 if min_shift is None else int(min_shift)
        if not 0 < m < n // 2:
            raise ValueError("min_shift must be in (0, n/2)")
        shifts = rng.integers(m, n - m + 1, size=n_sim)
        return np.array([_max_abs_z_from_B(dm, np.roll(dm._B, s, axis=0)) for s in shifts])

    raise ValueError("method must be 'gaussian', 'permutation' or 'shift'")


def simultaneous_band(
    dm: DependenceMap,
    alpha: float = 0.05,
    method: str = "gaussian",
    n_sim: int = 2000,
    rng=None,
    min_shift: Optional[int] = None,
) -> float:
    """Simultaneous critical value for |Z| over the grid; stored on ``dm``.

    Also stores ``dm.global_pvalue``: the share of null draws with
    max |Z| >= the observed max |Z| (a test of independence based on the
    sup-norm of the standardized empirical copula process over the grid).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    draws = null_max_distribution(dm, method=method, n_sim=n_sim, rng=rng, min_shift=min_shift)
    c = float(np.quantile(draws, 1 - alpha))
    dm.critical_value = c
    dm.critical_method = method
    dm.alpha = alpha
    dm.global_pvalue = float((1 + np.sum(draws >= dm.max_abs_z)) / (1 + draws.size))
    return c
