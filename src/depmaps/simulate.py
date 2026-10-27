"""Copula samplers used in examples and tests (uniform margins)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def gaussian_copula(n: int, rho: float, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = rng.standard_normal((n, 2)) @ L.T
    return stats.norm.cdf(z)


def t_copula(n: int, rho: float, df: float, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = rng.standard_normal((n, 2)) @ L.T
    w = np.sqrt(df / rng.chisquare(df, size=n))[:, None]
    return stats.t.cdf(z * w, df)


def clayton_copula(n: int, theta: float, rng=None) -> np.ndarray:
    """Clayton copula (lower-tail dependence), theta > 0, Marshall-Olkin sampler."""
    rng = np.random.default_rng(rng)
    g = rng.gamma(1.0 / theta, 1.0, size=n)
    e = rng.exponential(size=(n, 2))
    return (1.0 + e / g[:, None]) ** (-1.0 / theta)


def gaussian_copula_cdf(u, v, rho):
    """C(u, v) for the Gaussian copula."""
    x, y = stats.norm.ppf(u), stats.norm.ppf(v)
    return stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf(np.column_stack([np.ravel(x), np.ravel(y)]))
