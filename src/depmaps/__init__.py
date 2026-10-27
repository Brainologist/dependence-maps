"""Dependence maps: signed standardized deviations of the empirical copula from independence."""
from .core import DependenceMap, dependence_map, make_grid, pseudo_obs, default_max_lag, andrews_max_lag
from .inference import null_covariance, null_max_distribution, simultaneous_band
from .plotting import DEPMAP_CMAP, plot_map, rank_scatter

__all__ = [
    "DependenceMap",
    "dependence_map",
    "make_grid",
    "pseudo_obs",
    "default_max_lag",
    "andrews_max_lag",
    "simultaneous_band",
    "null_max_distribution",
    "null_covariance",
    "plot_map",
    "rank_scatter",
    "DEPMAP_CMAP",
]

__version__ = "0.1.1"
