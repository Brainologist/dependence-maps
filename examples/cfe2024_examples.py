"""Examples from the CFE 2024 talk "Dependence Maps: A Graphical Tool for
Visualizing Dependence" (slides in docs/CFE2024_dependence_maps_slides.pdf),
reproduced with simulated data.

1. Linear dependence: Gaussian copula, rho = 0.8 (talk Example 1, slides 3, 19-20).
   Also shows why the map is standardized: the raw deviation C_n(u, v) - uv
   peaks at the median only because it must vanish at the edges.
2. Tail dependence without correlation: Student t copula, rho = 0, 3 degrees
   of freedom (talk Example 2, slides 4, 21-23).
3. Independence (talk slides 16-18): what noise looks like on a map, and why
   simultaneous rather than pointwise critical values are needed.
4. A U-shaped relationship in a persistent time series, in the spirit of the
   temperature vs day-ahead power price application (talk slides 31-33).
   Cold and hot hours both raise the price, so the signed map is blue on the
   cold side and red on the hot side. The series are strongly autocorrelated,
   so the map uses the long-run variance correction serial="hac".
5. Asymmetric tail dependence, in the spirit of the load vs power price
   application (talk slides 34-36): Clayton copula, strong in the lower tail.

All data are simulated; no external data are needed.

Run:  python cfe2024_examples.py [output directory]   (default: current directory)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import depmaps as dpm
from depmaps.simulate import clayton_copula, gaussian_copula, t_copula

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
OUT.mkdir(parents=True, exist_ok=True)
N = 5000


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def pair(x, y, title, name, xlabel="u", ylabel="v", **map_kwargs):
    """Rank scatter plot beside the signed map with a simultaneous 5% band."""
    dm = dpm.dependence_map(x, y, grid=19, **map_kwargs)
    dm.simultaneous_band(rng=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    dpm.rank_scatter(x, y, ax=ax1, xlabel=xlabel, ylabel=ylabel)
    ax1.set_title(title, fontsize=10, loc="left")
    dm.plot(ax=ax2, xlabel=xlabel, ylabel=ylabel,
            title=f"Dependence map (dots: not significant at 5%)")
    save(fig, name)
    return dm


# 1. Gaussian copula, rho = 0.8: raw vs standardized ---------------------------
uv = gaussian_copula(N, 0.8, rng=1)
dm = dpm.dependence_map(uv[:, 0], uv[:, 1], grid=19)
dm.simultaneous_band(rng=0)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
dpm.rank_scatter(uv[:, 0], uv[:, 1], ax=axes[0])
axes[0].set_title("Gaussian copula, ρ = 0.8", fontsize=10, loc="left")
dm.plot(ax=axes[1], which="raw", xlabel="u", ylabel="v", title="Raw deviation Cₙ(u, v) − uv")
dm.plot(ax=axes[2], xlabel="u", ylabel="v", title="Standardized map Z(u, v)")
save(fig, "example1_gaussian")

# 2. Student t copula, rho = 0, nu = 3 -----------------------------------------
uv = t_copula(N, 0.0, 3, rng=2)
pair(uv[:, 0], uv[:, 1], "Student t copula, ρ = 0, ν = 3", "example2_t_tail_dependence")

# 3. Independence ----------------------------------------------------------------
uv = np.random.default_rng(3).random((N, 2))
dm = pair(uv[:, 0], uv[:, 1], "Independent uniforms", "example3_independence")
print(f"  independence: max |Z| = {dm.max_abs_z:.2f} vs critical value {dm.critical_value:.2f}")

# 4. U-shaped dependence in persistent hourly series -------------------------
rng = np.random.default_rng(4)
hours = np.arange(2 * 365 * 24)
noise = np.empty(hours.size)
noise[0] = 0.0
for t in range(1, hours.size):                       # persistent weather noise
    noise[t] = 0.98 * noise[t - 1] + rng.normal(scale=1.5)
temperature = 55 - 22 * np.cos(2 * np.pi * hours / (365 * 24)) \
    - 6 * np.cos(2 * np.pi * hours / 24) + noise
cold = np.clip(52 - temperature, 0, None)             # heating demand
hot = np.clip(temperature - 62, 0, None)              # cooling demand
price_noise = np.empty(hours.size)
price_noise[0] = 0.0
for t in range(1, hours.size):
    price_noise[t] = 0.9 * price_noise[t - 1] + rng.normal(scale=4.0)
price = 35 + 0.08 * cold**2 + 0.10 * hot**2 + price_noise
dm = pair(temperature, price, "Simulated hourly temperature and power price",
          "example4_u_shape", xlabel="Temperature (u)", ylabel="Price (v)",
          serial="hac")
print(f"  U-shape: plug-in lag = {dm.max_lag} h, median variance inflation = {np.median(dm.kappa):.1f}")

# 5. Clayton copula: lower-tail dependence -------------------------------------
uv = clayton_copula(N, 2.0, rng=5)
pair(uv[:, 0], uv[:, 1], "Clayton copula, θ = 2 (lower-tail dependence)", "example5_lower_tail")
