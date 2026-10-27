"""Quickstart: a dependence map for two variables with zero correlation.

The data follow a Student t copula with zero correlation and three degrees
of freedom (Example 2 of the CFE 2024 talk, slides 4 and 21-23). The
correlation coefficient is close to zero, yet the variables are clearly
dependent in all four corners of the unit square.

Run:  python quickstart.py   (writes quickstart.png to the current directory)
"""
import matplotlib.pyplot as plt
import numpy as np

import depmaps as dpm
from depmaps.simulate import t_copula

uv = t_copula(5000, rho=0.0, df=3, rng=1)
x, y = uv[:, 0], uv[:, 1]
print(f"Pearson correlation: {np.corrcoef(x, y)[0, 1]:.3f}")

dm = dpm.dependence_map(x, y, grid=19)
dm.simultaneous_band(alpha=0.05, rng=1)
print(f"max |Z| = {dm.max_abs_z:.2f}, 5% simultaneous critical value = {dm.critical_value:.2f}, "
      f"p-value of the independence test = {dm.global_pvalue:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
dpm.rank_scatter(x, y, ax=ax1)
ax1.set_title("Ranks (empirical copula sample)", fontsize=10, loc="left")
dm.plot(ax=ax2, xlabel="u", ylabel="v", title="Dependence map")
fig.tight_layout()
fig.savefig("quickstart.png", dpi=150)
