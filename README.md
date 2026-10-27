# dependence-maps

Dependence maps: a graphical tool for visualizing dependence between two scalar random variables.

A dependence map shows, on a grid of quantile levels (u, v), the signed and standardized deviation of the empirical copula from the independence copula. It indicates where in the joint distribution two variables depart from independence: around the median, in the joint upper or lower tail, or in the off-diagonal corners. The method was presented at CFE-CMStatistics 2024 ([slides](https://github.com/Brainologist/dependence-maps/blob/main/docs/CFE2024_dependence_maps_slides.pdf)).

![Dependence map of simulated hourly temperature and power price](https://raw.githubusercontent.com/Brainologist/dependence-maps/main/docs/images/example4_u_shape.png)

## Installation

```bash
pip install dependence-maps
```

The package is imported as `depmaps` (the name `depmap` was already taken on PyPI). It requires Python 3.9 or later, NumPy, SciPy and Matplotlib. pandas is optional and is only needed for `to_frame()` and the `data=` argument:

```bash
pip install "dependence-maps[pandas]"
```

For the development version, clone [the repository](https://github.com/Brainologist/dependence-maps) and run `pip install -e ".[test]"`, then `pytest`.

## Quickstart

The data below follow a Student t copula with zero correlation and three degrees of freedom. The correlation coefficient is close to zero, but the two variables are far from independent.

```python
import matplotlib.pyplot as plt
import depmaps as dpm
from depmaps.simulate import t_copula

uv = t_copula(5000, rho=0.0, df=3, rng=1)
x, y = uv[:, 0], uv[:, 1]

dm = dpm.dependence_map(x, y, grid=19)   # quantile levels 0.05, ..., 0.95
dm.simultaneous_band(alpha=0.05)          # critical value for max |Z| over the grid
print(dm.global_pvalue)                   # p-value of the test of independence

dm.plot()                                 # dots mark cells that are not significant
plt.show()
```

![Student t copula with zero correlation](https://raw.githubusercontent.com/Brainologist/dependence-maps/main/docs/images/quickstart.png)

`x` and `y` can be NumPy arrays, lists or pandas Series of equal length. Pairs with a missing value are dropped. Columns of a DataFrame can be passed by name, as in `dpm.dependence_map("load", "price", data=df)`.

## Examples from the CFE 2024 talk

[`examples/cfe2024_examples.py`](https://github.com/Brainologist/dependence-maps/blob/main/examples/cfe2024_examples.py) reproduces the examples of the talk with simulated data and writes the figures below. Slide numbers refer to the [slides](https://github.com/Brainologist/dependence-maps/blob/main/docs/CFE2024_dependence_maps_slides.pdf).

### Linear dependence (Example 1, slides 3 and 19–20)

Gaussian copula with ρ = 0.8. The raw deviation Cₙ(u, v) − uv (middle panel) is largest at the median, but only because it must vanish at the edges of the unit square. The standardized map (right) shows dependence of similar strength along the whole diagonal.

![Gaussian copula: raw and standardized maps](https://raw.githubusercontent.com/Brainologist/dependence-maps/main/docs/images/example1_gaussian.png)

### Tail dependence without correlation (Example 2, slides 4 and 21–23)

Student t copula with ρ = 0 and three degrees of freedom. Joint extremes of the same sign occur more often than under independence (red corners) and extremes of opposite sign less often (blue corners). The centre of the map shows nothing.

![Student t copula](https://raw.githubusercontent.com/Brainologist/dependence-maps/main/docs/images/example2_t_tail_dependence.png)

### Independence (slides 16–18)

Even under independence the map shows streaks, since neighbouring cells are strongly correlated. Significance is therefore judged against a simultaneous critical value, about 3.5 for a 19×19 grid, rather than the pointwise 1.96.

![Independence](https://raw.githubusercontent.com/Brainologist/dependence-maps/main/docs/images/example3_independence.png)

### U-shaped dependence in a time series (compare slides 31–33)

Simulated hourly temperature and power price, with the price rising in both cold and hot weather, much as in the temperature and day-ahead price application of the talk. The map is blue on the cold side and red on the hot side, whereas a map of absolute deviations would show both sides as a single region of strong dependence. Both series are strongly autocorrelated, so the map is computed with `serial="hac"`. This is the figure at the top of the page.

### Lower-tail dependence (compare slides 34–36)

Clayton copula with θ = 2, where dependence is strongest in the joint lower tail. The load and day-ahead price application of the talk shows a similar asymmetry.

![Clayton copula](https://raw.githubusercontent.com/Brainologist/dependence-maps/main/docs/images/example5_lower_tail.png)

## The statistic

For a sample (X₁, Y₁), …, (Xₙ, Yₙ) with empirical copula Cₙ, the map is

    Z(u, v) = √n (Cₙ(u, v) − uv) / √(u(1 − u) v(1 − v)).

When there are ties, u and v are replaced by the largest attainable marginal levels not exceeding them, which keeps the statistic exact.

The map is intuitive and easy to read since:

1. Z(u, v) is a statistic for a 2×2 table. If X is dichotomized at its u-quantile and Y at its v-quantile, then Z(u, v) = √n φ(u, v), where φ is the phi coefficient of the resulting table, and Z² is its Pearson χ² statistic. φ does not depend on the sample size and is available as `dm.phi`. Note, however, that the margins of the table bound |φ| from above. The bound equals 1 on the diagonal u = v but falls quickly away from it, and at (0.1, 0.9) φ cannot exceed 1/9 even under perfect dependence. `dm.phi_rel` = φ/φ_max corrects for this and equals ±1 everywhere under perfect positive or negative dependence.
2. Under independence and i.i.d. sampling, Z(u, v) is asymptotically N(0, 1) at every point, and the map as a whole converges to a Gaussian field with covariance ρ(u, u′)ρ(v, v′), where ρ(s, t) = (s∧t − st)/√(s(1 − s)t(1 − t)). This covariance contains no unknown quantities, so simultaneous critical values can be obtained by simulation, without resampling. Continuous margins are not required.
3. The sign carries information. Red cells indicate positive quadrant dependence at (u, v), meaning that the joint lower and joint upper orthants contain more observations than under independence, while blue cells indicate the opposite. A map of |Z| cannot distinguish the two and shows a U-shaped relationship as a single region of strong dependence.

The raw deviation Cₙ − uv is also available (`which="raw"`), but we do not recommend using it on its own. By the Fréchet bounds it tends to zero at the edges of the unit square regardless of the strength of dependence, which exaggerates dependence near the median.

## Usage

```python
import depmaps as dpm

dm = dpm.dependence_map(x, y, grid=19)          # levels 0.05, ..., 0.95
dm.simultaneous_band(alpha=0.05)                 # critical value for max |Z|
dm.plot()                                        # signed map; dots mark non-significant cells
dm.plot(which="phi_rel")                         # effect size relative to its bound
dm.to_frame()                                    # map as a table (needs pandas)
dm.global_pvalue                                 # sup-norm test of independence

# serially dependent data
dm = dpm.dependence_map("load", "price", data=df, grid=19,
                        serial="hac")               # lag by the Andrews (1991) rule
dm = dpm.dependence_map("load", "price", data=df, grid=19,
                        serial="hac", max_lag=168)  # or a fixed lag
dm.simultaneous_band(method="gaussian")          # HAC Gaussian null
dm.simultaneous_band(method="shift")             # circular shifts of Y
```

## Inference

Simultaneous critical values can be computed in four ways:

| method | null distribution | valid for |
|---|---|---|
| `gaussian` | Gaussian field with known covariance | i.i.d. data |
| `gaussian` with `serial="hac"` | Gaussian field with covariance Σ_h w_h Γ_A(h) ⊗ Γ_B(h), estimated | stationary, weakly dependent series |
| `permutation` | random permutations of Y | i.i.d. data, exact under ties |
| `shift` | random circular shifts of Y | stationary series |

The HAC Gaussian null uses a dense covariance matrix over all pairs of grid cells and is therefore limited to 2,500 cells (a 49×49 grid). For finer grids, use `shift`.

Under serial dependence, the pointwise null variance of √n(Cₙ − uv) equals Σ_h γ_A(h)γ_B(h), where A_t = 1{U_t ≤ u} − u and B_t = 1{V_t ≤ v} − v are centred indicator series. With `serial="hac"`, the ratio κ(u, v) of this long-run variance to its i.i.d. counterpart is estimated with Bartlett weights and the map reports Z/√κ. The truncation lag is chosen by the AR(1) plug-in rule of Andrews (1991) applied to A_t B_t. The Newey–West rule of thumb is available as `max_lag="nw"`, but it is much too short for persistent series such as hourly prices. For persistent and seasonal series κ can be large and may keep growing with the lag, so results should be reported for several lags.

## Tests

Running `pytest` checks both the code and the underlying theory. Among other things, the tests verify that:

- the raw deviation at the centre of the Gaussian copula equals arcsin(ρ)/2π, and Z² equals the χ² statistic of the corresponding 2×2 table;
- the statistic remains exact under ties and is standard normal under independence;
- the simultaneous band has the correct size, and Gaussian and permutation critical values agree;
- the map detects dependence that is confined to the tails;
- for AR(1) series, the HAC correction restores unit null variance and gives a band of approximately correct size;
- φ/φ_max equals ±1 under perfect dependence, and the plug-in lag increases with persistence;
- the copula samplers have uniform margins and the correct Kendall's τ;
- invalid input (grid and lag arguments, misaligned indexes) is rejected with a clear error.

## Citing

If you use the package, please cite:

> Medovikov, I. (2024). *Dependence Maps: A Graphical Tool for Visualizing Dependence.* 18th International Conference on Computational and Financial Econometrics (CFE-CMStatistics 2024), King's College London, 14–16 December 2024.

The repository includes a `CITATION.cff` file, which GitHub uses for its "Cite this repository" button.

## License

MIT; see [`LICENSE`](https://github.com/Brainologist/dependence-maps/blob/main/LICENSE).
