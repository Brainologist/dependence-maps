import numpy as np
import pytest
from scipy import stats

import depmaps as dpm
from depmaps.simulate import gaussian_copula, t_copula


def test_pseudo_obs_ties_use_max_rank():
    x = np.array([3.0, 1.0, 3.0, 2.0])
    np.testing.assert_allclose(dpm.pseudo_obs(x), [1.0, 0.25, 1.0, 0.5])


def test_grid_default():
    np.testing.assert_allclose(dpm.make_grid(19), np.arange(1, 20) / 20)


def test_gaussian_center_matches_arcsin_formula():
    # C(1/2, 1/2) - 1/4 = arcsin(rho) / (2 pi) for the Gaussian copula
    uv = gaussian_copula(200_000, 0.8, rng=1)
    dm = dpm.dependence_map(uv[:, 0], uv[:, 1], grid=[0.5])
    assert dm.raw[0, 0] == pytest.approx(np.arcsin(0.8) / (2 * np.pi), abs=3e-3)


def test_raw_matches_gaussian_copula_off_center():
    from depmaps.simulate import gaussian_copula_cdf

    uv = gaussian_copula(200_000, 0.5, rng=2)
    dm = dpm.dependence_map(uv[:, 0], uv[:, 1], grid=[0.2, 0.7])
    for i, u in enumerate([0.2, 0.7]):
        for j, v in enumerate([0.2, 0.7]):
            target = float(np.atleast_1d(gaussian_copula_cdf(u, v, 0.5))[0]) - u * v
            assert dm.raw[i, j] == pytest.approx(target, abs=3e-3)


def test_z_squared_is_2x2_chi_square():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(500)
    y = 0.3 * x + rng.standard_normal(500)
    dm = dpm.dependence_map(x, y, grid=[0.3, 0.6])
    qx, qy = np.quantile(x, 0.3, method="inverted_cdf"), np.quantile(y, 0.6, method="inverted_cdf")
    tab = np.array(
        [[np.sum((x <= qx) & (y <= qy)), np.sum((x <= qx) & (y > qy))],
         [np.sum((x > qx) & (y <= qy)), np.sum((x > qx) & (y > qy))]]
    )
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    assert dm.z[0, 1] ** 2 == pytest.approx(chi2, rel=1e-10)


def test_ties_are_exact_and_duplicates_dropped():
    rng = np.random.default_rng(4)
    x = rng.integers(0, 4, 1000).astype(float)  # 4 distinct values
    y = x + rng.integers(0, 3, 1000)
    dm = dpm.dependence_map(x, y, grid=19)
    assert dm.a.size <= 3  # at most 3 interior attainable levels
    assert np.all(np.isfinite(dm.z))
    # Z^2 equals chi-square of the dichotomized table at attained levels
    cut = np.sort(np.unique(x))[np.searchsorted(np.cumsum(np.bincount(x.astype(int))) / 1000, dm.a[0])]
    ycut_level = dm.b[0]
    V = dpm.pseudo_obs(y)
    tab = np.array(
        [[np.sum((x <= cut) & (V <= ycut_level + 1e-12)), np.sum((x <= cut) & (V > ycut_level + 1e-12))],
         [np.sum((x > cut) & (V <= ycut_level + 1e-12)), np.sum((x > cut) & (V > ycut_level + 1e-12))]]
    )
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    assert dm.z[0, 0] ** 2 == pytest.approx(chi2, rel=1e-10)


def test_invariant_to_monotone_transforms():
    uv = t_copula(3000, 0.3, 4, rng=5)
    d1 = dpm.dependence_map(uv[:, 0], uv[:, 1])
    d2 = dpm.dependence_map(np.exp(5 * uv[:, 0]), -1 / uv[:, 1] ** 3)
    np.testing.assert_allclose(d1.z, d2.z)


def test_sign_flips_under_decreasing_transform():
    uv = gaussian_copula(3000, 0.6, rng=6)
    d1 = dpm.dependence_map(uv[:, 0], uv[:, 1], grid=[0.5])
    d2 = dpm.dependence_map(uv[:, 0], -uv[:, 1], grid=[0.5])
    assert d1.z[0, 0] > 0 and d2.z[0, 0] < 0
    assert abs(d1.z[0, 0] + d2.z[0, 0]) < 0.2


def test_null_pointwise_standard_normal():
    rng = np.random.default_rng(7)
    zs = np.array(
        [dpm.dependence_map(rng.random(1000), rng.random(1000), grid=[0.1, 0.5]).z.ravel()
         for _ in range(1500)]
    )
    assert np.abs(zs.mean(axis=0)).max() < 0.1
    assert np.all(np.abs(zs.var(axis=0) - 1) < 0.12)


def test_gaussian_and_permutation_critical_values_agree():
    rng = np.random.default_rng(8)
    dm = dpm.dependence_map(rng.random(3000), rng.random(3000), grid=9)
    cg = dpm.simultaneous_band(dm, method="gaussian", n_sim=4000, rng=1)
    cp = dpm.simultaneous_band(dm, method="permutation", n_sim=1500, rng=2)
    assert cg == pytest.approx(cp, rel=0.06)
    assert 2.5 < cg < 3.6  # well above the pointwise 1.96


def test_simultaneous_band_size_iid():
    rng = np.random.default_rng(9)
    dm0 = dpm.dependence_map(rng.random(800), rng.random(800), grid=9)
    c = dpm.simultaneous_band(dm0, alpha=0.05, n_sim=5000, rng=3)
    rej = np.mean([dpm.dependence_map(rng.random(800), rng.random(800), grid=9).max_abs_z > c
                   for _ in range(600)])
    assert 0.02 < rej < 0.09


def test_power_against_tail_dependence_only():
    # t copula with rho=0: zero correlation, yet strong corner deviations
    uv = t_copula(5000, 0.0, 3, rng=10)
    dm = dpm.dependence_map(uv[:, 0], uv[:, 1], grid=9)
    c = dm.simultaneous_band(n_sim=3000, rng=4)
    sig = dm.significant()
    assert sig[0, 0] and sig[-1, -1]  # lower-left and upper-right
    assert dm.z[0, -1] < 0 and dm.z[-1, 0] < 0  # off-diagonal corners negative
    assert abs(dm.z[4, 4]) < c  # nothing at the median


def _ar1(n, phi, rng):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_hac_kappa_near_one_for_iid():
    rng = np.random.default_rng(11)
    dm = dpm.dependence_map(rng.random(5000), rng.random(5000), grid=9, serial="hac", max_lag=20)
    assert np.abs(dm.kappa - 1).max() < 0.15


def test_hac_restores_null_variance_for_ar1():
    rng = np.random.default_rng(12)
    zi, za = [], []
    for _ in range(300):
        x, y = _ar1(2000, 0.9, rng), _ar1(2000, 0.9, rng)
        dm = dpm.dependence_map(x, y, grid=[0.5], serial="hac", max_lag=60)
        zi.append(dm.z_iid[0, 0])
        za.append(dm.z[0, 0])
    assert np.var(zi) > 3.0          # i.i.d. standardization badly oversized
    assert 0.7 < np.var(za) < 1.35   # HAC standardization ~ N(0,1)


def test_shift_band_size_for_ar1():
    rng = np.random.default_rng(13)
    rej = 0
    reps = 120
    for _ in range(reps):
        x, y = _ar1(1500, 0.8, rng), _ar1(1500, 0.8, rng)
        dm = dpm.dependence_map(x, y, grid=5)
        dpm.simultaneous_band(dm, method="shift", n_sim=300, rng=rng)
        rej += dm.global_pvalue <= 0.05
    assert rej / reps < 0.12


def test_hac_gaussian_band_runs_and_is_reasonable():
    rng = np.random.default_rng(14)
    x, y = _ar1(3000, 0.8, rng), _ar1(3000, 0.8, rng)
    dm = dpm.dependence_map(x, y, grid=7, serial="hac", max_lag=40)
    c = dpm.simultaneous_band(dm, method="gaussian", n_sim=3000, rng=5)
    assert 2.0 < c < 4.0


def test_missing_values_dropped_and_dataframe_input():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(15)
    df = pd.DataFrame({"load": rng.random(500), "lmp": rng.random(500)})
    df.iloc[::7, 0] = np.nan
    dm = dpm.dependence_map("load", "lmp", data=df, grid=5)
    assert dm.n == df.dropna().shape[0]
    assert dm.xlabel == "load"
    assert dm.to_frame().shape == (5, 5)


def test_plot_smoke():
    import matplotlib
    matplotlib.use("Agg")
    uv = gaussian_copula(2000, 0.5, rng=16)
    dm = dpm.dependence_map(uv[:, 0], uv[:, 1])
    dm.simultaneous_band(n_sim=500, rng=1)
    dm.plot(pointwise_level=0.05)
    dm.plot(style="contour", absolute=True)


def test_phi_is_z_over_sqrt_n():
    uv = gaussian_copula(1000, 0.4, rng=17)
    dm = dpm.dependence_map(uv[:, 0], uv[:, 1], grid=5)
    np.testing.assert_allclose(dm.phi, dm.z_iid / np.sqrt(dm.n))
    assert np.all(np.abs(dm.phi) <= 1)


# --- added in 0.1.1 ---------------------------------------------------------

def test_phi_rel_is_one_under_comonotonicity_and_minus_one_under_countermonotonicity():
    x = np.random.default_rng(18).random(2000)
    up = dpm.dependence_map(x, x ** 3, grid=9)
    down = dpm.dependence_map(x, -x, grid=9)
    np.testing.assert_allclose(up.phi_rel, 1.0, atol=1e-9)
    np.testing.assert_allclose(down.phi_rel, -1.0, atol=1e-9)
    assert up.phi[0, -1] < 0.15  # raw phi fades off the diagonal


def test_andrews_lag_grows_with_persistence():
    rng = np.random.default_rng(19)
    lags = []
    for phi in (0.0, 0.5, 0.95):
        x, y = _ar1(5000, phi, rng), _ar1(5000, phi, rng)
        lags.append(dpm.dependence_map(x, y, grid=5, serial="hac").max_lag)
    assert lags[0] < lags[1] < lags[2]
    assert lags[2] > 30


def test_hac_gaussian_band_size_ar1():
    rng = np.random.default_rng(20)
    rej = []
    for r in range(120):
        x, y = _ar1(1500, 0.8, rng), _ar1(1500, 0.8, rng)
        dm = dpm.dependence_map(x, y, grid=5, serial="hac", max_lag=40)
        dm.simultaneous_band(n_sim=800, rng=r)
        rej.append(dm.global_pvalue <= 0.05)
    assert np.mean(rej) < 0.13


def test_warns_on_gaps_with_serial_correction():
    rng = np.random.default_rng(21)
    x, y = rng.random(500), rng.random(500)
    x[10] = np.nan
    with pytest.warns(UserWarning, match="contiguous"):
        dpm.dependence_map(x, y, grid=5, serial="hac", max_lag=5)


def test_rejects_misaligned_series_and_bad_lag():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(22)
    x = pd.Series(rng.random(100), index=range(100))
    y = pd.Series(rng.random(100), index=range(1, 101))
    with pytest.raises(ValueError, match="indexes"):
        dpm.dependence_map(x, y)
    with pytest.raises(ValueError, match="max_lag"):
        dpm.dependence_map(rng.random(50), rng.random(50), grid=3, serial="hac", max_lag=60)


def test_resampling_needs_indicators():
    rng = np.random.default_rng(23)
    dm = dpm.dependence_map(rng.random(200), rng.random(200), grid=5, keep_indicators=False)
    with pytest.raises(ValueError, match="keep_indicators"):
        dm.simultaneous_band(method="permutation", n_sim=10)
    dm.simultaneous_band(method="gaussian", n_sim=100)  # needs no indicators


def test_copula_samplers_have_uniform_margins_and_right_dependence():
    from depmaps.simulate import clayton_copula

    uv = clayton_copula(20000, 2.0, rng=24)
    assert stats.kstest(uv[:, 0], "uniform").pvalue > 0.01
    tau = stats.kendalltau(uv[:5000, 0], uv[:5000, 1])[0]
    assert tau == pytest.approx(2.0 / (2.0 + 2.0), abs=0.03)  # Clayton tau = theta/(theta+2)
    uv = t_copula(20000, 0.5, 4, rng=25)
    assert stats.kstest(uv[:, 1], "uniform").pvalue > 0.01
    tau = stats.kendalltau(uv[:5000, 0], uv[:5000, 1])[0]
    assert tau == pytest.approx(2 / np.pi * np.arcsin(0.5), abs=0.03)


# --- added in second review ---------------------------------------------------

def test_plain_lists_and_bad_arguments():
    rng = np.random.default_rng(26)
    x, y = list(rng.random(200)), list(rng.random(200))
    assert dpm.dependence_map(x, y, grid=5).n == 200
    for bad in (19.7, True, (0.25, 0.75)):
        with pytest.raises(ValueError, match="grid"):
            dpm.dependence_map(x, y, grid=bad)
    with pytest.raises(ValueError, match="y is required"):
        dpm.dependence_map(x)
    with pytest.raises(ValueError, match="max_lag"):
        dpm.dependence_map(x, y, grid=3, serial="hac", max_lag="foo")
    with pytest.warns(UserWarning, match="ignored"):
        dpm.dependence_map(x, y, grid=3, max_lag=10)
    dm = dpm.dependence_map(x, y, grid=5)
    with pytest.raises(ValueError, match="alpha"):
        dm.simultaneous_band(alpha=0)
    with pytest.raises(ValueError, match="which"):
        dm.plot(which="Z")
    with pytest.raises(ValueError, match="two grid levels"):
        dpm.dependence_map(x, y, grid=[0.5]).plot(style="contour")


def test_hac_gaussian_refuses_huge_grids():
    rng = np.random.default_rng(27)
    dm = dpm.dependence_map(rng.random(3000), rng.random(3000), grid=60, serial="hac", max_lag=5)
    with pytest.raises(ValueError, match="shift"):
        dm.simultaneous_band(method="gaussian", n_sim=10)
