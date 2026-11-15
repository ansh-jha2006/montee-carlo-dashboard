import numpy as np
import pytest

import bond_math as bm

FV, C, T, F = 1000.0, 0.05, 10, 2


def test_par_bond_ytm_equals_coupon():
    assert bm.yield_to_maturity(FV, C, T, 1000.0, F) == pytest.approx(C, abs=1e-9)


def test_price_ytm_round_trip():
    ytm = bm.yield_to_maturity(FV, C, T, 950.0, F)
    assert ytm == pytest.approx(0.0566, abs=2e-4)
    assert bm.bond_price(FV, C, T, ytm, F) == pytest.approx(950.0, abs=1e-6)


@pytest.mark.parametrize("freq", [1, 2, 4, 12])
@pytest.mark.parametrize("price", [600.0, 950.0, 1000.0, 1200.0])
def test_round_trip_across_conventions(freq, price):
    ytm = bm.yield_to_maturity(FV, C, T, price, freq)
    assert bm.bond_price(FV, C, T, ytm, freq) == pytest.approx(price, abs=1e-6)


def test_zero_coupon_duration_is_maturity():
    assert bm.macaulay_duration(FV, 0.0, 7, 0.04, F) == pytest.approx(7.0)


def test_par_bond_known_duration():
    # Closed form: (1+y/f)/y * (1 - (1+y/f)^-n) with y=.05, f=2, n=20
    expected = (1.025 / 0.05) * (1 - 1.025 ** -20)
    assert bm.macaulay_duration(FV, C, T, 0.05, F) == pytest.approx(expected, rel=1e-9)
    assert bm.modified_duration(FV, C, T, 0.05, F) == pytest.approx(expected / 1.025, rel=1e-9)


def test_duration_and_convexity_match_finite_differences():
    y, h = 0.0566, 1e-4
    p = bm.bond_price(FV, C, T, y, F)
    up, dn = bm.bond_price(FV, C, T, y + h, F), bm.bond_price(FV, C, T, y - h, F)
    assert bm.modified_duration(FV, C, T, y, F) == pytest.approx(-(up - dn) / (2 * h * p), rel=1e-5)
    assert bm.convexity(FV, C, T, y, F) == pytest.approx((up + dn - 2 * p) / (p * h**2), rel=1e-4)


def test_vectorised_price_matches_scalar():
    ys = np.array([0.01, 0.05, 0.09])
    vec = bm.bond_price(FV, C, T, ys, F)
    assert vec.shape == (3,)
    for v, y in zip(vec, ys):
        assert v == pytest.approx(bm.bond_price(FV, C, T, y, F))


def test_zero_yield_does_not_crash():
    assert bm.bond_price(FV, C, T, 0.0, F) == pytest.approx(FV + FV * C * T)


def test_unsolvable_prices_return_none():
    assert bm.yield_to_maturity(FV, C, T, 1e12, F) is None
    assert bm.yield_to_maturity(FV, C, T, 0.0, F) is None


def test_deep_discount_needs_wide_bracket():
    ytm = bm.yield_to_maturity(FV, C, T, 1.0, F)
    assert ytm is not None and ytm > 1.0
    assert bm.bond_price(FV, C, T, ytm, F) == pytest.approx(1.0, abs=1e-6)


def test_current_yield():
    assert bm.current_yield(FV, C, 950.0) == pytest.approx(50 / 950)


def test_monte_carlo_is_reproducible_and_centered():
    ytm = bm.yield_to_maturity(FV, C, T, 950.0, F)
    a = bm.monte_carlo_prices(FV, C, T, ytm, 20000, 0.01, F, seed=1)
    b = bm.monte_carlo_prices(FV, C, T, ytm, 20000, 0.01, F, seed=1)
    assert np.array_equal(a, b)
    stats = bm.summarize_prices(a)
    # Convexity makes the mean slightly above the market price, never far from it.
    assert 950 < stats["mean_price"] < 960
    assert stats["p5"] < stats["mean_price"] < stats["p95"]
    assert stats["min_price"] <= stats["p5"] and stats["p95"] <= stats["max_price"]


def test_monte_carlo_extreme_vol_stays_finite():
    p = bm.monte_carlo_prices(FV, C, T, 0.05, 5000, 5.0, F, seed=0)
    assert np.all(np.isfinite(p))


@pytest.mark.parametrize("args", [(0, C, T), (FV, -0.01, T), (FV, C, 0)])
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        bm.cash_flows(*args)
