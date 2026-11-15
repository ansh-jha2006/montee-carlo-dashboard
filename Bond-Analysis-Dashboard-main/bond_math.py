"""Bond math used by the dashboard.

Conventions (applied consistently by every function):
  * ``coupon_rate`` and ``ytm`` are annual rates as DECIMALS (0.05 == 5%).
  * ``years`` is time to maturity in years; the number of coupon periods is
    ``round(years * freq)``.
  * ``freq`` is coupon payments per year (2 == semiannual). YTM is a nominal
    annual rate compounded ``freq`` times per year (bond-equivalent yield).
  * Durations are in years, convexity is in years^2.

This module has no Streamlit dependency so it can be unit-tested directly.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

MIN_YTM = -0.99  # floor used to keep simulated yields in the valid pricing domain


def _validate(face_value: float, coupon_rate: float, years: float, freq: int) -> None:
    if face_value <= 0:
        raise ValueError("face_value must be positive")
    if coupon_rate < 0:
        raise ValueError("coupon_rate must be non-negative")
    if years <= 0:
        raise ValueError("years must be positive")
    if freq < 1 or int(freq) != freq:
        raise ValueError("freq must be a positive integer")


def cash_flows(face_value: float, coupon_rate: float, years: float, freq: int = 2):
    """Return ``(times_in_years, cash_flows)``; the last cash flow is coupon + face value."""
    _validate(face_value, coupon_rate, years, freq)
    n = max(1, int(round(years * freq)))
    times = np.arange(1, n + 1) / freq
    flows = np.full(n, face_value * coupon_rate / freq, dtype=float)
    flows[-1] += face_value
    return times, flows


def bond_price(face_value, coupon_rate, years, ytm, freq: int = 2):
    """Price for a given YTM. ``ytm`` may be a scalar or an array (vectorised)."""
    times, flows = cash_flows(face_value, coupon_rate, years, freq)
    ytm_arr = np.asarray(ytm, dtype=float)
    discount = (1.0 + ytm_arr[..., None] / freq) ** (freq * times)
    price = np.sum(flows / discount, axis=-1)
    return float(price) if price.ndim == 0 else price


def yield_to_maturity(face_value, coupon_rate, years, price, freq: int = 2):
    """Solve for the YTM that reprices the bond to ``price``.

    Returns ``None`` if ``price`` is not positive or no root exists.
    """
    _validate(face_value, coupon_rate, years, freq)
    if price <= 0:
        return None

    def objective(y):
        return bond_price(face_value, coupon_rate, years, y, freq) - price

    lo, hi = MIN_YTM, 1.0
    f_lo = objective(lo)
    if f_lo < 0:  # price is above anything this domain can produce
        return None
    f_hi = objective(hi)
    while f_hi > 0 and hi < 100.0:  # very deep-discount bonds need a wider bracket
        hi *= 2.0
        f_hi = objective(hi)
    if f_hi > 0:
        return None
    return float(optimize.brentq(objective, lo, hi, xtol=1e-12))


def macaulay_duration(face_value, coupon_rate, years, ytm, freq: int = 2) -> float:
    times, flows = cash_flows(face_value, coupon_rate, years, freq)
    pv = flows / (1.0 + ytm / freq) ** (freq * times)
    return float(np.sum(times * pv) / np.sum(pv))


def modified_duration(face_value, coupon_rate, years, ytm, freq: int = 2) -> float:
    return macaulay_duration(face_value, coupon_rate, years, ytm, freq) / (1.0 + ytm / freq)


def convexity(face_value, coupon_rate, years, ytm, freq: int = 2) -> float:
    times, flows = cash_flows(face_value, coupon_rate, years, freq)
    base = 1.0 + ytm / freq
    pv = flows / base ** (freq * times)
    weights = times * (times + 1.0 / freq) / base**2
    return float(np.sum(weights * pv) / np.sum(pv))


def current_yield(face_value, coupon_rate, price) -> float:
    """Annual coupon income divided by market price (decimal)."""
    if price <= 0:
        raise ValueError("price must be positive")
    return face_value * coupon_rate / price


def monte_carlo_prices(face_value, coupon_rate, years, ytm, iterations,
                       yield_vol=0.01, freq: int = 2, seed=None):
    """Simulate bond prices from normally distributed yields around ``ytm``.

    ``yield_vol`` is the standard deviation of the yield in absolute terms
    (0.01 == 1 percentage point). Simulated yields are floored at ``MIN_YTM``.
    The YTM is solved once by the caller, not once per iteration.
    """
    rng = np.random.default_rng(seed)
    ytms = np.maximum(rng.normal(ytm, yield_vol, int(iterations)), MIN_YTM)
    return bond_price(face_value, coupon_rate, years, ytms, freq)


def summarize_prices(prices) -> dict:
    prices = np.asarray(prices, dtype=float)
    return {
        "mean_price": float(np.mean(prices)),
        "std_dev": float(np.std(prices, ddof=1)) if prices.size > 1 else 0.0,
        "p5": float(np.percentile(prices, 5)),
        "p95": float(np.percentile(prices, 95)),
        "min_price": float(np.min(prices)),
        "max_price": float(np.max(prices)),
    }
