import numpy as np
import pytest

from riskmart import risk


def pv_cashflows(y, c, T, f=2):
    """Independent pricer: explicit sum of discounted cash flows."""
    n = int(round(T * f))
    pv = sum(100 * c / f / (1 + y / f) ** k for k in range(1, n + 1))
    return pv + 100 / (1 + y / f) ** n


@pytest.mark.parametrize("y,c,T", [(0.04, 0.05, 10), (0.08, 0.07, 5), (0.01, 0.02, 2), (0.0, 0.03, 3)])
def test_bond_price_matches_cashflow_sum(y, c, T):
    assert risk.bond_price(y, c, T) == pytest.approx(pv_cashflows(y, c, T), rel=1e-12)


def test_par_bond_prices_at_100():
    assert risk.bond_price(0.045, 0.045, 7) == pytest.approx(100.0)


def test_dv01_sign_and_size():
    s = risk.bond_sensitivities(1_000_000, 0.045, 0.045, 10)
    assert 700 < s["dv01_usd"] < 900            # ~8 year modified duration * $100 per bp per $1m
    assert risk.bond_sensitivities(-1_000_000, 0.045, 0.045, 10)["dv01_usd"] == pytest.approx(-s["dv01_usd"])


def test_var_es_known_values():
    pnl = np.arange(-100, 100, dtype=float)      # uniform, 200 points
    assert risk.var_hist(pnl, 0.99) == pytest.approx(-np.quantile(pnl, 0.01))
    assert risk.es_hist(pnl, 0.99) >= risk.var_hist(pnl, 0.99)
    assert risk.var_param(np.array([-1.0, 1.0] * 50), 0.99) == pytest.approx(2.3263478740408408 * np.std([-1.0, 1.0] * 50, ddof=1))


def test_kupiec_properties():
    assert risk.kupiec(250, 2, 0.99)["lr"] < 1.0                 # near expected (2.5) -> fits
    assert risk.kupiec(250, 12, 0.99)["reject_at_5pct"] is True  # far too many exceptions
    assert risk.kupiec(250, 0, 0.99)["lr"] > 0                   # zero exceptions handled (no log(0))


def test_backtest_counts_exceptions_exactly():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, 800)
    b = risk.backtest(pnl, 0.99, 500, 250)
    manual = sum(pnl[t] < -(-np.quantile(pnl[t - 500:t], 0.01)) for t in range(550, 800))
    assert b["exceptions"] == manual and b["tested_days"] == 250
    assert risk.backtest(pnl, 0.99, 500, 1000)["tested_days"] == 300     # clipped to available history


def test_report_internal_consistency(conn):
    r = risk.build_report(conn, 0.95)
    assert r["portfolio"]["es_hist"] >= r["portfolio"]["var_hist"] > 0
    assert r["portfolio"]["diversification_benefit"] == pytest.approx(
        sum(v["var_hist"] for v in r["by_asset_class"].values()) - r["portfolio"]["var_hist"])
    assert len(r["pnl_distribution"]) == r["window"]["days"]
    assert r["portfolio"]["cs01_usd"] < 0           # long credit -> loses when spreads widen
