"""Valuation and risk measures.

Method: historical simulation with FULL REVALUATION. Each historical one-day risk-factor change
is applied to today's positions; bonds are repriced (not delta-approximated). The portfolio is
held constant (no ageing/roll-down), so P&L series are *hypothetical P&L of today's book*.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

from . import config

BP = 1e-4


# ---------------------------------------------------------------- valuation
def bond_price(y: float, coupon: float, maturity: float, freq: int = 2) -> float:
    """Price per 100 face of a fixed-rate bond at a flat yield y (periodic compounding, annuity form)."""
    n = int(round(maturity * freq))
    i = y / freq
    c = 100.0 * coupon / freq
    if abs(i) < 1e-12:
        return c * n + 100.0
    v = (1.0 + i) ** -n
    return c * (1.0 - v) / i + 100.0 * v


def bond_sensitivities(notional: float, y: float, coupon: float, maturity: float, freq: int = 2) -> dict:
    """DV01 / CS01 in USD per 1bp for a position of given (signed) market value, by central bump."""
    p0 = bond_price(y, coupon, maturity, freq)
    up = bond_price(y + BP, coupon, maturity, freq)
    dn = bond_price(y - BP, coupon, maturity, freq)
    return {"price": p0, "dv01_usd": notional * (dn - up) / (2 * p0)}  # +ve = gains when yields fall


# ---------------------------------------------------------------- data access
def load_changes(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT obs_date, series_id, chg FROM factor_changes", conn)
    return df.pivot(index="obs_date", columns="series_id", values="chg").sort_index().dropna()


def load_last_levels(conn: sqlite3.Connection) -> tuple[str, dict]:
    d = conn.execute("SELECT MAX(obs_date) FROM factor_levels").fetchone()[0]
    rows = conn.execute("SELECT series_id, level FROM factor_levels WHERE obs_date=?", (d,)).fetchall()
    return d, {k: v for k, v in rows}


def load_positions(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM positions ORDER BY position_id", conn)
    # normalise SQL NULL to None: depending on the pandas version it may load as NaN (which is truthy)
    df["spread_factor_id"] = df["spread_factor_id"].astype(object).where(df["spread_factor_id"].notna(), None)
    return df


# ---------------------------------------------------------------- scenarios
def scenario_pnl(positions: pd.DataFrame, changes: pd.DataFrame, last_levels: dict) -> pd.DataFrame:
    """Per-position scenario P&L (USD), one row per historical date."""
    out = {}
    for p in positions.itertuples():
        if p.instrument == "DELTA1":
            r = changes[p.factor_id]
            r_eff = 1.0 / (1.0 + r) - 1.0 if p.invert else r
            out[p.position_id] = p.notional_usd * r_eff
        else:
            y0 = last_levels[p.factor_id] / 100.0
            dy = changes[p.factor_id]
            if p.spread_factor_id:
                y0 += last_levels[p.spread_factor_id] / 100.0
                dy = dy + changes[p.spread_factor_id]
            p0 = bond_price(y0, p.coupon, p.maturity_years, int(p.freq))
            p1 = np.array([bond_price(y0 + d, p.coupon, p.maturity_years, int(p.freq)) for d in dy.values])
            out[p.position_id] = pd.Series(p.notional_usd * (p1 / p0 - 1.0), index=changes.index)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- measures
def var_hist(pnl: np.ndarray, cl: float) -> float:
    return float(-np.quantile(pnl, 1.0 - cl))


def es_hist(pnl: np.ndarray, cl: float) -> float:
    q = np.quantile(pnl, 1.0 - cl)
    return float(-pnl[pnl <= q].mean())


def var_param(pnl: np.ndarray, cl: float) -> float:
    """Zero-mean normal VaR on the scenario P&L standard deviation."""
    return float(stats.norm.ppf(cl) * np.std(pnl, ddof=1))


def kupiec(n: int, x: int, cl: float) -> dict:
    """Kupiec proportion-of-failures likelihood-ratio test."""
    p = 1.0 - cl
    def ll(pp):
        a = (n - x) * np.log(1 - pp) if n - x else 0.0
        b = x * np.log(pp) if x else 0.0
        return a + b
    lr = -2.0 * (ll(p) - ll(x / n))
    pval = float(1.0 - stats.chi2.cdf(lr, 1))
    return {"lr": float(lr), "p_value": pval, "reject_at_5pct": bool(pval < 0.05)}


def backtest(pnl: np.ndarray, cl: float, lookback: int, days: int) -> dict:
    """Rolling out-of-sample: VaR_t from the `lookback` P&Ls before t; exception if pnl_t < -VaR_t."""
    n = len(pnl)
    days = min(days, n - lookback)
    if days <= 0:
        return {"tested_days": 0, "exceptions": 0, "expected": 0.0, "kupiec": None}
    exc = 0
    for t in range(n - days, n):
        v = var_hist(pnl[t - lookback:t], cl)
        exc += int(pnl[t] < -v)
    return {"tested_days": days, "exceptions": exc, "expected": days * (1 - cl),
            "kupiec": kupiec(days, exc, cl)}


# ---------------------------------------------------------------- report
def build_report(conn: sqlite3.Connection, cl: float = 0.99, lookback: int = config.DEFAULT_LOOKBACK,
                 backtest_days: int = config.DEFAULT_BACKTEST_DAYS) -> dict:
    changes_all = load_changes(conn)
    as_of, levels = load_last_levels(conn)
    pos = load_positions(conn)
    pnl_all = scenario_pnl(pos, changes_all, levels)
    lookback = min(lookback, len(pnl_all))
    pnl = pnl_all.iloc[-lookback:]
    port = pnl.sum(axis=1).values

    by_class = {}
    for ac, grp in pos.groupby("asset_class"):
        v = pnl[grp.position_id].sum(axis=1).values
        by_class[ac] = {"var_hist": var_hist(v, cl), "es_hist": es_hist(v, cl),
                        "exposure_usd": float(grp.notional_usd.sum())}
    standalone = sum(x["var_hist"] for x in by_class.values())
    worst_i = int(np.argmin(port))

    sens = []
    for p in pos[pos.instrument == "BOND"].itertuples():
        y0 = levels[p.factor_id] / 100.0 + (levels[p.spread_factor_id] / 100.0 if p.spread_factor_id else 0.0)
        s = bond_sensitivities(p.notional_usd, y0, p.coupon, p.maturity_years, int(p.freq))
        sens.append({"position_id": p.position_id, "asset_class": p.asset_class, "yield": y0,
                     "dv01_usd": s["dv01_usd"],
                     "cs01_usd": -s["dv01_usd"] if p.spread_factor_id else 0.0})  # +1bp spread = loss

    return {
        "as_of": as_of, "confidence": cl,
        "window": {"days": int(lookback), "start": pnl.index[0], "end": pnl.index[-1]},
        "portfolio": {
            "gross_exposure_usd": float(pos.notional_usd.abs().sum()),
            "net_exposure_usd": float(pos.notional_usd.sum()),
            "var_hist": var_hist(port, cl), "var_param": var_param(port, cl), "es_hist": es_hist(port, cl),
            "worst_day": {"date": pnl.index[worst_i], "pnl_usd": float(port[worst_i])},
            "sum_standalone_var": standalone,
            "diversification_benefit": standalone - var_hist(port, cl),
            "dv01_usd": float(sum(s["dv01_usd"] for s in sens)),
            "cs01_usd": float(sum(s["cs01_usd"] for s in sens)),
        },
        "by_asset_class": by_class,
        "sensitivities": sens,
        "backtest": backtest(pnl_all.sum(axis=1).values, cl, lookback, backtest_days),
        "pnl_distribution": [float(x) for x in port],
    }
