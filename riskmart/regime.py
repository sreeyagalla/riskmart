"""ML component: volatility-regime model and regime-conditional VaR.

Idea: plain historical VaR treats all past days as equally relevant. Here a 2-component Gaussian
mixture (unsupervised, scikit-learn) is fitted to log 20-day realised volatility of the portfolio's
hypothetical P&L. Days are labelled calm / stressed. Regime-conditional VaR is then the historical
VaR of the scenario days that FOLLOWED a day in the same regime as today.

Everything is fitted using only data before the day being forecast (no look-ahead), and the same
rolling backtest used for plain VaR is run for both so they can be compared honestly. The model
may or may not beat the baseline; the numbers are reported as they come out.
"""
from __future__ import annotations

import os
import sqlite3

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")   # silence joblib core-detection warning on Windows

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from . import config, risk

RV_WINDOW = 20
MIN_SAMPLE = 40          # below this the conditional sample is too thin -> fall back to unconditional


def _features(pnl: np.ndarray) -> np.ndarray:
    """log realised vol per day (NaN for the first RV_WINDOW-1 days)."""
    rv = pd.Series(pnl).rolling(RV_WINDOW).std().to_numpy()
    return np.log(rv)


def _fit_labels(x: np.ndarray):
    """Fit 2-state GMM on finite features. Returns (labels with 0=calm,1=stressed, p_stressed, gmm)."""
    ok = np.isfinite(x)
    gmm = GaussianMixture(2, random_state=0, n_init=1).fit(x[ok].reshape(-1, 1))
    stressed = int(np.argmax(gmm.means_.ravel()))
    post = np.full(len(x), np.nan)
    post[ok] = gmm.predict_proba(x[ok].reshape(-1, 1))[:, stressed]
    labels = np.where(np.isfinite(post), (post >= 0.5).astype(int), -1)
    return labels, post, gmm


def regime_var(window_pnl: np.ndarray, cl: float) -> dict:
    """Regime-conditional VaR for the day AFTER the window, using only the window's data."""
    x = _features(window_pnl)
    labels, post, gmm = _fit_labels(x)
    today = labels[-1]
    nxt = window_pnl[1:]                       # pnl on day j+1 ...
    prev = labels[:-1]                         # ... paired with regime of day j
    sample = nxt[prev == today]
    used_fallback = len(sample) < MIN_SAMPLE
    if used_fallback:
        sample = window_pnl
    return {"var": risk.var_hist(sample, cl), "regime": int(today), "p_stressed": float(post[-1]),
            "sample_size": int(len(sample)), "fallback": bool(used_fallback)}


def backtest_regime(pnl: np.ndarray, cl: float, lookback: int, days: int) -> dict:
    n = len(pnl)
    days = min(days, n - lookback)
    if days <= 0:
        return {"tested_days": 0}
    exc_r = exc_p = fb = 0
    for t in range(n - days, n):
        w = pnl[t - lookback:t]
        r = regime_var(w, cl)
        fb += r["fallback"]
        exc_r += int(pnl[t] < -r["var"])
        exc_p += int(pnl[t] < -risk.var_hist(w, cl))
    return {"tested_days": days, "expected": days * (1 - cl), "fallback_days": fb,
            "plain": {"exceptions": exc_p, "kupiec": risk.kupiec(days, exc_p, cl)},
            "regime": {"exceptions": exc_r, "kupiec": risk.kupiec(days, exc_r, cl)}}


def build_regime_report(conn: sqlite3.Connection, cl: float = 0.99,
                        lookback: int = config.DEFAULT_LOOKBACK,
                        backtest_days: int = config.DEFAULT_BACKTEST_DAYS,
                        with_backtest: bool = False) -> dict:
    changes = risk.load_changes(conn)
    as_of, levels = risk.load_last_levels(conn)
    pos = risk.load_positions(conn)
    pnl_df = risk.scenario_pnl(pos, changes, levels)
    pnl = pnl_df.sum(axis=1).to_numpy()
    lookback = min(lookback, len(pnl))
    w = pnl[-lookback:]
    r = regime_var(w, cl)
    x = _features(w)
    labels, _, gmm = _fit_labels(x)
    rv = np.exp(gmm.means_.ravel())
    return {
        "as_of": as_of, "confidence": cl, "window_days": int(lookback),
        "current": {"regime": "stressed" if r["regime"] == 1 else "calm",
                    "p_stressed": r["p_stressed"],
                    "recent_20d_pnl_vol_usd": float(np.exp(x[-1]))},
        "regime_means_vol_usd": {"calm": float(rv.min()), "stressed": float(rv.max())},
        "share_of_days_stressed": float((labels[labels >= 0] == 1).mean()),
        "var_regime": r["var"], "var_plain": risk.var_hist(w, cl),
        "conditional_sample_size": r["sample_size"], "used_fallback": r["fallback"],
        "backtest": backtest_regime(pnl, cl, lookback, backtest_days) if with_backtest else None,
        "regime_by_day": [{"date": d, "stressed": int(l)} for d, l in zip(pnl_df.index[-lookback:], labels)],
    }
