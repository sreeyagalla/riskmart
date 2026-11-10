"""Data-quality checks, run on every ETL load and persisted to dq_results.

Checks flag; they never silently modify data (the only transformation is the bounded
forward-fill, which is itself measured and reported here).
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from . import config


def _robust_outliers(chg: pd.Series) -> int:
    med = chg.median()
    mad = (chg - med).abs().median()
    if mad == 0:
        return 0
    z = 0.6745 * (chg - med).abs() / mad
    return int((z > config.OUTLIER_MADS).sum())


def run_checks(conn: sqlite3.Connection, run_id: int, parsed: dict[str, pd.DataFrame],
               panel: pd.DataFrame) -> list[dict]:
    res: list[dict] = []

    def add(name, sid, status, value, detail):
        res.append(dict(run_id=run_id, check_name=name, series_id=sid, status=status,
                        value=None if value is None else float(value), detail=detail))

    kind = {r[0]: r[1] for r in conn.execute("SELECT series_id, kind FROM series_meta")}
    for sid, df in parsed.items():
        # 1. reconciliation: distinct source dates == rows landed in the mart
        src = df["obs_date"].nunique()
        loaded = conn.execute("SELECT COUNT(*) FROM raw_observations WHERE series_id=?", (sid,)).fetchone()[0]
        add("row_reconciliation", sid, "pass" if src == loaded else "fail", loaded,
            f"source dates={src}, loaded={loaded}")
        # 2. duplicate dates in the source file
        dups = int(df["obs_date"].duplicated().sum())
        add("duplicate_dates", sid, "pass" if dups == 0 else "fail", dups, f"{dups} duplicate dates in source")
        # 3. missing-value markers
        miss = int(df["value"].isna().sum())
        frac = miss / max(len(df), 1)
        add("missing_values", sid, "pass" if frac < 0.05 else "warn", miss,
            f"{miss} missing observations ({frac:.1%} of source rows; holidays/non-trading days)")
        # 4. non-positive prices
        if kind[sid] == "price":
            bad = int((df["value"] <= 0).sum())
            add("non_positive_price", sid, "pass" if bad == 0 else "fail", bad, f"{bad} observations <= 0")
        # 5. outliers in the changes actually used by the risk engine
        ch = conn.execute("SELECT chg FROM factor_changes WHERE series_id=?", (sid,)).fetchall()
        s = pd.Series([c[0] for c in ch], dtype=float)
        n_out = _robust_outliers(s) if len(s) else 0
        add("outlier_changes", sid, "pass" if n_out == 0 else "warn", n_out,
            f"{n_out} one-day changes beyond {config.OUTLIER_MADS:g} robust sigmas (flagged, kept in data)")
        # 6. forward-fill usage
        p = panel[panel.series_id == sid]
        nf = int(p["filled"].sum())
        add("forward_filled", sid, "pass" if nf / max(len(p), 1) < 0.02 else "warn", nf,
            f"{nf} of {len(p)} panel points forward-filled (limit {config.FFILL_LIMIT} days)")
        # 7. staleness relative to the panel's last date (deterministic, not wall-clock)
        last_src = df.dropna(subset=["value"])["obs_date"].max()
        last_panel = panel["obs_date"].max()
        lag = (pd.Timestamp(last_panel) - pd.Timestamp(last_src)).days
        add("staleness", sid, "pass" if lag <= 7 else "warn", lag,
            f"last source obs {last_src} vs panel end {last_panel} ({lag} days)")

    n_days = panel["obs_date"].nunique()
    first_dates = next(iter(parsed.values()))["obs_date"].nunique()
    add("panel_dates", None, "pass", n_days,
        f"{n_days} aligned dates, {panel['obs_date'].min()} to {panel['obs_date'].max()} "
        f"(first series has {first_dates} source dates; the rest precede the shortest series' start or have unfillable gaps)")
    need = config.DEFAULT_LOOKBACK + config.DEFAULT_BACKTEST_DAYS + 1
    add("history_depth", None, "pass" if n_days >= need else "warn", n_days,
        f"{n_days} dates available; {need} needed for default {config.DEFAULT_LOOKBACK}d lookback + "
        f"{config.DEFAULT_BACKTEST_DAYS}d backtest")
    with conn:
        conn.executemany("INSERT INTO dq_results VALUES (:run_id,:check_name,:series_id,:status,:value,:detail)", res)
    return res
