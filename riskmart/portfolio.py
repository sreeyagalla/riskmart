"""Sample portfolio. SYNTHETIC: positions are generated from a seeded, documented rule.

Market data in the mart is real; the holdings are not (no public dataset has a real book).
Rule: fixed instrument list; notional = base size * uniform(0.6, 1.4) drawn from numpy
default_rng(seed); bond coupons are the base coupon +/- up to 50bp (rounded to 1bp).
"""
from __future__ import annotations

import sqlite3

import numpy as np

# (id, name, asset_class, instrument, factor, spread_factor, invert, base_notional_usd, coupon, maturity, freq)
BOOK = [
    ("EQ_SPX",   "Long S&P 500 index exposure",  "Equity",    "DELTA1", "SP500",        None,           0,  25_000_000, None,  None, None),
    ("EQ_NDX",   "Long NASDAQ Comp. exposure",   "Equity",    "DELTA1", "NASDAQCOM",    None,           0,  10_000_000, None,  None, None),
    ("FX_EUR",   "Long EUR vs USD",              "FX",        "DELTA1", "DEXUSEU",      None,           0,   8_000_000, None,  None, None),
    ("FX_JPY",   "Long JPY vs USD",              "FX",        "DELTA1", "DEXJPUS",      None,           1,   6_000_000, None,  None, None),
    ("CM_BRENT", "Short Brent crude",            "Commodity", "DELTA1", "DCOILBRENTEU", None,           0,  -5_000_000, None,  None, None),
    ("UST_2Y",   "Long 2y US Treasury",          "Rates",     "BOND",   "DGS2",         None,           0,  30_000_000, 0.040, 2,    2),
    ("UST_5Y",   "Short 5y US Treasury",         "Rates",     "BOND",   "DGS5",         None,           0, -12_000_000, 0.042, 5,    2),
    ("UST_10Y",  "Long 10y US Treasury",         "Rates",     "BOND",   "DGS10",        None,           0,  20_000_000, 0.045, 10,   2),
    ("IG_10Y",   "Long 10y IG corporate bond",   "Credit",    "BOND",   "DGS10",        "BAMLC0A0CM",   0,  15_000_000, 0.050, 10,   2),
    ("HY_5Y",    "Long 5y HY corporate bond",    "Credit",    "BOND",   "DGS5",         "BAMLH0A0HYM2", 0,   7_000_000, 0.070, 5,    2),
]


def generate_positions(seed: int = 42) -> list[tuple]:
    rng = np.random.default_rng(seed)
    rows = []
    for pid, name, ac, inst, f, sf, inv, base, cpn, mat, freq in BOOK:
        notional = round(base * rng.uniform(0.6, 1.4) / 1000) * 1000
        if cpn is not None:
            cpn = round(cpn + rng.uniform(-0.005, 0.005), 4)
        rows.append((pid, name, ac, inst, f, sf, inv, float(notional), cpn, mat, freq))
    return rows


def load_positions(conn: sqlite3.Connection, seed: int = 42) -> int:
    rows = generate_positions(seed)
    with conn:
        conn.execute("DELETE FROM positions")
        conn.executemany("INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)
