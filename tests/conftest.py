"""Test fixtures. The fixture market data is SYNTHETIC (seeded random walks written in FRED CSV
format) so tests are offline and deterministic; the real-data run is in test_real_data.py."""
import numpy as np
import pandas as pd
import pytest

from riskmart import config
from riskmart.etl import run_etl

N_DAYS = 900


def make_csvs(seed=7, n=N_DAYS, gap_rows=(100, 300, 301)):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n).strftime("%Y-%m-%d")
    spec = {"SP500": ("p", 4000, .010), "NASDAQCOM": ("p", 12000, .013), "DEXUSEU": ("p", 1.1, .004),
            "DEXJPUS": ("p", 110, .004), "DCOILBRENTEU": ("p", 70, .02), "DGS2": ("r", 3.0, .05),
            "DGS5": ("r", 3.3, .05), "DGS10": ("r", 3.6, .05), "BAMLC0A0CM": ("r", 1.2, .02),
            "BAMLH0A0HYM2": ("r", 4.0, .08)}
    out = {}
    for sid, (kind, start, vol) in spec.items():
        if kind == "p":
            lv = start * np.exp(np.cumsum(rng.normal(0, vol, n)))
        else:
            lv = np.maximum(start + np.cumsum(rng.normal(0, vol, n)), 0.3)
        vals = [f"{x:.4f}" for x in lv]
        if sid != "SP500":                       # holiday-style missing markers on non-equity series
            for g in gap_rows:
                vals[g] = "."
        out[sid] = "observation_date," + sid + "\n" + "\n".join(f"{d},{v}" for d, v in zip(dates, vals)) + "\n"
    return out


@pytest.fixture(scope="session")
def csvs():
    return make_csvs()


@pytest.fixture(scope="session")
def db_path(tmp_path_factory, csvs):
    p = tmp_path_factory.mktemp("mart") / "test.db"
    run_etl(p, fetcher=lambda sid: csvs[sid], source="synthetic test fixture")
    return p


@pytest.fixture()
def conn(db_path):
    import sqlite3
    c = sqlite3.connect(str(db_path))
    yield c
    c.close()
