"""Same grounding check, run against the REAL mart built by `python -m riskmart etl`.
Skipped when the mart has not been built (or was built from the test fixture)."""
import sqlite3

import pytest
from riskmart import config, qa
from independent import compute

pytestmark = pytest.mark.skipif(not config.DEFAULT_DB.exists(), reason="run `python -m riskmart etl` first")


def test_real_data_answers_match_independent_computation():
    conn = sqlite3.connect(config.DEFAULT_DB)
    if "FRED" not in conn.execute("SELECT source FROM etl_runs ORDER BY run_id DESC LIMIT 1").fetchone()[0]:
        pytest.skip("mart not built from FRED")
    ind = compute(config.DEFAULT_DB, 0.99)
    a = qa.answer_question(conn, "What is the 99% VaR of the portfolio?")
    assert next(f.value for f in a.figures if f.label == "var_hist") == pytest.approx(ind["var_hist"], rel=1e-9)
    a = qa.answer_question(conn, "What is the 99% expected shortfall?")
    assert next(f.value for f in a.figures if f.label == "es_hist") == pytest.approx(ind["es_hist"], rel=1e-9)
    n_days = conn.execute("SELECT COUNT(DISTINCT obs_date) FROM factor_levels").fetchone()[0]
    assert n_days > 700
