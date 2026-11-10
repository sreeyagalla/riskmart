import numpy as np
import pytest
from fastapi.testclient import TestClient

from riskmart import qa, regime, risk
from riskmart.api import create_app
from test_grounding import assert_all_numbers_traced, fig


def two_regime_series(seed=1, n=600):
    """Synthetic P&L with a known calm/stressed structure (calm sd 1, stressed sd 5, ~100-day persistence)."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n, int)
    for i in range(1, n):
        state[i] = state[i - 1] if rng.random() < 0.99 else 1 - state[i - 1]
    return rng.normal(0, np.where(state == 1, 5.0, 1.0)), state


def test_gmm_recovers_known_regimes():
    pnl, state = two_regime_series()
    x = regime._features(pnl)
    labels, post, _ = regime._fit_labels(x)
    ok = labels >= 0
    agree = (labels[ok] == state[ok]).mean()
    assert agree > 0.85          # unsupervised labelling should mostly recover the planted states
    assert labels[~ok].tolist() == [-1] * (regime.RV_WINDOW - 1)


def test_stressed_regime_var_exceeds_calm_regime_var():
    pnl, state = two_regime_series()
    stressed_window, calm_window = None, None
    for end in range(300, 600):                         # find days ending in each regime
        r = regime.regime_var(pnl[end - 300:end], 0.95)
        if r["regime"] == 1 and stressed_window is None and r["p_stressed"] > .95: stressed_window = r["var"]
        if r["regime"] == 0 and calm_window is None and r["p_stressed"] < .05: calm_window = r["var"]
    assert stressed_window and calm_window and stressed_window > 1.5 * calm_window


def test_regime_var_equals_hand_computed_conditional_quantile(conn):
    g = regime.build_regime_report(conn, 0.95, lookback=500)
    ch, (_, lv), pos = risk.load_changes(conn), risk.load_last_levels(conn), risk.load_positions(conn)
    pnl = risk.scenario_pnl(pos, ch, lv).sum(axis=1).to_numpy()[-500:]
    lab = np.array([-1 if d["stressed"] < 0 else d["stressed"] for d in g["regime_by_day"]])
    sample = sorted(pnl[1:][lab[:-1] == lab[-1]])
    pos_ = 0.05 * (len(sample) - 1)
    lo, hi = int(np.floor(pos_)), int(np.ceil(pos_))
    expected = -(sample[lo] + (sample[hi] - sample[lo]) * (pos_ - lo))
    assert g["var_regime"] == pytest.approx(expected, rel=1e-9)
    assert g["conditional_sample_size"] == len(sample)


def test_backtest_comparison_counts_both_models_on_same_days(conn):
    g = regime.build_regime_report(conn, 0.95, lookback=500, backtest_days=60, with_backtest=True)
    b = g["backtest"]
    assert b["tested_days"] == 60 and b["plain"]["kupiec"] and b["regime"]["kupiec"]
    assert 0 <= b["regime"]["exceptions"] <= 60


def test_regime_answer_is_grounded(conn):
    a = qa.answer_question(conn, "What volatility regime are we in?")
    g = regime.build_regime_report(conn, 0.99)
    assert a.supported and fig(a, "var_regime") == pytest.approx(g["var_regime"], rel=1e-9)
    assert fig(a, "var_plain") == pytest.approx(g["var_plain"], rel=1e-9)
    assert_all_numbers_traced(a)


def test_regime_endpoint(db_path):
    c = TestClient(create_app(db_path))
    r = c.get("/api/regime?confidence=0.95").json()
    assert r["current"]["regime"] in ("calm", "stressed") and r["backtest"] is None
    assert c.get("/api/regime?confidence=0.3").status_code == 422
