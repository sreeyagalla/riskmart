import sqlite3

import pandas as pd

from riskmart import config, etl
from riskmart.etl import build_panel, parse_fred_csv, run_etl

from conftest import N_DAYS, make_csvs


def test_parse_fred_missing_marker():
    df = parse_fred_csv("observation_date,X\n2024-01-01,1.5\n2024-01-02,.\n", "X")
    assert df["value"].isna().tolist() == [False, True]
    assert df["value"].iloc[0] == 1.5


def test_panel_forward_fill_is_flagged_and_bounded():
    dates = pd.bdate_range("2024-01-01", periods=12).strftime("%Y-%m-%d")
    a = pd.DataFrame({"series_id": "A", "obs_date": dates, "value": range(1, 13)})
    bvals = [float(i) for i in range(1, 13)]
    for i in (3, 8, 9, 10, 11):       # 1-day gap at 3; a 4-day gap at 8..11
        bvals[i] = float("nan")
    b = pd.DataFrame({"series_id": "B", "obs_date": dates, "value": bvals})
    panel = build_panel({"A": a, "B": b})
    pb = panel[panel.series_id == "B"].set_index("obs_date")
    assert pb.loc[dates[3], "filled"] == 1 and pb.loc[dates[3], "level"] == 3.0
    assert pb.loc[dates[11], "level"] == 8.0           # 4-day gap < limit 5: filled
    assert pb["filled"].sum() == 5


def test_gap_longer_than_limit_drops_the_date():
    dates = pd.bdate_range("2024-01-01", periods=20).strftime("%Y-%m-%d")
    a = pd.DataFrame({"series_id": "A", "obs_date": dates, "value": 1.0})
    v = [1.0] * 20
    for i in range(5, 5 + config.FFILL_LIMIT + 2):
        v[i] = float("nan")
    b = pd.DataFrame({"series_id": "B", "obs_date": dates, "value": v})
    panel = build_panel({"A": a, "B": b})
    kept = set(panel["obs_date"])
    # 7 missing in a row: the first 5 are filled, the last 2 are unfillable and dropped
    assert panel["obs_date"].nunique() == 20 - 2
    assert dates[5 + config.FFILL_LIMIT] not in kept and dates[11] not in kept
    assert dates[5 + config.FFILL_LIMIT - 1] in kept


def test_run_etl_loads_and_reconciles(conn):
    n_raw = conn.execute("SELECT COUNT(*) FROM raw_observations").fetchone()[0]
    assert n_raw == 10 * N_DAYS
    # 3 gap rows per non-SP500 series are NULL in raw but forward-filled in the panel
    assert conn.execute("SELECT COUNT(*) FROM raw_observations WHERE value IS NULL").fetchone()[0] == 9 * 3
    assert conn.execute("SELECT COUNT(DISTINCT obs_date) FROM factor_levels").fetchone()[0] == N_DAYS
    assert conn.execute("SELECT SUM(filled) FROM factor_levels").fetchone()[0] == 9 * 3
    fails = conn.execute("SELECT COUNT(*) FROM dq_results WHERE status='fail'").fetchone()[0]
    assert fails == 0


def test_factor_changes_view_matches_pandas(conn):
    lv = pd.read_sql_query("SELECT * FROM factor_levels", conn).pivot(index="obs_date", columns="series_id", values="level")
    v = pd.read_sql_query("SELECT * FROM factor_changes", conn).pivot(index="obs_date", columns="series_id", values="chg")
    pd.testing.assert_series_equal(v["SP500"], lv["SP500"].pct_change().dropna(), check_names=False)
    pd.testing.assert_series_equal(v["DGS10"], (lv["DGS10"].diff() / 100).dropna(), check_names=False)


def test_etl_is_idempotent_and_positions_are_seeded(tmp_path, csvs):
    p = tmp_path / "a.db"
    run_etl(p, fetcher=lambda s: csvs[s], seed=42)
    run_etl(p, fetcher=lambda s: csvs[s], seed=42)
    c = sqlite3.connect(p)
    assert c.execute("SELECT COUNT(*) FROM raw_observations").fetchone()[0] == 10 * N_DAYS
    first = c.execute("SELECT notional_usd FROM positions ORDER BY position_id").fetchall()
    run_etl(p, fetcher=lambda s: csvs[s], seed=42)
    assert first == c.execute("SELECT notional_usd FROM positions ORDER BY position_id").fetchall()
    run_etl(p, fetcher=lambda s: csvs[s], seed=43)
    assert first != c.execute("SELECT notional_usd FROM positions ORDER BY position_id").fetchall()


def test_dq_catches_injected_problems(tmp_path):
    bad = make_csvs()
    lines = bad["DEXUSEU"].splitlines()
    lines[50] = lines[50].split(",")[0] + ",-1.0"          # non-positive price
    lines.append(lines[60])                                 # duplicate date
    bad["DEXUSEU"] = "\n".join(lines) + "\n"
    p = tmp_path / "bad.db"
    run_etl(p, fetcher=lambda s: bad[s])
    c = sqlite3.connect(p)
    rows = {(r[0], r[1]): r[2] for r in c.execute("SELECT check_name, series_id, status FROM dq_results")}
    assert rows[("non_positive_price", "DEXUSEU")] == "fail"
    assert rows[("duplicate_dates", "DEXUSEU")] == "fail"
    assert rows[("duplicate_dates", "SP500")] == "pass"
