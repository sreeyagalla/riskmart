"""ETL: extract (FRED CSV) -> transform (parse, align, fill) -> load (SQLite) -> data-quality checks."""
from __future__ import annotations

import io
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config, dq

Fetcher = Callable[[str], str]


def fetch_fred(series_id: str, start: str = config.FETCH_START, retries: int = 3) -> str:
    url = config.FRED_URL.format(sid=series_id, start=start)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # network errors: retry with backoff
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not fetch {series_id}: {last}")


def parse_fred_csv(text: str, series_id: str) -> pd.DataFrame:
    """Parse FRED CSV. Missing markers ('.') become NaN. Duplicates are kept so DQ can see them."""
    df = pd.read_csv(io.StringIO(text), na_values=["."], dtype={0: str})
    df.columns = ["obs_date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.insert(0, "series_id", series_id)
    return df


def build_panel(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align all series on the calendar of the first (equity) series, forward-filling short gaps.

    Returns long frame: obs_date, series_id, level, filled. Dates that still have a gap after
    filling (or lie before every series has started) are dropped for all series.
    """
    wide = {}
    for sid, df in raw.items():
        wide[sid] = df.drop_duplicates("obs_date", keep="last").set_index("obs_date")["value"]
    first = next(iter(raw))
    calendar = wide[first].dropna().index
    start = max(s.dropna().index.min() for s in wide.values())
    calendar = calendar[calendar >= start]
    out = []
    for sid, s in wide.items():
        full = s.reindex(s.index.union(calendar)).sort_index()   # union so fills can bridge
        ff = full.ffill(limit=config.FFILL_LIMIT)
        aligned = ff.reindex(calendar)
        was_filled = full.isna().reindex(calendar) & aligned.notna()
        out.append(pd.DataFrame({"obs_date": calendar, "series_id": sid,
                                 "level": aligned.values, "filled": was_filled.astype(int).values}))
    panel = pd.concat(out, ignore_index=True)
    bad_dates = panel.loc[panel["level"].isna(), "obs_date"].unique()
    return panel[~panel["obs_date"].isin(bad_dates)].reset_index(drop=True)


def connect(db_path: Path | str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(config.SCHEMA_SQL.read_text(encoding="utf-8"))


def run_etl(db_path: Path | str = config.DEFAULT_DB, fetcher: Fetcher = fetch_fred,
            source: str = "FRED fredgraph.csv", seed: int = 42) -> dict:
    """Full refresh. Raw/panel/positions are rebuilt; run history and DQ results append."""
    from . import portfolio
    conn = connect(db_path)
    init_schema(conn)
    parsed: dict[str, pd.DataFrame] = {}
    for sid, *_ in config.SERIES:
        parsed[sid] = parse_fred_csv(fetcher(sid), sid)

    panel = build_panel(parsed)
    with conn:
        conn.execute("DELETE FROM positions")      # children first (foreign keys)
        conn.execute("DELETE FROM factor_levels")
        conn.execute("DELETE FROM raw_observations")
        conn.execute("DELETE FROM series_meta")
        conn.executemany("INSERT INTO series_meta VALUES (?,?,?,?,?)", config.SERIES)
        raw_rows = 0
        for sid, df in parsed.items():
            d = df.drop_duplicates("obs_date", keep="last")
            rows = [(r.series_id, r.obs_date, None if pd.isna(r.value) else float(r.value))
                    for r in d.itertuples()]
            conn.executemany("INSERT INTO raw_observations VALUES (?,?,?)", rows)
            raw_rows += len(rows)
        conn.executemany("INSERT INTO factor_levels VALUES (?,?,?,?)",
                         [(r.obs_date, r.series_id, float(r.level), int(r.filled))
                          for r in panel.itertuples()])
        cur = conn.execute("INSERT INTO etl_runs (started_at, source, rows_raw, rows_panel) VALUES (?,?,?,?)",
                           (datetime.now(timezone.utc).isoformat(timespec="seconds"), source,
                            raw_rows, len(panel)))
        run_id = cur.lastrowid
    portfolio.load_positions(conn, seed=seed)
    results = dq.run_checks(conn, run_id, parsed, panel)
    conn.close()
    return {"run_id": run_id, "rows_raw": raw_rows, "rows_panel": len(panel),
            "dq": {s: sum(1 for r in results if r["status"] == s) for s in ("pass", "warn", "fail")}}
