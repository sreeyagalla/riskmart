-- riskmart data mart (SQLite; ANSI-leaning SQL, window functions in the view)
CREATE TABLE IF NOT EXISTS series_meta (
    series_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('price','rate')),
    unit        TEXT NOT NULL
);

-- Landing table: exactly what the source returned (missing markers stored as NULL).
CREATE TABLE IF NOT EXISTS raw_observations (
    series_id TEXT NOT NULL REFERENCES series_meta(series_id),
    obs_date  TEXT NOT NULL,
    value     REAL,
    PRIMARY KEY (series_id, obs_date)
);

-- Clean, aligned panel on a common calendar. filled=1 marks forward-filled points.
CREATE TABLE IF NOT EXISTS factor_levels (
    obs_date  TEXT NOT NULL,
    series_id TEXT NOT NULL REFERENCES series_meta(series_id),
    level     REAL NOT NULL,
    filled    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (obs_date, series_id)
);

CREATE TABLE IF NOT EXISTS positions (
    position_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    asset_class     TEXT NOT NULL,
    instrument      TEXT NOT NULL CHECK (instrument IN ('DELTA1','BOND')),
    factor_id       TEXT NOT NULL REFERENCES series_meta(series_id),
    spread_factor_id TEXT REFERENCES series_meta(series_id),
    invert          INTEGER NOT NULL DEFAULT 0,
    notional_usd    REAL NOT NULL,           -- signed market value in USD (negative = short)
    coupon          REAL,                    -- annual coupon rate, decimal
    maturity_years  REAL,
    freq            INTEGER
);

CREATE TABLE IF NOT EXISTS etl_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    source      TEXT NOT NULL,
    rows_raw    INTEGER NOT NULL,
    rows_panel  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS dq_results (
    run_id     INTEGER NOT NULL REFERENCES etl_runs(run_id),
    check_name TEXT NOT NULL,
    series_id  TEXT,
    status     TEXT NOT NULL CHECK (status IN ('pass','warn','fail')),
    value      REAL,
    detail     TEXT NOT NULL
);

-- One-day risk-factor changes. Prices -> simple return; rates/spreads -> absolute move in
-- decimal (percentage points / 100). LAG() is evaluated per series in date order.
DROP VIEW IF EXISTS factor_changes;
CREATE VIEW factor_changes AS
SELECT obs_date, series_id, level, chg
FROM (
    SELECT f.obs_date, f.series_id, f.level,
           CASE m.kind
                WHEN 'price' THEN f.level * 1.0 / LAG(f.level) OVER w - 1.0
                ELSE (f.level - LAG(f.level) OVER w) / 100.0
           END AS chg
    FROM factor_levels f
    JOIN series_meta m USING (series_id)
    WINDOW w AS (PARTITION BY f.series_id ORDER BY f.obs_date)
)
WHERE chg IS NOT NULL;
