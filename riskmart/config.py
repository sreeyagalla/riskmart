"""Static configuration: the public risk-factor series and where the data lives."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_DB = DATA_DIR / "riskmart.db"
SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
FETCH_START = "2016-01-01"

# kind: 'price' -> risk-factor change is a simple return; 'rate' -> change is an absolute move
# in percentage points (converted to decimal by the SQL view).
SERIES = [
    # series_id,       name,                              asset_class, kind,    unit
    ("SP500",         "S&P 500 index",                    "Equity",    "price", "index pts"),
    ("NASDAQCOM",     "NASDAQ Composite index",           "Equity",    "price", "index pts"),
    ("DEXUSEU",       "USD per EUR spot",                 "FX",        "price", "USD/EUR"),
    ("DEXJPUS",       "JPY per USD spot",                 "FX",        "price", "JPY/USD"),
    ("DCOILBRENTEU",  "Brent crude spot",                 "Commodity", "price", "USD/bbl"),
    ("DGS2",          "US Treasury 2y CMT yield",         "Rates",     "rate",  "percent"),
    ("DGS5",          "US Treasury 5y CMT yield",         "Rates",     "rate",  "percent"),
    ("DGS10",         "US Treasury 10y CMT yield",        "Rates",     "rate",  "percent"),
    ("BAMLC0A0CM",    "ICE BofA US Corporate (IG) OAS",   "Credit",    "rate",  "percent"),
    ("BAMLH0A0HYM2",  "ICE BofA US High Yield OAS",       "Credit",    "rate",  "percent"),
]

ALLOWED_CONFIDENCE = (0.95, 0.975, 0.99)
DEFAULT_LOOKBACK = 500
DEFAULT_BACKTEST_DAYS = 250
FFILL_LIMIT = 5          # max consecutive days a gap may be forward-filled
OUTLIER_MADS = 8.0       # robust z-score threshold for flagging (not removing) outliers
