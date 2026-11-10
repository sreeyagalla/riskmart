"""Static demo builder (for GitHub Pages).

Pages can only serve static files, so the Python engine runs HERE, once, and precomputes the grounded
answer + result cards + sources for every supported question variant. The browser then runs the same chat
UI against that snapshot. Nothing is computed or invented in the browser: it parses the question (a JS port
of the rule planner, parity-tested against Python) and replays the stored answer for that exact intent.

Only aggregate risk figures for the synthetic book and per-day regime labels are published; no raw market
series are included.
"""
from __future__ import annotations

import itertools
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import agent, config, qa

STATIC = Path(__file__).parent / "static"
CLASSES = (None,) + qa.CLASSES

# which parameters change the answer for each intent (the snapshot key only uses these)
RELEVANT = {
    "portfolio_var": ("confidence", "method"),
    "portfolio_es": ("confidence",),
    "var_by_asset_class": ("confidence", "asset_class"),
    "worst_day": ("confidence",),
    "exposure": ("asset_class",),
    "cs01": (), "dv01": (),
    "backtest": ("confidence",),
    "data_quality": (),
    "regime": ("confidence",),
}


def key_of(intent: dict) -> str:
    parts = [intent["intent"]]
    for dim in RELEVANT[intent["intent"]]:
        v = intent.get(dim)
        parts.append("-" if v is None else f"{v:g}" if isinstance(v, float) else str(v))
    return "|".join(parts)


def _variants(name: str):
    dims = RELEVANT[name]
    grid = {"confidence": config.ALLOWED_CONFIDENCE, "method": qa.METHODS, "asset_class": CLASSES}
    for combo in itertools.product(*(grid[d] for d in dims)):
        yield qa.Intent(name, **dict(zip(dims, combo)))


def build(conn: sqlite3.Connection) -> dict:
    entries = {}
    for name in RELEVANT:
        for intent in _variants(name):
            ans = qa.answer_intent(conn, intent, "rules")
            entries[key_of(intent.__dict__)] = {
                "intent": intent.__dict__,
                "tools": agent.TOOLS[name],
                "cards": agent._cards(conn, intent, ans),
                "answer": ans.answer,
                "sources": [{"label": f.label, "display": f.display, "source": f.source} for f in ans.figures],
            }
    as_of, n_days, first = conn.execute("SELECT MAX(obs_date), COUNT(DISTINCT obs_date), MIN(obs_date) FROM factor_levels").fetchone()
    spec = qa.rules_spec()
    spec["mod_class"] = agent._MOD_CLASS
    spec["relevant"] = {k: list(v) for k, v in RELEVANT.items()}
    return {
        "meta": {"data_through": as_of, "data_from": first, "panel_dates": n_days,
                 "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "n_entries": len(entries),
                 "source": conn.execute("SELECT source FROM etl_runs ORDER BY run_id DESC LIMIT 1").fetchone()[0]},
        "rules": spec,
        "entries": entries,
    }


def write(conn: sqlite3.Connection, out: Path | str) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    snap = build(conn)
    (out / "snapshot.json").write_text(json.dumps(snap, separators=(",", ":")), encoding="utf-8")
    shutil.copy(STATIC / "static_backend.js", out / "static_backend.js")
    html = (STATIC / "chat.html").read_text(encoding="utf-8")
    inject = '<script>window.RISKMART_STATIC=true</script>\n<script src="static_backend.js"></script>\n'
    assert "<!--STATIC_HOOK-->" in html
    (out / "index.html").write_text(html.replace("<!--STATIC_HOOK-->", inject), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return snap["meta"]
