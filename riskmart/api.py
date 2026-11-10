"""REST API + static dashboard.  JSON everywhere; /api/report.xml for XML consumers."""
from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from . import config, qa, regime, risk

STATIC = Path(__file__).parent / "static"


class Ask(BaseModel):
    question: str


def _to_xml(tag: str, obj) -> ET.Element:
    el = ET.Element(tag)
    if isinstance(obj, dict):
        for k, v in obj.items():
            el.append(_to_xml(str(k), v))
    elif isinstance(obj, list):
        for v in obj:
            el.append(_to_xml("item", v))
    else:
        el.text = str(obj)
    return el


def create_app(db_path: Path | str = config.DEFAULT_DB) -> FastAPI:
    app = FastAPI(title="riskmart", version="0.1.0")

    def conn() -> sqlite3.Connection:
        if not Path(db_path).exists():
            raise HTTPException(503, "mart not built: run `python -m riskmart etl`")
        return sqlite3.connect(str(db_path))

    def report(cl: float, lookback: int, include_dist: bool = False) -> dict:
        if cl not in config.ALLOWED_CONFIDENCE:
            raise HTTPException(422, f"confidence must be one of {config.ALLOWED_CONFIDENCE}")
        c = conn()
        try:
            r = risk.build_report(c, cl=cl, lookback=lookback)
        finally:
            c.close()
        if not include_dist:
            r.pop("pnl_distribution")
        return r

    @app.get("/api/health")
    def health():
        c = conn()
        n = c.execute("SELECT COUNT(DISTINCT obs_date) FROM factor_levels").fetchone()[0]
        c.close()
        return {"status": "ok", "panel_dates": n}

    @app.get("/api/report")
    def get_report(confidence: float = 0.99, lookback: int = Query(config.DEFAULT_LOOKBACK, ge=100, le=2000),
                   distribution: bool = False):
        return report(confidence, lookback, distribution)

    @app.get("/api/report.xml")
    def get_report_xml(confidence: float = 0.99, lookback: int = Query(config.DEFAULT_LOOKBACK, ge=100, le=2000)):
        xml = ET.tostring(_to_xml("riskReport", report(confidence, lookback)), encoding="unicode")
        return Response('<?xml version="1.0" encoding="UTF-8"?>' + xml, media_type="application/xml")

    @app.get("/api/regime")
    def get_regime(confidence: float = 0.99, lookback: int = Query(config.DEFAULT_LOOKBACK, ge=100, le=2000),
                   backtest: bool = False):
        """ML volatility-regime model. backtest=true adds a ~10 s rolling comparison vs plain VaR."""
        if confidence not in config.ALLOWED_CONFIDENCE:
            raise HTTPException(422, f"confidence must be one of {config.ALLOWED_CONFIDENCE}")
        c = conn()
        try:
            return regime.build_regime_report(c, cl=confidence, lookback=lookback, with_backtest=backtest)
        finally:
            c.close()

    @app.get("/api/positions")
    def positions():
        c = conn()
        try:
            df = risk.load_positions(c).astype(object)
            return df.where(df.notna(), None).to_dict(orient="records")
        finally:
            c.close()

    @app.get("/api/dq")
    def dq():
        c = conn()
        try:
            cur = c.execute("SELECT check_name, series_id, status, value, detail FROM dq_results "
                            "WHERE run_id=(SELECT MAX(run_id) FROM etl_runs) ORDER BY "
                            "CASE status WHEN 'fail' THEN 0 WHEN 'warn' THEN 1 ELSE 2 END, series_id, check_name")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            c.close()

    @app.post("/api/ask")
    def ask(body: Ask):
        c = conn()
        try:
            return qa.answer_question(c, body.question).to_dict()
        finally:
            c.close()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
