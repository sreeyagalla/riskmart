"""Grounded natural-language Q&A over the risk mart.

Guardrail architecture (the language layer is NEVER allowed to produce a number):
  1. understand(question) -> a validated Intent {intent, confidence, method, asset_class}.
     Default parser is rule-based (no API key). If ANTHROPIC_API_KEY is set, an LLM may be used
     *only* to emit that same JSON intent; its output goes through the same validator.
  2. execute(intent)      -> numbers come from riskmart.risk (computed from the SQL mart).
  3. compose()            -> answer text is assembled from Figure.display strings only, and every
     Figure is returned as provenance. Anything unsupported gets a refusal with zero numbers.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import config, risk

INTENTS = ("portfolio_var", "portfolio_es", "var_by_asset_class", "worst_day", "exposure",
           "cs01", "dv01", "backtest", "data_quality", "regime")
METHODS = ("historical", "parametric")
CLASSES = ("Equity", "FX", "Commodity", "Rates", "Credit")

SUPPORTED_EXAMPLES = [
    "What is the 99% VaR of the portfolio?",
    "What is the 95% expected shortfall?",
    "Show VaR by asset class",
    "What was the worst day in the lookback window?",
    "What is our exposure by asset class?",
    "What is the credit spread sensitivity (CS01)?",
    "What is the DV01?",
    "How did the VaR backtest do?",
    "Are there data quality issues?",
    "What volatility regime are we in?",
]

REFUSAL = ("I can only answer from the computed risk report. I can't answer that from the data this "
           "system holds. Supported questions include: " + "; ".join(SUPPORTED_EXAMPLES))


@dataclass
class Figure:
    label: str
    value: object
    display: str
    source: str


@dataclass
class Intent:
    intent: str
    confidence: float = 0.99
    method: str = "historical"
    asset_class: Optional[str] = None


@dataclass
class Answer:
    supported: bool
    answer: str
    intent: Optional[dict] = None
    figures: list = field(default_factory=list)
    parser: str = "rules"

    def to_dict(self):
        return {"supported": self.supported, "answer": self.answer, "intent": self.intent,
                "figures": [f.__dict__ for f in self.figures], "parser": self.parser}


# ------------------------------------------------------------------ validation
def validate_intent(obj) -> Optional[Intent]:
    """Strict schema check. Anything off-schema -> None (=> refusal)."""
    if not isinstance(obj, dict) or set(obj) - {"intent", "confidence", "method", "asset_class"}:
        return None
    if obj.get("intent") not in INTENTS:
        return None
    conf = obj.get("confidence", 0.99)
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) or conf not in config.ALLOWED_CONFIDENCE:
        return None
    method = obj.get("method", "historical")
    if method not in METHODS:
        return None
    ac = obj.get("asset_class")
    if ac is not None and ac not in CLASSES:
        return None
    return Intent(obj["intent"], float(conf), method, ac)


# ------------------------------------------------------------------ rule parser
_FORECAST = re.compile(r"\b(predict|forecast|tomorrow|next (day|week|month|year)|will|should i|buy|sell|"
                       r"recommend|price target|why|because)\b", re.I)
_CLASS_WORDS = {"equity": "Equity", "equities": "Equity", "stock": "Equity", "stocks": "Equity",
                "fx": "FX", "currency": "FX", "currencies": "FX", "commodity": "Commodity",
                "commodities": "Commodity", "oil": "Commodity", "rates": "Rates", "treasury": "Rates",
                "treasuries": "Rates", "credit": "Credit"}
_ALLOWED_CAPS = {"var", "es", "fx", "cs01", "dv01", "i", "what", "how", "show", "are", "is", "which", "was",
                 "were", "do", "does", "give", "tell", "list", "our", "the", "kupiec", "ig", "hy", "usd",
                 "equity", "credit", "rates", "commodity", "treasury", "treasuries", "portfolio", "data",
                 "ust", "dq", "etl"}


def parse_rules(q: str):
    """Returns (Intent | None, refusal_reason | None)."""
    ql = q.lower()
    if _FORECAST.search(q):
        return None, "forecast/advice"
    # unknown named entities (e.g. company names) -> refuse rather than guess
    for w in re.findall(r"\b[A-Z][A-Za-z0-9&]+\b", q)[1:]:
        if w.lower() not in _ALLOWED_CAPS:
            return None, f"unknown entity {w}"
    conf = 0.99
    m = re.search(r"(\d{2}(?:\.\d)?)\s*(%|percent)", ql)
    if m:
        conf = float(m.group(1)) / 100.0
        if conf not in config.ALLOWED_CONFIDENCE:
            return None, "unsupported confidence level"
    method = "parametric" if re.search(r"parametric|normal|variance[- ]covariance|delta[- ]normal", ql) else "historical"
    ac = None
    for word, cls in _CLASS_WORDS.items():
        if re.search(rf"\b{word}\b", ql):
            ac = cls
            break
    by_class = bool(re.search(r"by (asset )?class|each asset|per asset|breakdown", ql))
    if re.search(r"regime|stressed|calm market|volatility state", ql):
        i = "regime"
    elif re.search(r"data quality|dq|quality (check|issue)|missing|stale|outlier", ql):
        i = "data_quality"
    elif re.search(r"backtest|back-test|exception|kupiec", ql):
        i = "backtest"
    elif re.search(r"\bcs01\b|spread (sensitivity|risk|dv01)|credit spread", ql):
        i = "cs01"
    elif re.search(r"\bdv01\b|duration|interest[- ]rate (sensitivity|risk)", ql):
        i = "dv01"
    elif re.search(r"worst|largest loss|biggest loss|max(imum)? loss", ql):
        i = "worst_day"
    elif re.search(r"exposure|notional|gross|net position|market value", ql):
        i = "exposure"
    elif re.search(r"expected shortfall|\bes\b|cvar|tail loss", ql):
        i = "portfolio_es"
    elif by_class and re.search(r"\bvar\b|value[- ]at[- ]risk", ql):
        i = "var_by_asset_class"
    elif re.search(r"\bvar\b|value[- ]at[- ]risk", ql):
        i = "var_by_asset_class" if ac else "portfolio_var"
    else:
        return None, "no supported intent"
    return validate_intent({"intent": i, "confidence": conf, "method": method, "asset_class": ac}), None


# ------------------------------------------------------------------ optional LLM intent parser
LLM_SYSTEM = (
    "You translate a risk manager's question into a JSON intent. Output ONLY a JSON object with keys: "
    f"intent (one of {list(INTENTS)}), confidence (one of {list(config.ALLOWED_CONFIDENCE)}), "
    f"method (one of {list(METHODS)}), asset_class (null or one of {list(CLASSES)}). "
    'If the question cannot be answered by one of those intents, output {"intent": null}. '
    "Never answer the question, never include numbers other than the confidence.")


def anthropic_call(question: str) -> str:
    body = json.dumps({"model": os.environ.get("RISKMART_LLM_MODEL", "claude-haiku-5-5"), "max_tokens": 200,
                       "system": LLM_SYSTEM, "messages": [{"role": "user", "content": question}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
        "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["content"][0]["text"]


def parse_llm(question: str, call: Callable[[str], str]) -> Optional[Intent]:
    try:
        text = call(question).strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        return validate_intent(json.loads(text))
    except Exception:
        return None


# ------------------------------------------------------------------ execution + composition
def _usd(v):  return f"${abs(v):,.0f}"
def _pct(v):  return f"{v * 100:g}%"


def answer_question(conn: sqlite3.Connection, question: str,
                    llm_call: Optional[Callable[[str], str]] = None) -> Answer:
    parser = "rules"
    intent, _ = parse_rules(question)
    if intent is None and llm_call is None and os.environ.get("ANTHROPIC_API_KEY"):
        llm_call = anthropic_call
    if intent is None and llm_call is not None:
        # The LLM may only rescue a question the rules did not understand, and the forecast /
        # unknown-entity refusals are NOT overridable.
        _, why = parse_rules(question)
        if why in (None, "no supported intent"):
            intent = parse_llm(question, llm_call)
            parser = "llm"
    if intent is None:
        return Answer(False, REFUSAL, parser=parser)

    rep = risk.build_report(conn, cl=intent.confidence)
    P, W = rep["portfolio"], rep["window"]
    figs: list[Figure] = []

    def fig(label, value, display, source):
        f = Figure(label, value, display, source)
        figs.append(f)
        return display

    cl = fig("confidence", intent.confidence, _pct(intent.confidence), "request")
    win = fig("window_days", W["days"], str(W["days"]), "risk.build_report")
    ctx = f"{win}-day historical window ending {fig('window_end', W['end'], W['end'], 'risk.build_report')}"
    i = intent.intent

    if i == "portfolio_var":
        if intent.method == "parametric":
            t = f"The {cl} 1-day parametric (normal) VaR is {fig('var_param', P['var_param'], _usd(P['var_param']), 'risk.var_param')} ({ctx})."
        else:
            t = f"The {cl} 1-day historical-simulation VaR is {fig('var_hist', P['var_hist'], _usd(P['var_hist']), 'risk.var_hist')} ({ctx})."
    elif i == "portfolio_es":
        t = f"The {cl} 1-day expected shortfall is {fig('es_hist', P['es_hist'], _usd(P['es_hist']), 'risk.es_hist')} ({ctx})."
    elif i == "var_by_asset_class":
        items = rep["by_asset_class"]
        if intent.asset_class:
            items = {intent.asset_class: items[intent.asset_class]}
        parts = [f"{k}: {fig('var_hist_' + k, v['var_hist'], _usd(v['var_hist']), 'risk.var_hist')}" for k, v in items.items()]
        t = f"Standalone {cl} 1-day historical VaR by asset class ({ctx}): " + "; ".join(parts) + "."
        if not intent.asset_class:
            t += (f" Portfolio VaR is {fig('var_hist', P['var_hist'], _usd(P['var_hist']), 'risk.var_hist')}, "
                  f"{fig('diversification_benefit', P['diversification_benefit'], _usd(P['diversification_benefit']), 'risk.build_report')} "
                  "below the sum of standalone VaRs (diversification).")
    elif i == "worst_day":
        w = P["worst_day"]
        t = (f"The worst scenario day in the {ctx} was {fig('worst_date', w['date'], w['date'], 'risk.build_report')}, "
             f"a hypothetical loss of {fig('worst_pnl', w['pnl_usd'], _usd(w['pnl_usd']), 'risk.build_report')} on today's positions.")
    elif i == "exposure":
        items = rep["by_asset_class"]
        if intent.asset_class:
            items = {intent.asset_class: items[intent.asset_class]}
        parts = [f"{k}: {'-' if v['exposure_usd'] < 0 else ''}{fig('exposure_' + k, v['exposure_usd'], _usd(v['exposure_usd']), 'positions')}"
                 for k, v in items.items()]
        t = "Net market-value exposure by asset class: " + "; ".join(parts) + "."
        if not intent.asset_class:
            t += (f" Gross exposure is {fig('gross', P['gross_exposure_usd'], _usd(P['gross_exposure_usd']), 'positions')}.")
    elif i in ("cs01", "dv01"):
        key = "cs01_usd" if i == "cs01" else "dv01_usd"
        tot = P[key]
        name = "credit-spread sensitivity (CS01)" if i == "cs01" else "interest-rate sensitivity (DV01)"
        if i == "cs01":
            t = (f"The portfolio's {name} is {fig('cs01_usd', tot, _usd(tot), 'risk.bond_sensitivities')} per +1bp "
                 "of spread widening (a loss).")
        else:
            t = (f"The portfolio's {name} is {fig('dv01_usd', tot, _usd(tot), 'risk.bond_sensitivities')} "
                 f"per -1bp move in yields ({'a gain' if tot >= 0 else 'a loss'}).")
    elif i == "backtest":
        b = rep["backtest"]
        if not b["tested_days"]:
            t = "There is not enough history to backtest."
        else:
            k = b["kupiec"]
            verdict = "rejected" if k["reject_at_5pct"] else "not rejected"
            t = (f"Over {fig('bt_days', b['tested_days'], str(b['tested_days']), 'risk.backtest')} out-of-sample days, "
                 f"the {cl} VaR had {fig('bt_exc', b['exceptions'], str(b['exceptions']), 'risk.backtest')} exceptions "
                 f"(expected {fig('bt_exp', b['expected'], f'{b['expected']:.1f}', 'risk.backtest')}); "
                 f"the Kupiec test p-value is {fig('kupiec_p', k['p_value'], f'{k['p_value']:.3f}', 'risk.kupiec')}, "
                 f"so calibration is {verdict} at 5%. Hypothetical P&L of today's book, not realised P&L.")
    elif i == "regime":
        from . import regime
        g = regime.build_regime_report(conn, cl=intent.confidence)
        pst = f"{g['current']['p_stressed'] * 100:.0f}%"
        t = (f"The ML regime model (two-state Gaussian mixture on the {fig('rv_window', regime.RV_WINDOW, str(regime.RV_WINDOW), 'regime.RV_WINDOW')}-day "
             f"realised P&L volatility) classifies today as {g['current']['regime']}, with a "
             f"{fig('p_stressed', g['current']['p_stressed'], pst, 'regime.regime_var')} "
             f"probability of the stressed state. Regime-conditional {cl} 1-day VaR is "
             f"{fig('var_regime', g['var_regime'], _usd(g['var_regime']), 'regime.regime_var')} versus "
             f"{fig('var_plain', g['var_plain'], _usd(g['var_plain']), 'risk.var_hist')} unconditional, "
             f"from {fig('cond_sample', g['conditional_sample_size'], str(g['conditional_sample_size']), 'regime.regime_var')} scenario days.")
    else:  # data_quality
        rows = conn.execute("SELECT status, COUNT(*) FROM dq_results WHERE run_id=(SELECT MAX(run_id) FROM etl_runs) "
                            "GROUP BY status").fetchall()
        c = {s: n for s, n in rows}
        t = (f"Latest ETL run data-quality checks: {fig('dq_pass', c.get('pass', 0), str(c.get('pass', 0)), 'dq_results')} passed, "
             f"{fig('dq_warn', c.get('warn', 0), str(c.get('warn', 0)), 'dq_results')} warnings, "
             f"{fig('dq_fail', c.get('fail', 0), str(c.get('fail', 0)), 'dq_results')} failures.")
    return Answer(True, t, intent=intent.__dict__, figures=figs, parser=parser)
