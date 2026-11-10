"""Chat agent: turns a question into a stream of UI events.

    plan -> tool_call* -> tool_result* -> card* -> token* -> sources -> done
    (or)  plan -> approval_required            (human-in-the-loop, nothing executed)
    (or)  refusal                              (guardrail: nothing computed, no figures)

Honest notes on what is and is not "AI" here:
  * The planner is rule-based by default; with ANTHROPIC_API_KEY set the qa layer may use an LLM,
    but only to emit a validated intent. It never writes numbers or answer text.
  * The streamed answer is the grounded, templated text from riskmart.qa, delivered word by word.
  * Tool calls are the real riskmart functions that run to produce the figures (see TOOLS).
"""
from __future__ import annotations

import re
import secrets
import sqlite3
import time
from typing import Iterator, Optional

from . import qa, regime

TOOLS = {
    "portfolio_var": ["risk.load_changes (SQL view factor_changes)", "risk.scenario_pnl (full revaluation)", "risk.var_hist / risk.var_param"],
    "portfolio_es": ["risk.load_changes (SQL view factor_changes)", "risk.scenario_pnl (full revaluation)", "risk.es_hist"],
    "var_by_asset_class": ["risk.load_changes (SQL view factor_changes)", "risk.scenario_pnl (full revaluation)", "risk.var_hist per asset class"],
    "worst_day": ["risk.load_changes (SQL view factor_changes)", "risk.scenario_pnl (full revaluation)", "argmin over scenario P&L"],
    "exposure": ["SQL: positions table", "aggregate by asset class"],
    "cs01": ["risk.bond_sensitivities (+/-1bp bump)", "sum over credit bonds"],
    "dv01": ["risk.bond_sensitivities (+/-1bp bump)", "sum over bonds"],
    "backtest": ["risk.scenario_pnl (full revaluation)", "risk.backtest (rolling)", "risk.kupiec"],
    "data_quality": ["SQL: dq_results (latest ETL run)"],
    "regime": ["risk.scenario_pnl (full revaluation)", "regime.GaussianMixture fit", "regime.regime_var"],
}
PARAM_LABELS = {"confidence", "window_days", "window_end", "rv_window", "bt_days"}
REFRESH = re.compile(r"\b(refresh|update|reload|re-?run|re-?fetch|re-?download)\b.*\b(data|etl|market|mart|series)\b", re.I)
_MOD_CLASS = {"equity": "Equity", "equities": "Equity", "stocks": "Equity", "fx": "FX", "currency": "FX",
              "commodity": "Commodity", "commodities": "Commodity", "rates": "Rates", "credit": "Credit"}


def resolve_followup(question: str, ctx: Optional[dict]) -> Optional[qa.Intent]:
    """'and at 95%?' / 'what about credit?' / 'parametric?' re-use the previous intent with new parameters."""
    if not ctx:
        return None
    base = qa.validate_intent({k: ctx.get(k) for k in ("intent", "confidence", "method", "asset_class") if k in ctx})
    if base is None or qa._FORECAST.search(question):
        return None
    for w in re.findall(r"\b[A-Z][A-Za-z0-9&]+\b", question):
        if w.lower() not in qa._ALLOWED_CAPS:
            return None
    ql, changed = question.lower(), False
    conf, method, ac = base.confidence, base.method, base.asset_class
    m = re.search(r"(\d{2}(?:\.\d)?)\s*(%|percent)", ql)
    if m:
        conf, changed = float(m.group(1)) / 100.0, True
    if re.search(r"parametric|normal", ql):
        method, changed = "parametric", True
    elif re.search(r"historical", ql):
        method, changed = "historical", True
    for w, c in _MOD_CLASS.items():
        if re.search(rf"\b{w}\b", ql):
            ac, changed = c, True
            break
    if not changed:
        return None
    return qa.validate_intent({"intent": base.intent, "confidence": conf, "method": method, "asset_class": ac})


def _cards(conn, intent: qa.Intent, ans: qa.Answer) -> list[dict]:
    cards = []
    main = [f for f in ans.figures if f.label not in PARAM_LABELS and not f.label.startswith(("var_hist_", "exposure_"))]
    if main:
        cards.append({"kind": "metrics", "title": intent.intent.replace("_", " "),
                      "items": [{"label": f.label.replace("_", " "), "value": f.display} for f in main]})
    per = [f for f in ans.figures if f.label.startswith(("var_hist_", "exposure_"))]
    if per:
        pre = "var_hist_" if per[0].label.startswith("var_hist_") else "exposure_"
        cards.append({"kind": "bars", "title": "VaR by asset class" if pre == "var_hist_" else "Net exposure by asset class",
                      "items": [{"label": f.label[len(pre):], "value": float(f.value), "display": f.display} for f in per]})
    if intent.intent == "regime":
        g = regime.build_regime_report(conn, cl=intent.confidence)
        cards.append({"kind": "strip", "title": "Volatility regime by day",
                      "items": [d["stressed"] for d in g["regime_by_day"]],
                      "dates": [g["regime_by_day"][0]["date"], g["regime_by_day"][-1]["date"]]})
    return cards


def stream(conn: sqlite3.Connection, question: str, context: Optional[dict] = None,
           new_approval=None, token_delay: float = 0.0) -> Iterator[dict]:
    q = question.strip()
    if REFRESH.search(q):
        yield {"type": "plan", "intent": "refresh_data", "parser": "rules"}
        aid = new_approval("refresh_data")
        yield {"type": "approval_required", "id": aid, "action": "refresh_data",
               "title": "Refresh market data?",
               "detail": "Re-downloads all 10 public FRED series and rebuilds the SQLite mart (overwrites the "
                         "current factor data and re-runs data-quality checks). Nothing has been changed yet."}
        yield {"type": "done"}
        return

    intent, why = qa.parse_rules(q)
    parser = "rules"
    if intent is None and why == "no supported intent":
        fu = resolve_followup(q, context)
        if fu is not None:
            intent, parser = fu, "rules+context"
    if intent is None:
        ans = qa.answer_question(conn, q)         # may rescue via optional LLM; else refusal
        if not ans.supported:
            yield {"type": "refusal", "reason": why or "unsupported"}
            yield from _tokens(ans.answer, token_delay)
            yield {"type": "done"}
            return
        intent_d = ans.intent
        intent, parser = qa.Intent(**intent_d), ans.parser
    yield {"type": "plan", "intent": intent.__dict__, "parser": parser}
    names = TOOLS[intent.intent]
    for i, n in enumerate(names):
        yield {"type": "tool_call", "id": i, "name": n}
    t0 = time.perf_counter()
    ans = qa.answer_intent(conn, intent, parser)
    ms = int((time.perf_counter() - t0) * 1000)
    for i, n in enumerate(names):
        yield {"type": "tool_result", "id": i, "ok": True,
               "summary": f"done ({ms} ms for the whole tool chain)" if i == len(names) - 1 else "done"}
    for c in _cards(conn, intent, ans):
        yield {"type": "card", **c}
    yield from _tokens(ans.answer, token_delay)
    yield {"type": "sources", "items": [{"label": f.label, "display": f.display, "source": f.source} for f in ans.figures]}
    yield {"type": "final", "answer": ans.answer, "intent": intent.__dict__}
    yield {"type": "done"}


def _tokens(text: str, delay: float) -> Iterator[dict]:
    for tok in re.findall(r"\S+\s*", text):
        if delay:
            time.sleep(delay)
        yield {"type": "token", "text": tok}


class Approvals:
    """Server-side gate: an action may only run if a matching approval was issued and is used once."""
    ALLOWED = {"refresh_data"}

    def __init__(self):
        self._pending: dict[str, str] = {}

    def issue(self, action: str) -> str:
        assert action in self.ALLOWED
        aid = secrets.token_urlsafe(12)
        self._pending[aid] = action
        return aid

    def consume(self, aid: str) -> Optional[str]:
        return self._pending.pop(aid, None)
