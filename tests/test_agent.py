import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from riskmart import agent, etl
from riskmart.api import create_app
from independent import compute
from test_grounding import assert_all_numbers_traced, fig
from riskmart import qa
from conftest import make_csvs


def events(conn, q, ctx=None):
    issued = []
    ev = list(agent.stream(conn, q, ctx, new_approval=lambda a: issued.append(a) or "id-1"))
    return ev, issued


def types(ev):
    return [e["type"] for e in ev]


def test_event_order_and_streamed_text_equals_grounded_answer(conn):
    ev, _ = events(conn, "What is the 99% VaR of the portfolio?")
    t = types(ev)
    assert t[0] == "plan" and t[-1] == "done"
    assert t.index("tool_call") < t.index("tool_result") < t.index("card") < t.index("token") < t.index("sources")
    streamed = "".join(e["text"] for e in ev if e["type"] == "token")
    final = next(e for e in ev if e["type"] == "final")["answer"]
    assert streamed.strip() == final
    assert streamed.strip() == qa.answer_question(conn, "What is the 99% VaR of the portfolio?").answer
    # every tool_call has a matching tool_result
    assert sorted(e["id"] for e in ev if e["type"] == "tool_call") == sorted(e["id"] for e in ev if e["type"] == "tool_result")


def test_cards_show_the_same_numbers_as_the_independent_computation(conn, db_path):
    ind = compute(db_path, 0.99)
    ev, _ = events(conn, "Show VaR by asset class")
    bars = next(e for e in ev if e["type"] == "card" and e["kind"] == "bars")
    for it in bars["items"]:
        assert it["value"] == pytest.approx(ind["by_class"][it["label"]], rel=1e-9)
    ev, _ = events(conn, "What is the 99% VaR of the portfolio?")
    metrics = next(e for e in ev if e["type"] == "card" and e["kind"] == "metrics")
    assert metrics["items"][0]["value"] == f"${ind['var_hist']:,.0f}"


@pytest.mark.parametrize("q", ["What will the S&P 500 do tomorrow?", "VaR of Tesla?", "tell me a joke"])
def test_refusals_emit_no_tools_cards_or_figures(conn, q):
    ev, issued = events(conn, q)
    t = types(ev)
    assert "refusal" in t and not ({"tool_call", "card", "sources", "approval_required"} & set(t))
    assert issued == []
    assert "$" not in "".join(e.get("text", "") for e in ev)


def test_followup_uses_previous_intent_and_matches_independent_value(conn, db_path):
    ev, _ = events(conn, "What is the 99% VaR of the portfolio?")
    ctx = next(e for e in ev if e["type"] == "final")["intent"]
    ev2, _ = events(conn, "And at 95%?", ctx)
    fin = next(e for e in ev2 if e["type"] == "final")
    assert fin["intent"]["intent"] == "portfolio_var" and fin["intent"]["confidence"] == 0.95
    assert f"${compute(db_path, 0.95)['var_hist']:,.0f}" in fin["answer"]
    ev3, _ = events(conn, "what about credit?", {"intent": "var_by_asset_class", "confidence": 0.95, "method": "historical", "asset_class": None})
    assert next(e for e in ev3 if e["type"] == "final")["intent"]["asset_class"] == "Credit"


def test_followup_does_not_bypass_guardrails(conn):
    ctx = {"intent": "portfolio_var", "confidence": 0.99, "method": "historical", "asset_class": None}
    assert "refusal" in types(events(conn, "and will it be higher tomorrow at 95%?", ctx)[0])
    assert "refusal" in types(events(conn, "and for Tesla at 95%?", ctx)[0])
    assert "refusal" in types(events(conn, "and at 95%?", None)[0])           # no context -> nothing to follow up


def test_refresh_requests_approval_and_executes_nothing(conn):
    before = conn.execute("SELECT COUNT(*) FROM etl_runs").fetchone()[0]
    ev, issued = events(conn, "Please refresh the market data")
    assert types(ev) == ["plan", "approval_required", "done"] and issued == ["refresh_data"]
    assert conn.execute("SELECT COUNT(*) FROM etl_runs").fetchone()[0] == before


def parse_sse(text):
    return [json.loads(b[6:]) for b in text.split("\n\n") if b.startswith("data: ")]


def test_http_stream_and_human_in_the_loop_gate(tmp_path):
    csvs = make_csvs()
    db = tmp_path / "hitl.db"
    etl.run_etl(db, fetcher=lambda s: csvs[s])
    calls = []
    client = TestClient(create_app(db, fetcher=lambda s: calls.append(s) or csvs[s], token_delay=0))

    ev = parse_sse(client.post("/api/chat", json={"question": "What is the 95% expected shortfall?"}).text)
    assert types(ev)[-1] == "done" and any(e["type"] == "token" for e in ev)

    ev = parse_sse(client.post("/api/chat", json={"question": "refresh the data"}).text)
    aid = next(e["id"] for e in ev if e["type"] == "approval_required")
    assert calls == []                                                          # nothing ran before approval
    assert client.post("/api/chat/approve", json={"id": "forged", "approve": True}).status_code == 403
    assert calls == []
    rej = client.post("/api/chat/approve", json={"id": aid, "approve": False}).json()
    assert rej["status"] == "rejected" and calls == []
    assert client.post("/api/chat/approve", json={"id": aid, "approve": True}).status_code == 403   # single-use

    aid2 = next(e["id"] for e in parse_sse(client.post("/api/chat", json={"question": "reload market data"}).text)
                if e["type"] == "approval_required")
    ok = client.post("/api/chat/approve", json={"id": aid2, "approve": True}).json()
    assert ok["status"] == "executed" and len(calls) == 10 and ok["result"]["dq"]["fail"] == 0
    assert client.post("/api/chat/approve", json={"id": aid2, "approve": True}).status_code == 403
    assert client.get("/chat").status_code == 200
