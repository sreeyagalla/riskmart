"""The static (GitHub Pages) demo must behave exactly like the Python agent it replays."""
import json
import shutil
import subprocess

import pytest

from riskmart import agent, snapshot
from riskmart.qa import SUPPORTED_EXAMPLES

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

VAR_CTX = {"intent": "portfolio_var", "confidence": 0.99, "method": "historical", "asset_class": None}
CASES = [{"q": q} for q in SUPPORTED_EXAMPLES] + [
    {"q": q} for q in [
        "What is the 95% VaR of the portfolio?", "97.5% value at risk", "parametric VaR at 99%", "normal var 95%",
        "VaR by asset class at 95%", "95% VaR for credit", "VaR for FX", "equity VaR 97.5%",
        "expected shortfall 97.5%", "What is the 99% CVaR?", "worst day at 95%", "exposure to credit",
        "net exposure by asset class", "gross exposure", "CS01", "what's the DV01", "backtest at 95%", "any outliers or stale data?",
        "is the market stressed?", "regime at 95%",
        "What will the S&P 500 do tomorrow?", "Should I buy Tesla stock?", "Tesla 99% VaR", "Apple VaR please",
        "What is the 90% VaR of the portfolio?", "tell me a joke", "Why did the portfolio lose money?",
        "refresh the market data", "please reload the data", "re-run the ETL",
        "What is the VaR for the portfolio today?", "Hello", "",
    ]
] + [
    {"q": "And at 95%?", "ctx": VAR_CTX}, {"q": "what about credit?", "ctx": VAR_CTX},
    {"q": "parametric?", "ctx": VAR_CTX}, {"q": "and at 97.5% for fx", "ctx": VAR_CTX},
    {"q": "and at 95%?", "ctx": None}, {"q": "and for Tesla at 95%?", "ctx": VAR_CTX},
    {"q": "and will it be higher tomorrow at 95%?", "ctx": VAR_CTX}, {"q": "and at 90%?", "ctx": VAR_CTX},
]


def norm(events):
    """Event stream minus fields that legitimately differ (timings, approval ids, token splits)."""
    out, text = [], ""
    for e in events:
        t = e["type"]
        if t == "token":
            text += e["text"]
            continue
        if t == "tool_result":
            e = {"type": t, "id": e["id"], "ok": e["ok"]}
        if t == "approval_required":
            e = {"type": t, "action": e["action"], "title": e["title"]}
        if t == "refusal":
            e = {"type": t}
        out.append(e)
    return out, text.strip()


def test_static_replay_matches_python_agent(conn, tmp_path):
    snap = tmp_path / "snapshot.json"
    meta = snapshot.write(conn, tmp_path)
    assert meta["n_entries"] == 45                      # 6+3+18+3+6+1+1+3+1+3 parameter variants
    assert (tmp_path / "index.html").exists()
    assert "RISKMART_STATIC" in (tmp_path / "index.html").read_text(encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(CASES))
    res = subprocess.run(["node", "tests/static_parity.js", str(snap), str(cases)], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    js_runs = json.loads(res.stdout)
    assert len(js_runs) == len(CASES)
    for case, js_events in zip(CASES, js_runs):
        py_events = list(agent.stream(conn, case["q"], case.get("ctx"), new_approval=lambda a: "x"))
        assert norm(js_events) == norm(py_events), f"mismatch for {case!r}"


def test_every_snapshot_answer_is_traceable(conn):
    from test_grounding import assert_all_numbers_traced
    from riskmart import qa
    snap = snapshot.build(conn)
    for key, e in snap["entries"].items():
        ans = qa.Answer(True, e["answer"], figures=[qa.Figure(s["label"], None, s["display"], s["source"]) for s in e["sources"]])
        assert_all_numbers_traced(ans)
