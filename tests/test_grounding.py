"""THE guardrail tests: every number the Q&A layer emits must (a) equal an independently computed
value and (b) be traceable to a returned Figure; unsupported questions get a refusal with no numbers."""
import json
import re

import pytest
from scipy import stats

from riskmart import qa
from independent import compute

NUM = re.compile(r"(?<![A-Za-z\d])\d[\d,]*\.?\d*")   # not part of a label like CS01 / DV01
# horizon / bump-size / significance-level constants that are part of the method's wording
CONSTANTS = ("1-day", "+1bp", "-1bp", "at 5%")


def numbers_in(text):
    for c in CONSTANTS:
        text = text.replace(c, " ")
    return [t.rstrip(".,") for t in NUM.findall(text)]


def assert_all_numbers_traced(a):
    allowed = set()
    for f in a.figures:
        allowed.update(numbers_in(f.display))
    for tok in numbers_in(a.answer):
        assert tok in allowed, f"number {tok!r} in answer is not backed by a figure: {a.answer}"


def fig(a, label):
    return next(f.value for f in a.figures if f.label == label)


@pytest.mark.parametrize("cl", [0.95, 0.975, 0.99])
def test_var_es_answers_equal_independent_computation(conn, db_path, cl):
    ind = compute(db_path, cl)
    a = qa.answer_question(conn, f"What is the {cl * 100:g}% VaR of the portfolio?")
    assert a.supported and fig(a, "var_hist") == pytest.approx(ind["var_hist"], rel=1e-9)
    assert_all_numbers_traced(a)
    assert f"${abs(ind['var_hist']):,.0f}" in a.answer          # the displayed number itself matches

    a = qa.answer_question(conn, f"expected shortfall at {cl * 100:g}%")
    assert fig(a, "es_hist") == pytest.approx(ind["es_hist"], rel=1e-9)
    assert_all_numbers_traced(a)

    a = qa.answer_question(conn, f"parametric VaR at {cl * 100:g}%")
    assert fig(a, "var_param") == pytest.approx(stats.norm.ppf(cl) * ind["var_param_sigma"], rel=1e-9)
    assert_all_numbers_traced(a)


def test_by_class_worst_day_cs01_exposure(conn, db_path):
    ind = compute(db_path, 0.99)
    a = qa.answer_question(conn, "Show VaR by asset class")
    for ac, v in ind["by_class"].items():
        assert fig(a, f"var_hist_{ac}") == pytest.approx(v, rel=1e-9)
    assert fig(a, "var_hist") == pytest.approx(ind["var_hist"], rel=1e-9)
    assert_all_numbers_traced(a)

    a = qa.answer_question(conn, "What was the worst day?")
    assert fig(a, "worst_pnl") == pytest.approx(ind["worst_pnl"], rel=1e-9)
    assert fig(a, "worst_date") == ind["worst_date"]
    assert_all_numbers_traced(a)

    a = qa.answer_question(conn, "What is the credit spread sensitivity (CS01)?")
    assert fig(a, "cs01_usd") == pytest.approx(ind["cs01"], rel=1e-3)   # bump (central vs forward) tolerance
    assert_all_numbers_traced(a)

    a = qa.answer_question(conn, "What is our exposure by asset class?")
    for ac, v in ind["exposure"].items():
        assert fig(a, f"exposure_{ac}") == pytest.approx(v)
    assert fig(a, "gross") == pytest.approx(ind["gross"])
    assert_all_numbers_traced(a)


def test_single_asset_class_scope(conn, db_path):
    ind = compute(db_path, 0.95)
    a = qa.answer_question(conn, "95% VaR for credit")
    assert [f.label for f in a.figures if f.label.startswith("var_hist_")] == ["var_hist_Credit"]
    assert fig(a, "var_hist_Credit") == pytest.approx(ind["by_class"]["Credit"], rel=1e-9)


UNSUPPORTED = [
    "What will the S&P 500 do tomorrow?",
    "Should I buy Tesla stock?",
    "What is the 99% VaR of Tesla?",
    "What is the 90% VaR of the portfolio?",          # unsupported confidence level: don't guess
    "What is the weather in London?",
    "Predict next week's loss",
    "Tell me a joke",
    "Why did the portfolio lose money in April?",
]


@pytest.mark.parametrize("q", UNSUPPORTED)
def test_unsupported_questions_are_refused_with_no_figures(conn, q):
    a = qa.answer_question(conn, q)
    assert a.supported is False and a.figures == []
    assert a.answer == qa.REFUSAL
    # the refusal text contains the example questions' percentages only, never a computed value
    assert "$" not in a.answer


def test_llm_cannot_inject_numbers_or_bypass_validation(conn):
    # 1) a hostile/hallucinating "LLM" that answers the question itself instead of emitting an intent
    a = qa.answer_question(conn, "gimme a number for risk tho", llm_call=lambda q: "The VaR is $42 million.")
    assert not a.supported and a.figures == []
    # 2) off-schema intent / extra keys / invalid confidence are all rejected by the validator
    for bad in ['{"intent": "portfolio_var", "confidence": 0.42}',
                '{"intent": "drop_tables"}',
                '{"intent": "portfolio_var", "answer": "VaR is $1"}',
                '{"intent": null}', "not json at all"]:
        assert not qa.answer_question(conn, "gimme a number for risk tho", llm_call=lambda q, b=bad: b).supported
    # 3) a valid intent from the LLM is accepted, but the numbers still come from computation
    good = json.dumps({"intent": "portfolio_var", "confidence": 0.95, "method": "historical", "asset_class": None})
    a = qa.answer_question(conn, "gimme a number for risk tho", llm_call=lambda q: good)
    assert a.supported and a.parser == "llm"
    assert_all_numbers_traced(a)
    # 4) the LLM cannot override the forecast/unknown-entity refusals
    assert not qa.answer_question(conn, "What will the VaR be tomorrow?", llm_call=lambda q: good).supported
    assert not qa.answer_question(conn, "What is the VaR of Tesla?", llm_call=lambda q: good).supported
