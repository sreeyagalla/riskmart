import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from riskmart.api import create_app


@pytest.fixture()
def client(db_path):
    return TestClient(create_app(db_path))


def test_health_and_static_dashboard(client):
    assert client.get("/api/health").json()["status"] == "ok"
    r = client.get("/")
    assert r.status_code == 200 and "Market risk dashboard" in r.text


def test_report_json_and_xml_agree(client):
    j = client.get("/api/report?confidence=0.95").json()
    x = ET.fromstring(client.get("/api/report.xml?confidence=0.95").text)
    assert float(x.find("portfolio/var_hist").text) == pytest.approx(j["portfolio"]["var_hist"])
    assert x.find("window/end").text == j["window"]["end"]
    assert "pnl_distribution" not in j


def test_validation_and_endpoints(client):
    assert client.get("/api/report?confidence=0.5").status_code == 422
    assert client.get("/api/report?lookback=5").status_code == 422
    assert len(client.get("/api/positions").json()) == 10
    dq = client.get("/api/dq").json()
    assert dq and {"check_name", "status", "detail"} <= set(dq[0])


def test_ask_endpoint(client):
    a = client.post("/api/ask", json={"question": "What is the 99% VaR of the portfolio?"}).json()
    j = client.get("/api/report?confidence=0.99").json()
    assert a["supported"]
    assert any(abs(f["value"] - j["portfolio"]["var_hist"]) < 1e-6 for f in a["figures"] if isinstance(f["value"], float))
    assert client.post("/api/ask", json={"question": "will it rain?"}).json()["supported"] is False


def test_503_when_mart_missing(tmp_path):
    assert TestClient(create_app(tmp_path / "none.db")).get("/api/health").status_code == 503
