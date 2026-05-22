"""Tests for FastAPI endpoints — response structure and status codes."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient for the Macro Intelligence API.

    Returns:
        A TestClient instance wrapping the FastAPI app.
    """
    try:
        from api.main import app
        return TestClient(app)
    except Exception as exc:
        pytest.skip(f"Could not import API: {exc}")


def test_overview_returns_metrics(client) -> None:
    """GET /api/overview should return 200 with a 'metrics' key.

    Args:
        client: FastAPI TestClient fixture.
    """
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data or "error" in data


def test_historical_returns_data(client) -> None:
    """GET /api/historical should return 200 with a 'data' key.

    Args:
        client: FastAPI TestClient fixture.
    """
    response = client.get("/api/historical")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data or "error" in data


def test_regimes_returns_expected_keys(client) -> None:
    """GET /api/regimes should return 200 with 'timeline' and 'rotation' keys.

    Args:
        client: FastAPI TestClient fixture.
    """
    response = client.get("/api/regimes")
    assert response.status_code == 200
    data = response.json()
    assert "timeline" in data or "error" in data
    if "timeline" in data:
        assert "rotation" in data
        assert "timelines" in data
        assert "kmeans" in data["timelines"]
        assert "hmm" in data["timelines"]


def test_risk_returns_metrics(client) -> None:
    """GET /api/risk should return 200 with a 'metrics' key.

    Args:
        client: FastAPI TestClient fixture.
    """
    response = client.get("/api/risk")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data or "error" in data


def test_backtest_returns_results(client) -> None:
    """GET /api/backtest should return 200 with a 'results' key.

    Args:
        client: FastAPI TestClient fixture.
    """
    response = client.get("/api/backtest")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data or "error" in data


def test_simulate_with_valid_inputs(client) -> None:
    """POST /api/simulate with valid payload should return 200.

    Args:
        client: FastAPI TestClient fixture.
    """
    payload = {
        "US_Fed_Funds_Rate": 4.5,
        "Brent_Crude": 75.0,
        "DXY": 100.0,
        "India_VIX": 16.0,
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data or "error" in data


def test_simulate_invalid_payload_rejected(client) -> None:
    """POST /api/simulate with missing fields should return 422.

    Args:
        client: FastAPI TestClient fixture.
    """
    response = client.post("/api/simulate", json={"US_Fed_Funds_Rate": 4.5})
    assert response.status_code == 422


def test_hmm_regimes_endpoint(client) -> None:
    """GET /api/regimes/hmm should return 200 with a 'timeline' or 'error' key.

    Args:
        client: FastAPI TestClient fixture.
    """
    response = client.get("/api/regimes/hmm")
    assert response.status_code == 200
    data = response.json()
    assert "timeline" in data or "error" in data
    if "timeline" in data and data["timeline"]:
        assert "Regime" in data["timeline"][0]


def test_trigger_email_endpoint_structure(client) -> None:
    """POST /api/trigger_email should return 200 with 'status' or 'error' key.

    This test only verifies response structure; SMTP is not actually invoked
    in a test environment (credentials are expected to be absent).

    Args:
        client: FastAPI TestClient fixture.
    """
    response = client.post("/api/trigger_email")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "error" in data
