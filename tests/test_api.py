"""
Tests for GreenFed-XAI FastAPI Backend Endpoints
================================================
Validates all REST API routes: prediction, XAI attributions, carbon sensitivity,
sensor audit, benchmark matrix, federated defense simulation, and blockchain verification.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["project"] == "GreenFed-XAI"


def test_api_overview():
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["total_consortium_banks"] == 8
    assert len(data["consortium_banks"]) == 8
    assert data["norm_threshold_tau"] == 2.50


def test_api_predict():
    payload = {
        "coupon_rate": 3.75,
        "tenor": 7.0,
        "carbon_footprint_score": 85.0,
        "esg_controversy_score": 10.0,
        "credit_rating": "AA",
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_spread_pct" in data
    assert "predicted_spread_bps" in data
    assert data["credit_rating_encoded"] == 1  # AA is encoded as 1 (AAA=0, AA=1)


def test_api_shap_explanation():
    payload = {
        "coupon_rate": 3.75,
        "tenor": 7.0,
        "carbon_footprint_score": 85.0,
        "esg_controversy_score": 10.0,
        "credit_rating": "AA",
    }
    res = client.post("/api/shap", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "base_value_bps" in data
    assert "feature_attributions_bps" in data
    assert data["efficiency_axiom_verified"] is True
    assert data["residual_error_bps"] < 0.15


def test_api_carbon_sensitivity():
    payload = {
        "coupon_rate": 3.75,
        "tenor": 7.0,
        "carbon_footprint_score": 75.0,
        "esg_controversy_score": 10.0,
        "credit_rating": "AA",
    }
    res = client.post("/api/carbon-sensitivity", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["eligible"] is True
    assert data["score_increment"] == 10.0
    assert "delta_bps" in data


def test_api_sensors():
    res = client.get("/api/sensors?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "total_readings" in data
    assert len(data["sample_records"]) <= 10


def test_api_benchmark():
    res = client.get("/api/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert data["total_scenarios"] == 11


def test_api_fl_simulation():
    payload = {"attack_type": "gaussian", "norm_threshold": 2.50}
    res = client.post("/api/fl/simulate-round", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_banks"] == 8
    assert data["quorum_met"] is True
    assert data["accepted_updates"] == 6  # 6 honest accepted, 2 gaussian rejected


def test_api_audit_lifecycle():
    # 1. Commit record
    commit_payload = {
        "bond_id": "BOND_TEST_API_001",
        "bond_params": {
            "coupon_rate": 4.0,
            "tenor": 5.0,
            "carbon_footprint_score": 80.0,
            "esg_controversy_score": 12.0,
            "credit_rating": "A",
        },
    }
    res = client.post("/api/audit/commit", json=commit_payload)
    assert res.status_code == 200
    commit_data = res.json()
    rec_id = commit_data["record_id"]
    on_chain_digest = commit_data["on_chain_digest"]
    assert rec_id.startswith("REC_")
    assert len(on_chain_digest) == 64

    # 2. Verify record (clean)
    verify_res = client.post("/api/audit/verify", json={"record_id": rec_id})
    assert verify_res.status_code == 200
    verify_data = verify_res.json()
    assert verify_data["match"] is True
    assert verify_data["tamper_detected"] is False

    # 3. Tamper record
    tamper_res = client.post("/api/audit/tamper", json={"record_id": rec_id, "tampered_bps": 50.0})
    assert tamper_res.status_code == 200
    tamper_data = tamper_res.json()
    assert tamper_data["tamper_detected"] is True
    assert tamper_data["verification_status"] == "INTEGRITY_VIOLATION_HASH_MISMATCH"
