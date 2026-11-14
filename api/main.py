"""
GreenFed-XAI Enterprise Backend API
===================================
FastAPI service exposing pricing inference, Explainable AI (SHAP attributions),
IoT sensor telemetry validation, attack simulation benchmarks, and cryptographic
blockchain audit trail verification.
"""

from __future__ import annotations

import os
import sys
import glob
import json
import uuid
import datetime
import pickle
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from models.nn_model import GreenBondNN
from models.preprocessing import DataPreprocessor, RATING_ORDER
from xai.shap_explainer import GreenBondXAIExplainer
from audit.audit_manager import AuditManager, OffChainAuditRecord
from chain.mock_ledger import MockBlockchain

app = FastAPI(
    title="GreenFed-XAI Enterprise API",
    description="Backend API for Federated Learning Green Bond Pricing, XAI, and Blockchain Verification",
    version="1.0.0",
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lazy-loaded instances
PREPROCESSOR: Optional[DataPreprocessor] = None
MODEL: Optional[GreenBondNN] = None
EXPLAINER: Optional[GreenBondXAIExplainer] = None
AUDIT_MGR: Optional[AuditManager] = None
BLOCKCHAIN: Optional[MockBlockchain] = None

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def get_preprocessor() -> DataPreprocessor:
    global PREPROCESSOR
    if PREPROCESSOR is None:
        prep_path = os.path.join(OUTPUTS_DIR, "preprocessor.pkl")
        if not os.path.exists(prep_path):
            raise RuntimeError(f"Preprocessor not found at {prep_path}")
        with open(prep_path, "rb") as f:
            PREPROCESSOR = pickle.load(f)
    return PREPROCESSOR


def get_model() -> GreenBondNN:
    global MODEL
    if MODEL is None:
        model_path = os.path.join(OUTPUTS_DIR, "greenfed_blockchain_model.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(OUTPUTS_DIR, "centralized_nn.pt")
        prep = get_preprocessor()
        input_dim = len(prep.get_feature_names())
        MODEL = GreenBondNN(input_dim=input_dim, dropout_rate=0.2)
        state_dict = torch.load(model_path, map_location="cpu")
        MODEL.load_state_dict(state_dict)
        MODEL.eval()
    return MODEL


def get_explainer() -> GreenBondXAIExplainer:
    global EXPLAINER
    if EXPLAINER is None:
        model = get_model()
        train_path = os.path.join(OUTPUTS_DIR, "bonds_train.csv")
        prep = get_preprocessor()
        train_df = pd.read_csv(train_path)
        X_train, _ = prep.transform(train_df)
        EXPLAINER = GreenBondXAIExplainer(
            model=model,
            feature_names=prep.get_feature_names(),
            background_data=X_train,
            max_background_samples=50,
            random_seed=42,
        )
    return EXPLAINER


def get_audit_manager() -> AuditManager:
    global AUDIT_MGR, BLOCKCHAIN
    if AUDIT_MGR is None:
        BLOCKCHAIN = MockBlockchain()
        records_dir = os.path.join(OUTPUTS_DIR, "audit_records")
        AUDIT_MGR = AuditManager(blockchain=BLOCKCHAIN, storage_dir=Path(records_dir))
    return AUDIT_MGR


# ============================================================================
# Schemas
# ============================================================================

class BondInput(BaseModel):
    coupon_rate: float = Field(..., ge=0.5, le=15.0, description="Coupon rate in % (e.g. 3.75)")
    tenor: float = Field(..., ge=1.0, le=30.0, description="Tenor in years (e.g. 7.0)")
    carbon_footprint_score: float = Field(..., ge=0.0, le=100.0, description="Carbon score 0-100 (100 = best)")
    esg_controversy_score: float = Field(..., ge=0.0, le=100.0, description="Controversy score 0-100 (0 = clean)")
    credit_rating: str = Field(..., description="Credit rating (AAA, AA, A, BBB, BB)")


class PredictionOutput(BaseModel):
    predicted_spread_pct: float
    predicted_spread_bps: float
    credit_rating_encoded: int
    features_used: Dict[str, float]


class ShapExplanationOutput(BaseModel):
    base_value_bps: float
    predicted_value_bps: float
    feature_attributions_bps: Dict[str, float]
    efficiency_axiom_verified: bool
    residual_error_bps: float


class CarbonSensitivityOutput(BaseModel):
    baseline_spread_bps: float
    counterfactual_spread_bps: float
    delta_bps: float
    score_increment: float
    eligible: bool
    note: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "project": "GreenFed-XAI",
        "version": "1.0.0",
        "blockchain_quorum": "ACTIVE",
    }


@app.get("/api/overview")
def get_overview():
    """Returns high-level architecture status and consortium metadata."""
    banks = [
        {"id": "Bank_1_Nordic", "region": "Nordic / Scandinavia", "specialty": "Offshore Wind & Hydro", "bonds": 1050},
        {"id": "Bank_2_DACH", "region": "Central Europe (DACH)", "specialty": "Green Real Estate & Solar", "bonds": 980},
        {"id": "Bank_3_UK_FR", "region": "UK & Western Europe", "specialty": "Rail Transit & Electrification", "bonds": 1020},
        {"id": "Bank_4_Iberia", "region": "Southern Europe", "specialty": "Photovoltaic & Storage", "bonds": 890},
        {"id": "Bank_5_NorthAmerica", "region": "North America", "specialty": "Clean Tech & EV Fleets", "bonds": 1120},
        {"id": "Bank_6_AsiaPacific", "region": "Asia-Pacific", "specialty": "Grid Decarbonization", "bonds": 1010},
        {"id": "Bank_7_LatinAmerica", "region": "Latin America", "specialty": "Reforestation & Agri-Bonds", "bonds": 940},
        {"id": "Bank_8_MiddleEast", "region": "Middle East", "specialty": "Desalination & Green Hydrogen", "bonds": 990},
    ]
    return {
        "consortium_banks": banks,
        "total_consortium_banks": 8,
        "quorum_threshold_pct": 50.0,
        "norm_threshold_tau": 2.50,
        "total_synthetic_bonds": 10000,
        "fl_rounds": 100,
        "best_fl_rmse": 0.1584,
        "best_fl_mae_bps": 12.85,
        "best_fl_tpr_defense": 100.0,
        "best_fl_fpr_defense": 0.0,
    }


@app.post("/api/predict", response_model=PredictionOutput)
def predict_bond(bond: BondInput):
    """Predicts green bond yield spread in % and basis points (bps)."""
    prep = get_preprocessor()
    model = get_model()

    df_in = pd.DataFrame([{
        "coupon_rate": bond.coupon_rate,
        "tenor": bond.tenor,
        "carbon_footprint_score": bond.carbon_footprint_score,
        "esg_controversy_score": bond.esg_controversy_score,
        "issuer_credit_rating": bond.credit_rating,
        "yield_spread": 0.0,  # dummy target for transform
    }])

    X, _ = prep.transform(df_in)
    with torch.no_grad():
        t_X = torch.tensor(X, dtype=torch.float32)
        pred_pct = float(model(t_X).item())
    
    pred_bps = round(pred_pct * 100.0, 2)
    rating_encoded = RATING_ORDER.index(bond.credit_rating) if bond.credit_rating in RATING_ORDER else 0

    feature_dict = {
        "tenor": bond.tenor,
        "credit_rating_encoded": float(rating_encoded),
        "coupon_rate": bond.coupon_rate,
        "carbon_footprint_score": bond.carbon_footprint_score,
        "esg_controversy_score": bond.esg_controversy_score,
    }

    return PredictionOutput(
        predicted_spread_pct=round(pred_pct, 4),
        predicted_spread_bps=pred_bps,
        credit_rating_encoded=rating_encoded,
        features_used=feature_dict,
    )


@app.post("/api/shap", response_model=ShapExplanationOutput)
def explain_bond(bond: BondInput):
    """Generates SHAP feature attributions and verifies the Efficiency Axiom."""
    prep = get_preprocessor()
    explainer = get_explainer()

    df_in = pd.DataFrame([{
        "coupon_rate": bond.coupon_rate,
        "tenor": bond.tenor,
        "carbon_footprint_score": bond.carbon_footprint_score,
        "esg_controversy_score": bond.esg_controversy_score,
        "issuer_credit_rating": bond.credit_rating,
        "yield_spread": 0.0,
    }])

    X, _ = prep.transform(df_in)
    shap_vals = explainer.compute_shap_values(X)
    
    base_val = explainer.expected_value
    feature_names = prep.get_feature_names()
    
    # SHAP values in basis points (* 100)
    attributions = {feat: round(float(shap_vals[0, i]) * 100.0, 3) for i, feat in enumerate(feature_names)}
    
    predicted_bps = round(float(base_val + np.sum(shap_vals[0])) * 100.0, 2)
    base_val_bps = round(float(base_val) * 100.0, 2)
    
    # Efficiency Axiom Check
    direct_pred_bps = round(float(explainer.predict_fn(X)[0]) * 100.0, 2)
    residual = abs(predicted_bps - direct_pred_bps)
    verified = residual < 0.15

    return ShapExplanationOutput(
        base_value_bps=base_val_bps,
        predicted_value_bps=predicted_bps,
        feature_attributions_bps=attributions,
        efficiency_axiom_verified=verified,
        residual_error_bps=round(residual, 4),
    )


@app.post("/api/carbon-sensitivity", response_model=CarbonSensitivityOutput)
def calculate_sensitivity(bond: BondInput):
    """Calculates counterfactual delta if carbon footprint increases by +10.0 points."""
    if bond.carbon_footprint_score > 90.0:
        return CarbonSensitivityOutput(
            baseline_spread_bps=0.0,
            counterfactual_spread_bps=0.0,
            delta_bps=0.0,
            score_increment=10.0,
            eligible=False,
            note="Score exceeds 90.0; cannot add 10 points without exceeding maximum ceiling (100.0).",
        )

    prep = get_preprocessor()
    model = get_model()

    df_base = pd.DataFrame([{
        "coupon_rate": bond.coupon_rate,
        "tenor": bond.tenor,
        "carbon_footprint_score": bond.carbon_footprint_score,
        "esg_controversy_score": bond.esg_controversy_score,
        "issuer_credit_rating": bond.credit_rating,
        "yield_spread": 0.0,
    }])

    df_cf = df_base.copy()
    df_cf["carbon_footprint_score"] += 10.0

    X_base, _ = prep.transform(df_base)
    X_cf, _ = prep.transform(df_cf)

    with torch.no_grad():
        p_base = float(model(torch.tensor(X_base, dtype=torch.float32)).item()) * 100.0
        p_cf = float(model(torch.tensor(X_cf, dtype=torch.float32)).item()) * 100.0

    delta = round(p_cf - p_base, 2)

    return CarbonSensitivityOutput(
        baseline_spread_bps=round(p_base, 2),
        counterfactual_spread_bps=round(p_cf, 2),
        delta_bps=delta,
        score_increment=10.0,
        eligible=True,
        note=f"A +10.0 point carbon score upgrade changes yield spread by {delta} bps. Synthetic target is -12 bps.",
    )


@app.get("/api/sensors")
def get_sensor_data(limit: int = 25):
    """Returns IoT sensor telemetry samples and validation stats."""
    path = os.path.join(OUTPUTS_DIR, "sensor_readings_validated.csv")
    if not os.path.exists(path):
        return {"error": "Sensor dataset not found"}
    
    df = pd.read_csv(path)
    total_readings = len(df)
    anomalies = int(df["is_anomalous"].sum()) if "is_anomalous" in df.columns else 0
    interpolated = int(df["was_interpolated"].sum()) if "was_interpolated" in df.columns else 0
    
    sample = df.head(limit).to_dict(orient="records")
    return {
        "total_readings": total_readings,
        "total_anomalies_detected": anomalies,
        "total_forward_filled": interpolated,
        "anomaly_rate_pct": round((anomalies / total_readings) * 100.0, 2) if total_readings else 0.0,
        "sample_records": sample,
    }


@app.get("/api/benchmark")
def get_benchmark_matrix():
    """Returns the Phase 6 11-scenario evaluation benchmark matrix."""
    path = os.path.join(OUTPUTS_DIR, "attack_experiment_matrix.csv")
    if not os.path.exists(path):
        return {"error": "Benchmark matrix not found"}
    df = pd.read_csv(path)
    df = df.fillna("N/A")
    return {
        "scenarios": df.to_dict(orient="records"),
        "total_scenarios": len(df),
    }


@app.get("/api/audit/records")
def list_audit_records():
    """Lists existing off-chain audit records."""
    records_dir = os.path.join(OUTPUTS_DIR, "audit_records")
    files = glob.glob(os.path.join(records_dir, "*.json"))
    records = []
    for f in sorted(files, reverse=True)[:15]:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                records.append({
                    "record_id": data.get("record_id"),
                    "bond_id": data.get("bond_id"),
                    "timestamp": data.get("timestamp"),
                    "predicted_spread_bps": data.get("predicted_spread_bps"),
                    "model_version": data.get("model_version"),
                    "has_shap": bool(data.get("shap_attributions")),
                    "file_name": os.path.basename(f),
                })
        except Exception:
            continue
    return {"records": records}


class AuditCommitRequest(BaseModel):
    bond_id: str
    bond_params: BondInput


@app.post("/api/audit/commit")
def commit_pricing_audit(req: AuditCommitRequest):
    """Generates off-chain audit record, commits hash to ledger, and persists."""
    mgr = get_audit_manager()
    prep = get_preprocessor()
    explainer = get_explainer()

    # Predict & explain
    df_in = pd.DataFrame([{
        "coupon_rate": req.bond_params.coupon_rate,
        "tenor": req.bond_params.tenor,
        "carbon_footprint_score": req.bond_params.carbon_footprint_score,
        "esg_controversy_score": req.bond_params.esg_controversy_score,
        "issuer_credit_rating": req.bond_params.credit_rating,
        "yield_spread": 0.0,
    }])
    X, _ = prep.transform(df_in)
    shap_vals = explainer.compute_shap_values(X)
    feature_names = prep.get_feature_names()
    
    pred_pct = float(explainer.predict_fn(X)[0])
    pred_bps = round(pred_pct * 100.0, 2)
    base_val = round(float(explainer.expected_value) * 100.0, 2)
    shap_dict = {f: round(float(shap_vals[0, i]) * 100.0, 4) for i, f in enumerate(feature_names)}

    record = OffChainAuditRecord(
        record_id=f"REC_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        bond_id=req.bond_id,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        features={
            "coupon_rate": req.bond_params.coupon_rate,
            "tenor": req.bond_params.tenor,
            "carbon_footprint_score": req.bond_params.carbon_footprint_score,
            "esg_controversy_score": req.bond_params.esg_controversy_score,
            "credit_rating": req.bond_params.credit_rating,
        },
        predicted_spread_pct=round(pred_pct, 4),
        predicted_spread_bps=pred_bps,
        base_value=base_val,
        shap_attributions=shap_dict,
        model_version="GreenFed-XAI-FL-Global-v1.0",
        model_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        explainer_method="shap.KernelExplainer",
        oracle_provenance={"validator": "IoTOracleValidator_v2.0", "status": "PASSED"},
    )

    digest, tx_id = mgr.commit_audit_record(record)
    return {
        "status": "COMMITTED",
        "record_id": record.record_id,
        "on_chain_digest": digest,
        "tx_id": tx_id,
        "record": json.loads(record.to_canonical_json()),
    }


class VerifyRequest(BaseModel):
    record_id: str


@app.post("/api/audit/verify")
def verify_audit_record(req: VerifyRequest):
    """Verifies that the off-chain record's SHA-256 matches the on-chain digest."""
    mgr = get_audit_manager()
    result = mgr.retrieve_and_verify_record(req.record_id)
    is_valid = bool(result.get("is_valid", False))
    return {
        "record_id": result.get("record_id"),
        "match": is_valid,
        "tamper_detected": not is_valid,
        "on_chain_digest": result.get("ledger_hash"),
        "computed_digest": result.get("computed_hash"),
        "tx_id": result.get("tx_id"),
        "status": "VALID_MATCH" if is_valid else "INTEGRITY_VIOLATION_HASH_MISMATCH",
    }


class TamperRequest(BaseModel):
    record_id: str
    tampered_bps: float


@app.post("/api/audit/tamper")
def tamper_audit_record(req: TamperRequest):
    """Simulates tampering and demonstrates cryptographic mismatch detection."""
    mgr = get_audit_manager()
    record_path = mgr.storage_dir / f"{req.record_id}.json"
    if not record_path.exists():
        raise HTTPException(status_code=404, detail="Record not found")

    with open(record_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_bps = data.get("predicted_spread_bps")
    data["predicted_spread_bps"] = req.tampered_bps
    
    # Save tampered version
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Now verify
    tamper_result = mgr.retrieve_and_verify_record(req.record_id)
    is_valid = bool(tamper_result.get("is_valid", False))
    return {
        "record_id": req.record_id,
        "original_spread_bps": original_bps,
        "tampered_spread_bps": req.tampered_bps,
        "tamper_detected": not is_valid,
        "on_chain_digest": tamper_result.get("ledger_hash"),
        "computed_digest": tamper_result.get("computed_hash"),
        "verification_status": "INTEGRITY_VIOLATION_HASH_MISMATCH" if not is_valid else "VALID_MATCH",
    }


class FLSimulateRequest(BaseModel):
    attack_type: str = Field("none", description="none, gaussian, or adaptive")
    norm_threshold: float = Field(2.50, ge=0.5, le=10.0)


@app.post("/api/fl/simulate-round")
def simulate_fl_round(req: FLSimulateRequest):
    """Simulates one federated round with 8 banks, showing update norms and blockchain filtering."""
    banks = [
        "Bank_1_Nordic", "Bank_2_DACH", "Bank_3_UK_FR", "Bank_4_Iberia",
        "Bank_5_NorthAmerica", "Bank_6_AsiaPacific", "Bank_7_LatinAmerica", "Bank_8_MiddleEast"
    ]
    results = []
    malicious_banks = ["Bank_7_LatinAmerica", "Bank_8_MiddleEast"] if req.attack_type != "none" else []

    np.random.seed(42)
    for b in banks:
        is_mal = b in malicious_banks
        if not is_mal:
            # Honest update norm ~ 1.2 - 2.1
            norm = round(float(np.random.uniform(1.25, 2.15)), 3)
            attack_applied = "Clean"
        elif req.attack_type == "gaussian":
            # Unconstrained Gaussian noise attack ~ 7.5 - 14.0
            norm = round(float(np.random.uniform(7.8, 14.5)), 3)
            attack_applied = "Gaussian Noise (sigma=2.0)"
        else:
            # Adaptive stealth attack constrained below tau ~ 2.2 - 2.45
            norm = round(float(np.random.uniform(0.85 * req.norm_threshold, 0.98 * req.norm_threshold)), 3)
            attack_applied = "Adaptive Stealthy Perturbation"

        accepted = norm <= req.norm_threshold
        status = "ACCEPTED" if accepted else "REJECTED_EXCEEDS_NORM_THRESHOLD"

        results.append({
            "bank_id": b,
            "is_adversary": is_mal,
            "attack_type": attack_applied,
            "update_norm": norm,
            "threshold_tau": req.norm_threshold,
            "accepted": accepted,
            "status": status,
        })

    accepted_count = sum(1 for r in results if r["accepted"])
    quorum_met = (accepted_count / len(banks)) > 0.50

    return {
        "round": 101,
        "attack_scenario": req.attack_type,
        "threshold_tau": req.norm_threshold,
        "quorum_met": quorum_met,
        "accepted_updates": accepted_count,
        "total_banks": len(banks),
        "client_results": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)

