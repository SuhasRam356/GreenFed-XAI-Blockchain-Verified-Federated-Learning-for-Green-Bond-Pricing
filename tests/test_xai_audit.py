"""
Unit and Integration Tests for Phase 7: Explainable AI (SHAP) & Audit Records
=============================================================================
"""

import json
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
import pytest
import torch

from audit.audit_manager import AuditManager, OffChainAuditRecord
from chain.mock_ledger import MockBlockchain
from models.nn_model import GreenBondNN
from xai.run_xai_pipeline import run_xai_audit_pipeline
from xai.shap_explainer import GreenBondXAIExplainer


@pytest.fixture
def mock_xai_env():
    """Create a minimal synthetic setup for testing XAI and audit features."""
    rng = np.random.default_rng(42)
    n_features = 8
    n_train = 100
    n_test = 20

    feature_names = [
        "credit_rating_num",
        "tenor_years",
        "coupon_pct",
        "carbon_score",
        "esg_controversy_score",
        "market_interest_rate",
        "liquidity_factor",
        "volatility_index",
    ]

    X_train = rng.normal(0, 1, size=(n_train, n_features)).astype(np.float32)
    y_train = (1.5 + 0.2 * X_train[:, 0] - 0.12 * X_train[:, 3]).astype(np.float32)

    X_test = rng.normal(0, 1, size=(n_test, n_features)).astype(np.float32)
    y_test = (1.5 + 0.2 * X_test[:, 0] - 0.12 * X_test[:, 3]).astype(np.float32)

    # Test DataFrame with raw features
    test_df = pd.DataFrame({
        "bond_id": [f"BOND_{i:04d}" for i in range(n_test)],
        "credit_rating": ["AA"] * (n_test // 2) + ["BBB"] * (n_test - n_test // 2),
        "credit_rating_num": X_test[:, 0],
        "tenor_years": rng.uniform(3, 10, size=n_test),
        "coupon_pct": rng.uniform(2, 5, size=n_test),
        "carbon_score": rng.uniform(50, 95, size=n_test),  # Some > 90 to test boundary
        "esg_controversy_score": rng.uniform(1, 4, size=n_test),
        "yield_spread": y_test,
    })

    model = GreenBondNN(input_dim=n_features, dropout_rate=0.0)

    # Mock preprocessor
    class MockPreprocessor:
        def __init__(self):
            self.feature_names = feature_names

        def transform(self, df: pd.DataFrame) -> np.ndarray:
            # Simple identity projection for testing
            arr = np.zeros((len(df), n_features), dtype=np.float32)
            arr[:, 0] = df["credit_rating_num"].values if "credit_rating_num" in df else 0.0
            arr[:, 3] = (df["carbon_score"].values - 75.0) / 15.0
            return arr

    return model, feature_names, X_train, X_test, test_df, MockPreprocessor()


class TestGreenBondXAIExplainer:
    """Tests for SHAP computation, additivity, and sensitivity analysis."""

    def test_explainer_initialization_background_subsampling(self, mock_xai_env):
        model, feature_names, X_train, _, _, _ = mock_xai_env
        explainer = GreenBondXAIExplainer(
            model=model,
            feature_names=feature_names,
            background_data=X_train,
            max_background_samples=25,
            random_seed=42,
        )
        assert len(explainer.background) == 25
        assert explainer.background.shape[1] == len(feature_names)
        assert isinstance(explainer.expected_value, float)

    def test_compute_shap_values_shape(self, mock_xai_env):
        model, feature_names, X_train, X_test, _, _ = mock_xai_env
        explainer = GreenBondXAIExplainer(
            model=model,
            feature_names=feature_names,
            background_data=X_train,
            max_background_samples=20,
        )
        # Evaluate on 5 test samples
        shap_vals = explainer.compute_shap_values(X_test[:5], nsamples=50)
        assert shap_vals.shape == (5, len(feature_names))
        assert not np.isnan(shap_vals).any()

    def test_verify_additivity_axiom(self, mock_xai_env):
        model, feature_names, X_train, X_test, _, _ = mock_xai_env
        explainer = GreenBondXAIExplainer(
            model=model,
            feature_names=feature_names,
            background_data=X_train,
            max_background_samples=20,
        )
        X_sample = X_test[:3]
        shap_vals = explainer.compute_shap_values(X_sample, nsamples=80)
        res = explainer.verify_additivity(X_sample, shap_vals, tolerance=0.25)

        assert "passed" in res
        assert "max_discrepancy" in res
        assert res["max_discrepancy"] >= 0.0

    def test_compute_feature_importance_ranking(self, mock_xai_env):
        model, feature_names, X_train, X_test, _, _ = mock_xai_env
        explainer = GreenBondXAIExplainer(
            model=model,
            feature_names=feature_names,
            background_data=X_train,
            max_background_samples=20,
        )
        shap_vals = explainer.compute_shap_values(X_test[:4], nsamples=50)
        imp_df = explainer.compute_feature_importance(shap_vals)

        assert len(imp_df) == len(feature_names)
        assert list(imp_df.columns) == ["feature", "mean_abs_shap", "importance_pct"]
        assert imp_df["importance_pct"].sum() == pytest.approx(100.0, rel=1e-2)
        # Must be sorted descending
        assert imp_df["mean_abs_shap"].is_monotonic_decreasing

    def test_carbon_score_sensitivity_eligible_filtering(self, mock_xai_env):
        model, feature_names, X_train, X_test, test_df, preprocessor = mock_xai_env
        explainer = GreenBondXAIExplainer(
            model=model,
            feature_names=feature_names,
            background_data=X_train,
            max_background_samples=20,
        )
        sens = explainer.evaluate_carbon_score_sensitivity(
            test_df=test_df,
            X_test=X_test,
            preprocessor=preprocessor,
            score_increase=10.0,
        )

        assert sens["n_eligible"] > 0
        assert sens["n_eligible"] <= len(test_df)
        assert "mean_delta_bps" in sens
        assert "median_delta_bps" in sens
        assert sens["score_increase"] == 10.0

    def test_generate_xai_plots(self, mock_xai_env, tmp_path):
        model, feature_names, X_train, X_test, test_df, _ = mock_xai_env
        explainer = GreenBondXAIExplainer(
            model=model,
            feature_names=feature_names,
            background_data=X_train,
            max_background_samples=20,
        )
        shap_vals = explainer.compute_shap_values(X_test[:4], nsamples=50)
        plots = explainer.generate_xai_plots(
            shap_values=shap_vals,
            X_eval=X_test[:4],
            eval_df=test_df.iloc[:4],
            output_dir=tmp_path,
            num_individual_samples=2,
        )
        assert len(plots) == 2
        for p in plots:
            assert p.exists()
            assert p.stat().st_size > 1000  # Non-empty image file


class TestAuditManager:
    """Tests for structured audit records and blockchain anchoring."""

    def test_off_chain_audit_record_canonical_json_determinism(self):
        rec1 = OffChainAuditRecord(
            record_id="REC_001",
            bond_id="BOND_001",
            timestamp="2026-11-14T00:00:00Z",
            features={"coupon": 3.5, "rating": "AAA"},
            predicted_spread_pct=1.45,
            predicted_spread_bps=145.0,
            base_value=1.50,
            shap_attributions={"coupon": -0.05},
            model_version="1.0.0",
            model_hash="abc123hash",
            explainer_method="KernelExplainer",
            oracle_provenance={"validated": True},
        )
        rec2 = OffChainAuditRecord(
            record_id="REC_001",
            bond_id="BOND_001",
            timestamp="2026-11-14T00:00:00Z",
            features={"rating": "AAA", "coupon": 3.5},  # inverted dictionary order
            predicted_spread_pct=1.45,
            predicted_spread_bps=145.0,
            base_value=1.50,
            shap_attributions={"coupon": -0.05},
            model_version="1.0.0",
            model_hash="abc123hash",
            explainer_method="KernelExplainer",
            oracle_provenance={"validated": True},
        )
        # Canonical JSON must sort keys identically
        assert rec1.to_canonical_json() == rec2.to_canonical_json()
        assert rec1.compute_sha256() == rec2.compute_sha256()
        assert len(rec1.compute_sha256()) == 64

    def test_commit_and_retrieve_valid_record(self, tmp_path):
        blockchain = MockBlockchain(mining_difficulty=1)
        manager = AuditManager(blockchain=blockchain, storage_dir=tmp_path)

        rec = OffChainAuditRecord(
            record_id="REC_VALID",
            bond_id="BOND_100",
            timestamp="2026-11-14T00:00:00Z",
            features={"carbon_score": 85.0},
            predicted_spread_pct=1.20,
            predicted_spread_bps=120.0,
            base_value=1.50,
            shap_attributions={"carbon_score": -0.30},
            model_version="1.0.0",
            model_hash="hash123",
            explainer_method="KernelExplainer",
            oracle_provenance={"oracle_passed": True},
        )

        audit_hash, tx_id = manager.commit_audit_record(rec)
        assert len(audit_hash) == 64
        assert (tmp_path / "REC_VALID.json").exists()

        verification = manager.retrieve_and_verify_record("REC_VALID")
        assert verification["is_valid"] is True
        assert verification["computed_hash"] == audit_hash
        assert verification["ledger_hash"] == audit_hash

    def test_tamper_detection_on_prediction(self, tmp_path):
        blockchain = MockBlockchain(mining_difficulty=1)
        manager = AuditManager(blockchain=blockchain, storage_dir=tmp_path)

        rec = OffChainAuditRecord(
            record_id="REC_TAMPER",
            bond_id="BOND_200",
            timestamp="2026-11-14T00:00:00Z",
            features={"carbon_score": 85.0},
            predicted_spread_pct=1.20,
            predicted_spread_bps=120.0,
            base_value=1.50,
            shap_attributions={"carbon_score": -0.30},
            model_version="1.0.0",
            model_hash="hash123",
            explainer_method="KernelExplainer",
            oracle_provenance={"oracle_passed": True},
        )
        manager.commit_audit_record(rec)

        # Demonstrate tampering on predicted_spread_bps: change 120.0 -> 5.0
        tamper_res = manager.demonstrate_tampering(
            record_id="REC_TAMPER",
            tampered_field="predicted_spread_bps",
            tampered_value=5.0,
        )

        assert tamper_res["tamper_detected"] is True
        assert tamper_res["is_valid"] is False
        assert tamper_res["computed_hash"] != tamper_res["ledger_hash"]

    def test_tamper_detection_on_shap_attributions(self, tmp_path):
        blockchain = MockBlockchain(mining_difficulty=1)
        manager = AuditManager(blockchain=blockchain, storage_dir=tmp_path)

        rec = OffChainAuditRecord(
            record_id="REC_SHAP_TAMPER",
            bond_id="BOND_300",
            timestamp="2026-11-14T00:00:00Z",
            features={"carbon_score": 85.0},
            predicted_spread_pct=1.20,
            predicted_spread_bps=120.0,
            base_value=1.50,
            shap_attributions={"carbon_score": -0.30},
            model_version="1.0.0",
            model_hash="hash123",
            explainer_method="KernelExplainer",
            oracle_provenance={"oracle_passed": True},
        )
        manager.commit_audit_record(rec)

        # Modify SHAP attribution dictionary
        tamper_res = manager.demonstrate_tampering(
            record_id="REC_SHAP_TAMPER",
            tampered_field="shap_attributions",
            tampered_value={"carbon_score": +0.50},  # Falsified positive impact
        )
        assert tamper_res["tamper_detected"] is True
        assert tamper_res["is_valid"] is False


class TestEndToEndPipeline:
    """Test full execution of run_xai_audit_pipeline."""

    def test_run_xai_audit_pipeline_quick(self, tmp_path):
        # Point pipeline to tmp_path with quick mode
        summary = run_xai_audit_pipeline(quick=True, num_eval_samples=4, output_dir=tmp_path)

        assert "model_hash" in summary
        assert "additivity" in summary
        assert "top_features" in summary
        assert len(summary["top_features"]) > 0
        assert summary["anchored_records_count"] > 0
        assert summary["tamper_demonstration"]["tamper_detected"] is True
        assert (tmp_path / "shap_feature_importance.png").exists()
        assert (tmp_path / "shap_individual_waterfall.png").exists()
        assert (tmp_path / "shap_feature_importance.csv").exists()
        assert (tmp_path / "carbon_score_sensitivity.json").exists()

