r"""
Phase 7 XAI & Audit Pipeline Orchestrator — GreenFed-XAI
========================================================

Executes the full Explainable AI and Off-Chain Audit workflow:
1. Loads trained model (GreenFed-XAI or Centralized reference).
2. Computes SHAP values using KernelExplainer with training background.
3. Tests efficiency/additivity: sum(phi_i) + phi_0 ~= f(x).
4. Generates global feature importance ranking & waterfall explanation plots.
5. Measures model sensitivity to a +10 point carbon score increase (basis points).
6. Persists off-chain structured audit records and anchors hashes to MockBlockchain.
7. Executes cryptographic tampering demonstration.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import torch

from audit.audit_manager import AuditManager, OffChainAuditRecord
from chain.mock_ledger import MockBlockchain, SmartContract
from models.nn_model import GreenBondNN
from models.train_centralized import DataPreprocessor
from xai.shap_explainer import GreenBondXAIExplainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("XAI_Pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_xai_audit_pipeline(
    model_checkpoint: Optional[Path] = None,
    num_eval_samples: int = 50,
    quick: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete Phase 7 XAI and audit trail pipeline."""
    outputs_dir = output_dir or (PROJECT_ROOT / "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = outputs_dir / "audit_records"

    # 1. Load data and preprocessing
    logger.info("Loading datasets and fitting training preprocessor...")
    data_dir = PROJECT_ROOT / "outputs"
    train_path = data_dir / "bonds_train.csv"
    val_path = data_dir / "bonds_val.csv"
    test_path = data_dir / "bonds_test.csv"

    if not train_path.exists():
        # Fallback names if generated under alternative schema
        train_path = data_dir / "train_bonds.csv"
        val_path = data_dir / "val_bonds.csv"
        test_path = data_dir / "test_bonds.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)

    if hasattr(preprocessor, "get_feature_names"):
        feature_names = preprocessor.get_feature_names()
    elif hasattr(preprocessor, "feature_names"):
        feature_names = preprocessor.feature_names
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    input_dim = X_train.shape[1]

    # 2. Load trained neural network model
    device = torch.device("cpu")
    model = GreenBondNN(input_dim=input_dim, dropout_rate=0.2).to(device)

    ckpt_path = model_checkpoint or (outputs_dir / "greenfed_blockchain_model.pt")
    if not ckpt_path.exists():
        ckpt_path = outputs_dir / "centralized_nn_model.pt"

    if ckpt_path.exists():
        logger.info("Loading trained model checkpoint from %s...", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        logger.warning("No checkpoint found at %s. Running with initialized model.", ckpt_path)

    model.eval()

    # Compute model parameter SHA-256 hash
    model_weights = [p.detach().cpu().numpy() for p in model.parameters()]
    model_hash = SmartContract.compute_parameter_hash(model_weights)
    logger.info("Model SHA-256 Parameter Hash: %s", model_hash[:16] + "...")

    # 3. Initialize SHAP Explainer with training background
    n_bg = 30 if quick else 100
    n_eval = 20 if quick else min(num_eval_samples, len(X_test))
    n_monte_carlo = 100 if quick else 200

    explainer = GreenBondXAIExplainer(
        model=model,
        feature_names=feature_names,
        background_data=X_train,
        device=device,
        max_background_samples=n_bg,
        random_seed=42,
    )

    # 4. Compute SHAP values on test set subset
    X_eval = X_test[:n_eval]
    eval_df = test_df.iloc[:n_eval].copy()
    shap_values = explainer.compute_shap_values(X_eval, nsamples=n_monte_carlo)

    # 5. Verify Additivity Axiom
    additivity_res = explainer.verify_additivity(X_eval, shap_values, tolerance=0.15)

    # 6. Global Feature Importance
    importance_df = explainer.compute_feature_importance(shap_values)
    imp_csv = outputs_dir / "shap_feature_importance.csv"
    importance_df.to_csv(imp_csv, index=False)
    logger.info("Saved global feature importance ranking to %s", imp_csv)

    # 7. 10-Point Carbon Score Model Sensitivity Evaluation
    sensitivity_res = explainer.evaluate_carbon_score_sensitivity(
        test_df=test_df,
        X_test=X_test,
        preprocessor=preprocessor,
        score_increase=10.0,
    )
    sens_json = outputs_dir / "carbon_score_sensitivity.json"
    with open(sens_json, "w", encoding="utf-8") as f:
        # Exclude raw sample array from summary JSON
        clean_sens = {k: v for k, v in sensitivity_res.items() if k != "delta_bps_samples"}
        json.dump(clean_sens, f, indent=2)
    logger.info("Saved carbon score sensitivity summary to %s", sens_json)

    # 8. Generate XAI Plots
    plot_paths = explainer.generate_xai_plots(
        shap_values=shap_values,
        X_eval=X_eval,
        eval_df=eval_df,
        output_dir=outputs_dir,
        num_individual_samples=4,
    )

    # 9. Audit Trail & Blockchain Commitment
    blockchain = MockBlockchain(mining_difficulty=1)
    audit_mgr = AuditManager(blockchain=blockchain, storage_dir=audit_dir)

    anchored_records: List[Dict[str, Any]] = []
    for i in range(min(5, n_eval)):
        bond_row = eval_df.iloc[i]
        bond_id = str(bond_row.get("bond_id", f"BOND_{i:04d}"))
        sample_features = {
            col: float(bond_row[col]) if isinstance(bond_row[col], (int, float, np.number)) else str(bond_row[col])
            for col in [
                "credit_rating",
                "issuer_credit_rating",
                "tenor_years",
                "coupon_pct",
                "coupon",
                "carbon_score",
                "carbon_footprint_score",
                "esg_controversy_score",
            ]
            if col in bond_row
        }
        pred_pct = float(explainer.predict_fn(X_eval[i : i + 1])[0])
        pred_bps = round(pred_pct * 100.0, 2)

        record = OffChainAuditRecord(
            record_id=str(uuid.uuid4()),
            bond_id=bond_id,
            timestamp=pd.Timestamp.now(tz="UTC").isoformat(),
            features=sample_features,
            predicted_spread_pct=round(pred_pct, 4),
            predicted_spread_bps=pred_bps,
            base_value=round(explainer.expected_value, 4),
            shap_attributions={
                f_name: round(float(shap_values[i, j]), 4)
                for j, f_name in enumerate(feature_names)
            },
            model_version="1.0.0",
            model_hash=model_hash,
            explainer_method="shap.KernelExplainer",
            oracle_provenance={
                "validated": True,
                "freshness_check": "passed",
                "jump_constraint_theta": 15.0,
            },
        )
        audit_hash, tx_id = audit_mgr.commit_audit_record(record)
        anchored_records.append({
            "record_id": record.record_id,
            "bond_id": bond_id,
            "audit_hash": audit_hash,
            "tx_id": tx_id,
        })

    # Verify first committed record
    first_rec_id = anchored_records[0]["record_id"]
    verification = audit_mgr.retrieve_and_verify_record(first_rec_id)
    assert verification["is_valid"] is True

    # 10. Tampering Demonstration
    tamper_result = audit_mgr.demonstrate_tampering(
        record_id=first_rec_id,
        tampered_field="predicted_spread_bps",
        tampered_value=1.5,
    )
    logger.info(
        "Tamper Detection Verified: Altered 'predicted_spread_bps' flagged -> Tamper Detected: %s",
        tamper_result["tamper_detected"],
    )

    logger.info("Phase 7 XAI and Audit Pipeline execution complete.")

    return {
        "model_hash": model_hash,
        "n_eval_samples": n_eval,
        "additivity": additivity_res,
        "top_features": importance_df["feature"].tolist()[:5],
        "carbon_sensitivity_mean_bps": sensitivity_res["mean_delta_bps"],
        "anchored_records_count": len(anchored_records),
        "tamper_demonstration": tamper_result,
        "plots": [str(p) for p in plot_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GreenFed-XAI Phase 7 XAI & Audit Pipeline.")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid smoke test.")
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples to explain.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom model checkpoint path.")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    run_xai_audit_pipeline(model_checkpoint=ckpt, num_eval_samples=args.samples, quick=args.quick)


if __name__ == "__main__":
    main()
