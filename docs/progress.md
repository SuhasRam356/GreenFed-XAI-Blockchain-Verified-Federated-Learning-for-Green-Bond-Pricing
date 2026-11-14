# Project Implementation Progress & Milestones

| Master Phase | Milestone Name | Status | Key Deliverables & Artifacts |
|:---|:---|:---:|:---|
| **Phase 0** | Scope, Foundations, & Setup | Completed | Environment locked, dependencies pinned, assumptions cataloged. |
| **Phase 1** | Synthetic Bond Data & Features | Completed | `data/generate_bonds.py`, 10,000 bonds, train/val/test splits. |
| **Phase 2** | IoT Sensor Simulation & Oracle Gate | Completed | `iot/simulate_sensors.py`, `iot/oracle_validator.py`, forward-fill fallback. |
| **Phase 3** | Centralized Baseline Models | Completed | `models/train_centralized.py` (Linear Regression, XGBoost, NN with MAE in bps & near-zero MAPE policy). |
| **Phase 4** | Federated Learning Reference | Completed | `fl/dataset_partitioner.py`, `fl/client.py`, `fl/strategy_fedavg.py`, `fl/server.py`. |
| **Phase 5** | Blockchain Verification & Norm Filter | Completed | `chain/mock_ledger.py`, `fl/strategy_blockchain.py`, Table I baseline update. |
| **Phase 6** | Attack Scenarios & Evaluation Matrix | Completed | `attacks/fl_attacks.py`, `experiments/run_matrix.py`, complete 11-scenario benchmark, $\tau$-sensitivity sweep. |
| **Phase 7** | Explainable AI (SHAP) & Audit Trail | Completed | `xai/shap_explainer.py`, `audit/audit_manager.py`, `tests/test_xai_audit.py`, feature importance rankings, sensitivity analysis, immutable audit logs. |
| **Phase 8** | Hyperledger Fabric Integration | Completed (Code) | `chain/chaincode/smart_contract.go` (Go 1.27 compiled and tested). |
| **Phase 9** | Interactive Dashboard & Final Handover | Next | Streamlit demo, final IEEE academic report, viva preparation guide. |

---

## Phase 7 Explainable AI & Off-Chain Audit Trail

### Key Results Summary
- **SHAP Explanation Engine**: `GreenBondXAIExplainer` utilizes `shap.KernelExplainer` on the PyTorch neural network baseline with a 100-sample k-means/random representative background partition. Additivity axiom verified with maximum residual error $< 10^{-6}$.
- **Global Feature Attributions**:
  1. `tenor`: 22.29% (mean |SHAP| = 0.2396)
  2. `credit_rating_encoded`: 22.20% (mean |SHAP| = 0.2385)
  3. `carbon_footprint_score`: 19.36% (mean |SHAP| = 0.2081)
  4. `esg_controversy_score`: 18.20% (mean |SHAP| = 0.1956)
  5. `coupon_rate`: 17.95% (mean |SHAP| = 0.1929)
- **10-Point Carbon Score Model Sensitivity**: Evaluated across 1,345 eligible test bonds ($C_t \le 90$); observed mean yield spread delta is $-2.45\text{ bps}$ (median $-2.53\text{ bps}$, std $15.08\text{ bps}$, synthetic generator target $-12\text{ bps}$). Explicitly documented as model sensitivity under synthetic generation, not a causal macroeconomic guarantee.
- **Off-Chain Audit Trail & Tamper Detection**:
  - Full pricing audit record serialized to deterministic, canonical JSON format with sorted keys and normalized floating-point precision.
  - 32-byte SHA-256 cryptographic digest committed on-chain to `MockBlockchain`.
  - Cryptographic verification re-computes digest and validates match with ledger.
  - Zero-tolerance tamper detection demonstrated: any 1-bit alteration to features, predictions, or attributions produces digest mismatch (`tamper_detected == True`).

---

## Phase 6 Benchmark Results Matrix

### Single-Run Benchmark (100-Round Converged Matrix)

| Scenario ID | Scenario Name | Threat Model / Defense | RMSE (%) | MAE (bps) | MAPE (%) | TPR (%) | FPR (%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | Centralized Clean Reference | None (Centralized Upper Bound) | 0.0868 | 6.93 | 2.72% | N/A | N/A |
| **2** | Clean FedAvg (Weighted) | Sample-Weighted (No Blockchain) | 0.1506 | 12.12 | 4.44% | N/A | N/A |
| **3** | Clean FedAvg (Equal) | Uniform Client Averaging | 0.1506 | 12.12 | 4.44% | N/A | N/A |
| **4** | Clean FedAvg + Hash Logging | Blockchain Hash Audit (No Norm Filter) | 0.1506 | 12.12 | 4.44% | N/A | 0.0% |
| **5** | Clean FedAvg + Norm Filter | Full GreenFed-XAI ($\tau = 2.50$) | 0.1506 | 12.12 | 4.44% | N/A | 0.0% |
| **6** | Attacked FedAvg (No Defense) | 25% Gaussian Noise ($\sigma = 2.0$) | 11.8139 | 455.82 | 159.66% | 0.0% | 0.0% |
| **7** | Attacked FedAvg + Hash Only | Hash Check Only (No Norm Filter) | 11.8139 | 455.82 | 159.66% | 0.0% | 0.0% |
| **8** | Attacked FedAvg + Norm Filter | **GreenFed-XAI ($\tau = 2.50$)** | **0.1584** | **12.85** | **4.69%** | **100.0%** | **0.0%** |
| **9** | Attacked FedAvg + Adaptive | Constrained Perturbation ($\|\Delta w\| \le 0.95\tau$) | 0.1800 | 14.64 | 5.37% | 0.0% | 0.0% |
| **10** | Sensor Attack (No Oracle) | Raw Greenwashed Telemetry | 0.1506 | 12.12 | 4.44% | N/A | N/A |
| **11** | Sensor Attack (With Oracle) | IoT Oracle Validated + Mitigated | 0.1445 | 11.61 | 4.29% | N/A | N/A |

### Multi-Seed Empirical Verification (Mean ± Std across Seeds)

| Scenario ID | Scenario Name | Attack Type | RMSE (%) | MAE (bps) | MAPE (%) | TPR (%) | FPR (%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | Centralized Clean Reference | None | 0.0973 ± 0.0112 | 7.71 ± 0.78 | 3.00 ± 0.30% | N/A | N/A |
| **2** | Clean FedAvg (Weighted) | None | 0.7452 ± 0.3259 | 60.60 ± 26.21 | 24.57 ± 10.71% | N/A | N/A |
| **3** | Clean FedAvg (Equal) | None | 0.7452 ± 0.3259 | 60.59 ± 26.21 | 24.57 ± 10.71% | N/A | N/A |
| **4** | Clean FedAvg + Hash Logging | None | 0.7452 ± 0.3259 | 60.60 ± 26.21 | 24.57 ± 10.71% | N/A | 0.0% |
| **5** | Clean FedAvg + Norm Filter | None | 0.7452 ± 0.3259 | 60.60 ± 26.21 | 24.57 ± 10.71% | N/A | 0.0% |
| **6** | Attacked FedAvg (No Defense) | Gaussian Noise ($\sigma = 2.0$) | 30.9855 ± 13.7527 | 2221.71 ± 1363.53 | 837.59 ± 628.59% | 0.0% | 0.0% |
| **7** | Attacked FedAvg + Hash Only | Gaussian Noise ($\sigma = 2.0$) | 30.9855 ± 13.7527 | 2221.71 ± 1363.53 | 837.59 ± 628.59% | 0.0% | 0.0% |
| **8** | Attacked FedAvg + Norm Filter | Gaussian Noise ($\tau = 2.50$) | **0.7490 ± 0.3170** | **60.95 ± 25.44** | **24.66 ± 10.38%** | **100.0%** | **0.0%** |
| **9** | Attacked FedAvg + Adaptive | Adaptive Perturbation | 1.1706 ± 0.5227 | 98.89 ± 46.90 | 35.35 ± 12.50% | 0.0% | 0.0% |
| **10** | Sensor Attack (No Oracle) | Biased Telemetry | 0.3600 ± 0.0111 | 29.20 ± 1.03 | 11.26 ± 0.32% | N/A | N/A |
| **11** | Sensor Attack (With Oracle) | Validated Telemetry | 0.3600 ± 0.0110 | 29.20 ± 1.03 | 11.26 ± 0.32% | N/A | N/A |

