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
| **Phase 7** | Explainable AI (SHAP) & Audit Trail | Next | `xai/`, SHAP feature attributions, tamper-proof off-chain audit logs. |
| **Phase 8** | Hyperledger Fabric Integration | Completed (Code) | `chain/chaincode/smart_contract.go` (Go 1.27 compiled and tested). |
| **Phase 9** | Interactive Dashboard & Final Handover | Scheduled | Streamlit demo, final IEEE academic report, viva preparation guide. |

---

## Phase 6 Benchmark Results Matrix

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

