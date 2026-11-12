# Project Implementation Progress & Milestones

| Master Phase | Milestone Name | Status | Key Deliverables & Artifacts |
|:---|:---|:---:|:---|
| **Phase 0** | Scope, Foundations, & Setup | Completed | Environment locked, dependencies pinned, assumptions cataloged. |
| **Phase 1** | Synthetic Bond Data & Features | Completed | `data/generate_bonds.py`, 10,000 bonds, train/val/test splits. |
| **Phase 2** | IoT Sensor Simulation & Oracle Gate | Completed | `iot/simulate_sensors.py`, `iot/oracle_validator.py`, forward-fill fallback. |
| **Phase 3** | Centralized Baseline Models | Completed | `models/train_centralized.py` (Linear Regression, XGBoost, NN with MAE in bps & near-zero MAPE policy). |
| **Phase 4** | Federated Learning Reference | Completed | `fl/dataset_partitioner.py`, `fl/client.py`, `fl/strategy_fedavg.py`, `fl/server.py`. |
| **Phase 5** | Blockchain Verification & Norm Filter | Completed | `chain/mock_ledger.py`, `fl/strategy_blockchain.py`, Table I baseline update. |
| **Phase 6** | Attack Scenarios & Evaluation Matrix | Next | Malicious clients, Gaussian attack, adaptive attack, $\tau$-sensitivity. |
| **Phase 7** | Explainable AI (SHAP) & Audit Trail | Scheduled | `xai/`, SHAP feature attributions, tamper-proof off-chain audit logs. |
| **Phase 8** | Hyperledger Fabric Integration | Completed (Code) | `chain/chaincode/smart_contract.go` (Go 1.27 compiled and tested). |
| **Phase 9** | Interactive Dashboard & Final Handover | Scheduled | Streamlit demo, final IEEE academic report, viva preparation guide. |

