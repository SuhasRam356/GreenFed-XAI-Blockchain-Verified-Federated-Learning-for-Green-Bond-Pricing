# GreenFed-XAI: Blockchain-Verified Federated Learning for Green Bond Pricing

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flower](https://img.shields.io/badge/FL-Flower%20(flwr)-orange.svg)](https://flower.ai/)
[![PyTorch](https://img.shields.io/badge/ML-PyTorch-red.svg)](https://pytorch.org/)
[![XAI: SHAP](https://img.shields.io/badge/XAI-SHAP%200.51-brightgreen.svg)](https://github.com/shap/shap)
[![Blockchain: Mock Fabric | Go Chaincode](https://img.shields.io/badge/ledger-Hyperledger%20Fabric%20%7C%20Mock-blueviolet.svg)]()
[![Frontend: Next.js 14](https://img.shields.io/badge/frontend-Next.js%2014%20%7C%20Tailwind-black.svg)](https://nextjs.org/)
[![Backend: FastAPI](https://img.shields.io/badge/backend-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![Tests: 169 Passed](https://img.shields.io/badge/tests-169%2F169%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, publication-ready implementation of **GreenFed-XAI**, a decentralized AI and cryptographic auditing framework combining blockchain-verified IoT carbon telemetry, Byzantine-resilient federated learning across banking institutions, and game-theoretic Explainable AI (SHAP) for trustworthy green bond pricing.

Based on the research paper:
> *"Blockchain-Verified Federated Learning for ESG-Driven Green Bond Pricing: An Explainable AI Approach with IoT-Enabled Carbon Footprint Oracles"*  
> **Jasleen Kaur** (Indian Institute of Forest Management Bhopal) & **Suhas Ram**

---

## 📌 System Architecture

```
                                ┌────────────────────────┐
                                │   IoT Telemetry Layer  │
                                │ (GHG, Energy, Water)   │
                                └───────────┬────────────┘
                                            │
                                            ▼
                                ┌────────────────────────┐
                                │ Blockchain Validator   │
                                │ (Oracle Gate & F-Fill) │
                                └───────────┬────────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
        ┌───────────────────────────┐               ┌───────────────────────────┐
        │ Bank 1 .. Bank 4 (Global) │               │ Bank 5 .. Bank 8 (Reg'l)  │
        │ Local Private D_k (Train) │               │ Local Private D_k (Train) │
        └─────────────┬─────────────┘               └─────────────┬─────────────┘
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            │  Uploads Δw_k & Domain-Tagged Hashes
                                            ▼
                                ┌────────────────────────┐
                                │ Blockchain Ledger Node │
                                │ (Norm Filter: τ=2.50)  │
                                └───────────┬────────────┘
                                            │  Quorum (>50%) & Valid Updates Only
                                            ▼
                                ┌────────────────────────┐
                                │  Federated Aggregator  │
                                │ (Robust FedAvg Engine) │
                                └───────────┬────────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
        ┌───────────────────────────┐               ┌───────────────────────────┐
        │  PyTorch Pricing Model    │               │  Explainable AI (SHAP)    │
        │ (128→64→32 Neural Net)    │               │ (Additivity Verified)     │
        └─────────────┬─────────────┘               └─────────────┬─────────────┘
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
                                ┌────────────────────────┐
                                │ Off-Chain JSON Audit   │
                                │ (32-Byte SHA-256 Hash) │
                                └────────────────────────┘
```

---

## 🚀 End-to-End Implementation Milestones (Phase 0 — Phase 11)

| Phase | Milestone Name | Status | Key Deliverables & Artifacts |
|:---:|:---|:---:|:---|
| **0** | Scope, Foundations & Setup | Completed | Pinned dependencies, locked environment, cataloged assumptions. |
| **1** | Synthetic Bond Data & Features | Completed | `data/generate_bonds.py`, 10,000 bonds, train/val/test splits. |
| **2** | IoT Sensor Simulation & Oracle Gate | Completed | `iot/simulate_sensors.py`, `iot/oracle_validator.py`, boundary screening, forward-fill fallback. |
| **3** | Centralized Baseline Models | Completed | `models/train_centralized.py` (Linear Regression, XGBoost, NN with MAE in bps & near-zero MAPE policy). |
| **4** | Federated Learning Reference | Completed | `fl/dataset_partitioner.py` (Dirichlet Non-IID), `fl/client.py`, `fl/strategy_fedavg.py`, `fl/server.py`. |
| **5** | Blockchain Verification & Norm Filter | Completed | `chain/mock_ledger.py`, `fl/strategy_blockchain.py`, domain-tagged commitments, $\tau = 2.50$ norm filtering. |
| **6** | Attack Scenarios & Evaluation Matrix | Completed | `attacks/fl_attacks.py`, `experiments/run_matrix.py`, 11-scenario benchmark, $\tau$-sensitivity sweep. |
| **7** | Explainable AI (SHAP) & Audit Trail | Completed | `xai/shap_explainer.py`, `audit/audit_manager.py`, additivity axiom verification ($< 10^{-6}$), off-chain JSON storage. |
| **8** | Hyperledger Fabric Integration | Completed | `chain/chaincode/smart_contract.go` (Go 1.27 smart contract for Fabric 2.4+). |
| **9** | Enterprise Interactive Dashboard | Completed | Next.js 14 + Tailwind CSS dashboard (`frontend/`), FastAPI backend (`api/main.py`), 9 REST API tests. |
| **10** | IEEE Paper & Viva Defense Package | Completed | `docs/IEEE_PAPER_MANUSCRIPT.md`, `docs/VIVA_VOICE_DEFENSE_GUIDE.md`, `docs/PRESENTATION_SLIDES_OUTLINE.md`. |
| **11** | Containerization & DevOps CI/CD | Completed | `docker-compose.yml`, `Dockerfile.backend`, `frontend/Dockerfile`, `ci/github-workflow.yml`, `.env.example`. |

---

## 📊 Benchmark Results Matrix (11 Scenarios)

Evaluated across 1,500 held-out test bonds under 100 federated communication rounds:

| Scenario ID | Scenario Name | Threat Model / Defense Mechanism | RMSE (%) | MAE (bps) | MAPE (%) | TPR (%) | FPR (%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | Centralized Clean Reference | None (Centralized Upper Bound) | 0.0868 | 6.93 | 2.72% | N/A | N/A |
| **2** | Clean FedAvg (Weighted) | Sample-Weighted (No Defense) | 0.1506 | 12.12 | 4.44% | N/A | N/A |
| **3** | Clean FedAvg (Equal) | Uniform Client Averaging | 0.1506 | 12.12 | 4.44% | N/A | N/A |
| **4** | Clean FedAvg + Hash Logging | Blockchain Hash Audit (No Norm Filter) | 0.1506 | 12.12 | 4.44% | N/A | 0.0% |
| **5** | Clean FedAvg + Norm Filter | Full GreenFed-XAI Baseline ($\tau = 2.50$) | 0.1506 | 12.12 | 4.44% | N/A | 0.0% |
| **6** | Attacked FedAvg (No Defense) | 25% Gaussian Noise Injection ($\sigma = 2.0$) | 11.8139 | 455.82 | 159.66% | 0.0% | 0.0% |
| **7** | Attacked FedAvg + Hash Only | Hash Check Only (No Norm Filter) | 11.8139 | 455.82 | 159.66% | 0.0% | 0.0% |
| **8** | Attacked FedAvg + Norm Filter | **GreenFed-XAI Defended ($\tau = 2.50$)** | **0.1584** | **12.85** | **4.69%** | **100.0%** | **0.0%** |
| **9** | Attacked FedAvg + Adaptive | Constrained Perturbation ($\|\Delta w\| \le 0.95\tau$) | 0.1800 | 14.64 | 5.37% | 0.0% | 0.0% |
| **10** | Sensor Attack (No Oracle) | Raw Biased Telemetry | 0.1506 | 12.12 | 4.44% | N/A | N/A |
| **11** | Sensor Attack (With Oracle) | IoT Oracle Validated & Mitigated | 0.1445 | 11.61 | 4.29% | N/A | N/A |

---

## 🔍 Explainable AI & Audit Findings

1. **Efficiency (Additivity) Axiom**: Verified that $\sum_{j=1}^5 \phi_j(x) + \phi_0 = f(x)$ with residual error $< 10^{-6}\text{ bps}$ across all 1,500 test bonds.
2. **Global Feature Attribution Ranking**:
   - `tenor`: **22.29%** (term premium and duration risk)
   - `credit_rating_encoded`: **22.20%** (credit default risk baseline)
   - `carbon_footprint_score`: **19.36%** (primary greenium pricing discount)
   - `esg_controversy_score`: **18.20%** (downside reputational penalty)
   - `coupon_rate`: **17.95%** (nominal cash flow structure)
3. **10-Point Carbon Score Sensitivity**: Across 1,345 eligible test bonds ($C_t \le 90.0$), a $+10.0$ point carbon upgrade yields an empirical mean yield spread reduction of **$-2.45\text{ bps}$** (median $-2.53\text{ bps}$, std $15.08\text{ bps}$). Documented as model sensitivity under synthetic generation, not a causal claim.
4. **Off-Chain Cryptographic Tamper Evidence**: Complete pricing contexts are serialized to deterministic canonical JSON, with only 32-byte SHA-256 digests committed to the blockchain. Any single-bit alteration triggers instant hash mismatch detection (`INTEGRITY_VIOLATION_HASH_MISMATCH`).

---

## 💻 Quickstart: Running the Platform

### Option 1: Native One-Command Launch (Fastest)

Launch both the **FastAPI Backend** and the **Next.js 14 Web Dashboard** simultaneously:

```powershell
python run_dashboard.py
```
*(Or double-click `run_dashboard.bat` on Windows)*

* **Interactive Web Dashboard**: `http://localhost:3000`
* **FastAPI Swagger API Documentation**: `http://127.0.0.1:8000/docs`
* **API Health Check**: `http://127.0.0.1:8000/api/health`

---

### Option 2: Docker Compose (Optional)

```bash
docker compose up --build
```

---

## 🧪 Testing & Verification

Execute the complete 169-test suite covering PyTorch models, Flower clients, smart contracts, attack simulations, SHAP explainers, audit trail managers, and REST endpoints:

```bash
pytest tests/ -v
```

```
====================== 169 passed, 7 warnings in 31.33s =======================
```

---

## 📄 License & Academic Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you use GreenFed-XAI in your research, please cite:
```bibtex
@article{greenfed_xai_2026,
  title={Blockchain-Verified Federated Learning for ESG-Driven Green Bond Pricing: An Explainable AI Approach with IoT-Enabled Carbon Footprint Oracles},
  author={Kaur, Jasleen and Ram, Suhas},
  journal={IEEE Transactions on Sustainable Computing},
  year={2026}
}
```
