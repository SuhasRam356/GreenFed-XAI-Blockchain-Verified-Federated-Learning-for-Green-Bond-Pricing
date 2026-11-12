# Assumptions, Design Decisions, and Paper Deviations

## 1. Project Scope & Purpose
This codebase is an academic prototype inspired by the paper:
> *"Blockchain-Verified Federated Learning for ESG-Driven Green Bond Pricing: An Explainable AI Approach with IoT-Enabled Carbon Footprint Oracles"* (Jasleen Kaur, 2025).

As mandated by scientific integrity guidelines, this project is **not** a claim of exact reproduction of an unpublished proprietary system. All synthetic data, simulated attacks, and benchmarks are transparently documented with assumptions, units, and clear distinctions between physical reality and simulated abstractions.

---

## 2. Core Scientific & Security Principles

1. **Synthetic Data vs. Real Market Data**:
   - All bond data, IoT telemetry, and malicious attacks are synthetic and generated via reproducible pseudo-random seeds.
   - The yield spread formula is an interpretable synthetic benchmark constructed to test whether ML/FL models can recover non-trivial relationships across credit ratings, tenors, controversies, and carbon metrics.

2. **Yield Spread vs. Full Bond Price**:
   - The model predicts **yield spread** ($s$), measured in **percentage points** ($1.00 = 1.00\%$) or **basis points** ($1\text{ bp} = 0.01\%$).
   - It does **not** output full bond pricing ($P$), which would require discounting cash flows, term structure of interest rates, and specific call/put option features.

3. **Units & Evaluation Metrics**:
   - Spread target $y$: percentage points ($\%$).
   - $\text{RMSE}$: percentage points ($\%$).
   - $\text{MAE}$: reported in percentage points ($\%$) and converted to basis points ($\text{bps} = \text{MAE} \times 100$).
   - $\text{MAPE}$: percentage error ($\%$), computed with a protected denominator $\max(|y|, 10^{-6})$ and an explicit check for near-zero target counts.

4. **IoT Telemetry & Oracle Verification Boundaries**:
   - Blockchain cannot prove that a physical reading at a solar plant or sensor probe is ground-truth fact; it provides an **immutable, tamper-evident log** of submitted claims.
   - Cryptographic signatures prove authenticity of the signer key, but cannot prevent compromised hardware or stolen keys from signing false telemetry.
   - The oracle gate evaluates physical jump constraints ($|\Delta C_t| \le \theta$, with $\theta = 15$) and cryptographic validity, accompanied by an explicit forward-fill fallback for rejected readings.

5. **Federated Learning & Model Update Norm Filtering**:
   - Simulated FL across 8 banks on a single machine isolates data partitions to simulate privacy; it does not replace multi-party secure hardware enclaves or differential privacy.
   - The quantity measured for poisoning defense is the **update norm** $\|\Delta w_k\|_2 = \|w_{t+1}^{(k)} - w_t\|_2$, where $w_{t+1}^{(k)}$ is obtained after local SGD/Adam optimization.
   - Norm thresholding ($\tau = 2.5$) can detect large, unconstrained perturbations; it does not constitute an impenetrable defense against subtle or adaptive in-distribution attacks.

6. **Blockchain Ledger: Mock vs. Hyperledger Fabric**:
   - Local experiments utilize `chain/mock_ledger.py`, a pure Python in-memory permissioned ledger providing deterministic hashing, SHA-256 block linkage, and query capabilities.
   - The Hyperledger Fabric Go chaincode in `chain/chaincode/smart_contract.go` provides the production-grade smart contract definition for deployment on Fabric 2.4+.
