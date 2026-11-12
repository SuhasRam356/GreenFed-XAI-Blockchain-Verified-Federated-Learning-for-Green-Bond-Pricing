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

---

## 7. Phase 6 Attack Threat Models & Experiment Findings

1. **Malicious Client Fraction (25.0% vs. 20.0%)**:
   - The paper mentions a "20% malicious client attack" while simultaneously establishing an 8-bank consortium ($K = 8$).
   - Mathematically, compromise in an 8-bank system operates in discrete increments of $12.5\%$ ($1/8 = 12.5\%$, $2/8 = 25.0\%$, $3/8 = 37.5\%$).
   - In GreenFed-XAI, we compromise exactly 2 banks (Bank 6 and Bank 7), which is explicitly and transparently labeled as **25.0%** ($2/8$).

2. **Attack Threat Models**:
   - **Unconstrained Gaussian Noise**: Malicious clients add zero-mean Gaussian noise ($\sigma = 2.0$) to parameter updates ($\Delta w_k$). This produces massive update vectors ($\|\Delta w_k\|_2 \approx 200$), causing complete convergence collapse in unprotected FedAvg (MAPE explodes from $4.44\%$ to $159.66\%$, MAE explodes to $455.82\text{ bps}$).
   - **Cryptographic Hash Limitation (Hash-Checks-Only)**: In Scenario 7, malicious clients submit valid SHA-256 hashes of their poisoned weights. The smart contract validates that `SHA-256(weights) == submitted_hash`. The check passes, and all poisoned updates are accepted. This proves the foundational distributed security principle: cryptographic hashing guarantees transmission integrity and non-repudiation, but *cannot* detect poisoned model updates without semantic verification.
   - **Adaptive Perturbation (Norm Evasion Boundary)**: Malicious clients craft an adversarial perturbation scaled strictly below the norm threshold ($\|\Delta w_k\|_2 = 0.95 \cdot \tau = 2.375 < 2.50$). This update successfully evades the norm filter ($\text{TPR} = 0.0\%$), causing moderate model drift ($\text{MAPE} = 5.37\%$, $+0.93\%$ over baseline). This documents the exact theoretical evasion boundary of pure Euclidean norm filtering.

3. **Empirical Justification of $\tau = 2.50$ Threshold**:
   - Through a comprehensive $\tau$-sensitivity sweep across $\tau \in [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]$:
     - For $\tau \le 2.0$: Early-round honest updates have $\|\Delta w\|_2 \approx 2.47$, resulting in a $100\%$ False Positive Rate ($\text{FPR} = 100\%$, falsely rejecting honest banks).
     - For $\tau = 2.5$: $\text{TPR} = 100.0\%$ (all 10 poisoned updates caught) and $\text{FPR} = 0.0\%$ (0 honest updates rejected), restoring model accuracy to $\text{MAPE} = 4.69\%$ and $\text{MAE} = 12.85\text{ bps}$ (within $0.7\text{ bps}$ of clean baseline).
     - For $\tau \ge 3.0$: While Gaussian noise ($\|\Delta w\|_2 \approx 200$) is still caught, higher thresholds expand the attack surface for adaptive evasion.
   - Thus, $\tau = 2.50$ is empirically and mathematically justified as the Pareto-optimal threshold.

