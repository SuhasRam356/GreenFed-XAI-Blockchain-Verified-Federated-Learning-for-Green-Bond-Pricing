# GreenFed-XAI: Master Defense Presentation Slide Deck Outline

**Presentation Format**: 10 High-Impact Slides (15–20 minutes)  
**Target Audience**: Academic Viva Committee, Fixed Income Portfolio Managers, Quantitative Risk Directors

---

### Slide 1: Title & Executive Overview
- **Title**: Blockchain-Verified Federated Learning for ESG-Driven Green Bond Pricing
- **Subtitle**: An Explainable AI Approach with IoT Carbon Oracles and Cryptographic Audit Trails
- **Presenter**: Suhas Ram / Jasleen Kaur
- **Key Message**: A decentralized AI platform resolving institutional data fragmentation, greenwashing vulnerabilities, and model poisoning in global sustainable debt markets.

---

### Slide 2: The 4 Structural Market Failures
- **Problem 1 (Data Silos)**: Proprietary bond covenants and private trading spreads cannot be pooled due to banking secrecy and GDPR.
- **Problem 2 (Greenwashing)**: Backward-looking, self-reported annual ESG reports are vulnerable to strategic corporate inflation.
- **Problem 3 (Adversarial Poisoning)**: Naive federated learning (FedAvg) collapses when rogue consortium nodes inject malicious gradients.
- **Problem 4 (Black-Box Opacity)**: Regulators (EU GBS, BCBS 239, SEC) mandate mathematical explainability and cryptographic auditability for debt underwriting.

---

### Slide 3: Unified System Architecture
- **Visual Diagram**: 5-tier architecture connecting IoT telemetry, local bank nodes, smart contract ledger, PyTorch FL engine, and SHAP explainer.
- **Key Invariants**:
  - Raw loan covenants never leave local bank firewalls.
  - Telemetry is screened before gradient optimization.
  - Model updates are authenticated and threshold-filtered on-chain.
  - Explanations are anchored cryptographically to prevent post-hoc tampering.

---

### Slide 4: IoT Sensor Telemetry & Oracle Gating
- **Continuous Telemetry**: Real-time monitoring of CO2e avoidance, energy capacity factors, and water efficiency across 10,000 readings.
- **Double-Barrier Oracle Screening**:
  - Physical boundary gate rejects impossible physical values (e.g., negative GHG avoidance).
  - Jump/freshness threshold rejects sudden rate-of-change spikes ($|C_t - C_{t-1}| > 15$).
- **Forward-Fill Mitigation**: Automatically substitutes historical valid reading ($C_t = C_{t-1}$), preserving gradient stability without sample loss.

---

### Slide 5: Blockchain Consensus & Norm-Threshold Defense
- **Consortium Profile**: 8 global syndicate banks partitioned under Dirichlet non-IID skew ($\alpha = 0.5$).
- **Domain-Tagged Commitment**:
  $$h_k = \text{SHA-256}(\text{"greenfed.fl.update.v1"} \,\|\, t \,\|\, k \,\|\, \text{CanonicalBytes}(\Delta w_k))$$
- **Norm Filter Rule**: Rejects any update exceeding threshold:
  $$\|\Delta w_k\|_2 \le \tau = 2.50$$
- **Consensus Quorum**: Requires $> 50\%$ accepted client submissions before updating global weights $w_{t+1}$.

---

### Slide 6: Empirical Attack Resilience Matrix (Table I)
- **Centralized Baseline**: $\text{RMSE} = 0.0868\%$, $\text{MAE} = 6.93\text{ bps}$.
- **Clean FedAvg Reference**: $\text{RMSE} = 0.1506\%$, $\text{MAE} = 12.12\text{ bps}$.
- **Catastrophic Failure without Defense**: Under 25% Gaussian noise injection ($\sigma = 2.0$), naive FedAvg error explodes to $\text{MAE} = 455.82\text{ bps}$ ($\text{MAPE} = 159.66\%$).
- **GreenFed-XAI Defended**: With norm filtering ($\tau = 2.50$), accuracy recovers to $\text{MAE} = 12.85\text{ bps}$ ($\text{RMSE} = 0.1584\%$).
- **Detection Efficacy**: Achieves **100.0% True Positive Rate** and **0.0% False Positive Rate**.

---

### Slide 7: Explainable AI (SHAP) & Efficiency Axiom
- **Game-Theoretic Decomposition**: Explains exact marginal contributions $\phi_j(x)$ for individual bond pricing.
- **Axiomatic Verification**: Proves efficiency (additivity) axiom:
  $$\sum_{j=1}^5 \phi_j(x) + \phi_0 = f(x)$$
  Observed residual error $< 10^{-6}\text{ bps}$ across all 1,500 test green bonds.
- **Global Feature Hierarchy**:
  1. Tenor (22.29%) | 2. Credit Rating (22.20%) | 3. Carbon Score (19.36%) | 4. Controversy (18.20%) | 5. Coupon (17.95%).

---

### Slide 8: 10-Point Carbon Sensitivity Analysis
- **Counterfactual Experiment**: Evaluated across 1,345 eligible test bonds ($S_i \le 90.0$) with $+10.0$ point carbon upgrade.
- **Empirical Sensitivity Result**: Mean yield spread delta is **$-2.45\text{ bps}$** (median $-2.53\text{ bps}$, std $15.08\text{ bps}$).
- **Methodological Transparency**: Disclosed as **model sensitivity under the synthetic generator**, distinguishing neural network interaction effects from causal macroeconomic claims.

---

### Slide 9: Off-Chain Cryptographic Audit Trail & Tamper Lab
- **Scalability Solution**: Storing full feature matrices and weights on-chain violates GDPR and BCBS 239 and bloats ledger state.
- **Hybrid Off-Chain Pattern**:
  - Full structured context stored off-chain in canonical JSON.
  - 32-byte SHA-256 fingerprint anchored to smart contract ledger.
- **Zero-Tolerance Tamper Evidence**: Modifying off-chain spread or features triggers instant cryptographic mismatch (`INTEGRITY_VIOLATION_HASH_MISMATCH`).

---

### Slide 10: Next.js 14 Interactive Dashboard & Conclusion
- **Full-Stack Deliverable**: Next.js 14 (TypeScript / Tailwind CSS) + FastAPI backend + PyTorch FL engine.
- **Interactive Capabilities**: Real-time green bond pricing, dynamic SHAP waterfall chart, live IoT telemetry table, round attack defense simulator, and cryptographic tamper testing lab.
- **Summary**: GreenFed-XAI establishes a verified, regulatory-compliant foundation for global institutional green bond pricing.
- **Q&A**: Inviting questions from the committee.

