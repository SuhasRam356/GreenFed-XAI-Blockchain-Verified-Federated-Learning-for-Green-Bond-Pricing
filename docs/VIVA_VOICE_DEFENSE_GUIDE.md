# GreenFed-XAI: Viva Voce & Master Defense Examination Guide

**Project**: GreenFed-XAI: Blockchain-Verified Federated Learning for Green Bond Pricing with Explainable AI & IoT Carbon Oracles  
**Audience**: University Examination Committee, Industry Defense Panel, Lead ML/Fintech Reviewers  
**Author**: Suhas Ram / Jasleen Kaur

---

## Part 1: High-Level Pitch & Opening Statement (2 Minutes)

> *"Good morning, esteemed committee members. Today, I present **GreenFed-XAI**, a decentralized AI and cryptographic architecture solving four critical market failures in sustainable debt capital markets: **institutional data silos**, **ESG greenwashing**, **adversarial poisoning attacks**, and the **regulatory black-box dilemma**.*  
>  
> *Green bonds trade at a 'greenium'—a yield discount reflecting sustainable covenants. However, banks cannot pool loan records due to privacy regulations, annual ESG reports are vulnerable to manipulation, collaborative AI is vulnerable to malicious poisoning, and regulators mandate explainability.*  
>  
> *GreenFed-XAI solves this by: (1) training deep pricing models across 8 banks via privacy-preserving Federated Learning under non-IID Dirichlet feature skew, (2) validating real-time IoT carbon telemetry via an automated oracle with forward-fill mitigation, (3) defending the consortium using a blockchain smart contract enforcing an $L_2$-norm gradient filter ($\tau = 2.50$) with $>50\%$ quorum consensus, (4) explaining pricing decisions using game-theoretic SHAP attributions satisfying the efficiency axiom with $<10^{-6}$ error, and (5) providing zero-tolerance tamper evidence through hybrid off-chain audit logs anchored by 32-byte SHA-256 ledger digests.*  
>  
> *Our empirical results across 10,000 bonds prove that GreenFed-XAI achieves near-centralized accuracy ($12.85\text{ bps}$ MAE) while achieving a 100% True Positive Rate in rejecting adversarial attacks that otherwise catastrophicly corrupt naive FedAvg ($455.82\text{ bps}$ error)."*

---

## Part 2: Financial Engineering & Pricing Questions

### Q1: Why predict the yield spread instead of the absolute bond price?
* **Model Answer**: Bond prices ($P$) are fundamentally dominated by the prevailing risk-free benchmark yield curve (e.g., US Treasuries or Euribor) and coupon discount discounting over time ($P = \sum \frac{C}{(1+y)^t} + \frac{F}{(1+y)^T}$). Macroeconomic rate cycles obscure the specific credit risk and environmental quality of the issuer. Yield spread (measured in basis points, where $100\text{ bps} = 1\%$) isolates the issuer-specific credit risk premium and greenium discount. Predicting spread directly follows institutional debt syndicate practice (e.g., Bloomberg Barclays Fixed Income Indices).

### Q2: What is the "Greenium" and how is it reflected in your model?
* **Model Answer**: The greenium (green premium) is the pricing discount where green-authenticated bonds trade at lower yield spreads ($y_{\text{green}} < y_{\text{vanilla}}$) than conventional peer bonds, lowering the issuer's cost of capital. In GreenFed-XAI, high carbon footprint scores ($S_i \to 100$) exert negative marginal attribution ($\phi_{\text{carbon}} < 0$) in the SHAP waterfall, reducing the predicted yield spread.

### Q3: Why is MAE reported in Basis Points (bps) instead of standard percentage?
* **Model Answer**: In institutional debt capital markets, pricing deviations are denominated in basis points ($1\text{ bps} = 0.01\%$). An error of $0.1285\%$ sounds small, but in fixed income that equals $12.85\text{ bps}$. On a \$500 million bond issuance over a 10-year tenor, a $12.85\text{ bps}$ deviation represents \$642,500 in annual interest mispricing. Reporting MAE in basis points aligns the machine learning evaluation directly with financial underwriting risk.

### Q4: How did you handle MAPE when yield spread targets approach zero?
* **Model Answer**: When bond spreads approach zero (e.g., AAA sovereign green bonds trading near benchmark rates), standard MAPE $\frac{|y - \hat{y}|}{|y|}$ divides by near-zero values, leading to numerical explosion ($> 10,000\%$). We enforced a protected near-zero MAPE policy: $\text{MAPE}_{\epsilon} = \frac{1}{N} \sum \frac{|y_i - \hat{y}_i|}{|y_i| + \epsilon}$ with $\epsilon = 10^{-6}$, ensuring stable, statistically valid relative percentage reporting.

---

## Part 3: Federated Learning & Consortium Heterogeneity Questions

### Q5: How did you partition the dataset among the 8 banks? Is it IID or Non-IID?
* **Model Answer**: Financial data across global banks is inherently non-IID due to regional specialization and varying underwriting mandates. We implemented three partition schemes in `fl/dataset_partitioner.py`:
  1. **Stratified IID**: Uniform distribution across clients (baseline reference).
  2. **Dirichlet Non-IID ($\alpha = 0.5$)**: Generates statistical quantity and feature skew across clients.
  3. **Rating-Tier Non-IID**: Partitions prime-grade debt (AAA/AA) to Nordic/DACH banks and high-yield/emerging debt (BBB/BB) to Latin American/Middle Eastern banks.
  We mathematically quantified the non-IID divergence using pairwise Jensen-Shannon (JS) divergence ($D_{\text{JS}} \in [0.18, 0.42]$), verifying realistic bank heterogeneity.

### Q6: Why did sample-weighted FedAvg yield identical results to equal-client FedAvg in your Table I?
* **Model Answer**: In our primary balanced 8-client setup, each bank holds exactly 12.5% of the total dataset ($n_k = 875$ samples each). When $n_k = \frac{N}{K}$, the sample weighting coefficient $\alpha_k = \frac{n_k}{N} = \frac{1}{8}$, which is mathematically identical to equal averaging $\frac{1}{K}$. However, `fl/strategy_fedavg.py` dynamically computes and normalizes $\alpha_k$ so that in unbalanced real-world scenarios, larger banks have proportionate influence while preserving the invariant $\sum \alpha_k = 1.0$.

### Q7: What happens if client communication drops or all clients are rejected?
* **Model Answer**: In `fl/strategy_fedavg.py` and `fl/strategy_blockchain.py`, we implemented a zero-accepted fallback mechanism. If network latency drops client connections or if all clients are rejected due to security violations, the server retains the previous round's weights $w_{t+1} = w_t$. This guarantees that the global model never gets corrupted with uninitialized or zero weights.

---

## Part 4: Adversarial Threats & Blockchain Defense Questions

### Q8: What exact threat vectors did you evaluate against the federated network?
* **Model Answer**: We evaluated four explicit threat models in `attacks/fl_attacks.py`:
  1. **Unconstrained Gaussian Noise Injection ($\sigma = 2.0$)**: Adversaries add high-variance noise to weight updates, simulating brute-force Byzantine destruction.
  2. **Directional Sign-Flip Attack**: Flips gradient signs ($\Delta w_{\text{adv}} = -\Delta w_{\text{honest}}$) to force model divergence in the opposite direction of the loss gradient.
  3. **Constrained Adaptive Perturbation**: An intelligent adversary who knows the threshold $\tau$ and scales perturbations just below the boundary ($\|\Delta w\| \le 0.95\tau$) to conduct stealthy model drift.
  4. **Post-Commitment Payload Tampering**: An adversary who submits hash $h$ on-chain but transmits altered weights $w'$ over the wire.

### Q9: How does the blockchain smart contract norm filter ($\tau = 2.50$) work?
* **Model Answer**: Each client computes its local weight update $\Delta w_k = w_{t+1}^{(k)} - w_t$ and its Euclidean $L_2$ norm $\|\Delta w_k\|_2 = \sqrt{\sum (\Delta w_{k,j})^2}$. Honest updates have norms typically ranging between $1.2$ and $2.1$. The smart contract enforces threshold $\tau = 2.50$. Any client submission with $\|\Delta w_k\|_2 > 2.50$ is rejected on-chain with state code `REJECTED_EXCEEDS_NORM_THRESHOLD`.

### Q10: How did GreenFed-XAI achieve 100% True Positive Rate and 0% False Positive Rate?
* **Model Answer**: Under the 25% Gaussian noise attack ($\sigma = 2.0$), the injected update norms range between $7.8$ and $14.5$, while honest client updates consistently stay below $2.15$. Because the separation boundary between honest convergence steps and unconstrained Byzantine noise is distinct at $\tau = 2.50$, all malicious updates are rejected ($\text{TPR} = 100.0\%$) and zero honest updates are discarded ($\text{FPR} = 0.0\%$).

### Q11: What is the defense limitation against the adaptive attack (Scenario 9)?
* **Examiner Trap Warning**: *Examiners often ask: "Can your norm filter stop ALL attacks?"*
* **Model Answer**: No, and this is an important scientific limitation that we explicitly document. An adaptive adversary who knows $\tau = 2.50$ can deliberately constrain perturbations to $\|\Delta w\| = 0.95\tau = 2.375$. Because this update satisfies the norm rule, it bypasses the threshold filter ($\text{TPR} = 0.0\%$). However, because the adversary is strictly constrained in magnitude, the damage is capped: MAE increases only slightly from $12.85\text{ bps}$ to $14.64\text{ bps}$, preventing the catastrophic collapse seen in undefended models ($455.82\text{ bps}$).

### Q12: Why not just use server-side Python filtering? Why is a blockchain necessary?
* **Model Answer**: A centralized server filter suffers from **single point of failure**, **repudiation**, and **insider collusion**. In a syndicated consortium of competing multinational banks, no bank trusts another bank's central server to arbitrate who gets excluded. A smart contract on Hyperledger Fabric provides:
  1. **Decentralized Multi-Party Quorum**: Verification rules are executed and endorsed across peer nodes.
  2. **Non-Repudiation**: A bank cannot claim it submitted valid updates if the on-chain hash mismatches.
  3. **Immutable Audit Log**: Regulators can audit the exact state transitions for every historical round.

---

## Part 5: Explainable AI (SHAP) & Carbon Sensitivity Questions

### Q13: What is the Efficiency Axiom in SHAP, and how did you verify it?
* **Model Answer**: The Efficiency Axiom (also called the Additivity Axiom) states that the sum of the feature attributions plus the expected base value must equal the exact model prediction:
  $$\sum_{j=1}^M \phi_j(x) + \phi_0 = f(x)$$
  In `xai/shap_explainer.py`, we implemented `verify_additivity()`. Across all 1,500 test set green bonds, the maximum observed residual error was $< 1.0 \times 10^{-6}\text{ bps}$, proving mathematically that our explanations account for 100% of the model's output without attribution leakage.

### Q14: What are the top features influencing green bond pricing in your model?
* **Model Answer**: Across 1,500 test bonds, global mean absolute SHAP ranking established:
  1. `tenor` (22.29%): Term premium and duration risk.
  2. `credit_rating_encoded` (22.20%): Baseline default risk spread.
  3. `carbon_footprint_score` (19.36%): Primary greenium discount driver.
  4. `esg_controversy_score` (18.20%): Downside penalty and compliance risk.
  5. `coupon_rate` (17.95%): Cash flow baseline.
  This proves that while financial fundamentals govern the nominal spread baseline, environmental metrics account for over $37.5\%$ of the total pricing variance.

### Q15: In your $+10\text{-point}$ carbon score upgrade test, why did the model yield $-2.45\text{ bps}$ instead of the synthetic generator's $-12\text{ bps}$? Does this prove causality?
* **Crucial Answer (Shows Deep Academic Integrity)**: 
  * "First, **no, this does NOT prove real-world causality**. We explicitly disclose this in the paper and dashboard as **model sensitivity under the synthetic data generator**, not a causal macroeconomic guarantee.
  * Second, the synthetic data generator defines the target spread with a baseline parameter of $-1.2\text{ bps}$ per point ($-12\text{ bps}$ for 10 points) under *ideal ceteris paribus* conditions. However, the neural network learns non-linear multi-variable interactions with tenor duration, credit rating, and controversy penalties across noisy data splits.
  * Across the 1,345 eligible test bonds ($S_i \le 90$), the neural network's empirical sensitivity averages $-2.45\text{ bps}$ (median $-2.53\text{ bps}$, std $15.08\text{ bps}$). Disclosing this difference demonstrates scientific rigor rather than presenting idealized theoretical numbers."

---

## Part 6: Off-Chain Architecture & Regulatory Compliance Questions

### Q16: Why not store the full SHAP values and bond data directly on the blockchain?
* **Model Answer**: Storing full bond features and floating-point SHAP matrices on-chain causes:
  1. **Blockchain State Bloat**: Exponential storage explosion across thousands of syndicated bonds.
  2. **Regulatory Non-Compliance (GDPR Article 17 & BCBS 239)**: Under GDPR ("Right to Erasure"), proprietary client financial data cannot be permanently written into an immutable ledger.
  GreenFed-XAI solves this by implementing an enterprise **hybrid off-chain storage pattern**: complete records are retained in off-chain JSON document stores, and only a 32-byte SHA-256 canonical digest is committed to the blockchain.

### Q17: How does your tamper detection work in practice?
* **Model Answer**: 
  1. Off-chain records are encoded using deterministic canonical JSON (sorted keys, normalized float precision).
  2. The SHA-256 hash $H = \text{SHA-256}(\text{CanonicalJSON})$ is anchored to the ledger.
  3. During audit, the auditor re-hashes the stored JSON. If an adversary modifies the predicted spread by even $0.01\text{ bps}$ or alters an ESG feature off-chain, the cryptographic hash immediately changes, triggering an `INTEGRITY_VIOLATION_HASH_MISMATCH` security alert.

---

## Part 7: Summary Scorecard for Defense

| Defense Dimension | GreenFed-XAI Metric | Standard / Baseline |
|:---|:---:|:---:|
| **Test Suite Coverage** | **169 / 169 Tests Passing (100%)** | Industry benchmark $\ge 80\%$ |
| **Pricing MAE (Defended)** | **12.85 bps** (clean reference: 6.93 bps) | Catastrophic without defense: 455.82 bps |
| **Attack True Positive Rate** | **100.0%** at $\tau = 2.50$ | 0.0% in standard FedAvg |
| **Attack False Positive Rate** | **0.0%** (zero honest clients rejected) | Acceptable threshold $< 5.0\%$ |
| **SHAP Efficiency Error** | **$< 10^{-6}\text{ bps}$** (exact additivity) | Axiomatic tolerance $< 0.15\text{ bps}$ |
| **Full Stack UI Polish** | **Next.js 14 + Tailwind + FastAPI** | Typical academic Streamlit prototype |

