"""
Blockchain Verification Layer — GreenFed-XAI
=============================================

This package implements the blockchain-based verification mechanism for
federated learning model updates, as described in Section IV-D of the
IEEE paper: "Blockchain-Verified Federated Learning for ESG-Driven
Green Bond Pricing."

Architecture Overview:
    1. Mock Blockchain Ledger (``mock_ledger.py``):
       A production-grade Python implementation of a permissioned blockchain
       ledger that simulates Hyperledger Fabric's consensus and verification
       mechanics. Used for local simulation and unit testing.

    2. Hyperledger Fabric Chaincode (``chaincode/``):
       Go smart contract implementing ``SubmitHash``, ``VerifyUpdate``, and
       ``RejectPoisoned`` functions compatible with Fabric 2.4+ peer CLI
       deployment.

Verification Protocol (from the paper):
    For each federated learning round t:
        1. Client k computes updated parameters w_{t+1}^{(k)}.
        2. Client k computes SHA-256 hash:
               H_k = SHA256(serialize(w_{t+1}^{(k)}))
        3. Hash H_k is submitted to the blockchain via SubmitHash.
        4. Smart contract verifies:
               a) Hash integrity: recomputed hash matches submitted hash.
               b) Gradient norm check: ||Δw_k||_2 ≤ τ (threshold = 2.5).
        5. Updates failing either check are rejected (REJECT_POISONED).
        6. Only accepted updates participate in FedAvg aggregation.

    The paper reports 94% of poisoned updates are rejected under attack,
    with MAPE remaining at ~3.5% vs. 9.8% without verification.

Modules:
    - ``mock_ledger``: MockBlockchain, Block, Transaction, SmartContract
    - ``chaincode/smart_contract.go``: Hyperledger Fabric Go chaincode
"""

from chain.mock_ledger import (
    Block,
    MockBlockchain,
    SmartContract,
    Transaction,
    TransactionType,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    "Block",
    "MockBlockchain",
    "SmartContract",
    "Transaction",
    "TransactionType",
    "VerificationResult",
    "VerificationStatus",
]

