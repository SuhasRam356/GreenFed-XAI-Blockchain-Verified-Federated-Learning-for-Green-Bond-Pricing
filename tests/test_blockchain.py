r"""
Unit and Integration Tests for Phase 4: Blockchain Verification Layer
====================================================================

This test suite covers:
1. Block creation, SHA-256 hashing, and deterministic representation.
2. Blockchain initialization, genesis block validity, and transaction mining.
3. Chain integrity and cryptographic tamper detection.
4. SmartContract parameter hashing, commitment submission, and verification.
5. SmartContract gradient norm threshold enforcement (τ = 2.5).
6. SmartContract two-phase process_update pipeline (accepted vs rejected variants).
7. On-chain audit trail querying by round, client, and transaction type.
8. BlockchainVerifiedStrategy update filtering and ledger state transitions.
9. End-to-end multi-round Federated Learning simulation with blockchain verification.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from chain.mock_ledger import (
    Block,
    MockBlockchain,
    SmartContract,
    Transaction,
    TransactionType,
    VerificationResult,
    VerificationStatus,
)
from fl.client import GreenBondFlowerClient
from fl.dataset_partitioner import BankDatasetPartitioner, ClientDataset
from fl.server import FederatedSimulationEngine
from fl.strategy_blockchain import BlockchainVerifiedStrategy, RoundBlockchainAudit
from models.nn_model import GreenBondNN


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_blockchain() -> MockBlockchain:
    """Provide a fresh mock blockchain with low mining difficulty for fast tests."""
    return MockBlockchain(mining_difficulty=1, max_transactions_per_block=4)


@pytest.fixture
def smart_contract(mock_blockchain: MockBlockchain) -> SmartContract:
    """Provide a SmartContract attached to the mock blockchain."""
    return SmartContract(blockchain=mock_blockchain, tau=2.5)


@pytest.fixture
def dummy_parameters() -> List[np.ndarray]:
    """Provide deterministic dummy neural network layer weights."""
    np.random.seed(42)
    return [
        np.random.randn(64, 10).astype(np.float32),
        np.random.randn(64).astype(np.float32),
        np.random.randn(1, 64).astype(np.float32),
        np.random.randn(1).astype(np.float32),
    ]


@pytest.fixture
def dummy_client_dataset() -> ClientDataset:
    """Create a minimal synthetic ClientDataset for client verification tests."""
    np.random.seed(101)
    N_train = 64
    N_val = 16
    X_train = np.random.randn(N_train, 10).astype(np.float32)
    y_train = np.random.uniform(1.0, 4.0, size=(N_train, 1)).astype(np.float32)
    X_val = np.random.randn(N_val, 10).astype(np.float32)
    y_val = np.random.uniform(1.0, 4.0, size=(N_val, 1)).astype(np.float32)
    return ClientDataset(
        client_id=0,
        bank_name="Test Bank 0",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )


# =============================================================================
# 1. Block and Transaction Unit Tests
# =============================================================================


def test_transaction_creation_and_serialization() -> None:
    tx = Transaction(
        tx_id="tx_123456",
        tx_type=TransactionType.SUBMIT_HASH,
        client_id=1,
        round_idx=3,
        timestamp="2026-11-12T00:00:00Z",
        data={"parameter_hash": "a1b2c3d4"},
    )
    d = tx.to_dict()
    assert d["tx_id"] == "tx_123456"
    assert d["tx_type"] == "SUBMIT_HASH"
    assert d["client_id"] == 1
    assert d["round_idx"] == 3
    assert d["data"]["parameter_hash"] == "a1b2c3d4"


def test_block_hash_computation_deterministic() -> None:
    tx = Transaction(
        tx_id="tx_genesis",
        tx_type=TransactionType.SUBMIT_HASH,
        client_id=-1,
        round_idx=0,
        timestamp="2026-01-01T00:00:00Z",
        data={"init": True},
    )
    b1 = Block(
        index=0,
        timestamp="2026-01-01T00:00:00Z",
        transactions=[tx],
        previous_hash="0" * 64,
        nonce=10,
    )
    b2 = Block(
        index=0,
        timestamp="2026-01-01T00:00:00Z",
        transactions=[tx],
        previous_hash="0" * 64,
        nonce=10,
    )
    h1 = b1.compute_hash()
    h2 = b2.compute_hash()
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex string


# =============================================================================
# 2. MockBlockchain Mechanics & Tamper Detection
# =============================================================================


def test_genesis_block_initialization(mock_blockchain: MockBlockchain) -> None:
    assert len(mock_blockchain.chain) == 1
    genesis = mock_blockchain.chain[0]
    assert genesis.index == 0
    assert genesis.previous_hash == "0" * 64
    assert len(genesis.transactions) == 1
    assert genesis.hash.startswith("0")
    valid, errors = mock_blockchain.validate_chain()
    assert valid is True
    assert len(errors) == 0


def test_blockchain_mining_and_block_linkage(mock_blockchain: MockBlockchain) -> None:
    # Add 5 transactions to trigger mining (max_tx = 4)
    for i in range(5):
        tx = Transaction(
            tx_id=f"tx_{i}",
            tx_type=TransactionType.SUBMIT_HASH,
            client_id=i,
            round_idx=1,
            timestamp="2026-11-12T00:00:00Z",
            data={"counter": i},
        )
        mock_blockchain.add_transaction(tx)

    # 1 genesis block + 1 mined block (containing first 4 txs)
    assert len(mock_blockchain.chain) == 2
    assert len(mock_blockchain.pending_transactions) == 1

    # Mine remaining pending tx
    mock_blockchain.mine_pending_transactions()
    assert len(mock_blockchain.chain) == 3
    assert len(mock_blockchain.pending_transactions) == 0

    # Check linkage
    assert mock_blockchain.chain[1].previous_hash == mock_blockchain.chain[0].hash
    assert mock_blockchain.chain[2].previous_hash == mock_blockchain.chain[1].hash

    valid, errors = mock_blockchain.validate_chain()
    assert valid is True


def test_blockchain_tamper_detection(mock_blockchain: MockBlockchain) -> None:
    tx = Transaction(
        tx_id="tx_valid",
        tx_type=TransactionType.ACCEPT_UPDATE,
        client_id=0,
        round_idx=1,
        timestamp="2026-11-12T00:00:00Z",
        data={"val": 100},
    )
    mock_blockchain.add_transaction(tx)
    mock_blockchain.mine_pending_transactions()

    valid_before, _ = mock_blockchain.validate_chain()
    assert valid_before is True

    # Tamper with block 1 payload
    mock_blockchain.chain[1].transactions[0] = Transaction(
        tx_id="tx_tampered",
        tx_type=TransactionType.ACCEPT_UPDATE,
        client_id=0,
        round_idx=1,
        timestamp="2026-11-12T00:00:00Z",
        data={"val": 999999},  # Altered data
    )

    valid_after, errors = mock_blockchain.validate_chain()
    assert valid_after is False
    assert any("TAMPER DETECTED at block 1" in e for e in errors)


def test_blockchain_broken_link_detection(mock_blockchain: MockBlockchain) -> None:
    tx = Transaction(
        tx_id="tx_link",
        tx_type=TransactionType.ACCEPT_UPDATE,
        client_id=1,
        round_idx=1,
        timestamp="2026-11-12T00:00:00Z",
        data={},
    )
    mock_blockchain.add_transaction(tx)
    mock_blockchain.mine_pending_transactions()

    # Corrupt the previous hash pointer
    mock_blockchain.chain[1].previous_hash = "deadbeef" * 8
    # Update hash so block itself matches but link is broken
    mock_blockchain.chain[1].hash = mock_blockchain.chain[1].compute_hash()

    valid, errors = mock_blockchain.validate_chain()
    assert valid is False
    assert any("CHAIN BREAK" in e for e in errors)


# =============================================================================
# 3. SmartContract Verification Protocol
# =============================================================================


def test_smart_contract_parameter_hash_determinism(
    dummy_parameters: List[np.ndarray],
) -> None:
    h1 = SmartContract.compute_parameter_hash(dummy_parameters)
    h2 = SmartContract.compute_parameter_hash(dummy_parameters)
    assert h1 == h2
    assert len(h1) == 64

    # Altering a single float should change the hash
    altered_params = [dummy_parameters[0].copy(), dummy_parameters[1].copy()]
    altered_params[0][0, 0] += 0.001
    h_altered = SmartContract.compute_parameter_hash(altered_params)
    assert h1 != h_altered


def test_smart_contract_gradient_norm_check(smart_contract: SmartContract) -> None:
    # tau = 2.5
    assert smart_contract.check_gradient_norm(0.5) is True
    assert smart_contract.check_gradient_norm(2.5) is True
    assert smart_contract.check_gradient_norm(2.5001) is False
    assert smart_contract.check_gradient_norm(10.0) is False
    # Explicit threshold override
    assert smart_contract.check_gradient_norm(3.0, tau=5.0) is True


def test_smart_contract_process_update_accepted(
    smart_contract: SmartContract, dummy_parameters: List[np.ndarray]
) -> None:
    param_hash = SmartContract.compute_parameter_hash(dummy_parameters)
    smart_contract.submit_hash(client_id=2, parameter_hash=param_hash, round_idx=1)

    result = smart_contract.process_update(
        client_id=2,
        parameters=dummy_parameters,
        grad_norm=1.85,
        round_idx=1,
    )

    assert result.status == VerificationStatus.ACCEPTED
    assert result.hash_valid is True
    assert result.norm_valid is True
    assert result.client_id == 2
    assert result.grad_norm == 1.85
    assert result.submitted_hash == param_hash
    assert result.computed_hash == param_hash


def test_smart_contract_process_update_rejected_norm_exceeded(
    smart_contract: SmartContract, dummy_parameters: List[np.ndarray]
) -> None:
    param_hash = SmartContract.compute_parameter_hash(dummy_parameters)
    smart_contract.submit_hash(client_id=3, parameter_hash=param_hash, round_idx=1)

    # Poisoning attack simulation: norm = 7.42 > 2.5
    result = smart_contract.process_update(
        client_id=3,
        parameters=dummy_parameters,
        grad_norm=7.42,
        round_idx=1,
    )

    assert result.status == VerificationStatus.REJECTED_NORM_EXCEEDED
    assert result.hash_valid is True
    assert result.norm_valid is False
    assert result.grad_norm == 7.42


def test_smart_contract_process_update_rejected_hash_mismatch(
    smart_contract: SmartContract, dummy_parameters: List[np.ndarray]
) -> None:
    # Submit fake/spoofed hash
    smart_contract.submit_hash(client_id=4, parameter_hash="f" * 64, round_idx=2)

    result = smart_contract.process_update(
        client_id=4,
        parameters=dummy_parameters,
        grad_norm=1.2,
        round_idx=2,
    )

    assert result.status == VerificationStatus.REJECTED_HASH_MISMATCH
    assert result.hash_valid is False
    assert result.norm_valid is True


def test_smart_contract_process_update_rejected_both(
    smart_contract: SmartContract, dummy_parameters: List[np.ndarray]
) -> None:
    # Mismatch hash AND excessive gradient norm
    smart_contract.submit_hash(client_id=5, parameter_hash="0" * 64, round_idx=2)

    result = smart_contract.process_update(
        client_id=5,
        parameters=dummy_parameters,
        grad_norm=12.5,
        round_idx=2,
    )

    assert result.status == VerificationStatus.REJECTED_BOTH
    assert result.hash_valid is False
    assert result.norm_valid is False


# =============================================================================
# 4. On-Chain Ledger Queries & Audit Reports
# =============================================================================


def test_ledger_queries_and_audit_summaries(
    smart_contract: SmartContract, dummy_parameters: List[np.ndarray]
) -> None:
    param_hash = SmartContract.compute_parameter_hash(dummy_parameters)

    # Round 1: client 0 accepts, client 1 rejects
    smart_contract.submit_hash(0, param_hash, round_idx=1)
    smart_contract.process_update(0, dummy_parameters, grad_norm=1.2, round_idx=1)

    smart_contract.submit_hash(1, param_hash, round_idx=1)
    smart_contract.process_update(1, dummy_parameters, grad_norm=8.5, round_idx=1)

    # Query round 1 transactions
    r1_txs = smart_contract.blockchain.get_transactions_by_round(1)
    assert len(r1_txs) >= 4  # 2 submits + 2 verifies + 1 accept + 1 reject

    # Rejection history
    rejections = smart_contract.blockchain.get_rejection_history()
    assert len(rejections) == 1
    assert rejections[0].client_id == 1
    assert "Gradient norm exceeded" in rejections[0].data["reason"]

    # Round summary
    summary = smart_contract.blockchain.get_round_summary(1)
    assert summary["total_submissions"] == 2
    assert summary["total_accepted"] == 1
    assert summary["total_rejected"] == 1
    assert summary["acceptance_rate"] == 0.5
    assert summary["client_statuses"][0] == "ACCEPTED"
    assert summary["client_statuses"][1] == "REJECTED"


# =============================================================================
# 5. BlockchainVerifiedStrategy Strategy Unit Tests
# =============================================================================


def test_blockchain_verified_strategy_filters_poisoned_updates(
    dummy_parameters: List[np.ndarray],
) -> None:
    strategy = BlockchainVerifiedStrategy(
        tau=2.5,
        mining_difficulty=1,
    )

    # Create 3 client fit results: 2 clean, 1 poisoned
    client_results = []
    for cid, norm in [(0, 1.2), (1, 8.5), (2, 2.1)]:
        fit_res_mock = type(
            "FitResMock",
            (),
            {
                "parameters": type(
                    "ParamMock",
                    (),
                    {
                        "tensors": dummy_parameters,
                        "ndarrays": dummy_parameters,
                    },
                )(),
                "num_examples": 100,
                "metrics": {
                    "client_id": cid,
                    "grad_norm": norm,
                    "train_loss": 0.05,
                },
            },
        )()
        client_proxy_mock = type("ClientProxyMock", (), {"cid": str(cid)})()
        client_results.append((client_proxy_mock, fit_res_mock))

    accepted, meta = strategy.verify_and_filter_updates(
        server_round=1, results=client_results
    )

    assert len(accepted) == 2
    assert meta["total_submitted"] == 3
    assert meta["total_accepted"] == 2
    assert meta["total_rejected"] == 1
    assert meta["rejected_clients"] == [1]
    assert meta["accepted_clients"] == [0, 2]
    assert meta["chain_valid"] is True

    # Audit report verification
    report = strategy.get_blockchain_audit_report()
    assert report["overall_statistics"]["total_accepted"] == 2
    assert report["overall_statistics"]["total_rejected"] == 1
    assert report["overall_statistics"]["unique_rejected_clients"] == [1]


# =============================================================================
# 6. End-to-End Simulation Test: Blockchain-Verified FL
# =============================================================================


def test_end_to_end_blockchain_verified_fl_simulation(tmp_path: Path) -> None:
    """Run an end-to-end multi-round FL simulation with BlockchainVerifiedStrategy."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 200
    n_features = 8
    X_synthetic = np.random.randn(n_samples, n_features).astype(np.float32)
    y_synthetic = (
        1.5 + 0.1 * X_synthetic[:, 0] - 0.05 * X_synthetic[:, 1]
    ).reshape(-1, 1).astype(np.float32)

    import pandas as pd
    df_dummy = pd.DataFrame({"dummy": np.arange(n_samples)})

    # 2 simulated clients
    c0 = ClientDataset(
        client_id=0,
        bank_name="Alpha Bank",
        X_train=X_synthetic[:80],
        y_train=y_synthetic[:80],
        X_val=X_synthetic[80:100],
        y_val=y_synthetic[80:100],
        raw_df=df_dummy.iloc[:100],
    )
    c1 = ClientDataset(
        client_id=1,
        bank_name="Beta Capital",
        X_train=X_synthetic[100:180],
        y_train=y_synthetic[100:180],
        X_val=X_synthetic[180:200],
        y_val=y_synthetic[180:200],
        raw_df=df_dummy.iloc[100:200],
    )

    device = torch.device("cpu")
    model = GreenBondNN(input_dim=n_features, dropout_rate=0.0).to(device)
    initial_weights = [
        val.cpu().detach().numpy().copy() for _, val in model.state_dict().items()
    ]

    clients = [
        GreenBondFlowerClient(
            client_id=0, dataset=c0, device=device, local_epochs=1, batch_size=32
        ),
        GreenBondFlowerClient(
            client_id=1, dataset=c1, device=device, local_epochs=1, batch_size=32
        ),
    ]

    strategy = BlockchainVerifiedStrategy(
        eval_dataset=(X_synthetic[80:100], y_synthetic[80:100]),
        model_template=model,
        device=device,
        tau=2.5,
        mining_difficulty=1,
        min_fit_clients=2,
        min_available_clients=2,
    )

    engine = FederatedSimulationEngine(
        clients=clients,
        strategy=strategy,
        initial_weights=initial_weights,
        eval_test_dataset=(X_synthetic[180:200], y_synthetic[180:200]),
    )

    results = engine.run_rounds(num_rounds=2, local_epochs=1)

    assert results["num_rounds"] == 2
    assert results["test_rmse"] > 0.0
    assert len(strategy.blockchain_audit_history) == 2

    # Save audit artifacts and test exports
    csv_path = tmp_path / "test_audit.csv"
    strategy.save_audit_csv(csv_path)
    assert csv_path.exists()

    json_export = strategy.blockchain.export_chain_json()
    assert "GreenFed-XAI Blockchain Genesis Block" in json_export
    valid, errors = strategy.blockchain.validate_chain()
    assert valid is True
    assert len(errors) == 0


def test_smart_contract_duplicate_submission_detection(
    smart_contract: SmartContract, dummy_parameters: List[np.ndarray]
) -> None:
    """Smart contract must detect and reject duplicate submissions in the same round."""
    param_hash = SmartContract.compute_parameter_hash(dummy_parameters)
    smart_contract.submit_hash(client_id=0, parameter_hash=param_hash, round_idx=1)

    # First submission: should succeed
    res1 = smart_contract.process_update(
        client_id=0,
        parameters=dummy_parameters,
        grad_norm=1.0,
        round_idx=1,
    )
    assert res1.status == VerificationStatus.ACCEPTED

    # Second submission by the same client in the same round: duplicate rejection
    res2 = smart_contract.process_update(
        client_id=0,
        parameters=dummy_parameters,
        grad_norm=1.0,
        round_idx=1,
    )
    assert res2.status == VerificationStatus.REJECTED_DUPLICATE_SUBMISSION


def test_smart_contract_replay_across_rounds_detection(
    smart_contract: SmartContract, dummy_parameters: List[np.ndarray]
) -> None:
    """Smart contract must reject updates replayed from past rounds."""
    # Client submits an update stamped for round 1 when active round is round 2
    res = smart_contract.process_update(
        client_id=1,
        parameters=dummy_parameters,
        grad_norm=1.0,
        round_idx=1,
        expected_round=2,
    )
    assert res.status == VerificationStatus.REJECTED_REPLAY_ATTACK


def test_quorum_enforcement_in_blockchain_strategy() -> None:
    """Strategy must reject aggregation if >50% participation quorum is not met."""
    device = torch.device("cpu")
    model = GreenBondNN(input_dim=5, dropout_rate=0.0).to(device)

    strategy = BlockchainVerifiedStrategy(
        model_template=model,
        device=device,
        tau=2.5,
        quorum_fraction=0.5,  # Requires strictly > 50%
        mining_difficulty=1,
    )

    # 4 clients total, but 3 are poisoned (norm 10.0 > 2.5) and only 1 is honest (norm 1.0)
    # Accepted = 1 / 4 = 25% <= 50% -> Quorum NOT met!
    p_mock = [np.zeros((5, 128), dtype=np.float32)]
    fit_results = []
    for cid in range(4):
        norm = 1.0 if cid == 0 else 10.0
        client_proxy = type("ClientProxyMock", (), {"cid": str(cid)})()
        fit_res = type(
            "FitResMock",
            (),
            {
                "parameters": type("ParamMock", (), {"ndarrays": p_mock})(),
                "num_examples": 100,
                "metrics": {"client_id": cid, "grad_norm": norm},
            },
        )()
        fit_results.append((client_proxy, fit_res))

    accepted, meta = strategy.verify_and_filter_updates(1, fit_results)
    # Because quorum (1/4 = 25% <= 50%) failed, accepted_results must be empty!
    assert len(accepted) == 0


# =============================================================================
# 7. Phase 5 Scientific Integrity & Attack Boundary Tests
# =============================================================================


def test_canonical_update_commitment_determinism_and_binding(
    dummy_parameters: List[np.ndarray],
) -> None:
    """Canonical update commitment must bind to round, client, and domain tags."""
    # 1. Determinism
    h1 = SmartContract.compute_canonical_update_commitment(
        dummy_parameters, client_id=2, round_idx=5
    )
    h2 = SmartContract.compute_canonical_update_commitment(
        dummy_parameters, client_id=2, round_idx=5
    )
    assert h1 == h2
    assert len(h1) == 64

    # 2. Cross-client replay prevention: same weights, different client ID -> different commitment
    h_client3 = SmartContract.compute_canonical_update_commitment(
        dummy_parameters, client_id=3, round_idx=5
    )
    assert h1 != h_client3

    # 3. Cross-round replay prevention: same weights, different round ID -> different commitment
    h_round6 = SmartContract.compute_canonical_update_commitment(
        dummy_parameters, client_id=2, round_idx=6
    )
    assert h1 != h_round6

    # 4. Domain separation: different domain tag -> different commitment
    h_domain = SmartContract.compute_canonical_update_commitment(
        dummy_parameters, client_id=2, round_idx=5, domain_tag="greenfed.fl.update.v2"
    )
    assert h1 != h_domain


def test_smart_contract_canonical_commitment_verification(
    smart_contract: SmartContract, dummy_parameters: List[np.ndarray]
) -> None:
    """Smart contract must verify update submitted with canonical commitment."""
    canonical_hash = SmartContract.compute_canonical_update_commitment(
        dummy_parameters, client_id=1, round_idx=4
    )
    smart_contract.submit_hash(client_id=1, parameter_hash=canonical_hash, round_idx=4)

    result = smart_contract.process_update(
        client_id=1,
        parameters=dummy_parameters,
        grad_norm=1.5,
        round_idx=4,
    )
    assert result.status == VerificationStatus.ACCEPTED
    assert result.hash_valid is True
    assert result.norm_valid is True


def test_legitimate_large_update_rejection(
    smart_contract: SmartContract, dummy_parameters: List[np.ndarray]
) -> None:
    """Demonstrate that an honest client with legitimate large update can be rejected if tau is too strict.

    This empirically proves the paper & guide principle:
    'A norm filter may detect some large attacks, but honest heterogeneous
    clients may be rejected. Do not describe it as a complete poisoning defense.'
    """
    param_hash = SmartContract.compute_parameter_hash(dummy_parameters)
    smart_contract.submit_hash(client_id=4, parameter_hash=param_hash, round_idx=1)

    # Legitimate bank with high portfolio volatility or non-IID drift: norm = 3.1 > tau = 2.5
    result = smart_contract.process_update(
        client_id=4,
        parameters=dummy_parameters,
        grad_norm=3.1,  # Exceeds tau = 2.5
        round_idx=1,
    )
    # Must be rejected due to norm threshold despite honest valid hash
    assert result.status == VerificationStatus.REJECTED_NORM_EXCEEDED
    assert result.hash_valid is True
    assert result.norm_valid is False


def test_malicious_small_update_stealth(
    smart_contract: SmartContract, dummy_parameters: List[np.ndarray]
) -> None:
    """Demonstrate that an adversary with valid hash and small norm bypasses filtering.

    This empirically proves the paper & guide principle:
    'A hash match checks that an update matches its earlier commitment;
    a malicious client can commit a hash of a malicious update.
    Hashing does not prove honest training or benign data.'
    """
    # Malicious client perturbs parameters subtly
    poisoned_params = [p + 0.005 for p in dummy_parameters]
    # Malicious client commits a perfectly valid SHA-256 hash of its poisoned update
    poisoned_hash = SmartContract.compute_parameter_hash(poisoned_params)
    smart_contract.submit_hash(client_id=5, parameter_hash=poisoned_hash, round_idx=2)

    # Malicious update kept under norm threshold (norm = 1.4 <= 2.5)
    result = smart_contract.process_update(
        client_id=5,
        parameters=poisoned_params,
        grad_norm=1.4,
        round_idx=2,
    )
    # Passes both checks because hash matches commitment and norm <= tau!
    assert result.status == VerificationStatus.ACCEPTED
    assert result.hash_valid is True
    assert result.norm_valid is True


def test_no_accepted_updates_fallback_preserves_global_weights() -> None:
    """When 100% of client updates are rejected, strategy must preserve previous model weights without crashing."""
    device = torch.device("cpu")
    model = GreenBondNN(input_dim=5, dropout_rate=0.0).to(device)

    # Set up strategy with initial weights
    initial_ndarrays = [p.detach().numpy().copy() for p in model.parameters()]
    from flwr.common import ndarrays_to_parameters

    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    strategy = BlockchainVerifiedStrategy(
        model_template=model,
        device=device,
        tau=2.5,
        mining_difficulty=1,
        initial_parameters=initial_parameters,
    )
    # Set latest weights
    strategy.latest_global_ndarrays = initial_ndarrays

    # All 4 clients submit updates with massive norm 50.0 > 2.5
    fit_results = []
    p_mock = [np.ones((5, 128), dtype=np.float32)]
    for cid in range(4):
        client_proxy = type("ClientProxyMock", (), {"cid": str(cid)})()
        fit_res = type(
            "FitResMock",
            (),
            {
                "parameters": type("ParamMock", (), {"ndarrays": p_mock})(),
                "num_examples": 100,
                "metrics": {"client_id": cid, "grad_norm": 50.0},
            },
        )()
        fit_results.append((client_proxy, fit_res))

    # Run aggregate_fit
    aggregated_params, metrics = strategy.aggregate_fit(
        server_round=1, results=fit_results, failures=[]
    )

    # Fallback must be triggered
    assert metrics["status"] == "fallback_previous_weights"
    assert metrics["accepted_count"] == 0
    assert aggregated_params is not None

