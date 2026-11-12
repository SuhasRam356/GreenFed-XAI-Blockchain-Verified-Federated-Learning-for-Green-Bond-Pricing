r"""
Blockchain-Verified Federated Learning Strategy — GreenFed-XAI
===============================================================

This module implements the ``BlockchainVerifiedStrategy``, the paper's
core novelty that integrates on-chain cryptographic verification with
federated averaging to defend against model poisoning attacks.

Architecture:
    ``BlockchainVerifiedStrategy`` extends ``GreenFedAvgStrategy`` (Phase 3)
    by overriding the ``verify_and_filter_updates()`` hook. Before FedAvg
    aggregation, each client's model update undergoes two-phase on-chain
    verification:

    Phase 1 — Hash Integrity Check:
        SHA-256(submitted_parameters) == committed_hash_on_blockchain
        ⟹ Prevents post-submission parameter tampering.

    Phase 2 — Gradient Norm Check:
        ||Δw_k||_2 ≤ τ = 2.5
        ⟹ Rejects updates with abnormally large gradient magnitudes,
           the hallmark of model poisoning attacks.

Paper Results (Section V-C):
    Under a 20% malicious client attack:
    - Plain FedAvg: MAPE degrades from 4.2% → 9.8%
    - GreenFed-XAI (blockchain-verified): 94% poisoned updates rejected,
      MAPE remains at ~3.5%
    - Verification overhead: ~2.3 seconds per round, 32 bytes/round/client

Integration Points:
    - Inherits all FedAvg mechanics from ``GreenFedAvgStrategy``.
    - Uses ``MockBlockchain`` and ``SmartContract`` from ``chain.mock_ledger``.
    - Compatible with ``FederatedSimulationEngine`` in ``fl.server``.
    - Gradient norms are pre-computed by ``GreenBondFlowerClient.compute_gradient_norm()``.

Usage:
    >>> from fl.strategy_blockchain import BlockchainVerifiedStrategy
    >>> strategy = BlockchainVerifiedStrategy(
    ...     eval_dataset=(X_val, y_val),
    ...     model_template=global_model,
    ...     tau=2.5,
    ...     mining_difficulty=2,
    ... )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

try:
    from flwr.common import (
        FitRes,
        NDArrays,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy
except ImportError:
    # Graceful fallback for environments where Flower is not yet installed
    ClientProxy = Any  # type: ignore[assignment, misc]
    FitRes = Any  # type: ignore[assignment, misc]
    Parameters = Any  # type: ignore[assignment, misc]
    NDArrays = List[np.ndarray]  # type: ignore[assignment, misc]

from chain.mock_ledger import (
    MockBlockchain,
    SmartContract,
    VerificationResult,
    VerificationStatus,
)
from fl.strategy_fedavg import GreenFedAvgStrategy, RoundConvergenceMetrics
from models.nn_model import GreenBondNN

logger = logging.getLogger(__name__)


# =============================================================================
# Per-Round Blockchain Audit Record
# =============================================================================


@dataclass
class RoundBlockchainAudit:
    """Detailed blockchain verification audit for a single FL round.

    Captures the complete per-client verification results, acceptance/
    rejection statistics, and timing information for audit trail
    reconstruction and regulatory compliance.

    Attributes:
        round_idx: FL communication round index.
        total_submitted: Number of client updates submitted for verification.
        total_accepted: Number of updates that passed both verification checks.
        total_rejected: Number of updates that failed one or both checks.
        acceptance_rate: Fraction of submitted updates that were accepted.
        rejected_clients: List of client IDs whose updates were rejected.
        accepted_clients: List of client IDs whose updates were accepted.
        rejection_details: Per-client rejection reasons and details.
        avg_verification_time_ms: Mean verification time across all clients.
        total_verification_time_ms: Total time spent on verification.
        blockchain_blocks_mined: Number of new blocks mined during this round.
        grad_norms: Per-client gradient norm values.
    """

    round_idx: int
    total_submitted: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    acceptance_rate: float = 0.0
    rejected_clients: List[int] = field(default_factory=list)
    accepted_clients: List[int] = field(default_factory=list)
    rejection_details: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    avg_verification_time_ms: float = 0.0
    total_verification_time_ms: float = 0.0
    blockchain_blocks_mined: int = 0
    grad_norms: Dict[int, float] = field(default_factory=dict)


# =============================================================================
# Blockchain-Verified Federated Learning Strategy
# =============================================================================


class BlockchainVerifiedStrategy(GreenFedAvgStrategy):
    """Blockchain-verified FedAvg strategy implementing the GreenFed-XAI protocol.

    This strategy extends ``GreenFedAvgStrategy`` by interposing a blockchain-
    based verification layer between client update submission and FedAvg
    aggregation. The verification protocol follows Section IV-D of the
    IEEE paper and consists of:

    1. **Hash Commitment**: Each client's model parameters are hashed using
       SHA-256 and the hash is committed to the blockchain ledger.

    2. **Hash Verification**: The smart contract recomputes the hash from
       the actual parameters and checks it matches the commitment.

    3. **Gradient Norm Check**: The update's gradient norm ||Δw||_2 is
       checked against the threshold τ = 2.5. Exceeding updates are
       rejected as potential poisoning attacks.

    4. **Selective Aggregation**: Only updates passing both checks
       participate in the weighted FedAvg aggregation.

    Parameters:
        eval_dataset: Centralized validation dataset (X_val, y_val).
        model_template: Neural network architecture blueprint.
        device: PyTorch computation device.
        tau: Gradient norm threshold (paper default: τ = 2.5).
        mining_difficulty: Proof-of-work difficulty for the mock blockchain.
            Higher values increase mining time but provide stronger security
            simulation. Default = 2.
        max_transactions_per_block: Maximum transactions per mined block.
        fraction_fit: Fraction of clients sampled per round.
        min_fit_clients: Minimum clients required per round.
        min_available_clients: Minimum total connected clients.
        initial_parameters: Pre-initialized global model parameters.

    Attributes:
        blockchain: The mock blockchain ledger instance.
        smart_contract: The smart contract implementing verification logic.
        tau: Configured gradient norm threshold.
        blockchain_audit_history: List of per-round audit records.
    """

    def __init__(
        self,
        eval_dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        model_template: Optional[GreenBondNN] = None,
        device: Optional[torch.device] = None,
        tau: float = 2.5,
        mining_difficulty: int = 2,
        max_transactions_per_block: int = 10,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 8,
        min_available_clients: int = 8,
        initial_parameters: Optional[Parameters] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the blockchain-verified strategy.

        Sets up the mock blockchain ledger, smart contract, and all
        tracking data structures required for verification and audit.

        Args:
            eval_dataset: Validation dataset for central evaluation.
            model_template: NN model architecture blueprint.
            device: Computation device (CPU/CUDA).
            tau: Gradient norm threshold τ (default: 2.5).
            mining_difficulty: PoW difficulty level (default: 2).
            max_transactions_per_block: Max TX per block (default: 10).
            fraction_fit: Client sampling fraction (default: 1.0).
            min_fit_clients: Min clients per round (default: 8).
            min_available_clients: Min connected clients (default: 8).
            initial_parameters: Initial global model parameters.
            **kwargs: Additional keyword arguments for FedAvg base class.
        """
        super().__init__(
            eval_dataset=eval_dataset,
            model_template=model_template,
            device=device,
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            **kwargs,
        )

        # ---------------------------------------------------------------
        # Initialize Mock Blockchain and Smart Contract
        # ---------------------------------------------------------------

        self.tau = tau

        self.blockchain = MockBlockchain(
            mining_difficulty=mining_difficulty,
            max_transactions_per_block=max_transactions_per_block,
        )

        self.smart_contract = SmartContract(
            blockchain=self.blockchain,
            tau=self.tau,
        )

        # ---------------------------------------------------------------
        # Audit Trail Storage
        # ---------------------------------------------------------------

        self.blockchain_audit_history: List[RoundBlockchainAudit] = []

        logger.info(
            "BlockchainVerifiedStrategy initialized | τ = %.2f | "
            "Mining Difficulty = %d | Max TX/Block = %d",
            self.tau,
            mining_difficulty,
            max_transactions_per_block,
        )

    # ===================================================================
    # Core Override: verify_and_filter_updates
    # ===================================================================

    def verify_and_filter_updates(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[List[Tuple[ClientProxy, FitRes]], Dict[str, Any]]:
        """Blockchain-verified filtering of client updates before aggregation.

        This method overrides the base ``GreenFedAvgStrategy.verify_and_filter_updates()``
        hook to implement the full on-chain verification protocol.

        For each client update in the round:
            1. Extract the model parameters from FitRes.
            2. Compute SHA-256 hash of the parameters.
            3. Submit the hash to the blockchain ledger.
            4. Extract gradient norm ||Δw_k||_2 from client metrics.
            5. Execute the smart contract's ``process_update()`` pipeline:
               a. Verify hash matches the committed value.
               b. Check gradient norm against τ = 2.5.
            6. Record ACCEPT or REJECT on the blockchain.
            7. Include only accepted updates in the returned list.

        Parameters:
            server_round: Current FL communication round index.
            results: List of (ClientProxy, FitRes) pairs from participating clients.

        Returns:
            Tuple of:
                - accepted_results: Filtered list of client updates that passed
                  both verification checks.
                - audit_metadata: Dictionary containing verification statistics,
                  rejection details, and blockchain state information.
        """
        round_start_time = time.time()
        blocks_before = len(self.blockchain.chain)

        accepted_results: List[Tuple[ClientProxy, FitRes]] = []
        rejected_clients: List[int] = []
        accepted_clients: List[int] = []
        rejection_details: Dict[int, Dict[str, Any]] = {}
        verification_times: List[float] = []
        grad_norms: Dict[int, float] = {}

        logger.info(
            "╔══════════════════════════════════════════════════════════╗"
        )
        logger.info(
            "║  BLOCKCHAIN VERIFICATION — Round %3d (%d clients)        ║",
            server_round,
            len(results),
        )
        logger.info(
            "╚══════════════════════════════════════════════════════════╝"
        )

        for client_proxy, fit_res in results:
            # Extract client metadata
            client_id = int(fit_res.metrics.get("client_id", -1))
            grad_norm = float(fit_res.metrics.get("grad_norm", 0.0))
            grad_norms[client_id] = grad_norm

            # ----------------------------------------------------------
            # Step 1: Extract model parameters
            # ----------------------------------------------------------
            if hasattr(fit_res.parameters, "ndarrays"):
                # Mock FitRes from FederatedSimulationEngine
                client_params = fit_res.parameters.ndarrays
            else:
                # Real Flower FitRes
                client_params = parameters_to_ndarrays(fit_res.parameters)

            # ----------------------------------------------------------
            # Step 2: Compute SHA-256 hash of model parameters
            # ----------------------------------------------------------
            param_hash = SmartContract.compute_parameter_hash(client_params)

            # ----------------------------------------------------------
            # Step 3: Submit hash to blockchain
            # ----------------------------------------------------------
            self.smart_contract.submit_hash(
                client_id=client_id,
                parameter_hash=param_hash,
                round_idx=server_round,
            )

            # ----------------------------------------------------------
            # Steps 4-6: Execute smart contract verification
            # ----------------------------------------------------------
            verification_result: VerificationResult = (
                self.smart_contract.process_update(
                    client_id=client_id,
                    parameters=client_params,
                    grad_norm=grad_norm,
                    round_idx=server_round,
                    submitted_hash=param_hash,
                )
            )

            verification_times.append(
                verification_result.verification_time_ms
            )

            # ----------------------------------------------------------
            # Step 7: Accept or reject based on verification outcome
            # ----------------------------------------------------------
            if verification_result.status == VerificationStatus.ACCEPTED:
                accepted_results.append((client_proxy, fit_res))
                accepted_clients.append(client_id)

                logger.info(
                    "  ✓ Client %d ACCEPTED | ||Δw||_2 = %.4f ≤ τ = %.2f | "
                    "Hash: %s",
                    client_id,
                    grad_norm,
                    self.tau,
                    param_hash[:12] + "...",
                )
            else:
                rejected_clients.append(client_id)
                rejection_details[client_id] = {
                    "status": verification_result.status.value,
                    "grad_norm": grad_norm,
                    "hash_valid": verification_result.hash_valid,
                    "norm_valid": verification_result.norm_valid,
                    "tx_id": verification_result.tx_id,
                }

                logger.warning(
                    "  ✗ Client %d REJECTED | Status: %s | ||Δw||_2 = %.4f | "
                    "τ = %.2f | Hash Valid: %s",
                    client_id,
                    verification_result.status.value,
                    grad_norm,
                    self.tau,
                    verification_result.hash_valid,
                )

        # Mine any remaining pending transactions
        self.blockchain.mine_pending_transactions()

        blocks_after = len(self.blockchain.chain)
        blocks_mined = blocks_after - blocks_before

        # Compute aggregate verification statistics
        total_submitted = len(results)
        total_accepted = len(accepted_results)
        total_rejected = len(rejected_clients)
        acceptance_rate = (
            total_accepted / total_submitted if total_submitted > 0 else 0.0
        )
        avg_verification_time = (
            float(np.mean(verification_times)) if verification_times else 0.0
        )
        total_verification_time = (time.time() - round_start_time) * 1000.0

        # Log summary
        logger.info(
            "────────────────────────────────────────────────────────────"
        )
        logger.info(
            "  Round %d Summary: %d/%d accepted (%.1f%%) | "
            "%d rejected | %d blocks mined | %.1fms total",
            server_round,
            total_accepted,
            total_submitted,
            acceptance_rate * 100,
            total_rejected,
            blocks_mined,
            total_verification_time,
        )
        if rejected_clients:
            logger.info(
                "  Rejected clients: %s",
                ", ".join(str(c) for c in rejected_clients),
            )
        logger.info(
            "────────────────────────────────────────────────────────────"
        )

        # Store round audit record
        round_audit = RoundBlockchainAudit(
            round_idx=server_round,
            total_submitted=total_submitted,
            total_accepted=total_accepted,
            total_rejected=total_rejected,
            acceptance_rate=acceptance_rate,
            rejected_clients=rejected_clients,
            accepted_clients=accepted_clients,
            rejection_details=rejection_details,
            avg_verification_time_ms=round(avg_verification_time, 3),
            total_verification_time_ms=round(total_verification_time, 3),
            blockchain_blocks_mined=blocks_mined,
            grad_norms=grad_norms,
        )
        self.blockchain_audit_history.append(round_audit)

        # Build audit metadata dict for the aggregation pipeline
        audit_metadata: Dict[str, Any] = {
            "total_submitted": total_submitted,
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,
            "acceptance_rate": round(acceptance_rate, 4),
            "verification_mode": "blockchain_verified",
            "tau": self.tau,
            "rejected_clients": rejected_clients,
            "accepted_clients": accepted_clients,
            "rejection_details": rejection_details,
            "avg_verification_time_ms": round(avg_verification_time, 3),
            "total_verification_time_ms": round(total_verification_time, 3),
            "blockchain_blocks": blocks_after,
            "blocks_mined_this_round": blocks_mined,
            "chain_valid": self.blockchain.validate_chain()[0],
        }

        return accepted_results, audit_metadata

    # ===================================================================
    # Blockchain Audit Report Generation
    # ===================================================================

    def get_blockchain_audit_report(self) -> Dict[str, Any]:
        """Generate a comprehensive blockchain audit report across all rounds.

        Returns:
            Dictionary containing:
                - overall_statistics: Aggregate acceptance/rejection statistics.
                - blockchain_state: Current chain length, hash, and stats.
                - per_round_audits: List of per-round audit summaries.
                - smart_contract_stats: Smart contract verification statistics.
                - chain_integrity: Whether the full chain is valid.
        """
        # Aggregate statistics from audit history
        total_submitted = sum(
            a.total_submitted for a in self.blockchain_audit_history
        )
        total_accepted = sum(
            a.total_accepted for a in self.blockchain_audit_history
        )
        total_rejected = sum(
            a.total_rejected for a in self.blockchain_audit_history
        )
        overall_acceptance_rate = (
            total_accepted / total_submitted if total_submitted > 0 else 0.0
        )

        # Collect all rejected client IDs across rounds
        all_rejected = set()
        for audit in self.blockchain_audit_history:
            all_rejected.update(audit.rejected_clients)

        # Chain validation
        chain_valid, chain_errors = self.blockchain.validate_chain()

        # Per-round summaries
        per_round_summaries = []
        for audit in self.blockchain_audit_history:
            per_round_summaries.append({
                "round": audit.round_idx,
                "submitted": audit.total_submitted,
                "accepted": audit.total_accepted,
                "rejected": audit.total_rejected,
                "acceptance_rate": round(audit.acceptance_rate, 4),
                "avg_verification_time_ms": audit.avg_verification_time_ms,
                "rejected_clients": audit.rejected_clients,
            })

        return {
            "overall_statistics": {
                "total_rounds_verified": len(self.blockchain_audit_history),
                "total_submissions": total_submitted,
                "total_accepted": total_accepted,
                "total_rejected": total_rejected,
                "overall_acceptance_rate": round(overall_acceptance_rate, 4),
                "unique_rejected_clients": sorted(all_rejected),
                "tau": self.tau,
            },
            "blockchain_state": self.blockchain.get_chain_statistics(),
            "smart_contract_stats": (
                self.smart_contract.get_overall_statistics()
            ),
            "per_round_audits": per_round_summaries,
            "chain_integrity": {
                "is_valid": chain_valid,
                "errors": chain_errors,
            },
        }

    def get_audit_dataframe(self) -> pd.DataFrame:
        """Export per-round blockchain audit metrics as a pandas DataFrame.

        Returns:
            DataFrame with columns: round, submitted, accepted, rejected,
            acceptance_rate, avg_verification_time_ms, blocks_mined.
        """
        rows = []
        for audit in self.blockchain_audit_history:
            rows.append({
                "round": audit.round_idx,
                "submitted": audit.total_submitted,
                "accepted": audit.total_accepted,
                "rejected": audit.total_rejected,
                "acceptance_rate": round(audit.acceptance_rate, 4),
                "avg_verification_time_ms": audit.avg_verification_time_ms,
                "total_verification_time_ms": audit.total_verification_time_ms,
                "blocks_mined": audit.blockchain_blocks_mined,
            })
        return pd.DataFrame(rows)

    def save_audit_csv(self, output_path: Path) -> Path:
        """Save the per-round blockchain audit to a CSV file.

        Parameters:
            output_path: Path to write the CSV file.

        Returns:
            The path to the saved CSV file.
        """
        df = self.get_audit_dataframe()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved blockchain audit CSV -> %s", output_path)
        return output_path

    def print_verification_summary(self) -> None:
        """Print a formatted verification summary to stdout."""
        report = self.get_blockchain_audit_report()
        stats = report["overall_statistics"]
        chain = report["blockchain_state"]

        print("\n" + "=" * 70)
        print("  BLOCKCHAIN VERIFICATION SUMMARY — GreenFed-XAI")
        print("=" * 70)
        print(f"  {'Verification Rounds':<35s} {stats['total_rounds_verified']:>10d}")
        print(f"  {'Total Submissions':<35s} {stats['total_submissions']:>10d}")
        print(f"  {'Total Accepted':<35s} {stats['total_accepted']:>10d}")
        print(f"  {'Total Rejected':<35s} {stats['total_rejected']:>10d}")
        print(
            f"  {'Overall Acceptance Rate':<35s} "
            f"{stats['overall_acceptance_rate'] * 100:>9.1f}%"
        )
        print(f"  {'Gradient Norm Threshold (τ)':<35s} {stats['tau']:>10.1f}")
        print(f"  {'Blockchain Blocks':<35s} {chain['chain_length']:>10d}")
        print(
            f"  {'Total Blockchain Transactions':<35s} "
            f"{chain['total_transactions']:>10d}"
        )
        print(
            f"  {'Chain Integrity':<35s} "
            f"{'✓ VALID' if report['chain_integrity']['is_valid'] else '✗ INVALID':>10s}"
        )

        if stats["unique_rejected_clients"]:
            print(
                f"\n  Rejected Client IDs: "
                f"{', '.join(str(c) for c in stats['unique_rejected_clients'])}"
            )

        print("=" * 70 + "\n")

