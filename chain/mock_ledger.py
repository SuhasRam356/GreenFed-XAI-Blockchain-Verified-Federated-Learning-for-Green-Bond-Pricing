r"""
Mock Blockchain Ledger — GreenFed-XAI
======================================

A production-grade Python simulation of a permissioned blockchain ledger
designed to replicate the verification mechanics of Hyperledger Fabric 2.4
as specified in the IEEE paper's Section IV-D.

This module is used for:
    - Local simulation and rapid prototyping without Docker/Fabric.
    - Comprehensive unit testing of the verification protocol.
    - Educational demonstration of blockchain-FL integration concepts.

Blockchain Data Model:
    Each block contains:
        - index: Sequential block number (genesis = 0).
        - timestamp: ISO 8601 creation timestamp.
        - transactions: List of Transaction objects recording FL events.
        - previous_hash: SHA-256 hash of the preceding block (chain linkage).
        - nonce: Proof-of-work nonce (simplified mining simulation).
        - hash: SHA-256 hash of the current block's serialized content.

    Transactions record four event types:
        - SUBMIT_HASH: Client submits SHA-256 fingerprint of model parameters.
        - VERIFY_UPDATE: Smart contract verifies hash integrity + gradient norm.
        - ACCEPT_UPDATE: Update passes all checks; included in aggregation.
        - REJECT_POISONED: Update fails verification; excluded from aggregation.

Smart Contract Logic:
    The ``SmartContract`` class implements the on-chain verification protocol:
        1. ``submit_hash(client_id, parameter_hash, round_idx)``
           → Stores the client's submitted hash on the ledger.
        2. ``verify_update(client_id, parameters, submitted_hash)``
           → Recomputes SHA-256 from raw parameters; checks match.
        3. ``check_gradient_norm(grad_norm, tau)``
           → Returns True if ||Δw||_2 ≤ τ (default τ = 2.5).
        4. ``process_update(client_id, parameters, grad_norm, round_idx, submitted_hash)``
           → Full verification pipeline: hash check + norm check → accept/reject.

Chain Integrity:
    ``MockBlockchain.validate_chain()`` traverses the entire ledger and
    verifies that each block's ``previous_hash`` field matches the actual
    SHA-256 hash of the preceding block. Any tamper is detected immediately.

Consensus Simulation:
    A simplified proof-of-work is simulated via ``mining_difficulty`` (number
    of leading zeros required in the block hash). Default difficulty = 2
    provides realistic mining behavior without excessive CPU cost.

Usage Example:
    >>> from chain.mock_ledger import MockBlockchain, SmartContract
    >>> blockchain = MockBlockchain(mining_difficulty=2)
    >>> contract = SmartContract(blockchain=blockchain, tau=2.5)
    >>> param_hash = SmartContract.compute_parameter_hash(parameters)
    >>> result = contract.process_update(
    ...     client_id=0,
    ...     parameters=parameters,
    ...     grad_norm=1.8,
    ...     round_idx=1,
    ...     submitted_hash=param_hash,
    ... )
    >>> print(result.status)  # VerificationStatus.ACCEPTED
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class TransactionType(enum.Enum):
    """Blockchain transaction types for the FL verification protocol.

    Each type corresponds to a distinct event in the federated learning
    verification lifecycle:

    SUBMIT_HASH:
        A participating bank submits the SHA-256 hash of its local model
        update to the blockchain before the aggregation step. This
        establishes a cryptographic commitment that cannot be altered
        retroactively.

    VERIFY_UPDATE:
        The smart contract performs the two-phase verification:
        (a) recomputes the parameter hash and checks it matches the
        submitted commitment, and (b) checks the gradient norm
        against the threshold τ = 2.5.

    ACCEPT_UPDATE:
        The update passed both verification checks and is included in
        the weighted FedAvg aggregation for the current round.

    REJECT_POISONED:
        The update failed one or both verification checks. It is
        permanently excluded from aggregation, and the rejection
        reason is recorded immutably on the ledger for audit.
    """

    SUBMIT_HASH = "SUBMIT_HASH"
    VERIFY_UPDATE = "VERIFY_UPDATE"
    ACCEPT_UPDATE = "ACCEPT_UPDATE"
    REJECT_POISONED = "REJECT_POISONED"


class VerificationStatus(enum.Enum):
    """Outcome status of the smart contract verification process.

    ACCEPTED:
        Both hash integrity and gradient norm checks passed.
        The update will participate in FedAvg aggregation.

    REJECTED_HASH_MISMATCH:
        The recomputed SHA-256 hash of the submitted parameters does
        not match the hash committed to the blockchain. This indicates
        potential tampering between submission and verification.

    REJECTED_NORM_EXCEEDED:
        The gradient norm ||Δw||_2 exceeds the threshold τ = 2.5.
        This is the primary defense against model poisoning attacks,
        as poisoned updates typically exhibit abnormally large
        gradient magnitudes.

    REJECTED_BOTH:
        Both the hash integrity check and the gradient norm check
        failed simultaneously. This is a strong indicator of a
        coordinated adversarial attack.
    """

    ACCEPTED = "ACCEPTED"
    REJECTED_HASH_MISMATCH = "REJECTED_HASH_MISMATCH"
    REJECTED_NORM_EXCEEDED = "REJECTED_NORM_EXCEEDED"
    REJECTED_BOTH = "REJECTED_BOTH"
    REJECTED_DUPLICATE_SUBMISSION = "REJECTED_DUPLICATE_SUBMISSION"
    REJECTED_REPLAY_ATTACK = "REJECTED_REPLAY_ATTACK"
    REJECTED_QUORUM_NOT_MET = "REJECTED_QUORUM_NOT_MET"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class Transaction:
    """Immutable record of a single blockchain transaction.

    Each transaction captures one event in the federated learning
    verification lifecycle (hash submission, verification, acceptance,
    or rejection) with full metadata for audit trail reconstruction.

    Attributes:
        tx_id: Unique transaction identifier (SHA-256 of contents).
        tx_type: Category of the FL verification event.
        client_id: Integer ID of the participating bank (0 to K-1).
        round_idx: Global FL communication round index.
        timestamp: ISO 8601 timestamp of transaction creation.
        data: Arbitrary metadata dict (e.g., hash value, norm, reason).
    """

    tx_id: str
    tx_type: TransactionType
    client_id: int
    round_idx: int
    timestamp: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transaction to a JSON-compatible dictionary.

        Returns:
            Dictionary with all transaction fields, including the
            transaction type serialized as its string value.
        """
        result = asdict(self)
        result["tx_type"] = self.tx_type.value
        return result


@dataclass(frozen=True)
class VerificationResult:
    """Complete result of the smart contract verification process.

    Encapsulates the outcome of verifying a single client's model
    update, including the verification status, detailed check results,
    and the blockchain transaction ID for audit reference.

    Attributes:
        client_id: Integer ID of the verified bank.
        round_idx: FL communication round of the verification.
        status: Final verification outcome (ACCEPTED or REJECTED_*).
        hash_valid: Whether the recomputed hash matched the submitted hash.
        norm_valid: Whether ||Δw||_2 ≤ τ.
        grad_norm: Actual gradient norm value.
        tau: Threshold τ used for the norm check.
        submitted_hash: The SHA-256 hash committed by the client.
        computed_hash: The SHA-256 hash recomputed by the smart contract.
        tx_id: Blockchain transaction ID recording this verification.
        verification_time_ms: Time taken for verification in milliseconds.
    """

    client_id: int
    round_idx: int
    status: VerificationStatus
    hash_valid: bool
    norm_valid: bool
    grad_norm: float
    tau: float
    submitted_hash: str
    computed_hash: str
    tx_id: str
    verification_time_ms: float


@dataclass
class Block:
    """A single block in the mock blockchain.

    Each block contains a list of transactions, a cryptographic link
    to the previous block (via ``previous_hash``), and a proof-of-work
    nonce that must produce a hash with the required number of leading
    zeros (determined by ``mining_difficulty``).

    Attributes:
        index: Sequential block number (0 = genesis block).
        timestamp: ISO 8601 creation timestamp.
        transactions: List of Transaction objects in this block.
        previous_hash: SHA-256 hash of the previous block in the chain.
        nonce: Integer nonce found during proof-of-work mining.
        hash: SHA-256 hash of this block's serialized content + nonce.
    """

    index: int
    timestamp: str
    transactions: List[Transaction]
    previous_hash: str
    nonce: int = 0
    hash: str = ""

    def compute_hash(self) -> str:
        """Compute the SHA-256 hash of this block's content.

        The hash is computed over a deterministic JSON serialization of
        the block's index, timestamp, serialized transactions, previous
        hash, and nonce. This ensures that any modification to the block
        content will produce a different hash, enabling tamper detection.

        Returns:
            Hexadecimal SHA-256 hash string (64 characters).
        """
        # Build a deterministic serialization of block content
        tx_data = [tx.to_dict() for tx in self.transactions]
        block_content = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "transactions": tx_data,
                "previous_hash": self.previous_hash,
                "nonce": self.nonce,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(block_content.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize block to a JSON-compatible dictionary.

        Returns:
            Dictionary containing all block metadata, with transactions
            serialized as lists of dictionaries.
        """
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash,
        }


# =============================================================================
# Mock Blockchain Implementation
# =============================================================================


class MockBlockchain:
    """Production-grade mock blockchain simulating Hyperledger Fabric.

    Implements a permissioned blockchain ledger with:
        - Genesis block creation and initialization.
        - Simplified proof-of-work mining with configurable difficulty.
        - Immutable transaction recording with SHA-256 chain linkage.
        - Full chain validation for tamper detection.
        - Comprehensive query capabilities for audit trail retrieval.

    The mock blockchain operates entirely in-memory and is designed for:
        - Unit testing the verification protocol without Docker.
        - Rapid prototyping of blockchain-FL integration patterns.
        - Educational demonstration of blockchain consensus mechanics.

    Parameters:
        mining_difficulty: Number of leading zeros required in block hashes
            during proof-of-work mining. Higher values increase mining time
            but provide stronger security simulation. Default = 2.
        max_transactions_per_block: Maximum number of transactions packed
            into a single block before a new block is mined. Default = 10.

    Attributes:
        chain: The ordered list of Block objects forming the blockchain.
        pending_transactions: Transactions waiting to be mined into a block.
        mining_difficulty: Configured proof-of-work difficulty level.
        total_transactions: Running count of all transactions recorded.
    """

    def __init__(
        self,
        mining_difficulty: int = 2,
        max_transactions_per_block: int = 10,
    ) -> None:
        """Initialize the mock blockchain with a genesis block.

        Args:
            mining_difficulty: Number of leading zeros in valid block hashes.
            max_transactions_per_block: Max transactions before auto-mining.
        """
        self.mining_difficulty = mining_difficulty
        self.max_transactions_per_block = max_transactions_per_block
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.total_transactions: int = 0

        # Create the genesis block (block index 0)
        self._create_genesis_block()

        logger.info(
            "MockBlockchain initialized | Difficulty: %d | "
            "Max TX/Block: %d | Genesis Hash: %s",
            self.mining_difficulty,
            self.max_transactions_per_block,
            self.chain[0].hash[:16] + "...",
        )

    # -----------------------------------------------------------------------
    # Genesis Block
    # -----------------------------------------------------------------------

    def _create_genesis_block(self) -> None:
        """Create and mine the genesis block (block 0).

        The genesis block has no predecessor, so its ``previous_hash``
        is set to a string of 64 zeros. It contains a single genesis
        transaction recording the blockchain's creation event.
        """
        genesis_tx = Transaction(
            tx_id=self._generate_tx_id("genesis", 0, -1),
            tx_type=TransactionType.SUBMIT_HASH,
            client_id=-1,
            round_idx=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={
                "message": "GreenFed-XAI Blockchain Genesis Block",
                "protocol_version": "1.0.0",
                "paper": "Blockchain-Verified FL for Green Bond Pricing",
                "tau_threshold": 2.5,
            },
        )

        genesis_block = Block(
            index=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            transactions=[genesis_tx],
            previous_hash="0" * 64,
            nonce=0,
        )

        # Mine the genesis block
        genesis_block = self._mine_block(genesis_block)
        self.chain.append(genesis_block)
        self.total_transactions = 1

    # -----------------------------------------------------------------------
    # Block Mining (Simplified Proof-of-Work)
    # -----------------------------------------------------------------------

    def _mine_block(self, block: Block) -> Block:
        """Mine a block by finding a nonce that produces a valid hash.

        The mining process iterates through nonce values until the block's
        SHA-256 hash begins with ``mining_difficulty`` leading zeros. This
        simulates the proof-of-work consensus mechanism used in many
        blockchain implementations.

        Parameters:
            block: The block to mine (nonce will be modified in-place).

        Returns:
            The mined block with valid nonce and hash values set.
        """
        target_prefix = "0" * self.mining_difficulty
        max_nonce_attempts = 1_000_000  # Safety bound to prevent infinite loop

        for nonce_attempt in range(max_nonce_attempts):
            block.nonce = nonce_attempt
            candidate_hash = block.compute_hash()

            if candidate_hash.startswith(target_prefix):
                block.hash = candidate_hash
                logger.debug(
                    "Block %d mined | Nonce: %d | Hash: %s",
                    block.index,
                    nonce_attempt,
                    candidate_hash[:16] + "...",
                )
                return block

        # Fallback: accept the last hash if mining exhausts attempts
        # (should only occur with unreasonably high difficulty)
        block.hash = block.compute_hash()
        logger.warning(
            "Block %d mining exhausted %d nonce attempts. "
            "Using best available hash.",
            block.index,
            max_nonce_attempts,
        )
        return block

    # -----------------------------------------------------------------------
    # Transaction Management
    # -----------------------------------------------------------------------

    def _generate_tx_id(
        self,
        tx_type_str: str,
        round_idx: int,
        client_id: int,
    ) -> str:
        """Generate a unique transaction ID via SHA-256 hashing.

        The transaction ID is derived from the transaction type, round
        index, client ID, current timestamp, and total transaction count
        to ensure global uniqueness across the entire ledger.

        Parameters:
            tx_type_str: String representation of the transaction type.
            round_idx: FL communication round index.
            client_id: Participating bank's integer ID.

        Returns:
            Hexadecimal SHA-256 hash string (64 characters).
        """
        content = (
            f"{tx_type_str}:{round_idx}:{client_id}:"
            f"{time.time_ns()}:{self.total_transactions}"
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def add_transaction(self, transaction: Transaction) -> str:
        """Add a transaction to the pending pool and auto-mine if full.

        If the pending transaction pool reaches ``max_transactions_per_block``,
        all pending transactions are automatically mined into a new block.

        Parameters:
            transaction: The Transaction object to record on the ledger.

        Returns:
            The transaction ID of the recorded transaction.
        """
        self.pending_transactions.append(transaction)
        self.total_transactions += 1

        logger.debug(
            "Transaction added | Type: %s | Client: %d | Round: %d | "
            "TX ID: %s",
            transaction.tx_type.value,
            transaction.client_id,
            transaction.round_idx,
            transaction.tx_id[:16] + "...",
        )

        # Auto-mine when pending pool is full
        if len(self.pending_transactions) >= self.max_transactions_per_block:
            self.mine_pending_transactions()

        return transaction.tx_id

    def mine_pending_transactions(self) -> Optional[Block]:
        """Mine all pending transactions into a new block.

        Creates a new block containing all currently pending transactions,
        links it to the previous block via its hash, performs proof-of-work
        mining, and appends the mined block to the chain.

        Returns:
            The newly mined Block, or None if no pending transactions exist.
        """
        if not self.pending_transactions:
            logger.debug("No pending transactions to mine.")
            return None

        previous_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.now(timezone.utc).isoformat(),
            transactions=list(self.pending_transactions),
            previous_hash=previous_block.hash,
        )

        mined_block = self._mine_block(new_block)
        self.chain.append(mined_block)
        self.pending_transactions.clear()

        logger.debug(
            "Mined block %d with %d transactions | Hash: %s",
            mined_block.index,
            len(mined_block.transactions),
            mined_block.hash[:16] + "...",
        )

        return mined_block

    # -----------------------------------------------------------------------
    # Chain Validation (Tamper Detection)
    # -----------------------------------------------------------------------

    def validate_chain(self) -> Tuple[bool, List[str]]:
        """Validate the entire blockchain for integrity and tamper detection.

        Performs the following checks on every block in the chain:
            1. Recomputes the block's SHA-256 hash and verifies it matches
               the stored hash value.
            2. Verifies that the block's ``previous_hash`` field matches
               the actual hash of the preceding block.
            3. Confirms the block hash satisfies the proof-of-work
               difficulty requirement (correct number of leading zeros).

        Returns:
            Tuple of (is_valid, error_messages):
                - is_valid: True if the entire chain is intact.
                - error_messages: List of detected integrity violations.
        """
        errors: List[str] = []

        if not self.chain:
            errors.append("CRITICAL: Blockchain is empty (no genesis block).")
            return False, errors

        # Validate genesis block
        genesis = self.chain[0]
        if genesis.previous_hash != "0" * 64:
            errors.append(
                f"CRITICAL: Genesis block has invalid previous_hash: "
                f"{genesis.previous_hash}"
            )

        # Validate each subsequent block
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Check 1: Recompute and verify current block hash
            recomputed_hash = current_block.compute_hash()
            if current_block.hash != recomputed_hash:
                errors.append(
                    f"TAMPER DETECTED at block {i}: stored hash "
                    f"{current_block.hash[:16]}... != recomputed "
                    f"{recomputed_hash[:16]}..."
                )

            # Check 2: Verify chain linkage
            if current_block.previous_hash != previous_block.hash:
                errors.append(
                    f"CHAIN BREAK at block {i}: previous_hash "
                    f"{current_block.previous_hash[:16]}... != "
                    f"block {i-1} hash {previous_block.hash[:16]}..."
                )

            # Check 3: Verify proof-of-work
            target_prefix = "0" * self.mining_difficulty
            if not current_block.hash.startswith(target_prefix):
                errors.append(
                    f"INVALID POW at block {i}: hash "
                    f"{current_block.hash[:16]}... does not meet "
                    f"difficulty {self.mining_difficulty}"
                )

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(
                "Chain validation PASSED | %d blocks | %d total transactions",
                len(self.chain),
                self.total_transactions,
            )
        else:
            logger.error(
                "Chain validation FAILED | %d errors detected: %s",
                len(errors),
                "; ".join(errors),
            )

        return is_valid, errors

    # -----------------------------------------------------------------------
    # Query & Audit Trail Functions
    # -----------------------------------------------------------------------

    def get_transactions_by_round(self, round_idx: int) -> List[Transaction]:
        """Retrieve all transactions associated with a specific FL round.

        Searches both mined blocks and pending transactions for entries
        matching the given round index.

        Parameters:
            round_idx: The FL communication round to query.

        Returns:
            List of Transaction objects from the specified round,
            ordered by their position in the chain.
        """
        transactions: List[Transaction] = []

        # Search mined blocks
        for block in self.chain:
            for tx in block.transactions:
                if tx.round_idx == round_idx:
                    transactions.append(tx)

        # Search pending (unmined) transactions
        for tx in self.pending_transactions:
            if tx.round_idx == round_idx:
                transactions.append(tx)

        return transactions

    def get_transactions_by_client(self, client_id: int) -> List[Transaction]:
        """Retrieve all transactions associated with a specific client/bank.

        Parameters:
            client_id: The bank's integer ID (0 to K-1).

        Returns:
            List of Transaction objects from the specified client.
        """
        transactions: List[Transaction] = []

        for block in self.chain:
            for tx in block.transactions:
                if tx.client_id == client_id:
                    transactions.append(tx)

        for tx in self.pending_transactions:
            if tx.client_id == client_id:
                transactions.append(tx)

        return transactions

    def get_transactions_by_type(
        self, tx_type: TransactionType
    ) -> List[Transaction]:
        """Retrieve all transactions of a specific type.

        Parameters:
            tx_type: The TransactionType to filter by.

        Returns:
            List of matching Transaction objects.
        """
        transactions: List[Transaction] = []

        for block in self.chain:
            for tx in block.transactions:
                if tx.tx_type == tx_type:
                    transactions.append(tx)

        for tx in self.pending_transactions:
            if tx.tx_type == tx_type:
                transactions.append(tx)

        return transactions

    def get_rejection_history(self) -> List[Transaction]:
        """Retrieve all REJECT_POISONED transactions from the ledger.

        This is a convenience method for audit purposes, returning the
        complete history of rejected (potentially poisoned) model updates.

        Returns:
            List of rejection Transaction objects, ordered chronologically.
        """
        return self.get_transactions_by_type(TransactionType.REJECT_POISONED)

    def get_round_summary(self, round_idx: int) -> Dict[str, Any]:
        """Generate a comprehensive summary of a specific FL round.

        Parameters:
            round_idx: The FL communication round to summarize.

        Returns:
            Dictionary containing:
                - round_idx: The queried round index.
                - total_submissions: Number of hash submissions.
                - total_accepted: Number of accepted updates.
                - total_rejected: Number of rejected updates.
                - acceptance_rate: Fraction of submissions accepted.
                - client_statuses: Per-client accept/reject status dict.
                - rejection_reasons: List of rejection reason strings.
        """
        round_txs = self.get_transactions_by_round(round_idx)

        submissions = [
            tx
            for tx in round_txs
            if tx.tx_type == TransactionType.SUBMIT_HASH
        ]
        acceptances = [
            tx
            for tx in round_txs
            if tx.tx_type == TransactionType.ACCEPT_UPDATE
        ]
        rejections = [
            tx
            for tx in round_txs
            if tx.tx_type == TransactionType.REJECT_POISONED
        ]

        n_submissions = len(submissions)
        n_accepted = len(acceptances)
        n_rejected = len(rejections)

        client_statuses: Dict[int, str] = {}
        for tx in acceptances:
            client_statuses[tx.client_id] = "ACCEPTED"
        for tx in rejections:
            client_statuses[tx.client_id] = "REJECTED"

        rejection_reasons: List[str] = []
        for tx in rejections:
            reason = tx.data.get("reason", "Unknown")
            rejection_reasons.append(
                f"Client {tx.client_id}: {reason}"
            )

        acceptance_rate = (
            n_accepted / n_submissions if n_submissions > 0 else 0.0
        )

        return {
            "round_idx": round_idx,
            "total_submissions": n_submissions,
            "total_accepted": n_accepted,
            "total_rejected": n_rejected,
            "acceptance_rate": acceptance_rate,
            "client_statuses": client_statuses,
            "rejection_reasons": rejection_reasons,
        }

    def get_chain_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics about the blockchain state.

        Returns:
            Dictionary containing chain length, total transactions,
            pending count, transactions by type, and the latest block hash.
        """
        # Count transactions by type across all blocks
        type_counts: Dict[str, int] = {t.value: 0 for t in TransactionType}

        for block in self.chain:
            for tx in block.transactions:
                type_counts[tx.tx_type.value] += 1

        for tx in self.pending_transactions:
            type_counts[tx.tx_type.value] += 1

        latest_hash = self.chain[-1].hash if self.chain else "N/A"

        return {
            "chain_length": len(self.chain),
            "total_transactions": self.total_transactions,
            "pending_transactions": len(self.pending_transactions),
            "transaction_counts": type_counts,
            "latest_block_hash": latest_hash,
            "mining_difficulty": self.mining_difficulty,
        }

    def export_chain_json(self) -> str:
        """Export the entire blockchain as a formatted JSON string.

        Returns:
            Indented JSON string representation of the full blockchain,
            suitable for saving to disk or displaying in a dashboard.
        """
        chain_data = {
            "metadata": {
                "chain_length": len(self.chain),
                "total_transactions": self.total_transactions,
                "mining_difficulty": self.mining_difficulty,
                "exported_at": datetime.now(timezone.utc).isoformat(),
            },
            "blocks": [block.to_dict() for block in self.chain],
            "pending_transactions": [
                tx.to_dict() for tx in self.pending_transactions
            ],
        }
        return json.dumps(chain_data, indent=2, default=str)


# =============================================================================
# Smart Contract Implementation
# =============================================================================


class SmartContract:
    """Blockchain smart contract for FL model update verification.

    Implements the verification protocol described in the IEEE paper's
    Section IV-D. Each client's model update undergoes a two-phase
    verification process:

    Phase 1 — Hash Integrity Check:
        The smart contract recomputes the SHA-256 hash of the submitted
        model parameters and verifies it matches the hash committed by
        the client to the blockchain before aggregation.

    Phase 2 — Gradient Norm Check:
        The gradient norm ||Δw||_2 (L2 norm of the parameter update
        vector) is compared against the threshold τ = 2.5. Updates
        exceeding this threshold are flagged as potential poisoning
        attacks and rejected.

    Parameters:
        blockchain: The MockBlockchain instance to record transactions.
        tau: Gradient norm threshold for poisoning detection (default: 2.5).

    Attributes:
        blockchain: Reference to the underlying blockchain ledger.
        tau: Configured gradient norm threshold.
        submitted_hashes: In-memory index of submitted hashes per round.
        verification_history: List of all VerificationResult objects.
    """

    def __init__(
        self,
        blockchain: MockBlockchain,
        tau: float = 2.5,
        enforce_norm: bool = True,
    ) -> None:
        """Initialize the smart contract with blockchain and threshold.

        Args:
            blockchain: The mock blockchain ledger for transaction recording.
            tau: Gradient norm threshold τ (paper default: 2.5).
            enforce_norm: If True, enforce ||Δw||_2 <= tau. If False (hash-only mode),
                norm checks always pass, isolating hash integrity testing.
        """
        self.blockchain = blockchain
        self.tau = tau
        self.enforce_norm = enforce_norm

        # In-memory hash index: {round_idx: {client_id: submitted_hash}}
        self.submitted_hashes: Dict[int, Dict[int, str]] = {}

        # Set of (round_idx, client_id) tuples processed to detect duplicate submissions
        self.processed_updates: Set[Tuple[int, int]] = set()

        # Complete verification history for audit
        self.verification_history: List[VerificationResult] = []

        logger.info(
            "SmartContract initialized | τ = %.2f | "
            "Blockchain blocks: %d",
            self.tau,
            len(self.blockchain.chain),
        )

    # -----------------------------------------------------------------------
    # Cryptographic Hash Computation
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_parameter_hash(parameters: List[np.ndarray]) -> str:
        """Compute the SHA-256 hash of serialized model parameters.

        Creates a deterministic byte representation of the parameter
        arrays by converting each to contiguous float32 format and
        concatenating their raw bytes before hashing.

        Parameters:
            parameters: List of NumPy arrays representing model weights.

        Returns:
            Hexadecimal SHA-256 hash string (64 characters).
        """
        hasher = hashlib.sha256()

        for i, param_array in enumerate(parameters):
            # Ensure deterministic byte representation
            contiguous_array = np.ascontiguousarray(
                param_array, dtype=np.float32
            )
            # Include array shape metadata for collision resistance
            shape_bytes = str(contiguous_array.shape).encode("utf-8")
            hasher.update(shape_bytes)
            hasher.update(contiguous_array.tobytes())

        return hasher.hexdigest()

    # -----------------------------------------------------------------------
    # Hash Submission (Phase 1a — Commitment)
    # -----------------------------------------------------------------------

    def submit_hash(
        self,
        client_id: int,
        parameter_hash: str,
        round_idx: int,
    ) -> str:
        """Submit a client's parameter hash commitment to the blockchain.

        Records the SHA-256 hash of the client's model update on the
        immutable ledger. This establishes a cryptographic commitment
        that cannot be altered retroactively.

        Parameters:
            client_id: Integer ID of the submitting bank (0 to K-1).
            parameter_hash: SHA-256 hex digest of model parameters.
            round_idx: Current FL communication round index.

        Returns:
            Transaction ID of the recorded submission.

        Raises:
            ValueError: If client_id or round_idx is negative.
        """
        if client_id < 0:
            raise ValueError(
                f"Invalid client_id: {client_id}. Must be non-negative."
            )
        if round_idx < 0:
            raise ValueError(
                f"Invalid round_idx: {round_idx}. Must be non-negative."
            )

        # Store in the in-memory hash index
        if round_idx not in self.submitted_hashes:
            self.submitted_hashes[round_idx] = {}
        self.submitted_hashes[round_idx][client_id] = parameter_hash

        # Record on the blockchain ledger
        tx = Transaction(
            tx_id=self.blockchain._generate_tx_id(
                TransactionType.SUBMIT_HASH.value, round_idx, client_id
            ),
            tx_type=TransactionType.SUBMIT_HASH,
            client_id=client_id,
            round_idx=round_idx,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={
                "parameter_hash": parameter_hash,
                "hash_algorithm": "SHA-256",
                "bytes_stored": 32,  # SHA-256 = 256 bits = 32 bytes
            },
        )
        tx_id = self.blockchain.add_transaction(tx)

        logger.debug(
            "Hash submitted | Client: %d | Round: %d | Hash: %s | TX: %s",
            client_id,
            round_idx,
            parameter_hash[:16] + "...",
            tx_id[:16] + "...",
        )

        return tx_id

    # -----------------------------------------------------------------------
    # Hash Verification (Phase 1b — Integrity Check)
    # -----------------------------------------------------------------------

    def verify_hash(
        self,
        client_id: int,
        parameters: List[np.ndarray],
        round_idx: int,
    ) -> Tuple[bool, str, str]:
        """Verify that submitted parameters match the committed hash.

        Recomputes the SHA-256 hash of the raw model parameters and
        checks it against the hash previously committed to the blockchain
        via ``submit_hash()``.

        Parameters:
            client_id: Integer ID of the bank being verified.
            parameters: The actual model parameter arrays to verify.
            round_idx: FL round index for hash lookup.

        Returns:
            Tuple of (hash_matches, computed_hash, submitted_hash):
                - hash_matches: True if hashes are identical.
                - computed_hash: SHA-256 recomputed from parameters.
                - submitted_hash: Hash retrieved from the blockchain.
        """
        computed_hash = self.compute_parameter_hash(parameters)

        # Retrieve the previously submitted hash
        submitted_hash = (
            self.submitted_hashes.get(round_idx, {}).get(client_id, "")
        )

        if not submitted_hash:
            logger.warning(
                "No submitted hash found for client %d in round %d",
                client_id,
                round_idx,
            )
            return False, computed_hash, ""

        hash_matches = computed_hash == submitted_hash

        if not hash_matches:
            logger.warning(
                "HASH MISMATCH | Client: %d | Round: %d | "
                "Computed: %s | Submitted: %s",
                client_id,
                round_idx,
                computed_hash[:16] + "...",
                submitted_hash[:16] + "...",
            )

        return hash_matches, computed_hash, submitted_hash

    # -----------------------------------------------------------------------
    # Gradient Norm Check (Phase 2 — Poisoning Detection)
    # -----------------------------------------------------------------------

    def check_gradient_norm(
        self,
        grad_norm: float,
        tau: Optional[float] = None,
    ) -> bool:
        """Check whether the gradient norm is within acceptable bounds.

        Implements the paper's primary defense against model poisoning
        attacks. Updates with ||Δw||_2 > τ are classified as potentially
        poisoned and will be rejected.

        Parameters:
            grad_norm: The L2 norm of the client's parameter update vector.
            tau: Override threshold (defaults to self.tau = 2.5).

        Returns:
            True if grad_norm ≤ τ (update is acceptable).
        """
        if not self.enforce_norm:
            return True

        threshold = tau if tau is not None else self.tau
        is_valid = grad_norm <= threshold

        if not is_valid:
            logger.warning(
                "GRADIENT NORM EXCEEDED | ||Δw||_2 = %.4f > τ = %.2f",
                grad_norm,
                threshold,
            )

        return is_valid

    # -----------------------------------------------------------------------
    # Full Verification Pipeline
    # -----------------------------------------------------------------------

    def process_update(
        self,
        client_id: int,
        parameters: List[np.ndarray],
        grad_norm: float,
        round_idx: int,
        submitted_hash: Optional[str] = None,
        expected_round: Optional[int] = None,
    ) -> VerificationResult:
        """Execute the complete two-phase verification pipeline.

        Performs both hash integrity verification and gradient norm checking,
        records the verification result on the blockchain, and returns a
        comprehensive VerificationResult object.

        The verification pipeline:
            1. If ``submitted_hash`` is provided and no prior submission
               exists, auto-submit the hash to the ledger.
            2. Check for replay attack (round index mismatch against active round).
            3. Check for duplicate update submission by the same client in this round.
            4. Recompute SHA-256 hash from raw parameters and compare against committed hash.
            5. Check gradient norm against τ (Phase 2).
            6. Record ACCEPT_UPDATE or REJECT_POISONED on the ledger.
            7. Return VerificationResult with full audit metadata.

        Parameters:
            client_id: Integer ID of the bank being verified (0 to K-1).
            parameters: The model parameter arrays submitted by the client.
            grad_norm: L2 norm of the parameter update vector ||Δw||_2.
            round_idx: Current FL communication round index.
            submitted_hash: Optional pre-computed hash. If provided and
                no prior submission exists, it is submitted automatically.
            expected_round: Optional active round index to detect replay attacks.

        Returns:
            VerificationResult containing the status, check details,
            and blockchain transaction ID.
        """
        verification_start = time.time()

        # Auto-submit hash if provided and not already submitted
        if submitted_hash is not None:
            existing = (
                self.submitted_hashes.get(round_idx, {}).get(client_id)
            )
            if existing is None:
                self.submit_hash(client_id, submitted_hash, round_idx)

        # Replay Attack Detection
        if expected_round is not None and round_idx != expected_round:
            hash_valid, computed_hash, stored_hash = False, "", ""
            norm_valid = False
            status = VerificationStatus.REJECTED_REPLAY_ATTACK
            logger.warning(
                "REPLAY ATTACK DETECTED | Client: %d | Submitted Round: %d != Active Round: %d",
                client_id,
                round_idx,
                expected_round,
            )
        # Duplicate Submission Detection
        elif (round_idx, client_id) in self.processed_updates:
            hash_valid, computed_hash, stored_hash = False, "", ""
            norm_valid = False
            status = VerificationStatus.REJECTED_DUPLICATE_SUBMISSION
            logger.warning(
                "DUPLICATE SUBMISSION DETECTED | Client: %d already submitted for Round: %d",
                client_id,
                round_idx,
            )
        else:
            self.processed_updates.add((round_idx, client_id))

            # Phase 1: Hash integrity verification
            hash_valid, computed_hash, stored_hash = self.verify_hash(
                client_id, parameters, round_idx
            )

            # Phase 2: Gradient norm check
            norm_valid = self.check_gradient_norm(grad_norm)

            # Determine verification status
            if hash_valid and norm_valid:
                status = VerificationStatus.ACCEPTED
            elif not hash_valid and not norm_valid:
                status = VerificationStatus.REJECTED_BOTH
            elif not hash_valid:
                status = VerificationStatus.REJECTED_HASH_MISMATCH
            else:
                status = VerificationStatus.REJECTED_NORM_EXCEEDED

        verification_time_ms = (time.time() - verification_start) * 1000.0

        # Record verification transaction on the ledger
        verify_tx = Transaction(
            tx_id=self.blockchain._generate_tx_id(
                TransactionType.VERIFY_UPDATE.value, round_idx, client_id
            ),
            tx_type=TransactionType.VERIFY_UPDATE,
            client_id=client_id,
            round_idx=round_idx,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={
                "hash_valid": hash_valid,
                "norm_valid": norm_valid,
                "grad_norm": float(grad_norm),
                "tau": self.tau,
                "computed_hash": computed_hash,
                "submitted_hash": stored_hash,
                "verification_time_ms": round(verification_time_ms, 3),
            },
        )
        self.blockchain.add_transaction(verify_tx)

        # Record accept or reject transaction
        if status == VerificationStatus.ACCEPTED:
            outcome_tx = Transaction(
                tx_id=self.blockchain._generate_tx_id(
                    TransactionType.ACCEPT_UPDATE.value, round_idx, client_id
                ),
                tx_type=TransactionType.ACCEPT_UPDATE,
                client_id=client_id,
                round_idx=round_idx,
                timestamp=datetime.now(timezone.utc).isoformat(),
                data={
                    "grad_norm": float(grad_norm),
                    "parameter_hash": computed_hash,
                    "message": "Update verified and accepted for aggregation",
                },
            )
        else:
            # Build detailed rejection reason
            reasons: List[str] = []
            if not hash_valid:
                reasons.append(
                    f"Hash mismatch (computed={computed_hash[:16]}... "
                    f"vs submitted={stored_hash[:16]}...)"
                )
            if not norm_valid:
                reasons.append(
                    f"Gradient norm exceeded (||Δw||_2={grad_norm:.4f} "
                    f"> τ={self.tau})"
                )

            outcome_tx = Transaction(
                tx_id=self.blockchain._generate_tx_id(
                    TransactionType.REJECT_POISONED.value,
                    round_idx,
                    client_id,
                ),
                tx_type=TransactionType.REJECT_POISONED,
                client_id=client_id,
                round_idx=round_idx,
                timestamp=datetime.now(timezone.utc).isoformat(),
                data={
                    "grad_norm": float(grad_norm),
                    "reason": "; ".join(reasons),
                    "status": status.value,
                    "parameter_hash": computed_hash,
                    "message": "Update rejected — excluded from aggregation",
                },
            )

        outcome_tx_id = self.blockchain.add_transaction(outcome_tx)

        # Build the verification result
        result = VerificationResult(
            client_id=client_id,
            round_idx=round_idx,
            status=status,
            hash_valid=hash_valid,
            norm_valid=norm_valid,
            grad_norm=float(grad_norm),
            tau=self.tau,
            submitted_hash=stored_hash,
            computed_hash=computed_hash,
            tx_id=outcome_tx_id,
            verification_time_ms=round(verification_time_ms, 3),
        )

        self.verification_history.append(result)

        # Log the outcome
        log_fn = logger.info if status == VerificationStatus.ACCEPTED else logger.warning
        log_fn(
            "Verification | Client: %d | Round: %d | Status: %s | "
            "||Δw||_2: %.4f | τ: %.2f | Time: %.3fms",
            client_id,
            round_idx,
            status.value,
            grad_norm,
            self.tau,
            verification_time_ms,
        )

        return result

    # -----------------------------------------------------------------------
    # Quorum Verification (>50% Participation Rule)
    # -----------------------------------------------------------------------

    def check_quorum(
        self,
        round_idx: int,
        total_clients: int = 8,
        min_participation_ratio: float = 0.5,
    ) -> Tuple[bool, int, float]:
        """Verify whether the round reached the >50% participation quorum.

        As specified in Section IV-D of the IEEE paper, a global communication
        round is only quorate and eligible for aggregation if more than 50%
        of authorized clients successfully pass verification.

        Parameters:
            round_idx: FL communication round index.
            total_clients: Total consortium size (default: 8).
            min_participation_ratio: Minimum ratio required (> 0.5).

        Returns:
            Tuple of (is_quorate, num_accepted, participation_ratio).
        """
        round_accepted = [
            r for r in self.verification_history
            if r.round_idx == round_idx and r.status == VerificationStatus.ACCEPTED
        ]
        n_accepted = len(round_accepted)
        ratio = (n_accepted / total_clients) if total_clients > 0 else 0.0
        is_quorate = ratio > min_participation_ratio
        return is_quorate, n_accepted, ratio

    # -----------------------------------------------------------------------
    # Audit & Analytics
    # -----------------------------------------------------------------------

    def get_round_verification_summary(
        self, round_idx: int
    ) -> Dict[str, Any]:
        """Get a summary of all verifications performed in a given round.

        Parameters:
            round_idx: The FL communication round to summarize.

        Returns:
            Dictionary with accepted/rejected counts, acceptance rate,
            per-client details, and average verification time.
        """
        round_results = [
            r for r in self.verification_history if r.round_idx == round_idx
        ]

        if not round_results:
            return {
                "round_idx": round_idx,
                "total_verified": 0,
                "total_accepted": 0,
                "total_rejected": 0,
                "acceptance_rate": 0.0,
                "avg_verification_time_ms": 0.0,
                "client_details": {},
            }

        accepted = [
            r for r in round_results
            if r.status == VerificationStatus.ACCEPTED
        ]
        rejected = [
            r for r in round_results
            if r.status != VerificationStatus.ACCEPTED
        ]

        client_details: Dict[int, Dict[str, Any]] = {}
        for r in round_results:
            client_details[r.client_id] = {
                "status": r.status.value,
                "grad_norm": r.grad_norm,
                "hash_valid": r.hash_valid,
                "norm_valid": r.norm_valid,
                "verification_time_ms": r.verification_time_ms,
            }

        avg_time = float(
            np.mean([r.verification_time_ms for r in round_results])
        )

        n_total = len(round_results)
        n_accepted = len(accepted)

        return {
            "round_idx": round_idx,
            "total_verified": n_total,
            "total_accepted": n_accepted,
            "total_rejected": len(rejected),
            "acceptance_rate": n_accepted / n_total if n_total > 0 else 0.0,
            "avg_verification_time_ms": round(avg_time, 3),
            "client_details": client_details,
        }

    def get_overall_statistics(self) -> Dict[str, Any]:
        """Compute overall verification statistics across all rounds.

        Returns:
            Dictionary containing total verifications, acceptance/rejection
            counts and rates, average norms, and per-round breakdowns.
        """
        if not self.verification_history:
            return {
                "total_verifications": 0,
                "total_accepted": 0,
                "total_rejected": 0,
                "overall_acceptance_rate": 0.0,
                "avg_accepted_norm": 0.0,
                "avg_rejected_norm": 0.0,
                "avg_verification_time_ms": 0.0,
            }

        accepted = [
            r for r in self.verification_history
            if r.status == VerificationStatus.ACCEPTED
        ]
        rejected = [
            r for r in self.verification_history
            if r.status != VerificationStatus.ACCEPTED
        ]

        total = len(self.verification_history)
        n_accepted = len(accepted)
        n_rejected = len(rejected)

        avg_accepted_norm = (
            float(np.mean([r.grad_norm for r in accepted]))
            if accepted
            else 0.0
        )
        avg_rejected_norm = (
            float(np.mean([r.grad_norm for r in rejected]))
            if rejected
            else 0.0
        )
        avg_time = float(
            np.mean(
                [r.verification_time_ms for r in self.verification_history]
            )
        )

        return {
            "total_verifications": total,
            "total_accepted": n_accepted,
            "total_rejected": n_rejected,
            "overall_acceptance_rate": n_accepted / total if total > 0 else 0.0,
            "avg_accepted_norm": round(avg_accepted_norm, 4),
            "avg_rejected_norm": round(avg_rejected_norm, 4),
            "avg_verification_time_ms": round(avg_time, 3),
            "tau": self.tau,
        }

