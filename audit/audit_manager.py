r"""
Off-Chain Audit Records and Blockchain Verification Bridge — GreenFed-XAI
========================================================================

This module implements the off-chain auditable prediction framework and
tamper-evident blockchain commitment interface specified in Phase 7.

Architecture & Security Model:
------------------------------
1. Privacy-Preserving Off-Chain Storage:
   - Sensitive financial features and granular SHAP explanation matrices
     are stored off-chain in structured JSON documents.
   - Storing large models and sensitive client data directly on-chain would
     violate data protection regulations (e.g. GDPR, Basel III) and bloat ledger size.

2. On-Chain Cryptographic Anchoring:
   - The smart contract ledger stores only:
     (a) Record Identifier (record_id)
     (b) Canonical SHA-256 digest of the entire audit record
     (c) ISO 8601 Timestamp
     (d) Model version & hash
   - Size overhead: exactly 32 bytes for the cryptographic digest.

3. Canonical Serialization:
   - Deterministic JSON encoding (alphabetically sorted keys, normalized float precision)
     guarantees that recomputing the hash from the retained off-chain record yields
     identical bytes across different systems.

4. Tampering Detection:
   - If an unauthorized party alters predictions, features, or SHAP values
     in the off-chain storage layer, recomputing the SHA-256 digest immediately
     flags a cryptographic mismatch against the ledger's immutable commitment.

5. Scientific Limitations:
   - A cryptographic hash proves only that the retrieved off-chain record is identical
     to what was committed at timestamp T.
   - Hashing alone does NOT prove that the prediction was mathematically optimal or
     that the model itself was free from historical data bias.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from chain.mock_ledger import MockBlockchain, Transaction, TransactionType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OffChainAuditRecord:
    """Structured, tamper-evident audit record for an explainable prediction.

    Attributes:
    -----------
    record_id : str
        Unique pseudonymous identifier (UUIDv4).
    bond_id : str
        Green bond identifier reference.
    timestamp : str
        ISO 8601 UTC creation timestamp.
    features : Dict[str, Any]
        Input features used for the prediction (unscaled or authorized fields).
    predicted_spread_pct : float
        Neural network yield spread prediction in percentage points (%).
    predicted_spread_bps : float
        Prediction converted to basis points (100 * spread_pct).
    base_value : float
        SHAP background expectation (E[f(x)]).
    shap_attributions : Dict[str, float]
        Per-feature SHAP additive contributions.
    model_version : str
        Model architecture and checkpoint version tag.
    model_hash : str
        Cryptographic SHA-256 fingerprint of the model parameters.
    explainer_method : str
        XAI explanation methodology (e.g. 'shap.KernelExplainer').
    oracle_provenance : Dict[str, Any]
        Audit reference linking IoT oracle telemetry validation status.
    """

    record_id: str
    bond_id: str
    timestamp: str
    features: Dict[str, Any]
    predicted_spread_pct: float
    predicted_spread_bps: float
    base_value: float
    shap_attributions: Dict[str, float]
    model_version: str
    model_hash: str
    explainer_method: str
    oracle_provenance: Dict[str, Any]

    def to_canonical_json(self) -> str:
        """Produce deterministic canonical JSON string for hashing.

        Returns:
        --------
        str
            Deterministic JSON string with sorted keys and normalized representations.
        """
        data = asdict(self)
        # Round floats for deterministic cross-platform hashing
        def normalize_floats(obj: Any) -> Any:
            if isinstance(obj, float):
                return round(obj, 6)
            if isinstance(obj, dict):
                return {k: normalize_floats(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [normalize_floats(v) for v in obj]
            return obj

        clean_dict = normalize_floats(data)
        return json.dumps(clean_dict, sort_keys=True, indent=None, ensure_ascii=True)

    def compute_sha256(self) -> str:
        """Compute SHA-256 fingerprint over canonical JSON bytes.

        Returns:
        --------
        str
            64-character hexadecimal SHA-256 hash.
        """
        canonical_bytes = self.to_canonical_json().encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()


class AuditManager:
    """Orchestrates off-chain audit record persistence and on-chain anchoring.

    Parameters:
    -----------
    blockchain : MockBlockchain
        Underlying blockchain ledger instance.
    storage_dir : Path
        Directory for off-chain JSON storage.
    """

    def __init__(
        self,
        blockchain: MockBlockchain,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.blockchain = blockchain
        self.storage_dir = storage_dir or (Path("outputs") / "audit_records")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # In-memory index of on-chain anchored records: {record_id: {hash, tx_id}}
        self.anchored_records: Dict[str, Dict[str, Any]] = {}

    def commit_audit_record(
        self,
        record: OffChainAuditRecord,
    ) -> Tuple[str, str]:
        """Persist record off-chain and commit cryptographic hash on-chain.

        Parameters:
        -----------
        record : OffChainAuditRecord
            Audit record to anchor.

        Returns:
        --------
        Tuple[str, str]
            (audit_hash, tx_id)
        """
        # 1. Compute canonical SHA-256 hash
        audit_hash = record.compute_sha256()

        # 2. Save full structured document off-chain
        record_path = self.storage_dir / f"{record.record_id}.json"
        with open(record_path, "w", encoding="utf-8") as f:
            f.write(record.to_canonical_json())

        # 3. Create blockchain transaction for on-chain anchoring
        tx = Transaction(
            tx_id=self.blockchain._generate_tx_id("AUDIT", 0, -1),
            tx_type=TransactionType.SUBMIT_HASH,
            client_id=-1,
            round_idx=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={
                "event": "AUDIT_RECORD_COMMITTED",
                "record_id": record.record_id,
                "bond_id": record.bond_id,
                "audit_hash": audit_hash,
                "model_version": record.model_version,
                "model_hash": record.model_hash,
                "prediction_bps": record.predicted_spread_bps,
                "storage_uri": str(record_path),
            },
        )
        tx_id = self.blockchain.add_transaction(tx)

        # Store in local index
        self.anchored_records[record.record_id] = {
            "audit_hash": audit_hash,
            "tx_id": tx_id,
            "path": record_path,
        }

        logger.info(
            "Anchored Audit Record %s | Bond: %s | Hash: %s | TX: %s",
            record.record_id,
            record.bond_id,
            audit_hash[:16] + "...",
            tx_id[:16] + "...",
        )
        return audit_hash, tx_id

    def retrieve_and_verify_record(
        self,
        record_id: str,
    ) -> Dict[str, Any]:
        """Retrieve off-chain record and cryptographically verify against ledger.

        Parameters:
        -----------
        record_id : str
            Unique record ID to verify.

        Returns:
        --------
        Dict[str, Any]
            Verification outcome containing 'is_valid', 'computed_hash',
            'ledger_hash', and loaded record metadata.
        """
        record_path = self.storage_dir / f"{record_id}.json"
        if not record_path.exists():
            raise FileNotFoundError(f"Audit record '{record_id}' not found at {record_path}")

        # 1. Read stored document
        with open(record_path, "r", encoding="utf-8") as f:
            raw_json = f.read()

        data_dict = json.loads(raw_json)
        reconstructed_record = OffChainAuditRecord(**data_dict)

        # 2. Recompute SHA-256 hash
        computed_hash = reconstructed_record.compute_sha256()

        # 3. Query ledger for committed hash
        committed_info = self.anchored_records.get(record_id)
        if committed_info is None:
            # Query blockchain directly from transaction history
            committed_hash = ""
            tx_id = ""
            for block in self.blockchain.chain:
                for tx in block.transactions:
                    if tx.data.get("record_id") == record_id:
                        committed_hash = tx.data.get("audit_hash", "")
                        tx_id = tx.tx_id
                        break
        else:
            committed_hash = committed_info["audit_hash"]
            tx_id = committed_info["tx_id"]

        is_valid = (computed_hash == committed_hash)

        if not is_valid:
            logger.warning(
                "TAMPER DETECTED in Audit Record %s | Computed: %s != Ledger: %s",
                record_id,
                computed_hash[:16] + "...",
                committed_hash[:16] + "...",
            )
        else:
            logger.info("Audit Record %s verified successfully (Hash Match)", record_id)

        return {
            "record_id": record_id,
            "is_valid": is_valid,
            "computed_hash": computed_hash,
            "ledger_hash": committed_hash,
            "tx_id": tx_id,
            "bond_id": reconstructed_record.bond_id,
            "predicted_spread_bps": reconstructed_record.predicted_spread_bps,
        }

    def demonstrate_tampering(
        self,
        record_id: str,
        tampered_field: str = "predicted_spread_bps",
        tampered_value: Any = 5.0,
    ) -> Dict[str, Any]:
        """Intentionally modify off-chain file to demonstrate tamper detection.

        Parameters:
        -----------
        record_id : str
            Record ID to tamper with.
        tampered_field : str
            Field key to modify.
        tampered_value : Any
            New unauthorized value to insert.

        Returns:
        --------
        Dict[str, Any]
            Verification outcome confirming that tampering was detected.
        """
        record_path = self.storage_dir / f"{record_id}.json"
        with open(record_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original_value = data.get(tampered_field)
        data[tampered_field] = tampered_value

        # Write corrupted record to off-chain storage
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=True)

        logger.warning(
            "Simulated unauthorized edit on '%s': %s -> %s",
            tampered_field,
            original_value,
            tampered_value,
        )

        # Execute standard verification
        verification = self.retrieve_and_verify_record(record_id)
        verification["tampered_field"] = tampered_field
        verification["original_value"] = original_value
        verification["tampered_value"] = tampered_value
        verification["tamper_detected"] = (verification["is_valid"] is False)

        return verification
