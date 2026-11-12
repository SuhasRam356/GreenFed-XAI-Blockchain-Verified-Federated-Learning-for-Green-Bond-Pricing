r"""
Federated Learning Attack Implementations — GreenFed-XAI
========================================================

This module implements production-grade adversarial attack simulations for
federated learning, separating sensor data tampering (Phase 2) from model
poisoning attacks on federated model updates (Phase 6).

Scientific Constraints & Theoretical Context:
---------------------------------------------
1. Attack Separation:
   - Malicious IoT sensors corrupt reported telemetry before preprocessing
     (addressed by the off-chain/on-chain OracleValidator).
   - Malicious FL clients corrupt local model parameter updates (\Delta w_k)
     during decentralized optimization (addressed by the SmartContract
     norm filter and cryptographic hash commitment).
2. Explicit Fractions:
   - In a consortium of K = 8 banks, 2 compromised clients correspond to:
     2 / 8 = 25.0% malicious fraction (correcting the paper's "20%" misnomer).
3. Evaluated Threat Models:
   - Unconstrained Gaussian Noise: \Delta w_k = \Delta w_k + \mathcal{N}(0, \sigma^2),
     generating large updates (||Δw||_2 >> \tau = 2.5) that severely disrupt global
     convergence if unmitigated.
   - Constrained / Adaptive Perturbation: An evasion attack where the adversary
     projects or scales the perturbation such that ||Δw_{poison}||_2 <= \tau * (1 - \epsilon),
     demonstrating the theoretical boundary where norm-only defenses cannot detect
     in-distribution malicious drift.
   - Directional Sign-Flip: Inverting local optimization direction to maximize loss.
   - Payload Tampering: Modifying model weights after hash submission to test
     SHA-256 cryptographic non-repudiation and transmission integrity.
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from fl.client import GreenBondFlowerClient
from fl.dataset_partitioner import ClientDataset
from models.nn_model import GreenBondNN

logger = logging.getLogger(__name__)


class AttackType(str, Enum):
    """Categorization of federated learning adversarial threat models."""

    NONE = "none"
    GAUSSIAN_NOISE = "gaussian_noise"
    ADAPTIVE_PERTURBATION = "adaptive_perturbation"
    SIGN_FLIP = "sign_flip"
    PAYLOAD_TAMPER = "payload_tamper"


class MaliciousClientConfig:
    """Configuration hyperparameters defining an adversarial client's behavior.

    Parameters:
    -----------
    client_id : int
        Bank ID of the compromised participant (0 to K-1).
    attack_type : AttackType
        The mathematical attack variant to execute.
    noise_scale : float
        Magnitude/standard deviation of noise or perturbation factor.
    target_norm : Optional[float]
        Explicit norm ceiling for adaptive attacks (e.g. 2.4 when \tau = 2.5).
    seed : int
        Pseudorandom seed for reproducible perturbation generation.
    """

    def __init__(
        self,
        client_id: int,
        attack_type: AttackType = AttackType.GAUSSIAN_NOISE,
        noise_scale: float = 2.0,
        target_norm: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        self.client_id = client_id
        self.attack_type = (
            attack_type if isinstance(attack_type, AttackType) else AttackType(attack_type)
        )
        self.noise_scale = float(noise_scale)
        self.target_norm = float(target_norm) if target_norm is not None else None
        self.seed = int(seed)

    def __repr__(self) -> str:
        return (
            f"MaliciousClientConfig(client_id={self.client_id}, "
            f"type='{self.attack_type.value}', noise_scale={self.noise_scale}, "
            f"target_norm={self.target_norm}, seed={self.seed})"
        )


class MaliciousGreenBondClient(GreenBondFlowerClient):
    """Adversarial Flower client capable of injecting targeted perturbations.

    Subclasses ``GreenBondFlowerClient`` and overrides ``fit()`` to apply
    adversarial modifications to the computed parameter updates while keeping
    dataset access and internal mechanics realistic.

    Parameters:
    -----------
    client_id : int
        Unique integer ID of this bank.
    dataset : ClientDataset
        The private training/validation dataset of this bank.
    attack_config : MaliciousClientConfig
        Adversarial configuration describing the attack to inject.
    model : Optional[GreenBondNN]
        Neural network model instance.
    device : Optional[torch.device]
        Computation device (CPU/CUDA).
    learning_rate : float
        Local optimization learning rate.
    local_epochs : int
        Local training epochs per communication round.
    batch_size : int
        Mini-batch size.
    weight_decay : float
        L2 regularization weight decay.
    """

    def __init__(
        self,
        client_id: int,
        dataset: ClientDataset,
        attack_config: MaliciousClientConfig,
        model: Optional[GreenBondNN] = None,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        local_epochs: int = 3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            model=model,
            device=device,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
        )
        self.attack_config = attack_config
        self.rng = np.random.default_rng(attack_config.seed + client_id * 1000)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Perform local training and inject configured adversarial perturbation.

        Parameters:
        -----------
        parameters : List[np.ndarray]
            Current global model weights w_t broadcast by the server.
        config : Dict[str, Any]
            Round hyperparameters from the server (e.g. tau threshold).

        Returns:
        --------
        Tuple[List[np.ndarray], int, Dict[str, Any]]:
            - Poisoned local weights w_{t+1}^{(k)}.
            - Total training examples (|D_k|).
            - Metrics dictionary including update norms and attack flags.
        """
        # 1. Execute normal local training to obtain honest reference update
        honest_weights, num_samples, honest_metrics = super().fit(parameters, config)
        w_init = [p.copy() for p in parameters]
        honest_update = [w_h - w_i for w_h, w_i in zip(honest_weights, w_init)]
        honest_norm = self.compute_gradient_norm(w_init, honest_weights)

        current_round = int(config.get("current_round", 1))
        tau = float(config.get("tau", 2.5))

        attack_type = self.attack_config.attack_type

        # 2. Inject adversarial payload
        if attack_type == AttackType.GAUSSIAN_NOISE:
            # Large unconstrained zero-mean Gaussian noise: Δw_k = Δw_k + N(0, σ^2)
            poisoned_update = []
            for delta_layer in honest_update:
                noise = self.rng.normal(
                    loc=0.0,
                    scale=self.attack_config.noise_scale,
                    size=delta_layer.shape,
                ).astype(delta_layer.dtype)
                poisoned_update.append(delta_layer + noise)

            poisoned_weights = [
                w_i + d_p for w_i, d_p in zip(w_init, poisoned_update)
            ]

        elif attack_type == AttackType.ADAPTIVE_PERTURBATION:
            # Constrained evasion attack: crafted adversarial direction scaled
            # to stay strictly below tau: ||Δw||_2 <= min(target_norm, tau * 0.95)
            target = (
                self.attack_config.target_norm
                if self.attack_config.target_norm is not None
                else (tau * 0.95)
            )

            # Generate random adversarial direction or inverted gradient
            raw_perturbation = [
                self.rng.normal(loc=0.0, scale=1.0, size=delta_layer.shape).astype(delta_layer.dtype)
                for delta_layer in honest_update
            ]
            raw_norm = float(
                np.sqrt(sum(float(np.sum(p ** 2)) for p in raw_perturbation))
            )
            scaling = target / max(raw_norm, 1e-8)

            poisoned_update = [p * scaling for p in raw_perturbation]
            poisoned_weights = [
                w_i + d_p for w_i, d_p in zip(w_init, poisoned_update)
            ]

        elif attack_type == AttackType.SIGN_FLIP:
            # Reversal of local descent direction: Δw_k = -γ * Δw_honest
            gamma = self.attack_config.noise_scale
            poisoned_update = [-d * gamma for d in honest_update]
            poisoned_weights = [
                w_i + d_p for w_i, d_p in zip(w_init, poisoned_update)
            ]

        elif attack_type == AttackType.PAYLOAD_TAMPER:
            # Emulates post-hash transmission corruption: submits one update
            # with intentional layer perturbation
            poisoned_weights = [w.copy() for w in honest_weights]
            if len(poisoned_weights) > 0:
                poisoned_weights[0] = poisoned_weights[0] + 1.0

        else:
            # Clean / honest fallback
            poisoned_weights = honest_weights

        # 3. Compute true update norm of the returned parameters
        poison_norm = self.compute_gradient_norm(w_init, poisoned_weights)

        metrics = {
            "client_id": self.client_id,
            "bank_name": self.dataset.bank_name,
            "train_loss": honest_metrics.get("train_loss", 0.0),
            "grad_norm": poison_norm,
            "update_norm": poison_norm,
            "honest_update_norm": honest_norm,
            "is_malicious": attack_type != AttackType.NONE,
            "attack_type": attack_type.value,
            "round": current_round,
        }

        logger.debug(
            "[Round %d] Malicious Bank %d (%s) injected '%s' | Honest Norm: %.4f | Poisoned Norm: %.4f",
            current_round,
            self.client_id,
            self.dataset.bank_name,
            attack_type.value,
            honest_norm,
            poison_norm,
        )

        return poisoned_weights, num_samples, metrics


def create_federated_cohort(
    client_datasets: List[ClientDataset],
    malicious_client_ids: Optional[Sequence[int]] = None,
    attack_type: AttackType = AttackType.GAUSSIAN_NOISE,
    noise_scale: float = 2.0,
    target_norm: Optional[float] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
    learning_rate: float = 0.001,
    local_epochs: int = 3,
    batch_size: int = 64,
) -> Tuple[List[Union[GreenBondFlowerClient, MaliciousGreenBondClient]], Dict[str, Any]]:
    """Assemble a consortium of honest and malicious clients for FL experiments.

    Parameters:
    -----------
    client_datasets : List[ClientDataset]
        List of partitioned datasets (one per bank).
    malicious_client_ids : Optional[Sequence[int]]
        Explicit indices of banks to compromise. If None and attack_type is not NONE,
        defaults to the last 2 banks (Bank 6 and Bank 7).
    attack_type : AttackType
        Threat model to assign to compromised banks.
    noise_scale : float
        Noise standard deviation or perturbation factor.
    target_norm : Optional[float]
        Norm target for adaptive perturbation attack.
    seed : int
        RNG seed for reproducibility.
    device : Optional[torch.device]
        Torch computation device.
    learning_rate : float
        Client learning rate.
    local_epochs : int
        Client local training epochs.
    batch_size : int
        Client mini-batch size.

    Returns:
    --------
    Tuple[List[Client], Dict[str, Any]]:
        - Cohort of Flower client instances.
        - Consortium metadata (counts, fractions, malicious IDs).
    """
    total_clients = len(client_datasets)
    if attack_type == AttackType.NONE:
        malicious_ids_set = set()
    else:
        if malicious_client_ids is None:
            # Default: 2 malicious out of 8 banks
            malicious_ids_set = {total_clients - 2, total_clients - 1} if total_clients >= 2 else {0}
        else:
            malicious_ids_set = set(malicious_client_ids)

    n_malicious = len(malicious_ids_set)
    n_honest = total_clients - n_malicious
    malicious_fraction = (n_malicious / total_clients) * 100.0 if total_clients > 0 else 0.0

    logger.info(
        "Consortium Cohort: %d total banks | %d honest, %d malicious "
        "(%.1f%% malicious fraction; e.g. 2/8 = 25.0%%) | Attack: %s",
        total_clients,
        n_honest,
        n_malicious,
        malicious_fraction,
        attack_type.value if isinstance(attack_type, AttackType) else str(attack_type),
    )

    cohort: List[Union[GreenBondFlowerClient, MaliciousGreenBondClient]] = []

    for cds in client_datasets:
        cid = cds.client_id
        if cid in malicious_ids_set:
            cfg = MaliciousClientConfig(
                client_id=cid,
                attack_type=attack_type,
                noise_scale=noise_scale,
                target_norm=target_norm,
                seed=seed + cid,
            )
            client = MaliciousGreenBondClient(
                client_id=cid,
                dataset=cds,
                attack_config=cfg,
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                batch_size=batch_size,
            )
        else:
            client = GreenBondFlowerClient(
                client_id=cid,
                dataset=cds,
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                batch_size=batch_size,
            )
        cohort.append(client)

    meta = {
        "total_clients": total_clients,
        "n_honest": n_honest,
        "n_malicious": n_malicious,
        "malicious_fraction_pct": malicious_fraction,
        "malicious_client_ids": sorted(list(malicious_ids_set)),
        "attack_type": attack_type.value if isinstance(attack_type, AttackType) else str(attack_type),
        "noise_scale": noise_scale,
        "target_norm": target_norm,
        "seed": seed,
    }

    return cohort, meta


def compute_attack_confusion_matrix(
    audit_history: Sequence[Any],
    malicious_client_ids: Sequence[int],
    total_clients_per_round: int = 8,
) -> Dict[str, Any]:
    """Calculate explicit confusion matrix metrics for update verification.

    Theoretical Formulas with Denominators:
    ---------------------------------------
    - True Positives (TP): Malicious updates correctly rejected.
    - False Positives (FP): Honest updates incorrectly rejected.
    - True Negatives (TN): Honest updates correctly accepted.
    - False Negatives (FN): Malicious updates incorrectly accepted.

    - True Positive Rate (TPR / Detection Rate):
          TPR = TP / (TP + FN) = TP / Total Malicious Updates Submitted
    - False Positive Rate (FPR / False Alarm Rate):
          FPR = FP / (FP + TN) = FP / Total Honest Updates Submitted

    Parameters:
    -----------
    audit_history : Sequence[Any]
        List of ``RoundBlockchainAudit`` objects or dictionaries from each round.
    malicious_client_ids : Sequence[int]
        Ground-truth IDs of compromised banks.
    total_clients_per_round : int
        Expected total submissions per round (default: 8).

    Returns:
    --------
    Dict[str, Any]
        Dictionary with raw counts, rates, and explicit formulas.
    """
    malicious_set = set(malicious_client_ids)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for audit in audit_history:
        # Extract accepted and rejected client lists
        if hasattr(audit, "accepted_clients"):
            accepted = set(audit.accepted_clients)
            rejected = set(audit.rejected_clients)
        elif isinstance(audit, dict):
            accepted = set(audit.get("accepted_clients", []))
            rejected = set(audit.get("rejected_clients", []))
        else:
            continue

        for cid in accepted:
            if cid in malicious_set:
                fn += 1  # Poisoned update slipped through
            else:
                tn += 1  # Honest update passed

        for cid in rejected:
            if cid in malicious_set:
                tp += 1  # Poisoned update caught
            else:
                fp += 1  # Honest update rejected

    total_malicious = tp + fn
    total_honest = fp + tn

    tpr = (tp / total_malicious) if total_malicious > 0 else 0.0
    fpr = (fp / total_honest) if total_honest > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_malicious_updates": total_malicious,
        "total_honest_updates": total_honest,
        "tpr": float(tpr),
        "tpr_pct": float(tpr * 100.0),
        "fpr": float(fpr),
        "fpr_pct": float(fpr * 100.0),
        "precision": float(precision),
        "detection_rate": float(tpr * 100.0),
        "false_alarm_rate": float(fpr * 100.0),
        "formula_tpr": f"{tp} / {total_malicious} ({tpr * 100.0:.2f}%)",
        "formula_fpr": f"{fp} / {total_honest} ({fpr * 100.0:.2f}%)",
    }
