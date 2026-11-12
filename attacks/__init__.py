"""
GreenFed-XAI Attacks Module
===========================
Adversarial attack implementations and evaluation utilities for federated
learning on green bond pricing.
"""

from attacks.fl_attacks import (
    AttackType,
    MaliciousClientConfig,
    MaliciousGreenBondClient,
    compute_attack_confusion_matrix,
    create_federated_cohort,
)

__all__ = [
    "AttackType",
    "MaliciousClientConfig",
    "MaliciousGreenBondClient",
    "create_federated_cohort",
    "compute_attack_confusion_matrix",
]

