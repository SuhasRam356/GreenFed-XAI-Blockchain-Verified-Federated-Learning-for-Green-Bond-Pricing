"""
Unit Tests for Federated Learning Attacks Module — GreenFed-XAI
===============================================================
"""

import numpy as np
import pytest
import torch

from attacks.fl_attacks import (
    AttackType,
    MaliciousClientConfig,
    MaliciousGreenBondClient,
    compute_attack_confusion_matrix,
    create_federated_cohort,
)
from fl.dataset_partitioner import BankDatasetPartitioner, ClientDataset
from models.nn_model import GreenBondNN


@pytest.fixture
def mock_client_dataset():
    """Create a minimal synthetic ClientDataset for unit tests."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 12

    X = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)
    y = rng.normal(1.5, 0.5, size=(n_samples,)).astype(np.float32)

    df_dummy = type("MockDF", (), {"iloc": lambda self, idx: None})()

    dataset = ClientDataset(
        client_id=0,
        bank_name="Test Bank",
        X_train=X[:80],
        y_train=y[:80],
        X_val=X[80:],
        y_val=y[80:],
        raw_df=df_dummy,
    )
    return dataset


class TestMaliciousGreenBondClient:
    """Tests for MaliciousGreenBondClient functionality."""

    def test_honest_client_fallback(self, mock_client_dataset):
        """When AttackType.NONE, client behaves honestly."""
        cfg = MaliciousClientConfig(client_id=0, attack_type=AttackType.NONE)
        client = MaliciousGreenBondClient(
            client_id=0,
            dataset=mock_client_dataset,
            attack_config=cfg,
        )
        init_params = client.get_parameters()
        weights, n_samples, metrics = client.fit(
            init_params, {"current_round": 1, "local_epochs": 1}
        )
        assert not metrics["is_malicious"]
        assert metrics["attack_type"] == "none"
        assert metrics["update_norm"] >= 0.0

    def test_gaussian_noise_inflates_norm(self, mock_client_dataset):
        """Gaussian noise attack should substantially increase the update norm."""
        cfg = MaliciousClientConfig(
            client_id=0,
            attack_type=AttackType.GAUSSIAN_NOISE,
            noise_scale=2.5,
            seed=42,
        )
        client = MaliciousGreenBondClient(
            client_id=0,
            dataset=mock_client_dataset,
            attack_config=cfg,
        )
        init_params = client.get_parameters()
        weights, n_samples, metrics = client.fit(
            init_params, {"current_round": 1, "local_epochs": 1}
        )

        assert metrics["is_malicious"]
        assert metrics["attack_type"] == "gaussian_noise"
        # Gaussian noise with scale 2.5 across thousands of weights yields high norm >> 2.5
        assert metrics["update_norm"] > 2.5
        assert metrics["update_norm"] > metrics["honest_update_norm"]

    def test_adaptive_perturbation_under_threshold(self, mock_client_dataset):
        """Adaptive perturbation must constrain update norm strictly below tau."""
        tau = 2.5
        target_norm = 2.2
        cfg = MaliciousClientConfig(
            client_id=0,
            attack_type=AttackType.ADAPTIVE_PERTURBATION,
            target_norm=target_norm,
            seed=42,
        )
        client = MaliciousGreenBondClient(
            client_id=0,
            dataset=mock_client_dataset,
            attack_config=cfg,
        )
        init_params = client.get_parameters()
        weights, n_samples, metrics = client.fit(
            init_params, {"current_round": 1, "local_epochs": 1, "tau": tau}
        )

        assert metrics["is_malicious"]
        assert metrics["attack_type"] == "adaptive_perturbation"
        assert metrics["update_norm"] == pytest.approx(target_norm, rel=1e-3)
        assert metrics["update_norm"] < tau

    def test_sign_flip_reversal(self, mock_client_dataset):
        """Sign flip should reverse optimization direction."""
        cfg = MaliciousClientConfig(
            client_id=0,
            attack_type=AttackType.SIGN_FLIP,
            noise_scale=1.0,
            seed=42,
        )
        client = MaliciousGreenBondClient(
            client_id=0,
            dataset=mock_client_dataset,
            attack_config=cfg,
        )
        init_params = client.get_parameters()
        weights, n_samples, metrics = client.fit(
            init_params, {"current_round": 1, "local_epochs": 1}
        )
        assert metrics["is_malicious"]
        assert metrics["attack_type"] == "sign_flip"


class TestCohortCreation:
    """Tests for consortium cohort generation."""

    def test_cohort_fractions_and_selection(self, mock_client_dataset):
        """2 malicious clients out of 8 must equal exactly 25.0%."""
        datasets = []
        for i in range(8):
            ds = ClientDataset(
                client_id=i,
                bank_name=f"Bank {i}",
                X_train=mock_client_dataset.X_train,
                y_train=mock_client_dataset.y_train,
                X_val=mock_client_dataset.X_val,
                y_val=mock_client_dataset.y_val,
                raw_df=mock_client_dataset.raw_df,
            )
            datasets.append(ds)

        cohort, meta = create_federated_cohort(
            client_datasets=datasets,
            malicious_client_ids=[6, 7],
            attack_type=AttackType.GAUSSIAN_NOISE,
        )

        assert len(cohort) == 8
        assert meta["n_honest"] == 6
        assert meta["n_malicious"] == 2
        assert meta["malicious_fraction_pct"] == pytest.approx(25.0)
        assert meta["malicious_client_ids"] == [6, 7]

        # Verify class types
        assert not isinstance(cohort[0], MaliciousGreenBondClient)
        assert isinstance(cohort[6], MaliciousGreenBondClient)
        assert isinstance(cohort[7], MaliciousGreenBondClient)


class TestConfusionMatrix:
    """Tests for attack confusion matrix calculation."""

    def test_perfect_detection(self):
        """All malicious updates rejected, all honest accepted."""
        # 2 rounds, 8 clients per round, clients 6 & 7 malicious
        audit_history = [
            {"accepted_clients": [0, 1, 2, 3, 4, 5], "rejected_clients": [6, 7]},
            {"accepted_clients": [0, 1, 2, 3, 4, 5], "rejected_clients": [6, 7]},
        ]
        cm = compute_attack_confusion_matrix(audit_history, malicious_client_ids=[6, 7])

        assert cm["true_positives"] == 4
        assert cm["false_positives"] == 0
        assert cm["true_negatives"] == 12
        assert cm["false_negatives"] == 0
        assert cm["tpr"] == 1.0
        assert cm["fpr"] == 0.0

    def test_zero_defense_detection(self):
        """No updates rejected (plain FedAvg under attack)."""
        audit_history = [
            {"accepted_clients": [0, 1, 2, 3, 4, 5, 6, 7], "rejected_clients": []},
            {"accepted_clients": [0, 1, 2, 3, 4, 5, 6, 7], "rejected_clients": []},
        ]
        cm = compute_attack_confusion_matrix(audit_history, malicious_client_ids=[6, 7])

        assert cm["true_positives"] == 0
        assert cm["false_negatives"] == 4
        assert cm["true_negatives"] == 12
        assert cm["false_positives"] == 0
        assert cm["tpr"] == 0.0
        assert cm["fpr"] == 0.0
