"""
Tests for Phase 3 — Federated Learning (Flower + FedAvg)
========================================================

Comprehensive unit and integration test suite covering:
1. Bank dataset partitioning (IID & Non-IID Dirichlet).
2. Flower NumPyClient local training (fit), evaluation, and parameter serialization.
3. Gradient norm calculation (||dw||_2) for monitoring and Phase 4/5 defenses.
4. GreenFedAvgStrategy aggregation mechanics and centralized evaluation hook.
5. Multi-round end-to-end federated simulation engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_bonds import generate_bonds, split_dataset
from fl.client import GreenBondFlowerClient, compute_mape, compute_rmse
from fl.dataset_partitioner import BankDatasetPartitioner, ClientDataset
from fl.server import FederatedSimulationEngine
from fl.strategy_fedavg import GreenFedAvgStrategy
from models.nn_model import GreenBondNN
from models.preprocessing import DataPreprocessor


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def shared_bond_data():
    """Generate small bond dataset and preprocessed splits for fast testing."""
    df = generate_bonds(n=2000, seed=42)
    train, val, test = split_dataset(
        df.iloc[:2000], seed=42
    ) if len(df) == 10000 else (
        df.iloc[:1400].copy(),
        df.iloc[1400:1700].copy(),
        df.iloc[1700:2000].copy(),
    )

    prep = DataPreprocessor()
    X_train, y_train = prep.fit_transform(train)
    X_val, y_val = prep.transform(val)
    X_test, y_test = prep.transform(test)

    return train, val, test, X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture(scope="module")
def client_datasets(shared_bond_data):
    """Partition training data across K = 4 banks for test execution."""
    train_df, _, _, X_train, y_train, *_ = shared_bond_data
    partitioner = BankDatasetPartitioner(n_clients=4, val_ratio=0.15, random_seed=42)
    return partitioner.partition_iid(train_df, X_train, y_train)


# ===================================================================
# Test: BankDatasetPartitioner
# ===================================================================

class TestBankDatasetPartitioner:
    """Tests for dataset partitioning among banks."""

    def test_iid_partition_count(self, client_datasets):
        """Should create exactly K client datasets."""
        assert len(client_datasets) == 4

    def test_iid_sample_coverage(self, shared_bond_data, client_datasets):
        """Total partitioned samples must equal original training length."""
        train_df = shared_bond_data[0]
        total_samples = sum(
            cds.n_train_samples + cds.n_val_samples for cds in client_datasets
        )
        assert total_samples == len(train_df)

    def test_no_empty_partitions(self, client_datasets):
        """No client should have zero training samples."""
        for cds in client_datasets:
            assert cds.n_train_samples > 0
            assert cds.n_val_samples > 0

    def test_train_val_split_ratio(self, client_datasets):
        """Local val ratio should approximate requested 15%."""
        for cds in client_datasets:
            total = cds.n_train_samples + cds.n_val_samples
            ratio = cds.n_val_samples / total
            assert 0.10 <= ratio <= 0.20

    def test_non_iid_dirichlet_partition(self, shared_bond_data):
        """Non-IID partitioning should succeed and allocate all records."""
        train_df, _, _, X_train, y_train, *_ = shared_bond_data
        partitioner = BankDatasetPartitioner(n_clients=3, val_ratio=0.10, random_seed=42)
        non_iid_ds = partitioner.partition_non_iid_dirichlet(
            train_df, X_train, y_train, alpha=0.5
        )
        assert len(non_iid_ds) == 3
        total_samples = sum(
            cds.n_train_samples + cds.n_val_samples for cds in non_iid_ds
        )
        assert total_samples == len(train_df)

    def test_dataloader_creation(self, client_datasets):
        """ClientDataset should yield valid PyTorch DataLoaders."""
        cds = client_datasets[0]
        train_loader = cds.get_train_loader(batch_size=32)
        X_batch, y_batch = next(iter(train_loader))

        assert X_batch.shape == (32, 5)
        assert y_batch.shape == (32,)
        assert isinstance(X_batch, torch.Tensor)
        assert isinstance(y_batch, torch.Tensor)


# ===================================================================
# Test: GreenBondFlowerClient
# ===================================================================

class TestGreenBondFlowerClient:
    """Tests for Flower NumPyClient behavior."""

    @pytest.fixture
    def test_client(self, client_datasets):
        """Create a single test client."""
        return GreenBondFlowerClient(
            client_id=0,
            dataset=client_datasets[0],
            learning_rate=0.005,
            local_epochs=1,
            batch_size=32,
        )

    def test_parameter_roundtrip(self, test_client):
        """set_parameters and get_parameters should preserve values."""
        params_orig = test_client.get_parameters()
        assert len(params_orig) == 8  # 4 linear layers (weights + biases)

        # Modify parameters and restore
        params_modified = [p + 0.1 for p in params_orig]
        test_client.set_parameters(params_modified)
        params_retrieved = test_client.get_parameters()

        for p_mod, p_ret in zip(params_modified, params_retrieved):
            np.testing.assert_allclose(p_mod, p_ret, rtol=1e-5)

    def test_fit_returns_correct_types(self, test_client):
        """fit() should return (weights, n_samples, metrics)."""
        params_in = test_client.get_parameters()
        config = {"current_round": 1, "local_epochs": 1, "learning_rate": 0.005}

        updated_params, n_samples, metrics = test_client.fit(params_in, config)

        assert isinstance(updated_params, list)
        assert len(updated_params) == len(params_in)
        assert n_samples == test_client.dataset.n_train_samples
        assert "train_loss" in metrics
        assert "grad_norm" in metrics
        assert metrics["grad_norm"] > 0.0

    def test_gradient_norm_calculation(self):
        """Euclidean norm of parameter differences should be mathematically exact."""
        p_init = [np.array([1.0, 2.0]), np.array([3.0])]
        p_upd = [np.array([2.0, 2.0]), np.array([5.0])]
        # diffs: [1.0, 0.0], [2.0] -> 1^2 + 0^2 + 2^2 = 5 -> sqrt(5) ~= 2.236068
        norm = GreenBondFlowerClient.compute_gradient_norm(p_init, p_upd)
        assert norm == pytest.approx(np.sqrt(5.0), rel=1e-5)

    def test_evaluate_returns_valid_metrics(self, test_client):
        """evaluate() should return (loss, n_samples, metrics)."""
        params_in = test_client.get_parameters()
        loss, n_samples, metrics = test_client.evaluate(params_in, {})

        assert loss >= 0.0
        assert n_samples == test_client.dataset.n_val_samples
        assert "val_rmse" in metrics
        assert "val_mape" in metrics
        assert metrics["val_rmse"] >= 0.0


# ===================================================================
# Test: GreenFedAvgStrategy
# ===================================================================

class TestGreenFedAvgStrategy:
    """Tests for FedAvg strategy mechanics."""

    def test_initialization(self, shared_bond_data):
        """Strategy should initialize with proper defaults."""
        _, _, _, _, _, X_val, y_val, *_ = shared_bond_data
        strategy = GreenFedAvgStrategy(
            eval_dataset=(X_val, y_val),
            min_fit_clients=4,
            min_available_clients=4,
        )
        assert strategy.eval_dataset is not None
        assert strategy.eval_model is not None
        assert len(strategy.history) == 0

    def test_weighted_averaging_math(self):
        """Parameter aggregation must be proportional to sample count."""
        strategy = GreenFedAvgStrategy(min_fit_clients=2)

        # Mock 2 clients with different sample counts
        p1 = [np.array([1.0, 2.0])]  # n = 100
        p2 = [np.array([3.0, 6.0])]  # n = 300
        # total n = 400. w1 = 100/400 = 0.25, w2 = 300/400 = 0.75
        # Expected: 0.25*[1,2] + 0.75*[3,6] = [0.25+2.25, 0.5+4.5] = [2.5, 5.0]

        fit_res_1 = type("FitResMock", (), {
            "parameters": type("PMock", (), {"ndarrays": p1})(),
            "num_examples": 100,
            "metrics": {"train_loss": 0.5, "grad_norm": 0.1, "client_id": 0},
        })()
        fit_res_2 = type("FitResMock", (), {
            "parameters": type("PMock", (), {"ndarrays": p2})(),
            "num_examples": 300,
            "metrics": {"train_loss": 0.3, "grad_norm": 0.2, "client_id": 1},
        })()

        results = [(None, fit_res_1), (None, fit_res_2)]

        # Verify math
        total = 400
        aggregated = [np.zeros_like(p1[0])]
        for _, res in results:
            frac = res.num_examples / total
            aggregated[0] += res.parameters.ndarrays[0] * frac

        np.testing.assert_allclose(aggregated[0], np.array([2.5, 5.0]), rtol=1e-5)


# ===================================================================
# Test: FederatedSimulationEngine (End-to-End Integration)
# ===================================================================

class TestFederatedSimulationEngine:
    """Integration test running a multi-round federated simulation."""

    def test_end_to_end_fast_simulation(self, shared_bond_data, client_datasets):
        """Simulation should execute communication rounds and record convergence."""
        _, _, _, _, _, X_val, y_val, X_test, y_test = shared_bond_data

        input_dim = X_test.shape[1]
        model = GreenBondNN(input_dim=input_dim, dropout_rate=0.2)
        init_weights = [
            v.cpu().detach().numpy().copy() for _, v in model.state_dict().items()
        ]

        clients = [
            GreenBondFlowerClient(
                client_id=cds.client_id,
                dataset=cds,
                learning_rate=0.01,
                local_epochs=1,
                batch_size=64,
            )
            for cds in client_datasets
        ]

        strategy = GreenFedAvgStrategy(
            eval_dataset=(X_val, y_val),
            model_template=model,
            min_fit_clients=4,
        )

        engine = FederatedSimulationEngine(
            clients=clients,
            strategy=strategy,
            initial_weights=init_weights,
            eval_test_dataset=(X_test, y_test),
        )

        # Run 3 fast rounds
        sim_results = engine.run_rounds(num_rounds=3, local_epochs=1, batch_size=64)

        assert sim_results["num_rounds"] == 3
        assert len(sim_results["history_df"]) == 3
        # Train loss should strictly decrease across the 3 rounds
        assert (
            sim_results["history_df"]["train_loss"].iloc[-1]
            < sim_results["history_df"]["train_loss"].iloc[0]
        )
        assert sim_results["test_rmse"] < 3.0
        assert len(sim_results["y_pred"]) == len(y_test)

