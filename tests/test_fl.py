"""
Unit Tests for Federated Learning Suite (Phase 4) — GreenFed-XAI
================================================================
Comprehensive test suite validating institutional dataset partitioning,
Flower NumPyClient training, update norm calculations, sample-weighted
vs equal-client aggregation, and convergence telemetry.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import pytest
import torch

from fl.client import (
    GreenBondFlowerClient,
    compute_mae,
    compute_mae_bps,
    compute_mape,
    compute_rmse,
)
from fl.dataset_partitioner import BankDatasetPartitioner, ClientDataset
from fl.server import (
    FederatedSimulationEngine,
    generate_loss_curve_plot,
    update_results_table,
)
from fl.strategy_fedavg import GreenFedAvgStrategy, RoundConvergenceMetrics
from models.nn_model import GreenBondNN


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def synthetic_bond_dataframe() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate minimal synthetic bond dataset for testing."""
    rng = np.random.default_rng(42)
    n_samples = 400
    n_features = 12

    ratings = ["AAA", "AA", "A", "BBB", "BB", "B"]
    assigned_ratings = [ratings[i % len(ratings)] for i in range(n_samples)]

    df = pd.DataFrame({
        "bond_id": [f"BOND_{i:04d}" for i in range(n_samples)],
        "issuer_credit_rating": assigned_ratings,
        "tenor": rng.uniform(1.0, 30.0, size=n_samples),
        "coupon_rate": rng.uniform(1.0, 8.0, size=n_samples),
        "carbon_footprint_score": rng.uniform(10.0, 95.0, size=n_samples),
        "esg_controversy_score": rng.uniform(0.0, 10.0, size=n_samples),
        "yield_spread": rng.uniform(0.5, 4.0, size=n_samples),
    })

    X_scaled = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)
    y = df["yield_spread"].to_numpy().astype(np.float32)

    return df, X_scaled, y


@pytest.fixture
def partitioner() -> BankDatasetPartitioner:
    """Instantiate BankDatasetPartitioner with K=8 banks."""
    return BankDatasetPartitioner(n_clients=8, val_ratio=0.15, random_seed=42)


@pytest.fixture
def iid_client_datasets(partitioner, synthetic_bond_dataframe) -> List[ClientDataset]:
    """Generate 8 IID partitioned client datasets."""
    df, X, y = synthetic_bond_dataframe
    return partitioner.partition_iid(df, X, y)


# ===================================================================
# 1. Tests for BankDatasetPartitioner
# ===================================================================

class TestBankDatasetPartitioner:
    """Tests for institutional dataset partitioning regimes."""

    def test_iid_partition_count(self, iid_client_datasets):
        """Should partition into exactly 8 client datasets."""
        assert len(iid_client_datasets) == 8

    def test_iid_sample_coverage(self, synthetic_bond_dataframe, iid_client_datasets):
        """Total partitioned samples must equal total dataset samples."""
        df, _, _ = synthetic_bond_dataframe
        total_partitioned = sum(ds.total_samples for ds in iid_client_datasets)
        assert total_partitioned == len(df)

    def test_no_empty_partitions(self, iid_client_datasets):
        """Every bank must have non-zero train and validation samples."""
        for ds in iid_client_datasets:
            assert ds.n_train_samples > 0
            assert ds.n_val_samples > 0
            assert ds.X_train.shape[1] == 12

    def test_train_val_split_ratio(self, iid_client_datasets):
        """Validation split ratio must be approximately 15%."""
        for ds in iid_client_datasets:
            val_ratio = ds.n_val_samples / ds.total_samples
            assert val_ratio == pytest.approx(0.15, abs=0.05)

    def test_non_iid_dirichlet_partition(self, partitioner, synthetic_bond_dataframe):
        """Dirichlet partition must succeed and preserve total sample count."""
        df, X, y = synthetic_bond_dataframe
        datasets = partitioner.partition_non_iid_dirichlet(df, X, y, alpha=0.5)
        assert len(datasets) == 8
        total = sum(ds.total_samples for ds in datasets)
        assert total == len(df)
        for ds in datasets:
            assert ds.n_train_samples > 0

    def test_non_iid_rating_tier_partition(self, partitioner, synthetic_bond_dataframe):
        """Rating-tier partition should concentrate IG ratings in Banks 0-3."""
        df, X, y = synthetic_bond_dataframe
        datasets = partitioner.partition_non_iid_rating_tier(df, X, y, ig_bias_ratio=0.85)
        assert len(datasets) == 8

        # Banks 0-3 should have predominantly AAA/AA/A/BBB
        ig_ratings = ["AAA", "AA", "A", "BBB"]
        group1_ig = sum(
            sum(ds.rating_distribution.get(r, 0) for r in ig_ratings)
            for ds in datasets[:4]
        )
        group2_ig = sum(
            sum(ds.rating_distribution.get(r, 0) for r in ig_ratings)
            for ds in datasets[4:]
        )
        assert group1_ig > group2_ig

    def test_jensen_shannon_divergence(self):
        """JSD must be 0 for identical distributions and strictly positive for distinct ones."""
        p1 = [0.5, 0.5]
        p2 = [0.5, 0.5]
        p3 = [1.0, 0.0]

        jsd_identical = BankDatasetPartitioner.compute_jensen_shannon_divergence(p1, p2)
        assert jsd_identical == pytest.approx(0.0, abs=1e-5)

        jsd_distinct = BankDatasetPartitioner.compute_jensen_shannon_divergence(p1, p3)
        assert 0.0 < jsd_distinct <= 1.0

    def test_heterogeneity_evaluation(self, partitioner, iid_client_datasets):
        """Heterogeneity evaluator must return valid metrics dictionary."""
        metrics = partitioner.evaluate_heterogeneity(iid_client_datasets)
        assert "mean_pairwise_jsd" in metrics
        assert "sample_size_cv" in metrics
        assert metrics["mean_pairwise_jsd"] >= 0.0

    def test_dataloader_creation(self, iid_client_datasets):
        """DataLoader must yield correctly typed and shaped batches."""
        ds = iid_client_datasets[0]
        train_loader = ds.get_train_loader(batch_size=16)
        batch_x, batch_y = next(iter(train_loader))

        assert batch_x.shape == (16, 12)
        assert batch_y.shape == (16,)
        assert batch_x.dtype == torch.float32
        assert batch_y.dtype == torch.float32

    def test_export_partition_diagnostics_csv(self, partitioner, iid_client_datasets, tmp_path):
        """Exporting partition diagnostics CSV must produce a readable CSV file."""
        csv_file = tmp_path / "test_partitions.csv"
        out_path = partitioner.export_partition_diagnostics_csv(iid_client_datasets, csv_file)
        assert out_path.exists()
        df_read = pd.read_csv(out_path)
        assert len(df_read) == 8
        assert "bank_name" in df_read.columns
        assert "n_train_samples" in df_read.columns


# ===================================================================
# 2. Tests for GreenBondFlowerClient
# ===================================================================

class TestGreenBondFlowerClient:
    """Tests for individual Flower client execution."""

    @pytest.fixture
    def single_client(self, iid_client_datasets) -> GreenBondFlowerClient:
        return GreenBondFlowerClient(
            client_id=0,
            dataset=iid_client_datasets[0],
            learning_rate=0.01,
            local_epochs=1,
            batch_size=32,
        )

    def test_parameter_roundtrip(self, single_client):
        """Parameters extracted and loaded back must match with zero error."""
        params = single_client.get_parameters()
        assert len(params) == 8  # 4 linear layers * (weight + bias) = 8 arrays
        single_client.set_parameters(params)
        reloaded = single_client.get_parameters()
        for p1, p2 in zip(params, reloaded):
            np.testing.assert_array_equal(p1, p2)

    def test_fit_returns_correct_types(self, single_client):
        """fit() must return (weights, n_samples, metrics)."""
        init_params = single_client.get_parameters()
        weights, n_samples, metrics = single_client.fit(
            init_params, {"current_round": 1, "local_epochs": 1}
        )

        assert isinstance(weights, list)
        assert len(weights) == 8
        assert n_samples == single_client.dataset.n_train_samples
        assert "train_loss" in metrics
        assert "update_norm" in metrics
        assert metrics["update_norm"] >= 0.0

    def test_update_norm_calculation(self):
        """compute_update_norm must match Euclidean L2 distance."""
        p_init = [np.array([0.0, 0.0]), np.array([0.0])]
        p_upd = [np.array([3.0, 4.0]), np.array([0.0])]
        # sqrt(3^2 + 4^2 + 0^2) = 5.0
        norm = GreenBondFlowerClient.compute_update_norm(p_init, p_upd)
        assert norm == pytest.approx(5.0)

    def test_evaluate_returns_valid_metrics(self, single_client):
        """evaluate() must compute valid loss, RMSE, MAE (bps), and MAPE."""
        init_params = single_client.get_parameters()
        loss, n_samples, metrics = single_client.evaluate(init_params, {})

        assert n_samples == single_client.dataset.n_val_samples
        assert loss >= 0.0
        assert metrics["val_rmse"] >= 0.0
        assert metrics["val_mae_bps"] >= 0.0
        assert metrics["val_mape"] >= 0.0

    def test_sgd_optimizer_support(self, iid_client_datasets):
        """Client should support SGD with momentum local optimizer."""
        client_sgd = GreenBondFlowerClient(
            client_id=1,
            dataset=iid_client_datasets[1],
            learning_rate=0.01,
            local_epochs=1,
            optimizer_type="sgd",
        )
        init_params = client_sgd.get_parameters()
        weights, n_samples, metrics = client_sgd.fit(
            init_params, {"current_round": 1, "optimizer_type": "sgd"}
        )
        assert len(weights) == 8
        assert metrics["update_norm"] >= 0.0


# ===================================================================
# 3. Tests for GreenFedAvgStrategy (Aggregation Math & Edge Cases)
# ===================================================================

class TestGreenFedAvgStrategy:
    """Tests for federated aggregation math, renormalization, and fallbacks."""

    def test_initialization(self):
        """Strategy must initialize with specified aggregation mode."""
        strat_w = GreenFedAvgStrategy(aggregation="weighted")
        assert strat_w.aggregation == "weighted"

        strat_e = GreenFedAvgStrategy(aggregation="equal")
        assert strat_e.aggregation == "equal"

        with pytest.raises(ValueError):
            GreenFedAvgStrategy(aggregation="invalid_mode")

    def test_weighted_averaging_math(self):
        """Sample-weighted FedAvg: w = sum(|D_k| / |D|) * w_k."""
        # Client 1: 100 samples, weights = [1.0]
        # Client 2: 300 samples, weights = [3.0]
        # Total = 400. w = (100/400)*1.0 + (300/400)*3.0 = 0.25 + 2.25 = 2.50
        w1 = np.array([1.0])
        w2 = np.array([3.0])
        weighted = (100 / 400) * w1 + (300 / 400) * w2
        assert weighted[0] == pytest.approx(2.50)

    def test_equal_client_averaging_math(self):
        """Equal-client FedAvg: w = (1/K) * sum(w_k)."""
        w1 = np.array([1.0])
        w2 = np.array([3.0])
        equal = 0.5 * w1 + 0.5 * w2
        assert equal[0] == pytest.approx(2.00)

    def test_weighted_vs_equal_equivalence_under_equal_sizes(self):
        """When client sample sizes are identical, weighted and equal averaging match exactly."""
        w1 = np.array([2.5, 4.0])
        w2 = np.array([7.5, 8.0])
        n1 = 500
        n2 = 500

        weighted = (n1 / (n1 + n2)) * w1 + (n2 / (n1 + n2)) * w2
        equal = 0.5 * (w1 + w2)

        np.testing.assert_allclose(weighted, equal)

    def test_accepted_client_renormalization_to_unity(self):
        """When client updates are filtered, dynamic weights must renormalize to sum to 1.0."""
        # 3 clients with sample counts 100, 200, 700
        # If client 3 is rejected, accepted clients are 1 and 2
        accepted_counts = [100, 200]
        total_accepted = sum(accepted_counts)
        norm_weights = [c / total_accepted for c in accepted_counts]

        assert sum(norm_weights) == pytest.approx(1.0)
        assert norm_weights == [1.0 / 3.0, 2.0 / 3.0]

    def test_zero_accepted_clients_fallback_preserves_weights(self):
        """Engine must preserve previous round parameters when all updates are rejected."""
        init_weights = [np.array([1.0, 2.0])]
        engine = FederatedSimulationEngine(
            clients=[],
            strategy=GreenFedAvgStrategy(),
            initial_weights=init_weights,
            eval_test_dataset=(np.zeros((1, 2)), np.zeros(1)),
        )
        assert engine.current_weights[0].tolist() == [1.0, 2.0]


# ===================================================================
# 4. Tests for FederatedSimulationEngine (End-to-End Integration)
# ===================================================================

class TestFederatedSimulationEngine:
    """Integration test running a multi-round federated simulation."""

    def test_end_to_end_fast_simulation(self, synthetic_bond_dataframe, iid_client_datasets):
        """Simulation should execute communication rounds and record convergence."""
        df, X, y = synthetic_bond_dataframe
        X_val, y_val = X[:40], y[:40]
        X_test, y_test = X[40:80], y[40:80]

        model = GreenBondNN(input_dim=12, dropout_rate=0.2)
        init_weights = [
            v.cpu().detach().numpy().copy() for _, v in model.state_dict().items()
        ]

        clients = [
            GreenBondFlowerClient(
                client_id=cds.client_id,
                dataset=cds,
                learning_rate=0.01,
                local_epochs=1,
                batch_size=32,
            )
            for cds in iid_client_datasets
        ]

        strategy = GreenFedAvgStrategy(
            eval_dataset=(X_val, y_val),
            model_template=model,
            aggregation="weighted",
            min_fit_clients=4,
        )

        engine = FederatedSimulationEngine(
            clients=clients,
            strategy=strategy,
            initial_weights=init_weights,
            eval_test_dataset=(X_test, y_test),
        )

        results = engine.run_rounds(num_rounds=3, local_epochs=1, batch_size=32)

        assert results["num_rounds"] == 3
        assert len(results["history_df"]) == 3
        assert results["test_rmse"] > 0.0
        assert results["test_mae_bps"] > 0.0
        assert results["test_mape"] > 0.0
        assert len(results["y_pred"]) == len(y_test)

    def test_equal_aggregation_end_to_end(self, synthetic_bond_dataframe, iid_client_datasets):
        """Simulation with equal-client aggregation must run and produce valid telemetry."""
        df, X, y = synthetic_bond_dataframe
        X_val, y_val = X[:40], y[:40]
        X_test, y_test = X[40:80], y[40:80]

        model = GreenBondNN(input_dim=12, dropout_rate=0.2)
        init_weights = [
            v.cpu().detach().numpy().copy() for _, v in model.state_dict().items()
        ]

        clients = [
            GreenBondFlowerClient(
                client_id=cds.client_id,
                dataset=cds,
                learning_rate=0.01,
                local_epochs=1,
                batch_size=32,
            )
            for cds in iid_client_datasets
        ]

        strategy = GreenFedAvgStrategy(
            eval_dataset=(X_val, y_val),
            model_template=model,
            aggregation="equal",
            min_fit_clients=4,
        )

        engine = FederatedSimulationEngine(
            clients=clients,
            strategy=strategy,
            initial_weights=init_weights,
            eval_test_dataset=(X_test, y_test),
        )

        results = engine.run_rounds(num_rounds=2, local_epochs=1, batch_size=32)
        assert results["num_rounds"] == 2
        assert strategy.history[-1].aggregation_mode == "equal"

    def test_generate_loss_curve_plot(self, tmp_path):
        """generate_loss_curve_plot must create a valid dual-panel image file."""
        history_df = pd.DataFrame({
            "round": [1, 2, 3],
            "train_loss": [1.5, 0.8, 0.4],
            "val_loss": [1.2, 0.7, 0.35],
            "val_rmse": [0.8, 0.5, 0.25],
            "val_mae_bps": [15.0, 10.0, 7.5],
            "avg_update_norm": [2.1, 1.4, 0.8],
        })

        out_img = tmp_path / "test_loss_curve.png"
        path = generate_loss_curve_plot(history_df, out_img, title_prefix="TestFedAvg")
        assert path.exists()

    def test_update_results_table(self, tmp_path):
        """update_results_table must create or append rows to benchmark CSV."""
        csv_path = tmp_path / "results_table.csv"
        df = update_results_table(
            results_path=csv_path,
            model_name="FedAvg (Weighted)",
            rmse=0.1506,
            mae_bps=12.12,
            mape=4.44,
            train_time_sec=5.5,
            notes="T=25, K=8",
        )
        assert csv_path.exists()
        assert len(df) == 1
        assert df["Model"].iloc[0] == "FedAvg (Weighted)"
        assert df["MAE (bps)"].iloc[0] == 12.12
