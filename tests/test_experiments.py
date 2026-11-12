"""
Unit Tests for Experiment Matrix Module — GreenFed-XAI
======================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from attacks.fl_attacks import AttackType
from experiments.run_matrix import (
    ExperimentResult,
    generate_experiment_plots,
    load_dataset_and_preprocessor,
    run_fl_benchmark_scenario,
    run_tau_sensitivity_sweep,
)
from fl.dataset_partitioner import ClientDataset
from models.nn_model import GreenBondNN


@pytest.fixture
def mock_fl_setup():
    """Create a minimal fast environment for testing experiment scenarios."""
    rng = np.random.default_rng(42)
    n_features = 12
    n_train = 60
    n_val = 20
    n_test = 20

    X_train = rng.normal(0, 1, size=(n_train, n_features)).astype(np.float32)
    y_train = np.abs(rng.normal(1.5, 0.5, size=(n_train,))).astype(np.float32) + 0.1

    X_val = rng.normal(0, 1, size=(n_val, n_features)).astype(np.float32)
    y_val = np.abs(rng.normal(1.5, 0.5, size=(n_val,))).astype(np.float32) + 0.1

    X_test = rng.normal(0, 1, size=(n_test, n_features)).astype(np.float32)
    y_test = np.abs(rng.normal(1.5, 0.5, size=(n_test,))).astype(np.float32) + 0.1

    df_dummy = type("MockDF", (), {"iloc": lambda self, idx: None})()

    # 4 clients
    clients = []
    for i in range(4):
        cds = ClientDataset(
            client_id=i,
            bank_name=f"Bank {i}",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            raw_df=df_dummy,
        )
        clients.append(cds)

    return clients, X_val, y_val, X_test, y_test, n_features


class TestExperimentExecution:
    """Tests for executing benchmark scenarios."""

    def test_clean_fl_scenario_execution(self, mock_fl_setup):
        """Clean FL scenario runs without errors and produces valid metrics."""
        clients, X_val, y_val, X_test, y_test, input_dim = mock_fl_setup
        device = torch.device("cpu")

        result = run_fl_benchmark_scenario(
            scenario_id=1,
            scenario_name="Test Clean Scenario",
            client_datasets=clients,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            input_dim=input_dim,
            device=device,
            defense_mode="none",
            attack_type=AttackType.NONE,
            num_rounds=1,
            local_epochs=1,
        )

        assert isinstance(result, ExperimentResult)
        assert result.test_rmse > 0.0
        assert result.test_mae_bps > 0.0
        assert result.test_mape > 0.0
        assert result.malicious_fraction_pct == 0.0

    def test_attacked_fl_with_blockchain_defense(self, mock_fl_setup):
        """Attacked FL with blockchain defense rejects malicious updates."""
        clients, X_val, y_val, X_test, y_test, input_dim = mock_fl_setup
        device = torch.device("cpu")

        result = run_fl_benchmark_scenario(
            scenario_id=2,
            scenario_name="Test Attacked Scenario",
            client_datasets=clients,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            input_dim=input_dim,
            device=device,
            defense_mode="blockchain_norm",
            attack_type=AttackType.GAUSSIAN_NOISE,
            tau=2.5,
            noise_scale=3.0,
            malicious_ids=[2, 3],  # 2 of 4 malicious (50%)
            num_rounds=1,
            local_epochs=1,
        )

        assert result.malicious_fraction_pct == pytest.approx(50.0)
        assert result.tpr_pct == 100.0  # Both malicious updates caught
        assert result.tp_count == 2
        assert result.fp_count == 0

    def test_tau_sensitivity_fast_sweep(self, mock_fl_setup):
        """Sensitivity sweep across 2 tau values produces structured DataFrame."""
        clients, X_val, y_val, X_test, y_test, input_dim = mock_fl_setup
        device = torch.device("cpu")

        tau_df = run_tau_sensitivity_sweep(
            client_datasets=clients,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            input_dim=input_dim,
            device=device,
            tau_values=[1.0, 5.0],
            num_rounds=1,
            noise_scale=2.5,
        )

        assert len(tau_df) == 2
        assert "tau" in tau_df.columns
        assert "tpr_pct" in tau_df.columns
        assert "fpr_pct" in tau_df.columns
        assert "test_mape" in tau_df.columns

    def test_plot_generation(self, tmp_path):
        """generate_experiment_plots generates both comparison and tau plots."""
        matrix_data = pd.DataFrame([
            {
                "scenario_name": "Scenario A",
                "test_mape": 3.5,
                "test_mae_bps": 7.5,
            },
            {
                "scenario_name": "Scenario B",
                "test_mape": 9.8,
                "test_mae_bps": 22.0,
            },
        ])
        tau_data = pd.DataFrame([
            {"tau": 1.5, "tpr_pct": 100.0, "fpr_pct": 10.0, "test_mape": 3.8},
            {"tau": 2.5, "tpr_pct": 100.0, "fpr_pct": 0.0, "test_mape": 3.5},
        ])

        generate_experiment_plots(matrix_data, tau_data, tmp_path)

        assert (tmp_path / "attack_matrix_comparison.png").exists()
        assert (tmp_path / "tau_sensitivity.png").exists()
