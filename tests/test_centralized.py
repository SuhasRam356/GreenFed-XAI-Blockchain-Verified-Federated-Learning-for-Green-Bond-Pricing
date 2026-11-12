"""
Tests for Phase 2 — Centralized Baselines
==========================================
Tests cover:
  - Data preprocessing (encoding, scaling, shapes)
  - Neural network architecture (layers, parameters, forward pass)
  - Metric computation (RMSE, MAPE)
  - Each baseline model (Linear Regression, XGBoost, Centralized NN)
  - Results table integrity
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_bonds import generate_bonds, split_dataset
from models.nn_model import GreenBondNN
from models.preprocessing import DataPreprocessor, RATING_ORDER, FEATURE_COLS
from models.train_centralized import (
    compute_rmse,
    compute_mae,
    compute_mae_bps,
    compute_mape,
    count_near_zero_targets,
    train_linear_regression,
    train_xgboost,
    train_neural_network,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def bond_data():
    """Generate and split bond data once for all tests."""
    df = generate_bonds(n=10_000, seed=42)
    train, val, test = split_dataset(df, seed=42)
    return train, val, test


@pytest.fixture(scope="module")
def preprocessed_data(bond_data):
    """Preprocess the bond data once for all tests."""
    train, val, test = bond_data
    prep = DataPreprocessor()
    X_train, y_train = prep.fit_transform(train)
    X_val, y_val = prep.transform(val)
    X_test, y_test = prep.transform(test)
    return prep, X_train, y_train, X_val, y_val, X_test, y_test


# ===================================================================
# Test: DataPreprocessor
# ===================================================================

class TestDataPreprocessor:
    """Tests for the preprocessing pipeline."""

    def test_fit_transform_shapes(self, preprocessed_data):
        """Output shapes must match expected dimensions."""
        prep, X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_data
        assert X_train.shape == (7000, 5), f"X_train shape: {X_train.shape}"
        assert X_val.shape == (1500, 5), f"X_val shape: {X_val.shape}"
        assert X_test.shape == (1500, 5), f"X_test shape: {X_test.shape}"
        assert y_train.shape == (7000,)
        assert y_val.shape == (1500,)
        assert y_test.shape == (1500,)

    def test_output_dtype(self, preprocessed_data):
        """Features and targets should be float32."""
        _, X_train, y_train, *_ = preprocessed_data
        assert X_train.dtype == np.float32
        assert y_train.dtype == np.float32

    def test_train_features_standardized(self, preprocessed_data):
        """Training features should have ~zero mean and ~unit std."""
        _, X_train, *_ = preprocessed_data
        means = np.abs(X_train.mean(axis=0))
        stds = X_train.std(axis=0)
        assert np.all(means < 0.1), f"Means not near zero: {means}"
        assert np.all(np.abs(stds - 1.0) < 0.1), f"Stds not near 1: {stds}"

    def test_no_nan_in_output(self, preprocessed_data):
        """No NaN values in preprocessed data."""
        _, X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_data
        assert not np.isnan(X_train).any()
        assert not np.isnan(y_train).any()
        assert not np.isnan(X_test).any()
        assert not np.isnan(y_test).any()

    def test_rating_encoding_order(self):
        """Credit ratings must be ordinal-encoded in correct order."""
        df = pd.DataFrame({
            "issuer_credit_rating": RATING_ORDER,
            "tenor": [10] * 6,
            "coupon_rate": [5.0] * 6,
            "esg_controversy_score": [5.0] * 6,
            "carbon_footprint_score": [50.0] * 6,
            "yield_spread": [1.0] * 6,
        })
        prep = DataPreprocessor()
        X, y = prep.fit_transform(df)
        # After scaling, first column should still be monotonically increasing
        raw_encoded = prep.rating_encoder.transform(
            np.array(RATING_ORDER).reshape(-1, 1)
        ).flatten()
        expected = [0, 1, 2, 3, 4, 5]
        np.testing.assert_array_equal(raw_encoded, expected)

    def test_feature_names(self, preprocessed_data):
        """get_feature_names() should return correct names."""
        prep = preprocessed_data[0]
        names = prep.get_feature_names()
        assert len(names) == 5
        assert "credit_rating_encoded" in names
        assert "tenor" in names
        assert "carbon_footprint_score" in names

    def test_fit_required_before_transform(self):
        """Calling transform without fit should raise."""
        prep = DataPreprocessor()
        df = pd.DataFrame({
            "issuer_credit_rating": ["AAA"],
            "tenor": [10],
            "coupon_rate": [5.0],
            "esg_controversy_score": [5.0],
            "carbon_footprint_score": [50.0],
            "yield_spread": [1.0],
        })
        with pytest.raises(AssertionError, match="fit"):
            prep.transform(df)


# ===================================================================
# Test: GreenBondNN architecture
# ===================================================================

class TestGreenBondNN:
    """Tests for the neural network model."""

    def test_architecture_layers(self):
        """Model should have the correct layer structure."""
        model = GreenBondNN(input_dim=5)
        layers = list(model.network)
        # Expected: Linear, ReLU, Dropout, Linear, ReLU, Dropout,
        #           Linear, ReLU, Dropout, Linear
        assert len(layers) == 10
        assert isinstance(layers[0], torch.nn.Linear)   # 5→128
        assert isinstance(layers[1], torch.nn.ReLU)
        assert isinstance(layers[2], torch.nn.Dropout)
        assert isinstance(layers[3], torch.nn.Linear)   # 128→64
        assert isinstance(layers[6], torch.nn.Linear)   # 64→32
        assert isinstance(layers[9], torch.nn.Linear)   # 32→1

    def test_layer_sizes(self):
        """Hidden layers must be 128, 64, 32 as per the paper."""
        model = GreenBondNN(input_dim=5)
        linear_layers = [l for l in model.network if isinstance(l, torch.nn.Linear)]
        assert linear_layers[0].in_features == 5
        assert linear_layers[0].out_features == 128
        assert linear_layers[1].in_features == 128
        assert linear_layers[1].out_features == 64
        assert linear_layers[2].in_features == 64
        assert linear_layers[2].out_features == 32
        assert linear_layers[3].in_features == 32
        assert linear_layers[3].out_features == 1

    def test_dropout_rate(self):
        """Dropout should be 0.2."""
        model = GreenBondNN(input_dim=5, dropout_rate=0.2)
        dropout_layers = [l for l in model.network if isinstance(l, torch.nn.Dropout)]
        assert len(dropout_layers) == 3
        for d in dropout_layers:
            assert d.p == 0.2

    def test_forward_pass_shape(self):
        """Forward pass should return correct output shape."""
        model = GreenBondNN(input_dim=5)
        x = torch.randn(32, 5)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (32,), f"Output shape: {out.shape}"

    def test_single_sample_forward(self):
        """Single sample should also work."""
        model = GreenBondNN(input_dim=5)
        x = torch.randn(1, 5)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1,)

    def test_parameter_count(self):
        """Check total parameter count is reasonable."""
        model = GreenBondNN(input_dim=5)
        n_params = model.count_parameters()
        # 5*128+128 + 128*64+64 + 64*32+32 + 32*1+1 = 640+128+8192+64+2048+32+32+1 = 11137
        expected = (5 * 128 + 128) + (128 * 64 + 64) + (64 * 32 + 32) + (32 * 1 + 1)
        assert n_params == expected, f"Expected {expected}, got {n_params}"

    def test_output_is_finite(self):
        """No NaN or Inf in output."""
        model = GreenBondNN(input_dim=5)
        x = torch.randn(100, 5)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()


# ===================================================================
# Test: Metrics
# ===================================================================

class TestMetrics:
    """Tests for RMSE and MAPE computation."""

    def test_rmse_perfect_prediction(self):
        """RMSE should be 0 for perfect predictions."""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        """RMSE for a known case."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        expected = np.sqrt(np.mean([0.01, 0.01, 0.01]))
        assert compute_rmse(y_true, y_pred) == pytest.approx(expected, rel=1e-5)

    def test_mape_perfect_prediction(self):
        """MAPE should be 0 for perfect predictions."""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mape(y, y) == pytest.approx(0.0)

    def test_mape_known_value(self):
        """MAPE for a known case."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # |10/100| + |20/200| = 0.1 + 0.1 = 0.2 → 10%
        expected = 10.0
        assert compute_mape(y_true, y_pred) == pytest.approx(expected, rel=1e-5)

    def test_rmse_non_negative(self):
        """RMSE is always non-negative."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(1.5, 0.5, 100)
        y_pred = rng.normal(1.5, 0.5, 100)
        assert compute_rmse(y_true, y_pred) >= 0

    def test_mape_non_negative(self):
        """MAPE is always non-negative."""
        rng = np.random.default_rng(42)
        y_true = np.abs(rng.normal(1.5, 0.5, 100)) + 0.1
        y_pred = np.abs(rng.normal(1.5, 0.5, 100)) + 0.1
        assert compute_mape(y_true, y_pred) >= 0

    def test_mae_and_bps(self):
        """MAE in basis points should be exactly 100 * MAE percentage."""
        y_true = np.array([2.0, 3.0])
        y_pred = np.array([2.1, 2.9])
        mae = compute_mae(y_true, y_pred)
        mae_bps = compute_mae_bps(y_true, y_pred)
        assert mae == pytest.approx(0.1)
        assert mae_bps == pytest.approx(10.0)

    def test_count_near_zero(self):
        """Should accurately identify targets close to zero."""
        y = np.array([0.01, 0.04, 0.06, 1.50])
        assert count_near_zero_targets(y, threshold=0.05) == 2


# ===================================================================
# Test: Model Training (lightweight)
# ===================================================================

class TestLinearRegression:
    """Tests for Linear Regression training."""

    def test_produces_predictions(self, preprocessed_data):
        """LR should produce predictions for all test samples."""
        _, X_train, y_train, _, _, X_test, y_test = preprocessed_data
        result = train_linear_regression(X_train, y_train, X_test, y_test)
        assert len(result["y_pred"]) == len(y_test)

    def test_rmse_reasonable(self, preprocessed_data):
        """LR RMSE should be in a reasonable range (< 0.2 for synthetic data)."""
        _, X_train, y_train, _, _, X_test, y_test = preprocessed_data
        result = train_linear_regression(X_train, y_train, X_test, y_test)
        assert result["rmse"] < 0.2, f"LR RMSE too high: {result['rmse']}"

    def test_mape_reasonable(self, preprocessed_data):
        """LR MAPE should be in a reasonable range."""
        _, X_train, y_train, _, _, X_test, y_test = preprocessed_data
        result = train_linear_regression(X_train, y_train, X_test, y_test)
        assert result["mape"] < 20.0, f"LR MAPE too high: {result['mape']}"


class TestXGBoost:
    """Tests for XGBoost training."""

    def test_produces_predictions(self, preprocessed_data):
        """XGBoost should produce predictions for all test samples."""
        _, X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_data
        result = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
        assert len(result["y_pred"]) == len(y_test)

    def test_better_than_linear(self, preprocessed_data):
        """XGBoost should beat Linear Regression on RMSE."""
        _, X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_data
        lr_result = train_linear_regression(X_train, y_train, X_test, y_test)
        xgb_result = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
        assert xgb_result["rmse"] < lr_result["rmse"], (
            f"XGBoost RMSE ({xgb_result['rmse']:.4f}) should be < "
            f"LR RMSE ({lr_result['rmse']:.4f})"
        )


class TestCentralizedNN:
    """Tests for the Centralized Neural Network."""

    def test_produces_predictions(self, preprocessed_data):
        """NN should produce predictions for all test samples."""
        _, X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_data
        result = train_neural_network(
            X_train, y_train, X_val, y_val, X_test, y_test,
            epochs=30, patience=10,
        )
        assert len(result["y_pred"]) == len(y_test)

    def test_rmse_reasonable(self, preprocessed_data):
        """NN RMSE should be in a reasonable range."""
        _, X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_data
        result = train_neural_network(
            X_train, y_train, X_val, y_val, X_test, y_test,
            epochs=50, patience=15,
        )
        assert result["rmse"] < 0.15, f"NN RMSE too high: {result['rmse']}"

    def test_predictions_finite(self, preprocessed_data):
        """All predictions must be finite (no NaN/Inf)."""
        _, X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_data
        result = train_neural_network(
            X_train, y_train, X_val, y_val, X_test, y_test,
            epochs=20, patience=10,
        )
        assert np.isfinite(result["y_pred"]).all()

    def test_model_returned(self, preprocessed_data):
        """Result should include the trained model."""
        _, X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_data
        result = train_neural_network(
            X_train, y_train, X_val, y_val, X_test, y_test,
            epochs=10, patience=5,
        )
        assert isinstance(result["model"], GreenBondNN)

