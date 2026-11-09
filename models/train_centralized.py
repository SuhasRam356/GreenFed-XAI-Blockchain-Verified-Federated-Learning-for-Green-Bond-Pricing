"""
Phase 2 — Centralized Baseline Training
=========================================

Trains three baseline models on the full training set and evaluates
on the 1,500-sample test set:

  1. Linear Regression
  2. XGBoost
  3. Centralized Neural Network (128→64→32, ReLU, Dropout 0.2)

Reports RMSE (%) and MAPE (%) to reproduce Table I of the paper.

Usage:
    python models/train_centralized.py
    python models/train_centralized.py --data-dir outputs/ --epochs 200
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.nn_model import GreenBondNN
from models.preprocessing import DataPreprocessor


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error (%)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%).

    Returns the error as a percentage (e.g., 5.0 means 5%).
    Handles near-zero values by clipping denominator.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    # Avoid division by zero — clip true values away from zero
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return float(mape)


# ---------------------------------------------------------------------------
# Model 1: Linear Regression
# ---------------------------------------------------------------------------

def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train and evaluate a Linear Regression model."""
    print("\n" + "=" * 60)
    print("  Model 1: Linear Regression")
    print("=" * 60)

    model = LinearRegression()
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.01, None)
    rmse = compute_rmse(y_test, y_pred)
    mape = compute_mape(y_test, y_pred)

    print(f"  Training time : {train_time:.2f}s")
    print(f"  RMSE (%)      : {rmse:.4f}")
    print(f"  MAPE (%)      : {mape:.2f}")

    return {
        "model_name": "Linear Regression",
        "model": model,
        "y_pred": y_pred,
        "rmse": rmse,
        "mape": mape,
        "train_time": train_time,
    }


# ---------------------------------------------------------------------------
# Model 2: XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train and evaluate an XGBoost model."""
    print("\n" + "=" * 60)
    print("  Model 2: XGBoost")
    print("=" * 60)

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=20,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.01, None)
    rmse = compute_rmse(y_test, y_pred)
    mape = compute_mape(y_test, y_pred)

    print(f"  Training time : {train_time:.2f}s")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  RMSE (%)      : {rmse:.4f}")
    print(f"  MAPE (%)      : {mape:.2f}")

    return {
        "model_name": "XGBoost",
        "model": model,
        "y_pred": y_pred,
        "rmse": rmse,
        "mape": mape,
        "train_time": train_time,
    }


# ---------------------------------------------------------------------------
# Model 3: Centralized Neural Network
# ---------------------------------------------------------------------------

def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 20,
) -> dict:
    """Train and evaluate the centralized neural network (128→64→32).

    Uses early stopping on validation loss.
    """
    print("\n" + "=" * 60)
    print("  Model 3: Centralized Neural Network (128→64→32)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    input_dim = X_train.shape[1]
    model = GreenBondNN(input_dim=input_dim, dropout_rate=0.2).to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    # Prepare data loaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_ds)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} — "
                f"train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}"
            )

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    train_time = time.time() - t0

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate on test set
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()
    y_pred = np.clip(y_pred, 0.01, None)

    rmse = compute_rmse(y_test, y_pred)
    mape = compute_mape(y_test, y_pred)

    print(f"\n  Best epoch    : {best_epoch}")
    print(f"  Training time : {train_time:.2f}s")
    print(f"  RMSE (%)      : {rmse:.4f}")
    print(f"  MAPE (%)      : {mape:.2f}")

    return {
        "model_name": "Centralized NN",
        "model": model,
        "y_pred": y_pred,
        "rmse": rmse,
        "mape": mape,
        "train_time": train_time,
        "best_epoch": best_epoch,
    }


# ---------------------------------------------------------------------------
# Results Table
# ---------------------------------------------------------------------------

def print_results_table(results: list[dict]):
    """Print a results table matching the paper's Table I format."""
    print("\n" + "=" * 60)
    print("  GREEN BOND YIELD SPREAD PREDICTION ERROR")
    print("  (Table I — Centralized Baselines)")
    print("=" * 60)
    print(f"  {'Model':<30s} {'RMSE (%)':>10s} {'MAPE (%)':>10s}")
    print("  " + "-" * 52)
    for r in results:
        print(f"  {r['model_name']:<30s} {r['rmse']:>10.4f} {r['mape']:>10.2f}")
    print("=" * 60)

    # Paper's reference values
    print("\n  Paper's reported values (for comparison):")
    print(f"  {'Linear Regression':<30s} {'0.68':>10s} {'7.9':>10s}")
    print(f"  {'XGBoost':<30s} {'0.42':>10s} {'5.0':>10s}")
    print(f"  {'Centralized NN':<30s} {'0.24':>10s} {'2.5':>10s}")
    print()


def save_results(results: list[dict], out_dir: Path):
    """Save results table as CSV."""
    rows = []
    for r in results:
        rows.append({
            "model": r["model_name"],
            "rmse_pct": round(r["rmse"], 4),
            "mape_pct": round(r["mape"], 2),
            "train_time_s": round(r["train_time"], 2),
        })
    df = pd.DataFrame(rows)
    path = out_dir / "baseline_results.csv"
    df.to_csv(path, index=False)
    print(f"  Saved results table → {path}")
    return df


def save_predictions(results: list[dict], y_test: np.ndarray, out_dir: Path):
    """Save test-set predictions from all models."""
    pred_df = pd.DataFrame({"y_true": y_test})
    for r in results:
        col_name = r["model_name"].lower().replace(" ", "_") + "_pred"
        pred_df[col_name] = r["y_pred"]
    path = out_dir / "baseline_predictions.csv"
    pred_df.to_csv(path, index=False)
    print(f"  Saved predictions    → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train centralized baselines")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = PROJECT_ROOT / "outputs"

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading datasets …")
    train_df = pd.read_csv(data_dir / "bonds_train.csv")
    val_df = pd.read_csv(data_dir / "bonds_val.csv")
    test_df = pd.read_csv(data_dir / "bonds_test.csv")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Preprocess
    print("Preprocessing …")
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)
    print(f"  Features: {X_train.shape[1]} ({preprocessor.get_feature_names()})")

    # Train all three baselines
    results = []

    # 1. Linear Regression
    lr_result = train_linear_regression(X_train, y_train, X_test, y_test)
    results.append(lr_result)

    # 2. XGBoost
    xgb_result = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(xgb_result)

    # 3. Centralized Neural Network
    nn_result = train_neural_network(
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )
    results.append(nn_result)

    # Print and save results
    print_results_table(results)
    save_results(results, out_dir)
    save_predictions(results, y_test, out_dir)

    # Save the NN model weights
    nn_model_path = out_dir / "centralized_nn.pt"
    torch.save(nn_result["model"].state_dict(), nn_model_path)
    print(f"  Saved NN weights     → {nn_model_path}")

    # Save the preprocessor for later phases
    import pickle
    prep_path = out_dir / "preprocessor.pkl"
    with open(prep_path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"  Saved preprocessor   → {prep_path}")


if __name__ == "__main__":
    main()

