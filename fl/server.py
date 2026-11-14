r"""
Federated Learning Simulation Orchestrator & Server — GreenFed-XAI
==================================================================

This module orchestrates decentralized training across the consortium of $K = 8$
participating financial institutions, evaluating performance on the held-out
1,500-sample test benchmark under rigorous scientific constraints.

Mathematical Workflow:
---------------------
1. Partition private bond portfolios across $K = 8$ banks under Stratified IID,
   Non-IID Dirichlet ($\alpha = 0.5$), or Non-IID Rating-Tier specialization.
2. Initialize global neural network $w_0 \sim \mathcal{N}(0, \sigma^2)$ ($128 \to 64 \to 32$ architecture).
3. In each global communication round $t = 1, \dots, T$:
   a. Aggregator broadcasts global parameter vector $w_t$.
   b. Each bank $k$ runs local optimization on its private dataset $\mathcal{D}_k$ for $E$ epochs:
      $$w_{t+1}^{(k)} \leftarrow \text{LocalOptimization}(w_t, \mathcal{D}_k)$$
   c. Banks report updated weights $w_{t+1}^{(k)}$ and update norms $\|\Delta w_k\|_2 = \|w_{t+1}^{(k)} - w_t\|_2$.
   d. Aggregator filters updates (via baseline verification hook or Phase 5 blockchain smart contract).
   e. Aggregator averages accepted weights:
      - Sample-Weighted: $w_{t+1} = \sum_{k \in S} \frac{|\mathcal{D}_k|}{\sum_{j \in S} |\mathcal{D}_j|} w_{t+1}^{(k)}$
      - Equal-Client: $w_{t+1} = \frac{1}{|S|} \sum_{k \in S} w_{t+1}^{(k)}$
   f. Evaluates $w_{t+1}$ on central validation benchmark (MSE, RMSE, MAE bps, MAPE).
4. Conducts final out-of-sample evaluation on the 1,500-bond test set.
5. Exports loss curves, convergence telemetry CSVs, and Table I benchmark records.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fl.client import (
    GreenBondFlowerClient,
    compute_mae,
    compute_mae_bps,
    compute_mape,
    compute_rmse,
)
from fl.dataset_partitioner import BankDatasetPartitioner, ClientDataset
from fl.strategy_blockchain import BlockchainVerifiedStrategy
from fl.strategy_fedavg import GreenFedAvgStrategy, RoundConvergenceMetrics
from models.nn_model import GreenBondNN
from models.preprocessing import DataPreprocessor

# Configure server logger
logger = logging.getLogger("GreenFedServer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [Server]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FederatedSimulationEngine:
    """Production-grade Federated Learning simulation engine.

    Executes multi-client decentralized learning rounds using exact Flower
    NumPyClient interfaces and FedAvg aggregation mechanics.

    Parameters:
    -----------
    clients : List[GreenBondFlowerClient]
        List of initialized Flower client instances (one per bank).
    strategy : GreenFedAvgStrategy
        Aggregation strategy managing client selection and parameter averaging.
    initial_weights : List[np.ndarray]
        List of NumPy parameter arrays for the global model at round 0.
    eval_test_dataset : Tuple[np.ndarray, np.ndarray]
        Test set features and targets (1,500 samples) strictly reserved for final verification.
    """

    def __init__(
        self,
        clients: List[GreenBondFlowerClient],
        strategy: GreenFedAvgStrategy,
        initial_weights: List[np.ndarray],
        eval_test_dataset: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.clients = clients
        self.strategy = strategy
        self.current_weights = [w.copy() for w in initial_weights]
        self.eval_test_dataset = eval_test_dataset
        self.n_clients = len(clients)

    def run_rounds(
        self,
        num_rounds: int = 25,
        local_epochs: int = 3,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        optimizer_type: str = "adam",
    ) -> Dict[str, Any]:
        """Execute T federated learning communication rounds.

        Parameters:
        -----------
        num_rounds : int
            Number of communication rounds (default: 25).
        local_epochs : int
            Local epochs per bank per round (default: 3).
        batch_size : int
            Mini-batch size for client SGD/Adam (default: 64).
        learning_rate : float
            Learning rate for client optimizers (default: 0.001).
        optimizer_type : str
            Client optimizer ('adam' or 'sgd').

        Returns:
        --------
        Dict[str, Any]
            Dictionary summarizing convergence history and final metrics.
        """
        logger.info(
            "Starting Federated Learning across %d banks for %d global rounds (aggregation: %s)...",
            self.n_clients,
            num_rounds,
            self.strategy.aggregation,
        )
        start_time = time.time()

        for round_idx in range(1, num_rounds + 1):
            round_start = time.time()
            client_fit_results = []

            # Step 1 & 2: Broadcast w_t and execute local optimization at all banks
            for client in self.clients:
                fit_config = {
                    "current_round": round_idx,
                    "local_epochs": local_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "optimizer_type": optimizer_type,
                    "tau": getattr(self.strategy, "tau", 2.5),
                }
                updated_weights, num_samples, metrics = client.fit(
                    self.current_weights, fit_config
                )

                # Wrap into fit representation compatible with Flower strategy
                fit_res_mock = type(
                    "FitResMock",
                    (),
                    {
                        "parameters": type(
                            "ParamMock",
                            (),
                            {"tensors": updated_weights, "ndarrays": updated_weights},
                        ),
                        "num_examples": num_samples,
                        "metrics": metrics,
                    },
                )()
                client_proxy_mock = type(
                    "ClientProxyMock",
                    (),
                    {"cid": str(client.client_id)},
                )()
                client_fit_results.append((client_proxy_mock, fit_res_mock))

            # Step 3: Aggregator filters and averages accepted weights
            accepted_results, audit_meta = self.strategy.verify_and_filter_updates(
                round_idx, client_fit_results
            )

            if not accepted_results:
                logger.error(
                    "Round %d: All updates rejected! Preserving previous round weights.",
                    round_idx,
                )
                continue

            total_samples = sum(res.num_examples for _, res in accepted_results)
            new_global_weights = [np.zeros_like(w) for w in self.current_weights]
            num_accepted = len(accepted_results)
            is_equal_agg = getattr(self.strategy, "aggregation", "weighted") == "equal"

            weighted_train_loss = 0.0
            client_norms: Dict[int, float] = {}

            for _, res in accepted_results:
                fraction = (1.0 / num_accepted) if is_equal_agg else (res.num_examples / total_samples)
                for i, layer in enumerate(res.parameters.ndarrays):
                    new_global_weights[i] += layer * fraction
                weighted_train_loss += (
                    float(res.metrics.get("train_loss", 0.0)) * fraction
                )
                cid = int(res.metrics.get("client_id", -1))
                norm_val = float(
                    res.metrics.get(
                        "update_norm", res.metrics.get("grad_norm", 0.0)
                    )
                )
                client_norms[cid] = norm_val

            self.current_weights = new_global_weights

            # Step 4: Central validation benchmark evaluation
            val_loss, val_rmse, val_mae, val_mae_bps, val_mape = (
                self.strategy.evaluate_global_model(self.current_weights)
            )
            round_duration = time.time() - round_start
            avg_norm = float(np.mean(list(client_norms.values()))) if client_norms else 0.0

            logger.info(
                "Round %3d/%3d (%.1fs) | Train Loss: %.6f | Val Loss: %.6f | "
                "RMSE: %.4f%% | MAE: %.2f bps | MAPE: %.2f%% | ||Δw||: %.4f",
                round_idx,
                num_rounds,
                round_duration,
                weighted_train_loss,
                val_loss,
                val_rmse,
                val_mae_bps,
                val_mape,
                avg_norm,
            )

            # Record round metric
            round_metric = RoundConvergenceMetrics(
                server_round=round_idx,
                train_loss=float(weighted_train_loss),
                val_loss=float(val_loss),
                val_rmse=float(val_rmse),
                val_mae_pct=float(val_mae),
                val_mae_bps=float(val_mae_bps),
                val_mape=float(val_mape),
                avg_update_norm=float(avg_norm),
                client_update_norms=client_norms,
                aggregation_mode=self.strategy.aggregation,
                num_accepted=num_accepted,
                duration_sec=round(round_duration, 3),
            )
            self.strategy.history.append(round_metric)

        total_duration = time.time() - start_time
        logger.info(
            "Federated training complete across %d rounds in %.2f seconds.",
            num_rounds,
            total_duration,
        )

        # Final out-of-sample test evaluation
        test_loss, test_rmse, test_mae, test_mae_bps, test_mape, y_pred = self.evaluate_test_set()
        logger.info(
            "Final Test Performance (1,500 samples): RMSE = %.4f%% | MAE = %.2f bps | MAPE = %.2f%%",
            test_rmse,
            test_mae_bps,
            test_mape,
        )

        return {
            "num_rounds": num_rounds,
            "total_duration": total_duration,
            "test_rmse": test_rmse,
            "test_mae_pct": test_mae,
            "test_mae_bps": test_mae_bps,
            "test_mape": test_mape,
            "test_loss": test_loss,
            "y_pred": y_pred,
            "final_weights": self.current_weights,
            "history_df": self.strategy.get_history_dataframe(),
        }

    def evaluate_test_set(self) -> Tuple[float, float, float, float, float, np.ndarray]:
        """Evaluate current global weights on the 1,500-sample test set.

        Returns:
        --------
        Tuple[float, float, float, float, float, np.ndarray]
            (test_loss, test_rmse, test_mae, test_mae_bps, test_mape, y_pred)
        """
        X_test, y_test = self.eval_test_dataset
        input_dim = X_test.shape[1]
        model = GreenBondNN(input_dim=input_dim, dropout_rate=0.2)

        keys = list(model.state_dict().keys())
        state_dict = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in zip(keys, self.current_weights)
        }
        model.load_state_dict(state_dict)
        model.eval()

        X_t = torch.tensor(X_test, dtype=torch.float32)
        y_t = torch.tensor(y_test, dtype=torch.float32).view(-1)

        with torch.no_grad():
            preds = model(X_t)
            loss = torch.nn.MSELoss()(preds, y_t).item()
            preds_np = preds.cpu().numpy().flatten()
            preds_np = np.clip(preds_np, 0.01, None)

        y_test_flat = y_test.flatten()
        rmse = compute_rmse(y_test_flat, preds_np)
        mae = compute_mae(y_test_flat, preds_np)
        mae_bps = compute_mae_bps(y_test_flat, preds_np)
        mape = compute_mape(y_test_flat, preds_np)

        return float(loss), float(rmse), float(mae), float(mae_bps), float(mape), preds_np


# ---------------------------------------------------------------------------
# Plotting & Reporting Helpers
# ---------------------------------------------------------------------------

def generate_loss_curve_plot(
    history_df: pd.DataFrame, output_path: Path, title_prefix: str = "FedAvg"
) -> Path:
    """Generate and save publication-quality loss and update-norm curve plots."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rounds = history_df["round"].to_numpy()
    train_loss = history_df["train_loss"].to_numpy()
    val_loss = history_df["val_loss"].to_numpy()
    val_rmse = history_df["val_rmse"].to_numpy()
    val_mae_bps = history_df.get("val_mae_bps", pd.Series(np.zeros_like(rounds))).to_numpy()
    update_norms = history_df.get("avg_update_norm", history_df.get("avg_grad_norm")).to_numpy()

    # Left Plot: Loss Convergence
    ax1.plot(rounds, train_loss, label="Training MSE Loss", color="#1f77b4", linewidth=2.0)
    ax1.plot(rounds, val_loss, label="Validation MSE Loss", color="#ff7f0e", linestyle="--", linewidth=2.0)
    ax1.set_xlabel("Global Communication Round (t)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=11, fontweight="bold")
    ax1.set_title(f"{title_prefix}: Loss Convergence Across Rounds", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Right Plot: Update Norm Trajectory & Spread MAE
    color_norm = "#2ca02c"
    ax2.plot(rounds, update_norms, label="Mean Update Norm ||Δw||_2", color=color_norm, linewidth=2.0)
    ax2.axhline(y=2.5, color="purple", linestyle=":", label="Paper Threshold τ = 2.5", alpha=0.7)
    ax2.set_xlabel("Global Communication Round (t)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Update Vector L2 Norm ||Δw||_2", color=color_norm, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_norm)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Secondary y-axis for MAE (bps)
    ax2_twin = ax2.twinx()
    color_mae = "#d62728"
    ax2_twin.plot(rounds, val_mae_bps, label="Validation MAE (bps)", color=color_mae, linestyle="-.", linewidth=2.0)
    ax2_twin.set_ylabel("Spread Error (Basis Points)", color=color_mae, fontsize=11, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor=color_mae)

    # Combine legends
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=True)
    ax2.set_title(f"{title_prefix}: Update Norm & Pricing Error Trajectory", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved loss curve plot to %s", output_path)
    return output_path


def update_results_table(
    results_path: Path,
    model_name: str,
    rmse: float,
    mae_bps: float,
    mape: float,
    train_time_sec: float,
    notes: str = "",
) -> pd.DataFrame:
    """Update or create the benchmark Table I summary CSV.
    
    Harmonizes with models/train_centralized.py schema ('model', 'rmse_pct',
    'mae_pct', 'mae_bps', 'mape_pct', 'train_time_s').
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    mae_pct = round(mae_bps / 100.0, 4)

    if results_path.exists():
        df = pd.read_csv(results_path)
        if "model" in df.columns:
            new_row = {
                "model": model_name,
                "rmse_pct": round(rmse, 4),
                "mae_pct": mae_pct,
                "mae_bps": round(mae_bps, 2),
                "mape_pct": round(mape, 2),
                "train_time_s": round(train_time_sec, 2),
            }
            if "notes" in df.columns or notes:
                new_row["notes"] = notes
            match = df["model"] == model_name
            if match.any():
                for k, v in new_row.items():
                    df.loc[match, k] = v
            else:
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            model_col = "Model" if "Model" in df.columns else df.columns[0]
            new_row = {
                "Model": model_name,
                "RMSE (%)": round(rmse, 4),
                "MAE (bps)": round(mae_bps, 2),
                "MAPE (%)": round(mape, 2),
                "Training Time (s)": round(train_time_sec, 2),
                "Notes": notes,
            }
            match = df[model_col] == model_name
            if match.any():
                for k, v in new_row.items():
                    if k in df.columns:
                        df.loc[match, k] = v
            else:
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        new_row = {
            "Model": model_name,
            "RMSE (%)": round(rmse, 4),
            "MAE (bps)": round(mae_bps, 2),
            "MAPE (%)": round(mape, 2),
            "Training Time (s)": round(train_time_sec, 2),
            "Notes": notes,
        }
        df = pd.DataFrame([new_row])

    df.to_csv(results_path, index=False)
    logger.info("Updated benchmark results table at %s", results_path)
    return df


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI orchestrator for Federated Learning."""
    parser = argparse.ArgumentParser(
        description="Run GreenFed-XAI Federated Learning Simulation across banks."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=25,
        help="Number of global communication rounds (default: 25)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=8,
        help="Number of participating banks (default: 8)",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=3,
        help="Number of local epochs per bank per round (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Local mini-batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Local learning rate for Adam optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "dirichlet", "rating_tier"],
        help="Partitioning scheme: 'iid', 'dirichlet', or 'rating_tier'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration parameter for non-IID partitioning (default: 0.5)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="weighted",
        choices=["weighted", "equal"],
        help="Aggregation weighting: 'weighted' (sample-size proportional) or 'equal' (uniform client weighting)",
    )
    parser.add_argument(
        "--blockchain",
        action="store_true",
        help="Enable Phase 5 blockchain verification layer (GreenFed-XAI)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=2.5,
        help="Update norm threshold τ for poisoning defense (default: 2.5)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke-test mode with 3 rounds and 1 local epoch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    eff_rounds = 3 if args.quick else args.rounds
    eff_epochs = 1 if args.quick else args.local_epochs

    logger.info(
        "Initializing FL Orchestrator | Rounds: %d | Banks: %d | Partition: %s | Aggregation: %s | Seed: %d",
        eff_rounds,
        args.clients,
        args.partition,
        args.aggregation,
        args.seed,
    )

    # 1. Load Data Splits
    logger.info("Loading training, validation, and test datasets...")
    train_df = pd.read_csv(outputs_dir / "bonds_train.csv")
    val_df = pd.read_csv(outputs_dir / "bonds_val.csv")
    test_df = pd.read_csv(outputs_dir / "bonds_test.csv")

    # 2. Preprocessing
    prep_path = outputs_dir / "preprocessor.pkl"
    if not prep_path.exists():
        logger.info("Fitting fresh DataPreprocessor...")
        preprocessor = DataPreprocessor()
        X_train, y_train = preprocessor.fit_transform(train_df)
        with open(prep_path, "wb") as f:
            pickle.dump(preprocessor, f)
    else:
        with open(prep_path, "rb") as f:
            preprocessor = pickle.load(f)
        X_train, y_train = preprocessor.transform(train_df)

    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)

    # 3. Partition Datasets across Banks
    partitioner = BankDatasetPartitioner(
        n_clients=args.clients, val_ratio=0.15, random_seed=args.seed
    )

    if args.partition == "dirichlet":
        client_datasets = partitioner.partition_non_iid_dirichlet(
            train_df, X_train, y_train, alpha=args.alpha
        )
    elif args.partition == "rating_tier":
        client_datasets = partitioner.partition_non_iid_rating_tier(
            train_df, X_train, y_train
        )
    else:
        client_datasets = partitioner.partition_iid(train_df, X_train, y_train)

    partitioner.print_partition_summary(client_datasets)
    partitioner.export_partition_diagnostics_csv(
        client_datasets, outputs_dir / "fl_bank_partitions.csv"
    )

    # 4. Instantiate Global Blueprint & Clients
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    global_model = GreenBondNN(input_dim=input_dim, dropout_rate=0.2).to(device)

    initial_weights = [
        val.cpu().detach().numpy().copy()
        for _, val in global_model.state_dict().items()
    ]

    clients = [
        GreenBondFlowerClient(
            client_id=cds.client_id,
            dataset=cds,
            device=device,
            learning_rate=args.lr,
            local_epochs=eff_epochs,
            batch_size=args.batch_size,
        )
        for cds in client_datasets
    ]

    # 5. Instantiate Aggregation Strategy
    if args.blockchain:
        logger.info(
            "Initializing BlockchainVerifiedStrategy (τ = %.2f, mock Fabric ledger, agg=%s)...",
            args.tau,
            args.aggregation,
        )
        strategy = BlockchainVerifiedStrategy(
            eval_dataset=(X_val, y_val),
            model_template=global_model,
            device=device,
            tau=args.tau,
            aggregation=args.aggregation,
            min_fit_clients=args.clients,
            min_available_clients=args.clients,
        )
        model_name = "GreenFed-XAI (blockchain-verified)"
        prefix = "greenfed_blockchain"
    else:
        logger.info(
            "Initializing baseline GreenFedAvgStrategy (no blockchain, agg=%s)...",
            args.aggregation,
        )
        strategy = GreenFedAvgStrategy(
            eval_dataset=(X_val, y_val),
            model_template=global_model,
            device=device,
            aggregation=args.aggregation,
            min_fit_clients=args.clients,
            min_available_clients=args.clients,
        )
        model_name = f"FedAvg ({args.aggregation})"
        prefix = f"fedavg_{args.aggregation}"

    # 6. Execute Federated Simulation
    engine = FederatedSimulationEngine(
        clients=clients,
        strategy=strategy,
        initial_weights=initial_weights,
        eval_test_dataset=(X_test, y_test),
    )

    results = engine.run_rounds(
        num_rounds=eff_rounds,
        local_epochs=eff_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # 7. Export Outputs & Artifacts
    history_df = results["history_df"]
    history_csv = outputs_dir / f"{prefix}_convergence.csv"
    history_df.to_csv(history_csv, index=False)
    logger.info("Saved convergence telemetry to %s", history_csv)

    plot_path = outputs_dir / f"{prefix}_loss_curve.png"
    generate_loss_curve_plot(history_df, plot_path, title_prefix=model_name)

    # Export test set predictions
    preds_df = pd.DataFrame({
        "bond_id": test_df["bond_id"],
        "actual_yield_spread": y_test.flatten(),
        "predicted_yield_spread": results["y_pred"],
        "absolute_error_bps": np.abs(y_test.flatten() - results["y_pred"]) * 100.0,
    })
    preds_csv = outputs_dir / f"{prefix}_predictions.csv"
    preds_df.to_csv(preds_csv, index=False)
    logger.info("Saved test set predictions to %s", preds_csv)

    # Update Table I
    update_results_table(
        results_path=outputs_dir / "baseline_results.csv",
        model_name=model_name,
        rmse=results["test_rmse"],
        mae_bps=results["test_mae_bps"],
        mape=results["test_mape"],
        train_time_sec=results["total_duration"],
        notes=f"T={eff_rounds}, K={args.clients}, E={eff_epochs}, {args.partition}, {args.aggregation}",
    )

    # Save final model checkpoint
    model_checkpoint_path = outputs_dir / f"{prefix}_model.pt"
    keys = list(global_model.state_dict().keys())
    final_state_dict = {
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in zip(keys, results["final_weights"])
    }
    torch.save(final_state_dict, model_checkpoint_path)
    logger.info("Saved final global model checkpoint to %s", model_checkpoint_path)


if __name__ == "__main__":
    main()
