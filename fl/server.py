"""
Federated Learning Server & Simulation Runner — GreenFed-XAI
============================================================

This module orchestrates the Federated Learning training pipeline across
the K = 8 simulated banking institutions as specified in Phase 3 of the
project implementation guide and Section IV-C of the IEEE paper.

Theoretical Workflow:
---------------------
1. Load proprietary bond partitions across K = 8 banks.
2. Initialize global neural network w_0 (128->64->32 architecture).
3. In each global round t = 1 ... T:
   a. Server broadcasts current global parameters w_t.
   b. Each bank k trains locally on its private dataset D_k for E epochs.
   c. Banks report updated parameters w_{t+1}^{(k)} and gradient norms ||dw_k||_2.
   d. Aggregator updates global model: w_{t+1} = \\sum_k (|D_k| / |D|) * w_{t+1}^{(k)}.
   e. Central evaluation is conducted on validation benchmark to record loss.
4. Final evaluation of the aggregated model on the 1,500 test set.
5. Generates loss curve visualizations and appends FedAvg results to Table I.

Usage:
------
    python fl/server.py --rounds 25 --clients 8 --local-epochs 3
    python fl/server.py --rounds 100  # full reproduction
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fl.client import GreenBondFlowerClient, compute_mape, compute_rmse
from fl.dataset_partitioner import BankDatasetPartitioner, ClientDataset
from fl.strategy_blockchain import BlockchainVerifiedStrategy
from fl.strategy_fedavg import GreenFedAvgStrategy
from models.nn_model import GreenBondNN
from models.preprocessing import DataPreprocessor

# Configure logger
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
        Test set features and targets (1,500 samples) for final verification.
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
    ) -> Dict[str, Any]:
        """Execute T federated learning communication rounds.

        Parameters:
        -----------
        num_rounds : int
            Number of communication rounds (default: 25).
        local_epochs : int
            Local epochs per bank per round (default: 3).
        batch_size : int
            Mini-batch size for client SGD (default: 64).
        learning_rate : float
            Learning rate for client optimizers (default: 0.001).

        Returns:
        --------
        Dict[str, Any]
            Dictionary summarizing convergence history and final metrics.
        """
        logger.info(
            "Starting Federated Learning across %d banks for %d global rounds...",
            self.n_clients,
            num_rounds,
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
                }
                updated_weights, num_samples, metrics = client.fit(
                    self.current_weights, fit_config
                )

                # Wrap into lightweight fit representation compatible with strategy
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
                    "Round %d: All updates rejected! Keeping previous round weights.",
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
                    res.metrics.get("train_loss", 0.0) * res.num_examples
                )
                client_norms[int(res.metrics.get("client_id", -1))] = float(
                    res.metrics.get("grad_norm", 0.0)
                )

            weighted_train_loss /= max(1, total_samples)
            self.current_weights = new_global_weights
            self.strategy.latest_global_ndarrays = self.current_weights

            # Step 4: Central validation evaluation
            val_loss, val_rmse, val_mape = 0.0, 0.0, 0.0
            if (
                self.strategy.eval_dataset is not None
                and self.strategy.eval_model is not None
            ):
                val_loss, val_rmse, val_mape = (
                    self.strategy._evaluate_global_model(self.current_weights)
                )

            # Record round metrics
            from fl.strategy_fedavg import RoundConvergenceMetrics

            avg_norm = float(np.mean(list(client_norms.values())))
            round_metric = RoundConvergenceMetrics(
                round_idx=round_idx,
                train_loss=float(weighted_train_loss),
                val_loss=float(val_loss),
                val_rmse=float(val_rmse),
                val_mape=float(val_mape),
                avg_update_norm=float(avg_norm),
                client_update_norms=client_norms,
            )
            self.strategy.history.append(round_metric)

            round_elapsed = time.time() - round_start
            if round_idx % 5 == 0 or round_idx == 1 or round_idx == num_rounds:
                logger.info(
                    "Round %3d/%3d (%.1fs) | Train Loss: %.6f | Val Loss: %.6f | RMSE: %.4f%% | MAPE: %.2f%% | ||dw||: %.4f",
                    round_idx,
                    num_rounds,
                    round_elapsed,
                    weighted_train_loss,
                    val_loss,
                    val_rmse,
                    val_mape,
                    avg_norm,
                )

        total_duration = time.time() - start_time
        logger.info(
            "Federated training complete across %d rounds in %.2f seconds.",
            num_rounds,
            total_duration,
        )

        # Final out-of-sample test evaluation
        test_loss, test_rmse, test_mape, y_pred = self.evaluate_test_set()
        logger.info(
            "Final Test Performance (1,500 samples): RMSE = %.4f%% | MAPE = %.2f%%",
            test_rmse,
            test_mape,
        )

        return {
            "num_rounds": num_rounds,
            "total_duration": total_duration,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
            "test_loss": test_loss,
            "y_pred": y_pred,
            "final_weights": self.current_weights,
            "history_df": self.strategy.get_history_dataframe(),
        }

    def evaluate_test_set(self) -> Tuple[float, float, float, np.ndarray]:
        """Evaluate current global weights on the 1,500-sample test set."""
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

        rmse = compute_rmse(y_test.flatten(), preds_np)
        mape = compute_mape(y_test.flatten(), preds_np)

        return float(loss), float(rmse), float(mape), preds_np


# ---------------------------------------------------------------------------
# Plotting & Reporting Helpers
# ---------------------------------------------------------------------------

def generate_loss_curve_plot(
    history_df: pd.DataFrame, output_path: Path
) -> Path:
    """Generate and save the Federated Learning loss curve plot."""
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: MSE Loss Curves
        ax1.plot(
            history_df["round"],
            history_df["train_loss"],
            label="Weighted Train Loss",
            color="#1f77b4",
            lw=2,
        )
        ax1.plot(
            history_df["round"],
            history_df["val_loss"],
            label="Central Val Loss",
            color="#ff7f0e",
            lw=2,
            linestyle="--",
        )
        ax1.set_title("FedAvg Convergence: MSE Loss vs. Round", fontsize=12)
        ax1.set_xlabel("Communication Round", fontsize=11)
        ax1.set_ylabel("Mean Squared Error", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=10)

        # Subplot 2: RMSE and MAPE Evaluation
        ax2.plot(
            history_df["round"],
            history_df["val_rmse"],
            label="Val RMSE (%)",
            color="#2ca02c",
            lw=2,
        )
        ax2.plot(
            history_df["round"],
            history_df["val_mape"],
            label="Val MAPE (%)",
            color="#d62728",
            lw=2,
            linestyle="-.",
        )
        ax2.set_title("FedAvg Prediction Error Across Rounds", fontsize=12)
        ax2.set_xlabel("Communication Round", fontsize=11)
        ax2.set_ylabel("Error Metric (%)", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        logger.info("Saved loss curve plot -> %s", output_path)
    except Exception as exc:
        logger.warning("Could not generate matplotlib plot: %s", exc)

    return output_path


def update_results_table(
    model_name: str,
    rmse: float,
    mape: float,
    duration: float,
    outputs_dir: Path,
) -> pd.DataFrame:
    """Append or update a model row in the results table alongside baselines."""
    results_path = outputs_dir / "baseline_results.csv"

    if results_path.exists():
        df_results = pd.read_csv(results_path)
        # Drop existing entry for this model if present to allow re-running
        df_results = df_results[
            df_results["model"] != model_name
        ].copy()
    else:
        df_results = pd.DataFrame(
            columns=["model", "rmse_pct", "mape_pct", "train_time_s"]
        )

    new_row = pd.DataFrame([
        {
            "model": model_name,
            "rmse_pct": round(rmse, 4),
            "mape_pct": round(mape, 2),
            "train_time_s": round(duration, 2),
        }
    ])

    df_updated = pd.concat([df_results, new_row], ignore_index=True)
    df_updated.to_csv(results_path, index=False)
    logger.info("Updated baseline results table -> %s", results_path)

    # Print formatted comparison table
    print("\n" + "=" * 65)
    print("  GREEN BOND YIELD SPREAD PREDICTION ERROR")
    print("  (Table I Reproduction — Centralized + Federated Baselines)")
    print("=" * 65)
    print(f"  {'Model':<30s} {'RMSE (%)':>12s} {'MAPE (%)':>12s}")
    print("  " + "-" * 58)
    for _, row in df_updated.iterrows():
        print(
            f"  {row['model']:<30s} {row['rmse_pct']:>12.4f} {row['mape_pct']:>12.2f}"
        )
    print("=" * 65)

    print("\n  Paper's reported benchmarks (for comparison):")
    print(f"  {'Linear Regression':<30s} {'0.68':>12s} {'7.9':>12s}")
    print(f"  {'XGBoost':<30s} {'0.42':>12s} {'5.0':>12s}")
    print(f"  {'FedAvg (no blockchain)':<30s} {'0.37':>12s} {'4.2':>12s}")
    print(f"  {'Centralized NN':<30s} {'0.24':>12s} {'2.5':>12s}")
    print(f"  {'GreenFed-XAI (Ours)':<30s} {'0.28':>12s} {'3.1':>12s}")
    print()

    return df_updated


# ---------------------------------------------------------------------------
# Main Orchestration Routine
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 3: Federated Learning WITHOUT Blockchain (FedAvg)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=25,
        help="Number of federated communication rounds (default: 25, paper: 100)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=8,
        help="Number of participating banking institutions (default: 8)",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=3,
        help="Local SGD epochs per client per round (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Local training mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for local optimizers (default: 0.001)",
    )
    parser.add_argument(
        "--non-iid",
        action="store_true",
        help="Use Dirichlet Non-IID partitioning instead of IID",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="weighted",
        choices=["weighted", "equal"],
        help="Aggregation weighting scheme: 'weighted' (sample-size proportional) or 'equal' (uniform client weighting)",
    )
    parser.add_argument(
        "--blockchain",
        action="store_true",
        help="Enable Phase 4 blockchain verification layer (GreenFed-XAI)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=2.5,
        help="Gradient norm threshold τ for poisoning defense (default: 2.5)",
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "outputs"
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data Splits
    logger.info("Loading training, validation, and test datasets...")
    train_df = pd.read_csv(data_dir / "bonds_train.csv")
    val_df = pd.read_csv(data_dir / "bonds_val.csv")
    test_df = pd.read_csv(data_dir / "bonds_test.csv")

    # 2. Load Preprocessor
    prep_path = data_dir / "preprocessor.pkl"
    if not prep_path.exists():
        logger.info("Fitting fresh DataPreprocessor...")
        preprocessor = DataPreprocessor()
        X_train, y_train = preprocessor.fit_transform(train_df)
    else:
        with open(prep_path, "rb") as f:
            preprocessor = pickle.load(f)
        X_train, y_train = preprocessor.transform(train_df)

    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)

    # 3. Partition Training Records across K = 8 Banks
    partitioner = BankDatasetPartitioner(n_clients=args.clients, val_ratio=0.15)
    if args.non_iid:
        client_datasets = partitioner.partition_non_iid_dirichlet(
            train_df, X_train, y_train, alpha=0.5
        )
    else:
        client_datasets = partitioner.partition_iid(train_df, X_train, y_train)

    partitioner.print_partition_summary(client_datasets)

    # 4. Instantiate Clients & Strategy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    global_model = GreenBondNN(input_dim=input_dim, dropout_rate=0.2).to(device)

    # Initial parameter vector
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
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
        )
        for cds in client_datasets
    ]

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
        logger.info("Initializing baseline GreenFedAvgStrategy (no blockchain, agg=%s)...", args.aggregation)
        strategy = GreenFedAvgStrategy(
            eval_dataset=(X_val, y_val),
            model_template=global_model,
            device=device,
            aggregation=args.aggregation,
            min_fit_clients=args.clients,
            min_available_clients=args.clients,
        )
        model_name = "FedAvg (no blockchain)"
        prefix = "fedavg"

    # 5. Run Federated Learning Simulation
    engine = FederatedSimulationEngine(
        clients=clients,
        strategy=strategy,
        initial_weights=initial_weights,
        eval_test_dataset=(X_test, y_test),
    )

    results = engine.run_rounds(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # 6. Save Artifacts & Deliverables
    # a. Convergence history CSV
    hist_path = outputs_dir / f"{prefix}_convergence.csv"
    strategy.save_history_csv(hist_path)

    # b. Loss curve plot
    plot_path = outputs_dir / f"{prefix}_loss_curve.png"
    generate_loss_curve_plot(results["history_df"], plot_path)

    # c. Final model weights
    model_path = outputs_dir / f"{prefix}_model.pt"
    keys = list(global_model.state_dict().keys())
    state_dict = {
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in zip(keys, results["final_weights"])
    }
    torch.save(state_dict, model_path)
    logger.info("Saved final model weights -> %s", model_path)

    # d. Predictions CSV
    preds_path = outputs_dir / f"{prefix}_predictions.csv"
    pd.DataFrame({
        "bond_id": test_df["bond_id"].values,
        "y_true": y_test,
        "y_pred": results["y_pred"],
        "absolute_error": np.abs(y_test - results["y_pred"]),
    }).to_csv(preds_path, index=False)
    logger.info("Saved test predictions -> %s", preds_path)

    # e. Blockchain audit export if in blockchain mode
    if args.blockchain and isinstance(strategy, BlockchainVerifiedStrategy):
        audit_csv_path = outputs_dir / "blockchain_audit_history.csv"
        strategy.save_audit_csv(audit_csv_path)
        ledger_export_path = outputs_dir / "blockchain_ledger_dump.json"
        with open(ledger_export_path, "w", encoding="utf-8") as f:
            f.write(strategy.blockchain.export_chain_json())
        logger.info("Exported full blockchain ledger -> %s", ledger_export_path)
        strategy.print_verification_summary()

    # f. Append to Table I results
    update_results_table(
        model_name,
        results["test_rmse"],
        results["test_mape"],
        results["total_duration"],
        outputs_dir,
    )


if __name__ == "__main__":
    main()

