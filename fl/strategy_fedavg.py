r"""
Federated Averaging Strategy (FedAvg) Suite — GreenFed-XAI
=========================================================

This module implements the production-grade Federated Averaging (FedAvg) strategy
customized for green bond yield spread prediction across heterogeneous financial institutions.

Theoretical Foundations & Aggregation Schemes:
----------------------------------------------
In each global communication round $t \in \{1, \dots, T\}$:
1. Coordinator broadcasts global model parameters $w_t$ to selected banks $S_t \subseteq \mathcal{K}$.
2. Each bank $k \in S_t$ optimizes locally on its private dataset $\mathcal{D}_k$, producing $w_{t+1}^{(k)}$.
3. The server aggregates client updates into the new global model $w_{t+1}$.

We support and explicitly analyze two aggregation paradigms:

1. Standard Sample-Weighted FedAvg (McMahan et al., 2017):
   $$w_{t+1} = \sum_{k \in S_t} \frac{|\mathcal{D}_k|}{\sum_{j \in S_t} |\mathcal{D}_j|} w_{t+1}^{(k)}$$
   - **Optimization Objective**: Minimizes overall empirical risk across all bond samples:
     $$\min_w F(w) = \sum_{k=1}^K \frac{|\mathcal{D}_k|}{|\mathcal{D}|} F_k(w) = \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \ell_i(w)$$
   - **Weighting**: Proportionate to portfolio size. Larger banks have greater influence.

2. Equal-Client Averaging (Uniform Democratic FedAvg):
   $$w_{t+1} = \frac{1}{|S_t|} \sum_{k \in S_t} w_{t+1}^{(k)}$$
   - **Optimization Objective**: Minimizes unweighted institutional risk:
     $$\min_w \bar{F}(w) = \frac{1}{K} \sum_{k=1}^K F_k(w)$$
   - **Weighting**: Identical influence ($1 / |S_t|$) across all institutions regardless of portfolio size.
     Matches the algorithm pseudocode displayed in Section IV of the IEEE paper.

Mathematical Equivalence vs. Divergence:
----------------------------------------
- **IID Equivalence**: When client datasets are of equal size ($|\mathcal{D}_k| = n \quad \forall k$),
  $$\frac{|\mathcal{D}_k|}{\sum_{j=1}^K |\mathcal{D}_j|} = \frac{n}{K \cdot n} = \frac{1}{K}$$
  Sample-weighted and equal-client FedAvg are mathematically identical.
- **Non-IID Divergence**: When portfolios vary in size (e.g. Bank 0 holds 1,200 bonds, Bank 7 holds 400 bonds),
  sample-weighted FedAvg assigns $3\times$ higher weight to Bank 0, whereas equal-client FedAvg weights
  each institution at exactly $12.5\%$.

Renormalization & Zero-Client Fallback:
---------------------------------------
- When client updates are filtered (e.g., by the Phase 5 blockchain norm gate), weights are dynamically
  renormalized strictly over accepted clients:
  $$\alpha_k = \frac{|\mathcal{D}_k|}{\sum_{j \in S_{\text{accepted}}} |\mathcal{D}_j|}, \quad \sum_{k \in S_{\text{accepted}}} \alpha_k = 1.0$$
- If no updates are accepted ($|S_{\text{accepted}}| = 0$), the strategy executes a robust fallback,
  retaining the prior global model $w_{t+1} \leftarrow w_t$ without numerical errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import flwr as fl
    from flwr.common import (
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        NDArrays,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg
except ImportError:
    # Fallback stubs for mock unit testing environments
    class FedAvg:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    ClientProxy = Any  # type: ignore[assignment, misc]
    FitRes = Any  # type: ignore[assignment, misc]
    Parameters = Any  # type: ignore[assignment, misc]
    NDArrays = List[np.ndarray]  # type: ignore[assignment, misc]

from fl.client import compute_mae, compute_mae_bps, compute_mape, compute_rmse
from models.nn_model import GreenBondNN

logger = logging.getLogger("FedAvgStrategy")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [Strategy]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class RoundConvergenceMetrics:
    r"""Comprehensive convergence telemetry for a single FL communication round.

    Attributes:
    -----------
    server_round : int
        Global communication round index.
    train_loss : float
        Weighted/equal mean training MSE loss across participating clients.
    val_loss : float
        Centralized validation MSE loss.
    val_rmse : float
        Centralized validation Root Mean Square Error (%).
    val_mae_pct : float
        Centralized validation Mean Absolute Error (%).
    val_mae_bps : float
        Centralized validation MAE expressed in basis points ($1\text{ bp} = 0.01\%$).
    val_mape : float
        Centralized validation MAPE (%) with protected denominator.
    avg_update_norm : float
        Mean Euclidean L2 norm of parameter update vectors across clients.
    min_update_norm : float
        Minimum client update norm in this round.
    max_update_norm : float
        Maximum client update norm in this round.
    std_update_norm : float
        Standard deviation of client update norms.
    client_update_norms : Dict[int, float]
        Per-client update norm mapping.
    client_sample_counts : Dict[int, int]
        Per-client training sample size mapping.
    client_aggregation_weights : Dict[int, float]
        Normalized aggregation weight assigned to each client.
    aggregation_mode : str
        Aggregation scheme used ('weighted' or 'equal').
    num_accepted : int
        Number of client updates accepted into aggregation.
    duration_sec : float
        Round execution wall-clock time in seconds.
    """

    server_round: int
    train_loss: float
    val_loss: float
    val_rmse: float
    val_mae_pct: float
    val_mae_bps: float
    val_mape: float
    avg_update_norm: float
    min_update_norm: float = 0.0
    max_update_norm: float = 0.0
    std_update_norm: float = 0.0
    client_update_norms: Dict[int, float] = field(default_factory=dict)
    client_sample_counts: Dict[int, int] = field(default_factory=dict)
    client_aggregation_weights: Dict[int, float] = field(default_factory=dict)
    aggregation_mode: str = "weighted"
    num_accepted: int = 8
    duration_sec: float = 0.0

    # Backward compatibility aliases
    @property
    def avg_grad_norm(self) -> float:
        """Alias for avg_update_norm."""
        return self.avg_update_norm

    @property
    def client_grad_norms(self) -> Dict[int, float]:
        """Alias for client_update_norms."""
        return self.client_update_norms


class GreenFedAvgStrategy(FedAvg):
    """Production-grade Federated Averaging strategy for Green Bond Pricing.

    Parameters:
    -----------
    eval_dataset : Optional[Tuple[np.ndarray, np.ndarray]]
        Isolated validation dataset (X_val, y_val) evaluated centrally after each round.
    model_template : Optional[GreenBondNN]
        PyTorch blueprint used for central evaluation.
    device : Optional[torch.device]
        Computation device (CPU/CUDA).
    aggregation : str
        Weighting scheme: 'weighted' (sample-size proportional) or 'equal' (uniform client weighting).
    fraction_fit : float
        Fraction of available clients sampled per round (default: 1.0).
    min_fit_clients : int
        Minimum clients required for training (default: 8).
    min_available_clients : int
        Minimum total connected clients (default: 8).
    """

    def __init__(
        self,
        eval_dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        model_template: Optional[GreenBondNN] = None,
        device: Optional[torch.device] = None,
        aggregation: str = "weighted",
        fraction_fit: float = 1.0,
        min_fit_clients: int = 8,
        min_available_clients: int = 8,
        initial_parameters: Optional[Parameters] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            **kwargs,
        )

        self.eval_dataset = eval_dataset
        self.aggregation = aggregation.lower()
        if self.aggregation not in {"weighted", "equal"}:
            raise ValueError(
                f"Unknown aggregation scheme: '{aggregation}'. Must be 'weighted' or 'equal'."
            )

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if eval_dataset is not None:
            input_dim = eval_dataset[0].shape[1]
            self.eval_model = (
                model_template.to(self.device)
                if model_template is not None
                else GreenBondNN(input_dim=input_dim, dropout_rate=0.2).to(self.device)
            )
        else:
            self.eval_model = None

        self.criterion = nn.MSELoss()

        # Telemetry history tracking
        self.history: List[RoundConvergenceMetrics] = []
        self.latest_global_ndarrays: Optional[NDArrays] = None

        logger.info(
            "GreenFedAvgStrategy initialized | Aggregation: '%s' | Min Clients: %d",
            self.aggregation,
            min_fit_clients,
        )

    # -----------------------------------------------------------------------
    # Verification Hook (Extensible by Blockchain Strategy in Phase 5)
    # -----------------------------------------------------------------------

    def verify_and_filter_updates(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[List[Tuple[ClientProxy, FitRes]], Dict[str, Any]]:
        """Hook to filter client updates before parameter aggregation.

        In Phase 4 (plain FedAvg), all submitted updates from authorized banks
        are accepted. In Phase 5, this method is overridden by BlockchainVerifiedStrategy
        to enforce SHA-256 integrity and ||Δw||_2 <= tau.

        Parameters:
        -----------
        server_round : int
            Current communication round index.
        results : List[Tuple[ClientProxy, FitRes]]
            Raw client fit submissions.

        Returns:
        --------
        Tuple[List[Tuple[ClientProxy, FitRes]], Dict[str, Any]]
            - Filtered list of accepted client results.
            - Audit metadata dictionary.
        """
        accepted = results
        audit_metadata: Dict[str, Any] = {
            "total_submitted": len(results),
            "total_accepted": len(results),
            "total_rejected": 0,
            "verification_mode": "plain_fedavg",
            "aggregation": self.aggregation,
        }
        return accepted, audit_metadata

    # -----------------------------------------------------------------------
    # Server-Side Parameter Aggregation (aggregate_fit)
    # -----------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate local model parameters using sample-weighted or equal averaging.

        Parameters:
        -----------
        server_round : int
            Current FL communication round.
        results : List[Tuple[ClientProxy, FitRes]]
            List of successful client training results.
        failures : List
            List of failed client connections.

        Returns:
        --------
        Tuple[Optional[Parameters], Dict[str, Scalar]]:
            - Aggregated global parameters.
            - Aggregation summary metrics dictionary.
        """
        round_start_time = time.time()

        if not results:
            logger.warning("Round %d: No client updates received!", server_round)
            return None, {}

        # 1. Filter updates through verification hook
        accepted_results, audit_meta = self.verify_and_filter_updates(
            server_round, results
        )

        if not accepted_results:
            logger.error(
                "Round %d: All updates rejected! Preserving previous model weights.",
                server_round,
            )
            # Fallback to previous global parameters
            if self.latest_global_ndarrays is not None:
                return ndarrays_to_parameters(self.latest_global_ndarrays), {
                    "round": server_round,
                    "accepted_count": 0,
                    "status": "fallback_previous_weights",
                }
            return None, {"error": "no_accepted_updates"}

        # 2. Extract weights, sample counts, and update norms
        total_samples = sum(fit_res.num_examples for _, fit_res in accepted_results)
        num_accepted = len(accepted_results)

        weighted_train_loss = 0.0
        update_norms: Dict[int, float] = {}
        sample_counts: Dict[int, int] = {}
        client_weights: Dict[int, float] = {}

        # 3. Compute normalized client aggregation weights
        for _, fit_res in accepted_results:
            cid = int(fit_res.metrics.get("client_id", -1))
            n_k = fit_res.num_examples
            sample_counts[cid] = n_k

            # Norm extraction (supporting both update_norm and grad_norm)
            norm_val = float(
                fit_res.metrics.get(
                    "update_norm", fit_res.metrics.get("grad_norm", 0.0)
                )
            )
            update_norms[cid] = norm_val

            train_loss = float(fit_res.metrics.get("train_loss", 0.0))

            if self.aggregation == "equal":
                weight_factor = 1.0 / num_accepted
            else:
                weight_factor = n_k / total_samples

            client_weights[cid] = weight_factor
            weighted_train_loss += train_loss * weight_factor

        # 4. Perform parameter averaging
        first_weights = parameters_to_ndarrays(accepted_results[0][1].parameters)
        aggregated_ndarrays = [np.zeros_like(layer) for layer in first_weights]

        for _, fit_res in accepted_results:
            cid = int(fit_res.metrics.get("client_id", -1))
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weight_factor = client_weights[cid]

            for i, layer in enumerate(client_ndarrays):
                aggregated_ndarrays[i] += layer * weight_factor

        self.latest_global_ndarrays = aggregated_ndarrays

        # 5. Evaluate on centralized benchmark
        val_loss, val_rmse, val_mae, val_mae_bps, val_mape = (
            self.evaluate_global_model(aggregated_ndarrays)
        )

        # 6. Compute update norm distribution moments
        norm_values = list(update_norms.values())
        avg_update_norm = float(np.mean(norm_values)) if norm_values else 0.0
        min_update_norm = float(np.min(norm_values)) if norm_values else 0.0
        max_update_norm = float(np.max(norm_values)) if norm_values else 0.0
        std_update_norm = float(np.std(norm_values)) if norm_values else 0.0

        duration_sec = time.time() - round_start_time

        # 7. Record rich convergence telemetry
        metrics_record = RoundConvergenceMetrics(
            server_round=server_round,
            train_loss=float(weighted_train_loss),
            val_loss=float(val_loss),
            val_rmse=float(val_rmse),
            val_mae_pct=float(val_mae),
            val_mae_bps=float(val_mae_bps),
            val_mape=float(val_mape),
            avg_update_norm=float(avg_update_norm),
            min_update_norm=float(min_update_norm),
            max_update_norm=float(max_update_norm),
            std_update_norm=float(std_update_norm),
            client_update_norms=update_norms,
            client_sample_counts=sample_counts,
            client_aggregation_weights=client_weights,
            aggregation_mode=self.aggregation,
            num_accepted=num_accepted,
            duration_sec=round(duration_sec, 3),
        )
        self.history.append(metrics_record)

        logger.info(
            "Round %3d (%s) | Train Loss: %.6f | Val Loss: %.6f | "
            "RMSE: %.4f%% | MAE: %.2f bps | MAPE: %.2f%% | ||Δw||_2: %.4f | Accepted: %d/%d",
            server_round,
            self.aggregation,
            weighted_train_loss,
            val_loss,
            val_rmse,
            val_mae_bps,
            val_mape,
            avg_update_norm,
            num_accepted,
            len(results),
        )

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        summary_metrics: Dict[str, Scalar] = {
            "round": server_round,
            "train_loss": weighted_train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_mae_bps": val_mae_bps,
            "val_mape": val_mape,
            "avg_update_norm": avg_update_norm,
            "num_accepted": num_accepted,
        }

        return aggregated_parameters, summary_metrics

    # -----------------------------------------------------------------------
    # Centralized Validation Benchmark Evaluation
    # -----------------------------------------------------------------------

    def evaluate_global_model(
        self, ndarrays: NDArrays
    ) -> Tuple[float, float, float, float, float]:
        """Evaluate parameter arrays on the isolated central validation benchmark.

        Returns:
        --------
        Tuple[float, float, float, float, float]
            (val_loss, val_rmse, val_mae, val_mae_bps, val_mape)
        """
        if self.eval_dataset is None or self.eval_model is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        X_eval, y_eval = self.eval_dataset

        # Load weights into central evaluation model
        keys = list(self.eval_model.state_dict().keys())
        state_dict = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in zip(keys, ndarrays)
        }
        self.eval_model.load_state_dict(state_dict)
        self.eval_model.eval()

        X_t = torch.tensor(X_eval, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_eval, dtype=torch.float32, device=self.device).view(-1)

        with torch.no_grad():
            preds = self.eval_model(X_t)
            loss = self.criterion(preds, y_t).item()
            preds_np = preds.cpu().numpy().flatten()
            preds_np = np.clip(preds_np, 0.01, None)

        y_eval_flat = y_eval.flatten()
        rmse = compute_rmse(y_eval_flat, preds_np)
        mae = compute_mae(y_eval_flat, preds_np)
        mae_bps = compute_mae_bps(y_eval_flat, preds_np)
        mape = compute_mape(y_eval_flat, preds_np)

        return float(loss), float(rmse), float(mae), float(mae_bps), float(mape)

    # -----------------------------------------------------------------------
    # History Export & Analytics
    # -----------------------------------------------------------------------

    def get_history_dataframe(self) -> pd.DataFrame:
        """Convert round convergence telemetry history to a pandas DataFrame."""
        rows = []
        for h in self.history:
            rows.append({
                "round": h.server_round,
                "train_loss": h.train_loss,
                "val_loss": h.val_loss,
                "val_rmse": h.val_rmse,
                "val_mae_pct": h.val_mae_pct,
                "val_mae_bps": h.val_mae_bps,
                "val_mape": h.val_mape,
                "avg_update_norm": h.avg_update_norm,
                "avg_grad_norm": h.avg_update_norm,  # backward compatibility
                "min_update_norm": h.min_update_norm,
                "max_update_norm": h.max_update_norm,
                "std_update_norm": h.std_update_norm,
                "num_accepted": h.num_accepted,
                "aggregation_mode": h.aggregation_mode,
                "duration_sec": h.duration_sec,
            })
        return pd.DataFrame(rows)

    def save_history_csv(self, filepath: Union[str, Path]) -> Path:
        """Export convergence history to CSV."""
        fp = Path(filepath)
        fp.parent.mkdir(parents=True, exist_ok=True)
        df = self.get_history_dataframe()
        df.to_csv(fp, index=False)
        logger.info("Saved convergence history CSV to %s", fp)
        return fp
