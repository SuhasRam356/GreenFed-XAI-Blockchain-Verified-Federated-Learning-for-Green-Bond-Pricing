r"""
Federated Averaging Strategy (FedAvg) — GreenFed-XAI
===================================================

This module implements the plain Federated Averaging (FedAvg) strategy
(McMahan et al., 2017) customized for green bond yield spread prediction.

Formulation:
------------
In each global communication round t:
1. The coordinator broadcasts global model weights w_t to all K participating banks.
2. Each bank k performs local training on its private dataset D_k, producing
   updated weights w_{t+1}^{(k)}.
3. The server aggregates the client weights proportional to their dataset sizes:
       w_{t+1} = \sum_{k=1}^K \frac{|D_k|}{|D|} w_{t+1}^{(k)}
   where |D| = \sum_k |D_k|.

Key Capabilities & Production Extensions:
------------------------------------------
- Tracks round-level loss curves (training loss, validation MSE, RMSE, MAPE).
- Evaluates the aggregated global model on an isolated validation/test benchmark.
- Records client update gradient norms (||\\Delta w_k||_2) across rounds.
- Designed with modular hook points (`verify_and_filter_updates`) that will be
  extended in Phase 4 by `GreenBlockchainStrategy` to incorporate on-chain
  cryptographic hash validation and gradient poisoning rejection (\\tau = 2.5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    # Fallback placeholders when Flower is installing
    class FedAvg:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass

    Parameters = Any  # type: ignore[assignment, misc]
    NDArrays = List[np.ndarray]  # type: ignore[assignment, misc]
    FitRes = Any  # type: ignore[assignment, misc]
    ClientProxy = Any  # type: ignore[assignment, misc]

from fl.client import compute_mape, compute_rmse
from models.nn_model import GreenBondNN

logger = logging.getLogger(__name__)


@dataclass
class RoundConvergenceMetrics:
    """Stores performance metrics collected for a single FL round."""

    round_idx: int
    train_loss: float
    val_loss: float
    val_rmse: float
    val_mape: float
    avg_grad_norm: float
    client_grad_norms: Dict[int, float] = field(default_factory=dict)


class GreenFedAvgStrategy(FedAvg):
    """Federated Averaging strategy tailored for Green Bond yield spread prediction.

    Parameters:
    -----------
    eval_dataset : Optional[Tuple[np.ndarray, np.ndarray]]
        Optional centralized benchmark dataset (X_val, y_val) used to evaluate
        the global model after every aggregation round.
    model_template : Optional[GreenBondNN]
        Neural network architecture blueprint for server-side evaluation.
    device : Optional[torch.device]
        PyTorch device (CPU or CUDA).
    fraction_fit : float
        Fraction of available clients sampled for training (default: 1.0 = all).
    min_fit_clients : int
        Minimum number of clients required to participate per round (default: 8).
    min_available_clients : int
        Minimum total clients that must be connected (default: 8).
    initial_parameters : Optional[Parameters]
        Pre-initialized model parameters broadcast in round 0.
    """

    def __init__(
        self,
        eval_dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        model_template: Optional[GreenBondNN] = None,
        device: Optional[torch.device] = None,
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

        # Convergence history tracking
        self.history: List[RoundConvergenceMetrics] = []
        self.latest_global_ndarrays: Optional[NDArrays] = None

    # -----------------------------------------------------------------------
    # Aggregation Hook (Extensible for Blockchain Verification in Phase 4)
    # -----------------------------------------------------------------------

    def verify_and_filter_updates(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[List[Tuple[ClientProxy, FitRes]], Dict[str, Any]]:
        """Hook to filter client updates before federated aggregation.

        In Phase 3 (plain FedAvg), all submitted updates are accepted.
        In Phase 4 (GreenFed-XAI), this method is overridden by the blockchain
        verification layer to check hash receipts and enforce ||dw||_2 <= tau.

        Parameters:
        -----------
        server_round : int
            Current communication round index.
        results : List[Tuple[ClientProxy, FitRes]]
            Raw list of (ClientProxy, FitRes) pairs from all participating clients.

        Returns:
        --------
        Tuple[List[Tuple[ClientProxy, FitRes]], Dict[str, Any]]
            - Accepted client updates.
            - Audit metadata dictionary.
        """
        accepted = results
        audit_metadata: Dict[str, Any] = {
            "total_submitted": len(results),
            "total_accepted": len(results),
            "total_rejected": 0,
            "verification_mode": "plain_fedavg",
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
        """Aggregate model weights using Federated Averaging weighted by sample count.

        Parameters:
        -----------
        server_round : int
            Current global communication round.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful client fit responses.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Client execution failures (if any).

        Returns:
        --------
        Tuple[Optional[Parameters], Dict[str, Scalar]]
            - Newly aggregated global parameters.
            - Strategy metrics dictionary.
        """
        if not results:
            logger.warning("[Round %d] No client fit results to aggregate.", server_round)
            return None, {}

        # Apply filtering hook (plain in Phase 3, blockchain-verified in Phase 4)
        accepted_results, audit_meta = self.verify_and_filter_updates(
            server_round, results
        )

        if not accepted_results:
            logger.error(
                "[Round %d] All client updates were rejected! Aggregation aborted.",
                server_round,
            )
            return None, {}

        # Extract client gradient norms & train losses
        grad_norms: Dict[int, float] = {}
        weighted_train_loss = 0.0
        total_samples = sum(fit_res.num_examples for _, fit_res in accepted_results)

        for _, fit_res in accepted_results:
            cid = int(fit_res.metrics.get("client_id", -1))
            gnorm = float(fit_res.metrics.get("grad_norm", 0.0))
            tloss = float(fit_res.metrics.get("train_loss", 0.0))
            grad_norms[cid] = gnorm
            weighted_train_loss += tloss * fit_res.num_examples

        weighted_train_loss /= max(1, total_samples)
        avg_grad_norm = float(np.mean(list(grad_norms.values()))) if grad_norms else 0.0

        # Weighted parameter averaging: w = \sum_k (|D_k| / |D|) * w_k
        first_weights = parameters_to_ndarrays(accepted_results[0][1].parameters)
        aggregated_ndarrays = [np.zeros_like(layer) for layer in first_weights]

        for _, fit_res in accepted_results:
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weight_factor = fit_res.num_examples / total_samples

            for i, layer in enumerate(client_ndarrays):
                aggregated_ndarrays[i] += layer * weight_factor

        self.latest_global_ndarrays = aggregated_ndarrays

        # Centralized benchmark evaluation if dataset provided
        val_loss, val_rmse, val_mape = 0.0, 0.0, 0.0
        if self.eval_dataset is not None and self.eval_model is not None:
            val_loss, val_rmse, val_mape = self._evaluate_global_model(
                aggregated_ndarrays
            )

        # Store convergence metrics
        metrics_record = RoundConvergenceMetrics(
            round_idx=server_round,
            train_loss=float(weighted_train_loss),
            val_loss=float(val_loss),
            val_rmse=float(val_rmse),
            val_mape=float(val_mape),
            avg_grad_norm=float(avg_grad_norm),
            client_grad_norms=grad_norms,
        )
        self.history.append(metrics_record)

        logger.info(
            "[Round %3d] Train Loss: %.6f | Val Loss: %.6f | RMSE: %.4f%% | MAPE: %.2f%% | ||dw||: %.4f",
            server_round,
            weighted_train_loss,
            val_loss,
            val_rmse,
            val_mape,
            avg_grad_norm,
        )

        aggregated_params = ndarrays_to_parameters(aggregated_ndarrays)
        metrics_out: Dict[str, Scalar] = {
            "train_loss": weighted_train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_mape": val_mape,
            "avg_grad_norm": avg_grad_norm,
            "clients_aggregated": len(accepted_results),
        }

        return aggregated_params, metrics_out

    # -----------------------------------------------------------------------
    # Centralized Evaluation of Aggregated Weights
    # -----------------------------------------------------------------------

    def _evaluate_global_model(
        self, parameters: NDArrays
    ) -> Tuple[float, float, float]:
        """Evaluate aggregated global parameters on the central benchmark split."""
        assert self.eval_model is not None and self.eval_dataset is not None
        X_eval, y_eval = self.eval_dataset

        # Load weights into eval model
        keys = list(self.eval_model.state_dict().keys())
        state_dict = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in zip(keys, parameters)
        }
        self.eval_model.load_state_dict(state_dict)
        self.eval_model.eval()

        X_t = torch.tensor(X_eval, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_eval, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            preds = self.eval_model(X_t)
            loss = self.criterion(preds, y_t).item()
            preds_np = preds.cpu().numpy()
            preds_np = np.clip(preds_np, 0.01, None)

        rmse = compute_rmse(y_eval, preds_np)
        mape = compute_mape(y_eval, preds_np)

        return float(loss), float(rmse), float(mape)

    # -----------------------------------------------------------------------
    # History Export & Analytics
    # -----------------------------------------------------------------------

    def get_history_dataframe(self) -> pd.DataFrame:
        """Export round-by-round convergence metrics as a pandas DataFrame."""
        rows = []
        for h in self.history:
            rows.append({
                "round": h.round_idx,
                "train_loss": h.train_loss,
                "val_loss": h.val_loss,
                "val_rmse": h.val_rmse,
                "val_mape": h.val_mape,
                "avg_grad_norm": h.avg_grad_norm,
            })
        return pd.DataFrame(rows)

    def save_history_csv(self, output_path: Path) -> Path:
        """Save history table to CSV."""
        df = self.get_history_dataframe()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved FL convergence history -> %s", output_path)
        return output_path

