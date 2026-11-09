r"""
Flower Client Implementation — GreenFed-XAI
===========================================

This module implements a production-grade Flower NumPyClient representing a
single financial institution (e.g., bank or asset manager) participating in the
decentralized green bond pricing consortium.

Theoretical Formulation:
------------------------
In each global communication round t:
1. Client k receives current global model weights w_t from the FL server.
2. Client k performs local optimization on its proprietary dataset D_k:
       w_{t+1}^{(k)} = w_t - \eta \nabla L_k(w_t)
   where L_k is the mean squared error (MSE) over D_k.
3. Client k computes its parameter update (gradient vector proxy):
       \Delta w_k = w_{t+1}^{(k)} - w_t
   and its Euclidean L2 norm:
       ||\Delta w_k||_2 = \sqrt{\sum_l ||\Delta w_{k, l}||_2^2}
4. Returns the updated parameters along with local training metadata (loss,
   gradient norm, sample count) to the aggregator.

The gradient norm computation implemented here directly feeds into:
- Phase 3: Monitoring convergence stability across heterogeneous banks.
- Phase 4: Blockchain smart contract verification (||\Delta w||_2 <= \tau = 2.5).
- Phase 5: Poisoning attack defense against malicious participants.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import flwr as fl
    from flwr.client import NumPyClient
except ImportError:
    # Graceful fallback for mock unit testing before flwr finishes loading
    class NumPyClient:  # type: ignore[no-redef]
        """Fallback stub when Flower is being initialized."""
        pass

from fl.dataset_partitioner import ClientDataset
from models.nn_model import GreenBondNN

logger = logging.getLogger(__name__)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Square Error (%)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error (%).

    Parameters:
    -----------
    y_true : np.ndarray
        Ground-truth yield spread targets.
    y_pred : np.ndarray
        Model predicted yield spreads.

    Returns:
    --------
    float
        MAPE expressed as percentage (e.g., 3.5 means 3.5%).
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    denom = np.clip(np.abs(y_t), 1e-6, None)
    return float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)


class GreenBondFlowerClient(NumPyClient):
    """Flower client representing an individual bank training on private bond data.

    Parameters:
    -----------
    client_id : int
        Unique integer ID of this bank (0 to K-1).
    dataset : ClientDataset
        The isolated private dataset belonging to this institution.
    model : Optional[GreenBondNN]
        Neural network model instance. If None, instantiated automatically.
    device : Optional[torch.device]
        Computation device (CPU or CUDA).
    learning_rate : float
        Learning rate for local Adam optimizer (default: 0.001).
    local_epochs : int
        Number of local training epochs per communication round (default: 3).
    batch_size : int
        Local training mini-batch size (default: 64).
    weight_decay : float
        L2 regularization penalty for local training (default: 1e-4).
    """

    def __init__(
        self,
        client_id: int,
        dataset: ClientDataset,
        model: Optional[GreenBondNN] = None,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        local_epochs: int = 3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.client_id = client_id
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        input_dim = dataset.X_train.shape[1]
        self.model = (
            model.to(self.device)
            if model is not None
            else GreenBondNN(input_dim=input_dim, dropout_rate=0.2).to(self.device)
        )

        self.criterion = nn.MSELoss()

    # -----------------------------------------------------------------------
    # Parameter Serialization (Flower NumPy <-> PyTorch Tensor)
    # -----------------------------------------------------------------------

    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """Extract model parameters as a list of NumPy ndarrays.

        Parameters:
        -----------
        config : Optional[Dict[str, Any]]
            Optional configuration dictionary passed from the FL server.

        Returns:
        --------
        List[np.ndarray]
            List of contiguous CPU NumPy arrays corresponding to PyTorch weights.
        """
        return [
            val.cpu().detach().numpy().copy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load external parameters into the local PyTorch model state dict.

        Parameters:
        -----------
        parameters : List[np.ndarray]
            List of NumPy arrays received from the server aggregator.
        """
        keys = list(self.model.state_dict().keys())
        if len(keys) != len(parameters):
            raise ValueError(
                f"Parameter length mismatch: model expects {len(keys)} layers, "
                f"received {len(parameters)}"
            )

        state_dict = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in zip(keys, parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    # -----------------------------------------------------------------------
    # Gradient Norm and Update Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_gradient_norm(
        initial_params: List[np.ndarray],
        updated_params: List[np.ndarray],
    ) -> float:
        r"""Compute the Euclidean L2 norm of the parameter update vector:

            ||\Delta w||_2 = \sqrt{\sum_l \sum_j (w_{new, j} - w_{old, j})^2}

        This metric measures the magnitude of the model adjustment performed
        by this client in the current round.

        Parameters:
        -----------
        initial_params : List[np.ndarray]
            Global weights w_t broadcast at the beginning of the round.
        updated_params : List[np.ndarray]
            Local weights w_{t+1}^{(k)} obtained after local SGD/Adam.

        Returns:
        --------
        float
            The total Euclidean L2 norm of the update vector.
        """
        total_squared_norm = 0.0
        for p_init, p_upd in zip(initial_params, updated_params):
            diff = p_upd - p_init
            total_squared_norm += float(np.sum(diff ** 2))
        return float(np.sqrt(total_squared_norm))

    # -----------------------------------------------------------------------
    # Local Training (fit)
    # -----------------------------------------------------------------------

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Perform local training using private banking dataset.

        Parameters:
        -----------
        parameters : List[np.ndarray]
            Global model parameters broadcast by server for round t.
        config : Dict[str, Any]
            Hyperparameter dictionary from server (e.g., local_epochs, lr).

        Returns:
        --------
        Tuple[List[np.ndarray], int, Dict[str, Any]]
            - Updated local model weights w_{t+1}^{(k)}.
            - Total number of training samples used (|D_k|).
            - Metadata dictionary containing train loss, grad norm, and diagnostics.
        """
        # Load broadcast global weights
        self.set_parameters(parameters)
        initial_weights = [p.copy() for p in parameters]

        # Extract round config overrides if provided by server
        epochs = int(config.get("local_epochs", self.local_epochs))
        lr = float(config.get("learning_rate", self.learning_rate))
        batch_size = int(config.get("batch_size", self.batch_size))
        current_round = int(config.get("current_round", 1))

        train_loader = self.dataset.get_train_loader(
            batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
        )

        self.model.train()
        epoch_losses: List[float] = []

        for ep in range(epochs):
            batch_losses: List[float] = []
            for X_b, y_b in train_loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                preds = self.model(X_b)
                loss = self.criterion(preds, y_b)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            epoch_loss = float(np.mean(batch_losses))
            epoch_losses.append(epoch_loss)

        # Retrieve updated parameters
        updated_weights = self.get_parameters()

        # Compute update vector L2 norm (||\Delta w||_2)
        grad_norm = self.compute_gradient_norm(initial_weights, updated_weights)
        final_loss = float(np.mean(epoch_losses))

        logger.debug(
            "[Round %d] Bank %d (%s) finished %d epochs | Loss: %.6f | ||dw||_2: %.4f",
            current_round,
            self.client_id,
            self.dataset.bank_name,
            epochs,
            final_loss,
            grad_norm,
        )

        metrics = {
            "client_id": self.client_id,
            "bank_name": self.dataset.bank_name,
            "train_loss": final_loss,
            "grad_norm": grad_norm,
            "round": current_round,
        }

        return updated_weights, self.dataset.n_train_samples, metrics

    # -----------------------------------------------------------------------
    # Local Evaluation (evaluate)
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate received model weights on local private validation records.

        Parameters:
        -----------
        parameters : List[np.ndarray]
            Model weights to evaluate.
        config : Dict[str, Any]
            Evaluation configuration parameters.

        Returns:
        --------
        Tuple[float, int, Dict[str, Any]]
            - Validation MSE loss.
            - Number of local validation samples.
            - Metrics dictionary with local RMSE (%) and MAPE (%).
        """
        self.set_parameters(parameters)
        self.model.eval()

        val_loader = self.dataset.get_val_loader(
            batch_size=self.batch_size, shuffle=False
        )

        total_loss = 0.0
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device)

                preds = self.model(X_b)
                loss = self.criterion(preds, y_b)
                total_loss += loss.item() * len(X_b)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_b.cpu().numpy())

        n_samples = max(1, self.dataset.n_val_samples)
        avg_loss = total_loss / n_samples

        if all_preds:
            y_pred_concat = np.concatenate(all_preds)
            y_pred_concat = np.clip(y_pred_concat, 0.01, None)
            y_true_concat = np.concatenate(all_targets)

            rmse = compute_rmse(y_true_concat, y_pred_concat)
            mape = compute_mape(y_true_concat, y_pred_concat)
        else:
            rmse = 0.0
            mape = 0.0

        metrics = {
            "client_id": self.client_id,
            "val_loss": float(avg_loss),
            "val_rmse": float(rmse),
            "val_mape": float(mape),
        }

        return float(avg_loss), self.dataset.n_val_samples, metrics

