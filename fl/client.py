r"""
Flower NumPyClient Implementation for Banking Institutions — GreenFed-XAI
=========================================================================

This module implements the production-grade decentralized Flower client
representing an individual financial institution (e.g., bank, pension fund,
or sovereign debt manager) participating in the GreenFed-XAI consortium.

Mathematical Optimization & Update Norm Formulation:
-----------------------------------------------------
In each global communication round $t \in \{1, \dots, T\}$:
1. The client receives global model parameter tensors $w_t \in \mathbb{R}^D$ from the aggregator.
2. The client loads $w_t$ into its local PyTorch neural network blueprint:
   $$w_{t, 0}^{(k)} \leftarrow w_t$$
3. Over $E$ local epochs, the client optimizes local empirical risk using mini-batch gradient steps:
   $$\min_w \mathcal{L}_k(w) = \frac{1}{|\mathcal{D}_k|} \sum_{(x_i, y_i) \in \mathcal{D}_k} \frac{1}{2} \left( f(x_i; w) - y_i \right)^2 + \frac{\lambda}{2} \|w\|_2^2$$
   yielding updated parameters $w_{t+1}^{(k)} = w_{t, E}^{(k)}$.
4. **Update Norm vs. Gradient Norm**:
   Because optimization involves multiple mini-batch steps under Adam (with first and second
   moment vectors $m_\tau, v_\tau$), the total displacement vector:
   $$\Delta w_k = w_{t+1}^{(k)} - w_t$$
   is a multi-step optimizer trajectory. We compute its exact Euclidean $L_2$ norm:
   $$\|\Delta w_k\|_2 = \sqrt{\sum_{l=1}^L \sum_{j} \left( w_{t+1, l, j}^{(k)} - w_{t, l, j} \right)^2}$$
   This is explicitly termed the **update norm**. For exactly one gradient step with vanilla SGD,
   $\Delta w = -\eta \nabla \mathcal{L}$; after multiple epochs of Adam, it represents the cumulative
   parameter displacement vector feeding into the smart contract threshold check $\|\Delta w_k\|_2 \le \tau$.

Evaluation Protocols:
---------------------
- Root Mean Square Error (RMSE, %): $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$
- Mean Absolute Error in Basis Points (MAE, bps): $100 \times \frac{1}{N} \sum |y_i - \hat{y}_i|$ ($1\text{ bp} = 0.01\%$)
- Mean Absolute Percentage Error (MAPE, %): Protected near-zero denominator policy ($\epsilon = 10^{-6}$):
  $$\text{MAPE} = \frac{100}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{\max(|y_i|, 10^{-6})}$$
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import flwr as fl
    from flwr.client import NumPyClient
except ImportError:
    # Graceful fallback stub for environments where Flower is dynamically loaded
    class NumPyClient:  # type: ignore[no-redef]
        """Fallback stub for NumPyClient."""
        pass

from fl.dataset_partitioner import ClientDataset
from models.nn_model import GreenBondNN

logger = logging.getLogger("FlowerClient")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Square Error (RMSE) in percentage points.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground-truth yield spread targets (%).
    y_pred : np.ndarray
        Model predicted yield spreads (%).

    Returns:
    --------
    float
        RMSE expressed in percentage points (e.g. 0.15 means 0.15%).
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    if len(y_t) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE) in percentage points.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground-truth yield spread targets (%).
    y_pred : np.ndarray
        Model predicted yield spreads (%).

    Returns:
    --------
    float
        MAE expressed in percentage points (e.g. 0.07 means 0.07%).
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(np.abs(y_t - y_p)))


def compute_mae_bps(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""Compute Mean Absolute Error (MAE) in basis points (1 bp = 0.01%).

    $$\text{MAE}_{\text{bps}} = 100 \times \text{MAE}_{\%}$$

    Parameters:
    -----------
    y_true : np.ndarray
        Ground-truth yield spread targets (%).
    y_pred : np.ndarray
        Model predicted yield spreads (%).

    Returns:
    --------
    float
        MAE expressed in basis points (e.g. 7.15 means 7.15 bps).
    """
    return compute_mae(y_true, y_pred) * 100.0


def compute_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    r"""Compute Mean Absolute Percentage Error (MAPE) with protected denominator.

    Enforces the Master Plan near-zero policy:
    $$\text{MAPE} = \frac{100}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{\max(|y_i|, \epsilon)}$$

    Parameters:
    -----------
    y_true : np.ndarray
        Ground-truth yield spread targets (%).
    y_pred : np.ndarray
        Model predicted yield spreads (%).
    epsilon : float
        Protected denominator floor (default: 1e-6).

    Returns:
    --------
    float
        MAPE expressed as percentage (e.g. 3.5 means 3.5%).
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    if len(y_t) == 0:
        return 0.0
    denom = np.clip(np.abs(y_t), epsilon, None)
    return float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)


class GreenBondFlowerClient(NumPyClient):
    """Production-grade Flower NumPyClient for green bond decentralized training.

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
        Learning rate for local optimizer (default: 0.001).
    local_epochs : int
        Number of local training epochs per communication round (default: 3).
    batch_size : int
        Local training mini-batch size (default: 64).
    weight_decay : float
        L2 regularization penalty (default: 1e-4).
    optimizer_type : str
        Local optimizer choice: 'adam' or 'sgd' (default: 'adam').
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
        optimizer_type: str = "adam",
    ) -> None:
        super().__init__()
        self.client_id = client_id
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type.lower()

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        input_dim = dataset.get_feature_dimension()
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
        """Extract model parameters as contiguous CPU NumPy ndarrays.

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
                f"Parameter layer mismatch for Bank {self.client_id}: "
                f"model expects {len(keys)} layers, received {len(parameters)}"
            )

        state_dict = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in zip(keys, parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    # -----------------------------------------------------------------------
    # Update Norm Computation
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_update_norm(
        initial_params: List[np.ndarray],
        updated_params: List[np.ndarray],
    ) -> float:
        r"""Compute the exact Euclidean $L_2$ norm of the parameter update vector:

        $$\|\Delta w\|_2 = \sqrt{\sum_{l=1}^L \sum_{j} (w_{new, l, j} - w_{old, l, j})^2}$$

        Parameters:
        -----------
        initial_params : List[np.ndarray]
            Global weights $w_t$ broadcast at the beginning of the round.
        updated_params : List[np.ndarray]
            Local weights $w_{t+1}^{(k)}$ obtained after local optimization.

        Returns:
        --------
        float
            Total Euclidean L2 norm of the parameter displacement vector.
        """
        total_squared_norm = 0.0
        for p_init, p_upd in zip(initial_params, updated_params):
            diff = p_upd - p_init
            total_squared_norm += float(np.sum(diff ** 2))
        return float(np.sqrt(total_squared_norm))

    @staticmethod
    def compute_gradient_norm(
        initial_params: List[np.ndarray],
        updated_params: List[np.ndarray],
    ) -> float:
        """Backward-compatible alias for compute_update_norm."""
        return GreenBondFlowerClient.compute_update_norm(initial_params, updated_params)

    # -----------------------------------------------------------------------
    # Local Training (fit)
    # -----------------------------------------------------------------------

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        r"""Perform local training using private institutional bond records.

        Parameters:
        -----------
        parameters : List[np.ndarray]
            Global model parameters broadcast by server for round t.
        config : Dict[str, Any]
            Hyperparameter dictionary from server (epochs, lr, batch_size).

        Returns:
        --------
        Tuple[List[np.ndarray], int, Dict[str, Any]]
            - Updated local model weights $w_{t+1}^{(k)}$.
            - Total training examples used ($|\mathcal{D}_k|$).
            - Telemetry metrics dictionary containing loss, update norm, timing.
        """
        fit_start = time.time()

        # 1. Load broadcast global parameters
        self.set_parameters(parameters)
        initial_weights = [p.copy() for p in parameters]

        # 2. Extract configuration overrides
        epochs = int(config.get("local_epochs", self.local_epochs))
        lr = float(config.get("learning_rate", self.learning_rate))
        batch_size = int(config.get("batch_size", self.batch_size))
        current_round = int(config.get("current_round", 1))
        opt_type = str(config.get("optimizer_type", self.optimizer_type)).lower()

        train_loader = self.dataset.get_train_loader(
            batch_size=batch_size, shuffle=True
        )

        # 3. Instantiate local optimizer
        if opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.weight_decay,
            )

        self.model.train()
        epoch_losses: List[float] = []
        batch_count = 0

        for ep in range(epochs):
            batch_losses: List[float] = []
            for X_b, y_b in train_loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device).view(-1)

                optimizer.zero_grad()
                preds = self.model(X_b)
                loss = self.criterion(preds, y_b)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                batch_count += 1

            epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            epoch_losses.append(epoch_loss)

        # 4. Extract updated weights and compute update norm
        updated_weights = self.get_parameters()
        update_norm = self.compute_update_norm(initial_weights, updated_weights)
        train_duration_ms = (time.time() - fit_start) * 1000.0

        final_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        logger.debug(
            "[Round %d] Bank %d (%s) finished %d epochs in %.1fms | Loss: %.6f | ||dw||: %.4f",
            current_round,
            self.client_id,
            self.dataset.bank_name,
            epochs,
            train_duration_ms,
            final_loss,
            update_norm,
        )

        metrics: Dict[str, Any] = {
            "client_id": self.client_id,
            "bank_name": self.dataset.bank_name,
            "train_loss": final_loss,
            "update_norm": update_norm,
            "grad_norm": update_norm,  # backward compatibility
            "round": current_round,
            "epochs": epochs,
            "batch_count": batch_count,
            "duration_ms": round(train_duration_ms, 2),
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
        """Evaluate received global parameters on local private validation records.

        Parameters:
        -----------
        parameters : List[np.ndarray]
            Model parameters to evaluate.
        config : Dict[str, Any]
            Evaluation configuration parameters.

        Returns:
        --------
        Tuple[float, int, Dict[str, Any]]
            - Validation MSE loss.
            - Number of local validation samples.
            - Metrics dictionary with RMSE (%), MAE (bps), and MAPE (%).
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
                y_b = y_b.to(self.device).view(-1)

                preds = self.model(X_b)
                loss = self.criterion(preds, y_b)
                total_loss += loss.item() * len(X_b)

                all_preds.append(preds.cpu().numpy().flatten())
                all_targets.append(y_b.cpu().numpy().flatten())

        n_samples = max(1, self.dataset.n_val_samples)
        avg_loss = total_loss / n_samples

        if all_preds:
            y_pred_concat = np.concatenate(all_preds)
            y_true_concat = np.concatenate(all_targets)

            rmse = compute_rmse(y_true_concat, y_pred_concat)
            mae = compute_mae(y_true_concat, y_pred_concat)
            mae_bps = compute_mae_bps(y_true_concat, y_pred_concat)
            mape = compute_mape(y_true_concat, y_pred_concat)
        else:
            rmse = 0.0
            mae = 0.0
            mae_bps = 0.0
            mape = 0.0

        metrics: Dict[str, Any] = {
            "client_id": self.client_id,
            "bank_name": self.dataset.bank_name,
            "val_loss": float(avg_loss),
            "val_rmse": float(rmse),
            "val_mae_pct": float(mae),
            "val_mae_bps": float(mae_bps),
            "val_mape": float(mape),
        }

        return float(avg_loss), self.dataset.n_val_samples, metrics
