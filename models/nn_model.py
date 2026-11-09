"""
Neural Network Model — GreenFed-XAI
=====================================

The paper's exact architecture:
  - 3 hidden layers: 128 → 64 → 32 neurons
  - ReLU activation
  - Dropout 0.2
  - Output: 1 neuron (yield spread prediction)

This module is shared across Phase 2 (centralized), Phase 3 (federated),
and Phase 4+ (blockchain-verified federated).
"""

import torch
import torch.nn as nn


class GreenBondNN(nn.Module):
    """Neural network for green bond yield spread prediction.

    Architecture (from the IEEE paper):
        Input → Linear(128) → ReLU → Dropout(0.2)
              → Linear(64)  → ReLU → Dropout(0.2)
              → Linear(32)  → ReLU → Dropout(0.2)
              → Linear(1)   (output: predicted yield spread)

    Parameters
    ----------
    input_dim : int
        Number of input features (after preprocessing / encoding).
    dropout_rate : float
        Dropout probability (default 0.2 as per the paper).
    """

    def __init__(self, input_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

