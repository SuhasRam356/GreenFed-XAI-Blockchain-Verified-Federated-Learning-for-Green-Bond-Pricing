"""
Bank Dataset Partitioner — GreenFed-XAI
=======================================

This module partitions proprietary green bond records among participating
financial institutions (banks and asset managers) to simulate decentralized,
privacy-preserving collaborative learning.

As formulated in Section III-C of the IEEE paper:
    Let K = {1, ..., K} be the set of participating institutions (K = 8).
    Each institution k has a local dataset D_k containing bond features
    (credit rating, tenor, coupon rate, ESG controversy score, and IoT-derived
    carbon footprint score C_t) and yield spread targets.

Key Capabilities:
-----------------
1. IID Partitioning: Equal, uniform distribution of bond portfolios across all K banks.
2. Non-IID Dirichlet Partitioning: Simulates realistic banking environments where
   different banks cater to distinct market sectors (e.g., tier-1 banks holding
   predominantly investment-grade AAA/AA bonds vs. regional lenders holding BB/B bonds).
3. Local Validation Splitting: Generates isolated train/validation partitions for
   each client to evaluate local model generalization without central data pooling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ClientDataset:
    """Encapsulates the proprietary private dataset held by an individual bank.

    Attributes:
    -----------
    client_id : int
        Unique integer index of the client institution (0 to K-1).
    bank_name : str
        Human-readable institution identifier (e.g., 'Bank_0_Alpha').
    X_train : np.ndarray
        Standardized local feature matrix for training (samples, features).
    y_train : np.ndarray
        Local yield spread target array for training (samples,).
    X_val : np.ndarray
        Standardized local feature matrix for local evaluation.
    y_val : np.ndarray
        Local yield spread target array for local evaluation.
    raw_df : pd.DataFrame
        Original unscaled DataFrame containing local records with metadata.
    rating_distribution : Dict[str, int]
        Count of bonds per credit rating in this institution's portfolio.
    """

    client_id: int
    bank_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    raw_df: pd.DataFrame
    rating_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def n_train_samples(self) -> int:
        """Return the number of local training samples."""
        return len(self.X_train)

    @property
    def n_val_samples(self) -> int:
        """Return the number of local validation samples."""
        return len(self.X_val)

    def get_train_loader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> DataLoader:
        """Construct a PyTorch DataLoader for local training.

        Parameters:
        -----------
        batch_size : int
            Number of samples per mini-batch. Default is 64.
        shuffle : bool
            Whether to shuffle samples each local epoch. Default is True.
        drop_last : bool
            Whether to drop incomplete final batch. Default is False.

        Returns:
        --------
        DataLoader
            Configured PyTorch DataLoader over local training tensors.
        """
        tensor_x = torch.tensor(self.X_train, dtype=torch.float32)
        tensor_y = torch.tensor(self.y_train, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def get_val_loader(
        self,
        batch_size: int = 64,
        shuffle: bool = False,
    ) -> DataLoader:
        """Construct a PyTorch DataLoader for local evaluation.

        Parameters:
        -----------
        batch_size : int
            Number of samples per mini-batch. Default is 64.
        shuffle : bool
            Whether to shuffle samples. Default is False for evaluation.

        Returns:
        --------
        DataLoader
            Configured PyTorch DataLoader over local validation tensors.
        """
        tensor_x = torch.tensor(self.X_val, dtype=torch.float32)
        tensor_y = torch.tensor(self.y_val, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )


class BankDatasetPartitioner:
    """Manages the partitioning of bond portfolios across K banking institutions.

    Parameters:
    -----------
    n_clients : int
        Number of participating institutions (default K = 8 as in paper).
    val_ratio : float
        Fraction of each bank's local dataset reserved for internal validation.
        Default is 0.15 (15%).
    random_seed : int
        Pseudorandom seed for reproducible data splits.
    """

    BANK_NAMES = [
        "Bank_0_NordicGreen",
        "Bank_1_AlpineCapital",
        "Bank_2_PacificAsset",
        "Bank_3_AtlanticMutual",
        "Bank_4_ContinentalTrust",
        "Bank_5_BorealInvest",
        "Bank_6_EquatorialFin",
        "Bank_7_MeridianHoldings",
    ]

    def __init__(
        self,
        n_clients: int = 8,
        val_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        if n_clients <= 0:
            raise ValueError(f"n_clients must be positive, got {n_clients}")
        if not (0.0 <= val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in [0.0, 1.0), got {val_ratio}")

        self.n_clients = n_clients
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def partition_iid(
        self,
        df_train: pd.DataFrame,
        X_scaled: np.ndarray,
        y: np.ndarray,
    ) -> List[ClientDataset]:
        """Partition records uniformly and independently among all K banks (IID).

        Parameters:
        -----------
        df_train : pd.DataFrame
            Original unscaled training DataFrame (7,000 records).
        X_scaled : np.ndarray
            Standardized feature matrix corresponding to df_train.
        y : np.ndarray
            Target yield spread values corresponding to df_train.

        Returns:
        --------
        List[ClientDataset]
            A list of K ClientDataset objects, each containing its slice of data.
        """
        n_total = len(df_train)
        if len(X_scaled) != n_total or len(y) != n_total:
            raise ValueError(
                f"Dimension mismatch: df_train ({n_total}), "
                f"X_scaled ({len(X_scaled)}), y ({len(y)})"
            )

        indices = np.arange(n_total)
        self._rng.shuffle(indices)

        # Split indices as evenly as possible across K clients
        client_index_splits = np.array_split(indices, self.n_clients)
        client_datasets: List[ClientDataset] = []

        for cid in range(self.n_clients):
            c_indices = client_index_splits[cid]
            c_df = df_train.iloc[c_indices].copy().reset_index(drop=True)
            c_X = X_scaled[c_indices]
            c_y = y[c_indices]

            # Internal train / validation split for this client
            c_train_len = int(len(c_indices) * (1.0 - self.val_ratio))
            train_idx = np.arange(c_train_len)
            val_idx = np.arange(c_train_len, len(c_indices))

            # Compute rating distribution
            rating_counts = c_df["issuer_credit_rating"].value_counts().to_dict()
            bank_name = (
                self.BANK_NAMES[cid]
                if cid < len(self.BANK_NAMES)
                else f"Bank_{cid}_Consortium"
            )

            client_ds = ClientDataset(
                client_id=cid,
                bank_name=bank_name,
                X_train=c_X[train_idx],
                y_train=c_y[train_idx],
                X_val=c_X[val_idx],
                y_val=c_y[val_idx],
                raw_df=c_df,
                rating_distribution=rating_counts,
            )
            client_datasets.append(client_ds)

            logger.info(
                "Initialized Client %d [%s]: %d train samples, %d val samples",
                cid,
                bank_name,
                client_ds.n_train_samples,
                client_ds.n_val_samples,
            )

        return client_datasets

    def partition_non_iid_dirichlet(
        self,
        df_train: pd.DataFrame,
        X_scaled: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.5,
    ) -> List[ClientDataset]:
        """Partition records using a Dirichlet distribution over credit ratings (Non-IID).

        Parameters:
        -----------
        df_train : pd.DataFrame
            Original unscaled training DataFrame.
        X_scaled : np.ndarray
            Standardized feature matrix.
        y : np.ndarray
            Target yield spreads.
        alpha : float
            Dirichlet concentration parameter. Lower alpha = higher heterogeneity.
            Default is 0.5.

        Returns:
        --------
        List[ClientDataset]
            A list of K ClientDataset objects with heterogeneous distributions.
        """
        n_total = len(df_train)
        ratings = df_train["issuer_credit_rating"].values
        unique_ratings = np.unique(ratings)

        client_indices: List[List[int]] = [[] for _ in range(self.n_clients)]

        # Allocate each rating category across clients via Dirichlet distribution
        for rating in unique_ratings:
            rating_idxs = np.where(ratings == rating)[0]
            self._rng.shuffle(rating_idxs)

            # Sample Dirichlet proportions
            proportions = self._rng.dirichlet(np.repeat(alpha, self.n_clients))
            proportions = (
                np.cumsum(proportions) * len(rating_idxs)
            ).astype(int)[:-1]

            splits = np.split(rating_idxs, proportions)
            for cid in range(self.n_clients):
                client_indices[cid].extend(splits[cid].tolist())

        client_datasets: List[ClientDataset] = []
        for cid in range(self.n_clients):
            c_indices = np.array(client_indices[cid], dtype=int)
            self._rng.shuffle(c_indices)

            c_df = df_train.iloc[c_indices].copy().reset_index(drop=True)
            c_X = X_scaled[c_indices]
            c_y = y[c_indices]

            c_train_len = max(1, int(len(c_indices) * (1.0 - self.val_ratio)))
            train_idx = np.arange(c_train_len)
            val_idx = np.arange(c_train_len, len(c_indices))

            rating_counts = c_df["issuer_credit_rating"].value_counts().to_dict()
            bank_name = (
                self.BANK_NAMES[cid]
                if cid < len(self.BANK_NAMES)
                else f"Bank_{cid}_Consortium"
            )

            client_ds = ClientDataset(
                client_id=cid,
                bank_name=bank_name,
                X_train=c_X[train_idx],
                y_train=c_y[train_idx],
                X_val=c_X[val_idx],
                y_val=c_y[val_idx],
                raw_df=c_df,
                rating_distribution=rating_counts,
            )
            client_datasets.append(client_ds)

            logger.info(
                "Non-IID Client %d [%s]: %d train samples, %d val samples",
                cid,
                bank_name,
                client_ds.n_train_samples,
                client_ds.n_val_samples,
            )

        return client_datasets

    def print_partition_summary(
        self, client_datasets: List[ClientDataset]
    ) -> None:
        """Print a formatted tabular summary of client portfolio distributions.

        Parameters:
        -----------
        client_datasets : List[ClientDataset]
            The partitioned client datasets.
        """
        print("\n" + "=" * 78)
        print("  FEDERATED LEARNING: PARTICIPATING BANK PORTFOLIO SUMMARY")
        print("=" * 78)
        header = f"  {'Client ID':<10s} {'Institution Name':<26s} {'Train':>8s} {'Val':>8s} {'Total':>8s}"
        print(header)
        print("  " + "-" * 74)

        total_train = 0
        total_val = 0

        for cds in client_datasets:
            tot = cds.n_train_samples + cds.n_val_samples
            total_train += cds.n_train_samples
            total_val += cds.n_val_samples
            print(
                f"  {cds.client_id:<10d} {cds.bank_name:<26s} "
                f"{cds.n_train_samples:>8d} {cds.n_val_samples:>8d} {tot:>8d}"
            )

        print("  " + "-" * 74)
        print(
            f"  {'TOTAL':<37s} {total_train:>8d} {total_val:>8d} {total_train + total_val:>8d}"
        )
        print("=" * 78 + "\n")

