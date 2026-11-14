r"""
Bank Dataset Partitioner & Institutional Portfolio Manager — GreenFed-XAI
========================================================================

This module implements enterprise production-grade dataset partitioning and
portfolio simulation across the $K = 8$ participating financial institutions
in the GreenFed-XAI decentralized consortium.

Mathematical & Regulatory Formulation:
--------------------------------------
As specified in Section III-C and Section IV-C of the IEEE paper:
    Let $\mathcal{K} = \{0, 1, \dots, K-1\}$ be the consortium of $K = 8$ banks.
    Each institution $k$ holds a private portfolio dataset $\mathcal{D}_k = \{(x_i^{(k)}, y_i^{(k)})\}_{i=1}^{n_k}$,
    where $x_i \in \mathbb{R}^d$ includes bond attributes (credit rating, tenor, coupon,
    ESG controversy, and IoT-verified carbon score $C_t$) and $y_i \in \mathbb{R}$
    denotes the annualized green bond yield spread in percentage points ($1\text{ bp} = 0.01\%$).

Partitioning Regimes:
---------------------
1. Stratified IID Partitioning:
   - Uniform distribution across all banks with stratified sampling across credit rating tiers.
   - Ensures identical empirical distributions: $\mathcal{P}_k(x, y) \approx \mathcal{P}_j(x, y) \quad \forall j, k$.
   - Sample sizes are identical ($n_k = \lfloor N_{\text{train}} / K \rfloor$).
2. Non-IID Dirichlet Partitioning:
   - Simulates real-world market segmentation where different institutions specialize in distinct
     credit rating categories (e.g., Tier-1 investment banks vs. regional lenders).
   - Rating proportions for each bank $k$ are drawn from a Dirichlet distribution:
     $$p_k \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_R)$$
     where $\alpha > 0$ controls the concentration/heterogeneity parameter (default $\alpha = 0.5$).
3. Non-IID Rating-Tier Market Specialization:
   - Explicit institutional specialization: Banks 0-3 specialize in Investment Grade bonds
     (AAA, AA, A, BBB), while Banks 4-7 specialize in High Yield / Speculative bonds (BB, B).
   - Reflects Basel III capital requirement constraints and institutional investment mandates.

Statistical Heterogeneity Diagnostics:
---------------------------------------
- Computes pairwise Jensen-Shannon (JS) divergence between institutional rating distributions.
- Computes sample size variance and portfolio feature moments (mean spread, carbon score, tenor).
- Exports comprehensive portfolio audit diagnostics to CSV for compliance inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Configure module-level logger
logger = logging.getLogger("DatasetPartitioner")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [Partitioner]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class InstitutionProfile:
    """Institutional metadata defining a consortium participant's profile.

    Attributes:
    -----------
    client_id : int
        Unique integer ID (0 to K-1).
    bank_name : str
        Formal institution name.
    institution_type : str
        Regulatory classification (e.g. 'Global Systemically Important Bank').
    jurisdiction : str
        Primary operating jurisdiction (e.g. 'Nordic EEA', 'Alpine DACH').
    esg_mandate : str
        Institutional ESG investment policy mandate.
    risk_profile : str
        Portfolio risk appetite ('Conservative', 'Balanced', 'High-Yield').
    """

    client_id: int
    bank_name: str
    institution_type: str
    jurisdiction: str
    esg_mandate: str
    risk_profile: str


# Canonical Consortium Profiles for the 8 Participating Banks
CONSORTIUM_PROFILES: Tuple[InstitutionProfile, ...] = (
    InstitutionProfile(
        client_id=0,
        bank_name="Bank_0_NordicGreen",
        institution_type="Global Systemically Important Bank (G-SIB)",
        jurisdiction="Nordic / EEA",
        esg_mandate="Strict Article 9 SFDR Impact Portfolio",
        risk_profile="Conservative / AAA-AA Focused",
    ),
    InstitutionProfile(
        client_id=1,
        bank_name="Bank_1_AlpineCapital",
        institution_type="European Commercial Bank",
        jurisdiction="DACH / Central Europe",
        esg_mandate="Article 8 Transition Infrastructure",
        risk_profile="Conservative / Investment Grade",
    ),
    InstitutionProfile(
        client_id=2,
        bank_name="Bank_2_PacificAsset",
        institution_type="Institutional Asset Manager",
        jurisdiction="Asia-Pacific",
        esg_mandate="Renewable Energy Project Finance",
        risk_profile="Balanced / A-BBB Focus",
    ),
    InstitutionProfile(
        client_id=3,
        bank_name="Bank_3_AtlanticMutual",
        institution_type="Mutual Savings & Loan Consortium",
        jurisdiction="North America",
        esg_mandate="Clean Municipal & Green Housing Bonds",
        risk_profile="Balanced / Investment Grade",
    ),
    InstitutionProfile(
        client_id=4,
        bank_name="Bank_4_ContinentalTrust",
        institution_type="Multinational Investment Bank",
        jurisdiction="Western Europe",
        esg_mandate="Decarbonization Debt Syndication",
        risk_profile="Balanced / Broad Market",
    ),
    InstitutionProfile(
        client_id=5,
        bank_name="Bank_5_BorealInvest",
        institution_type="Public Pension & Sovereign Affiliate",
        jurisdiction="Northern Europe",
        esg_mandate="Long-Tenor Circular Economy Bonds",
        risk_profile="Conservative / High Tenor Focus",
    ),
    InstitutionProfile(
        client_id=6,
        bank_name="Bank_6_EquatorialFin",
        institution_type="Emerging Market Development Bank",
        jurisdiction="Latin America & Caribbean",
        esg_mandate="Reforestation & Biodiversity Debt",
        risk_profile="High-Yield / Growth Focused",
    ),
    InstitutionProfile(
        client_id=7,
        bank_name="Bank_7_MeridianHoldings",
        institution_type="Private Debt & Specialized Infrastructure Fund",
        jurisdiction="Global Offshore",
        esg_mandate="Clean Tech Venture Debt",
        risk_profile="High-Yield / Opportunistic",
    ),
)


@dataclass
class ClientDataset:
    """Encapsulates the isolated private dataset held by an individual bank.

    Attributes:
    -----------
    client_id : int
        Unique integer index of the client institution (0 to K-1).
    bank_name : str
        Human-readable institution identifier.
    X_train : np.ndarray
        Standardized local feature matrix for training (samples, features).
    y_train : np.ndarray
        Local yield spread target vector for training (samples,).
    X_val : np.ndarray
        Standardized local feature matrix for local evaluation.
    y_val : np.ndarray
        Local yield spread target vector for local evaluation.
    raw_df : pd.DataFrame
        Original unscaled DataFrame containing local records with metadata.
    rating_distribution : Dict[str, int]
        Count of bonds per credit rating in this institution's portfolio.
    profile : Optional[InstitutionProfile]
        Institutional metadata and regulatory profile.
    """

    client_id: int
    bank_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    raw_df: pd.DataFrame
    rating_distribution: Dict[str, int] = field(default_factory=dict)
    profile: Optional[InstitutionProfile] = None

    def __post_init__(self) -> None:
        """Validate shapes and dimensions upon initialization."""
        if len(self.X_train) != len(self.y_train):
            raise ValueError(
                f"Client {self.client_id} train length mismatch: "
                f"X={len(self.X_train)}, y={len(self.y_train)}"
            )
        if len(self.X_val) != len(self.y_val):
            raise ValueError(
                f"Client {self.client_id} val length mismatch: "
                f"X={len(self.X_val)}, y={len(self.y_val)}"
            )
        # Ensure 1D target vector for seamless PyTorch loss computation
        if self.y_train.ndim > 1:
            self.y_train = self.y_train.flatten()
        if self.y_val.ndim > 1:
            self.y_val = self.y_val.flatten()

    @property
    def n_train_samples(self) -> int:
        """Return the number of local training samples."""
        return len(self.X_train)

    @property
    def n_val_samples(self) -> int:
        """Return the number of local validation samples."""
        return len(self.X_val)

    @property
    def total_samples(self) -> int:
        """Return total samples assigned to this institution."""
        return self.n_train_samples + self.n_val_samples

    def get_feature_dimension(self) -> int:
        """Return the number of feature dimensions."""
        return self.X_train.shape[1] if self.X_train.ndim > 1 else 0

    def compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute portfolio summary statistics for institutional diagnostics.

        Returns:
        --------
        Dict[str, Any]
            Portfolio feature averages, spread dispersion, and ratings breakdown.
        """
        stats: Dict[str, Any] = {
            "client_id": self.client_id,
            "bank_name": self.bank_name,
            "n_train": self.n_train_samples,
            "n_val": self.n_val_samples,
            "mean_yield_spread_pct": float(np.mean(self.y_train)) if len(self.y_train) > 0 else 0.0,
            "std_yield_spread_pct": float(np.std(self.y_train)) if len(self.y_train) > 0 else 0.0,
            "min_yield_spread_pct": float(np.min(self.y_train)) if len(self.y_train) > 0 else 0.0,
            "max_yield_spread_pct": float(np.max(self.y_train)) if len(self.y_train) > 0 else 0.0,
        }

        # Include unscaled raw features if available
        if isinstance(self.raw_df, pd.DataFrame) and not self.raw_df.empty:
            if "carbon_footprint_score" in self.raw_df.columns:
                stats["mean_carbon_score"] = float(self.raw_df["carbon_footprint_score"].mean())
            if "esg_controversy_score" in self.raw_df.columns:
                stats["mean_controversy_score"] = float(self.raw_df["esg_controversy_score"].mean())
            if "coupon_rate" in self.raw_df.columns:
                stats["mean_coupon_rate"] = float(self.raw_df["coupon_rate"].mean())
            if "tenor" in self.raw_df.columns:
                stats["mean_tenor_years"] = float(self.raw_df["tenor"].mean())

        stats["rating_distribution"] = self.rating_distribution
        return stats

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
        tensor_y = torch.tensor(self.y_train, dtype=torch.float32).view(-1)
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
        tensor_y = torch.tensor(self.y_val, dtype=torch.float32).view(-1)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )


class BankDatasetPartitioner:
    """Partitions green bond records across consortium members.

    Parameters:
    -----------
    n_clients : int
        Number of participating financial institutions (default: 8).
    val_ratio : float
        Fraction of local records reserved for local validation (default: 0.15).
    random_seed : int
        RNG seed for deterministic partitioning (default: 42).
    """

    BANK_NAMES: Tuple[str, ...] = tuple(p.bank_name for p in CONSORTIUM_PROFILES)
    RATING_ORDER: Tuple[str, ...] = ("AAA", "AA", "A", "BBB", "BB", "B")

    def __init__(
        self,
        n_clients: int = 8,
        val_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        if n_clients <= 0:
            raise ValueError(f"n_clients must be positive, got {n_clients}")
        if not (0.0 < val_ratio < 0.5):
            raise ValueError(f"val_ratio must be in (0, 0.5), got {val_ratio}")

        self.n_clients = n_clients
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Cache profiles
        self.profiles: List[InstitutionProfile] = []
        for i in range(n_clients):
            if i < len(CONSORTIUM_PROFILES):
                self.profiles.append(CONSORTIUM_PROFILES[i])
            else:
                self.profiles.append(
                    InstitutionProfile(
                        client_id=i,
                        bank_name=f"Bank_{i}_ConsortiumMember",
                        institution_type="Commercial Bank",
                        jurisdiction="Global",
                        esg_mandate="Article 8 ESG Fund",
                        risk_profile="Balanced",
                    )
                )

    # -----------------------------------------------------------------------
    # 1. Stratified IID Partitioning
    # -----------------------------------------------------------------------

    def partition_iid(
        self,
        df_train: pd.DataFrame,
        X_scaled: np.ndarray,
        y: np.ndarray,
    ) -> List[ClientDataset]:
        """Partition training records uniformly across K banks (IID).

        Performs stratified allocation across credit rating categories to
        ensure each bank receives an identical portfolio rating distribution.

        Parameters:
        -----------
        df_train : pd.DataFrame
            Original unscaled training DataFrame.
        X_scaled : np.ndarray
            Standardized feature matrix (samples, features).
        y : np.ndarray
            Yield spread target vector (samples,).

        Returns:
        --------
        List[ClientDataset]
            K ClientDataset objects with identical sample sizes and rating mixes.
        """
        n_samples = len(df_train)
        logger.info(
            "Executing Stratified IID Partitioning: %d samples across %d banks...",
            n_samples,
            self.n_clients,
        )

        client_indices: List[List[int]] = [[] for _ in range(self.n_clients)]

        # Group by credit rating for stratified assignment
        rating_col = "issuer_credit_rating"
        if rating_col in df_train.columns:
            for rating in self.RATING_ORDER:
                rating_idx = df_train[df_train[rating_col] == rating].index.to_numpy()
                shuffled_idx = self.rng.permutation(rating_idx)
                # Split evenly among clients
                split_chunks = np.array_split(shuffled_idx, self.n_clients)
                for client_id, chunk in enumerate(split_chunks):
                    client_indices[client_id].extend(chunk.tolist())
        else:
            # Fallback to random permutation if rating column is absent
            all_idx = self.rng.permutation(n_samples)
            split_chunks = np.array_split(all_idx, self.n_clients)
            for client_id, chunk in enumerate(split_chunks):
                client_indices[client_id].extend(chunk.tolist())

        # Construct ClientDataset instances
        datasets: List[ClientDataset] = []
        for client_id in range(self.n_clients):
            assigned_idx = np.array(client_indices[client_id])
            self.rng.shuffle(assigned_idx)

            client_raw = df_train.iloc[assigned_idx].copy().reset_index(drop=True)
            client_X = X_scaled[assigned_idx]
            client_y = y[assigned_idx]

            # Split local train / local validation
            n_val = int(len(assigned_idx) * self.val_ratio)
            n_train = len(assigned_idx) - n_val

            train_X, val_X = client_X[:n_train], client_X[n_train:]
            train_y, val_y = client_y[:n_train], client_y[n_train:]

            # Rating distribution
            rating_dist: Dict[str, int] = {}
            if rating_col in client_raw.columns:
                rating_dist = client_raw[rating_col].value_counts().to_dict()

            profile = self.profiles[client_id]
            ds = ClientDataset(
                client_id=client_id,
                bank_name=profile.bank_name,
                X_train=train_X,
                y_train=train_y,
                X_val=val_X,
                y_val=val_y,
                raw_df=client_raw,
                rating_distribution=rating_dist,
                profile=profile,
            )
            datasets.append(ds)
            logger.debug(
                "Initialized Client %d [%s]: %d train samples, %d val samples",
                client_id,
                ds.bank_name,
                ds.n_train_samples,
                ds.n_val_samples,
            )

        return datasets

    # -----------------------------------------------------------------------
    # 2. Non-IID Dirichlet Partitioning
    # -----------------------------------------------------------------------

    def partition_non_iid_dirichlet(
        self,
        df_train: pd.DataFrame,
        X_scaled: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.5,
    ) -> List[ClientDataset]:
        r"""Partition training records using a Dirichlet distribution (Non-IID).

        Each credit rating category $r$ is split across the $K$ banks according to
        proportions drawn from $\text{Dirichlet}(\alpha \cdot \mathbf{1}_K)$. Lower $\alpha$
        produces severe portfolio skewness (some banks holding mostly AAA, others BB).

        Parameters:
        -----------
        df_train : pd.DataFrame
            Original unscaled training DataFrame.
        X_scaled : np.ndarray
            Standardized feature matrix.
        y : np.ndarray
            Yield spread targets.
        alpha : float
            Dirichlet concentration parameter (default: 0.5; lower = more skewed).

        Returns:
        --------
        List[ClientDataset]
            K ClientDataset objects with heterogeneous sample sizes and rating mixes.
        """
        n_samples = len(df_train)
        logger.info(
            "Executing Non-IID Dirichlet Partitioning (alpha=%.2f): %d samples across %d banks...",
            alpha,
            n_samples,
            self.n_clients,
        )

        client_indices: List[List[int]] = [[] for _ in range(self.n_clients)]
        rating_col = "issuer_credit_rating"

        if rating_col not in df_train.columns:
            logger.warning(
                "Column '%s' not found. Falling back to random Dirichlet splits.",
                rating_col,
            )
            props = self.rng.dirichlet(np.repeat(alpha, self.n_clients))
            split_points = (np.cumsum(props) * n_samples).astype(int)[:-1]
            shuffled = self.rng.permutation(n_samples)
            chunks = np.split(shuffled, split_points)
            for cid, chunk in enumerate(chunks):
                client_indices[cid].extend(chunk.tolist())
        else:
            # Dirichlet split per rating category
            for rating in self.RATING_ORDER:
                rating_idx = df_train[df_train[rating_col] == rating].index.to_numpy()
                shuffled_idx = self.rng.permutation(rating_idx)
                n_rating_samples = len(shuffled_idx)

                # Draw proportions from Dirichlet(alpha)
                proportions = self.rng.dirichlet(np.repeat(alpha, self.n_clients))
                # Ensure minimum 1 sample per bank if samples permit
                counts = (proportions * n_rating_samples).astype(int)
                remainder = n_rating_samples - counts.sum()
                for i in range(remainder):
                    counts[i % self.n_clients] += 1

                current_start = 0
                for cid in range(self.n_clients):
                    c_count = counts[cid]
                    if c_count > 0:
                        client_indices[cid].extend(
                            shuffled_idx[current_start : current_start + c_count].tolist()
                        )
                        current_start += c_count

        # Build ClientDataset objects
        datasets: List[ClientDataset] = []
        for client_id in range(self.n_clients):
            assigned_idx = np.array(client_indices[client_id])
            self.rng.shuffle(assigned_idx)

            if len(assigned_idx) == 0:
                raise RuntimeError(
                    f"Client {client_id} received 0 samples under Dirichlet partition! "
                    f"Increase alpha or total dataset size."
                )

            client_raw = df_train.iloc[assigned_idx].copy().reset_index(drop=True)
            client_X = X_scaled[assigned_idx]
            client_y = y[assigned_idx]

            n_val = max(1, int(len(assigned_idx) * self.val_ratio))
            n_train = len(assigned_idx) - n_val

            train_X, val_X = client_X[:n_train], client_X[n_train:]
            train_y, val_y = client_y[:n_train], client_y[n_train:]

            rating_dist: Dict[str, int] = {}
            if rating_col in client_raw.columns:
                rating_dist = client_raw[rating_col].value_counts().to_dict()

            profile = self.profiles[client_id]
            ds = ClientDataset(
                client_id=client_id,
                bank_name=profile.bank_name,
                X_train=train_X,
                y_train=train_y,
                X_val=val_X,
                y_val=val_y,
                raw_df=client_raw,
                rating_distribution=rating_dist,
                profile=profile,
            )
            datasets.append(ds)

        return datasets

    # -----------------------------------------------------------------------
    # 3. Non-IID Rating-Tier Partitioning (Market Specialization)
    # -----------------------------------------------------------------------

    def partition_non_iid_rating_tier(
        self,
        df_train: pd.DataFrame,
        X_scaled: np.ndarray,
        y: np.ndarray,
        ig_bias_ratio: float = 0.85,
    ) -> List[ClientDataset]:
        """Partition by institutional credit tier specialization (Non-IID).

        Simulates market segmentation:
        - Banks 0-3 (Tier 1 Institutions) specialize in Investment Grade (AAA, AA, A, BBB).
        - Banks 4-7 (High-Yield / Regional) specialize in Speculative Grade (BB, B).

        Parameters:
        -----------
        df_train : pd.DataFrame
            Original unscaled training DataFrame.
        X_scaled : np.ndarray
            Standardized feature matrix.
        y : np.ndarray
            Yield spread target array.
        ig_bias_ratio : float
            Fraction of Investment Grade bonds allocated to Banks 0-3 (default: 0.85).

        Returns:
        --------
        List[ClientDataset]
            K ClientDataset objects reflecting institutional market mandates.
        """
        logger.info(
            "Executing Rating-Tier Non-IID Partitioning (IG bias=%.2f)...",
            ig_bias_ratio,
        )
        rating_col = "issuer_credit_rating"
        ig_ratings = {"AAA", "AA", "A", "BBB"}
        hy_ratings = {"BB", "B"}

        ig_indices = df_train[df_train[rating_col].isin(ig_ratings)].index.to_numpy()
        hy_indices = df_train[df_train[rating_col].isin(hy_ratings)].index.to_numpy()

        shuffled_ig = self.rng.permutation(ig_indices)
        shuffled_hy = self.rng.permutation(hy_indices)

        n_ig = len(shuffled_ig)
        n_hy = len(shuffled_hy)

        # Allocate 85% of IG to Banks 0-3, 15% to Banks 4-7
        ig_split_pt = int(n_ig * ig_bias_ratio)
        ig_for_group1 = np.array_split(shuffled_ig[:ig_split_pt], 4)
        ig_for_group2 = np.array_split(shuffled_ig[ig_split_pt:], 4)

        # Allocate 85% of HY to Banks 4-7, 15% to Banks 0-3
        hy_split_pt = int(n_hy * (1.0 - ig_bias_ratio))
        hy_for_group1 = np.array_split(shuffled_hy[:hy_split_pt], 4)
        hy_for_group2 = np.array_split(shuffled_hy[hy_split_pt:], 4)

        client_indices: List[List[int]] = [[] for _ in range(self.n_clients)]
        for i in range(4):
            client_indices[i].extend(ig_for_group1[i].tolist())
            client_indices[i].extend(hy_for_group1[i].tolist())
        for i in range(4):
            client_indices[4 + i].extend(ig_for_group2[i].tolist())
            client_indices[4 + i].extend(hy_for_group2[i].tolist())

        datasets: List[ClientDataset] = []
        for client_id in range(self.n_clients):
            assigned_idx = np.array(client_indices[client_id])
            self.rng.shuffle(assigned_idx)

            client_raw = df_train.iloc[assigned_idx].copy().reset_index(drop=True)
            client_X = X_scaled[assigned_idx]
            client_y = y[assigned_idx]

            n_val = max(1, int(len(assigned_idx) * self.val_ratio))
            n_train = len(assigned_idx) - n_val

            train_X, val_X = client_X[:n_train], client_X[n_train:]
            train_y, val_y = client_y[:n_train], client_y[n_train:]

            rating_dist = client_raw[rating_col].value_counts().to_dict()
            profile = self.profiles[client_id]

            ds = ClientDataset(
                client_id=client_id,
                bank_name=profile.bank_name,
                X_train=train_X,
                y_train=train_y,
                X_val=val_X,
                y_val=val_y,
                raw_df=client_raw,
                rating_distribution=rating_dist,
                profile=profile,
            )
            datasets.append(ds)

        return datasets

    # -----------------------------------------------------------------------
    # 4. Statistical Heterogeneity Diagnostics
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_jensen_shannon_divergence(
        dist_p: Sequence[float], dist_q: Sequence[float]
    ) -> float:
        r"""Compute the Jensen-Shannon divergence between two categorical distributions.

        $$JSD(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$$
        where $M = \frac{1}{2}(P + Q)$. JSD is symmetric and bounded in $[0, 1]$ (base 2).

        Parameters:
        -----------
        dist_p : Sequence[float]
            First probability vector.
        dist_q : Sequence[float]
            Second probability vector.

        Returns:
        --------
        float
            JSD value in $[0, 1]$.
        """
        p = np.asarray(dist_p, dtype=np.float64)
        q = np.asarray(dist_q, dtype=np.float64)

        # Normalize to probability simplex
        p = p / np.sum(p)
        q = q / np.sum(q)

        m = 0.5 * (p + q)

        def _kl(a: np.ndarray, b: np.ndarray) -> float:
            mask = (a > 0) & (b > 0)
            return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

        jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        return float(np.clip(jsd, 0.0, 1.0))

    def evaluate_heterogeneity(
        self, client_datasets: List[ClientDataset]
    ) -> Dict[str, Any]:
        """Compute comprehensive statistical heterogeneity metrics across banks.

        Returns:
        --------
        Dict[str, Any]
            Pairwise JSD matrix, mean pairwise JSD, sample size variance,
            and ratings entropy.
        """
        n_clients = len(client_datasets)
        ratings = list(self.RATING_ORDER)

        # Construct rating probability distribution for each bank
        prob_matrix = np.zeros((n_clients, len(ratings)), dtype=np.float64)
        for i, ds in enumerate(client_datasets):
            total = max(1, ds.total_samples)
            for j, r in enumerate(ratings):
                prob_matrix[i, j] = ds.rating_distribution.get(r, 0) / total

        # Compute pairwise JSD
        jsd_matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                jsd = self.compute_jensen_shannon_divergence(
                    prob_matrix[i], prob_matrix[j]
                )
                jsd_matrix[i, j] = jsd
                jsd_matrix[j, i] = jsd

        # Upper triangle mean (excluding self-comparisons)
        upper_indices = np.triu_indices(n_clients, k=1)
        mean_jsd = float(np.mean(jsd_matrix[upper_indices]))

        sample_sizes = [ds.n_train_samples for ds in client_datasets]
        sample_std = float(np.std(sample_sizes))
        sample_cv = sample_std / float(np.mean(sample_sizes)) if np.mean(sample_sizes) > 0 else 0.0

        return {
            "mean_pairwise_jsd": round(mean_jsd, 4),
            "sample_size_std": round(sample_std, 2),
            "sample_size_cv": round(sample_cv, 4),
            "pairwise_jsd_matrix": jsd_matrix,
            "rating_probability_matrix": prob_matrix,
        }

    # -----------------------------------------------------------------------
    # 5. Diagnostic Reporting & Export
    # -----------------------------------------------------------------------

    def print_partition_summary(
        self, client_datasets: List[ClientDataset]
    ) -> pd.DataFrame:
        """Print and return a structured diagnostic summary table.

        Parameters:
        -----------
        client_datasets : List[ClientDataset]
            Partitioned datasets across all consortium members.

        Returns:
        --------
        pd.DataFrame
            Summary DataFrame with sample counts and rating distributions.
        """
        rows: List[Dict[str, Any]] = []
        for ds in client_datasets:
            row: Dict[str, Any] = {
                "Bank ID": ds.client_id,
                "Bank Name": ds.bank_name,
                "Train Samples": ds.n_train_samples,
                "Val Samples": ds.n_val_samples,
                "Total": ds.total_samples,
            }
            for rating in self.RATING_ORDER:
                row[rating] = ds.rating_distribution.get(rating, 0)
            rows.append(row)

        df_summary = pd.DataFrame(rows)

        print("\n" + "=" * 95)
        print("          GREENFED-XAI CONSORTIUM PORTFOLIO PARTITION SUMMARY")
        print("=" * 95)
        print(df_summary.to_string(index=False))
        print("=" * 95 + "\n")

        # Heterogeneity diagnostics
        metrics = self.evaluate_heterogeneity(client_datasets)
        logger.info(
            "Portfolio Heterogeneity: Mean Pairwise JSD = %.4f | Sample Size CV = %.4f",
            metrics["mean_pairwise_jsd"],
            metrics["sample_size_cv"],
        )

        return df_summary

    def export_partition_diagnostics_csv(
        self, client_datasets: List[ClientDataset], output_path: Union[str, Path]
    ) -> Path:
        """Export portfolio diagnostic summary to CSV file.

        Parameters:
        -----------
        client_datasets : List[ClientDataset]
            Partitioned bank datasets.
        output_path : Union[str, Path]
            Destination CSV filepath.

        Returns:
        --------
        Path
            Path to the written CSV file.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for ds in client_datasets:
            stats = ds.compute_summary_statistics()
            row = {
                "client_id": stats["client_id"],
                "bank_name": stats["bank_name"],
                "n_train_samples": stats["n_train"],
                "n_val_samples": stats["n_val"],
                "mean_yield_spread_pct": stats["mean_yield_spread_pct"],
                "std_yield_spread_pct": stats["std_yield_spread_pct"],
                "min_yield_spread_pct": stats["min_yield_spread_pct"],
                "max_yield_spread_pct": stats["max_yield_spread_pct"],
            }
            if ds.profile is not None:
                row["institution_type"] = ds.profile.institution_type
                row["jurisdiction"] = ds.profile.jurisdiction
                row["risk_profile"] = ds.profile.risk_profile

            for r in self.RATING_ORDER:
                row[f"count_{r}"] = ds.rating_distribution.get(r, 0)

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(out_p, index=False)
        logger.info("Exported bank partition diagnostics to %s", out_p)
        return out_p
