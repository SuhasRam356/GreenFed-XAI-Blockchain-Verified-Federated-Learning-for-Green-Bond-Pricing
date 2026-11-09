"""
Phase 1a — Synthetic Green Bond Dataset Generator
===================================================

Generates a synthetic dataset of 10,000 green bonds as described in the
IEEE paper "GreenFed-XAI".  Each bond record contains:

  - issuer_credit_rating : categorical (AAA, AA, A, BBB, BB, B)
  - tenor               : int, maturity in years (1–30)
  - coupon_rate          : float, annual coupon (1–8 %)
  - esg_controversy_score: float, 0–10
  - carbon_footprint_score (Ct): float, 0–100  (IoT-derived)
  - yield_spread         : float, target variable (%)

Formula (from the implementation guide):
    spread = base(rating) + 0.03 × tenor + 0.12 × controversy
             − 0.012 × Ct + noise

Split: 7 000 train / 1 500 validation / 1 500 test.

Usage:
    python data/generate_bonds.py              # writes CSVs to outputs/
    python data/generate_bonds.py --out-dir .   # custom output directory
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_BONDS = 10_000

# Credit-rating tiers and their base yield spreads (in %)
RATING_TIERS = ["AAA", "AA", "A", "BBB", "BB", "B"]
RATING_BASE_SPREADS = {
    "AAA": 1.50,
    "AA":  1.80,
    "A":   2.15,
    "BBB": 2.55,
    "BB":  3.10,
    "B":   3.80,
}

# Rating probabilities (higher-quality issuers more common for green bonds)
RATING_PROBS = [0.10, 0.20, 0.25, 0.25, 0.15, 0.05]

# Feature ranges
TENOR_RANGE = (1, 30)        # years
COUPON_RANGE = (1.0, 8.0)    # %
ESG_CONTROVERSY_RANGE = (0.0, 10.0)
CARBON_SCORE_RANGE = (0.0, 100.0)

# Coefficients for the yield-spread formula
COEFF_TENOR = 0.03
COEFF_CONTROVERSY = 0.12
COEFF_CARBON = -0.012
NOISE_STD = 0.08   # Gaussian noise standard deviation

# Train / val / test split sizes
TRAIN_SIZE = 7_000
VAL_SIZE = 1_500
TEST_SIZE = 1_500


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_bonds(n: int = N_BONDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate *n* synthetic green-bond records.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bond_id, issuer_credit_rating, tenor,
        coupon_rate, esg_controversy_score, carbon_footprint_score,
        yield_spread.
    """
    rng = np.random.default_rng(seed)

    # --- Sample features ---------------------------------------------------
    ratings = rng.choice(RATING_TIERS, size=n, p=RATING_PROBS)
    tenors = rng.integers(TENOR_RANGE[0], TENOR_RANGE[1] + 1, size=n)
    coupons = rng.uniform(COUPON_RANGE[0], COUPON_RANGE[1], size=n).round(2)
    esg_controversy = rng.uniform(
        ESG_CONTROVERSY_RANGE[0], ESG_CONTROVERSY_RANGE[1], size=n
    ).round(2)
    # Carbon footprint score with 5 % sensor-error noise (as per the paper)
    carbon_base = rng.uniform(
        CARBON_SCORE_RANGE[0], CARBON_SCORE_RANGE[1], size=n
    )
    sensor_error = rng.normal(0, 0.05 * carbon_base)  # 5 % of true value
    carbon_scores = np.clip(carbon_base + sensor_error, 0, 100).round(2)

    # --- Compute target: yield spread (%) ----------------------------------
    base_spreads = np.array([RATING_BASE_SPREADS[r] for r in ratings])
    noise = rng.normal(0, NOISE_STD, size=n)

    yield_spread = (
        base_spreads
        + COEFF_TENOR * tenors
        + COEFF_CONTROVERSY * esg_controversy
        + COEFF_CARBON * carbon_scores
        + noise
    )
    # Spreads should be non-negative
    yield_spread = np.clip(yield_spread, 0.01, None).round(4)

    df = pd.DataFrame({
        "bond_id": np.arange(1, n + 1),
        "issuer_credit_rating": ratings,
        "tenor": tenors,
        "coupon_rate": coupons,
        "esg_controversy_score": esg_controversy,
        "carbon_footprint_score": carbon_scores,
        "yield_spread": yield_spread,
    })
    return df


def split_dataset(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """Shuffle and split into train / val / test sets.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    assert len(df) == TRAIN_SIZE + VAL_SIZE + TEST_SIZE, (
        f"Expected {TRAIN_SIZE + VAL_SIZE + TEST_SIZE} rows, got {len(df)}"
    )
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train = df_shuffled.iloc[:TRAIN_SIZE].copy()
    val = df_shuffled.iloc[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE].copy()
    test = df_shuffled.iloc[TRAIN_SIZE + VAL_SIZE:].copy()

    return train, val, test


def print_summary(df: pd.DataFrame, label: str = "Full dataset"):
    """Print basic statistics about *df*."""
    print(f"\n{'='*60}")
    print(f"  {label}  —  {len(df)} records")
    print(f"{'='*60}")
    print(df.describe(include="all").to_string())
    print()
    # Rating distribution
    print("Rating distribution:")
    print(df["issuer_credit_rating"].value_counts().sort_index().to_string())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic green bond dataset")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write CSVs into (default: <project>/outputs/)",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed"
    )
    args = parser.parse_args()

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic green bond dataset …")
    df = generate_bonds(seed=args.seed)
    print_summary(df)

    train, val, test = split_dataset(df, seed=args.seed)

    # Save full dataset + splits
    full_path = out_dir / "bonds.csv"
    train_path = out_dir / "bonds_train.csv"
    val_path = out_dir / "bonds_val.csv"
    test_path = out_dir / "bonds_test.csv"

    df.to_csv(full_path, index=False)
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Saved full dataset   → {full_path}  ({len(df)} rows)")
    print(f"Saved training set   → {train_path}  ({len(train)} rows)")
    print(f"Saved validation set → {val_path}  ({len(val)} rows)")
    print(f"Saved test set       → {test_path}  ({len(test)} rows)")


if __name__ == "__main__":
    main()

