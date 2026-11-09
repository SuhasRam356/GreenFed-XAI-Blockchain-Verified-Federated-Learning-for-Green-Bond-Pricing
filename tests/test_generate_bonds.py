"""
Tests for Phase 1a — Bond Dataset Generator
=============================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_bonds import (
    generate_bonds,
    split_dataset,
    N_BONDS,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    RATING_TIERS,
    RATING_BASE_SPREADS,
    COEFF_TENOR,
    COEFF_CONTROVERSY,
    COEFF_CARBON,
)


class TestGenerateBonds:
    """Tests for the bond generation function."""

    @pytest.fixture
    def bonds(self):
        """Generate a fresh bond dataset for testing."""
        return generate_bonds()

    def test_correct_number_of_bonds(self, bonds):
        """Must generate exactly 10,000 bonds."""
        assert len(bonds) == N_BONDS, f"Expected {N_BONDS}, got {len(bonds)}"

    def test_columns_present(self, bonds):
        """All required columns must be present."""
        expected_cols = {
            "bond_id",
            "issuer_credit_rating",
            "tenor",
            "coupon_rate",
            "esg_controversy_score",
            "carbon_footprint_score",
            "yield_spread",
        }
        assert expected_cols == set(bonds.columns), (
            f"Missing columns: {expected_cols - set(bonds.columns)}"
        )

    def test_bond_ids_unique(self, bonds):
        """Bond IDs must be unique and sequential 1..N."""
        assert bonds["bond_id"].is_unique
        assert bonds["bond_id"].min() == 1
        assert bonds["bond_id"].max() == N_BONDS

    def test_credit_ratings_valid(self, bonds):
        """All ratings must be from the defined set."""
        invalid = set(bonds["issuer_credit_rating"].unique()) - set(RATING_TIERS)
        assert len(invalid) == 0, f"Invalid ratings found: {invalid}"

    def test_all_ratings_represented(self, bonds):
        """All 6 rating tiers should appear in a 10K dataset."""
        present = set(bonds["issuer_credit_rating"].unique())
        assert present == set(RATING_TIERS), f"Missing ratings: {set(RATING_TIERS) - present}"

    def test_tenor_range(self, bonds):
        """Tenor must be 1–30 years (integers)."""
        assert bonds["tenor"].min() >= 1
        assert bonds["tenor"].max() <= 30
        assert bonds["tenor"].dtype in [np.int32, np.int64]

    def test_coupon_rate_range(self, bonds):
        """Coupon rate must be in [1, 8] %."""
        assert bonds["coupon_rate"].min() >= 1.0
        assert bonds["coupon_rate"].max() <= 8.0

    def test_esg_controversy_range(self, bonds):
        """ESG controversy score must be in [0, 10]."""
        assert bonds["esg_controversy_score"].min() >= 0.0
        assert bonds["esg_controversy_score"].max() <= 10.0

    def test_carbon_score_range(self, bonds):
        """Carbon footprint score must be in [0, 100]."""
        assert bonds["carbon_footprint_score"].min() >= 0.0
        assert bonds["carbon_footprint_score"].max() <= 100.0

    def test_yield_spread_positive(self, bonds):
        """Yield spread should be positive (clipped at 0.01)."""
        assert bonds["yield_spread"].min() >= 0.01

    def test_no_null_values(self, bonds):
        """No NaN or null values allowed."""
        assert bonds.isnull().sum().sum() == 0

    def test_reproducibility(self):
        """Same seed should produce identical datasets."""
        df1 = generate_bonds(seed=42)
        df2 = generate_bonds(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_data(self):
        """Different seeds should produce different datasets."""
        df1 = generate_bonds(seed=42)
        df2 = generate_bonds(seed=99)
        assert not df1["yield_spread"].equals(df2["yield_spread"])

    def test_yield_spread_formula_direction(self, bonds):
        """Higher credit quality should generally give lower spread."""
        mean_spreads = bonds.groupby("issuer_credit_rating")["yield_spread"].mean()
        # AAA should have lowest, B should have highest
        assert mean_spreads["AAA"] < mean_spreads["B"], (
            f"AAA spread ({mean_spreads['AAA']:.4f}) should be < "
            f"B spread ({mean_spreads['B']:.4f})"
        )

    def test_carbon_score_negative_effect_on_spread(self, bonds):
        """Higher carbon score should correlate with lower spread."""
        correlation = bonds["carbon_footprint_score"].corr(bonds["yield_spread"])
        assert correlation < 0, (
            f"Carbon score should negatively correlate with spread, "
            f"got r={correlation:.4f}"
        )

    def test_tenor_positive_effect_on_spread(self, bonds):
        """Higher tenor should correlate with higher spread."""
        correlation = bonds["tenor"].corr(bonds["yield_spread"])
        assert correlation > 0, (
            f"Tenor should positively correlate with spread, "
            f"got r={correlation:.4f}"
        )

    def test_controversy_positive_effect_on_spread(self, bonds):
        """Higher ESG controversy should correlate with higher spread."""
        correlation = bonds["esg_controversy_score"].corr(bonds["yield_spread"])
        assert correlation > 0, (
            f"Controversy should positively correlate with spread, "
            f"got r={correlation:.4f}"
        )


class TestSplitDataset:
    """Tests for the train/val/test split function."""

    @pytest.fixture
    def splits(self):
        df = generate_bonds()
        return split_dataset(df)

    def test_split_sizes(self, splits):
        """Train/val/test must have correct sizes."""
        train, val, test = splits
        assert len(train) == TRAIN_SIZE, f"Train: expected {TRAIN_SIZE}, got {len(train)}"
        assert len(val) == VAL_SIZE, f"Val: expected {VAL_SIZE}, got {len(val)}"
        assert len(test) == TEST_SIZE, f"Test: expected {TEST_SIZE}, got {len(test)}"

    def test_no_overlap(self, splits):
        """No bond should appear in more than one split."""
        train, val, test = splits
        train_ids = set(train["bond_id"])
        val_ids = set(val["bond_id"])
        test_ids = set(test["bond_id"])

        assert len(train_ids & val_ids) == 0, "Overlap between train and val"
        assert len(train_ids & test_ids) == 0, "Overlap between train and test"
        assert len(val_ids & test_ids) == 0, "Overlap between val and test"

    def test_all_bonds_covered(self, splits):
        """Union of all splits should equal full dataset."""
        train, val, test = splits
        all_ids = set(train["bond_id"]) | set(val["bond_id"]) | set(test["bond_id"])
        assert len(all_ids) == N_BONDS

    def test_split_reproducibility(self):
        """Same seed produces same splits."""
        df = generate_bonds()
        t1, v1, te1 = split_dataset(df, seed=42)
        t2, v2, te2 = split_dataset(df, seed=42)
        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(v1, v2)
        pd.testing.assert_frame_equal(te1, te2)

    def test_columns_preserved(self, splits):
        """All columns should be preserved in each split."""
        train, val, test = splits
        expected_cols = {
            "bond_id", "issuer_credit_rating", "tenor", "coupon_rate",
            "esg_controversy_score", "carbon_footprint_score", "yield_spread",
        }
        for name, split in [("train", train), ("val", val), ("test", test)]:
            assert set(split.columns) == expected_cols, f"{name} missing columns"

