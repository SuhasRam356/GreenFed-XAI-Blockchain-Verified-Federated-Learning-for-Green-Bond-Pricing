"""
Data Preprocessing Utilities — GreenFed-XAI
=============================================

Shared preprocessing pipeline used by all models:
  - Ordinal encoding of credit ratings
  - Standard scaling of numerical features
  - Splitting features (X) from target (y)

Re-used in Phases 2, 3, 4, etc.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


# Credit ratings ordered from best to worst (ordinal)
RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B"]

# Feature columns used by all models
FEATURE_COLS = [
    "issuer_credit_rating",
    "tenor",
    "coupon_rate",
    "esg_controversy_score",
    "carbon_footprint_score",
]
TARGET_COL = "yield_spread"


class DataPreprocessor:
    """Preprocesses green bond data for model training.

    Steps:
      1. Ordinal-encode ``issuer_credit_rating`` (AAA=0, AA=1, …, B=5).
      2. StandardScale all numerical features (zero mean, unit variance).

    The preprocessor must be ``fit()`` on training data only, then
    ``transform()`` applied to val/test to avoid data leakage.

    Parameters
    ----------
    feature_cols : list[str]
        Columns to use as features.
    target_col : str
        Column to predict.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        target_col: str = TARGET_COL,
    ):
        self.feature_cols = feature_cols or FEATURE_COLS
        self.target_col = target_col

        self.rating_encoder = OrdinalEncoder(
            categories=[RATING_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit encoder and scaler on training data."""
        X_raw = self._extract_features(df)
        X_encoded = self._encode_ratings(X_raw, fit=True)
        self.scaler.fit(X_encoded)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform a DataFrame into (X, y) numpy arrays.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
            Scaled feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target values (yield spread).
        """
        assert self._is_fitted, "Call fit() before transform()."
        X_raw = self._extract_features(df)
        X_encoded = self._encode_ratings(X_raw, fit=False)
        X_scaled = self.scaler.transform(X_encoded)
        y = df[self.target_col].values.astype(np.float32)
        return X_scaled.astype(np.float32), y

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit on *df* and return transformed (X, y)."""
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self) -> list[str]:
        """Return feature names after encoding (rating becomes numeric)."""
        names = []
        for col in self.feature_cols:
            if col == "issuer_credit_rating":
                names.append("credit_rating_encoded")
            else:
                names.append(col)
        return names

    # --- Internal helpers ---------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature columns from DataFrame."""
        missing = set(self.feature_cols) - set(df.columns)
        assert len(missing) == 0, f"Missing columns: {missing}"
        return df[self.feature_cols].copy()

    def _encode_ratings(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        """Ordinal-encode the credit rating column."""
        X_copy = X.copy()
        rating_col = X_copy[["issuer_credit_rating"]].values

        if fit:
            encoded = self.rating_encoder.fit_transform(rating_col)
        else:
            encoded = self.rating_encoder.transform(rating_col)

        X_copy["issuer_credit_rating"] = encoded.flatten()
        return X_copy.values.astype(np.float64)

