r"""
Explainable AI (XAI) Layer with SHAP — GreenFed-XAI
===================================================

This module implements the Explainable AI (XAI) analysis pipeline using
Shapley Additive Explanations (SHAP) for the GreenFed-XAI neural network.

Theoretical & Methodological Grounding:
---------------------------------------
1. Method & Output Scale:
   - Uses `shap.KernelExplainer` (model-agnostic exact local additive attribution)
     or `shap.GradientExplainer` on the PyTorch neural network.
   - Output scale: Yield spread in percentage points (%).
   - One basis point = 0.01% (i.e. 100 bps = 1.00%).

2. Background Dataset & Leakage Prevention:
   - Sampled strictly from the training dataset (N_bg = 100).
   - NEVER samples from validation or test datasets to avoid explanation tuning leakage.

3. Additivity (Efficiency Axiom):
   $$\sum_{j=1}^M \phi_j(x) + \phi_0 \approx f(x)$$
   where \phi_0 is the base value (expected model output over the background distribution)
   and \phi_j(x) is the attribution of feature j.

4. 10-Point Carbon Score Model Sensitivity Analysis:
   - Modifies C_t \leftarrow C_t + 10 for eligible test bonds (where C_t + 10 <= 100).
   - Recomputes yield spread predictions: \Delta s = f(x_{C_t+10}) - f(x).
   - Expressed in basis points: \Delta s_{bps} = 100 \times \Delta s.
   - Crucial Distinction: This evaluates model sensitivity (what the model learned
     from the synthetic relationship), NOT a causal macroeconomic market guarantee.

5. Scientific Limitations:
   - SHAP explains attributions under a specific background distribution.
   - Feature correlations (e.g. credit rating and coupon) can induce marginal sampling drift.
   - SHAP does not prove causation or greenwashing; it provides auditable evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from models.nn_model import GreenBondNN

logger = logging.getLogger(__name__)


class GreenBondXAIExplainer:
    """Production-grade SHAP explainer for green bond yield spread predictions.

    Parameters:
    -----------
    model : GreenBondNN
        Trained PyTorch neural network model.
    feature_names : List[str]
        Ordered names of model input features.
    background_data : np.ndarray
        Pre-processed background samples from the training set.
    device : Optional[torch.device]
        PyTorch computation device (CPU/CUDA).
    """

    def __init__(
        self,
        model: GreenBondNN,
        feature_names: List[str],
        background_data: np.ndarray,
        device: Optional[torch.device] = None,
        max_background_samples: int = 100,
        random_seed: int = 42,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.feature_names = list(feature_names)
        self.random_seed = random_seed

        # Subsample background data deterministically from training distribution
        if len(background_data) > max_background_samples:
            rng = np.random.default_rng(random_seed)
            indices = rng.choice(len(background_data), size=max_background_samples, replace=False)
            self.background = background_data[indices].astype(np.float32)
        else:
            self.background = background_data.astype(np.float32)

        # Background prediction function for KernelExplainer
        def model_predict_fn(x_arr: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tensor_in = torch.tensor(x_arr, dtype=torch.float32, device=self.device)
                preds = self.model(tensor_in).cpu().numpy().flatten()
            return preds

        self.predict_fn = model_predict_fn
        logger.info(
            "Initializing SHAP KernelExplainer with %d background samples (dimension: %d)...",
            len(self.background),
            self.background.shape[1],
        )
        self.explainer = shap.KernelExplainer(self.predict_fn, self.background)
        self.expected_value = float(np.mean(self.predict_fn(self.background)))

    def compute_shap_values(
        self,
        X_eval: np.ndarray,
        nsamples: int = 200,
    ) -> np.ndarray:
        """Compute SHAP attribution values for evaluation samples.

        Parameters:
        -----------
        X_eval : np.ndarray
            Standardized evaluation feature matrix.
        nsamples : int
            Number of Monte Carlo evaluations per sample for KernelExplainer.

        Returns:
        --------
        np.ndarray
            Matrix of SHAP attributions of shape (N_eval, M_features).
        """
        logger.info("Computing SHAP values for %d samples (nsamples=%d)...", len(X_eval), nsamples)
        shap_values = self.explainer.shap_values(X_eval, nsamples=nsamples)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return np.asarray(shap_values, dtype=np.float32)

    def verify_additivity(
        self,
        X_sample: np.ndarray,
        shap_values: np.ndarray,
        tolerance: float = 0.05,
    ) -> Dict[str, Any]:
        """Verify efficiency/additivity axiom: sum(phi_i) + phi_0 ~= f(x).

        Parameters:
        -----------
        X_sample : np.ndarray
            Input features of shape (N, M) or (M,).
        shap_values : np.ndarray
            SHAP attribution values of shape (N, M) or (M,).
        tolerance : float
            Maximum allowable discrepancy in percentage points.

        Returns:
        --------
        Dict[str, Any]
            Additivity check statistics including max absolute error and pass status.
        """
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
            shap_values = shap_values.reshape(1, -1)

        f_x = self.predict_fn(X_sample)
        sum_phi = np.sum(shap_values, axis=1) + self.expected_value
        discrepancies = np.abs(sum_phi - f_x)

        max_discrepancy = float(np.max(discrepancies))
        mean_discrepancy = float(np.mean(discrepancies))
        passed = max_discrepancy <= tolerance

        logger.info(
            "Additivity Check: Max Error = %.6f%% | Mean Error = %.6f%% | Passed (tol=%.4f): %s",
            max_discrepancy,
            mean_discrepancy,
            tolerance,
            passed,
        )

        return {
            "passed": passed,
            "max_discrepancy": max_discrepancy,
            "mean_discrepancy": mean_discrepancy,
            "expected_value": self.expected_value,
            "tolerance": tolerance,
        }

    def compute_feature_importance(
        self,
        shap_values: np.ndarray,
    ) -> pd.DataFrame:
        """Compute global feature importance ranking by mean absolute SHAP value.

        Parameters:
        -----------
        shap_values : np.ndarray
            SHAP values matrix of shape (N, M).

        Returns:
        --------
        pd.DataFrame
            Ranked features with mean absolute attribution and percentage share.
        """
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        total_importance = np.sum(mean_abs_shap) if np.sum(mean_abs_shap) > 0 else 1.0
        importance_pct = (mean_abs_shap / total_importance) * 100.0

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs_shap,
                "importance_pct": importance_pct,
            }
        ).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return df

    def evaluate_carbon_score_sensitivity(
        self,
        test_df: pd.DataFrame,
        X_test: np.ndarray,
        preprocessor: Any,
        score_increase: float = 10.0,
    ) -> Dict[str, Any]:
        r"""Estimate model response to a +10 point increase in carbon score (C_t).

        Strict Protocol:
        ----------------
        1. Identifies test bonds where original C_t + score_increase <= 100.0.
        2. Leaves all other features (credit rating, tenor, coupon, ESG controversy) identical.
        3. Transforms modified test set using original scaler (fitted strictly on training set).
        4. Recomputes neural network yield spread predictions:
           \Delta s = f(x_{modified}) - f(x_{original})
        5. Reports distribution in basis points (\Delta s_{bps} = 100 \times \Delta s).

        Parameters:
        -----------
        test_df : pd.DataFrame
            Original unscaled test set DataFrame containing 'carbon_score'.
        X_test : np.ndarray
            Standardized test features matching test_df.
        preprocessor : DataPreprocessor
            Preprocessing pipeline used for test feature transformation.
        score_increase : float
            Carbon score delta to evaluate (default: +10.0 points).

        Returns:
        --------
        Dict[str, Any]
            Sensitivity statistics: eligible counts, mean/median bps reduction, std, min, max.
        """
        logger.info(
            "Evaluating model sensitivity to +%.1f point carbon score increase...",
            score_increase,
        )

        carbon_col = (
            "carbon_footprint_score"
            if "carbon_footprint_score" in test_df.columns
            else ("carbon_score" if "carbon_score" in test_df.columns else None)
        )
        if carbon_col is None:
            raise KeyError(
                f"Neither 'carbon_footprint_score' nor 'carbon_score' found in test_df columns: {list(test_df.columns)}"
            )

        # 1. Filter eligible bonds where C_t + score_increase <= 100.0
        eligible_mask = (test_df[carbon_col] + score_increase) <= 100.0
        n_eligible = int(eligible_mask.sum())
        total_test = len(test_df)

        if n_eligible == 0:
            logger.warning("No eligible bonds with carbon_score + %.1f <= 100.0", score_increase)
            return {
                "n_eligible": 0,
                "total_bonds": total_test,
                "eligible_pct": 0.0,
                "mean_delta_bps": 0.0,
                "median_delta_bps": 0.0,
                "std_delta_bps": 0.0,
            }

        # 2. Extract baseline predictions on eligible bonds
        X_eligible = X_test[eligible_mask]
        orig_preds_pct = self.predict_fn(X_eligible)

        # 3. Create modified DataFrame with C_t + score_increase
        modified_df = test_df[eligible_mask].copy()
        modified_df[carbon_col] = modified_df[carbon_col] + score_increase

        # 4. Standardize using training preprocessor
        trans = preprocessor.transform(modified_df)
        X_modified = trans[0] if isinstance(trans, tuple) else trans
        mod_preds_pct = self.predict_fn(X_modified)

        # 5. Calculate delta: negative delta indicates yield spread reduction (greenium)
        delta_pct = mod_preds_pct - orig_preds_pct
        delta_bps = delta_pct * 100.0  # Convert percentage points to basis points

        mean_bps = float(np.mean(delta_bps))
        median_bps = float(np.median(delta_bps))
        std_bps = float(np.std(delta_bps))
        min_bps = float(np.min(delta_bps))
        max_bps = float(np.max(delta_bps))

        logger.info(
            "Carbon Score Sensitivity (+%.1f points) | Eligible: %d/%d (%.1f%%) | "
            "Mean Spread Delta: %.2f bps | Median: %.2f bps | Std: %.2f bps",
            score_increase,
            n_eligible,
            total_test,
            (n_eligible / total_test) * 100.0,
            mean_bps,
            median_bps,
            std_bps,
        )

        return {
            "score_increase": score_increase,
            "n_eligible": n_eligible,
            "total_bonds": total_test,
            "eligible_fraction_pct": round((n_eligible / total_test) * 100.0, 2),
            "mean_delta_bps": round(mean_bps, 2),
            "median_delta_bps": round(median_bps, 2),
            "std_delta_bps": round(std_bps, 2),
            "min_delta_bps": round(min_bps, 2),
            "max_delta_bps": round(max_bps, 2),
            "delta_bps_samples": delta_bps.tolist()[:100],  # sample for inspection
            "synthetic_target_effect_bps": -12.0,  # synthetic reference
            "interpretation_note": (
                f"A {score_increase}-point increase in carbon score produces a mean yield spread "
                f"reduction of {abs(mean_bps):.2f} bps. This reflects model sensitivity under "
                f"the synthetic generator (-12 bps per 10 points), not a causal macroeconomic guarantee."
            ),
        }

    def generate_xai_plots(
        self,
        shap_values: np.ndarray,
        X_eval: np.ndarray,
        eval_df: pd.DataFrame,
        output_dir: Path,
        num_individual_samples: int = 4,
    ) -> List[Path]:
        """Generate global feature importance and individual sample explanation figures.

        Parameters:
        -----------
        shap_values : np.ndarray
            SHAP attributions matrix (N, M).
        X_eval : np.ndarray
            Standardized evaluation features (N, M).
        eval_df : pd.DataFrame
            Original unscaled evaluation metadata (for readable labels).
        output_dir : Path
            Output directory for saved plot images.
        num_individual_samples : int
            Number of individual bond explanations to plot (default: 4).

        Returns:
        --------
        List[Path]
            Paths of saved visualization plots.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_plots: List[Path] = []

        # ---------------------------------------------------------------------
        # 1. Global Feature Importance Bar Plot
        # ---------------------------------------------------------------------
        imp_df = self.compute_feature_importance(shap_values)
        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(imp_df))
        ax.barh(y_pos, imp_df["mean_abs_shap"], color="#1f77b4", edgecolor="black", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(imp_df["feature"], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Absolute Attribution |SHAP value| (Spread %)", fontsize=11, fontweight="bold")
        ax.set_title("Global Feature Importance (GreenFed-XAI Neural Network)", fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        # Annotate values and percentages
        for i, (val, pct) in enumerate(zip(imp_df["mean_abs_shap"], imp_df["importance_pct"])):
            ax.text(val + 0.002, i, f"{val:.4f} ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        global_plot_path = output_dir / "shap_feature_importance.png"
        plt.savefig(global_plot_path, dpi=300)
        plt.close()
        saved_plots.append(global_plot_path)
        logger.info("Saved global SHAP feature importance plot to %s", global_plot_path)

        # ---------------------------------------------------------------------
        # 2. Individual Explanation Plots (Waterfall Style)
        # ---------------------------------------------------------------------
        n_plots = min(num_individual_samples, len(X_eval))
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i in range(n_plots):
            ax = axes[i]
            bond_meta = eval_df.iloc[i] if i < len(eval_df) else {}
            bond_id = bond_meta.get("bond_id", f"Bond_{i:04d}")
            rating = bond_meta.get("credit_rating", bond_meta.get("issuer_credit_rating", "N/A"))
            c_score = float(bond_meta.get("carbon_score", bond_meta.get("carbon_footprint_score", 0.0)))

            sample_phi = shap_values[i]
            pred_val = float(self.predict_fn(X_eval[i : i + 1])[0])

            # Top positive and negative contributors
            order = np.argsort(np.abs(sample_phi))[::-1]
            top_features = [self.feature_names[j] for j in order[:8]]
            top_vals = [sample_phi[j] for j in order[:8]]

            bar_colors = ["#d9534f" if v > 0 else "#5cb85c" for v in top_vals]
            y_p = np.arange(len(top_features))

            ax.barh(y_p, top_vals, color=bar_colors, edgecolor="black", alpha=0.85)
            ax.set_yticks(y_p)
            ax.set_yticklabels(top_features, fontsize=9)
            ax.invert_yaxis()
            ax.axvline(x=0.0, color="black", linestyle="-", linewidth=0.8)
            ax.set_xlabel("SHAP Value (Contribution to Spread %)", fontsize=9, fontweight="bold")
            ax.set_title(
                f"{bond_id} (Rating: {rating}, C_t: {c_score:.1f})\n"
                f"Pred: {pred_val:.3f}% ({pred_val*100:.1f} bps) | E[f(x)]: {self.expected_value:.3f}%",
                fontsize=10,
                fontweight="bold",
            )
            ax.grid(axis="x", linestyle="--", alpha=0.5)

            for j, v in enumerate(top_vals):
                offset = 0.005 if v >= 0 else -0.005
                ha = "left" if v >= 0 else "right"
                ax.text(v + offset, j, f"{v:+.3f}", va="center", ha=ha, fontsize=8, fontweight="bold")

        plt.tight_layout()
        indiv_plot_path = output_dir / "shap_individual_waterfall.png"
        plt.savefig(indiv_plot_path, dpi=300)
        plt.close()
        saved_plots.append(indiv_plot_path)
        logger.info("Saved individual SHAP explanation plots to %s", indiv_plot_path)

        return saved_plots
