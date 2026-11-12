"""
Phase 2 (Master Plan) — Oracle Validator with Fallback & Rate Tracking
=====================================================================

Simulates the blockchain smart contract's oracle validation logic.
A sensor reading is **rejected** if:
  1. Jump check: |Ct − C(t−1)| > θ (threshold θ = 15 points),
     representing physically impossible carbon score variations.
  2. Certificate check: sensor certificate is expired or invalid.

Fallback Strategy:
-------------------
When a reading is rejected by the smart oracle gate, the system applies
a robust **forward-fill policy**: the last verified valid carbon score
is substituted for downstream bond pricing aggregation, preventing missing
values from crashing the predictive pipeline.

Metrics & Confusion Matrix:
----------------------------
Reports exact confusion matrix values with explicit denominators:
  - Malicious Detection Rate = Detected Malicious / Total Malicious
  - Honest False Rejection Rate = Rejected Honest / Total Honest

Usage:
    python iot/oracle_validator.py --sensor-file outputs/sensor_readings.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Constants
RANDOM_SEED = 42
THRESHOLD_THETA = 15.0
CERT_EXPIRY_FRACTION = 0.03
VALIDATION_START = "2025-01-01T00:00:00"


class OracleValidator:
    """Simulates the blockchain oracle validation smart contract."""

    def __init__(
        self,
        theta: float = THRESHOLD_THETA,
        expired_sensors: Optional[Set[int]] = None,
        n_sensors: int = 500,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.theta = theta

        if expired_sensors is not None:
            self.expired_sensors = expired_sensors
        else:
            rng = np.random.default_rng(seed + 1000)
            n_expired = int(np.ceil(n_sensors * CERT_EXPIRY_FRACTION))
            self.expired_sensors = set(
                rng.choice(n_sensors, size=n_expired, replace=False)
            )

        # Track last verified valid carbon score per sensor for jump check and fallback
        self._last_score: Dict[int, float] = {}

    def validate_reading(
        self,
        sensor_id: int,
        carbon_score: float,
        return_fallback: bool = False,
    ) -> tuple[bool, str] | tuple[bool, str, float]:
        """Validate a single sensor reading.

        Parameters
        ----------
        sensor_id : int
        carbon_score : float
            The reported carbon footprint score (0–100).
        return_fallback : bool
            If True, returns (is_valid, reason, mitigated_value)

        Returns
        -------
        tuple[bool, str] or tuple[bool, str, float]
        """
        # Check 1: certificate expiry
        if sensor_id in self.expired_sensors:
            fallback = self._last_score.get(sensor_id, carbon_score)
            if return_fallback:
                return False, "EXPIRED_CERTIFICATE", fallback
            return False, "EXPIRED_CERTIFICATE"

        # Check 2: threshold check (skip for first reading of this sensor)
        if sensor_id in self._last_score:
            delta = abs(carbon_score - self._last_score[sensor_id])
            if delta > self.theta:
                fallback = self._last_score[sensor_id]
                if return_fallback:
                    return False, f"THRESHOLD_EXCEEDED (|ΔCt|={delta:.2f} > θ={self.theta})", fallback
                return False, f"THRESHOLD_EXCEEDED (|ΔCt|={delta:.2f} > θ={self.theta})"

        # Passed all checks — update last score
        self._last_score[sensor_id] = carbon_score
        if return_fallback:
            return True, "ACCEPTED", carbon_score
        return True, "ACCEPTED"

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate an entire sensor telemetry dataframe in chronological order.

        Appends columns:
          - is_valid: bool
          - rejection_reason: str
          - carbon_score_mitigated: float (with forward-fill fallback)
        """
        self._last_score = {}
        df_sorted = df.sort_values(["sensor_id", "hour"]).reset_index(drop=True)

        valid_flags = []
        reasons = []
        mitigated_scores = []

        for _, row in df_sorted.iterrows():
            res = self.validate_reading(
                sensor_id=int(row["sensor_id"]),
                carbon_score=float(row["carbon_score_reported"]),
                return_fallback=True,
            )
            is_valid, reason, mitigated = res
            valid_flags.append(is_valid)
            reasons.append(reason)
            mitigated_scores.append(round(mitigated, 2))

        df_sorted["is_valid"] = valid_flags
        df_sorted["rejection_reason"] = reasons
        df_sorted["carbon_score_mitigated"] = mitigated_scores

        return df_sorted


def generate_validation_report(df: pd.DataFrame) -> str:
    """Generate structured audit report with rates and denominators."""
    total_readings = len(df)
    n_valid = int(df["is_valid"].sum())
    n_rejected = total_readings - n_valid

    reasons = df[~df["is_valid"]]["rejection_reason"].value_counts()
    cert_expired = sum(1 for r in df["rejection_reason"] if "EXPIRED" in r)
    threshold_exceeded = sum(1 for r in df["rejection_reason"] if "THRESHOLD" in r)

    # Calculate honest vs malicious rates if labels present
    has_labels = "is_malicious" in df.columns
    attack_metrics = ""
    if has_labels:
        total_malicious = int(df["is_malicious"].sum())
        total_honest = total_readings - total_malicious

        mal_rejected = int(((df["is_malicious"]) & (~df["is_valid"])).sum())
        honest_rejected = int(((~df["is_malicious"]) & (~df["is_valid"])).sum())

        detection_rate = (mal_rejected / total_malicious * 100.0) if total_malicious > 0 else 0.0
        false_positive_rate = (honest_rejected / total_honest * 100.0) if total_honest > 0 else 0.0

        attack_metrics = f"""
  Adversarial Telemetry & Evaluation Rates:
    Total Malicious Telemetry Readings: {total_malicious:,}
    Malicious Readings Detected & Blocked: {mal_rejected:,}
    Malicious Detection Rate: {detection_rate:.2f}% ({mal_rejected}/{total_malicious})

    Total Honest Telemetry Readings: {total_honest:,}
    Honest Readings Rejected (False Positives): {honest_rejected:,}
    Honest False Rejection Rate: {false_positive_rate:.2f}% ({honest_rejected}/{total_honest})
"""

    report = f"""======================================================================
  ORACLE SMART CONTRACT VALIDATION REPORT
======================================================================
  Total readings processed : {total_readings:,}
  Valid readings accepted  : {n_valid:,} ({100.0 * n_valid / total_readings:.2f}%)
  Invalid readings rejected: {n_rejected:,} ({100.0 * n_rejected / total_readings:.2f}%)
  Accepted                 : {n_valid:,}
  Rejected                 : {n_rejected:,}

  Rejection Categorization:
    - Expired Certificates : {cert_expired:,}
    - Threshold Jumps (>θ) : {threshold_exceeded:,}
{attack_metrics}
  Malicious readings cross-analysis:
    Total malicious: {df['is_malicious'].sum() if has_labels else 0}

  Mitigation Policy:
    Rejected readings are safely forward-filled from the last verified
    valid observation, preventing telemetry gaps in bond spread pricing.
======================================================================"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate IoT sensor telemetry stream")
    parser.add_argument("--sensor-file", type=str, default="outputs/sensor_readings.csv", help="Input sensor CSV")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--theta", type=float, default=THRESHOLD_THETA, help="Jump threshold θ")
    args = parser.parse_args()

    sensor_path = Path(args.sensor_file)
    if not sensor_path.exists():
        print(f"Error: Sensor file '{sensor_path}' not found. Run iot/simulate_sensors.py first.")
        return

    df = pd.read_csv(sensor_path)
    validator = OracleValidator(theta=args.theta)
    df_val = validator.validate_dataframe(df)

    report = generate_validation_report(df_val)
    print(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_csv_path = out_dir / "sensor_readings_validated.csv"
    df_val.to_csv(val_csv_path, index=False)

    report_path = out_dir / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved validated sensor data -> {val_csv_path}")
    print(f"Saved validation audit report -> {report_path}")


if __name__ == "__main__":
    main()
