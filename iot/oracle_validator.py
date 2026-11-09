"""
Phase 1c — Oracle Validator
============================

Simulates the blockchain smart contract's oracle validation logic.
A reading is **rejected** if:

  1. |Ct − C(t−1)| > θ  (threshold ≈ 15 points)  — physically impossible
     sudden jump in carbon footprint score.
  2. The sensor certificate has expired (simulated by random expiry dates).

This mirrors the paper's description:
    "The oracle smart contract rejects a data reading if the absolute
     change |Ct − Ct−1| exceeds a threshold θ (indicating a physically
     impossible sudden jump) or if the sensor certificate has expired."

Usage:
    python iot/oracle_validator.py
    python iot/oracle_validator.py --sensor-file outputs/sensor_readings.csv
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
THRESHOLD_THETA = 15.0   # max allowed change per hour in carbon score

# Certificate expiry: simulate that some sensors have expired certs
CERT_EXPIRY_FRACTION = 0.03  # 3 % of sensors have expired certificates

# Validation start time (readings before this are considered "first reading")
VALIDATION_START = "2025-01-01T00:00:00"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class OracleValidator:
    """Simulates the blockchain oracle validation smart contract.

    Validates IoT sensor readings by checking:
      1. Carbon score change threshold (|ΔCt| ≤ θ).
      2. Sensor certificate validity.

    Parameters
    ----------
    theta : float
        Maximum allowed absolute change in carbon footprint score
        between consecutive hourly readings for the same sensor.
    expired_sensors : set[int] | None
        Set of sensor IDs whose certificates have expired. If None,
        a random 3% of sensors are marked as expired.
    """

    def __init__(
        self,
        theta: float = THRESHOLD_THETA,
        expired_sensors: set | None = None,
        n_sensors: int = 500,
        seed: int = RANDOM_SEED,
    ):
        self.theta = theta

        if expired_sensors is not None:
            self.expired_sensors = expired_sensors
        else:
            rng = np.random.default_rng(seed + 1000)  # different from simulator
            n_expired = int(np.ceil(n_sensors * CERT_EXPIRY_FRACTION))
            self.expired_sensors = set(
                rng.choice(n_sensors, size=n_expired, replace=False)
            )

        # Store last known carbon score per sensor for threshold checking
        self._last_score: dict[int, float] = {}

    def validate_reading(
        self,
        sensor_id: int,
        carbon_score: float,
    ) -> tuple[bool, str]:
        """Validate a single sensor reading.

        Parameters
        ----------
        sensor_id : int
        carbon_score : float
            The reported carbon footprint score (0–100).

        Returns
        -------
        tuple[bool, str]
            (is_valid, reason) — ``True`` if accepted, ``False`` if rejected.
        """
        # Check 1: certificate expiry
        if sensor_id in self.expired_sensors:
            return False, "EXPIRED_CERTIFICATE"

        # Check 2: threshold check (skip for first reading of this sensor)
        if sensor_id in self._last_score:
            delta = abs(carbon_score - self._last_score[sensor_id])
            if delta > self.theta:
                return False, f"THRESHOLD_EXCEEDED (|ΔCt|={delta:.2f} > θ={self.theta})"

        # Passed all checks — update last score
        self._last_score[sensor_id] = carbon_score
        return True, "ACCEPTED"

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate every reading in a DataFrame (must be sorted by time).

        Adds columns ``is_valid`` and ``rejection_reason`` to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: sensor_id, carbon_score_reported, hour.
            Must be sorted by (sensor_id, hour) or at least chronologically.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with two new columns.
        """
        # Reset state
        self._last_score = {}

        # Sort by sensor_id and hour for correct sequential validation
        df_sorted = df.sort_values(["sensor_id", "hour"]).reset_index(drop=True)

        valid_flags = []
        reasons = []

        for _, row in df_sorted.iterrows():
            is_valid, reason = self.validate_reading(
                sensor_id=int(row["sensor_id"]),
                carbon_score=float(row["carbon_score_reported"]),
            )
            valid_flags.append(is_valid)
            reasons.append(reason)

        df_sorted["is_valid"] = valid_flags
        df_sorted["rejection_reason"] = reasons

        return df_sorted


def generate_validation_report(df: pd.DataFrame) -> str:
    """Generate a human-readable validation report.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``OracleValidator.validate_dataframe``.

    Returns
    -------
    str
        Multi-line report string.
    """
    total = len(df)
    accepted = df["is_valid"].sum()
    rejected = total - accepted

    lines = [
        "=" * 60,
        "  Oracle Validation Report",
        "=" * 60,
        f"  Total readings     : {total:,}",
        f"  Accepted           : {accepted:,}  ({100*accepted/total:.2f}%)",
        f"  Rejected           : {rejected:,}  ({100*rejected/total:.2f}%)",
        "",
    ]

    # Break down by rejection reason
    if rejected > 0:
        lines.append("  Rejection breakdown:")
        reason_counts = (
            df.loc[~df["is_valid"], "rejection_reason"]
            .apply(lambda r: r.split(" (")[0])  # group threshold reasons
            .value_counts()
        )
        for reason, count in reason_counts.items():
            lines.append(f"    {reason:30s} : {count:,}")
        lines.append("")

    # Malicious sensor cross-analysis (if column present)
    if "is_malicious" in df.columns:
        mal_readings = df[df["is_malicious"]]
        hon_readings = df[~df["is_malicious"]]

        mal_rejected = (~mal_readings["is_valid"]).sum()
        hon_rejected = (~hon_readings["is_valid"]).sum()

        lines.extend([
            "  Cross-analysis with malicious flag:",
            f"    Malicious readings total    : {len(mal_readings):,}",
            f"    Malicious readings rejected : {mal_rejected:,}  "
            f"({100*mal_rejected/max(len(mal_readings),1):.2f}%)",
            f"    Honest readings total       : {len(hon_readings):,}",
            f"    Honest readings rejected    : {hon_rejected:,}  "
            f"({100*hon_rejected/max(len(hon_readings),1):.2f}%)",
            "",
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate IoT sensor readings using oracle logic"
    )
    parser.add_argument(
        "--sensor-file", type=str, default=None,
        help="Path to sensor_readings.csv (default: outputs/sensor_readings.csv)"
    )
    parser.add_argument("--theta", type=float, default=THRESHOLD_THETA)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.sensor_file:
        sensor_path = Path(args.sensor_file)
    else:
        sensor_path = project_root / "outputs" / "sensor_readings.csv"

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sensor_path.exists():
        print(f"Error: sensor file not found at {sensor_path}")
        print("Run iot/simulate_sensors.py first to generate sensor data.")
        return

    print(f"Loading sensor readings from {sensor_path} …")
    df = pd.read_csv(sensor_path)
    print(f"  Loaded {len(df):,} readings.")

    n_sensors = df["sensor_id"].nunique()
    validator = OracleValidator(theta=args.theta, n_sensors=n_sensors)
    print(f"  Expired sensor certificates: {sorted(validator.expired_sensors)}")
    print(f"  Threshold θ = {args.theta}")
    print()

    print("Validating readings …")
    df_validated = validator.validate_dataframe(df)

    report = generate_validation_report(df_validated)
    print(report)

    # Save validated readings
    out_path = out_dir / "sensor_readings_validated.csv"
    df_validated.to_csv(out_path, index=False)
    print(f"Saved validated readings → {out_path}")

    # Save validation report
    report_path = out_dir / "validation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved validation report  → {report_path}")


if __name__ == "__main__":
    main()

