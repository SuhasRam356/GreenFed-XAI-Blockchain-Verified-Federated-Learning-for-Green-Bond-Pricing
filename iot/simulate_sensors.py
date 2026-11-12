"""
Phase 2 (Master Plan) — IoT Sensor Simulation
==============================================

Simulates IoT carbon telemetry streams at green bond project sites,
distinguishing between:
  1. Latent ground-truth emissions (Et_true).
  2. Latent ground-truth carbon footprint score (Ct_true).
  3. Observed noisy emissions reading (Et_observed).
  4. Observed carbon footprint score (Ct_observed).
  5. Malicious sensor telemetry with greenwashing bias.

Mathematical Formulation:
--------------------------
For a project site with baseline emission factor E_max (kg CO₂e / kWh):
    Ct_true = 100 × (E_max − Et_true) / E_max

Sensors exhibit 5% Gaussian measurement error:
    Et_noisy = Et_true + N(0, (0.05 × Et_true)²)

Malicious sensors (5% of total sensor fleet) inject a greenwashing bias
(reporting emissions 20% lower than reality):
    Et_reported = Et_noisy × (1 − 0.20)
    Ct_reported = 100 × (E_max − Et_reported) / E_max

This separation allows the evaluation protocol to test the impact of
unmitigated sensor corruption on ground-truth bond pricing targets.

Usage:
    python iot/simulate_sensors.py --n-sensors 500 --hours 720 --out-dir outputs/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Default parameters
RANDOM_SEED = 42
N_SENSORS = 500
N_HOURS = 720  # 30 days × 24 hours
MALICIOUS_FRACTION = 0.05   # 5% of sensors are compromised
MALICIOUS_BIAS = 0.20       # 20% reduction in reported emissions (greenwashing)
SENSOR_ERROR_STD = 0.05     # 5% Gaussian measurement noise
EMAX = 1.0                  # Baseline benchmark emission factor (kg CO₂e / kWh)
EMISSION_RATE_RANGE = (0.05, 0.95)
HOURLY_DRIFT_STD = 0.005


def compute_carbon_score(emission_rate: float, e_max: float = EMAX) -> float:
    """Compute carbon footprint score Ct in range [0, 100].

    Parameters:
    -----------
    emission_rate : float
        Carbon emission intensity (kg CO₂e / kWh).
    e_max : float
        Project-specific benchmark emission baseline.

    Returns:
    --------
    float
        Carbon footprint score between 0.0 (dirtiest) and 100.0 (cleanest).
    """
    score = 100.0 * (e_max - emission_rate) / e_max
    return float(np.clip(score, 0.0, 100.0))


def simulate_sensors(
    n_sensors: int = N_SENSORS,
    n_hours: int = N_HOURS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Simulate IoT sensor telemetry distinguishing latent ground-truth from observations.

    Parameters:
    -----------
    n_sensors : int
        Total number of project IoT monitoring devices (default: 500).
    n_hours : int
        Number of hourly observation time-steps (default: 720 = 30 days).
    seed : int
        Random seed for full reproducibility.

    Returns:
    --------
    pd.DataFrame
        Hourly telemetry dataset with latent state and observed readings.
    """
    rng = np.random.default_rng(seed)

    # Designate compromised malicious sensors (5% fleet fraction)
    n_malicious = int(np.ceil(n_sensors * MALICIOUS_FRACTION))
    malicious_ids: Set[int] = set(
        rng.choice(n_sensors, size=n_malicious, replace=False)
    )

    records: List[Dict[str, object]] = []
    start_time = datetime(2025, 1, 1, 0, 0, 0)

    # Initial latent ground-truth emission rates
    latent_true_rates = rng.uniform(
        EMISSION_RATE_RANGE[0], EMISSION_RATE_RANGE[1], size=n_sensors
    )

    for hour in range(n_hours):
        timestamp = start_time + timedelta(hours=hour)

        # Latent true emissions random walk drift
        drift = rng.normal(0, HOURLY_DRIFT_STD, size=n_sensors)
        latent_true_rates = np.clip(
            latent_true_rates + drift,
            EMISSION_RATE_RANGE[0],
            EMISSION_RATE_RANGE[1],
        )

        for sid in range(n_sensors):
            is_mal = sid in malicious_ids
            et_true = float(latent_true_rates[sid])
            ct_true = compute_carbon_score(et_true, EMAX)

            # Natural sensor measurement error (5% Gaussian std)
            noise = float(rng.normal(0, SENSOR_ERROR_STD * et_true))
            et_noisy = max(0.001, et_true + noise)

            # Adversarial manipulation: malicious sensors report lower emissions
            if is_mal:
                et_reported = max(0.001, et_noisy * (1.0 - MALICIOUS_BIAS))
                is_attacked = True
            else:
                et_reported = et_noisy
                is_attacked = False

            ct_reported = compute_carbon_score(et_reported, EMAX)

            records.append({
                "sensor_id": sid,
                "timestamp": timestamp.isoformat(),
                "hour": hour,
                "emission_rate_true": round(et_true, 6),
                "emission_rate_reported": round(et_reported, 6),
                "carbon_score_raw": round(compute_carbon_score(et_noisy, EMAX), 2),
                "carbon_score_reported": round(ct_reported, 2),
                "is_malicious": is_mal,
            })

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame) -> None:
    """Print structured summary of latent vs reported sensor statistics."""
    n_sensors = df["sensor_id"].nunique()
    n_hours = df["hour"].nunique()
    n_mal = df.loc[df["is_malicious"], "sensor_id"].nunique()

    print(f"\n{'='*70}")
    print("  IOT CARBON SENSOR SIMULATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total Active Sensors     : {n_sensors}")
    print(f"  Malicious Sensors        : {n_mal} ({100.0 * n_mal / n_sensors:.1f}%)")
    print(f"  Honest Sensors           : {n_sensors - n_mal}")
    print(f"  Total Hourly Time-Steps  : {n_hours} hours")
    print(f"  Total Telemetry Records  : {len(df):,}")
    print("-" * 70)

    hon_df = df[~df["is_malicious"]]
    mal_df = df[df["is_malicious"]]

    print(f"  Honest Reported Score Ct : Mean = {hon_df['carbon_score_reported'].mean():.2f}, Std = {hon_df['carbon_score_reported'].std():.2f}")
    ct_true_mean = 100.0 * (EMAX - mal_df['emission_rate_true'].mean()) / EMAX
    inflation = mal_df['carbon_score_reported'].mean() - ct_true_mean
    print(f"  Malicious Reported Ct    : Mean = {mal_df['carbon_score_reported'].mean():.2f} (Greenwashing inflation: +{inflation:.2f} pts)")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IoT sensor carbon telemetry")
    parser.add_argument("--n-sensors", type=int, default=N_SENSORS, help="Number of virtual sensors")
    parser.add_argument("--hours", type=int, default=N_HOURS, help="Number of simulation hours")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {args.n_sensors} sensors over {args.hours} hours...")
    df = simulate_sensors(n_sensors=args.n_sensors, n_hours=args.hours, seed=args.seed)
    print_summary(df)

    out_path = out_dir / "sensor_readings.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved sensor telemetry dataset -> {out_path} ({len(df):,} records)")


if __name__ == "__main__":
    main()
