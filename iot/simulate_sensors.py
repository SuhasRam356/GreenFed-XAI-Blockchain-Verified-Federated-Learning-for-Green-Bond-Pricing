"""
Phase 1b — IoT Sensor Simulator
================================

Simulates 500 virtual IoT sensors deployed at green-bond project sites.
Each sensor produces one CO₂e reading per hour.

Key characteristics (from the paper):
  - 500 sensors, 1 reading/hour
  - 5 % Gaussian sensor error on each reading
  - 5 % of sensors are *malicious* and inject a +20 % positive bias

The carbon footprint score is computed as:
    Ct = 100 × (Emax − Et) / Emax

where Et is the current emission rate (kg CO₂e / kWh) and Emax is a
project-type baseline.

Usage:
    python iot/simulate_sensors.py
    python iot/simulate_sensors.py --n-sensors 500 --hours 720 --out-dir outputs/
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_SENSORS = 500
N_HOURS = 720  # 30 days × 24 hours
MALICIOUS_FRACTION = 0.05   # 5 % of sensors
MALICIOUS_BIAS = 0.20       # +20 % positive bias
SENSOR_ERROR_STD = 0.05     # 5 % sensor error
EMAX = 1.0                  # baseline emission rate (kg CO₂e / kWh)

# Simulate slowly-varying true emission rates per sensor
# (each sensor is at a different project site)
EMISSION_RATE_RANGE = (0.05, 0.95)  # Et ∈ [0.05, 0.95] × Emax

# Hourly drift: tiny random walk so readings aren't static
HOURLY_DRIFT_STD = 0.005


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate_sensors(
    n_sensors: int = N_SENSORS,
    n_hours: int = N_HOURS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Simulate IoT sensor readings.

    Returns
    -------
    pd.DataFrame
        Columns: sensor_id, timestamp, emission_rate, carbon_footprint_score,
                 is_malicious, reading_raw, reading_reported.
    """
    rng = np.random.default_rng(seed)

    # Assign malicious flag
    n_malicious = int(np.ceil(n_sensors * MALICIOUS_FRACTION))
    malicious_ids = set(rng.choice(n_sensors, size=n_malicious, replace=False))

    records = []
    start_time = datetime(2025, 1, 1, 0, 0, 0)

    # Initialise true emission rates per sensor
    true_rates = rng.uniform(
        EMISSION_RATE_RANGE[0], EMISSION_RATE_RANGE[1], size=n_sensors
    )

    for hour in range(n_hours):
        timestamp = start_time + timedelta(hours=hour)

        # Apply small random walk to emission rates (clipped to valid range)
        drift = rng.normal(0, HOURLY_DRIFT_STD, size=n_sensors)
        true_rates = np.clip(
            true_rates + drift,
            EMISSION_RATE_RANGE[0],
            EMISSION_RATE_RANGE[1],
        )

        for sid in range(n_sensors):
            is_mal = sid in malicious_ids

            # True emission reading
            et_true = true_rates[sid]

            # Add sensor error (5 % Gaussian noise)
            sensor_noise = rng.normal(0, SENSOR_ERROR_STD * et_true)
            et_noisy = et_true + sensor_noise

            # Malicious sensors add +20 % bias (making emissions look LOWER
            # → carbon score artificially HIGHER to enable greenwashing)
            if is_mal:
                et_reported = et_noisy * (1 - MALICIOUS_BIAS)
            else:
                et_reported = et_noisy

            # Clip to valid range
            et_reported = np.clip(et_reported, 0.001, EMAX)
            et_noisy = np.clip(et_noisy, 0.001, EMAX)

            # Compute carbon footprint scores
            ct_raw = 100 * (EMAX - et_noisy) / EMAX
            ct_reported = 100 * (EMAX - et_reported) / EMAX

            records.append({
                "sensor_id": sid,
                "timestamp": timestamp.isoformat(),
                "hour": hour,
                "emission_rate_true": round(et_true, 6),
                "emission_rate_reported": round(et_reported, 6),
                "carbon_score_raw": round(ct_raw, 2),
                "carbon_score_reported": round(ct_reported, 2),
                "is_malicious": is_mal,
            })

    df = pd.DataFrame(records)
    return df


def print_summary(df: pd.DataFrame):
    """Print statistics about the simulated sensor data."""
    n_sensors = df["sensor_id"].nunique()
    n_hours = df["hour"].nunique()
    n_mal = df.loc[df["is_malicious"], "sensor_id"].nunique()

    print(f"\n{'='*60}")
    print(f"  IoT Sensor Simulation Summary")
    print(f"{'='*60}")
    print(f"  Total sensors      : {n_sensors}")
    print(f"  Malicious sensors  : {n_mal} ({100*n_mal/n_sensors:.1f}%)")
    print(f"  Honest sensors     : {n_sensors - n_mal}")
    print(f"  Hours simulated    : {n_hours}")
    print(f"  Total readings     : {len(df):,}")
    print()
    print("  Carbon score (reported) statistics:")
    print(f"    Mean  : {df['carbon_score_reported'].mean():.2f}")
    print(f"    Std   : {df['carbon_score_reported'].std():.2f}")
    print(f"    Min   : {df['carbon_score_reported'].min():.2f}")
    print(f"    Max   : {df['carbon_score_reported'].max():.2f}")
    print()

    # Compare malicious vs honest carbon scores
    mal_scores = df.loc[df["is_malicious"], "carbon_score_reported"]
    hon_scores = df.loc[~df["is_malicious"], "carbon_score_reported"]
    print(f"  Malicious avg carbon score : {mal_scores.mean():.2f}")
    print(f"  Honest avg carbon score    : {hon_scores.mean():.2f}")
    print(f"  Bias difference            : {mal_scores.mean() - hon_scores.mean():.2f} points")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate IoT sensor readings for green bond projects"
    )
    parser.add_argument("--n-sensors", type=int, default=N_SENSORS)
    parser.add_argument("--hours", type=int, default=N_HOURS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Simulating IoT sensors …")
    df = simulate_sensors(
        n_sensors=args.n_sensors,
        n_hours=args.hours,
        seed=args.seed,
    )
    print_summary(df)

    out_path = out_dir / "sensor_readings.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved sensor readings → {out_path}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()

