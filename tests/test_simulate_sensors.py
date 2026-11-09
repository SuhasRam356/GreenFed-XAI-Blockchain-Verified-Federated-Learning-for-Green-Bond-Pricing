"""
Tests for Phase 1b — IoT Sensor Simulator
==========================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from iot.simulate_sensors import (
    simulate_sensors,
    N_SENSORS,
    N_HOURS,
    MALICIOUS_FRACTION,
    MALICIOUS_BIAS,
    EMAX,
)


class TestSimulateSensors:
    """Tests for the IoT sensor simulation."""

    @pytest.fixture(scope="class")
    def sensor_data(self):
        """Generate sensor data once for the test class (smaller for speed)."""
        return simulate_sensors(n_sensors=50, n_hours=48, seed=42)

    @pytest.fixture(scope="class")
    def full_sensor_data(self):
        """Full-size sensor data (500 sensors, 720 hours) — for stats tests."""
        return simulate_sensors(n_sensors=500, n_hours=24, seed=42)

    def test_correct_total_readings(self, sensor_data):
        """Total readings = n_sensors × n_hours."""
        expected = 50 * 48
        assert len(sensor_data) == expected, (
            f"Expected {expected} readings, got {len(sensor_data)}"
        )

    def test_columns_present(self, sensor_data):
        """All required columns must exist."""
        expected = {
            "sensor_id", "timestamp", "hour", "emission_rate_true",
            "emission_rate_reported", "carbon_score_raw",
            "carbon_score_reported", "is_malicious",
        }
        assert expected == set(sensor_data.columns)

    def test_sensor_id_range(self, sensor_data):
        """Sensor IDs should be 0 to n_sensors-1."""
        assert sensor_data["sensor_id"].min() == 0
        assert sensor_data["sensor_id"].max() == 49

    def test_all_sensors_have_all_hours(self, sensor_data):
        """Each sensor should have exactly n_hours readings."""
        counts = sensor_data.groupby("sensor_id").size()
        assert (counts == 48).all(), "Not all sensors have 48 readings"

    def test_carbon_score_range(self, sensor_data):
        """Carbon scores must be in [0, 100]."""
        assert sensor_data["carbon_score_reported"].min() >= 0.0
        assert sensor_data["carbon_score_reported"].max() <= 100.0
        assert sensor_data["carbon_score_raw"].min() >= 0.0
        assert sensor_data["carbon_score_raw"].max() <= 100.0

    def test_emission_rate_positive(self, sensor_data):
        """Emission rates must be positive."""
        assert sensor_data["emission_rate_true"].min() > 0
        assert sensor_data["emission_rate_reported"].min() > 0

    def test_malicious_fraction(self, full_sensor_data):
        """Approximately 5% of sensors should be malicious."""
        n_mal = full_sensor_data.loc[
            full_sensor_data["is_malicious"], "sensor_id"
        ].nunique()
        n_total = full_sensor_data["sensor_id"].nunique()
        ratio = n_mal / n_total
        assert 0.04 <= ratio <= 0.06, (
            f"Malicious fraction {ratio:.3f} outside expected range [0.04, 0.06]"
        )

    def test_malicious_sensors_have_higher_carbon_scores(self, full_sensor_data):
        """Malicious sensors should report inflated carbon scores on average.

        Malicious sensors reduce reported emissions by 20%, making their
        carbon scores artificially *higher*.
        """
        mal = full_sensor_data[full_sensor_data["is_malicious"]]
        hon = full_sensor_data[~full_sensor_data["is_malicious"]]

        mal_mean = mal["carbon_score_reported"].mean()
        hon_mean = hon["carbon_score_reported"].mean()

        # Malicious scores should be higher (bias makes projects look greener)
        assert mal_mean > hon_mean, (
            f"Malicious mean ({mal_mean:.2f}) should be > "
            f"honest mean ({hon_mean:.2f})"
        )

    def test_malicious_vs_raw_difference(self, sensor_data):
        """For malicious sensors, reported score should differ from raw."""
        mal = sensor_data[sensor_data["is_malicious"]]
        if len(mal) > 0:
            # Reported emission rate should be lower than raw (due to bias)
            diff = (
                mal["carbon_score_reported"].mean()
                - mal["carbon_score_raw"].mean()
            )
            assert diff > 0, (
                "Malicious reported scores should be higher than raw "
                f"(diff={diff:.4f})"
            )

    def test_honest_sensors_raw_close_to_reported(self, sensor_data):
        """For honest sensors, reported ≈ raw (just sensor error)."""
        hon = sensor_data[~sensor_data["is_malicious"]]
        if len(hon) > 0:
            diff = abs(
                hon["carbon_score_reported"].mean()
                - hon["carbon_score_raw"].mean()
            )
            # Should be very close (only sensor noise difference)
            assert diff < 1.0, (
                f"Honest sensors: reported vs raw diff = {diff:.4f}, "
                "expected < 1.0"
            )

    def test_reproducibility(self):
        """Same seed should produce identical data."""
        df1 = simulate_sensors(n_sensors=10, n_hours=10, seed=123)
        df2 = simulate_sensors(n_sensors=10, n_hours=10, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_no_null_values(self, sensor_data):
        """No NaN values allowed."""
        assert sensor_data.isnull().sum().sum() == 0

    def test_timestamp_format(self, sensor_data):
        """Timestamps should be valid ISO format."""
        # Just check the first few parse correctly
        for ts in sensor_data["timestamp"].head(5):
            pd.Timestamp(ts)  # will raise if invalid

