"""
Tests for Phase 1c — Oracle Validator
======================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from iot.oracle_validator import (
    OracleValidator,
    generate_validation_report,
    THRESHOLD_THETA,
)
from iot.simulate_sensors import simulate_sensors


class TestOracleValidatorUnit:
    """Unit tests for the OracleValidator class."""

    def test_first_reading_always_accepted(self):
        """First reading for a sensor should always be accepted."""
        validator = OracleValidator(theta=15.0, expired_sensors=set())
        is_valid, reason = validator.validate_reading(sensor_id=0, carbon_score=50.0)
        assert is_valid
        assert reason == "ACCEPTED"

    def test_small_change_accepted(self):
        """Change within threshold should be accepted."""
        validator = OracleValidator(theta=15.0, expired_sensors=set())
        validator.validate_reading(sensor_id=0, carbon_score=50.0)  # first
        is_valid, reason = validator.validate_reading(sensor_id=0, carbon_score=55.0)
        assert is_valid
        assert reason == "ACCEPTED"

    def test_exactly_at_threshold_accepted(self):
        """Change exactly equal to theta should be accepted."""
        validator = OracleValidator(theta=15.0, expired_sensors=set())
        validator.validate_reading(sensor_id=0, carbon_score=50.0)
        is_valid, reason = validator.validate_reading(sensor_id=0, carbon_score=65.0)
        assert is_valid  # |65 - 50| = 15 = θ, not > θ

    def test_large_change_rejected(self):
        """Change exceeding threshold should be rejected."""
        validator = OracleValidator(theta=15.0, expired_sensors=set())
        validator.validate_reading(sensor_id=0, carbon_score=50.0)
        is_valid, reason = validator.validate_reading(sensor_id=0, carbon_score=70.0)
        assert not is_valid
        assert "THRESHOLD_EXCEEDED" in reason

    def test_large_negative_change_rejected(self):
        """Large decrease should also be rejected."""
        validator = OracleValidator(theta=15.0, expired_sensors=set())
        validator.validate_reading(sensor_id=0, carbon_score=80.0)
        is_valid, reason = validator.validate_reading(sensor_id=0, carbon_score=60.0)
        assert not is_valid
        assert "THRESHOLD_EXCEEDED" in reason

    def test_expired_certificate_rejected(self):
        """Sensors with expired certificates should always be rejected."""
        validator = OracleValidator(
            theta=15.0,
            expired_sensors={5, 10},
        )
        is_valid, reason = validator.validate_reading(sensor_id=5, carbon_score=50.0)
        assert not is_valid
        assert reason == "EXPIRED_CERTIFICATE"

    def test_expired_cert_rejected_even_with_valid_score(self):
        """Expired cert takes priority over valid score change."""
        validator = OracleValidator(
            theta=15.0,
            expired_sensors={3},
        )
        # Even first reading should fail for expired sensor
        is_valid, _ = validator.validate_reading(sensor_id=3, carbon_score=50.0)
        assert not is_valid

    def test_non_expired_sensor_accepted(self):
        """Sensors not in the expired set should pass cert check."""
        validator = OracleValidator(
            theta=15.0,
            expired_sensors={5, 10},
        )
        is_valid, reason = validator.validate_reading(sensor_id=0, carbon_score=50.0)
        assert is_valid

    def test_different_sensors_independent(self):
        """Each sensor should have independent threshold tracking."""
        validator = OracleValidator(theta=15.0, expired_sensors=set())
        validator.validate_reading(sensor_id=0, carbon_score=50.0)
        validator.validate_reading(sensor_id=1, carbon_score=10.0)

        # Sensor 0: small change → accepted
        valid_0, _ = validator.validate_reading(sensor_id=0, carbon_score=55.0)
        assert valid_0

        # Sensor 1: large jump from 10 → 40 → rejected
        valid_1, _ = validator.validate_reading(sensor_id=1, carbon_score=40.0)
        assert not valid_1

    def test_sequential_small_changes_all_accepted(self):
        """Multiple small sequential changes should all be accepted."""
        validator = OracleValidator(theta=15.0, expired_sensors=set())
        scores = [50, 55, 60, 63, 58, 52, 48, 45]
        for score in scores:
            is_valid, reason = validator.validate_reading(
                sensor_id=0, carbon_score=score
            )
            assert is_valid, f"Score {score} rejected: {reason}"

    def test_custom_theta(self):
        """Custom theta should be respected."""
        validator = OracleValidator(theta=5.0, expired_sensors=set())
        validator.validate_reading(sensor_id=0, carbon_score=50.0)

        # 6-point change > theta=5 → rejected
        is_valid, _ = validator.validate_reading(sensor_id=0, carbon_score=56.0)
        assert not is_valid

        # Reset with new sensor — 4-point change < theta=5 → accepted
        validator.validate_reading(sensor_id=1, carbon_score=50.0)
        is_valid, _ = validator.validate_reading(sensor_id=1, carbon_score=54.0)
        assert is_valid


class TestOracleValidatorDataFrame:
    """Integration tests with simulated sensor data."""

    @pytest.fixture(scope="class")
    def validated_data(self):
        """Generate small sensor dataset and validate it."""
        sensor_df = simulate_sensors(n_sensors=20, n_hours=24, seed=42)
        validator = OracleValidator(
            theta=THRESHOLD_THETA,
            expired_sensors={2, 15},
            n_sensors=20,
        )
        return validator.validate_dataframe(sensor_df)

    def test_validation_columns_added(self, validated_data):
        """Validation adds is_valid and rejection_reason columns."""
        assert "is_valid" in validated_data.columns
        assert "rejection_reason" in validated_data.columns

    def test_some_readings_rejected(self, validated_data):
        """At least some readings should be rejected (expired certs)."""
        rejected = (~validated_data["is_valid"]).sum()
        assert rejected > 0, "Expected some rejections but found none"

    def test_expired_sensors_always_rejected(self, validated_data):
        """All readings from expired sensors should be rejected."""
        for sid in [2, 15]:
            sensor_readings = validated_data[validated_data["sensor_id"] == sid]
            if len(sensor_readings) > 0:
                rejected = (~sensor_readings["is_valid"]).sum()
                assert rejected == len(sensor_readings), (
                    f"Sensor {sid} (expired): {rejected}/{len(sensor_readings)} "
                    f"rejected, expected all"
                )

    def test_all_readings_have_reason(self, validated_data):
        """Every row should have a non-empty rejection reason or 'ACCEPTED'."""
        assert (validated_data["rejection_reason"].str.len() > 0).all()

    def test_accepted_readings_marked_correctly(self, validated_data):
        """Accepted readings should have reason 'ACCEPTED'."""
        accepted = validated_data[validated_data["is_valid"]]
        assert (accepted["rejection_reason"] == "ACCEPTED").all()

    def test_row_count_preserved(self, validated_data):
        """Validation shouldn't add or remove rows."""
        expected = 20 * 24
        assert len(validated_data) == expected


class TestValidationReport:
    """Tests for the validation report generation."""

    def test_report_is_string(self):
        """Report should be a string."""
        sensor_df = simulate_sensors(n_sensors=10, n_hours=10, seed=42)
        validator = OracleValidator(
            theta=THRESHOLD_THETA,
            expired_sensors={1},
            n_sensors=10,
        )
        df = validator.validate_dataframe(sensor_df)
        report = generate_validation_report(df)
        assert isinstance(report, str)
        assert len(report) > 50

    def test_report_contains_counts(self):
        """Report should include total, accepted, and rejected counts."""
        sensor_df = simulate_sensors(n_sensors=10, n_hours=10, seed=42)
        validator = OracleValidator(
            theta=THRESHOLD_THETA,
            expired_sensors={1},
            n_sensors=10,
        )
        df = validator.validate_dataframe(sensor_df)
        report = generate_validation_report(df)

        assert "Total readings" in report
        assert "Accepted" in report
        assert "Rejected" in report

    def test_report_contains_cross_analysis(self):
        """Report should include malicious cross-analysis when column present."""
        sensor_df = simulate_sensors(n_sensors=10, n_hours=10, seed=42)
        validator = OracleValidator(
            theta=THRESHOLD_THETA,
            expired_sensors=set(),
            n_sensors=10,
        )
        df = validator.validate_dataframe(sensor_df)
        report = generate_validation_report(df)

        assert "Malicious readings" in report or "Cross-analysis" in report

