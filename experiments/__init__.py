"""
GreenFed-XAI Experiments Module
===============================
Execution matrix and benchmarking suites reproducing and extending the
IEEE 2025 GreenFed-XAI paper results.
"""

from experiments.run_matrix import (
    ExperimentResult,
    run_complete_experiment_matrix,
    run_tau_sensitivity_sweep,
)

__all__ = [
    "ExperimentResult",
    "run_complete_experiment_matrix",
    "run_tau_sensitivity_sweep",
]
