"""
GreenFed-XAI Federated Learning Package
=======================================

This package implements the Federated Learning layer of the GreenFed-XAI
framework as described in Section III-C and Section IV-C of the IEEE paper:
"Blockchain-Verified Federated Learning for ESG-Driven Green Bond Pricing:
An Explainable AI Approach with IoT-Enabled Carbon Footprint Oracles".

Submodules:
-----------
- dataset_partitioner : Partitions green bond records across participating banks.
- client              : Flower-compliant local client for bank model training.
- strategy_fedavg     : Federated Averaging strategy tracking round-level convergence.
- server              : Orchestration runtime and simulation engine.
"""

from fl.dataset_partitioner import BankDatasetPartitioner, ClientDataset
from fl.client import GreenBondFlowerClient
from fl.strategy_fedavg import GreenFedAvgStrategy

__all__ = [
    "BankDatasetPartitioner",
    "ClientDataset",
    "GreenBondFlowerClient",
    "GreenFedAvgStrategy",
]

