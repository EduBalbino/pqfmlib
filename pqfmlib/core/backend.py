"""Backend and Estimator construction helpers."""

from __future__ import annotations

from typing import Optional

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

from pqfmlib.utils.config import Config


def make_aer_backend(
    *,
    seed: int = 42,
    mps: bool = False,
    use_gpu_statevector: bool = True,
    statevector_device: Optional[str] = None,
    mps_max_bond_dimension: Optional[int] = None,
    mps_truncation_threshold: float = 1e-16,
):
    """Build an Aer backend for ideal simulations."""
    if mps:
        backend = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=mps_max_bond_dimension,
            matrix_product_state_truncation_threshold=mps_truncation_threshold,
            device="CPU",
        )
    else:
        device = statevector_device or ("GPU" if use_gpu_statevector else "CPU")
        backend = AerSimulator(method="statevector", device=device)
    backend.options.seed_simulator = seed
    backend.options.seed_transpiler = seed
    return backend


def load_ibm_backend(ibm_qpu: str, *, channel: str = "ibm_cloud", token: Optional[str] = None):
    """Load a real IBM backend using Qiskit Runtime."""
    token = token or Config().QXToken
    service = QiskitRuntimeService(channel=channel, token=token)
    return service.backend(ibm_qpu), service


def make_estimator(backend, *, shots: int = 4096, resilience_level: int = 0):
    """Create an EstimatorV2 instance with default shots and resilience level."""
    options = {"default_shots": shots}
    if not isinstance(backend, AerSimulator):
        options["resilience_level"] = resilience_level
    return Estimator(backend, options=options)
