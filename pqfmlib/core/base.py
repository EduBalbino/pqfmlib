"""Base class for projected quantum feature maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

from pqfmlib.core.backend import load_ibm_backend, make_aer_backend, make_estimator
from pqfmlib.core.data import load_tabular_dataset


@dataclass
class BaseProjectiveQFM:
    """Shared runner state for PQFM implementations.

    Subclasses implement the physics-specific pieces: block construction,
    circuit construction, theta matrix construction, and observables.
    """

    name_file: str
    seed: int = 42
    simulation: bool = True
    fakebackend: bool = False
    ideal: bool = True
    shots: int = 4096
    ibm_qpu: str = "ibm_fez"
    q_enc: int = 20
    data_dir: str = "./data"
    output_root: str = "."
    mps: bool = False
    use_gpu_statevector: bool = False
    statevector_device: Optional[str] = None
    mps_max_bond_dimension: Optional[int] = None
    mps_truncation_threshold: float = 1e-16
    fakebackend_method: str = "density_matrix"
    fakebackend_device: str = "CPU"
    resilience_level: int = 0
    qiskit_channel: str = "ibm_cloud"
    save_circuit_drawings: bool = False

    def __post_init__(self) -> None:
        self._validate_resource_estimation_mode()
        if self.ideal:
            self.simulation = True
            self.fakebackend = False
        if self.mps and self.use_gpu_statevector:
            self.use_gpu_statevector = False
        np.random.seed(self.seed)
        self.service = None
        self.real_backend = None
        self.backend = None
        self.estimator = None
        self.df_full: pd.DataFrame | None = None
        self.X = None
        self.y = None
        self.X_q_all = None
        self.num_features = None
        self.base_folder = None

    def _validate_resource_estimation_mode(self) -> None:
        if self.ideal and bool(getattr(self, "resource_estimation", False)):
            raise ValueError("resource_estimation=True is invalid when ideal=True; ideal simulations do not use QPU resources.")

    def load_data(self) -> None:
        self.df_full, self.X, self.y = load_tabular_dataset(self.name_file, self.data_dir)
        self.X_q_all = self.X.copy()
        self.num_features = int(self.X.shape[1])

    def setup_backend_and_estimator(self) -> None:
        self._validate_resource_estimation_mode()
        if self.ideal:
            self.backend = self._make_aer_backend()
        else:
            self._load_real_backend()
            if self.simulation:
                if self.fakebackend:
                    self.backend = AerSimulator.from_backend(self.real_backend)
                    self.backend.set_options(
                        method=self.fakebackend_method,
                        device=self.fakebackend_device,
                        seed_simulator=self.seed,
                    )
                else:
                    self.backend = self._make_aer_backend()
            else:
                self.backend = self.real_backend
                self.backend.options.seed_transpiler = self.seed
        self.estimator = make_estimator(self.backend, shots=self.shots, resilience_level=self.resilience_level)

    def _load_real_backend(self) -> None:
        if self.real_backend is None:
            self.real_backend, self.service = load_ibm_backend(self.ibm_qpu, channel=self.qiskit_channel)

    def _make_aer_backend(self):
        return make_aer_backend(
            seed=self.seed,
            mps=self.mps,
            use_gpu_statevector=self.use_gpu_statevector,
            statevector_device=self.statevector_device,
            mps_max_bond_dimension=self.mps_max_bond_dimension,
            mps_truncation_threshold=self.mps_truncation_threshold,
        )

    def ensure_output_folder(self, name: str) -> str:
        if self.output_root in ("", ".", "./", ".\\"):
            self.base_folder = name
        else:
            self.base_folder = str(Path(self.output_root) / name)
        Path(self.base_folder).mkdir(parents=True, exist_ok=True)
        return self.base_folder

    def run(self):
        raise NotImplementedError("Subclasses must implement run().")
