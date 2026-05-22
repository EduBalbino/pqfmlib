"""Notebook-based Heisenberg Projective Quantum Feature Map.

This module implements the Heisenberg map from ``Heisenberg_map.ipynb``:

- one random single-qubit unitary per qubit;
- repeated even/odd nearest-neighbor chain layers;
- each encoded scalar feature drives RZZ, RXX, and RYY on one chain edge;
- feature angles are scaled as ``2*pi*tanh(x/3)``;
- observables are one-local Z, X, Y for every qubit.

The implementation intentionally keeps the notebook semantics, while exposing it
through the unified PQFMLib runner interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import SparsePauliOp, random_unitary
from qiskit.transpiler import Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from pqfmlib.core.base import BaseProjectiveQFM
from pqfmlib.core.execution import reorder_theta_by_parameter_names, submit_job_and_save_metadata
from pqfmlib.core.resources import test_circuit_t
from pqfmlib.core.serialization import metadata_to_column_name
from pqfmlib.hardware.coupling import get_coupling_edges_and_costs
from pqfmlib.hardware.feature_assignment import pick_connected_subset_greedy
from pqfmlib.hardware.layout import validate_phys_nodes
from pqfmlib.utils.io import load_json, save_circuit_draw, save_json


def heisenberg_chain_observables(
    num_qubits: int,
    *,
    measure_2local_diagonal: bool = False,
    edges_act: list[tuple[int, int]] | None = None,
):
    """Build Heisenberg observables.

    Always measures one-local Z, X, Y per qubit.

    Optionally measures two-local diagonal observables:
        ZZ, XX, YY

    By default, two-local observables are measured on nearest-neighbor
    chain edges only.
    """
    num_qubits = int(num_qubits)

    obs = []
    metadata = []

    def pauli_string(terms):
        p = ["I"] * num_qubits
        for idx, axis in terms:
            p[int(idx)] = str(axis).upper()
        return "".join(p)

    # One-local observables: Z, X, Y per qubit.
    for i in range(num_qubits):
        for axis in ("z", "x", "y"):
            obs.append(
                SparsePauliOp.from_list(
                    [(pauli_string([(i, axis)]), 1.0)]
                )
            )
            metadata.append((axis, i))

    # Two-local diagonal observables: ZZ, XX, YY.
    if measure_2local_diagonal:
        if edges_act is None:
            edges_act = [(i, i + 1) for i in range(num_qubits - 1)]

        edges = sorted({(min(i, j), max(i, j)) for i, j in edges_act})

        for i, j in edges:
            for axis in ("z", "x", "y"):
                axis_pair = axis + axis
                obs.append(
                    SparsePauliOp.from_list(
                        [(pauli_string([(i, axis), (j, axis)]), 1.0)]
                    )
                )
                metadata.append((axis_pair, i, j))

    return obs, metadata


def make_heisenberg_theta_matrix(
    X: np.ndarray,
    num_qubits: int,
    *,
    features_per_block: Optional[int] = None,
    use_tanh_scaling: bool = True,
):
    """Build the parameter matrix used by the notebook Heisenberg map.

    The notebook uses ``features_per_block = num_qubits - 1`` because each block
    encodes one feature per nearest-neighbor chain edge. If the dataset has more
    features than one block can cover, additional blocks are appended.
    """
    num_qubits = int(num_qubits)
    if features_per_block is None:
        features_per_block = num_qubits - 1
    features_per_block = int(features_per_block)

    if features_per_block <= 0:
        raise ValueError("features_per_block must be positive.")
    if features_per_block > num_qubits - 1:
        raise ValueError("features_per_block cannot exceed num_qubits - 1 for the chain map.")

    n_samples, num_features = X.shape
    theta_raw = X.astype(np.float64, copy=True)
    if use_tanh_scaling:
        theta_raw = 2.0 * np.pi * np.tanh(theta_raw / 3.0)

    num_blocks = int(np.ceil(num_features / features_per_block))
    total_slots = num_blocks * features_per_block

    theta_values = np.zeros((n_samples, total_slots), dtype=np.float64)
    theta_values[:, :num_features] = theta_raw

    return theta_values, {
        "num_features": int(num_features),
        "features_per_block": int(features_per_block),
        "num_blocks": int(num_blocks),
        "total_slots": int(total_slots),
        "use_tanh_scaling": bool(use_tanh_scaling),
    }


def build_heisenberg_chain_feature_circuit(
    num_qubits: int,
    *,
    num_blocks: int,
    features_per_block: Optional[int] = None,
    R: int = 2,
    alpha: float = 0.1,
    seed: int = 42,
    add_barriers: bool = True,
):
    """Build the notebook Heisenberg feature-map circuit.

    For each chain edge, the same scalar parameter is used in RZZ, RXX, and RYY.
    The even-edge sweep is followed by the odd-edge sweep, and this block is
    repeated ``R`` times.
    """
    from qiskit.circuit import ParameterVector

    num_qubits = int(num_qubits)
    num_blocks = int(num_blocks)
    R = int(R)

    if features_per_block is None:
        features_per_block = num_qubits - 1
    features_per_block = int(features_per_block)

    if features_per_block != num_qubits - 1:
        raise ValueError(
            "The notebook Heisenberg circuit expects features_per_block = num_qubits - 1. "
            "Use the default unless you intentionally modify the edge schedule."
        )

    qc = QuantumCircuit(num_qubits)

    for q in range(num_qubits):
        unitary = random_unitary(2, seed=seed + q)
        qc.unitary(unitary, [q], label=f"U_{q}")

    theta = ParameterVector("theta", num_blocks * features_per_block)
    phi_scale = float(alpha) / (2.0 * float(num_blocks))

    for _rep in range(R):
        index = 0
        for _block in range(num_blocks):
            for j in range(0, num_qubits - 1, 2):
                angle = phi_scale * theta[index]
                qc.rzz(angle, j, j + 1)
                qc.rxx(angle, j, j + 1)
                qc.ryy(angle, j, j + 1)
                index += 1

            for j in range(1, num_qubits - 1, 2):
                angle = phi_scale * theta[index]
                qc.rzz(angle, j, j + 1)
                qc.rxx(angle, j, j + 1)
                qc.ryy(angle, j, j + 1)
                index += 1

            if add_barriers:
                qc.barrier()

    return qc, list(theta), {
        "R": int(R),
        "alpha": float(alpha),
        "phi_scale": float(phi_scale),
        "num_blocks": int(num_blocks),
        "features_per_block": int(features_per_block),
    }


def _extract_initial_phys_nodes(qc_t, qc_reference: QuantumCircuit) -> list[int] | None:
    """Extract the initial physical layout chosen by the transpiler, if available."""
    try:
        layout = qc_t.layout.initial_virtual_layout(filter_ancillas=True)
        return [int(layout[q]) for q in qc_reference.qubits]
    except Exception:
        return None


@dataclass
class HeisenbergProjectiveQFM(BaseProjectiveQFM):
    """Heisenberg PQFM implementation based on the original notebook.

    Parameters
    ----------
    R:
        Number of repetitions of the even/odd Heisenberg chain schedule.
    alpha:
        Global interaction scale. The effective parameter scale is
        ``alpha / (2 * num_blocks)`` as in the notebook.
    use_tanh_scaling:
        If True, encode features as ``2*pi*tanh(x/3)``.
    use_fixed_phys_nodes:
        If False, the transpiler is allowed to choose a layout and the selected
        initial physical qubits are saved. If True, the given physical nodes are
        used as an initial layout, while routing remains enabled.
    """

    R: int = 2
    alpha: float = 0.1
    use_tanh_scaling: bool = True
    add_barriers: bool = True

    use_fixed_phys_nodes: bool = False
    fixed_phys_nodes_file: str = "phys_nodes.json"
    fixed_circuit_file_name: str = ""

    resource_estimation: bool = False

    # Kept for backward compatibility. The notebook measures only one-local Z/X/Y.
    axes: tuple[str, ...] = ("z", "x", "y")
    measure_2local_diagonal: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.phys_nodes = None
        self.obs_metadata = None
        self.Xq_all_raw = None
        self.job_id = None
        self.theta_info = None
        self.circuit_info = None

    def prepare_output_folder(self) -> None:
        features_per_block = int(self.q_enc) - 1
        parts = [
            f"heisenberg__R{int(self.R)}",
            "quantum_features",
            self.name_file,
            str(self.num_features),
            str(features_per_block),
        ]
        if self.ideal:
            parts.append("ideal")
        else:
            parts.append(self.ibm_qpu)
            if self.simulation:
                parts.append("fakebackend" if self.fakebackend else "simulacao")
        self.ensure_output_folder("_".join(parts))

    def prepare_layout(self) -> None:
        if self.ideal:
            self.phys_nodes = list(range(self.q_enc))
            return

        self._load_real_backend()
        if self.use_fixed_phys_nodes:
            coupling_edges, _edge_cost = get_coupling_edges_and_costs(self.real_backend)
            self.phys_nodes = load_json(self.fixed_phys_nodes_file)
            validate_phys_nodes(self.phys_nodes, self.q_enc, coupling_edges, self.ibm_qpu)
        else:
            # In notebook mode, the transpiler chooses the layout. The selected
            # physical nodes are extracted and saved after transpilation.
            self.phys_nodes = None

    def _transpile_or_load_circuit(self, qc_param: QuantumCircuit):
        """Transpile/load the Heisenberg circuit following the notebook semantics."""
        Path(self.base_folder).mkdir(parents=True, exist_ok=True)
        qpy_path = Path(self.base_folder) / "heisenberg_circuit.qpy"

        if self.fixed_circuit_file_name:
            fixed_path = Path(self.fixed_circuit_file_name)
            if fixed_path.suffix != ".qpy":
                fixed_path = fixed_path.with_suffix(".qpy")
            with open(fixed_path, "rb") as f:
                qc_t = qpy.load(f)[0]
            return qc_t

        if not self.simulation:
            if self.use_fixed_phys_nodes and self.phys_nodes is not None:
                initial_layout = Layout({qc_param.qubits[i]: self.phys_nodes[i] for i in range(self.q_enc)})
                qc_t = transpile(
                    qc_param,
                    backend=self.backend,
                    initial_layout=initial_layout,
                    optimization_level=3,
                    seed_transpiler=self.seed,
                )
            else:
                qc_t = transpile(
                    qc_param,
                    backend=self.backend,
                    optimization_level=3,
                    seed_transpiler=self.seed,
                )
                extracted = _extract_initial_phys_nodes(qc_t, qc_param)
                if extracted is not None:
                    self.phys_nodes = extracted
                    save_json(self.phys_nodes, Path(self.base_folder) / "phys_nodes.json")

            with open(qpy_path, "wb") as f:
                qpy.dump(qc_t, f)

        else:
            if self.fakebackend:
                if self.use_fixed_phys_nodes and self.phys_nodes is not None:
                    initial_layout = Layout({qc_param.qubits[i]: self.phys_nodes[i] for i in range(self.q_enc)})
                    qc_t = transpile(
                        qc_param,
                        backend=self.backend,
                        initial_layout=initial_layout,
                        optimization_level=3,
                        seed_transpiler=self.seed,
                    )
                else:
                    qc_t = transpile(
                        qc_param,
                        backend=self.backend,
                        optimization_level=3,
                        seed_transpiler=self.seed,
                    )
            else:
                pm = generate_preset_pass_manager(
                    optimization_level=3,
                    backend=self.backend,
                    seed_transpiler=self.seed,
                )
                qc_t = pm.run(qc_param)

        if self.save_circuit_drawings:
            save_circuit_draw(qc_t, self.base_folder, "heisenberg_circuit_transpiled")
            save_circuit_draw(qc_param, self.base_folder, "heisenberg_circuit_logical")

        return qc_t

    def execute_quantum_feature_map(self):
        theta_values_all, self.theta_info = make_heisenberg_theta_matrix(
            self.X_q_all,
            self.q_enc,
            features_per_block=self.q_enc - 1,
            use_tanh_scaling=self.use_tanh_scaling,
        )

        qc_param, param_order, self.circuit_info = build_heisenberg_chain_feature_circuit(
            self.q_enc,
            num_blocks=self.theta_info["num_blocks"],
            features_per_block=self.theta_info["features_per_block"],
            R=self.R,
            alpha=self.alpha,
            seed=self.seed,
            add_barriers=self.add_barriers,
        )

        qc_t = self._transpile_or_load_circuit(qc_param)

        if not self.simulation:
            test_circuit_t(
                qc_t,
                self.backend,
                self.resource_estimation,
                csv_path=str(Path(self.base_folder) / "resource_estimation.csv"),
                shots=self.shots,
                num_param_sets=int(theta_values_all.shape[0]),
            )
            if self.resource_estimation:
                return None

        obs_list, obs_metadata = heisenberg_chain_observables(
            self.q_enc,
            measure_2local_diagonal=self.measure_2local_diagonal,
        )

        if not self.simulation or self.fakebackend:
            obs_isa = [obs.apply_layout(qc_t.layout) for obs in obs_list]
        else:
            obs_isa = obs_list
        obs_isa_broadcast = [[obs] for obs in obs_isa]

        theta_reordered = reorder_theta_by_parameter_names(qc_t, param_order, theta_values_all)

        metadata_extra = {
            "q_enc": int(self.q_enc),
            "R": int(self.R),
            "alpha": float(self.alpha),
            "phi_scale": float(self.circuit_info["phi_scale"]),
            "features_per_block": int(self.theta_info["features_per_block"]),
            "num_blocks": int(self.theta_info["num_blocks"]),
            "total_slots": int(self.theta_info["total_slots"]),
            "use_tanh_scaling": bool(self.use_tanh_scaling),
        }
        if self.phys_nodes is not None:
            metadata_extra["phys_nodes"] = [int(x) for x in self.phys_nodes]

        if self.simulation:
            job = self.estimator.run([(qc_t, obs_isa_broadcast, theta_reordered)])
            result = job.result()[0]
            evs = np.asarray(result.data.evs, dtype=float)
            self.Xq_all_raw = evs.T
            self.obs_metadata = obs_metadata
            return None

        self.job_id = submit_job_and_save_metadata(
            self.estimator,
            qc_t,
            obs_isa_broadcast,
            theta_reordered,
            obs_metadata,
            base_folder=self.base_folder,
            **metadata_extra,
        )
        return self.job_id

    def save_quantum_features(self) -> None:
        q_col_names = [metadata_to_column_name(m) for m in self.obs_metadata]
        df_q_all = pd.DataFrame(self.Xq_all_raw, columns=q_col_names)
        df_q_all["y"] = self.y
        self.csv_output_path = str(Path(self.base_folder) / f"qfeatures_all_{self.name_file}_heisenberg.csv")
        self.npy_output_path = str(Path(self.base_folder) / f"qfeatures_all_{self.name_file}_heisenberg.npy")
        df_q_all.to_csv(self.csv_output_path, index=False)
        np.save(self.npy_output_path, self.Xq_all_raw)

    def run(self):
        self.load_data()
        self.setup_backend_and_estimator()
        self.prepare_output_folder()
        self.prepare_layout()
        result = self.execute_quantum_feature_map()
        if not self.simulation:
            return result
        self.save_quantum_features()
        print("Heisenberg quantum features generated successfully.")
        print("Features saved to:", self.csv_output_path)
        return {
            "Xq_all_raw": self.Xq_all_raw,
            "obs_metadata": self.obs_metadata,
            "csv_path": self.csv_output_path,
            "npy_path": self.npy_output_path,
            "base_folder": self.base_folder,
            "theta_info": self.theta_info,
            "circuit_info": self.circuit_info,
        }
