"""XYZ Projective Quantum Feature Map.

This is the main PQFMLib implementation. It supports multi-axis and
shared-feature encodings, diagonal terms, and cross-axis terms.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RZZGate

from pqfmlib.core.base import BaseProjectiveQFM
from pqfmlib.core.data import mutual_information_matrix
from pqfmlib.core.execution import run_projected_feature_job
from pqfmlib.core.observables import pauli_observables_axis_pairs
from pqfmlib.core.serialization import (
    blocks_from_jsonable,
    blocks_to_jsonable,
    metadata_to_column_name,
)
from pqfmlib.hardware import feature_assignment as GA
from pqfmlib.hardware.coupling import get_coupling_edges_and_costs
from pqfmlib.hardware.feature_assignment import pick_connected_subset_greedy
from pqfmlib.hardware.layout import edges_log_from_phys_nodes, validate_phys_nodes
from pqfmlib.utils.io import load_json, save_json

AXES_ALL = ("x", "y", "z")


def normalize_axes(features_per_qubit: int = 3, axes: Iterable[str] | None = None) -> tuple[str, ...]:
    """Normalize axis configuration."""
    if axes is None:
        if features_per_qubit not in (1, 2, 3):
            raise ValueError("features_per_qubit must be 1, 2, or 3.")
        return AXES_ALL[:features_per_qubit]
    axes = tuple(str(a).lower() for a in axes)
    invalid = [a for a in axes if a not in AXES_ALL]
    if invalid:
        raise ValueError(f"Invalid axes: {invalid}. Use only x, y, z.")
    if len(axes) == 0 or len(axes) > 3 or len(set(axes)) != len(axes):
        raise ValueError("axes must contain 1 to 3 unique axes.")
    return axes


def normalize_encoding_mode(encoding_mode: str = "multi_axis") -> str:
    """Normalize encoding mode aliases."""
    mode = str(encoding_mode).lower().strip()
    aliases = {
        "multi_axis": "multi_axis",
        "multiaxis": "multi_axis",
        "axis_features": "multi_axis",
        "different_features_per_axis": "multi_axis",
        "shared_feature": "shared_feature",
        "shared_axes": "shared_feature",
        "same_feature_all_axes": "shared_feature",
        "single_feature_all_axes": "shared_feature",
    }
    if mode not in aliases:
        raise ValueError("encoding_mode must be 'multi_axis' or 'shared_feature'.")
    return aliases[mode]


def sort_edges_by_importance_from_matrix(J):
    """Sort feature-feature edges by decreasing absolute mutual information."""
    n = J.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            val = float(J[i, j])
            if val > 0.0 and not np.isclose(val, 1.0):
                edges.append((int(i), int(j), val))
    edges.sort(key=lambda t: (-abs(float(t[2])), int(t[0]), int(t[1])))
    return edges


def build_blocks_edge_first(J, q_enc: int, remaining_policy="remove_features"):
    """Build feature blocks by greedily packing high-MI feature pairs."""
    edges_sorted = sort_edges_by_importance_from_matrix(J)
    n_feat = J.shape[0]
    remaining_features = set(range(n_feat))
    used_edges = set()
    blocks = []

    def edge_ok(a, b):
        a, b = int(a), int(b)
        if a not in remaining_features or b not in remaining_features:
            return False
        if (min(a, b), max(a, b)) in used_edges:
            return False
        return True

    idx = 0
    while True:
        seed_edge = None
        while idx < len(edges_sorted):
            a, b, _ = edges_sorted[idx]
            idx += 1
            if edge_ok(a, b):
                seed_edge = (int(a), int(b))
                break
        if seed_edge is None:
            if remaining_policy == "remove_features" and remaining_features:
                leftovers = sorted(int(f) for f in remaining_features)
                for start in range(0, len(leftovers), q_enc):
                    blocks.append({"feat_set": tuple(leftovers[start : start + q_enc])})
                remaining_features.clear()
            break
        feat_set = set(seed_edge)
        for u, v, _ in edges_sorted:
            if not edge_ok(u, v):
                continue
            new_feats = feat_set | {int(u), int(v)}
            if len(new_feats) <= q_enc:
                feat_set = new_feats
            if len(feat_set) == q_enc:
                break
        blocks.append({"feat_set": tuple(sorted(int(f) for f in feat_set))})
        if remaining_policy == "remove_features":
            remaining_features -= feat_set
        else:
            used_edges.add((min(seed_edge), max(seed_edge)))
        if len(remaining_features) < 2:
            break
    if remaining_policy == "remove_features" and remaining_features:
        leftovers = sorted(int(f) for f in remaining_features)
        for start in range(0, len(leftovers), q_enc):
            blocks.append({"feat_set": tuple(leftovers[start : start + q_enc])})
    return blocks


def build_blocks_edge_first_multiaxis(
    J,
    q_enc: int,
    features_per_qubit: int = 3,
    axes: Iterable[str] | None = None,
    remaining_policy="remove_features",
    encoding_mode: str = "multi_axis",
):
    """Build blocks according to the selected XYZ encoding capacity."""
    axes = normalize_axes(features_per_qubit, axes)
    encoding_mode = normalize_encoding_mode(encoding_mode)
    capacity = q_enc if encoding_mode == "shared_feature" else q_enc * len(axes)
    return build_blocks_edge_first(J, q_enc=capacity, remaining_policy=remaining_policy)


def _feature_importance_order(J_global, feat_set):
    feat_list = sorted(int(f) for f in feat_set)
    scores = []
    for f in feat_list:
        vals = np.array(J_global[f, feat_list], dtype=float, copy=True)
        vals[np.isclose(vals, 1.0)] = 0.0
        scores.append((f, float(np.sum(np.abs(vals)))))
    scores.sort(key=lambda t: (-float(t[1]), int(t[0])))
    return [f for f, _ in scores]


def _build_J_block_for_qubit_assignment(J_global, feat_list, q_enc):
    J_block = np.zeros((q_enc, q_enc), dtype=float)
    for i in range(q_enc):
        fi = int(feat_list[i])
        if fi < 0:
            continue
        for j in range(i + 1, q_enc):
            fj = int(feat_list[j])
            if fj < 0:
                continue
            val = float(J_global[fi, fj])
            if np.isclose(val, 1.0):
                val = 0.0
            J_block[i, j] = val
            J_block[j, i] = val
    return J_block


def _basis_to_z(qc: QuantumCircuit, axis: str, qubit: int):
    if axis == "x":
        qc.h(qubit)
    elif axis == "y":
        qc.rx(+np.pi / 2, qubit)
    elif axis == "z":
        pass
    else:
        raise ValueError(f"Invalid axis: {axis}")


def _basis_from_z(qc: QuantumCircuit, axis: str, qubit: int):
    if axis == "x":
        qc.h(qubit)
    elif axis == "y":
        qc.rx(-np.pi / 2, qubit)
    elif axis == "z":
        pass
    else:
        raise ValueError(f"Invalid axis: {axis}")


def add_exp_minus_i_theta_pauli(qc: QuantumCircuit, axis: str, i: int, theta):
    """Append exp(-i theta sigma_axis) on one qubit."""
    _basis_to_z(qc, axis, i)
    qc.rz(2 * theta, i)
    _basis_from_z(qc, axis, i)


def add_exp_minus_i_theta_pauli_pair(qc: QuantumCircuit, axis_i: str, i: int, axis_j: str, j: int, theta):
    """Append exp(-i theta sigma_axis_i sigma_axis_j) on an edge."""
    _basis_to_z(qc, axis_i, i)
    _basis_to_z(qc, axis_j, j)
    qc.append(RZZGate(2 * theta), [i, j])
    _basis_from_z(qc, axis_j, j)
    _basis_from_z(qc, axis_i, i)


def build_layer_mapping_and_Jterms_full_cross(
    J_global,
    feat_set,
    q_enc,
    edges_log,
    *,
    features_per_qubit: int = 3,
    axes: Iterable[str] | None = None,
    edge_weight=None,
    GA_assignment_fn=None,
    seed=42,
    n_keep=None,
    keep_diagonal_terms=True,
    keep_cross_terms=True,
    encoding_mode: str = "multi_axis",
):
    """Map features to XYZ slots and build J terms for the selected Hamiltonian."""
    axes = normalize_axes(features_per_qubit, axes)
    encoding_mode = normalize_encoding_mode(encoding_mode)

    if encoding_mode == "shared_feature":
        feat_order = _feature_importance_order(J_global, feat_set)
        if len(feat_order) < q_enc:
            feat_order += [-1] * (q_enc - len(feat_order))
        feat_order = feat_order[:q_enc]
        if GA_assignment_fn is not None:
            J_block = _build_J_block_for_qubit_assignment(J_global, feat_order, q_enc)
            perm_local, _ = GA_assignment_fn(
                J_block,
                edges_log,
                edge_weight=edge_weight,
                pop_size=80,
                ngen=300,
                k_tourn=3,
                elite_k=5,
                seed=seed,
            )
            feat_ids = [int(feat_order[perm_local[i]]) for i in range(q_enc)]
        else:
            feat_ids = [int(x) for x in feat_order]
        feat_ids_by_axis = {a: feat_ids[:] for a in axes}
    else:
        if GA_assignment_fn is not None:
            feat_ids_by_axis = GA_assignment_fn(
                J_global=J_global,
                feat_set=feat_set,
                q_enc=q_enc,
                edges_log=edges_log,
                axes=axes,
                features_per_qubit=features_per_qubit,
                edge_weight=edge_weight,
                keep_diagonal_terms=keep_diagonal_terms,
                keep_cross_terms=keep_cross_terms,
                seed=seed,
            )[0]
        else:
            slots = [(i, a) for i in range(q_enc) for a in axes]
            capacity = len(slots)
            feat_order = _feature_importance_order(J_global, feat_set)
            if len(feat_order) < capacity:
                feat_order += [-1] * (capacity - len(feat_order))
            feat_order = feat_order[:capacity]
            feat_ids_by_axis = {a: [-1] * q_enc for a in axes}
            for slot_idx, (i, a) in enumerate(slots):
                feat_ids_by_axis[a][i] = int(feat_order[slot_idx])

    allowed_axis_pairs = []
    for a in axes:
        for b in axes:
            if a == b and keep_diagonal_terms:
                allowed_axis_pairs.append((a, b))
            elif a != b and keep_cross_terms:
                allowed_axis_pairs.append((a, b))

    weighted_terms = []
    seen_edges = sorted({(min(i, j), max(i, j)) for i, j in edges_log})
    for i, j in seen_edges:
        for a, b in allowed_axis_pairs:
            fi = int(feat_ids_by_axis[a][i])
            fj = int(feat_ids_by_axis[b][j])
            if fi < 0 or fj < 0:
                continue
            Jij = float(J_global[fi, fj])
            if Jij <= 0.0 or np.isclose(Jij, 1.0):
                continue
            ew = 1.0 if edge_weight is None else float(edge_weight.get((i, j), edge_weight.get((j, i), 1.0)))
            weighted_terms.append(((a, b, i, j), Jij, abs(Jij) * ew))
    weighted_terms.sort(key=lambda t: t[2], reverse=True)
    if n_keep is not None:
        weighted_terms = weighted_terms[: int(n_keep)]
    return feat_ids_by_axis, {key: float(Jij) for key, Jij, _ in weighted_terms}


def build_full_cross_hamiltonian_circuit_param(
    num_qubits,
    edges_act,
    *,
    features_per_qubit: int = 3,
    axes: Iterable[str] | None = None,
    m: int = 1,
    initial_h: bool = True,
    keep_diagonal_terms: bool = True,
    keep_cross_terms: bool = True,
):
    """Build a first-order Trotterized XYZ Hamiltonian circuit."""
    axes = normalize_axes(features_per_qubit, axes)
    edge_list = sorted({(min(i, j), max(i, j)) for i, j in edges_act})
    axis_pairs = []
    for a in axes:
        for b in axes:
            if a == b and keep_diagonal_terms:
                axis_pairs.append((a, b))
            elif a != b and keep_cross_terms:
                axis_pairs.append((a, b))

    qc = QuantumCircuit(num_qubits)
    if initial_h:
        qc.h(range(num_qubits))

    n = int(num_qubits)
    n_edges = len(edge_list)
    theta_local = {a: ParameterVector(f"th_{a}", m * n) for a in axes}
    theta_pair = {ab: ParameterVector(f"tj_{ab[0]}{ab[1]}", m * n_edges) for ab in axis_pairs}

    for step in range(m):
        for a in axes:
            for i in range(n):
                add_exp_minus_i_theta_pauli(qc, a, i, theta_local[a][step * n + i])
        for ab in axis_pairs:
            a, b = ab
            for e_idx, (i, j) in enumerate(edge_list):
                add_exp_minus_i_theta_pauli_pair(qc, a, i, b, j, theta_pair[ab][step * n_edges + e_idx])

    param_order = []
    for a in axes:
        param_order.extend(list(theta_local[a]))
    for ab in axis_pairs:
        param_order.extend(list(theta_pair[ab]))
    return qc, param_order, {"axes": axes, "axis_pairs": axis_pairs, "edges": edge_list}


def make_theta_matrix_full_cross_blocks(
    X_q_reordered,
    blocks,
    edges_log,
    *,
    features_per_qubit: int = 3,
    axes: Iterable[str] | None = None,
    encoding_mode: str = "multi_axis",
    tau: float = 1.0,
    m_phys: int = 1,
    keep_diagonal_terms: bool = True,
    keep_cross_terms: bool = True,
):
    """Build the parameter matrix for all samples and all XYZ blocks."""
    axes = normalize_axes(features_per_qubit, axes)
    edge_list = sorted({(min(i, j), max(i, j)) for i, j in edges_log})
    axis_pairs = []
    for a in axes:
        for b in axes:
            if a == b and keep_diagonal_terms:
                axis_pairs.append((a, b))
            elif a != b and keep_cross_terms:
                axis_pairs.append((a, b))

    N, n_feat = X_q_reordered.shape
    q_enc = len(next(iter(blocks[0]["feat_ids_by_axis"].values())))
    n_edges = len(edge_list)
    n_layers = len(blocks)
    m_total = int(m_phys) * n_layers
    dt = float(tau) / float(m_phys)

    local_vals = {a: np.zeros((N, m_total * q_enc), dtype=np.float64) for a in axes}
    pair_vals = {ab: np.zeros((N, m_total * n_edges), dtype=np.float64) for ab in axis_pairs}

    for L, blk in enumerate(blocks):
        feat_ids_by_axis = blk["feat_ids_by_axis"]
        J_terms = blk["J_terms"]
        for step in range(m_phys):
            step_total = L * m_phys + step
            for r in range(N):
                for a in axes:
                    offset = step_total * q_enc
                    for i in range(q_enc):
                        fi = int(feat_ids_by_axis[a][i])
                        val = float(X_q_reordered[r, fi]) if 0 <= fi < n_feat else 0.0
                        local_vals[a][r, offset + i] = dt * val
                for ab in axis_pairs:
                    offset = step_total * n_edges
                    a, b = ab
                    for e_idx, (i, j) in enumerate(edge_list):
                        pair_vals[ab][r, offset + e_idx] = dt * float(J_terms.get((a, b, i, j), 0.0))

    arrays = [local_vals[a] for a in axes] + [pair_vals[ab] for ab in axis_pairs]
    return np.hstack(arrays), m_total


@dataclass
class XYZProjectiveQFM(BaseProjectiveQFM):
    """Runner for the XYZ Projective Quantum Feature Map."""

    use_fixed_blocks: bool = False
    use_fixed_phys_nodes: bool = False
    fixed_circuit_file_name: str = ""
    use_edge_error: bool = False
    features_per_qubit: int = 3
    axes: Optional[tuple[str, ...]] = None
    encoding_mode: str = "multi_axis"
    m: int = 1
    tau: float = 1.0
    keep_diagonal_terms: bool = True
    keep_cross_terms: bool = True
    n_keep_terms: Optional[int] = None
    measure_all_zz: bool = False
    measure_cross_observables: bool = False
    rho_thr: float = 0.0
    remaining_policy: str = "remove_features"
    fixed_blocks_file: str = "blocks.json"
    fixed_phys_nodes_file: str = "phys_nodes.json"
    resource_estimation: bool = False

    def __post_init__(self) -> None:
        self.encoding_mode = normalize_encoding_mode(self.encoding_mode)
        if self.encoding_mode == "shared_feature":
            self.axes = normalize_axes(1, self.axes or ("x", "y", "z"))
            if self.features_per_qubit != 1:
                raise ValueError("For shared_feature, set features_per_qubit=1 and choose axes separately.")
        else:
            self.axes = normalize_axes(self.features_per_qubit, self.axes)
            if self.features_per_qubit != len(self.axes):
                raise ValueError("For multi_axis, features_per_qubit must match len(axes).")
        super().__post_init__()
        if self.ideal:
            self.measure_all_zz = True
        self.J = None
        self.blocks_feat = None
        self.blocks = None
        self.edges_log = None
        self.edges_act = None
        self.edge_weight = None
        self.phys_nodes = None
        self.obs_metadata = None
        self.Xq_all_raw = None
        self.job_id = None

    def _block_capacity(self) -> int:
        return self.q_enc if self.encoding_mode == "shared_feature" else self.q_enc * len(self.axes)

    def _expected_n_blocks(self) -> int:
        return int(np.ceil(self.num_features / self._block_capacity()))

    def prepare_output_folder(self) -> None:
        axes_label = "".join(a.upper() for a in self.axes)
        parts = [
            "quantum_features",
            "XYZ",
            self.name_file,
            f"fperq{self.features_per_qubit}",
            f"axes{axes_label}",
            self.encoding_mode,
            str(self.num_features),
            str(self.q_enc),
        ]
        if self.ideal:
            parts.append("ideal")
        else:
            parts.append(self.ibm_qpu)
            if self.simulation:
                parts.append("fakebackend" if self.fakebackend else "simulacao")
                if self.measure_all_zz:
                    parts.append("All2Q")
        if self.measure_cross_observables:
            parts.append("CrossObs")
        if self.keep_cross_terms:
            parts.append("cross")
        elif self.keep_diagonal_terms:
            parts.append("diagonal")
        self.ensure_output_folder("_".join(parts))

    def compute_global_J_and_feature_blocks(self) -> None:
        if self.use_fixed_blocks:
            return
        self.J = mutual_information_matrix(self.X_q_all, seed=self.seed, rho_thr=self.rho_thr)
        self.blocks_feat = build_blocks_edge_first_multiaxis(
            self.J,
            q_enc=self.q_enc,
            features_per_qubit=len(self.axes),
            axes=self.axes,
            remaining_policy=self.remaining_policy,
            encoding_mode=self.encoding_mode,
        )

    def _validate_blocks_consistency(self) -> None:
        expected_blocks = self._expected_n_blocks()
        if len(self.blocks) != expected_blocks:
            raise ValueError(f"blocks.json has {len(self.blocks)} blocks, but {expected_blocks} are expected.")
        all_features = []
        for block_id, blk in enumerate(self.blocks):
            if "feat_ids_by_axis" not in blk:
                raise ValueError(f"Block {block_id} has no feat_ids_by_axis.")
            for axis in self.axes:
                if axis not in blk["feat_ids_by_axis"]:
                    raise ValueError(f"Block {block_id} is missing axis {axis}.")
                if len(blk["feat_ids_by_axis"][axis]) != self.q_enc:
                    raise ValueError(f"Block {block_id}, axis {axis}, has incompatible length.")
            if self.encoding_mode == "shared_feature":
                for qi in range(self.q_enc):
                    ids = {int(blk["feat_ids_by_axis"][a][qi]) for a in self.axes if int(blk["feat_ids_by_axis"][a][qi]) >= 0}
                    if len(ids) > 1:
                        raise ValueError(f"Shared-feature block {block_id}, qubit {qi}, has different features by axis.")
                    all_features.extend(list(ids))
            else:
                for axis in self.axes:
                    all_features.extend([int(f) for f in blk["feat_ids_by_axis"][axis] if int(f) >= 0])
        if self.remaining_policy == "remove_features":
            counts = Counter(all_features)
            duplicates = [f for f, count in counts.items() if count > 1]
            if duplicates:
                raise ValueError(f"blocks.json contains repeated features: {duplicates[:10]}")
            missing = sorted(set(range(self.num_features)) - set(all_features))
            if missing:
                raise ValueError(f"blocks.json does not cover all dataset features. Missing examples: {missing[:10]}")

    def prepare_blocks_and_edges(self) -> None:
        if self.ideal:
            self.phys_nodes = list(range(self.q_enc))
            self.edges_log = [(i, j) for i in range(self.q_enc) for j in range(i + 1, self.q_enc)]
            self.edges_act = self.edges_log
            self.edge_weight = None
            if self.use_fixed_blocks:
                self.blocks = blocks_from_jsonable(load_json(self.fixed_blocks_file))
                self._validate_blocks_consistency()
                return
            self.blocks = []
            for layer, blk in enumerate(self.blocks_feat):
                feat_ids_by_axis, J_terms = build_layer_mapping_and_Jterms_full_cross(
                    self.J,
                    blk["feat_set"],
                    self.q_enc,
                    self.edges_log,
                    features_per_qubit=len(self.axes),
                    axes=self.axes,
                    seed=self.seed + layer,
                    n_keep=self.n_keep_terms,
                    keep_diagonal_terms=self.keep_diagonal_terms,
                    keep_cross_terms=self.keep_cross_terms,
                    encoding_mode=self.encoding_mode,
                )
                self.blocks.append({"feat_ids_by_axis": feat_ids_by_axis, "J_terms": J_terms})
            self._validate_blocks_consistency()
            save_json(blocks_to_jsonable(self.blocks), Path(self.base_folder) / "blocks.json")
            return

        self._load_real_backend()
        coupling_edges, edge_cost = get_coupling_edges_and_costs(self.real_backend)
        if self.use_fixed_phys_nodes:
            self.phys_nodes = load_json(self.fixed_phys_nodes_file)
        else:
            self.phys_nodes = pick_connected_subset_greedy(self.q_enc, coupling_edges, edge_cost)
            save_json(self.phys_nodes, Path(self.base_folder) / "phys_nodes.json")
        validate_phys_nodes(self.phys_nodes, self.q_enc, coupling_edges, self.ibm_qpu)
        self.edges_log, self.edge_weight = edges_log_from_phys_nodes(coupling_edges, self.phys_nodes, edge_cost)
        self.edges_act = self.edges_log

        if self.use_fixed_blocks:
            self.blocks = blocks_from_jsonable(load_json(self.fixed_blocks_file))
            self._validate_blocks_consistency()
        else:
            edge_weight_used = self.edge_weight if self.use_edge_error else None
            ga_fn = GA.GA_assignment if self.encoding_mode == "shared_feature" else GA.GA_assignment_multiaxis
            self.blocks = []
            for layer, blk in enumerate(self.blocks_feat):
                feat_ids_by_axis, J_terms = build_layer_mapping_and_Jterms_full_cross(
                    self.J,
                    blk["feat_set"],
                    self.q_enc,
                    self.edges_log,
                    features_per_qubit=len(self.axes),
                    axes=self.axes,
                    edge_weight=edge_weight_used,
                    GA_assignment_fn=ga_fn,
                    seed=self.seed + layer,
                    n_keep=self.n_keep_terms,
                    keep_diagonal_terms=self.keep_diagonal_terms,
                    keep_cross_terms=self.keep_cross_terms,
                    encoding_mode=self.encoding_mode,
                )
                self.blocks.append({"feat_ids_by_axis": feat_ids_by_axis, "J_terms": J_terms})
            self._validate_blocks_consistency()
            save_json(blocks_to_jsonable(self.blocks), Path(self.base_folder) / "blocks.json")
        print("Physical subgraph:", self.phys_nodes)
        print("Number of blocks:", len(self.blocks))

    def _observable_edges(self):
        if self.ideal or self.measure_all_zz:
            return None
        return self.edges_act

    def execute_quantum_feature_map(self):
        theta_values_all, m_total = make_theta_matrix_full_cross_blocks(
            self.X_q_all,
            self.blocks,
            self.edges_log,
            features_per_qubit=len(self.axes),
            axes=self.axes,
            encoding_mode=self.encoding_mode,
            tau=self.tau,
            m_phys=self.m,
            keep_diagonal_terms=self.keep_diagonal_terms,
            keep_cross_terms=self.keep_cross_terms,
        )
        qc_param, param_order, _ = build_full_cross_hamiltonian_circuit_param(
            self.q_enc,
            self.edges_act,
            features_per_qubit=len(self.axes),
            axes=self.axes,
            m=m_total,
            keep_diagonal_terms=self.keep_diagonal_terms,
            keep_cross_terms=self.keep_cross_terms,
        )
        observable_edges = self._observable_edges()
        obs_list, obs_metadata = pauli_observables_axis_pairs(
            self.q_enc,
            axes=self.axes,
            edges_act=observable_edges,
            measure_cross_observables=self.measure_cross_observables,
        )
        metadata_extra = {
            "q_enc": int(self.q_enc),
            "features_per_qubit": int(self.features_per_qubit),
            "axes": list(self.axes),
            "encoding_mode": self.encoding_mode,
            "measure_cross_observables": bool(self.measure_cross_observables),
        }
        result = run_projected_feature_job(
            qc_param,
            param_order,
            theta_values_all,
            self.phys_nodes,
            self.backend,
            self.estimator,
            obs_list,
            obs_metadata,
            simulation=self.simulation,
            fakebackend=self.fakebackend,
            resource_estimation=self.resource_estimation,
            shots=self.shots,
            base_folder=self.base_folder,
            fixed_circuit=self.fixed_circuit_file_name,
            metadata_extra=metadata_extra,
            seed_transpiler=self.seed,
            qpy_filename="xyz_circuit.qpy",
            save_circuit_drawings=self.save_circuit_drawings,
        )
        if self.simulation:
            self.Xq_all_raw, self.obs_metadata = result
            return None
        self.job_id = result
        return self.job_id

    def save_quantum_features(self) -> None:
        q_col_names = [metadata_to_column_name(m) for m in self.obs_metadata]
        df_q_all = pd.DataFrame(self.Xq_all_raw, columns=q_col_names)
        df_q_all["y"] = self.y
        self.csv_output_path = str(Path(self.base_folder) / f"qfeatures_all_{self.name_file}.csv")
        self.npy_output_path = str(Path(self.base_folder) / f"qfeatures_all_{self.name_file}.npy")
        self.metadata_output_path = str(Path(self.base_folder) / f"qfeatures_metadata_{self.name_file}.json")
        df_q_all.to_csv(self.csv_output_path, index=False)
        np.save(self.npy_output_path, self.Xq_all_raw)
        save_json([list(m) if isinstance(m, tuple) else m for m in self.obs_metadata], self.metadata_output_path)

    def run(self):
        self.load_data()
        self.setup_backend_and_estimator()
        self.prepare_output_folder()
        self.compute_global_J_and_feature_blocks()
        self.prepare_blocks_and_edges()
        result = self.execute_quantum_feature_map()
        if not self.simulation:
            return result
        self.save_quantum_features()
        print("XYZ quantum features generated successfully.")
        print("Features saved to:", self.csv_output_path)
        return {
            "Xq_all_raw": self.Xq_all_raw,
            "obs_metadata": self.obs_metadata,
            "csv_path": self.csv_output_path,
            "npy_path": self.npy_output_path,
            "base_folder": self.base_folder,
            "encoding_mode": self.encoding_mode,
        }
