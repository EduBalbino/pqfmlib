"""CD-inspired Ising-glass Projective Quantum Feature Map."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RZZGate

from pqfmlib.core.base import BaseProjectiveQFM
from pqfmlib.core.data import mutual_information_matrix
from pqfmlib.core.execution import run_projected_feature_job
from pqfmlib.core.observables import z_observables, z_observables_edges
from pqfmlib.core.serialization import blocks_from_jsonable, blocks_to_jsonable
from pqfmlib.hardware import feature_assignment as GA
from pqfmlib.hardware.coupling import get_coupling_edges_and_costs
from pqfmlib.hardware.feature_assignment import pick_connected_subset_greedy
from pqfmlib.hardware.layout import edges_log_from_phys_nodes, validate_phys_nodes
from pqfmlib.maps.xyz import build_blocks_edge_first
from pqfmlib.utils.io import load_json, save_json


def add_exp_minus_i_theta_Y(qc, i, theta):
    """Append exp(-i theta Y_i)."""
    qc.rx(+np.pi / 2, i)
    qc.rz(2 * theta, i)
    qc.rx(-np.pi / 2, i)


def add_exp_minus_i_theta_YZ(qc, i, j, theta):
    """Append exp(-i theta Y_i Z_j)."""
    qc.rx(+np.pi / 2, i)
    qc.append(RZZGate(2 * theta), [i, j])
    qc.rx(-np.pi / 2, i)


def add_exp_minus_i_theta_ZY(qc, i, j, theta):
    """Append exp(-i theta Z_i Y_j)."""
    qc.rx(+np.pi / 2, j)
    qc.append(RZZGate(2 * theta), [i, j])
    qc.rx(-np.pi / 2, j)


def makeRt(h_x, J_dict, s, sum_hi2, sum_Jij2):
    """Auxiliary denominator used by the first-order CD coefficient."""
    sum_hi4 = sum(hi**4 for hi in h_x)
    sum_Jij4 = sum(Jij**4 for Jij in J_dict.values())
    sum_hi2Jii2 = 0.0
    for (i, j), Jij in J_dict.items():
        sum_hi2Jii2 += (h_x[i] ** 2 + h_x[j] ** 2) * (Jij**2)
    contrib_duplet = 2 * 6 * sum_hi2Jii2

    E = {(min(i, j), max(i, j)) for (i, j) in J_dict.keys()}

    def J(i, j):
        return J_dict[(i, j)] if (i, j) in J_dict else J_dict[(j, i)]

    sum_triplet = 0.0
    n = len(h_x)
    for i, j, k in combinations(range(n), 3):
        e1 = (min(i, j), max(i, j))
        e2 = (min(i, k), max(i, k))
        e3 = (min(j, k), max(j, k))
        if e1 in E and e2 in E and e3 in E:
            Jij, Jik, Jjk = J(i, j), J(i, k), J(j, k)
            sum_triplet += Jij**2 * Jik**2 + Jij**2 * Jjk**2 + Jik**2 * Jjk**2
    contrib_triplet = 6.0 * sum_triplet
    return ((1 - s) ** 2) * (sum_hi2 + 4 * sum_Jij2) + (s**2) * (
        sum_hi4 + 2 * sum_Jij4 + contrib_duplet + contrib_triplet
    )


def makeAlpha1(h_x, J_dict, s):
    """First-order CD coefficient used in the original CD-Ising pipeline."""
    sum_hi2 = sum(hi**2 for hi in h_x)
    sum_Jij2 = sum(Jij**2 for Jij in J_dict.values())
    alpha1 = -(1 / 4) * (sum_hi2 + 2 * sum_Jij2)
    Rt = makeRt(h_x, J_dict, s, sum_hi2, 2 * sum_Jij2)
    return alpha1 / Rt


def ds_curve(t, T):
    s = t / T
    return -((np.pi**2) / (4 * T)) * np.sin(np.pi * s) * np.sin(np.pi * (np.sin((np.pi / 2) * s) ** 2))


def s_curve(t_norm: float) -> float:
    return np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * t_norm) ** 2) ** 2


def build_layer_mapping_and_Jdict(J_global, feat_set, q_enc, edges_log, edge_weight=None, GA_assignment_fn=None, seed=42, n_keep=None):
    """Map one CD-Ising block to qubit slots and active J terms."""
    feat_list = list(feat_set)
    if len(feat_list) < q_enc:
        feat_list += [-1] * (q_enc - len(feat_list))
    feat_list = feat_list[:q_enc]

    J_block = np.zeros((q_enc, q_enc), dtype=float)
    for i in range(q_enc):
        fi = feat_list[i]
        if fi < 0:
            continue
        for j in range(i + 1, q_enc):
            fj = feat_list[j]
            if fj < 0:
                continue
            val = float(J_global[fi, fj])
            if np.isclose(val, 1.0):
                val = 0.0
            J_block[i, j] = val
            J_block[j, i] = J_block[i, j]

    if GA_assignment_fn is not None:
        perm_local, _ = GA_assignment_fn(
            J_block,
            edges_log,
            edge_weight=edge_weight,
            pop_size=60,
            ngen=200,
            k_tourn=3,
            elite_k=5,
            seed=seed,
        )
        feat_ids = [feat_list[perm_local[i]] for i in range(q_enc)]
    else:
        feat_ids = feat_list[:]

    edge_vals = []
    for i, j in edges_log:
        fi, fj = feat_ids[i], feat_ids[j]
        if fi < 0 or fj < 0:
            continue
        Jij = float(J_global[fi, fj])
        if Jij > 0.0 and not np.isclose(Jij, 1.0):
            edge_vals.append(((i, j), Jij))
    n_keep = len(edges_log) if n_keep is None else n_keep
    edge_vals.sort(key=lambda t: abs(t[1]), reverse=True)
    keep_set = {e for e, _ in edge_vals[:n_keep]}
    return feat_ids, {e: float(Jij) for e, Jij in edge_vals if e in keep_set}


def build_quench_circuit_param(num_qubits, edges_act, m=1):
    """Build the parametrized CD-Ising quench circuit."""
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    n = int(num_qubits)
    n_edges = len(edges_act)
    theta_y = ParameterVector("thy", m * n)
    theta_e = ParameterVector("the", m * n_edges)
    for step in range(m):
        for i in range(n):
            add_exp_minus_i_theta_Y(qc, i, theta_y[step * n + i])
        for e_idx, (i, j) in enumerate(edges_act):
            th = theta_e[step * n_edges + e_idx]
            add_exp_minus_i_theta_YZ(qc, i, j, th)
            add_exp_minus_i_theta_ZY(qc, i, j, th)
    return qc, list(theta_y) + list(theta_e)


def make_theta_matrix_all_blocks_hw(X_q_reordered, blocks, edges_log, tau, m_phys, s_curve_fn, ds_curve_fn, makeAlpha1_fn):
    """Build the CD-Ising theta matrix for all samples and blocks."""
    N, n_feat = X_q_reordered.shape
    q_enc = len(blocks[0]["feat_ids"])
    n_edges = len(edges_log)
    n_layers = len(blocks)
    m_total = int(m_phys) * n_layers
    dt = tau / m_phys
    theta_y_vals = np.zeros((N, m_total * q_enc), dtype=np.float64)
    theta_e_vals = np.zeros((N, m_total * n_edges), dtype=np.float64)

    for L, blk in enumerate(blocks):
        feat_ids = blk["feat_ids"]
        J_dict_layer = blk["J_dict_layer"]
        Jij_vec_layer = np.zeros(n_edges, dtype=np.float64)
        for k, (i, j) in enumerate(edges_log):
            Jij_vec_layer[k] = float(J_dict_layer.get((i, j), 0.0))
        for step in range(m_phys):
            step_total = L * m_phys + step
            t_mid = (step + 0.25) * dt
            s = s_curve_fn(t_mid / tau)
            ds = ds_curve_fn(t_mid, tau)
            for r in range(N):
                h_vec = np.zeros(q_enc, dtype=np.float64)
                for i in range(q_enc):
                    fi = feat_ids[i]
                    if 0 <= fi < n_feat:
                        h_vec[i] = float(X_q_reordered[r, fi])
                alpha1 = makeAlpha1_fn(h_vec, J_dict_layer, s)
                coeff_global = -2 * ds * dt * alpha1
                theta_y_vals[r, step_total * q_enc : (step_total + 1) * q_enc] = coeff_global * h_vec
                theta_e_vals[r, step_total * n_edges : (step_total + 1) * n_edges] = coeff_global * Jij_vec_layer
    return np.hstack([theta_y_vals, theta_e_vals]), m_total


@dataclass
class CDIsingProjectiveQFM(BaseProjectiveQFM):
    """Runner for the CD-inspired Ising-glass Projective Quantum Feature Map."""

    use_fixed_blocks: bool = False
    use_fixed_phys_nodes: bool = False
    fixed_circuit_file_name: str = ""
    use_edge_error: bool = False
    m: int = 1
    rho_thr: float = 0.0
    k_max: int = 2
    tau: float = 0.005
    fixed_blocks_file: str = "blocks.json"
    fixed_phys_nodes_file: str = "phys_nodes.json"
    measure_all_zz: bool = False
    resource_estimation: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.J = None
        self.blocks_feat = None
        self.blocks = None
        self.edges_log = None
        self.edges_act = None
        self.phys_nodes = None
        self.pairs_2q = None
        self.Xq_all_raw = None
        self.job_id = None

    def prepare_output_folder(self) -> None:
        parts = ["quantum_features", "CDIsing", self.name_file, f"k_max{self.k_max}", str(self.num_features), str(self.q_enc)]
        if self.ideal:
            parts.append("ideal")
        else:
            parts.append(self.ibm_qpu)
            if self.simulation:
                parts.append("fakebackend" if self.fakebackend else "simulacao")
                if self.measure_all_zz:
                    parts.append("ZZAll")
        self.ensure_output_folder("_".join(parts))

    def compute_global_J_and_feature_blocks(self) -> None:
        if self.use_fixed_blocks:
            return
        self.J = mutual_information_matrix(self.X_q_all, seed=self.seed, rho_thr=self.rho_thr)
        self.blocks_feat = build_blocks_edge_first(self.J, q_enc=self.q_enc, remaining_policy="remove_features")

    def prepare_blocks_and_edges(self) -> None:
        if self.ideal:
            self.phys_nodes = list(range(self.q_enc))
            self.edges_log = [(i, j) for i in range(self.q_enc) for j in range(i + 1, self.q_enc)]
            self.edges_act = self.edges_log
            if self.use_fixed_blocks:
                self.blocks = blocks_from_jsonable(load_json(self.fixed_blocks_file))
                return
            self.blocks = []
            for layer, blk in enumerate(self.blocks_feat):
                feat_ids, J_dict_layer = build_layer_mapping_and_Jdict(
                    self.J,
                    blk["feat_set"],
                    self.q_enc,
                    self.edges_log,
                    GA_assignment_fn=None,
                    seed=self.seed + layer,
                    n_keep=len(self.edges_log),
                )
                self.blocks.append({"feat_ids": feat_ids, "J_dict_layer": J_dict_layer})
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
        self.edges_log, edge_weight = edges_log_from_phys_nodes(coupling_edges, self.phys_nodes, edge_cost)
        self.edges_act = self.edges_log
        if self.use_fixed_blocks:
            self.blocks = blocks_from_jsonable(load_json(self.fixed_blocks_file))
        else:
            self.blocks = []
            edge_weight_used = edge_weight if self.use_edge_error else None
            for layer, blk in enumerate(self.blocks_feat):
                feat_ids, J_dict_layer = build_layer_mapping_and_Jdict(
                    self.J,
                    blk["feat_set"],
                    self.q_enc,
                    self.edges_log,
                    edge_weight=edge_weight_used,
                    GA_assignment_fn=GA.GA_assignment,
                    seed=self.seed + layer,
                    n_keep=len(self.edges_log),
                )
                self.blocks.append({"feat_ids": feat_ids, "J_dict_layer": J_dict_layer})
            save_json(blocks_to_jsonable(self.blocks), Path(self.base_folder) / "blocks.json")
        print("Physical subgraph:", self.phys_nodes)
        print("Number of blocks:", len(self.blocks))

    def execute_quantum_feature_map(self):
        theta_values_all, m_total = make_theta_matrix_all_blocks_hw(
            self.X_q_all,
            self.blocks,
            self.edges_log,
            self.tau,
            self.m,
            s_curve,
            ds_curve,
            makeAlpha1,
        )
        qc_param, param_order = build_quench_circuit_param(self.q_enc, self.edges_act, m=m_total)
        if self.measure_all_zz:
            obs_list, pairs_2q = z_observables(self.q_enc, k_max=self.k_max)
        else:
            obs_list, pairs_2q = z_observables_edges(self.q_enc, self.edges_act)
        obs_metadata = [("z", i) for i in range(self.q_enc)] + [("zz", i, j) for i, j in pairs_2q]
        metadata_extra = {"q_enc": int(self.q_enc), "pairs_2q": [list(p) for p in pairs_2q]}
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
            qpy_filename="cd_ising_circuit.qpy",
            save_circuit_drawings=self.save_circuit_drawings,
        )
        if self.simulation:
            self.Xq_all_raw, self.obs_metadata = result
            self.pairs_2q = pairs_2q
            return None
        self.job_id = result
        return self.job_id

    def save_quantum_features(self) -> None:
        n_samples, n_q_feats = self.Xq_all_raw.shape
        n1 = self.q_enc
        q1_names = [f"q1_z_{i}" for i in range(n1)]
        q2_names = [f"q2_z_{i}_{j}" for i, j in self.pairs_2q] if n_q_feats > n1 else []
        df_q_all = pd.DataFrame(self.Xq_all_raw, columns=q1_names + q2_names)
        df_q_all["y"] = self.y
        self.csv_output_path = str(Path(self.base_folder) / f"qfeatures_all_{self.name_file}.csv")
        self.npy_output_path = str(Path(self.base_folder) / f"qfeatures_all_{self.name_file}.npy")
        df_q_all.to_csv(self.csv_output_path, index=False)
        np.save(self.npy_output_path, self.Xq_all_raw)

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
        print("CD-Ising quantum features generated successfully.")
        print("Features saved to:", self.csv_output_path)
        return {
            "Xq_all_raw": self.Xq_all_raw,
            "pairs_2q": self.pairs_2q,
            "csv_path": self.csv_output_path,
            "npy_path": self.npy_output_path,
            "base_folder": self.base_folder,
        }
