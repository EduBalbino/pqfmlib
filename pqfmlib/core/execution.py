"""Generic Qiskit execution helpers for projected feature maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit import qpy, transpile
from qiskit.transpiler import Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from pqfmlib.core.resources import test_circuit_t
from pqfmlib.core.serialization import metadata_to_jsonable
from pqfmlib.utils.io import save_circuit_draw


def reorder_theta_by_parameter_names(qc_t, param_order, theta_values_all):
    """Reorder a theta matrix to match the transpiled circuit parameter order."""
    param_t = list(qc_t.parameters)
    param_order_names = [p.name for p in param_order]
    param_t_names = [p.name for p in param_t]
    if len(param_order_names) != len(set(param_order_names)):
        raise ValueError("Parameter names in param_order are not unique.")
    pos = {name: k for k, name in enumerate(param_order_names)}
    try:
        idx = [pos[name] for name in param_t_names]
    except KeyError as exc:
        missing = [name for name in param_t_names if name not in pos]
        raise KeyError(f"Some transpiled parameters are missing from param_order: {missing[:10]}") from exc
    return theta_values_all[:, idx]


def submit_job_and_save_metadata(
    estimator,
    qc_t,
    obs_isa_broadcast,
    theta_values_all,
    obs_metadata,
    base_folder="example",
    **extra_meta,
):
    """Submit one Estimator job and write job metadata to disk."""
    job = estimator.run([(qc_t, obs_isa_broadcast, theta_values_all)])
    job_id = job.job_id()
    out = {
        "job_id": job_id,
        "base_folder": base_folder,
        "n_param_sets": int(theta_values_all.shape[0]),
        "n_obs": int(len(obs_isa_broadcast)),
        "obs_metadata": metadata_to_jsonable(obs_metadata),
    }
    out.update(extra_meta)
    Path(base_folder).mkdir(parents=True, exist_ok=True)
    Path(base_folder, "job_meta.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Job submitted. job_id={job_id}")
    print(f"Metadata saved to {base_folder}/job_meta.json")
    return job_id


def run_projected_feature_job(
    qc_param,
    param_order,
    theta_values_all,
    phys_nodes,
    backend,
    estimator,
    obs_list,
    obs_metadata,
    *,
    simulation: bool = False,
    fakebackend: bool = False,
    resource_estimation: bool = False,
    shots: int = 1024,
    base_folder: str = "example",
    fixed_circuit: str = "",
    metadata_extra: dict | None = None,
    seed_transpiler: int = 42,
    qpy_filename: str = "qfm_circuit.qpy",
    save_circuit_drawings: bool = False,
):
    """Run or submit a projected feature-map Estimator job."""
    n = int(qc_param.num_qubits)
    if phys_nodes is None:
        phys_nodes = list(range(n))
    if len(phys_nodes) != n:
        raise ValueError(f"phys_nodes must have length {n}, but got {len(phys_nodes)}")

    N = int(theta_values_all.shape[0])
    if theta_values_all.ndim != 2:
        raise ValueError("theta_values_all must be a 2D matrix.")

    Path(base_folder).mkdir(parents=True, exist_ok=True)

    if not simulation:
        if fixed_circuit == "":
            initial_layout = Layout({qc_param.qubits[i]: phys_nodes[i] for i in range(n)})
            qc_t = transpile(
                qc_param,
                backend=backend,
                initial_layout=initial_layout,
                optimization_level=3,
                seed_transpiler=seed_transpiler,
                routing_method="none",
            )
            with open(Path(base_folder) / qpy_filename, "wb") as f:
                qpy.dump(qc_t, f)
        else:
            fixed_path = Path(fixed_circuit)
            if fixed_path.suffix != ".qpy":
                fixed_path = fixed_path.with_suffix(".qpy")
            with open(fixed_path, "rb") as f:
                qc_t = qpy.load(f)[0]
    else:
        if fakebackend:
            initial_layout = Layout({qc_param.qubits[i]: phys_nodes[i] for i in range(n)})
            qc_t = transpile(
                qc_param,
                backend=backend,
                initial_layout=initial_layout,
                optimization_level=3,
                seed_transpiler=seed_transpiler,
            )
        else:
            # Pure ideal simulation: keep the logical circuit.
            # Do not transpile against AerSimulator target.
            qc_t = qc_param.copy()            

    if save_circuit_drawings:
        save_circuit_draw(qc_t, base_folder, "circuit_transpiled")
        save_circuit_draw(qc_param, base_folder, "circuit_logical")

    if not simulation:
        test_circuit_t(
            qc_t,
            backend,
            resource_estimation,
            csv_path=str(Path(base_folder) / "resource_estimation.csv"),
            shots=shots,
            num_param_sets=N,
        )
        if resource_estimation:
            return None

    if not simulation or fakebackend:
        obs_isa = [obs.apply_layout(qc_t.layout) for obs in obs_list]
    else:
        obs_isa = obs_list
    obs_isa_broadcast = [[obs] for obs in obs_isa]

    theta_reordered = reorder_theta_by_parameter_names(qc_t, param_order, theta_values_all)

    if simulation:
        job = estimator.run([(qc_t, obs_isa_broadcast, theta_reordered)])
        result = job.result()[0]
        evs = np.asarray(result.data.evs, dtype=float)
        return evs.T, obs_metadata

    return submit_job_and_save_metadata(
        estimator,
        qc_t,
        obs_isa_broadcast,
        theta_reordered,
        obs_metadata,
        base_folder=base_folder,
        **(metadata_extra or {}),
    )
