"""Circuit resource-estimation helpers."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone


def backend_dt_seconds(backend) -> float:
    dt = None
    if getattr(backend, "target", None) is not None:
        dt = getattr(backend.target, "dt", None)
    if dt is None and hasattr(backend, "configuration"):
        dt = getattr(backend.configuration(), "dt", None)
    if dt is None:
        raise ValueError("Could not obtain backend dt.")
    return float(dt)


def estimate_circuit_length_s(qc_t, backend, dt_s: float) -> float:
    """Estimate circuit duration without submitting a job."""
    if hasattr(qc_t, "estimate_duration"):
        try:
            return float(qc_t.estimate_duration(target=backend.target, unit="s"))
        except TypeError:
            dur = qc_t.estimate_duration(target=getattr(backend, "target", None))
            return float(dur) * dt_s

    try:
        from qiskit.transpiler import InstructionDurations, PassManager
        from qiskit.transpiler.passes.scheduling import ASAPScheduleAnalysis

        durations = InstructionDurations.from_backend(backend)
        pm_sched = PassManager([ASAPScheduleAnalysis(durations)])
        qc_sched = pm_sched.run(qc_t)
        dur_ticks = getattr(qc_sched, "duration", None)
        if dur_ticks is None:
            raise RuntimeError("Scheduling pass did not produce qc.duration.")
        return float(dur_ticks) * dt_s
    except Exception as exc:
        raise RuntimeError("Failed to estimate circuit duration.") from exc


def test_circuit_t(
    qc_t,
    backend,
    resource_estimation: bool,
    *,
    csv_path: str = "resource_estimation.csv",
    shots: int = 1000,
    num_param_sets: int = 1,
    num_circuits: int = 1,
    rep_delay_s: float | None = None,
    overhead_s: float = 2.0,
):
    """Print resource metrics and optionally append a resource-estimation CSV row."""
    depth = int(qc_t.depth())
    dt_s = backend_dt_seconds(backend)
    ops = qc_t.count_ops()
    swap_count = int(ops.get("swap", 0))
    d2 = int(qc_t.depth(filter_function=lambda inst: inst.operation.num_qubits == 2))
    d1 = int(qc_t.depth(filter_function=lambda inst: inst.operation.num_qubits == 1))

    twoq = Counter()
    for ci in qc_t.data:
        if ci.operation.num_qubits == 2:
            twoq[ci.operation.name] += 1
    total_2q = int(sum(twoq.values()))

    print("Depth:", depth)
    print("dt (s):", dt_s)
    print("SWAP =", swap_count)
    print("2q-depth =", d2)
    print("1q-depth =", d1)
    print("2q gate types:", twoq)
    print("Total 2q gates =", total_2q)

    if not resource_estimation:
        return

    circuit_length_s = estimate_circuit_length_s(qc_t, backend, dt_s)
    if rep_delay_s is None:
        rep_delay_s = float(getattr(backend, "default_rep_delay", 250e-6))
    num_executions = int(shots) * int(num_param_sets) * int(num_circuits)
    estimated_qpu_time_s = float(overhead_s) + (float(rep_delay_s) + float(circuit_length_s)) * num_executions

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "depth": depth,
        "dt_s": dt_s,
        "swap": swap_count,
        "depth_2q": d2,
        "depth_1q": d1,
        "twoq_gate_types": json.dumps(dict(twoq), ensure_ascii=False),
        "total_2q_gates": total_2q,
        "circuit_length_s": float(circuit_length_s),
        "shots": int(shots),
        "num_param_sets": int(num_param_sets),
        "num_circuits": int(num_circuits),
        "num_executions": int(num_executions),
        "rep_delay_s": float(rep_delay_s),
        "overhead_s": float(overhead_s),
        "estimated_qpu_time_s": float(estimated_qpu_time_s),
    }

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[resource_estimation] circuit_length ≈ {circuit_length_s:.6e} s")
    print(f"[resource_estimation] estimated_qpu_time ≈ {estimated_qpu_time_s:.3f} s")
    print(f"[resource_estimation] CSV append: {csv_path}")
