"""Observable construction utilities."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

from qiskit.quantum_info import SparsePauliOp


def pauli_string(n: int, terms) -> str:
    """Return an n-qubit Pauli string from pairs (index, axis)."""
    p = ["I"] * int(n)
    for idx, axis in terms:
        p[int(idx)] = str(axis).upper()
    return "".join(p)


def z_observables(n: int, k_max: int = 2):
    """Build Z and ZZ/.../Z^k observables."""
    obs = []
    for i in range(n):
        obs.append(SparsePauliOp.from_list([(pauli_string(n, [(i, "Z")]), 1.0)]))
    pairs_2q = []
    for k in range(2, k_max + 1):
        for idx in combinations(range(n), k):
            obs.append(SparsePauliOp.from_list([(pauli_string(n, [(j, "Z") for j in idx]), 1.0)]))
            if k == 2:
                pairs_2q.append(tuple(idx))
    return obs, pairs_2q


def z_observables_edges(n: int, edges_act):
    """Build Z_i and Z_i Z_j observables on selected edges."""
    obs = []
    for i in range(n):
        obs.append(SparsePauliOp.from_list([(pauli_string(n, [(i, "Z")]), 1.0)]))
    uniq_edges = sorted({(min(i, j), max(i, j)) for i, j in edges_act})
    for i, j in uniq_edges:
        obs.append(SparsePauliOp.from_list([(pauli_string(n, [(i, "Z"), (j, "Z")]), 1.0)]))
    return obs, uniq_edges


def pauli_observables_axis_pairs(
    n: int,
    *,
    axes: Iterable[str] = ("x", "y", "z"),
    edges_act=None,
    measure_cross_observables: bool = False,
):
    """Build one-local and two-local Pauli observables for selected axes."""
    axes = tuple(str(a).lower() for a in axes)
    if edges_act is None:
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        edges = sorted({(min(i, j), max(i, j)) for i, j in edges_act})

    obs = []
    metadata = []

    for a in axes:
        for i in range(n):
            obs.append(SparsePauliOp.from_list([(pauli_string(n, [(i, a)]), 1.0)]))
            metadata.append((a, i))
        for i, j in edges:
            obs.append(SparsePauliOp.from_list([(pauli_string(n, [(i, a), (j, a)]), 1.0)]))
            metadata.append((a + a, i, j))

    if measure_cross_observables:
        for a in axes:
            for b in axes:
                if a == b:
                    continue
                for i, j in edges:
                    obs.append(SparsePauliOp.from_list([(pauli_string(n, [(i, a), (j, b)]), 1.0)]))
                    metadata.append((a + b, i, j))

    return obs, metadata
