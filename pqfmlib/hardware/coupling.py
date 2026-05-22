"""Coupling-map and hardware-error utilities."""

from __future__ import annotations


def get_coupling_edges_and_costs(backend):
    """Return coupling edges and two-qubit gate error costs from a backend."""
    coupling_edges = backend.coupling_map.get_edges()
    target = backend.target
    twoq_gate = None
    for gname in ["ecr", "cz", "cx"]:
        if gname in target.operation_names:
            twoq_gate = gname
            break
    if twoq_gate is None:
        raise RuntimeError("Could not find ecr/cz/cx in backend.target.operation_names.")

    edge_cost = {}
    for u, v in coupling_edges:
        props = target[twoq_gate].get((u, v)) or target[twoq_gate].get((v, u))
        edge_cost[(u, v)] = float(props.error) if props is not None and props.error is not None else 1.0
    return coupling_edges, edge_cost
