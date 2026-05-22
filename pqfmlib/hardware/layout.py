"""Physical-layout utilities."""

from __future__ import annotations


def edges_log_from_phys_nodes(coupling_edges, phys_nodes, edge_cost=None):
    """Convert physical coupling edges into local logical edges for a selected subgraph."""
    phys_set = set(phys_nodes)
    phys_to_log = {p: i for i, p in enumerate(phys_nodes)}
    edge_weight = {}
    edges_log = []
    for u, v in coupling_edges:
        if u in phys_set and v in phys_set:
            i, j = phys_to_log[u], phys_to_log[v]
            key = (min(i, j), max(i, j))
            edges_log.append(key)
            if edge_cost is not None:
                err = edge_cost.get((u, v), edge_cost.get((v, u), 1.0))
                edge_weight[key] = max(1e-6, 1.0 - err)
    return sorted(set(edges_log)), edge_weight


def validate_phys_nodes(phys_nodes, q_enc, coupling_edges=None, backend_name="backend"):
    """Validate length, uniqueness, availability, and optional connectivity of physical nodes."""
    if len(phys_nodes) != q_enc:
        raise ValueError(f"phys_nodes has {len(phys_nodes)} entries, but q_enc={q_enc}.")
    if len(set(phys_nodes)) != len(phys_nodes):
        raise ValueError("phys_nodes contains repeated physical qubits.")
    if coupling_edges is None:
        return

    physical_nodes_available = set()
    for u, v in coupling_edges:
        physical_nodes_available.add(u)
        physical_nodes_available.add(v)
    invalid = [p for p in phys_nodes if p not in physical_nodes_available]
    if invalid:
        raise ValueError(f"phys_nodes contains qubits not present in {backend_name}: {invalid}")

    phys_set = set(phys_nodes)
    adj = {p: set() for p in phys_nodes}
    for u, v in coupling_edges:
        if u in phys_set and v in phys_set:
            adj[u].add(v)
            adj[v].add(u)

    start = phys_nodes[0]
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    if len(seen) != len(phys_nodes):
        raise ValueError(f"phys_nodes does not form a connected subgraph on {backend_name}.")
