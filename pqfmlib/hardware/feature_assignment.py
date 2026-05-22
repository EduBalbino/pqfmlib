"""Genetic algorithms for hardware-aware feature assignment.

The public functions intentionally keep the ``GA`` prefix because this module is
meant to preserve the naming convention used in the original experimental code.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

Edge = Tuple[int, int]


def GA_fitness_perm(
    perm: List[int],
    edges: List[Edge],
    J: np.ndarray,
    edge_weight: Optional[Dict[Edge, float]] = None,
) -> float:
    """Evaluate a permutation by summing feature correlations over available edges."""
    score = 0.0
    for i, j in edges:
        w = 1.0 if edge_weight is None else edge_weight.get((i, j), edge_weight.get((j, i), 1.0))
        score += w * J[perm[i], perm[j]]
    return float(score)


def GA_tournament_select(pop: List[List[int]], fit: List[float], k: int) -> List[int]:
    """Select one parent by tournament selection."""
    idxs = random.sample(range(len(pop)), k)
    best = max(idxs, key=lambda t: fit[t])
    return pop[best]


def GA_ox_crossover(p1: List[int], p2: List[int]) -> List[int]:
    """Order crossover for permutations."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b] = p1[a:b]
    fill = [x for x in p2 if x not in child]
    ptr = 0
    for i in list(range(0, a)) + list(range(b, n)):
        child[i] = fill[ptr]
        ptr += 1
    return child


def GA_mutate_swap(p: List[int], pmut: float) -> None:
    """Swap two entries with probability pmut."""
    if random.random() < pmut:
        i, j = random.sample(range(len(p)), 2)
        p[i], p[j] = p[j], p[i]


def GA_mutate_scramble(p: List[int], pmut: float) -> None:
    """Scramble a random segment with probability pmut."""
    if random.random() < pmut:
        n = len(p)
        a, b = sorted(random.sample(range(n), 2))
        seg = p[a:b]
        random.shuffle(seg)
        p[a:b] = seg


def GA_assignment(
    J: np.ndarray,
    edges: List[Edge],
    edge_weight: Optional[Dict[Edge, float]] = None,
    pop_size: int = 80,
    ngen: int = 300,
    k_tourn: int = 3,
    elite_k: int = 5,
    pmut_init: float = 0.15,
    pmut_min: float = 0.05,
    pmut_max: float = 0.40,
    patience: int = 25,
    seed: int = 0,
):
    """Assign features to qubit slots using a permutation genetic algorithm."""
    random.seed(seed)
    np.random.seed(seed)

    n = int(J.shape[0])
    if n <= 1:
        raise ValueError("J must have at least two rows/columns.")
    if elite_k >= pop_size:
        raise ValueError("elite_k must be smaller than pop_size.")

    pop = [random.sample(range(n), n) for _ in range(pop_size)]
    pmut = pmut_init
    best_fit = -1e18
    best_perm = None
    no_improve = 0

    for g in range(ngen):
        fit = [GA_fitness_perm(p, edges, J, edge_weight) for p in pop]
        order = np.argsort(fit)[::-1]
        fit_sorted = [fit[i] for i in order]
        pop_sorted = [pop[i] for i in order]

        if fit_sorted[0] > best_fit + 1e-12:
            best_fit = fit_sorted[0]
            best_perm = pop_sorted[0][:]
            no_improve = 0
            pmut = max(pmut_min, pmut * 0.98)
        else:
            no_improve += 1

        if no_improve >= patience:
            pmut = min(pmut_max, pmut * 1.25)
            no_improve = 0

        new_pop = [pop_sorted[i][:] for i in range(elite_k)]
        while len(new_pop) < pop_size:
            p1 = GA_tournament_select(pop, fit, k_tourn)
            p2 = GA_tournament_select(pop, fit, k_tourn)
            c1 = GA_ox_crossover(p1, p2)
            c2 = GA_ox_crossover(p2, p1)
            GA_mutate_swap(c1, pmut)
            GA_mutate_swap(c2, pmut)
            GA_mutate_scramble(c1, pmut * 0.25)
            GA_mutate_scramble(c2, pmut * 0.25)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        if (g + 1) % 80 == 0:
            n_replace = max(1, pop_size // 20)
            for t in range(n_replace):
                new_pop[-1 - t] = random.sample(range(n), n)

        pop = new_pop

    return best_perm, best_fit


def pick_connected_subset_greedy(n_target, coupling_edges, edge_cost):
    """Pick a connected physical subgraph using a greedy low-error growth rule."""
    adj = {}
    for u, v in coupling_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    nodes = list(adj.keys())

    def node_score(u):
        costs = [edge_cost.get((u, v), edge_cost.get((v, u), 1.0)) for v in adj[u]]
        return np.mean(costs) if costs else 1e9

    seed = min(nodes, key=lambda u: (node_score(u), u))
    chosen = {seed}
    frontier = set(adj[seed])

    while len(chosen) < n_target:
        best_cand = None
        best_score = 1e18
        for cand in sorted(frontier):
            scores = []
            for u in sorted(chosen):
                if cand in adj.get(u, []):
                    scores.append(edge_cost.get((u, cand), edge_cost.get((cand, u), 1.0)))
            if scores:
                s = min(scores)
                if s < best_score:
                    best_score = s
                    best_cand = cand
        if best_cand is None:
            raise RuntimeError("Could not grow a connected subgraph with the requested size.")
        chosen.add(best_cand)
        frontier.remove(best_cand)
        for nb in adj[best_cand]:
            if nb not in chosen:
                frontier.add(nb)
    return sorted(chosen)


AXES_ALL = ("x", "y", "z")


def _normalize_axes_ga(axes=None, features_per_qubit: int = 3):
    if axes is None:
        if features_per_qubit not in (1, 2, 3):
            raise ValueError("features_per_qubit must be 1, 2, or 3.")
        return AXES_ALL[:features_per_qubit]
    axes = tuple(str(a).lower() for a in axes)
    invalid = [a for a in axes if a not in AXES_ALL]
    if invalid:
        raise ValueError(f"Invalid axes: {invalid}")
    if len(axes) == 0 or len(axes) > 3 or len(set(axes)) != len(axes):
        raise ValueError("axes must contain 1 to 3 unique axes.")
    return axes


def _axis_slots(q_enc: int, axes):
    return [(i, a) for i in range(q_enc) for a in tuple(axes)]


def _axis_slot_index(q_enc: int, axes):
    return {slot: idx for idx, slot in enumerate(_axis_slots(q_enc, axes))}


def build_multiaxis_slot_edges(
    q_enc: int,
    edges_log: List[Edge],
    axes=None,
    features_per_qubit: int = 3,
    keep_diagonal_terms: bool = True,
    keep_cross_terms: bool = True,
):
    """Expand qubit-qubit edges into slot-slot edges over selected axes."""
    axes = _normalize_axes_ga(axes, features_per_qubit)
    slot_idx = _axis_slot_index(q_enc, axes)
    slot_edges = []
    unique_edges = sorted({(min(i, j), max(i, j)) for i, j in edges_log})
    for i, j in unique_edges:
        for a in axes:
            for b in axes:
                if a == b and not keep_diagonal_terms:
                    continue
                if a != b and not keep_cross_terms:
                    continue
                slot_edges.append((slot_idx[(i, a)], slot_idx[(j, b)]))
    return slot_edges


def build_multiaxis_slot_edge_weight(
    q_enc: int,
    edges_log: List[Edge],
    edge_weight: Optional[Dict[Edge, float]],
    axes=None,
    features_per_qubit: int = 3,
    keep_diagonal_terms: bool = True,
    keep_cross_terms: bool = True,
):
    """Expand qubit-edge weights to slot-edge weights."""
    if edge_weight is None:
        return None
    axes = _normalize_axes_ga(axes, features_per_qubit)
    slot_idx = _axis_slot_index(q_enc, axes)
    slot_edge_weight = {}
    unique_edges = sorted({(min(i, j), max(i, j)) for i, j in edges_log})
    for i, j in unique_edges:
        w = float(edge_weight.get((i, j), edge_weight.get((j, i), 1.0)))
        for a in axes:
            for b in axes:
                if a == b and not keep_diagonal_terms:
                    continue
                if a != b and not keep_cross_terms:
                    continue
                slot_edge_weight[(slot_idx[(i, a)], slot_idx[(j, b)])] = w
    return slot_edge_weight


def _build_padded_J_for_feat_set(J_global: np.ndarray, feat_set, capacity: int):
    feat_list = [int(f) for f in feat_set]
    if len(feat_list) > capacity:
        scores = [(f, float(np.sum(np.abs(J_global[f, feat_list])))) for f in feat_list]
        scores.sort(key=lambda t: (-t[1], t[0]))
        feat_list = [f for f, _ in scores[:capacity]]
    padded_feat_ids = feat_list + [-1] * (capacity - len(feat_list))
    J_local = np.zeros((capacity, capacity), dtype=float)
    for a in range(capacity):
        fa = padded_feat_ids[a]
        if fa < 0:
            continue
        for b in range(a + 1, capacity):
            fb = padded_feat_ids[b]
            if fb < 0:
                continue
            J_local[a, b] = float(J_global[fa, fb])
            J_local[b, a] = J_local[a, b]
    return padded_feat_ids, J_local


def perm_to_feat_ids_by_axis(perm: List[int], padded_feat_ids: List[int], q_enc: int, axes=None, features_per_qubit: int = 3):
    """Convert a slot permutation into a dictionary axis -> feature ids per qubit."""
    axes = _normalize_axes_ga(axes, features_per_qubit)
    slots = _axis_slots(q_enc, axes)
    feat_ids_by_axis = {axis: [-1] * q_enc for axis in axes}
    for slot_idx, (qubit, axis) in enumerate(slots):
        local_feature_idx = int(perm[slot_idx])
        feat_ids_by_axis[axis][qubit] = int(padded_feat_ids[local_feature_idx])
    return feat_ids_by_axis


def GA_assignment_multiaxis(
    J_global: np.ndarray,
    feat_set,
    q_enc: int,
    edges_log: List[Edge],
    axes=None,
    features_per_qubit: int = 3,
    edge_weight: Optional[Dict[Edge, float]] = None,
    keep_diagonal_terms: bool = True,
    keep_cross_terms: bool = True,
    pop_size: int = 80,
    ngen: int = 300,
    k_tourn: int = 3,
    elite_k: int = 5,
    pmut_init: float = 0.15,
    pmut_min: float = 0.05,
    pmut_max: float = 0.40,
    patience: int = 25,
    seed: int = 0,
):
    """Assign features to multi-axis slots (qubit, axis) with a GA."""
    axes = _normalize_axes_ga(axes, features_per_qubit)
    capacity = int(q_enc) * len(axes)
    padded_feat_ids, J_local = _build_padded_J_for_feat_set(J_global, feat_set, capacity)
    slot_edges = build_multiaxis_slot_edges(
        q_enc=q_enc,
        edges_log=edges_log,
        axes=axes,
        features_per_qubit=features_per_qubit,
        keep_diagonal_terms=keep_diagonal_terms,
        keep_cross_terms=keep_cross_terms,
    )
    slot_edge_weight = build_multiaxis_slot_edge_weight(
        q_enc=q_enc,
        edges_log=edges_log,
        edge_weight=edge_weight,
        axes=axes,
        features_per_qubit=features_per_qubit,
        keep_diagonal_terms=keep_diagonal_terms,
        keep_cross_terms=keep_cross_terms,
    )
    best_perm, best_fit = GA_assignment(
        J_local,
        slot_edges,
        edge_weight=slot_edge_weight,
        pop_size=pop_size,
        ngen=ngen,
        k_tourn=k_tourn,
        elite_k=elite_k,
        pmut_init=pmut_init,
        pmut_min=pmut_min,
        pmut_max=pmut_max,
        patience=patience,
        seed=seed,
    )
    feat_ids_by_axis = perm_to_feat_ids_by_axis(
        best_perm,
        padded_feat_ids,
        q_enc=q_enc,
        axes=axes,
        features_per_qubit=features_per_qubit,
    )
    return feat_ids_by_axis, best_fit, best_perm, padded_feat_ids


# Backward-compatible lowercase aliases.
ga_assignment = GA_assignment
ga_assignment_multiaxis = GA_assignment_multiaxis
