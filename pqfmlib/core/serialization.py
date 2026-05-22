"""Serialization helpers for blocks and observable metadata."""

from __future__ import annotations


def metadata_to_jsonable(metadata):
    return [list(x) if isinstance(x, tuple) else x for x in metadata]


def normalize_metadata(obs_metadata):
    if obs_metadata is None:
        return []
    out = []
    for item in obs_metadata:
        if isinstance(item, tuple):
            out.append(item)
        elif isinstance(item, list):
            out.append(tuple(item))
        else:
            raise ValueError(f"Unexpected observable metadata item: {item}")
    return out


def metadata_to_column_name(metadata) -> str:
    if len(metadata) == 2:
        axis, i = metadata
        return f"q1_{axis}_{i}"
    if len(metadata) == 3:
        axis_pair, i, j = metadata
        return f"q2_{axis_pair}_{i}_{j}"
    return "q_" + "_".join(map(str, metadata))


def blocks_to_jsonable(blocks):
    out = []
    for b in blocks:
        if "feat_ids_by_axis" in b:
            j_terms = []
            for key, val in b["J_terms"].items():
                a, bb, i, j = key
                j_terms.append({
                    "axis_i": str(a),
                    "axis_j": str(bb),
                    "i": int(i),
                    "j": int(j),
                    "valor": float(val),
                })
            out.append({
                "feat_ids_by_axis": {
                    str(axis): [int(x) if x is not None else -1 for x in ids]
                    for axis, ids in b["feat_ids_by_axis"].items()
                },
                "J_terms": j_terms,
            })
        else:
            j_list = [
                {"i": int(i), "j": int(j), "valor": float(v)}
                for (i, j), v in b["J_dict_layer"].items()
            ]
            out.append({
                "feat_ids": [int(x) if x is not None else -1 for x in b["feat_ids"]],
                "J_dict_layer": j_list,
            })
    return out


def blocks_from_jsonable(blocks_in):
    blocks = []
    for b in blocks_in:
        if "feat_ids_by_axis" in b:
            feat_ids_by_axis = {
                str(axis): [int(x) for x in ids]
                for axis, ids in b["feat_ids_by_axis"].items()
            }
            J_terms = {}
            for d in b["J_terms"]:
                key = (str(d["axis_i"]), str(d["axis_j"]), int(d["i"]), int(d["j"]))
                J_terms[key] = float(d["valor"])
            blocks.append({"feat_ids_by_axis": feat_ids_by_axis, "J_terms": J_terms})
        else:
            j_dict = {(int(d["i"]), int(d["j"])): float(d["valor"]) for d in b["J_dict_layer"]}
            feat_ids = [int(x) for x in b["feat_ids"]]
            blocks.append({"feat_ids": feat_ids, "J_dict_layer": j_dict})
    return blocks
