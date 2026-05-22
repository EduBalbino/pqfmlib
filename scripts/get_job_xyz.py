"""Download and save XYZ PQFM features from an IBM Runtime job.

Edit BASE_FOLDER and NAME_FILE before running.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService

from pqfmlib.core.serialization import metadata_to_column_name, normalize_metadata
from pqfmlib.utils.config import Config
from pqfmlib.utils.io import load_json

BASE_FOLDER = "quantum_features_XYZ_ISPY1_preprocessed_fperq3_axesXYZ_multi_axis_370_124_ibm_kingston_cross"
NAME_FILE = "ISPY1_preprocessed"
DATA_DIR = "./data"


def job_is_done(status) -> bool:
    status_name = getattr(status, "name", str(status))
    return str(status_name).upper() in ("DONE", "COMPLETED")


def fallback_column_names(q_enc, n_q_feats, axes=None, pairs_2q=None):
    axes = list(axes or ["x", "y", "z"])
    q1_names = [f"q1_{axis}_{qi}" for axis in axes for qi in range(q_enc)]
    n2 = n_q_feats - len(q1_names)
    if n2 < 0:
        return [f"qfeat_{k}" for k in range(n_q_feats)]
    q2_names = []
    if pairs_2q:
        diagonal_pairs = [f"{a}{a}" for a in axes]
        full_pairs = [f"{a}{b}" for a in axes for b in axes]
        if n2 == len(pairs_2q) * len(diagonal_pairs):
            axis_pairs = diagonal_pairs
        elif n2 == len(pairs_2q) * len(full_pairs):
            axis_pairs = full_pairs
        else:
            axis_pairs = []
        for axis_pair in axis_pairs:
            for i, j in pairs_2q:
                q2_names.append(f"q2_{axis_pair}_{i}_{j}")
    names = q1_names + q2_names
    return names if len(names) == n_q_feats else [f"qfeat_{k}" for k in range(n_q_feats)]


def download_job(base_folder=BASE_FOLDER):
    meta = load_json(Path(base_folder) / "job_meta.json")
    service = QiskitRuntimeService(channel="ibm_cloud", token=Config().QXToken)
    job = service.job(meta["job_id"])
    status = job.status()
    print("Status:", status)
    if not job_is_done(status):
        print("The job has not finished yet.")
        return None
    result = job.result()[0]
    Xq_quantum = np.asarray(result.data.evs, dtype=float).T
    return Xq_quantum, meta


def main():
    df_full = pd.read_csv(Path(DATA_DIR) / f"{NAME_FILE}.csv")
    y = df_full.iloc[:, -1].values
    downloaded = download_job(BASE_FOLDER)
    if downloaded is None:
        return
    Xq_quantum, meta = downloaded
    if len(y) != Xq_quantum.shape[0]:
        raise ValueError("The number of labels does not match the number of returned samples.")

    obs_metadata = normalize_metadata(meta.get("obs_metadata"))
    if obs_metadata:
        q_col_names = [metadata_to_column_name(m) for m in obs_metadata]
    else:
        q_enc = int(meta.get("q_enc", Xq_quantum.shape[1]))
        axes = meta.get("axes", ["x", "y", "z"])
        pairs_2q = meta.get("pairs_2q", [])
        q_col_names = fallback_column_names(q_enc, Xq_quantum.shape[1], axes=axes, pairs_2q=pairs_2q)

    if len(q_col_names) != Xq_quantum.shape[1]:
        raise ValueError("Column-name count does not match the number of quantum features.")

    df_q_all = pd.DataFrame(Xq_quantum, columns=q_col_names)
    df_q_all["y"] = y
    csv_path = Path(BASE_FOLDER) / f"qfeatures_all_{NAME_FILE}.csv"
    npy_path = Path(BASE_FOLDER) / f"qfeatures_all_{NAME_FILE}.npy"
    df_q_all.to_csv(csv_path, index=False)
    np.save(npy_path, Xq_quantum)
    print("Features saved to:", csv_path)
    print("NPY saved to:", npy_path)


if __name__ == "__main__":
    main()
