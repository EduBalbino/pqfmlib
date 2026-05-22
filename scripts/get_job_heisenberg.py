"""Download and save Heisenberg PQFM features from an IBM Runtime job.

Edit BASE_FOLDER and NAME_FILE before running.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService

from pqfmlib.core.serialization import metadata_to_column_name, normalize_metadata
from pqfmlib.utils.config import Config
from pqfmlib.utils.io import load_json

BASE_FOLDER = "heisenberg__R2_quantum_features_ISPY1_preprocessed_370_124_ibm_kingston"
NAME_FILE = "ISPY1_preprocessed"
DATA_DIR = "./data"


def job_is_done(status) -> bool:
    status_name = getattr(status, "name", str(status))
    return str(status_name).upper() in ("DONE", "COMPLETED")


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
    return np.asarray(result.data.evs, dtype=float).T, meta


def fallback_heisenberg_names(n_q_feats, axes):
    axes = list(axes or ["z", "x", "y"])
    if n_q_feats % len(axes) != 0:
        return [f"qfeat_{k}" for k in range(n_q_feats)]
    n_qubits = n_q_feats // len(axes)
    return [f"q1_{axis}_{qi}" for axis in axes for qi in range(n_qubits)]


def main():
    df_full = pd.read_csv(Path(DATA_DIR) / f"{NAME_FILE}.csv")
    y = df_full.iloc[:, -1].values
    downloaded = download_job(BASE_FOLDER)
    if downloaded is None:
        return
    Xq_quantum, meta = downloaded
    obs_metadata = normalize_metadata(meta.get("obs_metadata"))
    if obs_metadata:
        q_col_names = [metadata_to_column_name(m) for m in obs_metadata]
    else:
        q_col_names = fallback_heisenberg_names(Xq_quantum.shape[1], meta.get("axes", ["z", "x", "y"]))
    if len(y) != Xq_quantum.shape[0]:
        raise ValueError("The number of labels does not match the number of returned samples.")

    df_q_all = pd.DataFrame(Xq_quantum, columns=q_col_names)
    df_q_all["y"] = y
    csv_path = Path(BASE_FOLDER) / f"qfeatures_all_{NAME_FILE}_heisenberg.csv"
    npy_path = Path(BASE_FOLDER) / f"qfeatures_all_{NAME_FILE}_heisenberg.npy"
    df_q_all.to_csv(csv_path, index=False)
    np.save(npy_path, Xq_quantum)
    print("Features saved to:", csv_path)
    print("NPY saved to:", npy_path)


if __name__ == "__main__":
    main()
