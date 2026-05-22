"""Download and save CD-Ising PQFM features from an IBM Runtime job.

Edit BASE_FOLDER and NAME_FILE before running.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService

from pqfmlib.utils.config import Config
from pqfmlib.utils.io import load_json

BASE_FOLDER = "quantum_features_CDIsing_AMI_test_156_shap_k_max2_156_156_ibm_kingston"
NAME_FILE = "AMI_test_156_shap"
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


def main():
    df_full = pd.read_csv(Path(DATA_DIR) / f"{NAME_FILE}.csv")
    y = df_full.iloc[:, -1].values
    downloaded = download_job(BASE_FOLDER)
    if downloaded is None:
        return
    Xq_quantum, meta = downloaded
    q_enc = int(meta.get("q_enc", 0))
    if q_enc <= 0:
        phys_nodes_path = Path(BASE_FOLDER) / "phys_nodes.json"
        q_enc = len(load_json(phys_nodes_path)) if phys_nodes_path.exists() else Xq_quantum.shape[1]
    pairs_2q = [tuple(p) for p in meta.get("pairs_2q", [])]

    q1_names = [f"q1_z_{qi}" for qi in range(q_enc)]
    q2_names = [f"q2_z_{i}_{j}" for i, j in pairs_2q]
    q_col_names = q1_names + q2_names
    if len(q_col_names) != Xq_quantum.shape[1]:
        q_col_names = [f"qfeat_{k}" for k in range(Xq_quantum.shape[1])]
    if len(y) != Xq_quantum.shape[0]:
        raise ValueError("The number of labels does not match the number of returned samples.")

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
