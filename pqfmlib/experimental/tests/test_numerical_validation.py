"""Compare XYZ results from Aer and TileLang on the IBM Kingston layout."""

import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import pqfmlib.maps.xyz as xyz_module
from pqfmlib import XYZProjectiveQFM

RESULTS_DIR = Path("output/xyz_tilelang_validation")
REPORT_PATH = Path("output/xyz_tilelang_validation.md")
TILELANG_SOURCE = Path(__file__).parents[1] / "xyz_tilelang.py"
QUBIT_COUNTS = (6, 12, 18)
INPUT_ROWS = 8
INPUT_FEATURES = 36
HOT_ROWS = 128
HOT_RUNS = 5
ABSOLUTE_TOLERANCE = 0.08
HOT_TOLERANCE = 1e-6

QFM_OPTIONS = {
    "name_file": "toy_xyz",
    "seed": 42,
    "ideal": False,
    "simulation": True,
    "fakebackend": False,
    "shots": 4096,
    "ibm_qpu": "ibm_kingston",
    "features_per_qubit": 1,
    "axes": ("x", "y", "z"),
    "encoding_mode": "shared_feature",
    "keep_diagonal_terms": True,
    "keep_cross_terms": False,
    "measure_all_zz": True,
    "m": 1,
    "tau": 1.0,
    "rho_thr": 0.0,
}


def _write_toy_dataset(data_dir: Path) -> np.ndarray:
    rng = np.random.default_rng(42)
    latent = rng.uniform(-0.8, 0.8, size=(INPUT_ROWS, 1))
    weights = np.linspace(0.65, 1.35, INPUT_FEATURES)[None, :]
    features = np.clip(
        latent * weights
        + rng.normal(0.0, 0.025, size=(INPUT_ROWS, INPUT_FEATURES)),
        -1.0,
        1.0,
    )
    frame = pd.DataFrame(
        features,
        columns=[f"x{index}" for index in range(INPUT_FEATURES)],
    )
    frame["y"] = (latent[:, 0] > 0.0).astype(int)
    data_dir.mkdir(parents=True)
    frame.to_csv(data_dir / "toy_xyz.csv", index=False)
    return features


def _qfm_options(data_dir: Path, output_root: Path, qubits: int) -> dict:
    return QFM_OPTIONS | {
        "data_dir": str(data_dir),
        "output_root": str(output_root),
        "q_enc": qubits,
    }


def _run_hot(study, qfm, features: np.ndarray) -> tuple[np.ndarray, float]:
    theta, _ = xyz_module.make_theta_matrix_full_cross_blocks(
        features,
        qfm.blocks,
        qfm.edges_log,
        features_per_qubit=len(qfm.axes),
        axes=qfm.axes,
        encoding_mode=qfm.encoding_mode,
        tau=qfm.tau,
        m_phys=qfm.m,
        keep_diagonal_terms=qfm.keep_diagonal_terms,
        keep_cross_terms=qfm.keep_cross_terms,
    )
    hot_theta = np.tile(theta, (HOT_ROWS // len(theta), 1))
    elapsed = []
    for _ in range(HOT_RUNS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        hot_features = study.TILELANG_BACKEND.run(
            hot_theta,
            n_qubits=qfm.q_enc,
            axes=qfm.axes,
            evolution_edges=qfm.edges_act,
            observable_edges=None,
        )
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)
    return hot_features, float(np.median(elapsed))


def _run_backends(tmp_path: Path, qubits: int):
    data_dir = tmp_path / "data"
    features = _write_toy_dataset(data_dir)

    original_executor = xyz_module.run_projected_feature_job
    started = time.perf_counter()
    aer_features = XYZProjectiveQFM(
        **_qfm_options(data_dir, tmp_path / "aer", qubits)
    ).run()["Xq_all_raw"]
    aer_seconds = time.perf_counter() - started

    # Run Aer first. Then install the TileLang function.
    study = import_module("qimed.nested_v2")
    xyz_module.run_projected_feature_job = study.run_xyz_tilelang
    original_qfm = study.XYZProjectiveQFM
    tilelang_qfm = None

    def capture_qfm(**options):
        nonlocal tilelang_qfm
        tilelang_qfm = original_qfm(**options)
        return tilelang_qfm

    study.XYZProjectiveQFM = capture_qfm
    try:
        started = time.perf_counter()
        tilelang_features = study.project_xyz(
            qubits,
            features,
            len(features),
            tmp_path / "tilelang" / f"q{qubits}",
        )
        tilelang_seconds = time.perf_counter() - started
        hot_features, hot_seconds = _run_hot(study, tilelang_qfm, features)
    finally:
        study.XYZProjectiveQFM = original_qfm
        xyz_module.run_projected_feature_job = original_executor

    times = {
        "aer_end_to_end_seconds": aer_seconds,
        "tilelang_end_to_end_seconds": tilelang_seconds,
        "tilelang_hot_seconds_median": hot_seconds,
        "tilelang_hot_rows_per_second": HOT_ROWS / hot_seconds,
    }
    return aer_features, tilelang_features, hot_features, times


def _save_results(
    qubits: int,
    aer_features: np.ndarray,
    tilelang_features: np.ndarray,
    hot_features: np.ndarray,
    times: dict,
) -> None:
    difference = np.abs(tilelang_features - aer_features)
    rows, output_count = tilelang_features.shape
    metrics = {
        "status": "PASS",
        "qubits": qubits,
        "rows": rows,
        "outputs": output_count,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "hot_absolute_tolerance": HOT_TOLERANCE,
        "hot_rows": HOT_ROWS,
        "hot_runs": HOT_RUNS,
        **times,
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / f"q{qubits:02d}_aer.npy", aer_features)
    np.save(RESULTS_DIR / f"q{qubits:02d}_tilelang.npy", tilelang_features)
    np.save(RESULTS_DIR / f"q{qubits:02d}_tilelang_hot.npy", hot_features)
    (RESULTS_DIR / f"q{qubits:02d}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RESULTS_DIR.glob("q*_metrics.json"))
    ]
    records = [item for item in records if "tilelang_hot_seconds_median" in item]
    source_lines = len(TILELANG_SOURCE.read_text(encoding="utf-8").splitlines())
    lines = [
        "# XYZ numerical comparison: Aer and TileLang",
        "",
        "## Validation status",
        "",
        "All table rows passed the shape, hot-output, and Aer-comparison "
        "assertions before the test saved their artifacts.",
        "",
        "## Validated implementation",
        "",
        f"- `xyz_tilelang.py` has {source_lines:,} lines. The cleanup baseline "
        f"was 1,757 lines ({1_757 - source_lines:,} lines removed).",
        "- Qubits 1 through 12 use the small-state kernel.",
        "- Qubit count 18 uses a specialized 9+9 cut with Schmidt rank 8.",
        "- Its first three crossing gates expand exactly from rank 1 to rank 8. "
        "The remaining crossing gates use structured rank-8 updates.",
        "- Qubit count 18 measures each 4,096-value tile with two 64 by 64 "
        "Tensor Core transforms and retains only the requested coefficients.",
        "- The generic q13+ fallback and the unused BigBang kernels were removed.",
        "- Qubit counts 13 through 17 and 19 or greater are not supported.",
        "",
        "## Experiment",
        "",
        "The test uses the IBM Kingston layout. Aer uses 4,096 shots. Each "
        "input has 36 features. The XYZ map uses shared features and a "
        "diagonal Hamiltonian. The test measures all same-axis observables.",
        "",
        f"The Aer comparison uses an absolute tolerance of "
        f"{ABSOLUTE_TOLERANCE:.2f}. The repeated hot output uses an absolute "
        f"tolerance of {HOT_TOLERANCE:g}.",
        "",
        "## Timing",
        "",
        "The end-to-end times include setup. The hot time does not include "
        "setup or compilation. It is the median time for five runs of 128 rows. "
        "The hot batch repeats the eight validated input rows 16 times.",
        "",
        "## Results",
        "",
        "| Status | Qubits | Hot rows | Hot median (ms) | Hot rows/s | Rows | Outputs | "
        "Aer end-to-end (s) | TileLang end-to-end (s) | "
        "Max difference | Mean difference | RMSE |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in records:
        lines.append(
            f"| {item.get('status', 'PASS')} | {item['qubits']} | {HOT_ROWS} | "
            f"{1000 * item['tilelang_hot_seconds_median']:.3f} | "
            f"{item['tilelang_hot_rows_per_second']:.1f} | "
            f"{item['rows']} | {item['outputs']} | "
            f"{item['aer_end_to_end_seconds']:.3f} | "
            f"{item['tilelang_end_to_end_seconds']:.3f} | "
            f"{item['max_absolute_error']:.6f} | "
            f"{item['mean_absolute_error']:.6f} | {item['rmse']:.6f} |"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        "\n"
        f"{' XYZ q' + str(qubits) + ': Aer and TileLang ':=^74}\n"
        f"  Input rows        {rows:>10}\n"
        f"  Output features   {output_count:>10}\n"
        f"  Aer end-to-end    {times['aer_end_to_end_seconds']:>10.3f} s\n"
        f"  TileLang end-to-end{times['tilelang_end_to_end_seconds']:>9.3f} s\n"
        f"  TileLang hot median{times['tilelang_hot_seconds_median']:>9.6f} s\n"
        f"  Hot rows/s        {times['tilelang_hot_rows_per_second']:>10.1f}\n"
        f"  Maximum difference{metrics['max_absolute_error']:>9.6f}\n"
        f"  Mean difference   {metrics['mean_absolute_error']:>10.6f}\n"
        f"  RMSE              {metrics['rmse']:>10.6f}\n"
        f"  Report            {str(REPORT_PATH):>10}\n"
        f"{'=' * 74}"
    )


@pytest.mark.parametrize("qubits", QUBIT_COUNTS)
def test_nested_v2_tilelang_matches_local_kingston(tmp_path: Path, qubits: int):
    aer, tilelang, hot, times = _run_backends(tmp_path, qubits)

    expected_outputs = 3 * qubits * (qubits + 1) // 2
    assert aer.shape == tilelang.shape == (INPUT_ROWS, expected_outputs)
    expected_hot = np.tile(tilelang, (HOT_ROWS // INPUT_ROWS, 1))
    np.testing.assert_allclose(hot, expected_hot, rtol=0.0, atol=HOT_TOLERANCE)
    np.testing.assert_allclose(
        tilelang,
        aer,
        rtol=0.0,
        atol=ABSOLUTE_TOLERANCE,
    )
    _save_results(qubits, aer, tilelang, hot, times)
