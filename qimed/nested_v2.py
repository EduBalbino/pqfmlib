#!/usr/bin/env python3
"""Leakage-free nested temporal selection for the QIMED PQFM v2 study."""

from collections import defaultdict
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from pqfmlib import XYZProjectiveQFM
import pqfmlib.maps.xyz as xyz_module
from pqfmlib.experimental.xyz_tilelang import XYZTileLangBackend

from qimed.plots import render_results


########## Dataset schema ##########

## Notebook-derived dataset schema: QIMED_testes.ipynb, cell 71.
FEATURE_COLS = """
age_at_enc gender race deceased marital_status class_code enc_type_grp has_reason
n_enc_total n_enc_30d n_enc_90d n_enc_365d days_since_last had_emer_90d
had_imp_90d month quarter day_of_week is_weekend has_renal_disease has_diabetes
has_hypertension has_mental_health n_conditions_total n_conditions_active
last_hba1c last_egfr last_systolic_bp last_diastolic_bp last_bmi n_labs_90d
n_vitals_90d n_procedures_90d n_procedures_365d had_surgical_90d had_dialysis_90d
""".split()
TARGET_COL = "readmitted_30d"
TIME_COL = "period_start"

## Notebook-derived classical column groups: QIMED_testes.ipynb, cell 77.
NUM_COLS = """
age_at_enc n_enc_total n_enc_30d n_enc_90d n_enc_365d days_since_last
n_conditions_total n_conditions_active n_procedures_90d n_procedures_365d
n_labs_90d n_vitals_90d
""".split()
LAB_COLS = """
last_hba1c last_egfr last_systolic_bp last_diastolic_bp last_bmi
""".split()
CAT_COLS = """
gender race marital_status class_code enc_type_grp month day_of_week quarter
""".split()
BIN_COLS = """
deceased has_reason had_emer_90d had_imp_90d has_renal_disease has_diabetes
has_hypertension has_mental_health had_surgical_90d had_dialysis_90d is_weekend
""".split()


########## Quantum input preprocessing ##########

def quantum_preprocess(frame: pd.DataFrame, fit_rows: int) -> np.ndarray:
    """Create 36 bounded values using only the training prefix."""
    encoded = frame[FEATURE_COLS].copy()

    numeric = frame[NUM_COLS].to_numpy(float)
    numeric = StandardScaler().fit(numeric[:fit_rows]).transform(numeric)
    encoded[NUM_COLS] = np.clip(numeric, -CLIP_SIGMA, CLIP_SIGMA) / CLIP_SIGMA

    labs = frame[LAB_COLS].to_numpy(float)
    labs = SimpleImputer(strategy="median").fit(labs[:fit_rows]).transform(labs)
    labs = StandardScaler().fit(labs[:fit_rows]).transform(labs)
    encoded[LAB_COLS] = np.clip(labs, -CLIP_SIGMA, CLIP_SIGMA) / CLIP_SIGMA

    binary = frame[BIN_COLS]
    if not binary.isin((0, 1)).all().all():
        raise ValueError("Invalid binary feature")
    encoded[BIN_COLS] = 2.0 * binary - 1.0

    if frame[CAT_COLS].isna().any().any():
        raise ValueError("Missing categorical feature")
    categorical = frame[CAT_COLS].astype(str)
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ).fit(categorical.iloc[:fit_rows])
    codes = encoder.transform(categorical)
    span = np.maximum(
        np.asarray([len(categories) - 1 for categories in encoder.categories_]), 1
    )
    # Temporal validation contains legitimate categories absent from training,
    # so zero is their explicit neutral encoding.
    encoded[CAT_COLS] = np.where(codes < 0, 0.0, 2.0 * codes / span - 1.0)

    output = encoded[FEATURE_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(output).all():
        raise ValueError("Quantum preprocessing produced non-finite values")
    return output


########## Classical models and study configuration ##########

PACKAGE_ROOT = Path(__file__).parent

DATASET = PACKAGE_ROOT / "data/dataset_modelo_readmissao.parquet"
OUTPUT = PACKAGE_ROOT / "results/qimed-pqfm-v2-q18-tilelang-4clf"

QUBITS = (18,)

## Notebook-derived expanding temporal CV counts: QIMED_testes.ipynb, cell 83.
## V2 implements the selection loop itself so it can fit each PQFM per split.
OUTER_SPLITS = 10
INNER_SPLITS = 3

PARAMETER_SAMPLES = 1

SEED = 42
CLIP_SIGMA = 4.0


## Notebook-derived classical models and search grids: QIMED_testes.ipynb, cell 81.
## Each entry is (unfitted estimator, unprefixed parameter grid, scale inputs).
MODELS = {
    "Logistic Regression": (
        LogisticRegression(class_weight="balanced", random_state=SEED),
        {"C": (1.0,), "max_iter": (1000,)},
        True,
    ),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {"max_depth": (6,), "min_samples_leaf": (50,)},
        False,
    ),
    "Random Forest": (
        RandomForestClassifier(
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        {
            "n_estimators": (200,),
            "max_depth": (10,),
            "min_samples_leaf": (20,),
        },
        False,
    ),
    "XGBoost": (
        XGBClassifier(
            eval_metric="logloss", verbosity=0, random_state=SEED
        ),
        {
            "n_estimators": (200,),
            "max_depth": (4,),
            "learning_rate": (0.05,),
            "subsample": (1.0,),
        },
        False,
    ),
}

QFM_OPTIONS = dict(
    seed=SEED,
    ideal=False,
    simulation=True,
    fakebackend=False,
    shots=4096,
    ibm_qpu="ibm_kingston",
    features_per_qubit=1,
    axes=("x", "y", "z"),
    encoding_mode="shared_feature",
    keep_diagonal_terms=True,
    keep_cross_terms=False,
    measure_all_zz=True,
    m=1,
    tau=1.0,
    rho_thr=0.0,
)

TILELANG_BACKEND = XYZTileLangBackend()
TILELANG_EDGES = {}


def run_xyz_tilelang(_qc, _order, theta, nodes, _backend, _estimator, _obs, metadata, **_kwargs):
    qubits = len(nodes)
    axes = tuple(dict.fromkeys(item[0][0] for item in metadata))
    observable_edges = sorted({item[1:] for item in metadata if len(item) == 3})
    return TILELANG_BACKEND.run(
        theta,
        n_qubits=qubits,
        axes=axes,
        evolution_edges=TILELANG_EDGES[qubits],
        observable_edges=observable_edges,
    ), metadata


xyz_module.run_projected_feature_job = run_xyz_tilelang


########## Split-local quantum projections ##########

def project_xyz(
    q: int,
    Xq: np.ndarray,
    train_rows: int,
    work_dir: Path,
) -> np.ndarray:
    """Fit XYZ on training rows and transform validation with that same map."""
    work_dir.mkdir(parents=True)
    training = pd.DataFrame(
        Xq[:train_rows],
        columns=[f"feature_{index:02d}" for index in range(Xq.shape[1])],
    )
    # PQFMLib requires a final target column, but XYZ is unsupervised.
    training[TARGET_COL] = np.zeros(train_rows, dtype=np.int8)
    training.to_csv(
        work_dir / "train.csv", index=False, float_format="%.17g"
    )

    qfm = XYZProjectiveQFM(
        name_file="train",
        q_enc=q,
        data_dir=str(work_dir),
        output_root=str(work_dir),
        **QFM_OPTIONS,
    )
    prepare = qfm.prepare_blocks_and_edges

    def prepare_and_capture_edges():
        prepare()
        TILELANG_EDGES[q] = tuple(qfm.edges_act)

    qfm.prepare_blocks_and_edges = prepare_and_capture_edges
    train = qfm.run()["Xq_all_raw"]
    qfm.X_q_all = Xq[train_rows:]
    qfm.execute_quantum_feature_map()
    return np.vstack((train, qfm.Xq_all_raw)).astype(np.float32)


########## Classical and hybrid representations ##########

def classical_features(
    raw: pd.DataFrame,
    train_rows: int,
    scale: bool,
) -> tuple[np.ndarray | sparse.spmatrix, np.ndarray | sparse.spmatrix]:
    ## Notebook-derived classical preprocessing: QIMED_testes.ipynb, cell 77.
    ## It is factored here because V2 concatenates its output with PQFM features.
    numeric = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scaler", StandardScaler()))
    labs = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        labs.append(("scaler", StandardScaler()))
    prep = ColumnTransformer([
        ("num", Pipeline(numeric), NUM_COLS),
        ("labs", Pipeline(labs), LAB_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_COLS),
        ("bin", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]), BIN_COLS),
    ])
    train = prep.fit_transform(raw.iloc[:train_rows])
    validation = prep.transform(raw.iloc[train_rows:])
    return train, validation


def _representations(
    raw: pd.DataFrame,
    Xq: np.ndarray,
    train_rows: int,
    qubits: tuple[int, ...],
    work_root: Path,
):
    classical = {
        scale: classical_features(raw, train_rows, scale)
        for scale in (False, True)
    }
    for q in qubits:
        quantum = project_xyz(
            q, Xq, train_rows,
            work_dir=work_root / f"q{q}",
        )
        q_train, q_validation = quantum[:train_rows], quantum[train_rows:]
        scaler = StandardScaler().fit(q_train)
        projected = {
            False: (q_train, q_validation),
            True: (
                scaler.transform(q_train),
                scaler.transform(q_validation),
            ),
        }
        yield q, {
            scale: tuple(
                sparse.hstack((a, b), format="csr")
                if sparse.issparse(a)
                else np.c_[a, b]
                for a, b in zip(classical[scale], projected[scale], strict=True)
            )
            for scale in (False, True)
        }


########## Inner-fold scoring and selection ##########

def _model(base, parameters: dict, y: np.ndarray):
    model = clone(base).set_params(**parameters)
    if isinstance(model, XGBClassifier):
        model.set_params(
            scale_pos_weight=float((y == 0).sum() / (y == 1).sum())
        )
    return model


def score_parameter(
    base,
    parameters: dict,
    X_train: np.ndarray,
    X_validation: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    outer_y_train: np.ndarray,
) -> float:
    ## Notebook-derived tuning objective: QIMED_testes.ipynb, cell 83.
    estimator = _model(base, parameters, outer_y_train)
    estimator.fit(X_train, y_train)
    probability = estimator.predict_proba(X_validation)[:, 1]
    return float(average_precision_score(y_validation, probability))


########## Nested study execution and artifacts ##########

def run() -> None:
    parameters = {
        name: list(ParameterSampler(
            dict(grid),
            n_iter=min(PARAMETER_SAMPLES, int(np.prod([len(v) for v in grid.values()]))),
            random_state=SEED,
        ))
        for name, (_, grid, _) in MODELS.items()
    }
    OUTPUT.mkdir(parents=True)
    df = pd.read_parquet(
        DATASET,
        columns=[*FEATURE_COLS, TARGET_COL, TIME_COL, "patient_id"],
    ).sort_values(TIME_COL, ignore_index=True)
    y = df[TARGET_COL].astype(np.int8).to_numpy()
    started = time.monotonic()
    all_metrics = []
    all_predictions = []

    for outer_fold, (outer_train, outer_validation) in enumerate(
        TimeSeriesSplit(n_splits=OUTER_SPLITS).split(df), 1
    ):
        with TemporaryDirectory(prefix=f"qimed-{outer_fold:02d}-") as temporary:
            root = Path(temporary)
            scores = defaultdict(list)
            outer_y_train = y[outer_train]

            for stage, (inner_train_local, inner_validation_local) in enumerate(
                TimeSeriesSplit(n_splits=INNER_SPLITS).split(outer_train), 1
            ):
                inner_train = outer_train[inner_train_local]
                inner_validation = outer_train[inner_validation_local]
                raw = df.iloc[np.r_[inner_train, inner_validation]]
                y_train, y_validation = y[inner_train], y[inner_validation]
                if len(set(y_train)) < 2 or len(set(y_validation)) < 2:
                    raise RuntimeError("Inner split does not contain both classes")
                Xq = quantum_preprocess(raw, len(inner_train))

                for q, matrices in _representations(
                    raw,
                    Xq,
                    len(inner_train),
                    QUBITS,
                    root / f"inner{stage}",
                ):
                    for model_name, (base, _, scale) in MODELS.items():
                        X_train, X_validation = matrices[scale]
                        candidates = parameters[model_name]
                        values = [
                            score_parameter(
                                base,
                                candidate,
                                X_train,
                                X_validation,
                                y_train,
                                y_validation,
                                outer_y_train,
                            )
                            for candidate in candidates
                        ]
                        for index, value in enumerate(values):
                            scores[model_name, q, index].append(value)

                print(
                    f"[outer {outer_fold:02d}/inner {stage:02d}] "
                    f"evaluated all {len(QUBITS)} qubit budgets",
                    flush=True,
                )

            choices = defaultdict(list)
            for model_name in MODELS:
                best_parameters = {
                    q: max(
                        range(len(parameters[model_name])),
                        key=lambda index: np.mean(scores[model_name, q, index]),
                    )
                    for q in QUBITS
                }
                hybrid_q = max(
                    QUBITS,
                    key=lambda q: np.mean(
                        scores[model_name, q, best_parameters[q]]
                    ),
                )
                choices[hybrid_q].append(
                    (model_name, best_parameters[hybrid_q])
                )

            selected_qubits = tuple(sorted(q for q in choices if q is not None))
            raw = df.iloc[np.r_[outer_train, outer_validation]]
            y_train, y_validation = y[outer_train], y[outer_validation]
            if len(set(y_train)) < 2 or len(set(y_validation)) < 2:
                raise RuntimeError("Outer split does not contain both classes")
            Xq = quantum_preprocess(raw, len(outer_train))

            for q, representations in _representations(
                raw, Xq, len(outer_train), selected_qubits, root / "outer"
            ):
                for model_name, parameter_index in choices[q]:
                    base, _, scale = MODELS[model_name]
                    X_train, X_validation = representations[scale]
                    selected_parameters = parameters[model_name][parameter_index]
                    inner_scores = scores[model_name, q, parameter_index]
                    estimator = _model(base, selected_parameters, y_train)
                    estimator.fit(X_train, y_train)
                    prediction = estimator.predict(X_validation).astype(np.int8)
                    probability = estimator.predict_proba(X_validation)[:, 1]
                    ## Notebook-derived outer-fold reporting metrics: QIMED_testes.ipynb, cell 83.
                    result = {
                        "AUC-ROC": roc_auc_score(y_validation, probability),
                        "AUC-PR": average_precision_score(y_validation, probability),
                        "Accuracy": accuracy_score(y_validation, prediction),
                        "F1": f1_score(
                            y_validation, prediction, zero_division=0
                        ),
                        "Precision": precision_score(
                            y_validation, prediction, zero_division=0
                        ),
                        "Recall": recall_score(y_validation, prediction),
                        "model": model_name,
                        "qubits": q,
                        "mean_inner_auc_pr": float(np.mean(inner_scores)),
                        "std_inner_auc_pr": float(np.std(inner_scores)),
                        "params_json": json.dumps(
                            selected_parameters, sort_keys=True
                        ),
                        "outer_fold": outer_fold,
                    }
                    all_metrics.append(result)
                    all_predictions.append(pd.DataFrame({
                        "model_row": outer_validation,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "qubits": q,
                        "target": y_validation,
                        "prediction": prediction,
                        "probability": probability,
                    }))

    metrics_frame = pd.DataFrame(all_metrics)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    metrics_frame.to_parquet(OUTPUT / "outer_metrics.parquet", index=False)
    predictions_frame.to_parquet(OUTPUT / "outer_predictions.parquet", index=False)
    study = {
        "study_id": "qimed-pqfm-nested-temporal-v2-q18-tilelang-4clf",
        "dataset": {
            "path": str(DATASET.resolve()),
            "rows": int(len(df)),
            "patients": int(df["patient_id"].nunique()),
            "target_counts": {
                str(label): int(count)
                for label, count in df[TARGET_COL].value_counts().sort_index().items()
            },
            "time_range": [
                df[TIME_COL].min().isoformat(),
                df[TIME_COL].max().isoformat(),
            ],
        },
        "backend": "pqfmlib.XYZProjectiveQFM.run",
        "qubits": QUBITS,
        "qfm_options": QFM_OPTIONS,
        "headline_source": "outer_only",
        "selection_metric": "average_precision",
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUTPUT / "study.json").write_text(
        json.dumps(study, indent=2, ensure_ascii=False) + "\n"
    )
    render_results(OUTPUT)


if __name__ == "__main__":
    run()
