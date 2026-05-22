"""Dataset loading and feature-relation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def validate_numeric_dataframe(df: pd.DataFrame) -> None:
    """Validate that all columns are numeric and finite enough for QFM preprocessing."""
    non_numeric = df.columns[~df.apply(lambda s: pd.api.types.is_numeric_dtype(s))]
    if len(non_numeric) > 0:
        raise ValueError(f"The dataset must contain only numeric columns. Non-numeric columns: {list(non_numeric)}")
    if df.isna().any().any():
        cols_nan = df.columns[df.isna().any()].tolist()
        raise ValueError(f"The dataset contains NaN values. Columns with NaN: {cols_nan}")


def load_tabular_dataset(name_file: str, data_dir: str = "./data"):
    """Load a CSV dataset whose last column is the target."""
    path = Path(data_dir) / f"{name_file}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    validate_numeric_dataframe(df)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy()
    return df, X, y


def mutual_information_matrix(X: np.ndarray, *, seed: int = 42, rho_thr: float = 0.0) -> np.ndarray:
    """Compute a normalized pairwise mutual-information matrix among features."""
    n_features = int(X.shape[1])
    MI = np.zeros((n_features, n_features), dtype=float)
    exact_duplicates = np.zeros((n_features, n_features), dtype=bool)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if np.allclose(X[:, i], X[:, j]):
                MI[i, j] = 1.0
                exact_duplicates[i, j] = True
                exact_duplicates[j, i] = True
            else:
                MI[i, j] = mutual_info_regression(X[:, [i]], X[:, j], random_state=seed)[0]
            MI[j, i] = MI[i, j]
    J = MI / (MI.max() + 1e-12)
    J[exact_duplicates] = 1.0
    J[J < rho_thr] = 0.0
    np.fill_diagonal(J, 0.0)
    return J
