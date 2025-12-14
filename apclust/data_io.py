"""
Data loading helpers for Affinity Propagation experiments.

The loaders aim to be lightweight and composable, following the coding style
favored by the CVXGRP community: minimal global state, explicit arguments, and
simple return types.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


@dataclass(frozen=True)
class DataMatrix:
    """
    Container for a numeric data matrix and optional axis labels.

    Attributes
    ----------
    values:
        Two-dimensional NumPy array with shape (n_samples, n_features).
    sample_ids:
        Optional iterable of sample identifiers aligned with the rows.
    feature_names:
        Optional iterable of feature identifiers aligned with the columns.
    """

    values: np.ndarray
    sample_ids: Optional[Tuple[str, ...]] = None
    feature_names: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("DataMatrix.values must be two-dimensional.")
        if self.sample_ids is not None and len(self.sample_ids) != self.values.shape[0]:
            raise ValueError("sample_ids length must match number of rows.")
        if self.feature_names is not None and len(self.feature_names) != self.values.shape[1]:
            raise ValueError("feature_names length must match number of columns.")

    def as_numpy(self) -> np.ndarray:
        """Return the underlying numeric matrix."""
        return self.values


def load_matrix(
    path: PathLike,
    *,
    key: Optional[str] = None,
    dtype: np.dtype = np.float64,
) -> DataMatrix:
    """
    Load a data matrix from ``.csv`` or ``.npy`` files.

    Parameters
    ----------
    path:
        Path to the file on disk.
    key:
        Optional dictionary key when loading a pickled ``.npy`` structure that
        stores multiple matrices (e.g., the Parmigiani bundle).
    dtype:
        Target numeric dtype. Defaults to ``np.float64`` for numerical
        stability in clustering computations.

    Returns
    -------
    DataMatrix
        Numeric matrix with rows representing samples and columns features.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        return _load_from_csv(path, dtype=dtype)

    if path.suffix.lower() == ".npy":
        return _load_from_npy(path, key=key, dtype=dtype)

    raise ValueError(f"Unsupported file extension: {path.suffix}")


def _load_from_csv(path: Path, *, dtype: np.dtype) -> DataMatrix:
    values = np.loadtxt(path, delimiter=",", dtype=dtype)
    values = np.atleast_2d(values)
    sample_ids = tuple(f"sample_{i}" for i in range(values.shape[0]))
    feature_names = tuple(f"feature_{j}" for j in range(values.shape[1]))
    return DataMatrix(values=values, sample_ids=sample_ids, feature_names=feature_names)


def _load_from_npy(path: Path, *, key: Optional[str], dtype: np.dtype) -> DataMatrix:
    raw = np.load(path, allow_pickle=True)

    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if key is None:
            raise ValueError("`.npy` file stores a dictionary. Provide the `key` argument.")
        container = raw.item()
        if key not in container:
            raise KeyError(f"Key '{key}' not found in {path.name}.")
        data = container[key]
    else:
        data = raw if key is None else raw.item().get(key)

    if isinstance(data, pd.DataFrame):
        return _matrix_from_dataframe(data, dtype=dtype)

    array = np.asarray(data, dtype=dtype)
    array = np.atleast_2d(array)
    sample_ids = tuple(f"sample_{i}" for i in range(array.shape[0]))
    feature_names = tuple(f"feature_{j}" for j in range(array.shape[1]))
    return DataMatrix(values=array, sample_ids=sample_ids, feature_names=feature_names)


def _matrix_from_dataframe(frame: pd.DataFrame, *, dtype: np.dtype) -> DataMatrix:
    frame = frame.copy()

    id_column = frame.columns[0]
    axis_tokens = frame[id_column].astype(str)

    numeric = frame.drop(columns=id_column, errors="ignore")
    numeric = numeric.apply(pd.to_numeric, errors="coerce")

    row_tokens = tuple(axis_tokens)
    column_tokens = tuple(str(col) for col in numeric.columns)

    row_sample_like = _looks_like_sample_ids(row_tokens)
    col_sample_like = _looks_like_sample_ids(column_tokens)
    row_gene_like = _looks_like_gene_ids(row_tokens)
    col_gene_like = _looks_like_gene_ids(column_tokens)

    # Case 1: rows look like samples (e.g., VDX IDs)
    if row_sample_like and not col_sample_like:
        sample_ids = row_tokens
        feature_names = column_tokens
        values = numeric.to_numpy(dtype=dtype)
        return DataMatrix(values=values, sample_ids=sample_ids, feature_names=feature_names)

    # Case 2: columns look like samples (e.g., Mainz, Transbig, VDX)
    if col_sample_like and not row_sample_like:
        sample_ids = column_tokens
        feature_names = row_tokens
        values = numeric.to_numpy(dtype=dtype).T
        return DataMatrix(values=values, sample_ids=sample_ids, feature_names=feature_names)

    # Case 3: rows look like gene IDs, columns do not look like samples
    if row_gene_like and not col_gene_like:
        sample_ids = column_tokens
        feature_names = row_tokens
        values = numeric.to_numpy(dtype=dtype).T
        return DataMatrix(values=values, sample_ids=sample_ids, feature_names=feature_names)

    # Fallback: treat rows as samples if no clear hint is found.
    values = numeric.to_numpy(dtype=dtype)
    sample_ids = row_tokens if len(row_tokens) == values.shape[0] else tuple(str(idx) for idx in frame.index)
    feature_names = column_tokens if len(column_tokens) == values.shape[1] else tuple(str(col) for col in numeric.columns)
    return DataMatrix(values=values, sample_ids=sample_ids, feature_names=feature_names)


def _looks_like_sample_ids(labels: Iterable[str], threshold: float = 0.5) -> bool:
    labels = list(labels)
    if not labels:
        return False
    pattern_hits = sum(_sample_token_predicate(label) for label in labels)
    score = pattern_hits / float(len(labels))
    return score >= threshold


def _sample_token_predicate(label: str) -> bool:
    label = label.upper()
    if "VDX_" in label or "TRANS" in label or "MAINZ" in label:
        return True
    if "_" in label:
        tail = label.split("_")[-1]
        return tail.isdigit()
    return False


def _looks_like_gene_ids(labels: Iterable[str], threshold: float = 0.5) -> bool:
    labels = list(labels)
    if not labels:
        return False
    hits = sum(_gene_token_predicate(label) for label in labels)
    return hits / float(len(labels)) >= threshold


def _gene_token_predicate(label: str) -> bool:
    lower = label.lower()
    return "_at" in lower or lower.startswith("affx")


def load_study_panels(path: PathLike, *, dtype: np.dtype = np.float64) -> Dict[str, DataMatrix]:
    """
    Load the HK_3, PAM50, and full gene panels for a given study.

    Parameters
    ----------
    path
        Path to a `*_dict.npy` file produced by the Masoero et al. pipeline.
    dtype
        Target dtype for numeric values (default: float64).

    Returns
    -------
    dict
        Mapping of panel name -> DataMatrix (keys: 'HK_3', 'PAM50', 'all').
    """

    path = Path(path)
    raw = np.load(path, allow_pickle=True)
    if not isinstance(raw.item(), dict):
        raise ValueError(f"{path} does not contain a panel dictionary.")

    panels = raw.item()
    expected_keys = {"HK_3", "PAM50", "all"}
    missing = expected_keys.difference(panels.keys())
    if missing:
        raise KeyError(f"Missing panels {missing} in {path}.")

    return {
        name: _matrix_from_dataframe(panels[name], dtype=dtype)
        for name in expected_keys
    }

