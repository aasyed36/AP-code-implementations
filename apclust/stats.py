"""
Descriptive statistics utilities for exploring data matrices.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .data_io import DataMatrix


def summarize_features(data: DataMatrix) -> pd.DataFrame:
    """
    Compute summary statistics for each feature (column) in the matrix.

    Returns a DataFrame with mean, standard deviation, min, max, and the first
    and third quartiles, aligned with ``data.feature_names``.
    """

    values = data.values
    features = _fallback_feature_names(data)
    summary = {
        "mean": values.mean(axis=0),
        "std": values.std(axis=0, ddof=1),
        "min": values.min(axis=0),
        "q1": np.quantile(values, 0.25, axis=0),
        "median": np.median(values, axis=0),
        "q3": np.quantile(values, 0.75, axis=0),
        "max": values.max(axis=0),
    }
    return pd.DataFrame(summary, index=features)


def summarize_samples(data: DataMatrix) -> pd.DataFrame:
    """
    Compute summary statistics for each sample (row) in the matrix.
    """

    values = data.values
    samples = _fallback_sample_ids(data)
    summary = {
        "mean": values.mean(axis=1),
        "std": values.std(axis=1, ddof=1),
        "min": values.min(axis=1),
        "q1": np.quantile(values, 0.25, axis=1),
        "median": np.median(values, axis=1),
        "q3": np.quantile(values, 0.75, axis=1),
        "max": values.max(axis=1),
    }
    return pd.DataFrame(summary, index=samples)


def feature_correlations(data: DataMatrix, *, method: str = "pearson") -> pd.DataFrame:
    """
    Compute the feature-by-feature correlation matrix.
    """

    frame = pd.DataFrame(data.values, columns=_fallback_feature_names(data))
    return frame.corr(method=method)


def _fallback_sample_ids(data: DataMatrix) -> pd.Index:
    if data.sample_ids is not None:
        return pd.Index(data.sample_ids, name="sample")
    count = data.values.shape[0]
    return pd.Index([f"sample_{i}" for i in range(count)], name="sample")


def _fallback_feature_names(data: DataMatrix) -> pd.Index:
    if data.feature_names is not None:
        return pd.Index(data.feature_names, name="feature")
    count = data.values.shape[1]
    return pd.Index([f"feature_{j}" for j in range(count)], name="feature")

