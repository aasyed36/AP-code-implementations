"""
Utilities for summarizing and comparing clustering results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


def cluster_counts(labels: np.ndarray, *, name: str = "cluster") -> pd.DataFrame:
    """
    Return a DataFrame of cluster sizes.
    """

    unique, counts = np.unique(labels, return_counts=True)
    frame = pd.DataFrame({name: unique, "size": counts})
    return frame.sort_values(by=name).reset_index(drop=True)


def partition_metrics(labels_a: np.ndarray, labels_b: np.ndarray) -> pd.Series:
    """
    Compute adjusted Rand index (ARI) and adjusted mutual information (AMI).
    """

    ari = adjusted_rand_score(labels_a, labels_b)
    ami = adjusted_mutual_info_score(labels_a, labels_b)
    return pd.Series({"ARI": ari, "AMI": ami})


def contingency_table(labels_a: np.ndarray, labels_b: np.ndarray) -> pd.DataFrame:
    """
    Cross-tabulate two cluster labelings.
    """

    return pd.crosstab(labels_a, labels_b, rownames=["labels_a"], colnames=["labels_b"])

