"""
Replicability utilities for Affinity Propagation.

Implements a bootstrap-based comparison between train and test studies,
mirroring the structure used by Masoero et al. (2023) for other clustering
methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

from .data_io import DataMatrix


@dataclass(frozen=True)
class ReplicabilityResult:
    """
    Container for bootstrap replicability statistics.
    """

    ari: np.ndarray
    ami: np.ndarray
    predicted_labels: np.ndarray
    true_labels: np.ndarray
    k_pred: np.ndarray
    k_true: np.ndarray


def run_affinity_replicability(
    train: DataMatrix,
    test: DataMatrix,
    *,
    num_boots: int = 50,
    boot_fraction: float = 0.8,
    random_state: Optional[int] = None,
    preference: Optional[float] = None,
    damping: float = 0.5,
    max_iter: int = 200,
    convergence_iter: int = 15,
    affinity: str = "euclidean",
) -> ReplicabilityResult:
    """
    Evaluate cross-study replicability for Affinity Propagation.

    Parameters
    ----------
    train, test
        Data matrices representing the training and testing cohorts.
    num_boots
        Number of bootstrap repetitions.
    boot_fraction
        Fraction of samples drawn (without replacement) in each bootstrap subset.
    random_state
        Seed controlling both bootstrap sampling and the AP initialisation.
    preference, damping, max_iter, convergence_iter, affinity
        Hyperparameters forwarded to ``sklearn.cluster.AffinityPropagation``.
    """

    rng = np.random.default_rng(random_state)
    boot_size = max(1, int(min(len(train.values), len(test.values)) * boot_fraction))

    ari = np.zeros(num_boots)
    ami = np.zeros(num_boots)
    k_pred = np.zeros(num_boots, dtype=int)
    k_true = np.zeros(num_boots, dtype=int)

    predicted_matrix = np.full((num_boots, len(test.values)), -1, dtype=int)
    true_matrix = np.full((num_boots, len(test.values)), -1, dtype=int)

    for b in range(num_boots):
        sub_train, train_idx = _subsample(train.values, boot_size, rng)
        sub_test, test_idx = _subsample(test.values, boot_size, rng)

        estimator_kwargs = dict(
            preference=preference,
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            affinity=affinity,
            random_state=random_state,
        )

        pred_labels = _fit_predict_affinity(sub_train, sub_test, estimator_kwargs)
        true_labels = _fit_predict_affinity(sub_test, sub_test, estimator_kwargs)

        k_pred[b] = len(np.unique(pred_labels))
        k_true[b] = len(np.unique(true_labels))

        padded_pred = _pad_labels(pred_labels, test_idx, test.values)
        padded_true = _pad_labels(true_labels, test_idx, test.values)

        predicted_matrix[b] = padded_pred
        true_matrix[b] = padded_true

        ari[b] = adjusted_rand_score(padded_pred, padded_true)
        ami[b] = adjusted_mutual_info_score(padded_pred, padded_true)

    return ReplicabilityResult(
        ari=ari,
        ami=ami,
        predicted_labels=predicted_matrix,
        true_labels=true_matrix,
        k_pred=k_pred,
        k_true=k_true,
    )


def _subsample(
    matrix: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = rng.choice(matrix.shape[0], size=size, replace=False)
    return matrix[indices], indices


def _fit_predict_affinity(
    train_values: np.ndarray,
    test_values: np.ndarray,
    estimator_kwargs: dict,
) -> np.ndarray:
    estimator = AffinityPropagation(**estimator_kwargs)
    estimator.fit(train_values)
    return estimator.predict(test_values)


def _pad_labels(labels: np.ndarray, indices: np.ndarray, full_matrix: np.ndarray) -> np.ndarray:
    padded = np.full(full_matrix.shape[0], -1, dtype=int)
    padded[indices] = labels

    missing = np.where(padded == -1)[0]
    if missing.size == 0:
        return padded

    subsample_matrix = full_matrix[indices]
    nearest = np.argmin(euclidean_distances(full_matrix[missing], subsample_matrix), axis=1)
    padded[missing] = labels[nearest]
    return padded

