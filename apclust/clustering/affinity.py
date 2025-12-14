"""
Thin wrapper around scikit-learn's AffinityPropagation estimator.

The goal is to expose a small, typed interface that matches the parameters we
care about while keeping notebook code compact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.cluster import AffinityPropagation

from ..data_io import DataMatrix


__all__ = ["AffinityResult", "run_affinity_propagation"]


@dataclass(frozen=True)
class AffinityResult:
    """
    Results from an AffinityPropagation run.

    Attributes
    ----------
    estimator:
        Fitted scikit-learn estimator. Exposes the full API if further
        inspection is required.
    labels:
        Cluster labels for each sample.
    exemplars:
        Indices of the chosen exemplars (cluster centers).
    n_iter:
        Number of iterations performed. Useful for diagnosing convergence.
    """

    estimator: AffinityPropagation
    labels: np.ndarray
    exemplars: np.ndarray
    n_iter: int

    def cluster_centers(self) -> np.ndarray:
        """
        Return the cluster centers in data space.
        """

        if not hasattr(self.estimator, "cluster_centers_"):
            raise AttributeError("Estimator does not expose cluster_centers_.")
        return self.estimator.cluster_centers_


def run_affinity_propagation(
    data: DataMatrix,
    *,
    preference: Optional[Sequence[float]] = None,
    damping: float = 0.5,
    max_iter: int = 200,
    convergence_iter: int = 15,
    affinity: str = "euclidean",
    copy: bool = True,
    verbose: bool = False,
    random_state: Optional[int] = None,
) -> AffinityResult:
    """
    Fit AffinityPropagation on the provided data matrix.

    Parameters mirror the scikit-learn estimator for ease of use.
    """

    values = data.values
    estimator = AffinityPropagation(
        damping=damping,
        preference=preference,
        max_iter=max_iter,
        convergence_iter=convergence_iter,
        affinity=affinity,
        copy=copy,
        verbose=verbose,
        random_state=random_state,
    )
    estimator.fit(values)
    return AffinityResult(
        estimator=estimator,
        labels=estimator.labels_.copy(),
        exemplars=(
            estimator.cluster_centers_indices_.copy()
            if estimator.cluster_centers_indices_ is not None
            else np.array([], dtype=int)
        ),
        n_iter=int(estimator.n_iter_),
    )


