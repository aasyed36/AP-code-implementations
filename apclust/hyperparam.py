"""
Hyperparameter exploration utilities for Affinity Propagation.

These helpers support the workflow:
    1. compute a similarity matrix (optional, but useful for later metric learning),
    2. derive a data-driven grid of preference values,
    3. run Affinity Propagation across a grid of (preference, damping) pairs,
       collecting convergence diagnostics and cluster statistics,
    4. optionally retain label assignments for downstream analysis or plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_rand_score
import time
from sklearn.metrics import pairwise_distances

from .data_io import DataMatrix


def compute_similarity_matrix(
    data: DataMatrix,
    *,
    squared: bool = True,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Construct the negative squared Euclidean similarity matrix used by AP.

    Parameters
    ----------
    data
        Input samples with shape (n_samples, n_features).
    squared
        If True (default), use squared Euclidean distances before negation.
        Otherwise, use standard Euclidean distances.
    dtype
        Target dtype for the resulting matrix.
    """

    metric = "sqeuclidean" if squared else "euclidean"
    distances = pairwise_distances(data.values, metric=metric).astype(dtype, copy=False)
    return -distances


def preference_grid(
    similarity_matrix: np.ndarray,
    *,
    quantiles: Iterable[float],
    include_median: bool = True,
) -> List[float]:
    """
    Derive preference values from the off-diagonal similarities.

    Parameters
    ----------
    similarity_matrix
        Square matrix of similarities (larger is more similar).
    quantiles
        Iterable of quantiles (between 0 and 1) to evaluate.
    include_median
        If True, ensure the 0.5 quantile is included even if not in `quantiles`.
    """

    sim = np.asarray(similarity_matrix)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("similarity_matrix must be square.")

    mask = ~np.eye(sim.shape[0], dtype=bool)
    off_diag = sim[mask]
    if off_diag.size == 0:
        raise ValueError("similarity_matrix must contain off-diagonal entries.")

    q_values = set(float(q) for q in quantiles)
    if include_median:
        q_values.add(0.5)

    prefs = sorted(
        float(np.quantile(off_diag, q))
        for q in q_values
        if 0.0 <= q <= 1.0
    )
    return prefs


@dataclass(frozen=True)
class GridRunResult:
    """
    Results from a grid search over Affinity Propagation hyperparameters.
    """

    records: pd.DataFrame
    labels: Dict[Tuple[float, float], np.ndarray]


def run_ap_grid(
    data: DataMatrix,
    *,
    preferences: Sequence[Optional[float]],
    dampings: Sequence[float],
    similarity_matrix: Optional[np.ndarray] = None,
    max_iter: int = 200,
    convergence_iter: int = 15,
    random_state: int = 0,
    save_labels: bool = False,
) -> GridRunResult:
    """
    Run Affinity Propagation over a grid of (preference, damping) pairs.

    Parameters
    ----------
    data
        DataMatrix containing samples to cluster. Required unless
        `similarity_matrix` is provided in precomputed mode.
    preferences
        Sequence of preference values (floats or None to use the median).
    dampings
        Sequence of damping factors in [0.5, 0.99).
    similarity_matrix
        Optional precomputed similarity matrix to use with `affinity='precomputed'`.
        If provided, its shape must match (n_samples, n_samples).
    max_iter, convergence_iter
        Stop criteria passed to `AffinityPropagation`.
    random_state
        Seed to ensure deterministic message initialisation.
    save_labels
        If True, retain label arrays keyed by (preference, damping).

    Returns
    -------
    GridRunResult
        `records` dataframe summarising each run, and optionally `labels`.
    """

    if similarity_matrix is not None:
        sim = np.asarray(similarity_matrix)
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
            raise ValueError("similarity_matrix must be square.")
        fit_data = sim
        affinity = "precomputed"
    else:
        fit_data = data.values
        affinity = "euclidean"

    records: List[dict] = []
    label_store: Dict[Tuple[float, float], np.ndarray] = {}

    for pref in preferences:
        for damping in dampings:
            estimator = AffinityPropagation(
                damping=damping,
                preference=pref,
                affinity=affinity,
                max_iter=max_iter,
                convergence_iter=convergence_iter,
                random_state=random_state,
            )

            estimator.fit(fit_data)

            labels = estimator.labels_.astype(int, copy=True)
            unique_labels = np.unique(labels)
            exemplars = estimator.cluster_centers_indices_

            record = {
                "preference": pref if pref is not None else np.median(estimator.affinity_matrix_),
                "input_preference": pref,
                "damping": damping,
                "n_clusters": unique_labels.size,
                "n_iter": estimator.n_iter_,
                "converged": estimator.n_iter_ < max_iter,
                "random_state": random_state,
                "exemplars": tuple(int(idx) for idx in exemplars) if exemplars is not None else tuple(),
            }
            records.append(record)

            if save_labels:
                label_store[(pref, damping)] = labels

    records_df = pd.DataFrame.from_records(records)
    records_df.sort_values(["preference", "damping"], inplace=True)
    records_df.reset_index(drop=True, inplace=True)

    return GridRunResult(records=records_df, labels=label_store)


def run_stability_probe(
    data: DataMatrix,
    *,
    preference: Optional[float] = None,
    dampings: Optional[Sequence[float]] = None,
    max_iters: Optional[Sequence[int]] = None,
    convergence_iter: int = 15,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Probe convergence of Affinity Propagation across (damping, max_iter) pairs.
    """

    damping_scan = np.linspace(0.5, 0.95, 10) if dampings is None else list(dampings)
    max_iter_scan = [50, 100, 150, 200] if max_iters is None else list(max_iters)

    records = []
    for damping in damping_scan:
        for max_iter in max_iter_scan:
            result = run_ap_grid(
                data,
                preferences=[preference],
                dampings=[damping],
                max_iter=max_iter,
                convergence_iter=convergence_iter,
                random_state=random_state,
                save_labels=False,
            )
            row = result.records.iloc[0].copy()
            row["max_iter_setting"] = max_iter
            records.append(row)

    df = pd.DataFrame(records)
    return df[["damping", "max_iter_setting", "n_iter", "converged"]]


def _run_once(
    data: DataMatrix,
    *,
    preference: Optional[float],
    damping: float,
    max_iter: int,
    convergence_iter: int = 15,
    random_state: int = 0,
) -> pd.Series:
    result = run_ap_grid(
        data,
        preferences=[preference],
        dampings=[damping],
        max_iter=max_iter,
        convergence_iter=convergence_iter,
        random_state=random_state,
        save_labels=False,
    )
    return result.records.iloc[0]


def find_damping_limit(
    data: DataMatrix,
    *,
    preference: Optional[float] = None,
    max_iter: int = 200,
    step: float = 0.05,
    tol: float = 1e-3,
    convergence_iter: int = 15,
    random_state: int = 0,
) -> float:
    """
    Largest damping in [0.5, 0.99) that converges under the specified max_iter.
    """

    d = 0.5
    last_good = d
    while d < 0.99:
        rec = _run_once(
            data,
            preference=preference,
            damping=d,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            random_state=random_state,
        )
        if rec["converged"]:
            last_good = d
            d += step
        else:
            high = d
            low = last_good
            break
    else:
        return 0.99

    while high - low > tol:
        mid = 0.5 * (low + high)
        rec = _run_once(
            data,
            preference=preference,
            damping=mid,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            random_state=random_state,
        )
        if rec["converged"]:
            low = mid
        else:
            high = mid
    return low


def find_min_max_iter(
    data: DataMatrix,
    *,
    preference: Optional[float] = None,
    damping: float = 0.5,
    max_iter_cap: int = 200,
    convergence_iter: int = 15,
    random_state: int = 0,
) -> int:
    """
    Smallest max_iter in [1, max_iter_cap] that converges at the given damping.
    """

    low = 1
    high = max_iter_cap
    min_good = max_iter_cap

    while low <= high:
        mid = (low + high) // 2
        rec = _run_once(
            data,
            preference=preference,
            damping=damping,
            max_iter=mid,
            convergence_iter=convergence_iter,
            random_state=random_state,
        )
        if rec["converged"]:
            min_good = mid
            high = mid - 1
        else:
            low = mid + 1
    return min_good


def run_multiple_ap(
    data: DataMatrix,
    *,
    n_runs: int = 3,
    preference: Optional[float] = None,
    damping: float = 0.5,
    max_iter: int = 200,
    convergence_iter: int = 15,
    random_state: Optional[int] = None,
) -> List[dict]:
    """
    Execute Affinity Propagation multiple times on the same dataset.
    """

    runs: List[dict] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = run_ap_grid(
            data,
            preferences=[preference],
            dampings=[damping],
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            random_state=random_state,
            save_labels=True,
        )
        elapsed = time.perf_counter() - start

        record = result.records.iloc[0].to_dict()
        record["labels"] = result.labels[(preference, damping)]
        record["max_iter"] = max_iter
        record["convergence_iter"] = convergence_iter
        record["runtime"] = elapsed
        runs.append(record)

    return runs


def summarize_ap_runs(
    runs: Sequence[dict],
    *,
    reference: int = 0,
) -> pd.DataFrame:
    """
    Summarise multiple AP runs, comparing each run to a reference run.
    """

    if not runs:
        return pd.DataFrame(
            columns=[
                "run",
                "n_clusters",
                "converged",
                "n_iter",
                "labels_equal_ref",
                "ari_to_ref",
            ]
        )

    if not (0 <= reference < len(runs)):
        raise IndexError("reference run index out of range.")

    ref_labels = runs[reference]["labels"]
    rows = []
    for idx, run in enumerate(runs):
        labels = run["labels"]
        equal = np.array_equal(labels, ref_labels)
        ari = adjusted_rand_score(ref_labels, labels)
        rows.append(
            {
                "run": idx + 1,
                "input_preference": run.get("input_preference"),
                "preference": run.get("preference"),
                "damping": run.get("damping"),
                "max_iter": run.get("max_iter"),
                "convergence_iter": run.get("convergence_iter"),
                "n_clusters": run["n_clusters"],
                "converged": run["converged"],
                "n_iter": run["n_iter"],
                "exemplar_count": len(run.get("exemplars", ())),
                "runtime": run.get("runtime"),
                "labels_equal_ref": equal,
                "ari_to_ref": ari,
            }
        )

    return pd.DataFrame(rows)

