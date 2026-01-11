"""
Monte Carlo subsampling (MCSS) driver for Affinity Propagation replicability studies.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import ShuffleSplit


_METRIC_REGISTRY: Mapping[str, callable] = {
    "ari": adjusted_rand_score,
    "ami": lambda y_true, y_pred: adjusted_mutual_info_score(
        y_true, y_pred, average_method="arithmetic"
    ),
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_metric(name: str) -> callable:
    if name not in _METRIC_REGISTRY:
        raise ValueError(f"Unsupported metric '{name}'.")
    return _METRIC_REGISTRY[name]


def run_mcss_ap(
    X: np.ndarray,
    *,
    dataset_name: str,
    out_dir: Path,
    b: int = 200,
    train_frac: float = 0.8,
    random_seed: int = 0,
    ap_params: Dict | None = None,
    metrics: Sequence[str] = ("ari", "ami"),
) -> None:
    """
    Run Monte Carlo subsampling replicability study with Affinity Propagation.

    Parameters
    ----------
    X
        Data matrix of shape (n_samples, n_features).
    dataset_name
        Name used to create the output subdirectory.
    out_dir
        Root directory where results will be written.
    b
        Number of subsampling iterations.
    train_frac
        Fraction of samples to include in each training split.
    random_seed
        Seed for the subsampling generator.
    ap_params
        Parameters passed to sklearn.cluster.AffinityPropagation.
    metrics
        Iterable of metric names to compute (supported: "ari", "ami").
    """

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n_samples, n_features = X.shape
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be in (0, 1).")
    if b <= 0:
        raise ValueError("b must be positive.")

    metrics = tuple(metrics)
    for metric in metrics:
        _ = _resolve_metric(metric)

    ap_params = dict(ap_params or {})

    target_dir = Path(out_dir) / dataset_name
    _ensure_dir(target_dir)

    config_path = target_dir / "config.json"
    config = {
        "dataset_name": dataset_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "train_frac": train_frac,
        "b": b,
        "random_seed": random_seed,
        "ap_params": ap_params,
        "metrics": metrics,
        "timestamp": _dt.datetime.now().isoformat(),
    }
    config_path.write_text(json.dumps(config, indent=2))

    splitter = ShuffleSplit(
        n_splits=b,
        train_size=train_frac,
        test_size=1 - train_frac,
        random_state=random_seed,
    )

    records: list[dict] = []

    for iteration, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        split_path = target_dir / f"split_indices_{iteration:03d}.npz"
        np.savez(split_path, train_idx=train_idx, test_idx=test_idx)

        X_train = X[train_idx]
        X_test = X[test_idx]

        # Fit on train subset
        ap_train = AffinityPropagation(**ap_params)
        start = time.perf_counter()
        ap_train.fit(X_train)
        train_runtime = time.perf_counter() - start

        train_n_iter = getattr(ap_train, "n_iter_", np.nan)
        train_max_iter = ap_train.get_params().get("max_iter", None)
        train_converged_attr = getattr(ap_train, "converged_", None)
        if train_converged_attr is None:
            if train_max_iter is None:
                train_converged = train_n_iter is not None
            else:
                train_converged = bool(train_n_iter < train_max_iter)
        else:
            train_converged = bool(train_converged_attr)
        train_centers = getattr(ap_train, "cluster_centers_indices_", None)
        if train_centers is None:
            train_centers = np.array([], dtype=int)
        train_labels = getattr(ap_train, "labels_", None)
        train_preference = getattr(ap_train, "preference_", None)

        try:
            predicted_test_labels = ap_train.predict(X_test)
        except Exception:
            predicted_test_labels = None

        train_exemplars_local = np.asarray(train_centers, dtype=int)
        train_exemplars_global = (
            train_idx[train_exemplars_local] if train_exemplars_local.size > 0 else np.array([], dtype=int)
        )

        train_npz_path = target_dir / f"ap_train_{iteration:03d}.npz"
        np.savez(
            train_npz_path,
            train_labels=train_labels if train_labels is not None else np.array([], dtype=int),
            train_exemplars_local=train_exemplars_local,
            train_exemplars_global=train_exemplars_global,
            train_n_clusters=len(train_exemplars_local),
            train_converged=train_converged,
            train_n_iter=train_n_iter,
            preference_input=ap_params.get("preference"),
            preference_resolved=train_preference,
            damping=ap_params.get("damping", 0.5),
            max_iter=ap_params.get("max_iter", 200),
            convergence_iter=ap_params.get("convergence_iter", 15),
            runtime_sec=train_runtime,
        )

        # Fit on test subset (direct AP)
        ap_test = AffinityPropagation(**ap_params)
        start = time.perf_counter()
        ap_test.fit(X_test)
        test_runtime = time.perf_counter() - start

        test_n_iter = getattr(ap_test, "n_iter_", np.nan)
        test_max_iter = ap_test.get_params().get("max_iter", None)
        test_converged_attr = getattr(ap_test, "converged_", None)
        if test_converged_attr is None:
            if test_max_iter is None:
                test_converged = test_n_iter is not None
            else:
                test_converged = bool(test_n_iter < test_max_iter)
        else:
            test_converged = bool(test_converged_attr)
        test_centers = getattr(ap_test, "cluster_centers_indices_", None)
        if test_centers is None:
            test_centers = np.array([], dtype=int)
        test_labels = getattr(ap_test, "labels_", None)
        test_preference = getattr(ap_test, "preference_", None)

        test_exemplars_local = np.asarray(test_centers, dtype=int)
        test_exemplars_global = (
            test_idx[test_exemplars_local] if test_exemplars_local.size > 0 else np.array([], dtype=int)
        )

        test_npz_path = target_dir / f"ap_test_{iteration:03d}.npz"
        np.savez(
            test_npz_path,
            test_labels=test_labels if test_labels is not None else np.array([], dtype=int),
            test_exemplars_local=test_exemplars_local,
            test_exemplars_global=test_exemplars_global,
            test_n_clusters=len(test_exemplars_local),
            test_converged=test_converged,
            test_n_iter=test_n_iter,
            preference_input=ap_params.get("preference"),
            preference_resolved=test_preference,
            damping=ap_params.get("damping", 0.5),
            max_iter=ap_params.get("max_iter", 200),
            convergence_iter=ap_params.get("convergence_iter", 15),
            runtime_sec=test_runtime,
        )

        metric_results: dict[str, float] = {}
        if predicted_test_labels is not None and test_labels is not None:
            for metric in metrics:
                func = _resolve_metric(metric)
                try:
                    metric_results[metric] = float(func(test_labels, predicted_test_labels))
                except Exception:
                    metric_results[metric] = float("nan")
        else:
            metric_results = {metric: float("nan") for metric in metrics}

        record = {
            "iteration": iteration,
            "n_samples": n_samples,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_converged": bool(train_converged),
            "test_converged": bool(test_converged),
            "train_n_iter": float(train_n_iter),
            "test_n_iter": float(test_n_iter),
            "train_n_clusters": int(len(train_exemplars_local)),
            "test_n_clusters": int(len(test_exemplars_local)),
            "train_runtime_sec": float(train_runtime),
            "test_runtime_sec": float(test_runtime),
            "preference_input": ap_params.get("preference"),
            "preference_train": train_preference,
            "preference_test": test_preference,
            "damping": ap_params.get("damping", 0.5),
            "max_iter": ap_params.get("max_iter", 200),
            "convergence_iter": ap_params.get("convergence_iter", 15),
        }
        record.update(metric_results)
        records.append(record)

    summary_df = pd.DataFrame(records)
    summary_path = target_dir / "mcss_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary_stats = {
        "metrics": {},
        "train_converged_fraction": float(summary_df["train_converged"].mean()),
        "test_converged_fraction": float(summary_df["test_converged"].mean()),
        "train_cluster_count_stats": {
            "mean": float(summary_df["train_n_clusters"].mean()),
            "std": float(summary_df["train_n_clusters"].std()),
        },
        "test_cluster_count_stats": {
            "mean": float(summary_df["test_n_clusters"].mean()),
            "std": float(summary_df["test_n_clusters"].std()),
        },
    }
    quantiles = [0.25, 0.5, 0.75]
    for metric in metrics:
        series = summary_df[metric]
        summary_stats["metrics"][metric] = {
            "mean": float(series.mean(skipna=True)),
            "std": float(series.std(skipna=True)),
            "quantiles": {str(q): float(series.quantile(q)) for q in quantiles},
        }

    stats_path = target_dir / "mcss_summary_stats.json"
    stats_path.write_text(json.dumps(summary_stats, indent=2))

