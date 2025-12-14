"""
Visualization helpers for exploratory data analysis.

These functions return Matplotlib figures or axes so they can be embedded in
Jupyter notebooks without embedding plotting logic inline.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data_io import DataMatrix
from .stats import _fallback_feature_names


def feature_histograms(
    data: DataMatrix,
    *,
    features: Optional[Sequence[str]] = None,
    bins: Union[int, Sequence[int]] = 30,
    figsize: Optional[Sequence[float]] = None,
) -> plt.Figure:
    """
    Plot histograms for selected features.
    """

    feature_index = _select_features(data, features)
    n_features = len(feature_index)
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (4 * n_cols, 3 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    values = data.values[:, feature_index]
    feature_names = np.asarray(_fallback_feature_names(data))[feature_index]
    if isinstance(bins, (list, tuple, np.ndarray)):
        if len(bins) != n_features:
            raise ValueError("Length of `bins` must match number of selected features.")
        bin_iter = bins
    else:
        bin_iter = [bins] * n_features

    for ax, column, name, bin_count in zip(axes, values.T, feature_names, bin_iter):
        ax.hist(column, bins=bin_count, color="#377eb8", alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("value")
        ax.set_ylabel("frequency")

    for ax in axes[n_features:]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def pairwise_scatter(
    data: DataMatrix,
    *,
    features: Optional[Sequence[str]] = None,
    alpha: float = 0.7,
    figsize: Optional[Sequence[float]] = None,
) -> plt.Figure:
    """
    Plot pairwise scatter plots for up to four selected features.
    """

    feature_index = _select_features(data, features, limit=4)
    feature_names = np.asarray(_fallback_feature_names(data))[feature_index]
    values = data.values[:, feature_index]

    n_features = len(feature_index)
    n_pairs = n_features * (n_features - 1) // 2
    fig, axes = plt.subplots(n_pairs, 1, figsize=figsize or (5, 4 * n_pairs))
    axes = np.atleast_1d(axes)

    for ax, (i, j) in zip(axes, combinations(range(n_features), 2)):
        ax.scatter(values[:, i], values[:, j], alpha=alpha, edgecolors="none")
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    return fig


def feature_scatter_matrix(
    data: DataMatrix,
    *,
    features: Optional[Sequence[str]] = None,
    figsize: Optional[Sequence[float]] = None,
    diagonal: str = "kde",
    alpha: float = 0.6,
) -> pd.DataFrame:
    """
    Create a scatter-matrix (pairplot) for selected features.
    """

    feature_index = _select_features(data, features, limit=10)
    columns = np.asarray(_fallback_feature_names(data))[feature_index]
    frame = pd.DataFrame(data.values[:, feature_index], columns=columns)
    axes = pd.plotting.scatter_matrix(
        frame,
        figsize=figsize or (3.0 * len(columns), 3.0 * len(columns)),
        diagonal=diagonal,
        alpha=alpha,
    )
    plt.tight_layout()
    return axes


def cluster_scatter_2d(
    data: DataMatrix,
    labels: np.ndarray,
    *,
    title: str = "",
    use_pca: bool = True,
    figsize: Optional[Sequence[float]] = None,
    cmap: str = "tab20",
    marker_size: float = 20.0,
) -> plt.Figure:
    """
    Plot a 2D scatter of clustered data.

    If the data has more than two features and ``use_pca`` is True, a PCA projection
    onto the first two components is used.
    """

    values = data.values
    if values.ndim != 2:
        raise ValueError("DataMatrix.values must be two-dimensional.")

    if use_pca and values.shape[1] > 2:
        projector = PCA(n_components=2, random_state=0)
        coords = projector.fit_transform(values)
        x_label, y_label = "PC1", "PC2"
    else:
        coords = values[:, :2]
        names = np.asarray(_fallback_feature_names(data))
        x_label = names[0] if names.size > 0 else "feature_0"
        y_label = names[1] if names.size > 1 else "feature_1"

    fig, ax = plt.subplots(figsize=figsize or (7, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap=cmap,
        s=marker_size,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or "Cluster assignment")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.colorbar(scatter, ax=ax, label="cluster id")
    fig.tight_layout()
    return fig


def plot_raw_gene_pairs(
    data: DataMatrix,
    labels: np.ndarray,
    *,
    title: str = "Cluster assignments in gene space",
    palette: str = "tab20",
    marker_size: float = 20.0,
) -> sns.axisgrid.PairGrid:
    """
    Pairwise scatter of gene pairs coloured by cluster label.
    """

    frame = pd.DataFrame(data.values, columns=data.feature_names)
    frame["cluster"] = labels

    grid = sns.pairplot(
        frame,
        hue="cluster",
        palette=palette,
        corner=True,
        plot_kws={"alpha": 0.8, "s": marker_size},
    )

    for ax in grid.axes.flatten():
        if ax is not None:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if grid.fig is not None:
        grid.fig.suptitle(title, y=1.02)

    return grid


def pca_projection(
    data: DataMatrix,
    *,
    n_components: int = 2,
    standardize: bool = True,
    figsize: Optional[Sequence[float]] = None,
) -> plt.Figure:
    """
    Plot a PCA projection of the data matrix.
    """

    values = data.values
    if standardize:
        scaler = StandardScaler()
        values = scaler.fit_transform(values)

    if n_components < 2:
        raise ValueError("n_components must be at least 2 for visualization.")

    pca = PCA(n_components=n_components, random_state=0)
    components = pca.fit_transform(values)

    fig, ax = plt.subplots(figsize=figsize or (6, 5))
    ax.scatter(components[:, 0], components[:, 1], alpha=0.7, edgecolors="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


def _select_features(
    data: DataMatrix,
    features: Optional[Sequence[str]],
    limit: Optional[int] = None,
) -> np.ndarray:
    all_features = np.asarray(_fallback_feature_names(data))
    if features is None:
        index = np.arange(all_features.shape[0])
    else:
        mapping = {name: idx for idx, name in enumerate(all_features)}
        missing = [name for name in features if name not in mapping]
        if missing:
            raise KeyError(f"Features not found: {missing}")
        index = np.array([mapping[name] for name in features], dtype=int)

    if limit is not None and index.size > limit:
        raise ValueError(f"Select at most {limit} features for this plot.")

    return index

