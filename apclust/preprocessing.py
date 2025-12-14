"""
Preprocessing utilities for constructing shared representations across studies.

Implements the PCA-based feature spaces described in Masoero et al. (2023):
  - Joint 15-dimensional PCA space for PAM50 genes
  - Joint 30-dimensional PCA space for the top 1% most variable genes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data_io import DataMatrix


@dataclass(frozen=True)
class StudyMatrices:
    """Container holding aligned matrices for multiple studies."""

    mainz: DataMatrix
    transbig: DataMatrix
    vdx: DataMatrix

    def values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.mainz.values, self.transbig.values, self.vdx.values


def pam50_joint_pca(studies: StudyMatrices, *, n_components: int = 15) -> StudyMatrices:
    """
    Fit PCA on concatenated PAM50 matrices and return projected scores for each study.
    """

    concatenated, splits = _concatenate_studies(studies)
    scaler = StandardScaler(with_mean=True, with_std=False)
    centered = scaler.fit_transform(concatenated)

    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(centered)

    split_scores = _split_scores(scores, splits)
    return _wrap_scores(split_scores, studies, suffix="PAM50_PCA")


def top_variable_joint_pca(
    studies: StudyMatrices,
    *,
    percentile: float = 1.0,
    n_components: int = 30,
) -> Tuple[StudyMatrices, np.ndarray]:
    """
    Construct joint PCA on the intersection of top-percentile variable genes.

    Returns projected scores and the indices (column positions) of the selected genes.
    """

    indices = _top_percentile_variable_indices(studies, percentile=percentile)
    reduced = _subset_studies(studies, indices)

    concatenated, splits = _concatenate_studies(reduced)
    scaler = StandardScaler(with_mean=True, with_std=False)
    centered = scaler.fit_transform(concatenated)

    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(centered)
    split_scores = _split_scores(scores, splits)
    projected = _wrap_scores(split_scores, reduced, suffix="TOP_PCA")
    return projected, indices


def _concatenate_studies(studies: StudyMatrices) -> Tuple[np.ndarray, Tuple[int, int]]:
    mainz, transbig, vdx = studies.values()
    concatenated = np.concatenate([mainz, transbig, vdx], axis=0)
    splits = (len(mainz), len(mainz) + len(transbig))
    return concatenated, splits


def _split_scores(scores: np.ndarray, splits: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_mainz, split_trans = splits
    mainz_scores = scores[:split_mainz]
    trans_scores = scores[split_mainz:split_trans]
    vdx_scores = scores[split_trans:]
    return mainz_scores, trans_scores, vdx_scores


def _wrap_scores(
    scores: Tuple[np.ndarray, np.ndarray, np.ndarray],
    template: StudyMatrices,
    *,
    suffix: str,
) -> StudyMatrices:
    mainz_scores, trans_scores, vdx_scores = scores
    mainz_dm = DataMatrix(mainz_scores, template.mainz.sample_ids, _pca_feature_names(mainz_scores.shape[1], suffix))
    trans_dm = DataMatrix(trans_scores, template.transbig.sample_ids, _pca_feature_names(trans_scores.shape[1], suffix))
    vdx_dm = DataMatrix(vdx_scores, template.vdx.sample_ids, _pca_feature_names(vdx_scores.shape[1], suffix))
    return StudyMatrices(mainz_dm, trans_dm, vdx_dm)


def _pca_feature_names(n_components: int, suffix: str) -> Tuple[str, ...]:
    return tuple(f"{suffix}_{i:02d}" for i in range(n_components))


def _top_percentile_variable_indices(studies: StudyMatrices, *, percentile: float) -> np.ndarray:
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100].")

    mainz, transbig, vdx = studies.values()
    fractions = percentile / 100.0
    counts = int(np.ceil(mainz.shape[1] * fractions))

    idx_mainz = _top_indices(mainz, counts)
    idx_trans = _top_indices(transbig, counts)
    idx_vdx = _top_indices(vdx, counts)

    intersection = set(idx_mainz).intersection(idx_trans, idx_vdx)
    if not intersection:
        raise ValueError("No overlapping genes in the specified percentile.")
    return np.array(sorted(intersection))


def _top_indices(matrix: np.ndarray, count: int) -> np.ndarray:
    variation = _coefficient_of_variation(matrix)
    order = np.argsort(variation)[::-1]
    return order[:count]


def _coefficient_of_variation(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    # Avoid division by zero: genes with zero variance are assigned zero CV.
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean != 0, std / np.abs(mean), 0.0)
    return cv


def _subset_studies(studies: StudyMatrices, indices: np.ndarray) -> StudyMatrices:
    mainz = studies.mainz.values[:, indices]
    trans = studies.transbig.values[:, indices]
    vdx = studies.vdx.values[:, indices]

    mainz_dm = DataMatrix(
        mainz,
        studies.mainz.sample_ids,
        tuple(np.array(studies.mainz.feature_names)[indices]),
    )
    trans_dm = DataMatrix(
        trans,
        studies.transbig.sample_ids,
        tuple(np.array(studies.transbig.feature_names)[indices]),
    )
    vdx_dm = DataMatrix(
        vdx,
        studies.vdx.sample_ids,
        tuple(np.array(studies.vdx.feature_names)[indices]),
    )
    return StudyMatrices(mainz_dm, trans_dm, vdx_dm)

