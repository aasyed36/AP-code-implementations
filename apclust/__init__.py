"""
Utilities for Affinity Propagation clustering experiments.

This package provides modular building blocks for data loading, exploratory
analysis, and downstream clustering workflows. Use the modules from notebooks
to keep interactive code light and reproducible.
"""

from .data_io import DataMatrix, load_matrix, load_study_panels
from .metrics import cluster_counts, contingency_table, partition_metrics
from .preprocessing import StudyMatrices, pam50_joint_pca, top_variable_joint_pca
from .replicability import ReplicabilityResult, run_affinity_replicability
from .hyperparam import (
    GridRunResult,
    compute_similarity_matrix,
    preference_grid,
    run_ap_grid,
    run_stability_probe,
    find_damping_limit,
    find_min_max_iter,
    run_multiple_ap,
    summarize_ap_runs,
)
from .plots import cluster_scatter_2d, plot_raw_gene_pairs
from .mcss import run_mcss_ap
from .synthetic import (
    generate_gaussian_mixture,
    generate_samples_original_1,
    toeplitz_covariance,
)

__all__ = [
    "DataMatrix",
    "load_matrix",
    "load_study_panels",
    "cluster_counts",
    "contingency_table",
    "partition_metrics",
    "StudyMatrices",
    "pam50_joint_pca",
    "top_variable_joint_pca",
    "ReplicabilityResult",
    "run_affinity_replicability",
    "GridRunResult",
    "compute_similarity_matrix",
    "preference_grid",
    "run_ap_grid",
    "run_stability_probe",
    "find_damping_limit",
    "find_min_max_iter",
    "run_multiple_ap",
    "summarize_ap_runs",
    "cluster_scatter_2d",
    "plot_raw_gene_pairs",
    "run_mcss_ap",
    "toeplitz_covariance",
    "generate_gaussian_mixture",
    "generate_samples_original_1",
]

