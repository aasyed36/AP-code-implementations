#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apclust.mcss import run_mcss_ap  # noqa: E402
from apclust.synthetic import (  # noqa: E402
    generate_gaussian_mixture,
    toeplitz_covariance,
)


MEAN_LEVELS = np.array([1.0, 4.0, 7.0, 10.0], dtype=np.float64)
SYNTHETIC_ROOT = REPO_ROOT / "Results" / "synthetic_cases"
MCSS_ROOT = REPO_ROOT / "Results" / "mcss"
DEFAULT_DAMPINGS = (0.50, 0.85)


@dataclass(frozen=True)
class Scenario:
    name: str
    regime: str
    family: str
    n_samples: int
    n_features: int
    covariance: dict
    noise_seed: int
    label_seed: int
    means: Sequence[float] = tuple(MEAN_LEVELS.tolist())
    label_probs: Sequence[float] | None = None


def build_scenarios() -> List[Scenario]:
    """Return the catalog of synthetic scenarios."""
    scenarios: list[Scenario] = []

    def add_family(
        *,
        name_prefix: str,
        regime: str,
        family: str,
        p_values: Iterable[int],
        t_values: Iterable[int],
        noise_offset: int,
        label_offset: int,
        covariance_factory,
    ) -> None:
        for p in p_values:
            for t in t_values:
                scenarios.append(
                    Scenario(
                        name=f"{name_prefix}_p{p}_t{t}",
                        regime=regime,
                        family=family,
                        n_samples=int(t),
                        n_features=int(p),
                        covariance=covariance_factory(p, t),
                        noise_seed=int(noise_offset + p + t),
                        label_seed=int(label_offset + p + t),
                    )
                )

    add_family(
        name_prefix="tgt_gt_iso",
        regime="t>p",
        family="identity_var1",
        p_values=(100, 200, 400),
        t_values=(10_000, 5_000, 1_000),
        noise_offset=100,
        label_offset=200,
        covariance_factory=lambda _p, _t: {"kind": "identity", "variance": 1.0},
    )

    add_family(
        name_prefix="tgt_gt_var10",
        regime="t>p",
        family="identity_var10",
        p_values=(100, 200, 400),
        t_values=(10_000, 5_000, 1_000),
        noise_offset=300,
        label_offset=400,
        covariance_factory=lambda _p, _t: {"kind": "identity", "variance": 10.0},
    )

    add_family(
        name_prefix="tgt_gt_toeplitz",
        regime="t>p",
        family="toeplitz",
        p_values=(100, 200, 400),
        t_values=(10_000, 5_000, 1_000),
        noise_offset=500,
        label_offset=600,
        covariance_factory=lambda _p, _t: {"kind": "toeplitz"},
    )

    add_family(
        name_prefix="t_leq_iso",
        regime="t<=p",
        family="identity_var1",
        p_values=(1_000, 2_000, 4_000),
        t_values=(100, 200, 1_000),
        noise_offset=700,
        label_offset=800,
        covariance_factory=lambda _p, _t: {"kind": "identity", "variance": 1.0},
    )

    add_family(
        name_prefix="t_leq_var10",
        regime="t<=p",
        family="identity_var10",
        p_values=(1_000, 2_000, 4_000),
        t_values=(100, 200, 1_000),
        noise_offset=900,
        label_offset=1_000,
        covariance_factory=lambda _p, _t: {"kind": "identity", "variance": 10.0},
    )

    add_family(
        name_prefix="t_leq_toeplitz",
        regime="t<=p",
        family="toeplitz",
        p_values=(1_000, 2_000, 4_000),
        t_values=(100, 200, 1_000),
        noise_offset=1_100,
        label_offset=1_200,
        covariance_factory=lambda _p, _t: {"kind": "toeplitz"},
    )

    return scenarios


def resolve_covariance(spec: dict, n_features: int) -> float | np.ndarray:
    """Render the covariance specification into a numeric object."""
    kind = spec.get("kind")
    if kind == "identity":
        variance = float(spec.get("variance", 1.0))
        if variance <= 0:
            raise ValueError("identity variance must be positive.")
        return variance
    if kind == "toeplitz":
        profile = spec.get("profile")
        return toeplitz_covariance(n_features, profile=profile)
    raise ValueError(f"Unsupported covariance kind: {kind!r}")


def load_or_generate_matrix(
    scenario: Scenario,
    *,
    overwrite: bool,
    write_csv: bool,
) -> np.ndarray:
    """Ensure the synthetic matrix exists on disk and return it."""
    case_dir = SYNTHETIC_ROOT / scenario.name
    matrix_npy = case_dir / "matrix.npy"
    matrix_csv = case_dir / "matrix.csv"
    metadata_path = case_dir / "metadata.json"

    if not overwrite and matrix_npy.exists():
        return np.load(matrix_npy)

    cov = resolve_covariance(scenario.covariance, scenario.n_features)
    data_matrix, component_labels = generate_gaussian_mixture(
        n_samples=scenario.n_samples,
        n_features=scenario.n_features,
        means=scenario.means,
        covariance=cov,
        noise_seed=scenario.noise_seed,
        label_seed=scenario.label_seed,
        labels=None,
        label_probs=scenario.label_probs,
    )

    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(matrix_npy, data_matrix)
    if write_csv:
        np.savetxt(matrix_csv, data_matrix, delimiter=",")

    metadata = {
        "name": scenario.name,
        "regime": scenario.regime,
        "family": scenario.family,
        "n_samples": scenario.n_samples,
        "n_features": scenario.n_features,
        "means": list(scenario.means),
        "covariance": scenario.covariance,
        "noise_seed": scenario.noise_seed,
        "label_seed": scenario.label_seed,
        "label_probs": list(scenario.label_probs) if scenario.label_probs is not None else None,
        "matrix_path": {
            "npy": str(matrix_npy.relative_to(REPO_ROOT)),
            "csv": str(matrix_csv.relative_to(REPO_ROOT)) if write_csv else None,
        },
        "note": "Component memberships are reproduced implicitly via noise_seed and label_seed.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return data_matrix


def run_mcss_for_scenario(
    scenario: Scenario,
    matrix: np.ndarray,
    *,
    dampings: Sequence[float],
    b: int,
    train_frac: float,
    random_seed: int,
    overwrite: bool,
    ap_random_state: int | None,
) -> None:
    """Run MCSS+AP sweeps for the requested damping values."""
    for damping in dampings:
        damping_dir = f"damping{damping:0.2f}"
        summary_path = MCSS_ROOT / scenario.name / damping_dir / "mcss_summary.csv"
        if summary_path.exists() and not overwrite:
            print(f"  [skip] {scenario.name} @ {damping_dir} (results exist)")
            continue

        dataset_key = str(Path(scenario.name) / damping_dir)
        ap_params = {"damping": damping}
        if ap_random_state is not None:
            ap_params["random_state"] = ap_random_state

        print(f"  [run]  {scenario.name} @ {damping_dir} → b={b}, train_frac={train_frac}")
        run_mcss_ap(
            matrix,
            dataset_name=dataset_key,
            out_dir=MCSS_ROOT,
            b=b,
            train_frac=train_frac,
            random_seed=random_seed,
            ap_params=ap_params,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic GMM datasets and run MCSS+AP sweeps."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Run only the named scenario (can be provided multiple times).",
    )
    parser.add_argument(
        "--regime",
        choices=("t>p", "t<=p", "all"),
        default="t>p",
        help="Filter scenarios by sample/feature regime.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit.",
    )
    parser.add_argument(
        "--damping",
        action="append",
        type=float,
        dest="dampings",
        help="Damping value for AP (can be repeated). Defaults to 0.50 and 0.85.",
    )
    parser.add_argument(
        "--b",
        type=int,
        default=200,
        help="Number of MCSS iterations (default: 200).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Training fraction for each MCSS split (default: 0.8).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for MCSS subsampling (default: 0).",
    )
    parser.add_argument(
        "--ap-random-state",
        type=int,
        default=0,
        help="Random state passed to AffinityPropagation (default: 0).",
    )
    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="Regenerate synthetic matrices even if cached.",
    )
    parser.add_argument(
        "--overwrite-mcss",
        action="store_true",
        help="Re-run MCSS sweeps even if summary exists.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing matrix.csv (only matrix.npy will be stored).",
    )
    return parser.parse_args()


def select_scenarios(
    scenarios: Sequence[Scenario],
    *,
    names: Sequence[str] | None,
    regime: str,
) -> list[Scenario]:
    """Filter the scenario catalog."""
    selected = []
    name_filter = set(names) if names else None
    for scenario in scenarios:
        if name_filter and scenario.name not in name_filter:
            continue
        if regime != "all" and scenario.regime != regime:
            continue
        selected.append(scenario)
    return selected


def main() -> None:
    args = parse_args()
    scenarios = build_scenarios()

    if args.list:
        print("Available scenarios:")
        for scenario in scenarios:
            print(f"  {scenario.name:>24}  ({scenario.regime}, {scenario.family})")
        return

    selected = select_scenarios(
        scenarios,
        names=args.scenarios,
        regime=args.regime,
    )
    if not selected:
        raise SystemExit("No scenarios selected. Use --list to inspect available names.")

    dampings = tuple(args.dampings) if args.dampings else DEFAULT_DAMPINGS

    for scenario in selected:
        print(f"[scenario] {scenario.name}  ({scenario.regime}, {scenario.family})")
        matrix = load_or_generate_matrix(
            scenario,
            overwrite=args.overwrite_data,
            write_csv=not args.no_csv,
        )
        run_mcss_for_scenario(
            scenario,
            matrix,
            dampings=dampings,
            b=args.b,
            train_frac=args.train_frac,
            random_seed=args.random_seed,
            overwrite=args.overwrite_mcss,
            ap_random_state=args.ap_random_state,
        )


if __name__ == "__main__":
    main()

