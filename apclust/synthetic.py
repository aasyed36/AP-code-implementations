from __future__ import annotations

import random
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np

ArrayLike = np.ndarray
SeedLike = Union[int, np.random.RandomState]


def _as_random_state(seed: Optional[SeedLike]) -> np.random.RandomState:
    """Return a RandomState no matter how the seed is specified."""
    if seed is None:
        return np.random.RandomState()
    if isinstance(seed, np.random.RandomState):
        return seed
    return np.random.RandomState(seed)


def toeplitz_covariance(
    n_features: int,
    *,
    profile: Optional[Dict[int, float]] = None,
    dtype: np.dtype = np.float64,
) -> ArrayLike:
    """Construct the Toeplitz covariance from Eq. (2)."""
    if n_features <= 0:
        raise ValueError("n_features must be positive.")

    if profile is None:
        profile = {
            0: 1.0,
            1: 0.9,
            2: 0.8,
            3: 0.7,
            4: 0.6,
            5: 0.5,
            6: 0.4,
            7: 0.3,
            8: 0.2,
            9: 0.1,
            10: 0.001,
        }

    cov = np.zeros((n_features, n_features), dtype=dtype)
    for offset, value in profile.items():
        if offset < 0:
            raise ValueError("Toeplitz profile offsets must be non-negative.")
        if abs(value) < 1e-15 or offset >= n_features:
            continue
        diag_indices = np.arange(0, n_features - offset, dtype=int)
        cov[diag_indices, diag_indices + offset] = value
        if offset > 0:
            cov[diag_indices + offset, diag_indices] = value

    if not np.all(np.diag(cov)):
        np.fill_diagonal(cov, profile.get(0, 1.0))

    return cov


def _resolve_means(
    means: Union[Sequence[float], ArrayLike],
    n_features: int,
) -> ArrayLike:
    mean_array = np.asarray(means, dtype=np.float64)
    if mean_array.ndim == 1:
        return np.repeat(mean_array[:, None], n_features, axis=1)
    if mean_array.shape == (mean_array.shape[0], n_features):
        return mean_array
    raise ValueError(
        f"Expected `means` to be 1-D of length K or shape (K, {n_features}). "
        f"Received array with shape {mean_array.shape!r}."
    )


def _label_array(
    n_samples: int,
    n_components: int,
    *,
    labels: Optional[Sequence[int]],
    label_probs: Optional[Sequence[float]],
    rng: random.Random,
) -> np.ndarray:
    if labels is not None:
        label_arr = np.asarray(labels, dtype=int)
        if label_arr.shape != (n_samples,):
            raise ValueError(
                f"Provided `labels` has shape {label_arr.shape}, expected ({n_samples},)."
            )
        if (label_arr < 0).any() or (label_arr >= n_components).any():
            raise ValueError("Labels must be in [0, n_components).")
        return label_arr.copy()

    if label_probs is None:
        label_probs = np.full(n_components, 1.0 / n_components)
    else:
        label_probs = np.asarray(label_probs, dtype=np.float64)
        if label_probs.shape != (n_components,):
            raise ValueError("`label_probs` must have length equal to n_components.")
        if not np.isclose(label_probs.sum(), 1.0):
            raise ValueError("`label_probs` must sum to 1.")

    population = list(range(n_components))
    weights = label_probs.tolist() if isinstance(label_probs, np.ndarray) else list(label_probs)
    return np.fromiter(
        (rng.choices(population, weights=weights, k=1)[0] for _ in range(n_samples)),
        dtype=int,
        count=n_samples,
    )


def _choices_rng(seed: Optional[SeedLike], *, fallback_seed: Optional[int] = None) -> random.Random:
    """Return a `random.Random` instance configured for component draws."""
    rng = random.Random()
    if seed is None:
        if fallback_seed is not None:
            rng.seed(int(fallback_seed))
        return rng
    if isinstance(seed, np.random.RandomState):
        derived_seed = int(seed.randint(0, 2**31 - 1))
        rng.seed(derived_seed)
        return rng
    rng.seed(int(seed))
    return rng


def _resolve_covariances(
    covariance: Union[float, ArrayLike, Sequence[Union[float, ArrayLike]]],
    *,
    n_components: int,
    n_features: int,
) -> Tuple[Tuple[str, Union[float, ArrayLike], Union[float, ArrayLike]], ...]:
    """Normalize covariance specification per component."""
    def _as_matrix(value: ArrayLike) -> np.ndarray:
        cov_arr = np.asarray(value, dtype=np.float64)
        if cov_arr.shape != (n_features, n_features):
            raise ValueError(
                f"`covariance` entries must have shape ({n_features}, {n_features}). "
                f"Received {cov_arr.shape}."
            )
        return cov_arr

    if np.isscalar(covariance):
        variance = float(covariance)
        if variance <= 0:
            raise ValueError("Variance must be strictly positive.")
        return tuple(("scalar", variance, np.sqrt(variance)) for _ in range(n_components))

    if isinstance(covariance, Sequence) and not isinstance(covariance, np.ndarray):
        if len(covariance) != n_components:
            raise ValueError(
                "`covariance` sequence length must equal the number of components when "
                "supplying per-component covariances."
            )
        resolved = []
        for entry in covariance:
            if np.isscalar(entry):
                variance = float(entry)
                if variance <= 0:
                    raise ValueError("Variance must be strictly positive.")
                resolved.append(("scalar", variance, np.sqrt(variance)))
            else:
                cov_arr = _as_matrix(entry)
                resolved.append(("matrix", cov_arr, np.linalg.cholesky(cov_arr)))
        return tuple(resolved)

    cov_arr = _as_matrix(covariance)
    shared = ("matrix", cov_arr, np.linalg.cholesky(cov_arr))
    return tuple(shared for _ in range(n_components))


def generate_gaussian_mixture(
    *,
    n_samples: int,
    n_features: int,
    means: Union[Sequence[float], ArrayLike],
    covariance: Union[float, ArrayLike],
    noise_seed: Optional[SeedLike] = None,
    label_seed: Optional[SeedLike] = None,
    labels: Optional[Sequence[int]] = None,
    label_probs: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample from the Gaussian mixture defined in the synthetic-data note."""
    if n_samples <= 0 or n_features <= 0:
        raise ValueError("`n_samples` and `n_features` must be positive.")

    mean_matrix = _resolve_means(means, n_features)
    n_components = mean_matrix.shape[0]

    noise_rng = _as_random_state(noise_seed)
    fallback_seed = None
    if isinstance(noise_seed, (int, np.integer)):
        fallback_seed = int(noise_seed)
    label_rng = _choices_rng(label_seed, fallback_seed=fallback_seed)
    component_labels = _label_array(
        n_samples,
        n_components,
        labels=labels,
        label_probs=label_probs,
        rng=label_rng,
    )

    data = np.empty((n_samples, n_features), dtype=np.float64)
    covariances = _resolve_covariances(
        covariance,
        n_components=n_components,
        n_features=n_features,
    )
    for idx, label in enumerate(component_labels):
        cov_kind, cov_value, cache = covariances[label]
        if cov_kind == "scalar":
            noise = noise_rng.standard_normal(size=n_features) * cache  # cache holds sqrt(var)
        else:
            noise = noise_rng.standard_normal(size=n_features) @ cache.T  # cache holds Cholesky
        data[idx] = mean_matrix[label] + noise

    return data, component_labels


_SAMPLES_ORIGINAL_1_LABELS: Tuple[int, ...] = (
    0,
    0,
    1,
    0,
    3,
    0,
    2,
    1,
    3,
    0,
    1,
    1,
    0,
    0,
    1,
    0,
    2,
    0,
    1,
    1,
    3,
    0,
    0,
    3,
    0,
    3,
    2,
    1,
    3,
    3,
    1,
    3,
    0,
    2,
    2,
    3,
    1,
    1,
    3,
    2,
    2,
    2,
    2,
    3,
    1,
    2,
    1,
    0,
    3,
    0,
    3,
    3,
    2,
    0,
    1,
    1,
    0,
    1,
    2,
    1,
    0,
    0,
    2,
    0,
    1,
    0,
    2,
    2,
    3,
    2,
    3,
    3,
    2,
    0,
    1,
    0,
    3,
    1,
    3,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
    2,
    0,
    2,
    1,
    3,
    1,
    1,
)


def generate_samples_original_1() -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the matrix saved as ``samples_original_1.csv``."""
    data, labels = generate_gaussian_mixture(
        n_samples=100,
        n_features=1000,
        means=np.array([1.0, 4.0, 7.0, 10.0], dtype=np.float64),
        covariance=1.0,
        noise_seed=123,
        labels=_SAMPLES_ORIGINAL_1_LABELS,
    )
    return data, labels
