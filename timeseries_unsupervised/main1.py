#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "scikit-learn",
# ]
# ///
"""
Random-walk time-series generator + latent-space visualization (PCA / t-SNE)

Dependencies:
  pip install numpy matplotlib scikit-learn
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def generate_random_walks(
    n_series: int = 100,
    length: int = 200,
    drift: float = 0.0,
    step_std: float = 1.0,
    start_value: float = 0.0,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Generate n_series random-walk time series of given length.
    Returns array of shape (n_series, length).
    """
    rng = np.random.default_rng(seed)
    print(f"rng: {rng}")
    steps = drift + step_std * rng.standard_normal(size=(n_series, length))
    print(f"steps: {steps}")
    walks = start_value + np.cumsum(steps, axis=1)
    print(f"walks: {walks}")
    #sys.exit(0)
    return walks


def series_to_matrix(
    walks: np.ndarray,
    representation: str = "levels_zscore",
) -> np.ndarray:
    """
    Convert time series to a 2D matrix (n_series, n_features) suitable for PCA/t-SNE.

    representation:
      - "levels": raw levels flattened (n_features = length)
      - "levels_zscore": per-series z-score then flatten (default)
      - "diff": first-differences (n_features = length-1)
      - "diff_zscore": per-series z-score of diffs then flatten
    """
    if walks.ndim != 2:
        raise ValueError("walks must have shape (n_series, length)")

    if representation == "levels":
        X = walks.copy()
    elif representation == "levels_zscore":
        mu = walks.mean(axis=1, keepdims=True)
        sd = walks.std(axis=1, keepdims=True) + 1e-12
        X = (walks - mu) / sd
    elif representation == "diff":
        X = np.diff(walks, axis=1)
    elif representation == "diff_zscore":
        diffs = np.diff(walks, axis=1)
        mu = diffs.mean(axis=1, keepdims=True)
        sd = diffs.std(axis=1, keepdims=True) + 1e-12
        X = (diffs - mu) / sd
    else:
        raise ValueError(f"Unknown representation: {representation}")

    return X  # shape: (n_series, n_features)


def embed_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    PCA embedding to n_components dimensions.
    (We also standardize features across the dataset to avoid scale dominance.)
    """
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(Xs)


def embed_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """
    t-SNE embedding to n_components dimensions.

    Note: perplexity must be < n_samples. If None, pick a safe default based on n.
    """
    n_samples = X.shape[0]
    if perplexity is None:
        # rule-of-thumb; keep it valid for small sample sizes
        perplexity = float(min(30, max(5, (n_samples - 1) // 3)))
    if perplexity >= n_samples:
        perplexity = float(max(2, n_samples - 1))

    Xs = StandardScaler().fit_transform(X)
    tsne = TSNE(
        n_components=n_components,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    )
    return tsne.fit_transform(Xs)


def plot_time_series(walks: np.ndarray, max_to_plot: int = 20) -> None:
    n = walks.shape[0]
    k = min(n, max_to_plot)
    plt.figure()
    for i in range(k):
        plt.plot(walks[i], alpha=0.8, linewidth=1, label=str(i))
    plt.title(f"Random walk time series (showing {k} of {n})")
    plt.xlabel("time")
    plt.ylabel("value")

    # If you plot too many series, the legend becomes unreadable.
    # This keeps it usable while still giving labels.
    if k <= 25:
        plt.legend(
            title="series id",
            ncol=2 if k > 12 else 1,
            fontsize=8,
            title_fontsize=9,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        plt.tight_layout(rect=[0, 0, 0.82, 1])  # leave space for legend
    else:
        plt.tight_layout()

    plt.show()
    #n = walks.shape[0]
    #k = min(n, max_to_plot)
    #plt.figure()
    #for i in range(k):
    #    plt.plot(walks[i], alpha=0.7, linewidth=1)
    #plt.title(f"Random walk time series (showing {k} of {n})")
    #plt.xlabel("time")
    #plt.ylabel("value")
    #plt.tight_layout()
    #plt.show()


def plot_embedding(Y: np.ndarray, title: str) -> None:
    if Y.shape[1] != 2:
        raise ValueError("This plot function expects a 2D embedding (n_samples, 2).")

    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], s=35, alpha=0.85)
    for i, (x, y) in enumerate(Y):
        plt.text(x, y, str(i), fontsize=8, alpha=0.75)

    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.show()


def main() -> None:
    # --- Config ---
    N_SERIES = 100      # <= 100 (or any number you want)
    LENGTH = 300        # time steps
    DRIFT = 0.02        # average step
    STEP_STD = 1.0      # step volatility
    SEED = 42

    REPRESENTATION = "levels_zscore"  # try: "diff_zscore" for “shape of increments”
    PLOT_MAX_SERIES = 25

    # --- Generate data ---
    walks = generate_random_walks(
        n_series=N_SERIES,
        length=LENGTH,
        drift=DRIFT,
        step_std=STEP_STD,
        seed=SEED,
    )

    # --- Convert to feature matrix for embeddings ---
    X = series_to_matrix(walks, representation=REPRESENTATION)

    # --- Embeddings ---
    Y_pca = embed_pca(X, n_components=2)
    Y_tsne = embed_tsne(X, n_components=2, perplexity=None, seed=0)

    # --- Visualize ---
    plot_time_series(walks, max_to_plot=PLOT_MAX_SERIES)
    plot_embedding(Y_pca, f"PCA latent space (representation={REPRESENTATION})")
    plot_embedding(Y_tsne, f"t-SNE latent space (representation={REPRESENTATION})")


if __name__ == "__main__":
    main()
