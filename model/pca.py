"""PCA-based spectral decomposition of the indicator matrix.

Computes eigendecomposition, varimax rotation, composite scores,
and bootstrap confidence intervals following METHODOLOGY.md §4.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger(__name__)


def compute_pca(X: np.ndarray) -> dict:
    """Compute PCA on the correlation matrix of X.

    Args:
        X: (n_countries, n_indicators) matrix, already standardized.

    Returns:
        dict with keys:
            eigenvalues: (K,) array sorted descending
            eigenvectors: (K, K) matrix, columns = eigenvectors
            variance_explained: (K,) fraction of variance per component
            cumulative_variance: (K,) cumulative variance
            n_components_retained: int, per Kaiser + 80% cumvar rule
            scores: (n_countries, K) component scores
    """
    # Remove any all-NaN columns
    valid_cols = ~np.all(np.isnan(X), axis=0)
    X_clean = X[:, valid_cols]

    # Replace remaining NaNs with column means
    col_means = np.nanmean(X_clean, axis=0)
    for j in range(X_clean.shape[1]):
        mask = np.isnan(X_clean[:, j])
        X_clean[mask, j] = col_means[j]

    # Correlation matrix
    n, k = X_clean.shape
    corr_matrix = np.corrcoef(X_clean, rowvar=False)

    # Handle NaN in correlation matrix
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    # Eigendecomposition (symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance explained
    total_var = eigenvalues.sum()
    variance_explained = eigenvalues / total_var
    cumulative_variance = np.cumsum(variance_explained)

    # Component retention: Kaiser (eigenvalue > 1) AND cumvar >= 0.80
    kaiser_count = int(np.sum(eigenvalues > 1.0))
    cumvar_count = int(np.searchsorted(cumulative_variance, 0.80) + 1)
    n_components = max(kaiser_count, cumvar_count)
    n_components = min(n_components, k)  # Can't exceed number of indicators

    # Component scores: Z = X_clean @ V
    scores = X_clean @ eigenvectors

    logger.info("PCA: %d components retained (Kaiser=%d, CumVar80=%d), "
                 "explaining %.1f%% of variance",
                 n_components, kaiser_count, cumvar_count,
                 cumulative_variance[n_components - 1] * 100)

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "variance_explained": variance_explained,
        "cumulative_variance": cumulative_variance,
        "n_components_retained": n_components,
        "scores": scores,
        "correlation_matrix": corr_matrix,
        "X_clean": X_clean,
    }


def apply_varimax(loadings: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """Apply varimax rotation to the retained loadings matrix.

    Args:
        loadings: (K, r) matrix of retained eigenvectors scaled by sqrt(eigenvalue).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Rotated loadings matrix (K, r).
    """
    try:
        from factor_analyzer import Rotator
        rotator = Rotator(method="varimax")
        rotated = rotator.fit_transform(loadings)
        logger.info("Varimax rotation applied via factor_analyzer.")
        return rotated
    except ImportError:
        logger.warning("factor_analyzer not available. Using manual varimax implementation.")

    # Manual varimax implementation (Sherin, 1966)
    p, k = loadings.shape
    R = np.eye(k)
    d = 0

    for _ in range(max_iter):
        old_d = d
        Lambda = loadings @ R
        u, s, vt = np.linalg.svd(
            loadings.T @ (Lambda ** 3 - (Lambda * np.sum(Lambda ** 2, axis=0, keepdims=True)) / p)
        )
        R = u @ vt
        d = np.sum(s)
        if abs(d - old_d) < tol:
            break

    return loadings @ R


def compute_composite_scores(X: np.ndarray, eigenvalues: np.ndarray,
                              eigenvectors: np.ndarray, n_components: int) -> np.ndarray:
    """Compute composite instability score as variance-weighted sum of component scores.

    s_i = Σ_k (λ_k / Σ_j λ_j) · z_ik

    Args:
        X: (n_countries, n_indicators) standardized matrix
        eigenvalues: All eigenvalues
        eigenvectors: All eigenvectors
        n_components: Number of retained components

    Returns:
        (n_countries,) composite scores
    """
    # Retained eigenvalues and vectors
    evals = eigenvalues[:n_components]
    evecs = eigenvectors[:, :n_components]

    # Component scores
    scores = X @ evecs  # (n, r)

    # Variance weights
    weights = evals / evals.sum()

    # Composite score
    composite = scores @ weights  # (n,)

    logger.info("Composite scores: mean=%.3f, std=%.3f, range=[%.3f, %.3f]",
                 composite.mean(), composite.std(), composite.min(), composite.max())
    return composite


def sanity_check_fsi(composite_scores: np.ndarray, fsi_scores: np.ndarray) -> dict:
    """Check correlation between composite scores and FSI headline.

    Target: ρ ≥ 0.70 (both Pearson and Spearman).

    Returns:
        dict with pearson_r, spearman_rho, pearson_p, spearman_p, pass_pearson, pass_spearman
    """
    # Align: remove NaNs
    mask = ~(np.isnan(composite_scores) | np.isnan(fsi_scores))
    c = composite_scores[mask]
    f = fsi_scores[mask]

    if len(c) < 10:
        logger.warning("Too few overlapping countries (%d) for FSI sanity check.", len(c))
        return {"pearson_r": np.nan, "spearman_rho": np.nan, "pass": False, "n": len(c)}

    pr, pp = pearsonr(c, f)
    sr, sp = spearmanr(c, f)

    result = {
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_rho": sr,
        "spearman_p": sp,
        "pass_pearson": abs(pr) >= 0.70,
        "pass_spearman": abs(sr) >= 0.70,
        "n": len(c),
    }

    status = "✅ PASS" if result["pass_spearman"] else "⚠️ FAIL"
    logger.info("FSI sanity check: %s — Pearson r=%.3f (p=%.2e), Spearman ρ=%.3f (p=%.2e)",
                 status, pr, pp, sr, sp)
    return result


def bootstrap_pca(X: np.ndarray, B: int = 1000, n_components: int | None = None,
                  random_state: int = 42) -> dict:
    """Bootstrap PCA with Procrustes alignment for confidence intervals.

    Resamples countries (rows) with replacement, computes PCA, aligns
    to reference loadings via Procrustes, and stores bootstrap distributions.

    Args:
        X: (n_countries, n_indicators) matrix
        B: Number of bootstrap iterations
        n_components: Components to retain (default: auto-detect from reference PCA)
        random_state: Random seed

    Returns:
        dict with:
            eigenvalues_ci: (n_components, 2) — 5th and 95th percentiles
            scores_ci: (n_countries, 2) — CI for composite scores
            composite_bootstrap: (B, n_countries) — all bootstrap composite scores
    """
    rng = np.random.RandomState(random_state)

    # Reference PCA
    ref = compute_pca(X)
    if n_components is None:
        n_components = ref["n_components_retained"]

    ref_loadings = ref["eigenvectors"][:, :n_components] * np.sqrt(ref["eigenvalues"][:n_components])
    ref_composite = compute_composite_scores(
        ref["X_clean"], ref["eigenvalues"], ref["eigenvectors"], n_components
    )

    n = X.shape[0]
    boot_eigenvalues = np.zeros((B, n_components))
    boot_composites = np.zeros((B, n))

    for b in range(B):
        # Resample countries
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X[idx]

        try:
            res = compute_pca(X_boot)
            boot_loadings = res["eigenvectors"][:, :n_components] * np.sqrt(res["eigenvalues"][:n_components])

            # Procrustes alignment: find R minimizing ||boot_loadings @ R - ref_loadings||
            u, _, vt = np.linalg.svd(boot_loadings.T @ ref_loadings)
            R = u @ vt
            aligned = boot_loadings @ R

            boot_eigenvalues[b] = res["eigenvalues"][:n_components]

            # Compute composite for original (not resampled) countries using aligned loadings
            aligned_evecs = ref["eigenvectors"].copy()
            aligned_evecs[:, :n_components] = aligned / np.sqrt(res["eigenvalues"][:n_components] + 1e-10)
            boot_composites[b] = compute_composite_scores(
                ref["X_clean"], res["eigenvalues"], aligned_evecs, n_components
            )
        except Exception:
            boot_eigenvalues[b] = ref["eigenvalues"][:n_components]
            boot_composites[b] = ref_composite

        if (b + 1) % 100 == 0:
            logger.info("  Bootstrap %d/%d complete", b + 1, B)

    eigenvalues_ci = np.percentile(boot_eigenvalues, [5, 95], axis=0).T
    scores_ci = np.percentile(boot_composites, [5, 95], axis=0).T

    logger.info("Bootstrap PCA complete: B=%d, %d components", B, n_components)
    return {
        "eigenvalues_ci": eigenvalues_ci,
        "scores_ci": scores_ci,
        "composite_bootstrap": boot_composites,
        "reference_composite": ref_composite,
    }
