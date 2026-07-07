"""Calibration of α via leave-one-year-out cross-validation.

Per METHODOLOGY.md §6.5.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calibrate_alpha(composite_scores_panel: dict[int, np.ndarray],
                    W_panel: dict[int, np.ndarray],
                    grid_min: float = 0.05, grid_max: float = 0.95,
                    grid_step: float = 0.01) -> dict:
    """Calibrate α to minimize one-year-ahead prediction error.

    For each held-out year t:
        1. s(t) = composite scores for year t
        2. W(t) = coupling matrix for year t
        3. Predict: x̂(t+1) = α·W(t)·s(t) + (1-α)·s(t)
        4. Compare to x_actual(t+1)

    Args:
        composite_scores_panel: {year: (N,) scores} dict
        W_panel: {year: (N,N) coupling matrix} dict
        grid_min, grid_max, grid_step: Grid search parameters

    Returns:
        dict with best_alpha, rmse_by_alpha, cv_results
    """
    from model.dynamics import steady_state

    years = sorted(set(composite_scores_panel.keys()) & set(W_panel.keys()))
    if len(years) < 3:
        logger.warning("Too few years for cross-validation (%d). Using default α=0.40", len(years))
        return {"best_alpha": 0.40, "rmse_by_alpha": {}, "cv_results": []}

    alphas = np.arange(grid_min, grid_max + grid_step / 2, grid_step)
    rmse_by_alpha = {}

    for alpha in alphas:
        errors = []
        for i in range(len(years) - 1):
            t = years[i]
            t1 = years[i + 1]

            s_t = composite_scores_panel[t]
            W_t = W_panel[t]
            x_actual = composite_scores_panel[t1]

            try:
                x_pred = alpha * (W_t @ s_t) + (1 - alpha) * s_t
                rmse = np.sqrt(np.mean((x_pred - x_actual) ** 2))
                errors.append(rmse)
            except Exception:
                continue

        if errors:
            rmse_by_alpha[alpha] = np.mean(errors)

    if not rmse_by_alpha:
        logger.warning("Calibration failed. Using default α=0.40")
        return {"best_alpha": 0.40, "rmse_by_alpha": {}, "cv_results": []}

    # Find best alpha
    best_alpha = min(rmse_by_alpha, key=rmse_by_alpha.get)
    best_rmse = rmse_by_alpha[best_alpha]

    logger.info("Calibrated α=%.3f (RMSE=%.4f). Grid searched %.2f to %.2f.",
                 best_alpha, best_rmse, grid_min, grid_max)

    return {
        "best_alpha": best_alpha,
        "best_rmse": best_rmse,
        "rmse_by_alpha": rmse_by_alpha,
    }


def bootstrap_alpha(composite_scores_panel: dict[int, np.ndarray],
                    W_panel: dict[int, np.ndarray],
                    B: int = 500, random_state: int = 42,
                    **kwargs) -> dict:
    """Bootstrap CI for calibrated α.

    Resamples countries within each fold.

    Returns:
        dict with alpha_ci (5th, 95th percentile), bootstrap_alphas
    """
    rng = np.random.RandomState(random_state)
    years = sorted(set(composite_scores_panel.keys()) & set(W_panel.keys()))
    n = next(iter(composite_scores_panel.values())).shape[0]

    boot_alphas = []
    for b in range(B):
        # Resample country indices
        idx = rng.choice(n, size=n, replace=True)

        # Create resampled panels
        resampled_scores = {y: composite_scores_panel[y][idx] for y in years}
        resampled_W = {y: W_panel[y][np.ix_(idx, idx)] for y in years}

        result = calibrate_alpha(resampled_scores, resampled_W, **kwargs)
        boot_alphas.append(result["best_alpha"])

        if (b + 1) % 100 == 0:
            logger.info("  Bootstrap α: %d/%d", b + 1, B)

    alpha_ci = np.percentile(boot_alphas, [5, 95])
    logger.info("Bootstrap α: median=%.3f, 90%% CI=[%.3f, %.3f]",
                 np.median(boot_alphas), alpha_ci[0], alpha_ci[1])

    return {
        "alpha_ci": alpha_ci,
        "bootstrap_alphas": np.array(boot_alphas),
        "alpha_median": np.median(boot_alphas),
    }
