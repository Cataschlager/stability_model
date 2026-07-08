"""Validation orchestrator.

Usage: python -m validation.run_all
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_historical_scores(panel_df, indicator_df, eigenvalues, eigenvectors, n_comp):
    """Project historical indicator panel into composite scores using PCA components."""
    cols = indicator_df.columns.tolist()
    scores_by_year = []
    years = sorted(panel_df["year"].unique())
    all_countries = indicator_df.index.tolist()
    
    for yr in years:
        df_yr = panel_df[panel_df["year"] == yr]
        if df_yr.empty:
            continue
        
        # Pivot to wide (iso3 × indicator)
        wide_yr = df_yr.pivot_table(index="iso3", columns="indicator", values="value", aggfunc="mean")
        
        # Align to standard indicators and country universe
        wide_yr = wide_yr.reindex(columns=cols)
        wide_yr = wide_yr.reindex(index=all_countries)
        
        # Impute missing values with column median
        for col in cols:
            median_val = wide_yr[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            wide_yr[col] = wide_yr[col].fillna(median_val)
            
        # Winsorize [0.01, 0.99]
        for col in cols:
            lo = wide_yr[col].quantile(0.01)
            hi = wide_yr[col].quantile(0.99)
            if pd.notna(lo) and pd.notna(hi):
                wide_yr[col] = wide_yr[col].clip(lower=lo, upper=hi)
                
        # Robust z-score standardization
        X_clean = wide_yr.values.astype(float)
        X_std = np.zeros_like(X_clean)
        for j in range(len(cols)):
            col_vals = X_clean[:, j]
            med = np.median(col_vals)
            mad = np.median(np.abs(col_vals - med)) * 1.4826
            if mad > 1e-6:
                X_std[:, j] = (col_vals - med) / mad
            else:
                X_std[:, j] = 0.0
                
        # Project component scores
        evals = eigenvalues[:n_comp]
        evecs = eigenvectors[:, :n_comp]
        comp_scores = X_std @ evecs
        
        # Variance-weighted sum
        weights = evals / evals.sum()
        composite = comp_scores @ weights
        
        df_scores = pd.DataFrame({
            "iso3": all_countries,
            "year": yr,
            "composite_score": composite
        })
        scores_by_year.append(df_scores)
        
    return pd.concat(scores_by_year, ignore_index=True)


def main():
    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    output_dir = PROJECT_ROOT / config["paths"]["output"]
    clean_dir = PROJECT_ROOT / config["paths"]["clean_data"]

    logger.info("=" * 60)
    logger.info("VALIDATION PIPELINE")
    logger.info("=" * 60)

    # Load model outputs
    scores_path = output_dir / "composite_scores.parquet"
    if not scores_path.exists():
        logger.error("No model outputs. Run 'make model' first.")
        sys.exit(1)

    scores_df = pd.read_parquet(scores_path)
    countries = scores_df["iso3"].tolist()

    # Historical event validation
    logger.info("\n--- Historical Event Reconstruction ---")
    panel_path = clean_dir / "indicator_panel.parquet"
    if panel_path.exists():
        from validation.historical_events import run_all_episodes
        panel = pd.read_parquet(panel_path)
        
        # Load PCA components and metadata to build historical scores panel
        eigenvalues = np.load(output_dir / "eigenvalues.npy")
        eigenvectors = np.load(output_dir / "eigenvectors.npy")
        metadata_df = pd.read_parquet(output_dir / "model_metadata.parquet")
        n_comp = int(metadata_df["n_components"].iloc[0])
        indicator_matrix = pd.read_parquet(clean_dir / "indicator_matrix.parquet")
        
        scores_panel = build_historical_scores(panel, indicator_matrix, eigenvalues, eigenvectors, n_comp)
        episode_results = run_all_episodes(scores_panel, countries)
        logger.info("\n%s", episode_results.to_string(index=False))
        episode_results.to_parquet(output_dir / "validation_episodes.parquet", index=False)
    else:
        logger.warning("No indicator panel for historical validation.")

    # Rank correlation with FSI
    logger.info("\n--- Rank Correlation with FSI ---")
    fsi_path = clean_dir / "fsi.parquet"
    if fsi_path.exists():
        from scipy.stats import spearmanr
        fsi_df = pd.read_parquet(fsi_path)
        fsi_total = fsi_df[fsi_df["indicator"] == "fsi_total"]
        if not fsi_total.empty:
            merged = scores_df.merge(
                fsi_total.groupby("iso3")["value"].last().reset_index(),
                on="iso3", how="inner"
            )
            if len(merged) > 10:
                rho, p = spearmanr(merged["composite_score"], merged["value"])
                status = "✅ PASS" if abs(rho) >= 0.75 else "⚠️ BELOW TARGET"
                logger.info("Spearman ρ with FSI: %.3f (p=%.2e) - %s (target ≥ 0.75)", rho, p, status)

    logger.info("\n✅ Validation complete.")


if __name__ == "__main__":
    main()
