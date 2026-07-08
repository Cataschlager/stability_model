"""Indicator engineering pipeline.

Loads cleaned data from all connectors, applies temporal alignment,
sign orientation, missingness filtering, MICE imputation, winsorization,
and robust z-score standardization. Outputs the (125 × K) indicator matrix.

Usage: python -m features.build_indicators
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


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)


def load_clean_data(data_dir: Path) -> pd.DataFrame:
    """Load all parquet files from data/clean/ and merge into a single long-form DataFrame."""
    all_frames = []
    for pq_file in sorted(data_dir.glob("*.parquet")):
        # Exclude pipeline output files from being re-loaded as inputs
        if pq_file.name.startswith("indicator_") or pq_file.name.startswith("imputed_"):
            continue
        try:
            df = pd.read_parquet(pq_file)
            if not df.empty:
                all_frames.append(df)
                logger.info("Loaded %s: %d rows", pq_file.name, len(df))
        except Exception as e:
            logger.warning("Failed to load %s: %s", pq_file.name, e)

    if not all_frames:
        logger.error("No parquet files found in %s", data_dir)
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info("Combined: %d total rows, %d indicators, %d countries",
                 len(combined), combined["indicator"].nunique(), combined["iso3"].nunique())
    return combined


def align_temporal(df: pd.DataFrame, window: int = 15, forward_fill_max: int = 2) -> pd.DataFrame:
    """Align all series to annual frequency over the specified window.

    - Forward-fill governance indicators (max forward_fill_max years)
    - Linear interpolation for economic series (single-year gaps)
    """
    if df.empty:
        return df

    end_year = int(df["year"].max())
    start_year = end_year - window + 1
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

    # Define which sources are "slowly-changing" (governance)
    governance_sources = {"VDEM", "WB_WGI", "FREEDOM_HOUSE", "TI_CPI", "POLITY5", "FSI", "NDGAIN"}
    economic_sources = {"WB_WDI", "IMF_WEO", "SIPRI"}

    filled_frames = []
    for (iso3, indicator), group in df.groupby(["iso3", "indicator"]):
        group = group.sort_values("year")
        source = group["source"].iloc[0] if "source" in group.columns else ""

        # Create full year range
        full_years = pd.DataFrame({"year": range(start_year, end_year + 1)})
        merged = full_years.merge(group[["year", "value"]], on="year", how="left")

        if source in governance_sources:
            # Forward-fill up to forward_fill_max years
            merged["value"] = merged["value"].ffill(limit=forward_fill_max)
        elif source in economic_sources:
            # Linear interpolation for single gaps
            merged["value"] = merged["value"].interpolate(method="linear", limit=1)

        merged["iso3"] = iso3
        merged["indicator"] = indicator
        merged["source"] = source
        filled_frames.append(merged)

    result = pd.concat(filled_frames, ignore_index=True)
    logger.info("Temporal alignment: %d rows after filling", len(result))
    return result


def sign_orient(df: pd.DataFrame) -> pd.DataFrame:
    """Invert indicators where needed so higher = more unstable."""
    from features.pillar_config import get_inversion_map

    inversion_map = get_inversion_map()
    df = df.copy()

    for indicator_code, should_invert in inversion_map.items():
        if should_invert:
            mask = df["indicator"] == indicator_code
            if mask.any():
                df.loc[mask, "value"] = -df.loc[mask, "value"]
                logger.debug("Inverted %s (%d values)", indicator_code, mask.sum())

    # Special handling: current account - take absolute value of deficit
    mask_ca = df["indicator"] == "current_account_pct_gdp"
    if mask_ca.any():
        # Only penalize deficits (negative values become positive; surpluses → 0)
        df.loc[mask_ca, "value"] = df.loc[mask_ca, "value"].clip(upper=0).abs()

    logger.info("Sign orientation complete.")
    return df


def flag_missingness(df: pd.DataFrame, threshold: float = 0.40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag and exclude country-indicator pairs with >threshold missingness.

    Returns (filtered_df, missingness_report).
    """
    # Pivot to wide form to assess missingness
    if df.empty:
        return df, pd.DataFrame()

    # Count total possible observations per indicator
    n_years = df["year"].nunique()
    miss_stats = []

    for (iso3, indicator), group in df.groupby(["iso3", "indicator"]):
        n_missing = group["value"].isna().sum()
        n_total = n_years
        miss_rate = n_missing / n_total if n_total > 0 else 1.0
        miss_stats.append({
            "iso3": iso3,
            "indicator": indicator,
            "n_missing": n_missing,
            "n_total": n_total,
            "miss_rate": miss_rate,
            "excluded": miss_rate > threshold,
        })

    report = pd.DataFrame(miss_stats)
    excluded = report[report["excluded"]]

    if not excluded.empty:
        logger.info("Excluding %d country-indicator pairs with >%.0f%% missingness",
                     len(excluded), threshold * 100)
        exclude_pairs = set(zip(excluded["iso3"], excluded["indicator"]))
        df = df[~df.apply(lambda r: (r["iso3"], r["indicator"]) in exclude_pairs, axis=1)]

    return df, report


def impute_mice(wide_df: pd.DataFrame, n_imputations: int = 10,
                max_iter: int = 50, random_state: int = 42) -> list[pd.DataFrame]:
    """MICE imputation using sklearn IterativeImputer.

    Args:
        wide_df: Countries × Indicators matrix (rows=countries, cols=indicators)
        n_imputations: Number of multiply-imputed datasets
        max_iter: Max iterations for convergence
        random_state: Base random seed

    Returns:
        List of n_imputations imputed DataFrames.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    logger.info("Running MICE imputation (m=%d, max_iter=%d)...", n_imputations, max_iter)

    imputed_datasets = []
    numeric_cols = wide_df.select_dtypes(include=[np.number]).columns.tolist()

    for i in range(n_imputations):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max_iter,
            random_state=random_state + i,
            sample_posterior=True,  # Enable stochastic imputation for multiple datasets
        )
        imputed_values = imputer.fit_transform(wide_df[numeric_cols])
        imputed_df = wide_df.copy()
        imputed_df[numeric_cols] = imputed_values
        imputed_datasets.append(imputed_df)
        logger.info("  Imputation %d/%d complete.", i + 1, n_imputations)

    return imputed_datasets


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Winsorize each numeric column at specified percentiles."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def robust_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize using median and MAD.

    z = (x - median) / (MAD * 1.4826)
    The 1.4826 constant makes MAD consistent with SD for normal distributions.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median = df[col].median()
        mad = np.nanmedian(np.abs(df[col] - median)) * 1.4826
        if mad > 0:
            df[col] = (df[col] - median) / mad
        else:
            df[col] = 0.0  # Constant column
    return df


def long_to_wide(df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Convert long-form (iso3, year, indicator, value) to wide (iso3 × indicators)."""
    if year is not None:
        df = df[df["year"] == year]

    wide = df.pivot_table(index="iso3", columns="indicator", values="value", aggfunc="mean")
    logger.info("Wide matrix shape: %s", wide.shape)
    return wide


def build_indicator_matrix(config_path: str = "config.yaml") -> None:
    """Main entry point. Orchestrates the full indicator engineering pipeline.

    Outputs:
      - data/clean/indicator_matrix.parquet (125 × K for latest year)
      - data/clean/indicator_panel.parquet (125 × K × T long-form)
      - data/clean/missingness_report.csv
    """
    config = load_config(config_path)
    clean_dir = PROJECT_ROOT / config["paths"]["clean_data"]
    output_dir = PROJECT_ROOT / config["paths"]["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load all cleaned data
    logger.info("=" * 60)
    logger.info("INDICATOR ENGINEERING PIPELINE")
    logger.info("=" * 60)

    df = load_clean_data(clean_dir)
    if df.empty:
        logger.error("No data loaded. Run 'make ingest' first.")
        sys.exit(1)

    # Load country universe from data/countries.csv to restrict the dataset
    countries_csv = PROJECT_ROOT / "data" / "countries.csv"
    if not countries_csv.exists():
        logger.error("countries.csv not found at %s. Run IMF WEO connector first.", countries_csv)
        sys.exit(1)
    countries_df = pd.read_csv(countries_csv)
    universe_iso3 = set(countries_df["iso3"].tolist())
    logger.info("Filtering dataset to country universe of %d countries from countries.csv", len(universe_iso3))
    df = df[df["iso3"].isin(universe_iso3)]

    # Step 2: Temporal alignment
    window = config["temporal"]["window_years"]
    ff_max = config["temporal"]["forward_fill_max"]
    df = align_temporal(df, window=window, forward_fill_max=ff_max)

    # Step 3: Sign orientation
    df = sign_orient(df)

    # Step 4: Missingness filtering
    threshold = config["imputation"]["missingness_threshold"]
    df, miss_report = flag_missingness(df, threshold=threshold)
    miss_report.to_csv(clean_dir / "missingness_report.csv", index=False)

    # Step 5: Pivot to wide for the latest year
    end_year = config["temporal"].get("base_year")
    if end_year is None:
        end_year = int(df["year"].max())
    else:
        end_year = int(end_year)
    logger.info("Pivoting data for base year: %d", end_year)
    wide_latest = long_to_wide(df, year=end_year)
    wide_latest = wide_latest.reindex(index=sorted(universe_iso3))
    logger.info("Wide matrix shape after reindexing: %s", wide_latest.shape)

    # Step 5b: Pre-imputation scaling (to prevent MICE ill-conditioning and posterior draw explosion)
    logger.info("Applying pre-imputation scaling to indicator matrix...")
    wide_scaled = robust_zscore(wide_latest)

    # Step 6: MICE imputation
    n_imp = config["imputation"]["n_imputations"]
    max_iter = config["imputation"]["max_iter"]
    seed = config["random_seed"]
    imputed_datasets = impute_mice(wide_scaled, n_imputations=n_imp, max_iter=max_iter, random_state=seed)

    # Step 7: Pool imputed datasets (simple mean for point estimate)
    pooled = sum(imputed_datasets) / len(imputed_datasets)

    # Step 8: Winsorize
    w_lower = config["standardization"]["winsorize_lower"]
    w_upper = config["standardization"]["winsorize_upper"]
    pooled = winsorize(pooled, lower=w_lower, upper=w_upper)

    # Step 9: Post-imputation Robust z-score to ensure final normalization
    pooled = robust_zscore(pooled)

    # Save outputs
    pooled.to_parquet(clean_dir / "indicator_matrix.parquet")
    logger.info("Saved indicator matrix: %s", pooled.shape)

    # Also save the long-form panel (for temporal analysis)
    df.to_parquet(clean_dir / "indicator_panel.parquet", index=False)
    logger.info("Saved indicator panel: %d rows", len(df))

    # Save each imputed dataset for uncertainty propagation
    for i, imp_df in enumerate(imputed_datasets):
        # Winsorize and standardize each dataset so they are fully scaled
        imp_scaled = winsorize(imp_df, lower=w_lower, upper=w_upper)
        imp_scaled = robust_zscore(imp_scaled)
        imp_scaled.to_parquet(clean_dir / f"imputed_{i}.parquet")

    logger.info("✅ Indicator engineering complete.")


if __name__ == "__main__":
    build_indicator_matrix()
