"""World Bank WDI connector.

Fetches economic and social indicators via the wbgapi package.
"""

import logging
from pathlib import Path

import pandas as pd

from ingestion.base import DataConnector

logger = logging.getLogger(__name__)

# WDI indicator codes and their human-readable names
WDI_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "SL.UEM.1524.ZS": "youth_unemployment_pct",
    "SI.POV.GINI": "gini_coefficient",
    "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp",
    "FI.RES.TOTL.MO": "reserves_months_imports",
    "SM.POP.REFG": "refugee_population",
    "VC.IDP.TOTL.HE": "idps",
}


class WorldBankWDIConnector(DataConnector):
    """Connector for World Bank World Development Indicators."""

    source_name = "worldbank_wdi"

    def download(self) -> Path:
        """Download WDI data via wbgapi."""
        import wbgapi as wb

        self._ensure_dirs()
        cache_path = self.raw_dir / "wdi_data.csv"

        if self._is_cached(cache_path):
            logger.info("[wdi] Using cached data at %s", cache_path)
            return cache_path

        logger.info("[wdi] Fetching WDI data via wbgapi...")

        # Determine year range from country universe
        if self.countries.empty:
            end_year = 2024
        else:
            end_year = int(self.countries["gdp_year"].iloc[0])
        start_year = end_year - 14
        year_range = range(start_year, end_year + 1)

        all_frames = []
        for wb_code, name in WDI_INDICATORS.items():
            try:
                logger.info("[wdi] Fetching %s (%s)...", wb_code, name)
                df = wb.data.DataFrame(
                    wb_code,
                    economy="all",
                    time=year_range,
                    labels=False,
                    columns="time",
                    numericTimeKeys=True,
                )
                if df is not None and not df.empty:
                    # wbgapi returns wide format: rows=countries, cols=years
                    df = df.reset_index()
                    df = df.melt(id_vars=["economy"], var_name="year", value_name="value")
                    df["indicator"] = name
                    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
                    all_frames.append(df)
                else:
                    logger.warning("[wdi] No data for %s", wb_code)
            except Exception as e:
                logger.warning("[wdi] Failed to fetch %s: %s", wb_code, e)

        if all_frames:
            combined = pd.concat(all_frames, ignore_index=True)
            combined.to_csv(cache_path, index=False)
            logger.info("[wdi] Cached %d rows", len(combined))
        else:
            pd.DataFrame().to_csv(cache_path, index=False)
            logger.error("[wdi] No data fetched.")

        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """Normalize WDI data to long-form."""
        raw = pd.read_csv(raw_path)
        if raw.empty:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        # wbgapi uses ISO-3166 alpha-3 codes in the 'economy' column
        raw = raw.rename(columns={"economy": "iso3"})
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")

        # Filter to country universe
        if not self.countries.empty:
            universe = set(self.country_codes)
            raw = raw[raw["iso3"].isin(universe)]

        raw["source"] = "WB_WDI"
        result = raw[["iso3", "year", "indicator", "value", "source"]].dropna(subset=["iso3", "year"])

        logger.info("[wdi] Cleaned: %d rows, %d countries, %d indicators",
                     len(result), result["iso3"].nunique(), result["indicator"].nunique())
        return result
