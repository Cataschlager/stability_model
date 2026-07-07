"""World Bank Worldwide Governance Indicators connector.

Fetches all 6 WGI dimensions via wbgapi (Source ID 3).
"""

import logging
from pathlib import Path

import pandas as pd

from ingestion.base import DataConnector

logger = logging.getLogger(__name__)

WGI_INDICATORS = {
    "GOV_WGI_VA.EST": "wgi_voice_accountability",
    "GOV_WGI_PV.EST": "wgi_political_stability",
    "GOV_WGI_GE.EST": "wgi_govt_effectiveness",
    "GOV_WGI_RQ.EST": "wgi_regulatory_quality",
    "GOV_WGI_RL.EST": "wgi_rule_of_law",
    "GOV_WGI_CC.EST": "wgi_control_corruption",
}


class WorldBankWGIConnector(DataConnector):
    """Connector for World Bank Worldwide Governance Indicators."""

    source_name = "worldbank_wgi"

    def download(self) -> Path:
        """Download WGI data via wbgapi with source=3."""
        import wbgapi as wb

        self._ensure_dirs()
        cache_path = self.raw_dir / "wgi_data.csv"

        if self._is_cached(cache_path):
            logger.info("[wgi] Using cached data at %s", cache_path)
            return cache_path

        logger.info("[wgi] Fetching WGI data via wbgapi (source=3)...")

        if self.countries.empty:
            end_year = 2024
        else:
            end_year = int(self.countries["gdp_year"].iloc[0])
        start_year = end_year - 14
        year_range = range(start_year, end_year + 1)

        all_frames = []
        for wb_code, name in WGI_INDICATORS.items():
            try:
                logger.info("[wgi] Fetching %s (%s)...", wb_code, name)
                df = wb.data.DataFrame(
                    wb_code,
                    economy="all",
                    time=year_range,
                    db=3,  # WGI database
                    labels=False,
                    columns="time",
                    numericTimeKeys=True,
                )
                if df is not None and not df.empty:
                    df = df.reset_index()
                    df = df.melt(id_vars=["economy"], var_name="year", value_name="value")
                    df["indicator"] = name
                    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
                    all_frames.append(df)
            except Exception as e:
                logger.warning("[wgi] Failed to fetch %s: %s", wb_code, e)

        if all_frames:
            combined = pd.concat(all_frames, ignore_index=True)
            combined.to_csv(cache_path, index=False)
            logger.info("[wgi] Cached %d rows", len(combined))
        else:
            pd.DataFrame().to_csv(cache_path, index=False)

        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """Normalize WGI data."""
        raw = pd.read_csv(raw_path)
        if raw.empty:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        raw = raw.rename(columns={"economy": "iso3"})
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")

        if not self.countries.empty:
            raw = raw[raw["iso3"].isin(set(self.country_codes))]

        raw["source"] = "WB_WGI"
        result = raw[["iso3", "year", "indicator", "value", "source"]].dropna(subset=["iso3", "year"])

        logger.info("[wgi] Cleaned: %d rows, %d countries", len(result), result["iso3"].nunique())
        return result
