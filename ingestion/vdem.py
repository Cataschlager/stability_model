"""V-Dem (Varieties of Democracy) connector.

Downloads bulk CSV and extracts democracy indices.
Uses streaming download to avoid OOM on the large file.
"""

import logging
from pathlib import Path

import pandas as pd
import pycountry

from ingestion.base import DataConnector

logger = logging.getLogger(__name__)

VDEM_URL = "https://github.com/vdeminstitute/vdemdata/raw/master/data/vdem.RData"

VDEM_INDICATORS = {
    "v2x_polyarchy": "vdem_polyarchy",
    "v2x_libdem": "vdem_liberal_democracy",
    "v2x_delibdem": "vdem_deliberative_democracy",
    "v2x_egaldem": "vdem_egalitarian_democracy",
    "v2x_partipdem": "vdem_participatory_democracy",
}


class VDemConnector(DataConnector):
    """Connector for V-Dem dataset (bulk RData download)."""

    source_name = "vdem"

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "vdem_core.RData"

        if self._is_cached(cache_path):
            logger.info("[vdem] Using cached data at %s", cache_path)
            return cache_path

        url = self.config.get("vdem_url", VDEM_URL)
        logger.info("[vdem] Downloading V-Dem RData from %s...", url)

        try:
            resp = self._http_get(url, timeout=300)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
            logger.info("[vdem] Download complete: %.1f MB", cache_path.stat().st_size / 1e6)
        except Exception as e:
            logger.error("[vdem] Download failed: %s. Skipping V-Dem.", e)
            cache_path.unlink(missing_ok=True)
            cache_path.touch()

        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            logger.warning("[vdem] Raw file empty - skipping V-Dem.")
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        logger.info("[vdem] Reading V-Dem RData...")
        try:
            import pyreadr
            result = pyreadr.read_r(str(raw_path))
            raw = result["vdem"]
        except Exception as e:
            logger.error("[vdem] Failed to read RData: %s", e)
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        cols_needed = ["country_text_id", "year"] + list(VDEM_INDICATORS.keys())
        raw = raw[[c for c in cols_needed if c in raw.columns]]

        # Ensure year is integer
        raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
        raw = raw.dropna(subset=["year"])
        raw["year"] = raw["year"].astype(int)

        # Map V-Dem country codes to ISO-3166 alpha-3
        raw["iso3"] = raw["country_text_id"].apply(self._vdem_to_iso3)
        raw = raw.dropna(subset=["iso3"])

        # Filter to country universe and time window
        if not self.countries.empty:
            raw = raw[raw["iso3"].isin(set(self.country_codes))]
            end_year = int(self.countries["gdp_year"].iloc[0])
        else:
            end_year = int(raw["year"].max())
        start_year = end_year - 14
        raw = raw[(raw["year"] >= start_year) & (raw["year"] <= end_year)]

        # Melt to long form
        frames = []
        for vdem_col, indicator_name in VDEM_INDICATORS.items():
            if vdem_col in raw.columns:
                sub = raw[["iso3", "year", vdem_col]].copy()
                sub = sub.rename(columns={vdem_col: "value"})
                sub["indicator"] = indicator_name
                sub["source"] = "VDEM"
                frames.append(sub[["iso3", "year", "indicator", "value", "source"]])

        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["iso3", "year", "indicator", "value", "source"]
        )
        logger.info("[vdem] Cleaned: %d rows", len(result))
        return result

    @staticmethod
    def _vdem_to_iso3(code: str) -> str | None:
        """V-Dem uses ISO-3166 alpha-3 codes mostly, with some exceptions."""
        if not isinstance(code, str):
            return None
        code = code.strip().upper()
        try:
            c = pycountry.countries.get(alpha_3=code)
            return c.alpha_3 if c else None
        except (KeyError, AttributeError):
            return None
