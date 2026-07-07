"""IMF World Economic Outlook connector.

Fetches country universe (top 125 by nominal GDP) and fiscal indicators.
Uses the imfp package for SDMX API access.
"""

import logging
from pathlib import Path

import pandas as pd
import pycountry

from ingestion.base import DataConnector

logger = logging.getLogger(__name__)

# IMF country code → ISO-3166 alpha-3 overrides for codes pycountry can't resolve
IMF_TO_ISO3_OVERRIDES = {
    "UVK": "XKX",  # Kosovo
    "WBG": "PSE",  # West Bank and Gaza → Palestine
    "TMP": "TLS",  # Timor-Leste
    "ZAR": "COD",  # Congo, Dem. Rep.
    "ROM": "ROU",  # Romania (old code)
}

# IMF WEO indicator codes we need
WEO_INDICATORS = {
    "NGDPD": "gdp_nominal_usd_bn",       # Nominal GDP, current prices, USD billions
    "GGXCNL_NGDP": "fiscal_balance_pct_gdp",  # General govt net lending/borrowing, % GDP
    "GGXWDG_NGDP": "gross_debt_pct_gdp",      # General govt gross debt, % GDP
}


class IMFWEOConnector(DataConnector):
    """Connector for IMF World Economic Outlook database.

    This connector:
    1. Fetches nominal GDP for all countries and selects the top 125.
    2. Saves the country universe to data/countries.csv.
    3. Fetches fiscal balance and gross debt indicators.
    """

    source_name = "imf_weo"

    def download(self) -> Path:
        """Download WEO data via imfp SDMX API."""
        import imfp

        self._ensure_dirs()
        cache_path = self.raw_dir / "weo_data.csv"

        if self._is_cached(cache_path):
            logger.info("[imf_weo] Using cached data at %s", cache_path)
            return cache_path

        logger.info("[imf_weo] Fetching WEO data via imfp...")

        all_frames = []
        for weo_code, name in WEO_INDICATORS.items():
            try:
                logger.info("[imf_weo] Fetching indicator: %s (%s)", weo_code, name)
                # Use imfp to get WEO data
                # The WEO database key in IMF's SDMX is typically "WEO"
                df = imfp.imf_dataset(
                    database_id="WEO",
                    indicator=weo_code,
                    freq="A",
                )
                if df is not None and not df.empty:
                    df["indicator"] = name
                    all_frames.append(df)
                else:
                    logger.warning("[imf_weo] No data returned for %s", weo_code)
            except Exception as e:
                logger.warning("[imf_weo] Failed to fetch %s via imfp: %s. Trying alternative...", weo_code, e)

        if not all_frames:
            # Fallback: try using the weo package for bulk download
            logger.info("[imf_weo] Falling back to weo package for bulk download...")
            try:
                import weo as weo_pkg
                # Download the most recent WEO vintage
                weo_pkg.download(year=2024, release="Oct", filename=str(self.raw_dir / "weo_raw.tsv"))
                w = weo_pkg.WEO(str(self.raw_dir / "weo_raw.tsv"))
                for weo_code, name in WEO_INDICATORS.items():
                    try:
                        series = w.get(weo_code)
                        if series is not None:
                            df = series.reset_index()
                            df["indicator"] = name
                            all_frames.append(df)
                    except Exception as e2:
                        logger.warning("[imf_weo] weo fallback failed for %s: %s", weo_code, e2)
            except Exception as e:
                logger.error("[imf_weo] Both imfp and weo packages failed: %s", e)

        if all_frames:
            combined = pd.concat(all_frames, ignore_index=True)
            combined.to_csv(cache_path, index=False)
            logger.info("[imf_weo] Cached %d rows to %s", len(combined), cache_path)
        else:
            # Create a minimal placeholder that will be caught by validation
            combined = pd.DataFrame(columns=["ref_area", "time_period", "value", "indicator"])
            combined.to_csv(cache_path, index=False)
            logger.error("[imf_weo] No data fetched. Check API connectivity.")

        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """Normalize WEO data and establish country universe."""
        raw = pd.read_csv(raw_path)

        if raw.empty:
            logger.error("[imf_weo] Raw data is empty.")
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        # Normalize column names (imfp returns various column naming conventions)
        col_map = {}
        for c in raw.columns:
            cl = c.lower().strip()
            if cl in ("ref_area", "iso", "country_code", "weo_country_code", "country"):
                col_map[c] = "country_code"
            elif cl in ("time_period", "year", "date"):
                col_map[c] = "year"
            elif cl in ("obs_value", "value"):
                col_map[c] = "value"
            elif cl == "indicator":
                col_map[c] = "indicator"
        raw = raw.rename(columns=col_map)

        # Ensure required columns exist
        for required in ["country_code", "year", "value", "indicator"]:
            if required not in raw.columns:
                logger.error("[imf_weo] Missing column '%s' in raw data.", required)
                return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        # Convert types
        raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

        # Map IMF country codes to ISO-3166 alpha-3
        raw["iso3"] = raw["country_code"].apply(self._imf_to_iso3)
        raw = raw.dropna(subset=["iso3", "year", "value"])

        # Establish country universe: top 125 by GDP in the most recent year
        gdp_df = raw[raw["indicator"] == "gdp_nominal_usd_bn"].copy()
        if gdp_df.empty:
            logger.error("[imf_weo] No GDP data found.")
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        # Cap most_recent_year at 2024 to prevent querying future projection years (2025-2031)
        most_recent_year = min(int(gdp_df["year"].max()), 2024)
        logger.info("[imf_weo] Most recent year with GDP data (capped at 2024): %d", most_recent_year)

        gdp_latest = (
            gdp_df[gdp_df["year"] == most_recent_year]
            .sort_values("value", ascending=False)
            .drop_duplicates(subset=["iso3"])
            .head(125)
        )

        # Save country universe
        countries = gdp_latest[["iso3"]].copy()
        countries["country_name"] = countries["iso3"].apply(self._iso3_to_name)
        countries["gdp_nominal_usd"] = gdp_latest["value"].values * 1e9  # Convert billions to USD
        countries["gdp_year"] = most_recent_year
        countries = countries.reset_index(drop=True)

        countries_path = self.project_root / "data" / "countries.csv"
        countries_path.parent.mkdir(parents=True, exist_ok=True)
        countries.to_csv(countries_path, index=False)
        logger.info("[imf_weo] Saved country universe (%d countries) to %s",
                     len(countries), countries_path)
        self._countries_df = countries

        # Filter all indicators to country universe and build long-form output
        universe_iso3 = set(countries["iso3"].tolist())
        filtered = raw[raw["iso3"].isin(universe_iso3)].copy()

        # Determine the 15-year window
        end_year = int(most_recent_year)
        start_year = end_year - 14
        filtered = filtered[(filtered["year"] >= start_year) & (filtered["year"] <= end_year)]

        result = filtered[["iso3", "year", "indicator", "value"]].copy()
        result["source"] = "IMF_WEO"

        logger.info("[imf_weo] Cleaned data: %d rows, %d countries, years %d–%d",
                     len(result), result["iso3"].nunique(), start_year, end_year)
        return result

    @staticmethod
    def _imf_to_iso3(code: str) -> str | None:
        """Convert IMF country code to ISO-3166 alpha-3."""
        if not isinstance(code, str):
            return None
        code = code.strip().upper()

        # Check overrides first
        if code in IMF_TO_ISO3_OVERRIDES:
            return IMF_TO_ISO3_OVERRIDES[code]

        # Try pycountry lookup
        try:
            country = pycountry.countries.get(alpha_3=code)
            if country:
                return country.alpha_3
        except (KeyError, AttributeError):
            pass

        # Try numeric code
        try:
            country = pycountry.countries.get(numeric=code)
            if country:
                return country.alpha_3
        except (KeyError, AttributeError):
            pass

        # Try alpha-2
        try:
            country = pycountry.countries.get(alpha_2=code[:2])
            if country:
                return country.alpha_3
        except (KeyError, AttributeError):
            pass

        return None

    @staticmethod
    def _iso3_to_name(iso3: str) -> str:
        """Convert ISO-3166 alpha-3 to country name."""
        try:
            country = pycountry.countries.get(alpha_3=iso3)
            return country.name if country else iso3
        except (KeyError, AttributeError):
            return iso3
