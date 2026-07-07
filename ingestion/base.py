"""Abstract base class for all data connectors."""

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
import requests

logger = logging.getLogger(__name__)


class DataConnector(ABC):
    """Base class for data source connectors.

    Each connector handles download, cleaning, validation, and parquet export
    for a single data source. Raw downloads are cached under
    data/raw/{source_name}/{date}/ and normalized outputs go to data/clean/.
    """

    source_name: str = "base"

    def __init__(self, countries_path: Path | None = None, config: dict | None = None):
        load_dotenv()
        self.config = config or {}
        self.project_root = Path(__file__).resolve().parent.parent
        
        # Scan for the most recent non-empty cached directory, falling back to today's date
        base_raw_dir = self.project_root / "data" / "raw" / self.source_name
        self.raw_dir = base_raw_dir / date.today().isoformat()
        if base_raw_dir.exists():
            date_dirs = sorted([d for d in base_raw_dir.iterdir() if d.is_dir()], reverse=True)
            for d in date_dirs:
                try:
                    files = [f for f in d.iterdir() if f.is_file() and f.stat().st_size > 100]
                    if files:
                        self.raw_dir = d
                        logger.info("[%s] Found existing cached directory from %s. Using it.", self.source_name, d.name)
                        break
                except Exception:
                    pass

        self.clean_dir = self.project_root / "data" / "clean"
        self.countries_path = countries_path or (self.project_root / "data" / "countries.csv")
        self._countries_df: pd.DataFrame | None = None

    @property
    def countries(self) -> pd.DataFrame:
        """Load the country universe. Lazy-loaded and cached."""
        if self._countries_df is None:
            if self.countries_path.exists():
                self._countries_df = pd.read_csv(self.countries_path)
            else:
                logger.warning("countries.csv not found at %s — run IMF WEO connector first.", self.countries_path)
                self._countries_df = pd.DataFrame(columns=["iso3", "country_name", "gdp_nominal_usd", "gdp_year"])
        return self._countries_df

    @property
    def country_codes(self) -> list[str]:
        """List of ISO-3166 alpha-3 codes in the universe."""
        return self.countries["iso3"].tolist()

    def _ensure_dirs(self) -> None:
        """Create cache and output directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)

    def _is_cached(self, path: Path) -> bool:
        """Check if a file exists in raw cache and is non-empty."""
        return path.exists() and path.stat().st_size > 100

    @abstractmethod
    def download(self) -> Path:
        """Download raw data from the source. Returns path to raw file(s)."""
        ...

    @abstractmethod
    def clean(self, raw_path: Path) -> pd.DataFrame:
        """Normalize raw data to a long-form DataFrame.

        For country-level indicators, columns: iso3, year, indicator, value, source.
        For dyadic data, columns: iso3_i, iso3_j, year, value, source.
        """
        ...

    def validate(self, df: pd.DataFrame) -> bool:
        """Basic validation: no all-NaN indicator columns, expected columns exist."""
        if df.empty:
            logger.warning("[%s] DataFrame is empty after cleaning.", self.source_name)
            return False

        required_cols = {"iso3", "year", "indicator", "value", "source"}
        dyadic_cols = {"iso3_i", "iso3_j", "year", "value", "source"}
        has_required = required_cols.issubset(df.columns) or dyadic_cols.issubset(df.columns)
        if not has_required:
            logger.error("[%s] Missing required columns. Got: %s", self.source_name, list(df.columns))
            return False

        if "indicator" in df.columns:
            for ind in df["indicator"].unique():
                vals = df.loc[df["indicator"] == ind, "value"]
                if vals.isna().all():
                    logger.warning("[%s] Indicator '%s' is all NaN.", self.source_name, ind)
                if vals.dropna().nunique() <= 1:
                    logger.warning("[%s] Indicator '%s' is constant.", self.source_name, ind)

        return True

    def to_parquet(self, df: pd.DataFrame) -> Path:
        """Save cleaned DataFrame as parquet."""
        self._ensure_dirs()
        out_path = self.clean_dir / f"{self.source_name}.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("[%s] Saved %d rows to %s", self.source_name, len(df), out_path)
        return out_path

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline: download → clean → validate → save."""
        logger.info("[%s] Starting ingestion...", self.source_name)
        self._ensure_dirs()
        try:
            raw_path = self.download()
            df = self.clean(raw_path)
            if self.validate(df):
                self.to_parquet(df)
                logger.info("[%s] ✅ Ingestion complete. %d rows.", self.source_name, len(df))
            else:
                logger.warning("[%s] ⚠️ Validation issues detected. Data saved anyway.", self.source_name)
                self.to_parquet(df)
            return df
        except Exception as e:
            logger.error("[%s] ❌ Ingestion failed: %s", self.source_name, e, exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def _http_get(url: str, params: dict = None, headers: dict = None,
                  max_retries: int = 3, timeout: int = 60) -> requests.Response:
        """HTTP GET with retries and exponential backoff."""
        headers_merged = headers.copy() if headers else {}
        if "User-Agent" not in headers_merged:
            headers_merged["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, headers=headers_merged, timeout=timeout)
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("HTTP request failed (attempt %d/%d): %s. Retrying in %ds...",
                                   attempt + 1, max_retries, e, wait)
                    time.sleep(wait)
                else:
                    raise

    @staticmethod
    def _get_env(key: str, required: bool = False) -> str | None:
        """Read environment variable, optionally raising if missing."""
        val = os.getenv(key)
        if required and not val:
            raise EnvironmentError(f"Required environment variable '{key}' is not set. See SETUP.md.")
        return val
