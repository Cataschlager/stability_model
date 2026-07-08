"""Bulk-download connectors for sources without APIs.

Covers: Polity5, Freedom House, Transparency International CPI,
Fragile States Index, SIPRI, ND-GAIN.
"""

import logging
from pathlib import Path

import pandas as pd
import pycountry

from ingestion.base import DataConnector

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

_fuzzy_iso3_cache = {}

def _fuzzy_iso3(name: str) -> str | None:
    """Try to resolve a country name to ISO-3166 alpha-3."""
    if not isinstance(name, str):
        return None
    name = name.strip()
    if name in _fuzzy_iso3_cache:
        return _fuzzy_iso3_cache[name]
    try:
        result = pycountry.countries.search_fuzzy(name)
        val = result[0].alpha_3 if result else None
    except LookupError:
        val = None
    _fuzzy_iso3_cache[name] = val
    return val


# ── Polity5 ──────────────────────────────────────────────────────────────────

class Polity5Connector(DataConnector):
    """Polity5 regime data (bulk Excel). Coverage ends ~2018."""

    source_name = "polity5"

    POLITY_URL = "https://www.systemicpeace.org/inscr/p5v2018.xls"

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "polity5.xls"
        if self._is_cached(cache_path):
            return cache_path
        logger.info("[polity5] Downloading from %s...", self.POLITY_URL)
        try:
            resp = self._http_get(self.config.get("polity5_url", self.POLITY_URL), timeout=120)
            cache_path.write_bytes(resp.content)
        except Exception as e:
            logger.error("[polity5] Download failed: %s. Place file manually.", e)
            cache_path.touch()
        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])
        try:
            raw = pd.read_excel(raw_path, engine="xlrd")
        except Exception:
            raw = pd.read_excel(raw_path)

        # Polity5 has columns: scode (3-letter), country, year, polity2, durable, xconst, etc.
        code_col = next((c for c in raw.columns if c.lower() in ("scode", "code")), None)
        name_col = next((c for c in raw.columns if c.lower() in ("country",)), None)

        indicators = {"polity2": "polity2_score", "durable": "regime_durability", "xconst": "executive_constraints"}
        frames = []
        for pol_col, ind_name in indicators.items():
            if pol_col not in raw.columns:
                continue
            sub = raw[["year", pol_col]].copy()
            if code_col:
                sub["iso3"] = raw[code_col].apply(lambda x: x.strip().upper() if isinstance(x, str) else None)
            elif name_col:
                sub["iso3"] = raw[name_col].apply(_fuzzy_iso3)
            else:
                continue
            sub = sub.rename(columns={pol_col: "value"})
            sub["indicator"] = ind_name
            sub["source"] = "POLITY5"
            sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
            # Polity uses -66, -77, -88 for special codes - treat as missing
            sub.loc[sub["value"].isin([-66, -77, -88]), "value"] = pd.NA
            frames.append(sub[["iso3", "year", "indicator", "value", "source"]].dropna(subset=["iso3"]))

        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["iso3", "year", "indicator", "value", "source"])

        if not self.countries.empty:
            result = result[result["iso3"].isin(set(self.country_codes))]
            end_year = int(self.countries["gdp_year"].iloc[0])
            start_year = end_year - 14
            result = result[(result["year"] >= start_year) & (result["year"] <= end_year)]

        logger.info("[polity5] Cleaned: %d rows (coverage ends ~2018)", len(result))
        return result


# ── Freedom House ────────────────────────────────────────────────────────────

class FreedomHouseConnector(DataConnector):
    """Freedom House Freedom in the World scores (bulk Excel)."""

    source_name = "freedom_house"

    FH_URL = "https://freedomhouse.org/sites/default/files/2024-02/All_data_FIW_2013-2024.xlsx"

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "freedom_house.xlsx"
        if self._is_cached(cache_path):
            return cache_path
        logger.info("[fh] Downloading Freedom House data...")
        try:
            resp = self._http_get(self.config.get("fh_url", self.FH_URL), timeout=120)
            cache_path.write_bytes(resp.content)
        except Exception as e:
            logger.error("[fh] Download failed: %s", e)
            cache_path.touch()
        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])
        
        for sheet in [1, 0, "FIW13-24", "FIW24"]:
            for header in [1, 0]:
                try:
                    raw_candidate = pd.read_excel(raw_path, sheet_name=sheet, header=header)
                    name_col_candidate = next((c for c in raw_candidate.columns if "country" in c.lower() or "territory" in c.lower()), None)
                    year_col_candidate = next((c for c in raw_candidate.columns if "edition" in c.lower() or "year" in c.lower()), None)
                    if name_col_candidate and year_col_candidate:
                        raw = raw_candidate
                        name_col = name_col_candidate
                        year_col = year_col_candidate
                        logger.info("[fh] Successfully loaded data from sheet: %s, header: %d", sheet, header)
                        break
                except Exception as e:
                    logger.warning("[fh] Failed to read sheet %s header %d: %s", sheet, header, e)
            if raw is not None:
                break

        if raw is None or name_col is None or year_col is None:
            logger.error("[fh] Could not identify country/year columns or load valid data sheet in Excel")
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        raw["iso3"] = raw[name_col].apply(_fuzzy_iso3)
        raw["year"] = pd.to_numeric(raw[year_col], errors="coerce").astype("Int64")

        indicators = {}
        for c in raw.columns:
            cl = c.strip().lower()
            if cl == "pr":
                indicators[c] = "fh_political_rights"
            elif cl == "cl":
                indicators[c] = "fh_civil_liberties"
            elif cl == "total":
                indicators[c] = "fh_aggregate_score"

        frames = []
        for col, ind_name in indicators.items():
            sub = raw[["iso3", "year"]].copy()
            sub["value"] = pd.to_numeric(raw[col], errors="coerce")
            sub["indicator"] = ind_name
            sub["source"] = "FREEDOM_HOUSE"
            frames.append(sub.dropna(subset=["iso3", "year", "value"]))

        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["iso3", "year", "indicator", "value", "source"])

        if not self.countries.empty:
            result = result[result["iso3"].isin(set(self.country_codes))]
        logger.info("[fh] Cleaned: %d rows", len(result))
        return result


# ── Transparency International CPI ──────────────────────────────────────────

class TransparencyIntlConnector(DataConnector):
    """Transparency International Corruption Perceptions Index (bulk download)."""

    source_name = "transparency_intl"

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "cpi.csv"
        if self._is_cached(cache_path):
            return cache_path
        # Try multiple URLs in order of reliability
        urls = [
            # Our World in Data curated version (stable)
            "https://ourworldindata.org/grapher/ti-corruption-perception-index.csv?v=1&csvType=full&useColumnShortNames=false",
            # TI direct data download (most authoritative)
            "https://images.transparencycdn.org/images/CPI2023_GlobalResultsTrends_20240116_211803.csv",
            # OWID GitHub raw
            "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Corruption%20Perceptions%20Index%20(Transparency%20International)/Corruption%20Perceptions%20Index%20(Transparency%20International).csv",
        ]
        logger.info("[ti] Downloading CPI data...")
        for url in urls:
            try:
                resp = self._http_get(url, timeout=60)
                if resp.status_code == 200 and len(resp.content) > 500:
                    cache_path.write_bytes(resp.content)
                    logger.info("[ti] Downloaded from: %s", url)
                    return cache_path
            except Exception as e:
                logger.warning("[ti] URL %s failed: %s", url, e)
        logger.error("[ti] All CPI download attempts failed.")
        cache_path.touch()
        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])
        raw = pd.read_csv(raw_path)

        # OWID format typically has: country, year, cpi_score
        name_col = next((c for c in raw.columns if "country" in c.lower()), None)
        code_col = next((c for c in raw.columns if "code" in c.lower()), None)

        if code_col:
            raw["iso3"] = raw[code_col].str.strip().str.upper()
        elif name_col:
            raw["iso3"] = raw[name_col].apply(_fuzzy_iso3)
        else:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        score_col = next((c for c in raw.columns if "cpi" in c.lower() or "score" in c.lower() or "corruption" in c.lower() or "perception" in c.lower()), None)
        if not score_col:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        year_col = next((c for c in raw.columns if "year" in c.lower()), None)
        if not year_col:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        result = raw[["iso3", year_col]].copy()
        result = result.rename(columns={year_col: "year"})
        result["value"] = pd.to_numeric(raw[score_col], errors="coerce")
        result["indicator"] = "cpi_score"
        result["source"] = "TI_CPI"
        result = result.dropna(subset=["iso3", "year", "value"])
        # Only use post-2012 data (methodology change)
        result = result[result["year"] >= 2012]

        if not self.countries.empty:
            result = result[result["iso3"].isin(set(self.country_codes))]
        logger.info("[ti] Cleaned: %d rows", len(result))
        return result


# ── Fragile States Index ────────────────────────────────────────────────────

class FSIConnector(DataConnector):
    """Fund for Peace Fragile States Index (bulk Excel)."""

    source_name = "fsi"

    FSI_INDICATORS = [
        ("C1", "fsi_security_apparatus"),
        ("C2", "fsi_factionalized_elites"),
        ("C3", "fsi_group_grievance"),
        ("E1", "fsi_economic_decline"),
        ("E2", "fsi_uneven_development"),
        ("E3", "fsi_human_flight"),
        ("P1", "fsi_state_legitimacy"),
        ("P2", "fsi_public_services"),
        ("P3", "fsi_human_rights"),
        ("S1", "fsi_demographic_pressures"),
        ("S2", "fsi_refugees_idps"),
        ("X1", "fsi_external_intervention"),
        ("Total", "fsi_total"),
    ]

    FSI_URLS = {
        2010: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2010.xlsx",
        2011: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2011.xlsx",
        2012: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2012.xlsx",
        2013: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2013.xlsx",
        2014: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2014.xlsx",
        2015: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2015.xlsx",
        2016: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2016.xlsx",
        2017: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2017.xlsx",
        2018: "https://fragilestatesindex.org/wp-content/uploads/2018/04/fsi-2018.xlsx",
        2019: "https://fragilestatesindex.org/wp-content/uploads/2019/04/fsi-2019.xlsx",
        2020: "https://fragilestatesindex.org/wp-content/uploads/2020/05/fsi-2020.xlsx",
        2021: "https://fragilestatesindex.org/wp-content/uploads/2021/05/fsi-2021.xlsx",
        2022: "https://fragilestatesindex.org/wp-content/uploads/2022/07/fsi-2022-download.xlsx",
        2023: "https://fragilestatesindex.org/wp-content/uploads/2023/06/FSI-2023-DOWNLOAD.xlsx",
    }

    def download(self) -> Path:
        self._ensure_dirs()
        # Download all years and save to fsi_{year}.xlsx
        for year, url in self.FSI_URLS.items():
            cache_path = self.raw_dir / f"fsi_{year}.xlsx"
            if self._is_cached(cache_path) and cache_path.stat().st_size > 1000:
                continue
            logger.info("[fsi] Downloading FSI %d from %s...", year, url)
            try:
                resp = self._http_get(url, timeout=60)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    cache_path.write_bytes(resp.content)
                    logger.info("[fsi] Downloaded FSI %d", year)
                else:
                    logger.warning("[fsi] URL %s returned code %d, content length %d", url, resp.status_code, len(resp.content))
            except Exception as e:
                logger.error("[fsi] Failed to download FSI %d: %s", year, e)
        return self.raw_dir

    def clean(self, raw_path: Path) -> pd.DataFrame:
        files = list(raw_path.glob("fsi_*.xlsx"))
        if not files:
            logger.warning("[fsi] No FSI files found in %s", raw_path)
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        all_frames = []
        for file_path in files:
            try:
                year = int(file_path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue

            try:
                raw = pd.read_excel(file_path)
            except Exception as e:
                logger.error("[fsi] Failed to read Excel %s: %s", file_path, e)
                continue

            name_col = next((c for c in raw.columns if "country" in c.lower()), None)
            if not name_col:
                logger.warning("[fsi] Country column not found in %s", file_path)
                continue

            raw["iso3"] = raw[name_col].apply(_fuzzy_iso3)
            raw["year"] = year

            for fsi_code, ind_name in self.FSI_INDICATORS:
                col = next((c for c in raw.columns if fsi_code.lower() == c.lower() or c.lower().startswith(fsi_code.lower() + ":") or c.lower().startswith(fsi_code.lower() + " ")), None)
                if not col and fsi_code == "Total":
                    col = next((c for c in raw.columns if "total" in c.lower() or "score" in c.lower()), None)
                if col:
                    sub = raw[["iso3", "year"]].copy()
                    sub["value"] = pd.to_numeric(raw[col], errors="coerce")
                    sub["indicator"] = ind_name
                    sub["source"] = "FSI"
                    all_frames.append(sub.dropna(subset=["iso3", "value"]))

        result = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(
            columns=["iso3", "year", "indicator", "value", "source"]
        )

        if not self.countries.empty:
            result = result[result["iso3"].isin(set(self.country_codes))]
        logger.info("[fsi] Cleaned: %d rows", len(result))
        return result


# ── SIPRI Military Expenditure ──────────────────────────────────────────────

class SIPRIConnector(DataConnector):
    """SIPRI Military Expenditure database."""

    source_name = "sipri"

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "sipri_milex.csv"
        if self._is_cached(cache_path):
            return cache_path
        # Try multiple OWID/GitHub sources
        urls = [
            "https://ourworldindata.org/grapher/military-expenditure-share-gdp.csv?v=1&csvType=full&useColumnShortNames=false",
            "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Military%20expenditure%20(%25%20of%20GDP)%20-%20SIPRI/Military%20expenditure%20(%25%20of%20GDP)%20-%20SIPRI.csv",
            # World Bank indicator: MS.MIL.XPND.GD.ZS
            "https://api.worldbank.org/v2/en/indicator/MS.MIL.XPND.GD.ZS?downloadformat=csv",
        ]
        logger.info("[sipri] Downloading military expenditure data...")
        for url in urls:
            try:
                resp = self._http_get(url, timeout=60)
                if resp.status_code == 200 and len(resp.content) > 200:
                    cache_path.write_bytes(resp.content)
                    logger.info("[sipri] Downloaded from: %s", url)
                    return cache_path
            except Exception as e:
                logger.warning("[sipri] URL %s failed: %s", url, e)

        # Fallback: use wbgapi which is already installed
        logger.info("[sipri] Trying wbgapi fallback for MS.MIL.XPND.GD.ZS...")
        try:
            import wbgapi as wb
            df = wb.data.DataFrame("MS.MIL.XPND.GD.ZS", time=range(2010, 2025))
            df.to_csv(cache_path)
            logger.info("[sipri] Downloaded via wbgapi: %d rows", len(df))
        except Exception as e:
            logger.error("[sipri] wbgapi fallback failed: %s", e)
            cache_path.touch()
        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        # Check if it's a wbgapi-style pivoted CSV (countries as rows, years as cols)
        raw = pd.read_csv(raw_path)
        if "economy" in raw.columns or (raw.index.name == "economy"):
            # wbgapi format: index=iso3, columns=YR2010..YR2024
            if "economy" not in raw.columns:
                raw = raw.reset_index()
            raw = raw.rename(columns={"economy": "iso3"})
            year_cols = [c for c in raw.columns if str(c).startswith("YR") or (str(c).isdigit() and 2000 < int(str(c)) < 2030)]
            if year_cols:
                long = raw.melt(id_vars=["iso3"], value_vars=year_cols, var_name="year", value_name="value")
                long["year"] = long["year"].astype(str).str.replace("YR", "").astype(int)
                long["indicator"] = "military_expenditure_pct_gdp"
                long["source"] = "SIPRI_WB"
                long = long.dropna(subset=["iso3", "year", "value"])
                if not self.countries.empty:
                    long = long[long["iso3"].isin(set(self.country_codes))]
                logger.info("[sipri] Cleaned (wbgapi): %d rows", len(long))
                return long
        return self._clean_owid_format(raw_path)

    def _clean_owid_format(self, raw_path: Path) -> pd.DataFrame:
        """Clean OWID-style CSV."""
        raw = pd.read_csv(raw_path)
        raw.columns = raw.columns.str.lower()
        code_col = next((c for c in raw.columns if "code" in c.lower()), None)
        name_col = next((c for c in raw.columns if "country" in c.lower() or "entity" in c.lower()), None)
        gdp_col = next((c for c in raw.columns if "gdp" in c.lower() or "share" in c.lower() or "military" in c.lower()), None)

        if code_col:
            raw["iso3"] = raw[code_col].str.strip().str.upper()
        elif name_col:
            raw["iso3"] = raw[name_col].apply(_fuzzy_iso3)
        else:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        if gdp_col:
            result = raw[["iso3", "year"]].copy()
            result["value"] = pd.to_numeric(raw[gdp_col], errors="coerce")
            result["indicator"] = "military_expenditure_pct_gdp"
            result["source"] = "SIPRI"
            result = result.dropna(subset=["iso3", "year", "value"])
        else:
            result = pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        if not self.countries.empty:
            result = result[result["iso3"].isin(set(self.country_codes))]
        logger.info("[sipri] Cleaned: %d rows", len(result))
        return result


# ── ND-GAIN ─────────────────────────────────────────────────────────────────

class NDGAINConnector(DataConnector):
    """ND-GAIN Country Index (climate vulnerability & readiness)."""

    source_name = "ndgain"

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "ndgain.zip"
        if self._is_cached(cache_path) and cache_path.stat().st_size > 10000:
            return cache_path

        url = "https://gain.nd.edu/assets/647440/ndgain_countryindex_2026.zip"
        logger.info("[ndgain] Downloading ND-GAIN ZIP from %s...", url)
        try:
            resp = self._http_get(url, timeout=120)
            if resp.status_code == 200 and len(resp.content) > 10000:
                cache_path.write_bytes(resp.content)
                logger.info("[ndgain] Downloaded ND-GAIN data (%d bytes)", len(resp.content))
                return cache_path
        except Exception as e:
            logger.error("[ndgain] Download failed: %s", e)

        # Fallback: if we already have it in root directory as ndgain_test.zip, copy it!
        test_zip = self.project_root / "ndgain_test.zip"
        if test_zip.exists():
            logger.info("[ndgain] Found fallback ndgain_test.zip in root. Copying...")
            import shutil
            shutil.copy(test_zip, cache_path)
            return cache_path

        cache_path.touch()
        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        import zipfile
        frames = []
        try:
            with zipfile.ZipFile(raw_path) as zf:
                targets = [
                    ("resources/gain/gain.csv", "ndgain_overall"),
                    ("resources/vulnerability/vulnerability.csv", "ndgain_vulnerability"),
                    ("resources/readiness/readiness.csv", "ndgain_readiness")
                ]
                for file_in_zip, ind_name in targets:
                    if file_in_zip in zf.namelist():
                        with zf.open(file_in_zip) as f:
                            raw = pd.read_csv(f)
                        
                        iso_col = next((c for c in raw.columns if "iso" in c.lower() or "code" in c.lower()), None)
                        if not iso_col:
                            continue
                        
                        raw = raw.rename(columns={iso_col: "iso3"})
                        raw["iso3"] = raw["iso3"].str.strip().str.upper()
                        
                        year_cols = [c for c in raw.columns if str(c).isdigit()]
                        if not year_cols:
                            continue
                        
                        long_df = raw.melt(id_vars=["iso3"], value_vars=year_cols, var_name="year", value_name="value")
                        long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype(int)
                        long_df["indicator"] = ind_name
                        long_df["source"] = "NDGAIN"
                        frames.append(long_df.dropna(subset=["iso3", "year", "value"]))
        except Exception as e:
            logger.error("[ndgain] Failed to unzip or clean: %s", e)
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["iso3", "year", "indicator", "value", "source"]
        )

        if not self.countries.empty:
            result = result[result["iso3"].isin(set(self.country_codes))]
        logger.info("[ndgain] Cleaned: %d rows", len(result))
        return result
