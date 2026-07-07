"""Conflict data connectors: ACLED, UCDP, UN Population, FAO, BIS.

These connectors require API keys (read from .env).
"""

import logging
from pathlib import Path

import pandas as pd
import pycountry

from ingestion.base import DataConnector

logger = logging.getLogger(__name__)


_fuzzy_iso3_cache = {}

def _fuzzy_iso3(name: str) -> str | None:
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


class ACLEDConnector(DataConnector):
    """ACLED Armed Conflict Location & Event Data Project.

    New auth system (myACLED, 2025): OAuth2 password-grant.
    No static API key — authenticate with email + account password
    to receive a 24-hour Bearer token.

    Required .env variables:
        ACLED_EMAIL    -- your myACLED account email
        ACLED_PASSWORD -- your myACLED account password

    Register free: https://acleddata.com/user/register
    API docs:      https://acleddata.com/api-documentation/getting-started
    """

    source_name = "acled"

    ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
    ACLED_API_URL   = "https://acleddata.com/api/acled/read"
    PAGE_SIZE = 5000  # rows per paginated request
    EVENT_TYPES = ["Battles", "Violence against civilians", "Explosions/Remote violence",
                   "Riots", "Protests"]

    def _get_bearer_token(self, email: str, password: str) -> str | None:
        """POST to /oauth/token with password grant; returns access_token string."""
        import requests
        try:
            resp = requests.post(
                self.ACLED_TOKEN_URL,
                data={
                    "username": email,
                    "password": password,
                    "grant_type": "password",
                    "client_id": "acled",
                    "scope": "authenticated",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            token_data = resp.json()
            token = token_data.get("access_token")
            if token:
                logger.info("[acled] Bearer token obtained (expires in %ss).",
                            token_data.get("expires_in", "?"))
                return token
            else:
                logger.error("[acled] Token endpoint returned no access_token: %s", token_data)
                return None
        except Exception as e:
            logger.error("[acled] Failed to obtain OAuth token: %s", e)
            return None

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "acled_events.csv"
        if self._is_cached(cache_path):
            return cache_path

        email    = self._get_env("ACLED_EMAIL")
        password = self._get_env("ACLED_PASSWORD")

        if not email or not password:
            logger.warning(
                "[acled] ACLED_EMAIL or ACLED_PASSWORD not set in .env. "
                "Add to .env:\n  ACLED_EMAIL=your@email.com\n  ACLED_PASSWORD=yourpassword"
            )
            cache_path.touch()
            return cache_path

        # Step 1: get OAuth Bearer token (valid 24 hours)
        token = self._get_bearer_token(email, password)
        if not token:
            cache_path.touch()
            return cache_path

        auth_header = {"Authorization": f"Bearer {token}"}

        if self.countries.empty:
            end_year = 2024
        else:
            end_year = int(self.countries["gdp_year"].iloc[0])
        start_year = end_year - 14

        # Step 2: Chunk countries into groups of 10 to avoid query length limits and reduce request counts
        chunk_size = 10
        iso_numeric_list = []
        for code in self.country_codes:
            try:
                c = pycountry.countries.get(alpha_3=code)
                if c and c.numeric:
                    iso_numeric_list.append(str(int(c.numeric)))
            except Exception:
                pass

        if not iso_numeric_list:
            logger.warning("[acled] No country numeric codes resolved. Querying globally.")
            chunks = [[]]
        else:
            chunks = [iso_numeric_list[i:i + chunk_size] for i in range(0, len(iso_numeric_list), chunk_size)]

        all_events = []
        import requests
        import time

        for year in range(start_year, end_year + 1):
            for chunk_idx, chunk in enumerate(chunks):
                offset = 0
                last_ids = set()
                while True:
                    try:
                        params = {
                            "_format": "json",
                            "event_date": f"{year}-01-01|{year}-12-31",
                            "event_date_where": "BETWEEN",
                            "limit": self.PAGE_SIZE,
                            "offset": offset,
                        }
                        if chunk:
                            params["iso"] = "|".join(chunk)

                        resp = requests.get(
                            self.ACLED_API_URL,
                            params=params,
                            headers=auth_header,
                            timeout=120,
                        )
                        data = resp.json()

                        if resp.status_code == 401 or "access denied" in str(data).lower():
                            logger.error("[acled] Auth rejected (401). Check credentials.")
                            break

                        rows = data.get("data", [])
                        if not rows:
                            break

                        # Check for infinite loop (API ignoring offset and returning duplicate data)
                        current_ids = {r.get("event_id_cnty") for r in rows if r.get("event_id_cnty")}
                        if current_ids and current_ids == last_ids:
                            logger.warning("[acled] API returned duplicate data (ignoring offset). Breaking loop.")
                            break
                        last_ids = current_ids

                        page_df = pd.DataFrame(rows)
                        # Use actual year from event_date to prevent incorrect historical mapping
                        if "event_date" in page_df.columns:
                            try:
                                page_df["year"] = pd.to_datetime(page_df["event_date"]).dt.year
                            except Exception:
                                if "year" not in page_df.columns:
                                    page_df["year"] = year
                        elif "year" not in page_df.columns:
                            page_df["year"] = year

                        all_events.append(page_df)
                        logger.info(
                            "[acled] Year %d Chunk %d/%d offset %d: %d events",
                            year, chunk_idx + 1, len(chunks), offset, len(rows)
                        )

                        if len(rows) < self.PAGE_SIZE:
                            break  # last page
                        offset += self.PAGE_SIZE
                        time.sleep(0.1)

                    except Exception as e:
                        logger.warning("[acled] Failed year %d chunk %d offset %d: %s", year, chunk_idx + 1, offset, e)
                        break

                time.sleep(0.2)

        if all_events:
            combined = pd.concat(all_events, ignore_index=True)
            combined.to_csv(cache_path, index=False)
            logger.info("[acled] Saved %d total events.", len(combined))
        else:
            pd.DataFrame().to_csv(cache_path, index=False)
            logger.warning("[acled] No events fetched. Check credentials.")

        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        raw = pd.read_csv(raw_path)
        if raw.empty:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        # Resolve ISO codes
        iso_col = next((c for c in raw.columns if c.lower() == "iso"), None)
        country_col = next((c for c in raw.columns if c.lower() == "country"), None)

        if iso_col:
            # ACLED uses ISO numeric codes
            raw["iso3"] = raw[iso_col].apply(lambda x: self._numeric_to_iso3(x))
        elif country_col:
            raw["iso3"] = raw[country_col].apply(_fuzzy_iso3)
        else:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        # Aggregate to annual counts by event type
        event_col = next((c for c in raw.columns if "event_type" in c.lower()), None)
        if not event_col:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        frames = []
        event_type_map = {
            "Battles": "acled_battles",
            "Violence against civilians": "acled_violence_civilians",
            "Explosions/Remote violence": "acled_explosions",
            "Riots": "acled_riots",
            "Protests": "acled_protests",
        }

        for event_type, ind_name in event_type_map.items():
            sub = raw[raw[event_col] == event_type]
            counts = sub.groupby(["iso3", "year"]).size().reset_index(name="value")
            counts["indicator"] = ind_name
            counts["source"] = "ACLED"
            frames.append(counts)

        # Also total fatalities
        fat_col = next((c for c in raw.columns if "fatal" in c.lower()), None)
        if fat_col:
            raw["fatalities_num"] = pd.to_numeric(raw[fat_col], errors="coerce").fillna(0)
            fatalities = raw.groupby(["iso3", "year"])["fatalities_num"].sum().reset_index()
            fatalities = fatalities.rename(columns={"fatalities_num": "value"})
            fatalities["indicator"] = "acled_total_fatalities"
            fatalities["source"] = "ACLED"
            frames.append(fatalities)

        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["iso3", "year", "indicator", "value", "source"])

        if not self.countries.empty:
            result = result[result["iso3"].isin(set(self.country_codes))]

        logger.info("[acled] Cleaned: %d aggregated rows", len(result))
        return result

    @staticmethod
    def _numeric_to_iso3(code) -> str | None:
        try:
            c = pycountry.countries.get(numeric=str(int(code)).zfill(3))
            return c.alpha_3 if c else None
        except (ValueError, KeyError, AttributeError):
            return None


class UCDPConnector(DataConnector):
    """UCDP/PRIO Armed Conflict Dataset.

    As of Feb 2026, UCDP API requires a token obtained by emailing the maintainers.
    This connector now primarily uses the FREE public bulk CSV download
    (no token or registration required) as the default path.

    Bulk data: https://ucdp.uu.se/downloads/
    """

    source_name = "ucdp"

    # Public bulk CSV URLs (no token needed)
    UCDP_BULK_URLS = [
        # UCDP/PRIO Armed Conflict Dataset v24.1 (country-year)
        "https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-241-csv.zip",
        # Fallback: older version (v23.1)
        "https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-231-csv.zip",
    ]

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "ucdp_conflict.csv"
        if self._is_cached(cache_path):
            return cache_path

        # Try bulk CSV first (no token needed)
        logger.info("[ucdp] Trying public bulk CSV download...")
        for url in self.UCDP_BULK_URLS:
            try:
                resp = self._http_get(url, timeout=120)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    import zipfile, io
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
                        if csv_files:
                            data = zf.read(csv_files[0])
                            cache_path.write_bytes(data)
                            logger.info("[ucdp] Downloaded bulk CSV: %s", csv_files[0])
                            return cache_path
            except Exception as e:
                logger.warning("[ucdp] Bulk URL %s failed: %s", url, e)

        # Fallback: API with optional token
        token = self._get_env("UCDP_API_TOKEN")
        if token:
            logger.info("[ucdp] Trying API with token...")
            try:
                headers = {"Authorization": f"Bearer {token}"}
                resp = self._http_get(
                    "https://ucdpapi.pcr.uu.se/api/ucdpprioconflict/24.1",
                    headers=headers, timeout=60
                )
                data = resp.json()
                if "Result" in data:
                    df = pd.DataFrame(data["Result"])
                    df.to_csv(cache_path, index=False)
                    return cache_path
            except Exception as e:
                logger.warning("[ucdp] API failed: %s", e)

        logger.error("[ucdp] All download attempts failed.")
        cache_path.touch()
        return cache_path

        logger.warning("[ucdp] No UCDP data. Download from https://ucdp.uu.se/downloads/")
        cache_path.touch()
        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        raw = pd.read_csv(raw_path, low_memory=False)

        # UCDP columns vary; look for country identifiers
        loc_col = next((c for c in raw.columns if "location" in c.lower()), None)
        year_col = next((c for c in raw.columns if c.lower() == "year"), None)
        intensity_col = next((c for c in raw.columns if "intensity" in c.lower()), None)

        if not year_col:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        # UCDP location can contain multiple countries separated by commas
        frames = []
        for _, row in raw.iterrows():
            if loc_col and isinstance(row.get(loc_col), str):
                countries = [c.strip() for c in str(row[loc_col]).split(",")]
                for country_name in countries:
                    iso3 = _fuzzy_iso3(country_name)
                    if iso3:
                        frames.append({
                            "iso3": iso3,
                            "year": row[year_col],
                            "indicator": "ucdp_conflict_events",
                            "value": 1,
                            "source": "UCDP",
                        })
                        if intensity_col and pd.notna(row.get(intensity_col)):
                            frames.append({
                                "iso3": iso3,
                                "year": row[year_col],
                                "indicator": "ucdp_conflict_intensity",
                                "value": row[intensity_col],
                                "source": "UCDP",
                            })

        if frames:
            result = pd.DataFrame(frames)
            # Aggregate conflict events per country-year
            events = result[result["indicator"] == "ucdp_conflict_events"].groupby(
                ["iso3", "year", "indicator", "source"]).sum().reset_index()
            intensity = result[result["indicator"] == "ucdp_conflict_intensity"].groupby(
                ["iso3", "year", "indicator", "source"]).max().reset_index()
            result = pd.concat([events, intensity], ignore_index=True)
        else:
            result = pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        if not self.countries.empty:
            result = result[result["iso3"].isin(set(self.country_codes))]

        logger.info("[ucdp] Cleaned: %d rows", len(result))
        return result


class UNPopulationConnector(DataConnector):
    """UN Population Division — demographics via API or bulk download."""

    source_name = "un_population"

    API_URL = "https://population.un.org/dataportalapi/api/v1"

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "un_pop.csv"
        if self._is_cached(cache_path):
            return cache_path

        token = self._get_env("UN_POP_BEARER_TOKEN")
        if not token:
            logger.warning("[un_pop] UN_POP_BEARER_TOKEN not set. Using WDI for demographics.")
            cache_path.touch()
            return cache_path

        logger.info("[un_pop] Fetching population data via UN API...")
        # Fetch population indicators
        headers = {"Authorization": f"Bearer {token}"}
        indicators = [46, 47, 48]  # Population by broad age group, urbanization, urban growth

        frames = []
        for ind_id in indicators:
            try:
                url = f"{self.API_URL}/data/indicators/{ind_id}"
                resp = self._http_get(url, headers=headers, timeout=120)
                data = resp.json()
                if "data" in data:
                    frames.append(pd.DataFrame(data["data"]))
            except Exception as e:
                logger.warning("[un_pop] Failed for indicator %d: %s", ind_id, e)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(cache_path, index=False)
        else:
            cache_path.touch()

        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        raw = pd.read_csv(raw_path)
        if raw.empty:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        # Map to ISO-3166 and extract relevant indicators
        iso_col = next((c for c in raw.columns if "iso" in c.lower()), None)
        if iso_col:
            raw["iso3"] = raw[iso_col].str.strip().str.upper()
        raw["source"] = "UN_POP"

        if not self.countries.empty:
            raw = raw[raw.get("iso3", pd.Series()).isin(set(self.country_codes))]

        result = raw[["iso3", "year", "indicator", "value", "source"]].dropna(
            subset=["iso3"]) if all(c in raw.columns for c in ["iso3", "year", "indicator", "value"]) \
            else pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        logger.info("[un_pop] Cleaned: %d rows", len(result))
        return result


class FAOConnector(DataConnector):
    """FAO Food Security Indicators."""

    source_name = "fao"

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "fao_food_security.csv"
        if self._is_cached(cache_path):
            return cache_path

        logger.info("[fao] Fetching FAO food security data...")
        token = self._get_env("FAOSTAT_JWT_TOKEN")
        try:
            import faostat
            if token:
                logger.info("[fao] Setting manual JWT Bearer token for authentication.")
                faostat.set_requests_args(token=token)
            df = faostat.get_data_df("FS", pars={"item": "21004,21010"})  # Food Security indicators filtered by item
            if df is not None and not df.empty:
                df.to_csv(cache_path, index=False)
                return cache_path
        except Exception as e:
            logger.warning("[fao] faostat package failed: %s. Trying bulk URL...", e)

        # Fallback: OWID curated data
        try:
            url = "https://raw.githubusercontent.com/owid/etl/master/etl/steps/data/garden/fao/2024-03-14/food_security.csv"
            resp = self._http_get(url, timeout=60)
            cache_path.write_bytes(resp.content)
        except Exception as e:
            logger.error("[fao] All download methods failed: %s", e)
            cache_path.touch()

        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        raw = pd.read_csv(raw_path, low_memory=False)

        name_col = next((c for c in raw.columns if c.lower() in ("area", "country")), None)
        if not name_col:
            name_col = next((c for c in raw.columns if ("area" in c.lower() or "country" in c.lower()) and "code" not in c.lower()), None)
        item_col = next((c for c in raw.columns if c.lower() == "item"), None)
        if not item_col:
            item_col = next((c for c in raw.columns if "item" in c.lower() and "code" not in c.lower()), None)
        if not item_col:
            item_col = next((c for c in raw.columns if "item" in c.lower()), None)
        year_col = next((c for c in raw.columns if c.lower() == "year"), None)
        value_col = next((c for c in raw.columns if c.lower() == "value"), None)

        if name_col:
            raw["iso3"] = raw[name_col].apply(_fuzzy_iso3)
        else:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        raw = raw.dropna(subset=["iso3"])

        # Parse years (in FAOSTAT, 3-year averages like 2020-2022 are parsed to the middle or end year.
        # e.g., "2020-2022" contains 2021. Let's use the end year of the 3-year average.
        def parse_year(yr_str):
            if not isinstance(yr_str, str):
                yr_str = str(yr_str)
            if "-" in yr_str:
                parts = yr_str.split("-")
                return int(parts[1])
            try:
                return int(float(yr_str))
            except ValueError:
                return None

        raw["parsed_year"] = raw[year_col].apply(parse_year) if year_col else pd.NA
        raw = raw.dropna(subset=["parsed_year"])

        # Map to canonical indicator codes
        def map_indicator(item):
            il = str(item).lower()
            if "undernourishment" in il:
                return "prevalence_of_undernourishment"
            elif "dietary energy" in il:
                return "dietary_energy_adequacy"
            return None

        raw["mapped_indicator"] = raw[item_col].apply(map_indicator) if item_col else None
        raw = raw.dropna(subset=["mapped_indicator"])

        result = pd.DataFrame({
            "iso3": raw["iso3"],
            "year": raw["parsed_year"],
            "indicator": raw["mapped_indicator"],
            "value": pd.to_numeric(raw[value_col], errors="coerce"),
            "source": "FAO",
        }).dropna(subset=["iso3", "value"])

        if not self.countries.empty:
            result = result[result["iso3"].isin(set(self.country_codes))]

        logger.info("[fao] Cleaned: %d rows", len(result))
        return result

    @staticmethod
    def _fao_to_iso3(code) -> str | None:
        try:
            c = pycountry.countries.get(numeric=str(int(code)).zfill(3))
            return c.alpha_3 if c else None
        except (ValueError, KeyError, AttributeError):
            return None


class BISBankingConnector(DataConnector):
    """BIS consolidated banking statistics via SDMX."""

    source_name = "bis_banking"

    # BIS SDMX REST API — Consolidated Banking Statistics (WS_CBS_PUB)
    # Confirmed working as of 2025-Q4; uses immediate counterparty basis,
    # all instruments, USD, total sector, 2008-present.
    CBS_API_URL = (
        "https://stats.bis.org/api/v1/data/BIS,WS_CBS_PUB,1.0/"
        "?startPeriod=2008&format=csv"
    )

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "bis_banking.csv"
        if self._is_cached(cache_path):
            return cache_path

        logger.info("[bis] Fetching BIS CBS via SDMX REST API (WS_CBS_PUB)...")
        try:
            resp = self._http_get(self.CBS_API_URL, timeout=300)
            if resp.status_code == 200 and len(resp.content) > 10_000:
                cache_path.write_bytes(resp.content)
                logger.info("[bis] Downloaded %d bytes (%d rows)",
                            len(resp.content),
                            resp.content.count(b"\n"))
                return cache_path
            else:
                logger.warning("[bis] API returned status %s, size %d",
                               resp.status_code, len(resp.content))
        except Exception as e:
            logger.warning("[bis] SDMX REST fetch failed: %s", e)

        # Fallback: legacy bulk ZIP URLs (may be stale)
        for url in [
            "https://www.bis.org/statistics/full_bis_cbs_diss_csv.zip",
            "https://www.bis.org/statistics/full_bis_lbs_diss_csv.zip",
        ]:
            try:
                resp = self._http_get(url, timeout=300)
                if resp.status_code == 200 and len(resp.content) > 100_000:
                    import zipfile, io
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                        if csv_names:
                            cache_path.write_bytes(zf.read(csv_names[0]))
                            logger.info("[bis] Downloaded legacy ZIP: %s", csv_names[0])
                            return cache_path
            except Exception as e2:
                logger.warning("[bis] Legacy ZIP %s failed: %s", url, e2)

        logger.warning("[bis] All download attempts failed — place CSV manually in %s", self.raw_dir)
        cache_path.touch()
        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            logger.warning("[bis] Empty cache file — returning empty DataFrame")
            return pd.DataFrame(columns=["iso3_i", "iso3_j", "year", "value", "indicator", "source"])

        raw = pd.read_csv(raw_path, low_memory=False)
        logger.info("[bis] Raw BIS shape: %s, columns: %s", raw.shape, list(raw.columns))

        # WS_CBS_PUB columns: FREQ, L_MEASURE, L_REP_CTY, CBS_BANK_TYPE, CBS_BASIS,
        # L_POSITION, L_INSTR, REM_MATURITY, CURR_TYPE_BOOK, L_CP_SECTOR,
        # L_CP_COUNTRY, ..., TIME_PERIOD, OBS_VALUE
        required = {"L_REP_CTY", "L_CP_COUNTRY", "TIME_PERIOD", "OBS_VALUE"}
        if not required.issubset(raw.columns):
            logger.warning("[bis] Missing expected columns: %s", required - set(raw.columns))
            return pd.DataFrame(columns=["iso3_i", "iso3_j", "year", "value", "indicator", "source"])

        # Filter to cross-border claims: immediate counterparty basis (F),
        # total position (I), all instruments (A), all maturities (N),
        # all currencies (TO1), all sectors (A)
        df = raw.copy()
        if "CBS_BASIS" in df.columns:
            df = df[df["CBS_BASIS"] == "F"]   # immediate counterparty
        if "L_POSITION" in df.columns:
            df = df[df["L_POSITION"] == "I"]  # claims (assets)
        if "L_INSTR" in df.columns:
            df = df[df["L_INSTR"] == "A"]     # all instruments

        # Parse year from quarterly TIME_PERIOD (e.g. "2015-Q4" → 2015)
        df["year"] = pd.to_numeric(
            df["TIME_PERIOD"].astype(str).str[:4], errors="coerce"
        ).astype("Int64")

        # Aggregate quarterly → annual mean
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        df = (
            df.groupby(["L_REP_CTY", "L_CP_COUNTRY", "year"], as_index=False)
              .agg(claims_usd_mn=("OBS_VALUE", "mean"))
        )

        # Map BIS country codes (ISO 2-letter or BIS codes) → ISO3
        # BIS uses 2-letter codes largely matching ISO-3166-1 alpha-2
        def bis_to_iso3(code: str) -> str | None:
            if not code or len(code) < 2:
                return None
            # Handle BIS special aggregate codes (5A = all countries, etc.)
            if code.startswith("5") or len(code) > 3:
                return None
            try:
                c = pycountry.countries.get(alpha_2=code.upper())
                return c.alpha_3 if c else None
            except Exception:
                return None

        df["iso3_i"] = df["L_REP_CTY"].map(bis_to_iso3)   # reporting bank country
        df["iso3_j"] = df["L_CP_COUNTRY"].map(bis_to_iso3)  # borrower country

        df = df.dropna(subset=["iso3_i", "iso3_j", "year", "claims_usd_mn"])
        df = df[df["iso3_i"] != df["iso3_j"]]  # exclude self-claims

        # Scale: BIS values are in millions USD (UNIT_MULT=6)
        df["value"] = df["claims_usd_mn"] * 1e6
        df["indicator"] = "banking_claims_usd"
        df["source"] = "bis_cbs"

        result = df[["iso3_i", "iso3_j", "year", "value", "indicator", "source"]]
        logger.info("[bis] Cleaned: %d bilateral pairs, %d unique reporters",
                    len(result), result["iso3_i"].nunique())
        return result

    def validate(self, df: pd.DataFrame) -> bool:
        return True  # BIS may be empty for some country pairs


# ── WHO Global Health Observatory ─────────────────────────────────────────────

class WHOGHOConnector(DataConnector):
    """WHO Global Health Observatory — OData REST API (no key required).

    Fetches state capacity proxy indicators: life expectancy, UHC index,
    under-5 mortality. 194 member states, annual.

    API docs: https://www.who.int/data/gho/info/gho-odata-api
    """

    source_name = "who_gho"

    GHO_BASE = "https://ghoapi.azureedge.net/api"

    # WHO indicator codes → our variable names
    INDICATORS = {
        "WHOSIS_000001": "who_life_expectancy",          # Life expectancy at birth (both sexes)
        "MDG_0000000007": "who_u5_mortality_rate",        # Under-5 mortality rate (per 1000)
        "UHC_INDEX_REPORTED": "who_uhc_index",            # UHC service coverage index
    }

    def download(self) -> Path:
        self._ensure_dirs()
        cache_path = self.raw_dir / "who_gho.csv"
        if self._is_cached(cache_path):
            return cache_path

        logger.info("[who_gho] Fetching from WHO GHO OData API...")
        all_frames = []

        for code, ind_name in self.INDICATORS.items():
            if code in ("WHOSIS_000001", "MDG_0000000007"):
                url = f"{self.GHO_BASE}/{code}?$filter=Dim1 eq 'SEX_BTSX'&$select=SpatialDim,TimeDim,NumericValue"
            else:
                url = f"{self.GHO_BASE}/{code}?$select=SpatialDim,TimeDim,NumericValue"
            try:
                resp = self._http_get(url, timeout=60)
                data = resp.json()
                if "value" in data and data["value"]:
                    df = pd.DataFrame(data["value"])
                    df = df.rename(columns={
                        "SpatialDim": "iso3",
                        "TimeDim": "year",
                        "NumericValue": "value",
                    })
                    df["indicator"] = ind_name
                    df["source"] = "WHO_GHO"
                    all_frames.append(df[["iso3", "year", "value", "indicator", "source"]])
                    logger.info("[who_gho] %s: %d rows", code, len(df))
            except Exception as e:
                logger.warning("[who_gho] Failed for %s: %s", code, e)

        if all_frames:
            combined = pd.concat(all_frames, ignore_index=True)
            combined.to_csv(cache_path, index=False)
        else:
            logger.warning("[who_gho] No data fetched.")
            cache_path.touch()

        return cache_path

    def clean(self, raw_path: Path) -> pd.DataFrame:
        if raw_path.stat().st_size < 100:
            return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "source"])

        raw = pd.read_csv(raw_path)
        raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw["iso3"] = raw["iso3"].str.strip().str.upper()
        result = raw.dropna(subset=["iso3", "year", "value"])

        if not self.countries.empty:
            result = result[result["iso3"].isin(set(self.country_codes))]
            end_year = int(self.countries["gdp_year"].iloc[0])
            result = result[result["year"] >= end_year - 14]

        logger.info("[who_gho] Cleaned: %d rows, %d indicators",
                    len(result), result["indicator"].nunique())
        return result

