# SETUP.md - Spectral Instability Model

> Installation, data-source registration, environment configuration, and pipeline execution.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Environment Configuration](#3-environment-configuration)
4. [Data Source Registration Guide](#4-data-source-registration-guide)
   - [Sources Requiring API Keys / Registration](#41-sources-requiring-api-keys--registration)
   - [Sources with No API Key Required](#42-sources-with-no-api-key-required)
   - [Restricted / Unavailable Sources](#43-restricted--unavailable-sources)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| **Python** | 3.11+ | 3.12 recommended. Check with `python3 --version`. |
| **uv** | 0.7+ | Fast Python package manager. Install via `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Make** | Any | Ships with Xcode CLI tools on macOS (`xcode-select --install`). |
| **Git** | 2.x | For cloning the repository. |
| **Disk space** | ~5 GB | Raw + processed data, virtual environment, and cached artefacts. |

Verify your environment:

```bash
python3 --version   # ≥ 3.11
uv --version        # ≥ 0.7
make --version
git --version
```

---

## 2. Installation

### 2.1 Clone the repository

```bash
git clone <REPO_URL> spectral-instability-model
cd spectral-instability-model
```

### 2.2 Create virtual environment and install dependencies

The project uses **uv** for dependency management. A `pyproject.toml` is provided at the repository root.

```bash
# Create a virtual environment and install all project dependencies
uv sync
```

This installs the full dependency set:

| Category | Packages |
|---|---|
| **Core scientific** | `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels` |
| **Data I/O** | `pyarrow` (Parquet support), `requests`, `python-dotenv` |
| **IMF / World Bank** | `weo`, `imfp`, `sdmx1`, `wbgapi` |
| **UN / FAO** | `comtradeapicall`, `faostat` |
| **Conflict data** | `acled` (unofficial ACLED wrapper) |
| **Geo / reference** | `pycountry` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **Web UI** | `streamlit` |

To add a package later:

```bash
uv add <package-name>
```

### 2.3 Verify the installation

```bash
uv run python -c "import pandas, numpy, scipy, sklearn, statsmodels; print('All core imports OK')"
```

---

## 3. Environment Configuration

All API keys and tokens are read from a **`.env`** file at the project root. A template is provided in `.env.example`.

### 3.1 Create your `.env` file

```bash
cp .env.example .env
```

### 3.2 Populate the keys

Open `.env` in your editor and fill in the values obtained by following the registration steps in [§4](#4-data-source-registration-guide).

```dotenv
# ──────────────────────────────────────────────
# Spectral Instability Model - Environment Vars
# ──────────────────────────────────────────────

# ACLED (Armed Conflict Location & Event Data)
# Register: https://acleddata.com/register/
ACLED_EMAIL=your_registered_email@institution.edu
ACLED_API_KEY=your_acled_api_key

# UCDP / PRIO (Uppsala Conflict Data Program)
# Request access token from UCDP API maintainer
UCDP_API_TOKEN=your_ucdp_token

# UN Population Division
# Generate: https://population.un.org/dataportal/
UN_POP_BEARER_TOKEN=your_un_population_token

# FAO FAOSTAT
# Register at FAOSTAT Developer Portal for JWT token
FAOSTAT_JWT_TOKEN=your_faostat_jwt_token

# UN Comtrade
# Subscribe: https://comtradeplus.un.org/
COMTRADE_SUBSCRIPTION_KEY=your_comtrade_subscription_key

# ──── No key required (leave as-is) ────
# IMF WEO / DOTS - public SDMX API
# World Bank WDI / WGI - public REST API (wbgapi)
# V-Dem, Polity5, Freedom House, TI CPI, FSI,
# BIS, SIPRI, ND-GAIN, CEPII, COW - bulk download
```

> **Security note:** Never commit `.env` to version control. The `.gitignore` already excludes it.

---

## 4. Data Source Registration Guide

### 4.1 Sources Requiring API Keys / Registration

#### 4.1.1 ACLED (Armed Conflict Location & Event Data Project)

ACLED provides georeferenced event-level data on political violence and protests worldwide.

| Field | Value |
|---|---|
| Auth method | OAuth token (email + API key) |
| Cost | Free for research use |
| Turnaround | Instant upon registration |

**Steps:**

1. Navigate to <https://acleddata.com/register/>.
2. Create a **myACLED** account. An institutional email (`.edu`, `.ac.uk`, etc.) is strongly recommended - personal-domain emails may be delayed or rejected.
3. Complete the registration form, specifying *Academic / Research* as the use case.
4. Confirm your email address.
5. Log in to <https://acleddata.com/> → **Dashboard** → **API Access**.
6. Copy your **API key**.
7. Add both your registered email and the API key to `.env`:
   ```
   ACLED_EMAIL=you@university.edu
   ACLED_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
   ```

**Verification:**

```bash
uv run python -c "
import os, requests
r = requests.get('https://api.acleddata.com/acled/read',
    params={'key': os.environ['ACLED_API_KEY'],
            'email': os.environ['ACLED_EMAIL'],
            'limit': 1})
print('ACLED OK' if r.status_code == 200 else f'ACLED FAIL: {r.status_code}')
"
```

---

#### 4.1.2 UCDP/PRIO (Uppsala Conflict Data Program)

UCDP provides battle-related deaths data, armed conflict datasets, and one-sided violence records.

| Field | Value |
|---|---|
| Auth method | Access token (Bearer header) |
| Cost | Free |
| Turnaround | Variable (email-based request) |
| Bulk alternative | CSV downloads at <https://ucdp.uu.se/downloads/> - **no auth required** |

**Steps (API access):**

1. Visit <https://ucdp.uu.se/apidocs/>.
2. Email the UCDP API maintainer (address listed on the API docs page) requesting an access token. Include your name, institution, and research purpose.
3. You will receive a token via email (allow 1–5 business days).
4. Add it to `.env`:
   ```
   UCDP_API_TOKEN=your_token_here
   ```

**Fallback (bulk CSV - no token required):**

If the API token is not yet available, the pipeline can ingest bulk CSV files directly:

1. Download the relevant datasets from <https://ucdp.uu.se/downloads/> (e.g., *UCDP/PRIO Armed Conflict Dataset*, *UCDP Georeferenced Event Dataset*).
2. Place the CSV files in `data/raw/ucdp/`.
3. The ingest module auto-detects local files when `UCDP_API_TOKEN` is unset.

---

#### 4.1.3 UN Population Division

Provides demographic indicators: population projections, fertility, mortality, migration.

| Field | Value |
|---|---|
| Auth method | Bearer token |
| Cost | Free |
| Turnaround | Instant |

**Steps:**

1. Navigate to <https://population.un.org/dataportal/>.
2. Click **Sign In** (top-right) and create a UN Data Portal account.
3. After login, go to **Profile** → **API Tokens** → **Generate New Token**.
4. Copy the token and add to `.env`:
   ```
   UN_POP_BEARER_TOKEN=Bearer xxxxxx...
   ```

---

#### 4.1.4 FAO FAOSTAT

Agricultural production, food prices, land use, trade, and emissions data.

| Field | Value |
|---|---|
| Auth method | JWT Bearer token |
| Cost | Free |
| Turnaround | Instant upon account creation |
| Bulk alternative | <https://www.fao.org/faostat/en/#data> - bulk CSV download, **no auth** |

**Steps (API access):**

1. Go to the FAOSTAT Developer Portal.
2. Register for a developer account.
3. Create an application and generate a JWT token.
4. Add to `.env`:
   ```
   FAOSTAT_JWT_TOKEN=eyJhbG...
   ```

**Fallback (bulk download):**

1. Visit <https://www.fao.org/faostat/en/#data>.
2. Select the relevant domain (e.g., *Production → Crops and livestock products*).
3. Download as CSV and place in `data/raw/fao/`.

---

#### 4.1.5 UN Comtrade

Bilateral trade flow data (import/export by commodity and partner).

| Field | Value |
|---|---|
| Auth method | Subscription key (header: `Ocp-Apim-Subscription-Key`) |
| Cost | Free tier available (~100 calls/day); premium tiers also available |
| Turnaround | Account creation is instant; **key approval may take 1–5 business days** |
| Alternative | IMF Direction of Trade Statistics (DOTS) via `imfp` - **no key needed** |

**Steps:**

1. Navigate to <https://comtradeplus.un.org/>.
2. Register for an account.
3. Subscribe to the **Comtrade API** product (free tier).
4. Wait for approval (you'll receive an email). This can take several days.
5. Retrieve your subscription key from your profile page.
6. Add to `.env`:
   ```
   COMTRADE_SUBSCRIPTION_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
   ```

**Alternative - IMF DOTS (no key):**

If Comtrade approval is pending, the pipeline falls back to IMF Direction of Trade Statistics via the `imfp` package, which requires no authentication:

```python
import imfp
dots = imfp.imf_dataset("DOT", freq="A", ref_area="US")
```

Set `USE_IMF_DOTS_FALLBACK=true` in `.env` to force this fallback.

---

### 4.2 Sources with No API Key Required

These sources are accessed either through public REST/SDMX APIs or via bulk file downloads. No registration is needed.

| Source | Access Method | URL / Package | Notes |
|---|---|---|---|
| **IMF WEO** | SDMX API or bulk CSV | `weo` / `imfp` / `sdmx1` | Published biannually (Apr, Oct). |
| **V-Dem** | Bulk CSV download | <https://v-dem.net/data/> | ~30 MB compressed. Country-year panel. |
| **Polity5** | Bulk Excel download | <https://www.systemicpeace.org/polityproject.html> | Coverage ends ~2018; supplement with V-Dem. |
| **World Bank WGI** | REST API | `wbgapi` | `wb.data.DataFrame('CC.EST', economy='all')` |
| **World Bank WDI** | REST API | `wbgapi` | `wb.data.DataFrame('NY.GDP.PCAP.CD')` |
| **Freedom House** | Bulk Excel download | <https://freedomhouse.org/report/freedom-world> | Annual ratings (PR + CL scores). |
| **Transparency Intl. CPI** | Bulk Excel/CSV | <https://www.transparency.org/en/cpi> | Annual scores and ranks. |
| **Fund for Peace FSI** | Bulk Excel download | <https://fragilestatesindex.org/excel/> | 12 sub-indicators + composite. |
| **BIS Banking Statistics** | SDMX API | `sdmx1` | Cross-border banking exposure data. |
| **SIPRI Milex** | Bulk Excel download | <https://www.sipri.org/databases/milex> | Military expenditure by country-year. |
| **ND-GAIN** | Bulk CSV download | <https://gain.nd.edu/our-work/country-index/download-data/> | Climate vulnerability + readiness. |
| **CEPII GeoDist** | Bulk CSV download | <http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6> | Bilateral distance, contiguity, etc. |
| **Correlates of War** | Bulk CSV download | <https://correlatesofwar.org/> | Trade, alliance, IGO membership data. |

**Bulk download placement:** Place all downloaded files in the appropriate subdirectory under `data/raw/<source_name>/` (e.g., `data/raw/vdem/`, `data/raw/sipri/`). The ingest pipeline auto-discovers files by directory.

---

### 4.3 Restricted / Unavailable Sources

#### Global Terrorism Database (START/GTD)

As of 2025, the GTD has moved to **restricted access**. Downloading the full dataset now requires a formal access request submitted to the National Consortium for the Study of Terrorism and Responses to Terrorism (START) at the University of Maryland.

- **Status:** Not used in the default pipeline.
- **Substitute:** ACLED political violence events (event types: *Battles*, *Explosions/Remote violence*, *Violence against civilians*) cover substantially similar ground. The substitution rationale is documented in `METHODOLOGY.md`.

#### Polity5

Polity5 coverage effectively ends around **2018** and has not been updated since the project's administrative transition.

- **Mitigation:** The pipeline ingests Polity5 for historical coverage (1800–2018) and supplements with **V-Dem** regime-type indicators (`v2x_regime`, `v2x_polyarchy`) for years 2019 onward. Crosswalk logic is in `src/ingest/polity_vdem_bridge.py`.

---

## 5. Running the Pipeline

The project uses a `Makefile` for orchestration. All commands assume you are in the project root.

```bash
# Display all available targets
make help

# ── Full pipeline ──
make all              # Runs: ingest → clean → impute → build → score → export

# ── Individual stages ──
make ingest           # Download / load raw data from APIs and bulk files
make clean            # Harmonise country codes, align time indices
make impute           # Multiple imputation (MICE via sklearn IterativeImputer)
make build            # Construct composite indicators, PCA, network layers
make score            # Compute spectral instability scores
make export           # Write final outputs to data/output/

# ── Utilities ──
make check-env        # Validate .env file and test all API connections
make test             # Run the test suite (pytest)
make lint             # Run ruff linter
make dashboard        # Launch Streamlit dashboard (streamlit run app.py)
```

### Typical first-run workflow

```bash
# 1. Ensure .env is populated
make check-env

# 2. Download all bulk data files (V-Dem, Polity5, SIPRI, etc.)
make ingest-bulk

# 3. Fetch API-sourced data
make ingest-api

# 4. Run the full pipeline
make all

# 5. Launch the dashboard
make dashboard
```

---

## 6. Troubleshooting

### 6.1 `ModuleNotFoundError` on import

You are likely running Python outside the uv-managed virtual environment.

```bash
# Always prefix commands with `uv run`:
uv run python src/main.py

# Or activate the venv manually:
source .venv/bin/activate
python src/main.py
```

### 6.2 ACLED API returns 403 Forbidden

- Verify that `ACLED_EMAIL` matches the email used during registration.
- Regenerate your API key from the myACLED dashboard if the key has been rotated.
- ACLED enforces rate limits (~500 requests/min). If you hit the limit, the ingest module retries with exponential back-off automatically.

### 6.3 UN Comtrade returns 401 Unauthorized

- Subscription key approval can take several business days. Check your email for approval confirmation.
- In the meantime, set `USE_IMF_DOTS_FALLBACK=true` in `.env` to use IMF DOTS data instead.

### 6.4 UCDP API is unavailable

- The UCDP API endpoint occasionally goes offline for maintenance.
- Download the bulk CSVs from <https://ucdp.uu.se/downloads/> and place them in `data/raw/ucdp/`. The pipeline will auto-detect local files.

### 6.5 `wbgapi` hangs or times out

The World Bank API can be slow. Set a longer timeout:

```python
import wbgapi as wb
wb.source.TIMEOUT = 60  # seconds
```

### 6.6 Parquet read/write errors

Ensure `pyarrow` is installed:

```bash
uv add pyarrow
uv run python -c "import pyarrow; print(pyarrow.__version__)"
```

### 6.7 `.env` file not loading

Ensure `python-dotenv` is installed and that your entry point calls `load_dotenv()` before accessing `os.environ`:

```python
from dotenv import load_dotenv
load_dotenv()  # reads .env from project root
```

### 6.8 V-Dem download is very large

The full V-Dem dataset is ~1 GB uncompressed. The ingest module downloads only the country-year core dataset by default. To fetch the full dataset (including episode and coder-level data), set `VDEM_FULL_DOWNLOAD=true` in `.env`.

### 6.9 Polity5 ↔ V-Dem bridging warnings

If you see warnings about unmapped country-year observations in the Polity–V-Dem crosswalk, check `logs/polity_vdem_bridge.log` for details. Common causes:
- Polity5 uses non-standard country codes for historical states (e.g., Yugoslavia, USSR).
- V-Dem covers some micro-states that Polity5 does not.

---

## Quick Reference - Environment Variables

| Variable | Required? | Source |
|---|---|---|
| `ACLED_EMAIL` | Yes (for conflict data) | [ACLED registration](#411-acled) |
| `ACLED_API_KEY` | Yes (for conflict data) | [ACLED registration](#411-acled) |
| `UCDP_API_TOKEN` | Optional (fallback: bulk CSV) | [UCDP request](#412-ucdpprio) |
| `UN_POP_BEARER_TOKEN` | Yes | [UN Data Portal](#413-un-population-division) |
| `FAOSTAT_JWT_TOKEN` | Optional (fallback: bulk CSV) | [FAOSTAT Portal](#414-fao-faostat) |
| `COMTRADE_SUBSCRIPTION_KEY` | Optional (fallback: IMF DOTS) | [UN Comtrade](#415-un-comtrade) |
| `USE_IMF_DOTS_FALLBACK` | Optional | Set `true` to bypass Comtrade |
| `VDEM_FULL_DOWNLOAD` | Optional | Set `true` for full V-Dem dataset |
