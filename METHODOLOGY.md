# METHODOLOGY.md — Spectral Model of Global Regime Instability

> Full methodological documentation for the spectral instability model.
> Every modeling decision is documented with citations in BibTeX format (see `references.bib`).

---

## Table of Contents

1. [Country Universe Selection](#1-country-universe-selection)
2. [Data Sources](#2-data-sources)
3. [Indicator Engineering](#3-indicator-engineering)
4. [Spectral Core — Latent Instability Factors](#4-spectral-core--latent-instability-factors)
5. [Country Coupling Matrix and Network Spectrum](#5-country-coupling-matrix-and-network-spectrum)
6. [Dynamic Propagation Model](#6-dynamic-propagation-model)
7. [Simulation Capabilities](#7-simulation-capabilities)
8. [Uncertainty Quantification](#8-uncertainty-quantification)
9. [Validation Strategy](#9-validation-strategy)

---

## 1. Country Universe Selection

### 1.1 Source

The country universe is drawn from the **IMF World Economic Outlook (WEO)** database, accessed programmatically via the SDMX API using the `weo` or `imfp` Python packages. No API key is required.

### 1.2 Selection Criteria

- Extract **nominal GDP in current USD** (`NGDPD`) for the most recent fully reported year.
- Rank all sovereign states and select the **top 125** by nominal GDP.
- Exclude territories, dependencies, and non-sovereign entities.

### 1.3 Identifiers

Each country is assigned:
- **ISO 3166-1 alpha-3 code** (primary key for all downstream joins)
- **ISO 3166-1 numeric code**
- **IMF country code** (for WEO/DOTS cross-referencing)
- **World Bank country code** (for WDI/WGI joins)

The mapping is performed via the `pycountry` package with manual overrides for known discrepancies (e.g., Kosovo `XKX`, Taiwan `TWN`).

### 1.4 Persistence and Versioning

The country list is persisted in `data/countries.csv` with columns: `iso3`, `country_name`, `gdp_nominal_usd`, `gdp_year`, `imf_code`, `wb_code`. The file is regenerated on each pipeline run but the snapshot used in any given simulation is versioned via a hash in the output metadata.

---

## 2. Data Sources

All data used in this model comes from official primary sources. No secondary aggregators are used except where explicitly noted. No synthetic or LLM-generated data is present anywhere in the pipeline.

### 2.1 Governance & Political

#### V-Dem (Varieties of Democracy) v14+

- **Provides:** Polyarchy index (`v2x_polyarchy`), liberal democracy (`v2x_libdem`), electoral democracy (`v2x_polyarchy`), deliberative democracy (`v2x_delibdem`), egalitarian democracy (`v2x_egaldem`), participatory democracy (`v2x_partipdem`).
- **Access:** Bulk CSV download from [v-dem.net](https://v-dem.net/data/). No API available.
- **Granularity:** Country-year panel, 1789–present.
- **Python method:** `pandas.read_csv()` on downloaded CSV.
- **Citation:** Coppedge, M. et al. (2024). V-Dem Dataset v14. Varieties of Democracy (V-Dem) Project.
- **Limitations:** Large file (~1 GB uncompressed). We use the country-year core dataset only.

#### Polity5 (Center for Systemic Peace)

- **Provides:** Composite polity2 score (−10 to +10), regime durability, executive constraints (`xconst`), executive recruitment (`xrreg`, `xrcomp`, `xropen`), political competition (`parcomp`).
- **Access:** Bulk Excel download from [systemicpeace.org](https://www.systemicpeace.org/inscrdata.html). No API.
- **Python method:** `pandas.read_excel()`.
- **Citation:** Marshall, M. G., & Gurr, T. R. (2020). Polity5: Political Regime Characteristics and Transitions, 1800–2018. Center for Systemic Peace.
- **⚠️ Limitation:** Core dataset coverage effectively ends **~2018**. The project has not been updated since its administrative transition. For years 2019 onward, we supplement with V-Dem regime-type indicators (see §3.1). This introduces a methodological break that is documented in `LIMITATIONS.md`.

#### Worldwide Governance Indicators (WGI)

- **Provides:** Six dimensions: Voice & Accountability (`VA.EST`), Political Stability & Absence of Violence (`PV.EST`), Government Effectiveness (`GE.EST`), Regulatory Quality (`RQ.EST`), Rule of Law (`RL.EST`), Control of Corruption (`CC.EST`). Each ranges from approximately −2.5 to +2.5.
- **Access:** World Bank API (Source ID 3) via `wbgapi` package.
- **Python method:** `wbgapi.data.DataFrame('CC.EST', economy='all')`.
- **Citation:** [@kaufmann2010wgi]
- **Limitations:** Published biennially with some gaps. Estimates have substantial uncertainty bands (provided in the source data).

#### Freedom House — Freedom in the World

- **Provides:** Aggregate freedom score (1–7, where 1 = most free), Political Rights (PR) subscore, Civil Liberties (CL) subscore, Freedom Status (Free / Partly Free / Not Free).
- **Access:** Bulk Excel download from [freedomhouse.org](https://freedomhouse.org/report/freedom-world).
- **Python method:** `pandas.read_excel()` on annual release files.
- **Citation:** Freedom House (annual). Freedom in the World. Washington, DC.
- **Limitations:** Methodology changes between survey years. The aggregate score direction (lower = more free) is inverted in §3.3.

#### Transparency International — Corruption Perceptions Index (CPI)

- **Provides:** CPI score (0–100, where 100 = least corrupt).
- **Access:** Bulk CSV/Excel from [transparency.org](https://www.transparency.org/en/cpi).
- **Python method:** `pandas.read_csv()` or `pandas.read_excel()`.
- **Citation:** Transparency International (annual). Corruption Perceptions Index.
- **Limitations:** Methodology changed in 2012 (scale shifted from 0–10 to 0–100). We use post-2012 data only for consistency. Inverted in §3.3.

#### Fund for Peace — Fragile States Index (FSI)

- **Provides:** 12 sub-indicators (0–10 each) plus composite score (0–120): Security Apparatus (C1), Factionalized Elites (C2), Group Grievance (C3), Economic Decline (E1), Uneven Economic Development (E2), Human Flight & Brain Drain (E3), State Legitimacy (P1), Public Services (P2), Human Rights & Rule of Law (P3), Demographic Pressures (S1), Refugees & IDPs (S2), External Intervention (X1).
- **Access:** Bulk Excel download from [fragilestatesindex.org](https://fragilestatesindex.org/). Also available via Kaggle/Mendeley consolidated CSVs.
- **Python method:** `pandas.read_excel()`.
- **Citation:** Fund for Peace (annual). Fragile States Index. Washington, DC.
- **Role in model:** Used primarily as a **validation benchmark** (§9). FSI sub-indicators are also included in the indicator matrix where they map to canonical pillars. The composite FSI score is NOT used as an input — it is used only for the sanity check correlation in §4.4.

### 2.2 Economic & Fiscal

#### World Bank — World Development Indicators (WDI)

- **Provides:** GDP growth (`NY.GDP.MKTP.KD.ZG`), inflation CPI (`FP.CPI.TOTL.ZG`), unemployment (`SL.UEM.TOTL.ZS`), youth unemployment (`SL.UEM.1524.ZS`), Gini coefficient (`SI.POV.GINI`), current account % GDP (`BN.CAB.XOKA.GD.ZS`), gross reserves in months of imports (`FI.RES.TOTL.MO`), refugee population by country of asylum (`SM.POP.REFG`), internally displaced persons (`VC.IDP.TOTL.HE`).
- **Access:** World Bank REST API via `wbgapi` package. No key required.
- **Python method:** `wbgapi.data.DataFrame(['NY.GDP.MKTP.KD.ZG', ...], economy='all')`.
- **Citation:** World Bank (annual). World Development Indicators. Washington, DC.
- **Limitations:** Gini coefficient has extensive missingness (~50% of country-years). Subject to MICE imputation with flagging per §3.2.

#### IMF World Economic Outlook (WEO)

- **Provides:** Fiscal balance % GDP (`GGXCNL_NGDP`), gross government debt % GDP (`GGXWDG_NGDP`), external debt metrics.
- **Access:** SDMX API (no key) or bulk CSV. Uses `weo` or `imfp` Python packages.
- **Citation:** International Monetary Fund (biannual). World Economic Outlook Database.
- **Limitations:** WEO contains both actual data and IMF staff estimates/projections for future years. We use only actual/estimated values, never projections.

#### BIS Consolidated Banking Statistics

- **Provides:** Cross-border claims and liabilities by reporting-country/counterparty-country pair. Used for the financial coupling sub-matrix `W_fin` (§5).
- **Access:** SDMX API via `sdmx1` package, or bulk CSV from [data.bis.org](https://data.bis.org/). No key required.
- **Citation:** Bank for International Settlements. Consolidated Banking Statistics.
- **⚠️ Limitation:** BIS reporting banks cover approximately **30 countries** (mainly OECD + major financial centers). For the remaining ~95 countries in our universe, bilateral financial exposure data is sparse or unavailable. Mitigation: (a) use mirror data where available (if country A reports claims on country B, we infer B's exposure to A's banking sector, though we discount mirror exposures by 0.5 to reflect the uncertainty and asymmetry inherent in inferred exposures), (b) remaining entries set to 0. The `W_fin` sub-matrix is therefore dominated by major financial centers. This limitation is acceptable because financial contagion channels are indeed concentrated in these centers.

### 2.3 Conflict & Security

#### UCDP/PRIO Armed Conflict Dataset

- **Provides:** Armed conflict episodes classified by type (interstate, intrastate, internationalized intrastate, extrasystemic), intensity level (minor: 25–999 battle-related deaths/year; war: ≥1000), conflict parties, dates.
- **Access:** REST API (requires Bearer token, request from UCDP maintainer) or bulk CSV from [ucdp.uu.se/downloads](https://ucdp.uu.se/downloads/). The pipeline auto-detects local CSV when API token is unavailable.
- **Python method:** `requests` with Bearer auth for API; `pandas.read_csv()` for bulk.
- **Citation:** Gleditsch, N. P. et al. (2002). Armed Conflict 1946–2001: A New Dataset. Journal of Peace Research, 39(5), 615–637.

#### UCDP Georeferenced Event Dataset (GED)

- **Provides:** Individual conflict events with geographic coordinates, date, fatalities, actors. Used for cross-border spillover analysis in the coupling matrix.
- **Access:** Same as UCDP/PRIO above.
- **Citation:** Sundberg, R., & Melander, E. (2013). Introducing the UCDP Georeferenced Event Dataset. Journal of Peace Research, 50(4), 523–532.

#### ACLED (Armed Conflict Location & Event Data Project)

- **Provides:** Event-level data categorized as: Battles, Violence against civilians, Explosions/Remote violence, Riots, Protests, Strategic developments. Each event has date, location, actors, fatalities.
- **Access:** REST API with OAuth authentication. Requires myACLED account registration.
- **Python method:** `requests` with API key + email, or unofficial `acled` package.
- **Citation:** Raleigh, C. et al. (2010). Introducing ACLED: An Armed Conflict Location and Event Dataset. Journal of Peace Research, 47(5), 651–660.
- **Usage in model:** We compute rolling 12-month event counts by type per country. ACLED also serves as the **substitute for the Global Terrorism Database** (see below).

#### SIPRI Military Expenditure Database

- **Provides:** Military expenditure in constant USD, as % of GDP, per capita, as % of government expenditure.
- **Access:** Bulk Excel from [sipri.org/databases/milex](https://www.sipri.org/databases/milex), or clean CSV from Our World in Data.
- **Python method:** `pandas.read_excel()` or `pandas.read_csv()` (OWID URL).
- **Citation:** Stockholm International Peace Research Institute (annual). SIPRI Military Expenditure Database.

#### Global Terrorism Database — SUBSTITUTED

- **Original plan:** Use START/GTD for terrorism event counts.
- **Status as of 2025:** The GTD has moved to **restricted access**. Downloading the full dataset requires a formal access request to START at the University of Maryland, with new DOJ regulations potentially further limiting access.
- **Substitute:** ACLED's political violence categories (Battles, Explosions/Remote violence, Violence against civilians) cover substantially similar events to what GTD tracked. ACLED's classification does not distinguish "terrorism" as a separate category, but the underlying events (bombings, assassinations, attacks on civilians) are captured.
- **Loss of fidelity:** We lose (a) the specific "terrorism" label and its associated GTD-specific metadata (weapon type, target type, attack type), and (b) historical data before ACLED's coverage start date (varies by region; Africa from 1997, global from 2016). For the 15-year panel (2009–2024), ACLED's global coverage begins partway through, which introduces a data gap for non-African countries pre-2016.
- **Mitigation:** For the historical panel, we use UCDP one-sided violence data to fill the pre-2016 gap for non-African countries, since one-sided violence against civilians correlates highly with terrorism events.

### 2.4 Social & Environmental

#### UN Population Division

- **Provides:** Youth bulge (population aged 15–29 as share of total), urbanization rate, urban population growth rate, age-structure distributions.
- **Access:** REST API with Bearer token from UN Data Portal, or bulk CSV download (no auth).
- **Python method:** `requests` with `Authorization: Bearer ...` header.
- **Citation:** United Nations, Department of Economic and Social Affairs, Population Division (2024). World Population Prospects.

#### FAO Food Security Indicators

- **Provides:** Prevalence of undernourishment (PoU), food price index, dietary energy supply adequacy, cereal import dependency ratio.
- **Access:** FAOSTAT API (JWT token) or bulk CSV download (no auth). Dataset code `FS`.
- **Python method:** `faostat` package or `pandas.read_csv()` on bulk download.
- **Citation:** FAO (annual). Suite of Food Security Indicators. FAOSTAT.

#### ND-GAIN Country Index

- **Provides:** Overall index, vulnerability score (exposure, sensitivity, adaptive capacity), readiness score (economic, governance, social). Higher readiness = better prepared; higher vulnerability = more exposed.
- **Access:** Bulk CSV download (ZIP) from [gain.nd.edu](https://gain.nd.edu/our-work/country-index/download-data/). No API.
- **Python method:** `pandas.read_csv()` on extracted CSVs.
- **Citation:** Chen, C. et al. (2015). University of Notre Dame Global Adaptation Initiative Country Index. ND-GAIN.
- **Note:** Readiness is inverted in §3.3 (high readiness = low instability).

#### World Bank WDI — Displacement Data

- **Provides:** Refugee population by country of asylum (`SM.POP.REFG`), internally displaced persons (`VC.IDP.TOTL.HE`).
- **Access:** Same as WDI above via `wbgapi`.

### 2.5 Linkage Data (for Coupling Matrix)

#### IMF Direction of Trade Statistics (DOTS)

- **Provides:** Bilateral trade flows (exports and imports) between country pairs, in USD.
- **Access:** SDMX API via `sdmx1` or `imfp` packages. No API key required.
- **Python method:** `imfp.imf_dataset("DOT", freq="A", ...)`.
- **Citation:** International Monetary Fund. Direction of Trade Statistics.
- **Rationale for choice over UN Comtrade:** DOTS requires no API key, has no rate limits for our scale, and provides the aggregate bilateral trade data we need. UN Comtrade provides commodity-level detail we do not require.

#### BIS Locational Banking Statistics

- **Provides:** Bilateral financial claims and liabilities between banking systems.
- **Access:** Same as BIS consolidated (§2.2) — SDMX API or bulk CSV.
- **Same coverage limitations as §2.2.**

#### CEPII GeoDist Database

- **Provides:** Bilateral distance (great-circle, population-weighted), contiguity indicator, common official language, common ethnolinguistic language, colonial relationship indicator, common colonizer, current colony.
- **Access:** Bulk CSV from [cepii.fr](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele.asp).
- **Python method:** `pandas.read_csv()`.
- **Citation:** [@mayer2011geodist]

#### Correlates of War — Alliance Data (v4.1)

- **Provides:** Formal alliance ties classified as: defense pact, neutrality/nonaggression pact, entente. Dyadic format with start/end years.
- **Access:** Bulk CSV from [correlatesofwar.org](https://correlatesofwar.org/data-sets/formal-alliances/).
- **Python method:** `pandas.read_csv()`.
- **Citation:** Gibler, D. M. (2009). International Military Alliances, 1648–2008. CQ Press.

#### Correlates of War — Militarized Interstate Disputes (MID v5)

- **Provides:** Dyadic militarized disputes with hostility level, revisionist state, fatality level.
- **Access:** Bulk CSV from correlatesofwar.org.
- **Usage:** Active MIDs indicate interstate rivalry/tension, which transmits instability bidirectionally.

#### UN DESA — Bilateral Migrant Stock

- **Provides:** Bilateral origin-destination matrix of international migrant stock (number of migrants born in country j living in country i).
- **Access:** Bulk Excel from [UN DESA](https://www.un.org/development/desa/pd/content/international-migrant-stock). Complex multi-tab matrices.
- **Python method:** `pandas.read_excel()` with extensive parsing of matrix format.
- **Citation:** United Nations, Department of Economic and Social Affairs (2020). International Migrant Stock.
- **Limitations:** Updated approximately every 5 years (most recent: 2020). We use the most recent available snapshot and hold constant for years between releases.

---

## 3. Indicator Engineering

Implemented in `features/build_indicators.py` with configuration in `features/pillar_config.py`.

### 3.1 Temporal Alignment

All series are aligned to **annual frequency** covering the **most recent 15 years** (e.g., 2009–2024 for a 2024 run).

- **Slowly-changing governance indicators** (V-Dem, WGI, Freedom House, Polity5): Forward-fill gaps of up to 2 years. This is defensible because institutional characteristics change slowly; a 1-year gap in WGI reporting does not indicate a governance change.
- **Economic series** (WDI, WEO): Linear interpolation for single-year gaps only. No extrapolation.
- **Event-based series** (ACLED, UCDP): Aggregated to annual counts. Missing years treated as zero events (conservative assumption documented in `LIMITATIONS.md`).
- **Polity5 post-2018:** For years after Polity5's coverage ends, we use V-Dem's `v2x_regime` (Regimes of the World ordinal classification) mapped to the Polity2 scale via a crosswalk trained on overlapping years (2000–2018). The crosswalk is a simple ordinal mapping: V-Dem closed autocracy → Polity2 ∈ [−10, −6], electoral autocracy → [−5, 0], electoral democracy → [1, 5], liberal democracy → [6, 10], with the exact value within each bin determined by V-Dem's polyarchy score. This introduces measurement error documented in `LIMITATIONS.md`.

### 3.2 Imputation

We use **Multivariate Imputation by Chained Equations (MICE)** [@vanbuuren2011mice; @rubin1987multiple; @vanbuuren2018flexible].

**Implementation:**
- `sklearn.impute.IterativeImputer` with `BayesianRidge` estimator (default MICE specification).
- `max_iter=50`, `random_state` fixed for reproducibility.
- `n_imputations=10` — we generate 10 multiply-imputed datasets and pool results following Rubin's rules [@rubin1987multiple].

**Regional restriction:**
- Imputation donors are restricted to countries within the same **UN M49 region** (e.g., Western Europe, Eastern Africa, South-Eastern Asia). This prevents implausible imputations such as using Nordic governance values to fill missing data for Sahel countries.
- Implementation: Run IterativeImputer separately for each region, then concatenate.

**Missingness flagging:**
- Any country-indicator pair with **>40% missingness** over the 15-year panel is flagged and excluded from the primary composite index. These indicators remain in an extended dataset for robustness checks.
- A missingness report is generated in `data/clean/missingness_report.csv`.

**Expected high-missingness indicators:** Gini coefficient (~50%), BIS financial exposures for non-reporting countries, some FSI sub-indicators for micro-states.

### 3.3 Sign Orientation

All indicators are oriented so that **higher values = more unstable/fragile**. This ensures interpretability: the leading eigenvector points in the direction of maximum instability.

| Indicator | Source | Original Direction | Action |
|---|---|---|---|
| Polyarchy index | V-Dem | Higher = more democratic | **Invert** (multiply by −1 before standardization) |
| Liberal democracy | V-Dem | Higher = more democratic | **Invert** |
| Electoral democracy | V-Dem | Higher = more democratic | **Invert** |
| Deliberative democracy | V-Dem | Higher = more democratic | **Invert** |
| Egalitarian democracy | V-Dem | Higher = more democratic | **Invert** |
| Polity2 score | Polity5 | Higher = more democratic | **Invert** |
| Executive constraints | Polity5 | Higher = more constrained | **Invert** |
| Regime durability | Polity5 | Higher = more durable | **Invert** |
| Voice & Accountability | WGI | Higher = better | **Invert** |
| Political Stability | WGI | Higher = better | **Invert** |
| Government Effectiveness | WGI | Higher = better | **Invert** |
| Regulatory Quality | WGI | Higher = better | **Invert** |
| Rule of Law | WGI | Higher = better | **Invert** |
| Control of Corruption | WGI | Higher = better | **Invert** |
| Freedom score (PR + CL) | Freedom House | Lower = more free (1–7) | **Keep** (higher already = less free) |
| CPI score | TI | Higher = less corrupt (0–100) | **Invert** |
| GDP growth | WDI | Higher = better | **Invert** |
| Reserves months of imports | WDI | Higher = better | **Invert** |
| Current account % GDP | WDI | Ambiguous (surplus vs deficit) | **Take absolute value of deficit** (negative → positive, positive → 0) |
| ND-GAIN readiness | ND-GAIN | Higher = more ready | **Invert** |
| ND-GAIN vulnerability | ND-GAIN | Higher = more vulnerable | **Keep** |
| FSI all 12 sub-indicators | FFP | Higher = more fragile | **Keep** |
| Inflation | WDI | Higher = worse | **Keep** |
| Unemployment | WDI | Higher = worse | **Keep** |
| Youth unemployment | WDI | Higher = worse | **Keep** |
| Gini coefficient | WDI | Higher = more unequal | **Keep** |
| Gross debt % GDP | IMF WEO | Higher = more indebted | **Keep** |
| Fiscal balance % GDP | IMF WEO | Deficit is negative | **Invert** (so deficit = positive = unstable) |
| External debt | IMF WEO | Higher = more exposed | **Keep** |
| Military expenditure % GDP | SIPRI | Higher = more militarized | **Keep** |
| ACLED battle events | ACLED | Higher = more violent | **Keep** |
| ACLED violence vs civilians | ACLED | Higher = more violent | **Keep** |
| ACLED riots | ACLED | Higher = more unrest | **Keep** |
| ACLED protests | ACLED | Higher = more protests | **Keep with caveat** (see §3.5 note) |
| UCDP conflict events | UCDP | Higher = more conflict | **Keep** |
| Youth bulge (15–29 share) | UN Pop | Higher = larger youth cohort | **Keep** (youth bulge associated with instability) |
| Urbanization rate | UN Pop | Higher = more urban | **Keep** (rapid urbanization strains services) |
| Undernourishment | FAO | Higher = more food insecure | **Keep** |
| Refugee population per capita | WDI | Higher = more displacement | **Keep** |
| IDPs | WDI | Higher = more displacement | **Keep** |

### 3.4 Outlier Treatment and Standardization

Following the OECD Handbook on Constructing Composite Indicators [@oecd2008handbook]:

**Step 1: Winsorization**
- All indicator values are winsorized at the **1st and 99th percentiles**. Values below the 1st percentile are set to P1; values above P99 are set to P99.
- Rationale: Small failed states and extreme outliers (e.g., Venezuela's hyperinflation, Somalia's conflict counts) would otherwise dominate the variance and distort PCA.

**Step 2: Robust z-score standardization**

$$z_i = \frac{x_i - \text{median}(x)}{\text{MAD}(x)}$$

where MAD = median absolute deviation = `median(|x_i − median(x)|) × 1.4826` (the constant scales MAD to be consistent with the standard deviation for normal distributions).

- Rationale: The median and MAD are robust to the remaining influence of outliers after winsorization. Standard mean/SD would still be pulled by extreme values common in governance and conflict data [@oecd2008handbook].

### 3.5 Pillar Assignment

Every indicator is assigned to one of **seven canonical pillars**. The assignment follows standard political science groupings and is informed by the FSI's own four-category framework, the WGI's governance dimensions, and the OECD composite indicator methodology [@oecd2008handbook].

#### Pillar 1: Political Legitimacy

Captures the degree to which the population perceives the government as legitimate, elections as fair, and political institutions as representative.

| Indicator | Source | Variable Code | Sign |
|---|---|---|---|
| Polyarchy index | V-Dem | `v2x_polyarchy` | Inverted |
| Electoral democracy | V-Dem | `v2x_polyarchy` | Inverted |
| Liberal democracy | V-Dem | `v2x_libdem` | Inverted |
| Polity2 score | Polity5 | `polity2` | Inverted |
| Political Rights | Freedom House | `PR` | Keep (high = worse) |
| Civil Liberties | Freedom House | `CL` | Keep (high = worse) |
| State Legitimacy | FSI | `P1` | Keep |
| Factionalized Elites | FSI | `C2` | Keep |

#### Pillar 2: State Capacity & Rule of Law

Captures the state's ability to deliver services, enforce laws, and control corruption.

| Indicator | Source | Variable Code | Sign |
|---|---|---|---|
| Government Effectiveness | WGI | `GE.EST` | Inverted |
| Rule of Law | WGI | `RL.EST` | Inverted |
| Regulatory Quality | WGI | `RQ.EST` | Inverted |
| Control of Corruption | WGI | `CC.EST` | Inverted |
| CPI score | TI | `cpi_score` | Inverted |
| Executive constraints | Polity5 | `xconst` | Inverted |
| Public Services | FSI | `P2` | Keep |
| Human Rights & Rule of Law | FSI | `P3` | Keep |

#### Pillar 3: Economic Performance

Captures macroeconomic health, employment, and distributional equity.

| Indicator | Source | Variable Code | Sign |
|---|---|---|---|
| GDP growth | WDI | `NY.GDP.MKTP.KD.ZG` | Inverted |
| Inflation (CPI) | WDI | `FP.CPI.TOTL.ZG` | Keep |
| Unemployment rate | WDI | `SL.UEM.TOTL.ZS` | Keep |
| Youth unemployment | WDI | `SL.UEM.1524.ZS` | Keep |
| Gini coefficient | WDI | `SI.POV.GINI` | Keep |
| Economic Decline | FSI | `E1` | Keep |
| Uneven Economic Dev. | FSI | `E2` | Keep |

#### Pillar 4: Fiscal & External Vulnerability

Captures sovereign fiscal health, debt sustainability, and external balance sheet risks.

| Indicator | Source | Variable Code | Sign |
|---|---|---|---|
| Fiscal balance % GDP | IMF WEO | `GGXCNL_NGDP` | Inverted (deficit → positive) |
| Gross govt debt % GDP | IMF WEO | `GGXWDG_NGDP` | Keep |
| External debt | IMF WEO | — | Keep |
| Current account deficit | WDI | `BN.CAB.XOKA.GD.ZS` | Absolute deficit |
| Reserves (months imports) | WDI | `FI.RES.TOTL.MO` | Inverted |

#### Pillar 5: Social Cohesion & Demography

Captures social stress factors: demographic pressure, displacement, food insecurity, group grievance.

| Indicator | Source | Variable Code | Sign |
|---|---|---|---|
| Youth bulge (15–29 %) | UN Pop | — | Keep |
| Urbanization rate | UN Pop | — | Keep |
| Urban growth rate | UN Pop | — | Keep |
| Refugee pop. per capita | WDI | `SM.POP.REFG` | Keep |
| IDPs | WDI | `VC.IDP.TOTL.HE` | Keep |
| Demographic Pressures | FSI | `S1` | Keep |
| Human Flight/Brain Drain | FSI | `E3` | Keep |
| Group Grievance | FSI | `C3` | Keep |
| Undernourishment prev. | FAO | `PoU` | Keep |

#### Pillar 6: Security & Violence

Captures armed conflict, political violence, militarization, and security sector dysfunction.

| Indicator | Source | Variable Code | Sign |
|---|---|---|---|
| UCDP conflict events | UCDP | — | Keep |
| ACLED battle events (12mo) | ACLED | `event_type=Battles` | Keep |
| ACLED violence vs civilians | ACLED | `event_type=Violence against civilians` | Keep |
| ACLED explosions/remote | ACLED | `event_type=Explosions/Remote violence` | Keep |
| ACLED riots (12mo) | ACLED | `event_type=Riots` | Keep |
| ACLED protests (12mo) | ACLED | `event_type=Protests` | **Keep with caveat** |
| Military expend. % GDP | SIPRI | — | Keep |
| Security Apparatus | FSI | `C1` | Keep |
| External Intervention | FSI | `X1` | Keep |

> **⚠️ Note on protests:** Protest counts are ambiguous as an instability indicator. Democracies typically have higher protest rates than autocracies, reflecting healthy civic engagement rather than instability. Including protests biases the model toward conflating democratic participation with instability. We include protests in the indicator matrix but assign them a **lower weight within the pillar** (0.5× relative to other indicators) and conduct a sensitivity analysis excluding protests entirely. Results with and without protests are reported in `VALIDATION.md`.

#### Pillar 7: Environmental & Resource Stress

Captures climate vulnerability, environmental fragility, and resource-driven stress.

| Indicator | Source | Variable Code | Sign |
|---|---|---|---|
| ND-GAIN vulnerability | ND-GAIN | `vulnerability` | Keep |
| ND-GAIN readiness | ND-GAIN | `readiness` | Inverted |
| Food price index | FAO | — | Keep |
| Urban growth rate | UN Pop | — | Keep |

### 3.6 Output

The indicator engineering step produces:

- **`X`** — a **(125 × K)** country × indicator matrix for the most recent year, where K ≈ 45–50 depending on availability and missingness exclusions.
- **`X_panel`** — a **(125 × K × T)** panel, where T = 15 years.
- **`pillar_assignment`** — mapping of each of the K indicators to its pillar.
- **`missingness_report`** — country × indicator missingness rates with flags.
- **`imputation_diagnostics`** — convergence plots and distribution comparisons for MICE.

All outputs are saved as Parquet files under `data/clean/`.

---

## 4. Spectral Core — Latent Instability Factors

Implemented in `model/pca.py`.

### 4.1 Eigendecomposition of the Indicator Correlation Matrix

We compute PCA on the correlation matrix rather than the covariance matrix to avoid scale effects [@jolliffe2002pca]:

$$\Sigma = \text{corr}(X)$$

where X is the (125 × K) standardized indicator matrix from §3.

The eigendecomposition:

$$\Sigma = V \Lambda V^T$$

where $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_K)$ with $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_K$, and $V = [v_1 | v_2 | \cdots | v_K]$ is the matrix of eigenvectors.

### 4.2 Component Retention

We retain components satisfying **both**:
1. **Kaiser criterion:** Eigenvalue $\lambda_k > 1$ (i.e., the component explains more variance than any single original variable) [@jolliffe2002pca].
2. **Cumulative variance ≥ 80%:** $\sum_{k=1}^{r} \lambda_k / \sum_{k=1}^{K} \lambda_k \geq 0.80$.

If these criteria conflict (e.g., Kaiser retains too few components for 80% variance), we use the more conservative criterion (retain more components).

### 4.3 Varimax Rotation

We apply **varimax rotation** [@kaiser1958varimax] to the retained loadings matrix to improve interpretability:

$$\tilde{V}_r = V_r R$$

where $R$ is the orthogonal rotation matrix that maximizes the variance of squared loadings within each component. This produces components where each variable loads strongly on at most one component, making pillar interpretation clearer.

Implementation: `scipy.spatial.transform` or `factor_analyzer` package.

### 4.4 Composite Instability Score

Each country's composite instability score is computed as a weighted sum of its component scores, weighted by the corresponding eigenvalues (variance explained):

$$s_i = \sum_{k=1}^{r} \frac{\lambda_k}{\sum_{j=1}^{r} \lambda_j} \cdot z_{ik}$$

where $z_{ik}$ is country $i$'s score on (rotated) component $k$.

This follows the OECD recommendation for variance-weighted aggregation [@oecd2008handbook].

### 4.5 Sanity Check

The leading eigenvector (first principal component scores) should correlate **≥ 0.7** (Pearson or Spearman) with the **FSI headline score**. This is a face-validity check: since the FSI is the most widely used fragility index, a model that produces radically different country rankings should be investigated.

- If $\rho \geq 0.7$: Pass. Proceed.
- If $\rho < 0.7$: Investigate. Likely causes include (a) a pillar with anomalous loadings, (b) an imputation artifact, (c) a sign-orientation error. Document the investigation in `VALIDATION.md` regardless of outcome.

### 4.6 Interpretation

The rotated components are examined for interpretability by inspecting which indicators load most heavily. We expect (but do not force) components to roughly align with the seven pillars. Cross-loading indicators are noted and discussed.

The **"systemic instability axis"** is defined as the first principal component. Countries with the highest scores on this axis are the most unstable along the dominant latent dimension.

---

## 5. Country Coupling Matrix and Network Spectrum

Implemented in `model/coupling.py`.

### 5.1 Overview

We construct a **(125 × 125)** non-negative coupling matrix $W$ whose entry $W_{ij}$ measures how much instability in country $j$ propagates to country $i$. The matrix is **asymmetric** — a small country's dependence on a large trading partner is not reciprocated equally.

$W$ is built as a weighted combination of five normalized sub-matrices:

$$W = w_{\text{trade}} \cdot \hat{W}_{\text{trade}} + w_{\text{fin}} \cdot \hat{W}_{\text{fin}} + w_{\text{geo}} \cdot \hat{W}_{\text{geo}} + w_{\text{pol}} \cdot \hat{W}_{\text{pol}} + w_{\text{mig}} \cdot \hat{W}_{\text{mig}}$$

where the hat denotes row-normalized sub-matrices and the default weights are:

| Channel | Weight | Rationale |
|---|---|---|
| Trade (`w_trade`) | 0.30 | Primary real-economy contagion channel; most complete data |
| Financial (`w_fin`) | 0.25 | Financial contagion is fast-acting but data is sparse |
| Geographic (`w_geo`) | 0.15 | Proximity enables refugee flows, border spillover |
| Political (`w_pol`) | 0.15 | Alliances and rivalries create commitment/tension channels |
| Migration (`w_mig`) | 0.15 | Diaspora ties, social remittances, refugee pressure |

These weights are exposed as configuration in `config.yaml` and their sensitivity is analyzed in §8.

### 5.2 Sub-Matrix Construction

#### $W_{\text{trade}}$ — Trade Dependency

$$W_{\text{trade}}[i,j] = \frac{\text{exports}_{ij} + \text{imports}_{ij}}{\text{total\_trade}_i}$$

where $\text{total\_trade}_i = \sum_k (\text{exports}_{ik} + \text{imports}_{ik})$.

- Source: IMF DOTS bilateral trade flows.
- This is naturally asymmetric: a small country that trades heavily with a large partner has high $W_{\text{trade}}[i,j]$, but the large partner's $W_{\text{trade}}[j,i]$ is small.
- Diagonal: $W_{\text{trade}}[i,i] = 0$ (no self-loops).

#### $W_{\text{fin}}$ — Financial Exposure

$$W_{\text{fin}}[i,j] = \frac{\text{claims}_{ij} + \text{liabilities}_{ij}}{\text{GDP}_i}$$

- Source: BIS consolidated + locational banking statistics.
- Normalized by country $i$'s GDP to capture relative exposure.
- For non-reporting countries: use mirror data where available; otherwise $W_{\text{fin}}[i,j] = 0$.
- Diagonal: 0.

#### $W_{\text{geo}}$ — Geographic Proximity

$$W_{\text{geo}}[i,j] = \frac{1}{d_{ij}} + \beta \cdot \text{contiguity}_{ij}$$

where $d_{ij}$ is the population-weighted great-circle distance from CEPII GeoDist [@mayer2011geodist], $\text{contiguity}_{ij} \in \{0, 1\}$, and $\beta = 0.5$ (configurable).

- Row-normalized before combining with other sub-matrices.
- Diagonal: 0 (distance to self is undefined; set to 0).

#### $W_{\text{pol}}$ — Political Ties

$$W_{\text{pol}}[i,j] = \sum_{\text{type}} \text{alliance}_{ij}^{(\text{type})} + \gamma \cdot \text{active\_MID}_{ij}$$

where alliance types are defense pact (+1), neutrality/nonaggression (+0.5), entente (+0.3), and $\gamma = 1.0$ for active militarized interstate disputes.

- Source: COW Alliance v4.1 + COW MID v5.
- **Rationale for including both alliances and rivalries:** Both transmit instability. Alliance partners are pulled into each other's crises through commitment mechanisms; rivals transmit instability through escalation spirals and arms races.
- Symmetric by construction (alliances and MIDs are dyadic). Row-normalized before combination.

#### $W_{\text{mig}}$ — Migration Linkages

$$W_{\text{mig}}[i,j] = \frac{\text{migrants\_from\_j\_in\_i}}{\text{population}_i}$$

- Source: UN DESA bilateral migrant stock.
- Captures diaspora connections, social remittance channels, and refugee-driven pressure.
- Asymmetric: receiving countries bear the exposure.
- Diagonal: 0.

### 5.3 Row Normalization

After combining:

$$W_{\text{norm}}[i,:] = \frac{W[i,:]}{\sum_j W[i,j]}$$

This ensures each row sums to 1, making $W$ **row-stochastic**. Substantively, this imposes a "total exposure budget" per country — the question is not how much total exposure a country has, but how that exposure is distributed across partners.

### 5.4 Spectral Analysis of $W$

Since $W$ is non-symmetric, it generally has **complex eigenvalues**. We compute the full eigendecomposition using `numpy.linalg.eig`.

**Key outputs:**

1. **Eigenvalue magnitudes** $|\lambda_k|$, sorted in decreasing order. Since $W$ is row-stochastic, $\rho(W) = \lambda_1 = 1$ by the Perron-Frobenius theorem.

2. **Spectral gap** $\delta = |\lambda_1| - |\lambda_2|$. A larger spectral gap means faster convergence of the dynamic model (§6) and cleaner separation of the dominant mode from secondary modes [@newman2010networks].

3. **Right eigenvector centrality** — the leading right eigenvector $v_R$ (corresponding to $\lambda_1 = 1$) identifies countries whose instability has the **broadest downstream impact** (systemic transmitters). This is the classical eigenvector centrality measure [@bonacich1972factoring].

4. **Left eigenvector** — the leading left eigenvector $v_L$ identifies countries **most exposed to receiving** instability from the network.

5. **Eigenvalue magnitude plot** — visualization of $|\lambda_k|$ vs. $k$ to characterize the spectrum.

---

## 6. Dynamic Propagation Model

Implemented in `model/dynamics.py`.

### 6.1 Model Specification

$$x(t+1) = \alpha \cdot W \cdot x(t) + (1 - \alpha) \cdot s + \eta(t)$$

where:
- $x(t) \in \mathbb{R}^{125}$ — instability vector at time $t$
- $s \in \mathbb{R}^{125}$ — structural baseline from §4 (composite instability scores)
- $\alpha \in [0, 1)$ — coupling strength (default 0.4)
- $\eta(t) \in \mathbb{R}^{125}$ — idiosyncratic shock term (default zero; nonzero in simulation scenarios)
- $W$ — the row-stochastic coupling matrix from §5

### 6.2 Theoretical Framework

This model is a **Friedkin-Johnsen opinion dynamics model** [@friedkin1990social] with stochastic perturbations, applied to cross-country instability propagation. It generalizes the DeGroot consensus model [@degroot1974consensus]:

- **Pure DeGroot** ($\alpha = 1$, $s = 0$, $\eta = 0$): $x(t+1) = W \cdot x(t)$. All countries converge to a weighted average — consensus. No structural anchoring.
- **Friedkin-Johnsen** ($0 < \alpha < 1$, $\eta = 0$): Each country's instability is a blend of network influence (neighbors' instability) and intrinsic susceptibility. Countries are "stubborn" — they don't fully converge to neighbors.
- **Our model** adds $\eta(t)$ for scenario simulation (exogenous shocks).

The parameter $\alpha$ interpolates between independent country dynamics ($\alpha = 0$: instability is purely structural) and full network contagion ($\alpha \to 1$: instability is purely transmitted).

See also [@acemoglu2011opinion] for a survey of opinion dynamics models and [@jackson2008social] for the network-theoretic foundations.

### 6.3 Stability Analysis

The system is stable (converges to a fixed point) if and only if the spectral radius of $\alpha W$ is less than 1:

$$\rho(\alpha W) = \alpha \cdot \rho(W) < 1$$

Since $W$ is row-stochastic, $\rho(W) = 1$ by the Perron-Frobenius theorem. Therefore:

$$\text{Stability} \iff \alpha < 1$$

This result comes from epidemic threshold theory [@chakrabarti2008epidemic; @pastorsatorras2001epidemic]. The key insight: **$\alpha$ is the operational tunable that controls whether the system is damped or explosive.** When $\alpha$ is close to 1, even small shocks are massively amplified through the network before dying out. When $\alpha$ is small, network effects are weak and each country's instability is close to its structural baseline.

We report $\alpha$ prominently in all outputs and document this stability threshold clearly in the UI.

### 6.4 Steady State

When $\eta(t) = 0$ and $\alpha < 1$, the system converges to:

$$x^* = (I - \alpha W)^{-1} \cdot (1 - \alpha) \cdot s$$

**Interpretation:** The steady state **amplifies the structural baseline** through the network. Countries with high eigenvector centrality in $W$ (systemic transmitters) and countries with high-centrality neighbors will see their instability amplified beyond their structural level. The matrix $(I - \alpha W)^{-1}$ is the **Leontief inverse** (borrowing the term from input-output economics), encoding both direct and indirect network transmission.

Column $i$ of $(I - \alpha W)^{-1}$ gives the **sensitivity** of every country's steady-state instability to a marginal increase in country $i$'s structural baseline.

### 6.5 Calibration of $\alpha$

Implemented in `model/calibrate.py`.

**Objective:** Calibrate $\alpha$ to minimize one-year-ahead prediction error of the composite instability index.

**Method:**
1. For each year $t$ in the 10-year calibration window (leave-one-year-out cross-validation):
   a. Compute $s(t)$ from that year's indicator matrix.
   b. Compute $W(t)$ from that year's linkage data.
   c. For a given $\alpha$, predict $\hat{x}(t+1) = \alpha W(t) s(t) + (1 - \alpha) s(t)$. This 1-step transient update is used rather than the infinite-time steady state, as it aligns with the 1-year predictive horizon.
   d. Compare $\hat{x}(t+1)$ against $x_{\text{actual}}(t+1)$ (the composite score computed from actual $t+1$ data).
2. Objective: Minimize RMSE over all hold-out years.
3. Search: Grid search over $\alpha \in [0.05, 0.95]$ with step 0.01.

**Reporting:**
- Calibrated $\hat{\alpha}$ and its cross-validated RMSE.
- RMSE as a function of $\alpha$ (sensitivity plot).
- Bootstrap 90% CI on $\hat{\alpha}$ (resample countries within each fold, B=500).

---

## 7. Simulation Capabilities

Implemented in `model/dynamics.py` (core) and `app/streamlit_app.py` + `app/cli.py` (interfaces).

### 7.1 Factor Shocks

**Input:** Country $i$, pillar $p$, magnitude $\delta$ (in standard deviations).

**Procedure:**
1. Increase country $i$'s indicators in pillar $p$ by $\delta$ standard deviations.
2. Recompute composite score $s_i^{\text{shocked}} = s_i + \delta \cdot w_p$, where $w_p$ is the pillar's contribution weight from PCA loadings.
3. Compute the new steady state: $x^{*\text{new}} = (I - \alpha W)^{-1} \cdot (1 - \alpha) \cdot s^{\text{shocked}}$.
4. Also simulate the trajectory $x(0) = x^*_{\text{baseline}}$, $x(t+1) = \alpha W x(t) + (1-\alpha) s^{\text{shocked}}$ for $N$ steps.

**Output:**
- Trajectory plots for the shocked country and its top-10 transmission partners (by $W$ weights).
- New equilibrium vs. baseline delta.
- Time to convergence (within 1% of new steady state).

### 7.2 Edge Shocks

**Input:** Country pair $(i, j)$, channel (trade/finance/geo/political/migration), new value or multiplier.

**Procedure:**
1. Modify the specific sub-matrix entry (e.g., zero out $W_{\text{trade}}[i,j]$ and $W_{\text{trade}}[j,i]$).
2. Recompute the combined $W$ and row-normalize.
3. Recompute the full eigendecomposition of the new $W$.
4. Compute new steady state.

**Output:**
- Spectral comparison: eigenvalue magnitudes of original vs. perturbed $W$.
- New eigenvector centrality rankings.
- Steady-state delta for all countries.

### 7.3 Multi-Country Compound Scenarios

**Input:** Multiple simultaneous shocks (factor and/or edge).

**Procedure:**
1. Apply all shocks simultaneously.
2. Check whether the spectral radius of the perturbed effective operator ($\alpha W_{\text{perturbed}}$) exceeds 1 — if so, the system has tipped from damped to explosive.
3. If stable, compute new steady state. If explosive, report the divergence rate.

**Output:**
- Stability assessment (damped vs. explosive).
- Steady-state comparisons (if stable).
- Critical $\alpha$ threshold for the perturbed system.

### 7.4 Sensitivity Decomposition

**Input:** Target country $i$.

**Procedure:** Extract column $i$ of $(I - \alpha W)^{-1}$.

**Output:** For every other country $j$, the partial derivative $\partial x^*_j / \partial s_i$ — how much a unit increase in country $i$'s structural instability raises country $j$'s steady-state instability.

### 7.5 Key Design Principle

The UI explicitly distinguishes:
- **Structural state** — where the country sits in latent instability space (from PCA), reflecting its intrinsic conditions.
- **Transmitted state** — how much instability arrives from its network neighbors (the amplification via $(I - \alpha W)^{-1}$).

Both are informative and are displayed separately. Conflating them is the most common failure mode of network-instability dashboards — a country with low structural instability but high eigenvector centrality neighbors (e.g., Jordan) will show up differently in each view.

---

## 8. Uncertainty Quantification

### 8.1 Bootstrap Framework

We use nonparametric bootstrap resampling [@efron1979bootstrap] with B = 1000 iterations to propagate uncertainty through the entire model pipeline.

**Sources of uncertainty resampled:**
1. **Indicator weights:** Resample the K indicators (with replacement) before computing PCA. This varies which indicators dominate each component.
2. **Imputation draws:** For each bootstrap iteration, draw one of the m = 10 multiply-imputed datasets. This propagates imputation uncertainty per Rubin's rules [@rubin1987multiple].
3. **Coupling matrix weights:** Perturb the 5 channel weights by drawing from a symmetric Dirichlet distribution centered on the default weights with concentration parameter $\kappa = 50$ (tight around defaults but allowing meaningful variation).

### 8.2 Handling Rotational Ambiguity

PCA eigenvectors are defined only up to sign, and the ordering of components can switch across bootstrap samples (especially when adjacent eigenvalues are close). Following [@timmerman2007confidence]:

1. Compute PCA on the original data to obtain reference loadings $V_0$.
2. For each bootstrap sample $b$, compute PCA loadings $V_b$.
3. Align $V_b$ to $V_0$ via Procrustes rotation: find the orthogonal $R$ minimizing $\|V_b R - V_0\|_F$.
4. Use the aligned loadings for inference.

### 8.3 Outputs

For every reported quantity, we provide the **point estimate** and **90% bootstrap confidence interval** (5th and 95th percentiles):

- Country composite instability scores.
- Country rankings (reported as rank intervals).
- Eigenvalues and variance explained.
- Eigenvector centrality rankings (right and left).
- Steady-state instability under each scenario.
- Calibrated $\alpha$.

---

## 9. Validation Strategy

Detailed results are reported in `VALIDATION.md`.

### 9.1 Historical Event Reconstruction

For each of these episodes, we run the model with that year's exogenous data and verify:
(a) The affected countries' instability scores rise sharply relative to the prior year.
(b) The model's "top transmission risks" list correctly identifies the secondary countries.

| Episode | Years | Primary Countries | Expected Secondary Transmission |
|---|---|---|---|
| Arab Spring | 2010–2012 | Tunisia, Egypt, Libya, Syria, Yemen, Bahrain | Jordan, Morocco, Algeria, Lebanon, Iraq |
| Euro sovereign debt crisis | 2010–2012 | Greece, Ireland, Portugal, Spain, Italy | France, Germany, Cyprus, Belgium |
| 2014 oil price collapse + Russia/Ukraine | 2014–2015 | Russia, Ukraine, Venezuela, Nigeria, Saudi Arabia | Kazakhstan, Belarus, Azerbaijan, Iraq |
| COVID-19 onset | 2020 | Global | Tourism-dependent small states, oil exporters |
| 2022 Russia–Ukraine war | 2022 | Russia, Ukraine | Energy-dependent EU states, grain importers (Egypt, Lebanon, Tunisia) |
| 2023 Sahel coup belt | 2023 | Niger, Mali, Burkina Faso, Gabon | Chad, Nigeria, Guinea, Senegal |

### 9.2 Rank Correlation with External Indices

Compute Spearman rank correlation between our composite instability index and:

1. **Fragile States Index (FSI)** headline score — target: $\rho \geq 0.75$.
2. **EIU Democracy Index** (inverted, so higher = less democratic = more instability-adjacent) — target: $\rho \geq 0.75$.

If targets are not met, we do **not** adjust the model to fit. Instead, we document the discrepancy, investigate its source (e.g., which countries drive divergence), and explain whether the divergence reflects a genuine modeling insight or a deficiency.

### 9.3 Conflict Onset Prediction (AUC)

Using the composite instability score as the sole predictor:

1. **Outcome variable:** Binary indicator for next-year ACLED-defined conflict onset (country transitions from <25 battle-related deaths to ≥25, following UCDP's minor conflict threshold).
2. **Hold-out period:** Most recent 3 years.
3. **Training:** Logistic regression on the remaining years.
4. **Metric:** Area Under the ROC Curve (AUC) — target: ≥ 0.80.

This is an intentionally simple test. If a composite index constructed from the inputs cannot discriminate conflict onset at AUC ≥ 0.80, the index is not capturing the relevant signal. If it does, it validates the indicator selection and weighting — not the network propagation, which is validated separately through §9.1.

---

## References

All citations use BibTeX keys defined in `references.bib`. Key methodological references:

- PCA methodology: [@jolliffe2002pca]
- Composite indicator construction: [@oecd2008handbook]
- MICE imputation: [@vanbuuren2011mice; @rubin1987multiple]
- Varimax rotation: [@kaiser1958varimax]
- Dynamic model (Friedkin-Johnsen): [@friedkin1990social; @degroot1974consensus]
- Opinion dynamics survey: [@acemoglu2011opinion]
- Network centrality: [@bonacich1972factoring; @newman2010networks]
- Epidemic thresholds: [@chakrabarti2008epidemic; @pastorsatorras2001epidemic]
- Financial contagion parallels: [@battiston2012debtrank; @billio2012econometric; @diebold2014network]
- Bootstrap uncertainty: [@efron1979bootstrap; @timmerman2007confidence]
- Conflict diffusion: [@gleditsch2002local; @gleditsch2006diffusion; @salehyan2006refugees]
- Political instability forecasting: [@goldstone2010global; @hegre2013predicting]
- Geographic distance data: [@mayer2011geodist]
- WGI methodology: [@kaufmann2010wgi]
