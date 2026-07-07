# LIMITATIONS.md — Known Limitations and Caveats

> This document enumerates what the spectral instability model **cannot** do, where its assumptions break down, and where users should exercise caution in interpretation.

---

## 1. Data Lag

All indicators are retrospective. The most recent data point is typically 1–2 years old (e.g., WGI has a ~18-month lag; V-Dem releases in spring for the prior year). The model describes **where instability was**, not where it is right now. For countries undergoing rapid change (coups, wars, economic crises), the model's snapshot may be outdated by the time results are published.

**Implication:** The model is an analytical tool for understanding structural vulnerabilities and contagion pathways, not a real-time early warning system.

## 2. Autocratic Data Manipulation

Authoritarian regimes systematically misreport economic statistics (GDP growth, unemployment, inflation), suppress conflict data (restricting journalist and NGO access), and manipulate governance assessments. This is well-documented for countries like China, Russia, Turkmenistan, and North Korea.

**Implication:** The model may **underestimate** instability in closed autocracies where the input data itself is unreliable. V-Dem and WGI partially account for this through expert surveys, but economic data from IMF WEO and WDI relies on national statistical offices that may be compromised.

## 3. Black Swan Events

The model captures **structural** vulnerability — the slow-moving conditions that make instability more likely. It does not and cannot predict:
- Assassination of a head of state
- Natural disasters (earthquakes, tsunamis, pandemics)
- Sudden commodity price crashes
- Military mutinies or palace coups
- Terrorist attacks

These are exogenous shocks that the model can only analyze *after* they occur (via the simulation interface). The model tells you which countries are dry tinder; it does not predict when the match will be struck.

## 4. Endogenous Policy Response

The model treats the structural baseline $s$ as exogenous. In reality, governments respond to rising instability with policy interventions — economic stimulus, security crackdowns, political reforms, international aid negotiations. These responses may prevent the instability trajectories the model projects.

**Implication:** The model's projected trajectories assume no policy change. Steady-state instability scores should be interpreted as "what would happen if current conditions persisted without intervention," not as forecasts.

## 5. Instability ≠ Regime Change

High instability scores do not imply regime change is imminent or even likely. Many highly fragile states persist for decades (e.g., DRC, Somalia, Afghanistan). The relationship between structural instability and actual regime transitions is mediated by factors the model does not capture: elite cohesion, security force loyalty, external intervention, opposition organization.

**Implication:** Do not interpret composite scores as probabilities of regime change. They measure structural vulnerability, not transition likelihood.

## 6. Polity5 Temporal Break

Polity5 coverage effectively ends ~2018. For subsequent years, we use a V-Dem crosswalk (see METHODOLOGY.md §3.1). This introduces measurement error because:
- The Polity2 scale and V-Dem's polyarchy index measure different (though correlated) aspects of democracy.
- The crosswalk mapping (ordinal bins scaled by polyarchy) is approximate.
- Countries at regime-type boundaries may be misclassified.

Users should be cautious about interpreting Polity5-derived indicators for post-2018 years.

## 7. BIS Financial Exposure Coverage (Data Bias)

The financial coupling sub-matrix ($W_{\text{fin}}$) is dominated by the ~30 BIS-reporting countries. For the remaining ~95 countries, bilateral financial exposures are largely unknown or inferred from mirror data. 
* **The Reporting Asymmetry:** Because developing countries do not report their banks' outbound claims to the BIS, they only appear as counterparties (borrowers) in data reported by major economies (like Japan or the US).
* **Artificial Concentration Bias:** When the sub-matrix is row-normalized, this reporting asymmetry concentrates almost all of a developing country's financial coupling weight on a few major lenders. For example, because Japan is the dominant regional lender in East Asia and many borrowers don't report claims elsewhere, Japan receives 90%+ of their financial coupling weight post-normalization. This artificially inflates the systemic risk of major regional lenders in the network.
* **Implication:** Financial contagion channels from/to non-reporting countries are underestimated, and the coupling matrix overweights trade and geography for developing nations.

## 8. Static Coupling Matrix

The coupling matrix $W$ is computed from the most recent available data for each channel. In reality, trade, financial, and political relationships evolve. A severed trade relationship (e.g., EU sanctions on Russia) changes $W$ in real time, but our $W$ is a snapshot.

**Mitigation:** The edge shock simulation (METHODOLOGY.md §7.2) allows users to manually modify $W$ entries. But the baseline $W$ reflects pre-shock relationships.

## 9. PCA Linearity Assumption (Linearity vs. Reality)

PCA assumes linear relationships among indicators. If instability is driven by nonlinear interactions (e.g., the combination of youth bulge AND unemployment is more explosive than either alone), PCA will not capture this. 
* **The Tipping Points Issue:** Political crises and regime instabilities are rarely linear processes. Geopolitical networks are prone to non-linear "tipping points" (regime phase transitions) where a country appears completely stable until a threshold is crossed, triggering a sudden, cascade collapse that linear modeling fails to forecast.
* **Implication:** The leading eigenvector describes the linear axis of maximum variance, which may miss critical non-linear interaction effects and threshold-based spillovers.

## 10. GDP Universe Limits (Exclusion of Volatile Zones)

The model's country universe is established by selecting the top 125 nations by nominal GDP. 
* **Exclusion of Low-Income Conflict Zones:** While this includes the vast majority of global economic activity, it systematically excludes highly volatile, low-income regional conflict zones—such as the Sahel region (e.g., Niger, Chad, Gabon). 
* **Implication:** Because these countries are omitted from the network nodes, the model is unable to forecast highly localized, lower-income regional spillovers, limiting its predictive application in sub-Saharan Africa.

## 11. Coupling Matrix Weight Sensitivity

The default channel weights (trade 0.30, finance 0.25, geography 0.15, political 0.15, migration 0.15) are reasonable priors but not empirically calibrated. Different weight choices produce different coupling structures and different eigenvector centrality rankings.

**Mitigation:** Bootstrap uncertainty quantification (METHODOLOGY.md §8) includes weight perturbation. The sensitivity analysis in VALIDATION.md reports how country rankings change under alternative weight schemes.

## 12. Migration Data Staleness

UN DESA bilateral migrant stock data is updated approximately every 5 years. The most recent release (2020) predates major displacement events (2022 Ukraine, 2023 Sudan, 2024 Gaza). The migration coupling sub-matrix is therefore stale for countries affected by recent large-scale displacement.

## 13. Protest Indicator Ambiguity

Protest event counts (from ACLED) are included in the Security & Violence pillar, but their sign-orientation is debatable. In democracies, high protest counts may indicate a healthy, engaged civil society rather than instability. In autocracies, protests genuinely signal regime stress. The model applies a uniform sign orientation (more protests = more instability), which may overestimate instability in democratic countries with active protest cultures (e.g., France, South Korea).

See METHODOLOGY.md §3.5 for the mitigation approach (reduced weight + sensitivity analysis).

## 14. No Subnational Resolution

The model operates at the country level. It cannot distinguish between:
- Stable capital regions and unstable peripheries (e.g., Nigeria: Lagos vs. northeast)
- Autonomous regions with distinct dynamics (e.g., Kurdistan, Xinjiang)
- City-state economies embedded in unstable regions (e.g., Singapore)

Country-level aggregation smooths over critical within-country variation.

## 14. Ecological Inference

PCA on country-level aggregates produces country-level factors. These cannot be used to make inferences about individuals or sub-populations within countries. A country scoring high on "economic stress" does not mean all its citizens are economically stressed — the distribution matters, and our model captures only aggregate indicators like Gini, not full distributions.

## 15. Model Reflexivity

In theory, if this model's outputs were used by policymakers or investors, their actions could change the underlying dynamics — creating a feedback loop the model does not account for. In practice, this is unlikely at the current stage, but it is a general limitation of political forecasting models (see Goodhart's Law).
