# Spectral Model of Global Regime Instability

A computational research prototype applying spectral graph theory and network science to model cross-border instability contagion across 125 nations. 

## Overview

This project constructs a two-layer analytical framework to assess geopolitical instability:

1. **Latent Instability Factors (PCA)** - Decomposes a high-dimensional matrix of 50 indicators across 7 canonical pillars (governance, economics, conflict, etc.) into a low-dimensional instability space. The resulting composite score correlates highly (Pearson *r*=0.92) with established benchmarks like the Fragile States Index.

![The model's steady-state instability ranking, isolating the 20 most vulnerable nations based on 50 structural and geopolitical indicators.](assets/imagetop20.png)

2. **Country Coupling Network (Spectral Graph Analysis)** - Builds a directed, weighted multiplex coupling matrix encoding trade, financial, geographic, and political linkages. 

![A 3D spectral embedding of the global coupling network, mapping distinct structural communities of instability transmission.](assets/image3d.png)

By applying a discrete-time dynamic propagation model (a Friedkin-Johnsen style mechanism), the system computes a steady-state instability score that accounts for both domestic structural fragility and the stabilizing or destabilizing influence of network neighbors. 

### Key Findings
* **Structural Inertia Dominates:** Calibration of the network coupling parameter (*α*) via leave-one-year-out cross-validation yields a low optimal value (*α* ≈ 0.05). This indicates that on a 1-year horizon, domestic structural factors overwhelmingly dominate over short-term network contagion effects.
* **The Empirical Limits of Contagion:** At empirically calibrated coupling strengths, the global system does not exhibit cascading domino effects. Instead, massive localized shocks are largely absorbed by the structural resilience of stable network neighbors.
* **Centrality Identification:** PageRank and Eigenvector centrality on the multiplex coupling matrix successfully identify systemic hubs (transmitters and receivers) of potential future instability shocks.

## Historical Validation

The model's ability to capture meaningful variance in instability was validated against recent historical shock episodes. As a baseline sanity check, the model's structural composite score exhibits a strong correlation (*r* = 0.92) with the external Fragile States Index. 

For dynamic propagation, the model was backtested against discrete historical crises. Due to data availability constraints in the 15-year panel (specifically regarding the 2020 COVID-19 pandemic and pre-2013 events like the Arab Spring), full temporal validation was restricted to episodes with complete indicator coverage.

**Successful Identifications (In-Sample):**
* **2014 Oil Price & Russian Sanctions Shock**
* **2022 Russian Invasion of Ukraine**
* **2023 Sahel Coup Contagion**

For these episodes, the network's steady-state output successfully ranked the ultimately affected countries in the highest deciles of systemic vulnerability prior to the crisis.

## Quick Start

```bash
# Clone and install
git clone https://github.com/Cataschlager/stability_model.git
cd stability_model
cp .env.example .env  # Add your API keys if you wish to re-download raw data
make install

# Run the full analytical pipeline
make all

# Launch the interactive Streamlit dashboard
make serve
```

See [SETUP.md](SETUP.md) for detailed installation and data source configuration.

## Methodology & Documentation

- [METHODOLOGY.md](METHODOLOGY.md) - Full methodological documentation with mathematical notation.
- [LIMITATIONS.md](LIMITATIONS.md) - Known caveats, including data missingness and bounds of predictive validity.
- [references.bib](references.bib) - Academic citations underpinning the model design.

## Interactive Dashboard

The repository includes a Streamlit application (`make serve`) that visualizes the model's outputs:
- **Most Unstable:** Steady-state scores vs. raw structural scores.
- **Systemic Transmitters:** Outbound eigenvector centrality.
- **Most Exposed:** Inbound eigenvector centrality.
- **World Map:** Geospatial visualization of instability and network metrics.

