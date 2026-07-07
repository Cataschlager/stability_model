# Spectral Model of Global Regime Instability

A computational model using eigenvalue/eigenvector decomposition to characterize and simulate regime instability propagation across the top 125 economies by nominal GDP.

## Overview

This project constructs a two-layer spectral model:

1. **Latent Instability Factors (PCA)** — Decomposes a high-dimensional indicator matrix across seven canonical pillars into a low-dimensional instability space, producing composite scores per country.
2. **Country Coupling Network (Spectral Graph Analysis)** — Builds a directed weighted coupling matrix encoding trade, financial, geographic, political, and migration linkages, then analyzes its spectrum to identify systemic transmitters and receivers of instability.

A discrete-time dynamic propagation model couples these layers, enabling scenario simulation of how shocks in one country transmit through the global network.

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd spectral-instability-model
cp .env.example .env  # Add your API keys
make install

# Run full pipeline
make all

# Launch simulation UI
make serve
```

See [SETUP.md](SETUP.md) for detailed installation and data source configuration.

## Repository Structure

```
├── README.md              # This file
├── SETUP.md               # Installation, API keys, environment setup
├── METHODOLOGY.md         # Full methodological documentation with citations
├── VALIDATION.md          # Validation results and historical reconstruction
├── LIMITATIONS.md         # Known limitations and caveats
├── references.bib         # BibTeX references
├── Makefile               # Reproducibility: `make all` runs everything
├── pyproject.toml         # Dependencies (pinned via uv)
├── data/
│   ├── raw/               # Immutable source downloads
│   └── clean/             # Normalized parquet files
├── ingestion/             # One module per data source
├── features/              # Indicator construction, imputation, standardization
├── model/
│   ├── pca.py             # Latent instability factor analysis
│   ├── coupling.py        # Country coupling matrix construction
│   ├── dynamics.py        # Dynamic propagation model
│   └── calibrate.py       # α calibration via cross-validation
├── validation/            # Historical event reconstruction, benchmarking
├── app/                   # Simulation web UI
├── notebooks/             # Exploratory analysis and figure generation
└── tests/                 # Unit tests on every transformation
```

## Key Outputs

- **Composite Instability Index** — Ranked list of 125 countries with bootstrap confidence intervals
- **Eigenvector Centrality Rankings** — Systemic transmitters and receivers
- **Interactive Simulator** — Factor shocks, edge shocks, compound scenarios
- **Spectral Diagnostics** — Eigenvalue plots, scree plots, spectral gap analysis

## License

[TBD]

## Citation

[TBD]
