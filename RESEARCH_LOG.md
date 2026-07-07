# Research Log: Spectral Instability Model

*This is an immutable, chronologically appended log of all major experiments, parameter calibrations, and validation runs. This ensures reproducibility and prevents "p-hacking" or loss of context for the final academic paper.*

---

## 2026-07-01: Baseline Model Initialization & Code Audit
**Context:** Full pipeline execution (Ingestion → PCA → Coupling → Dynamics). A deep code audit revealed the math is sound, but critical validation steps (α calibration, baseline comparisons, bootstrap uncertainty) were present in code but not orchestrated in the main pipeline.
**Decision:** Implement automated LaTeX injection. Next step is to integrate `calibrate_alpha()` into the main pipeline.

---

- **Auto-run (Calibration):** Optimal $\alpha$ = 0.050 (RMSE = 0.5854)

- **Auto-run (Calibration):** Optimal $\alpha$ = 0.050 (RMSE = 0.5850)

- **Auto-run (Calibration):** Optimal $\alpha$ = 0.050 (RMSE = 0.5850)

- **Auto-run (Calibration):** Optimal $\alpha$ = 0.050 (RMSE = 0.5850)

- **Auto-run (Calibration):** Optimal $\alpha$ = 0.050 (RMSE = 0.5850)

- **Auto-run (Calibration):** Optimal $\alpha$ = 0.050 (RMSE = 0.5850)
