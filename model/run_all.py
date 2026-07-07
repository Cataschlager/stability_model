"""Model orchestrator — runs PCA → coupling → calibrate → dynamics.

Usage: python -m model.run_all
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    clean_dir = PROJECT_ROOT / config["paths"]["clean_data"]
    output_dir = PROJECT_ROOT / config["paths"]["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SPECTRAL MODEL — FULL PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Load indicator matrix ────────────────────────────────────
    logger.info("\n--- Step 1: Load Indicator Matrix ---")
    matrix_path = clean_dir / "indicator_matrix.parquet"
    if not matrix_path.exists():
        logger.error("Indicator matrix not found. Run 'make features' first.")
        sys.exit(1)

    indicator_df = pd.read_parquet(matrix_path)
    countries = indicator_df.index.tolist()
    X = indicator_df.values.astype(float)
    n, k = X.shape
    logger.info("Loaded indicator matrix: %d countries × %d indicators", n, k)

    # ── Step 2: PCA ──────────────────────────────────────────────────────
    logger.info("\n--- Step 2: PCA Decomposition ---")
    from model.pca import compute_pca, apply_varimax, compute_composite_scores, sanity_check_fsi

    pca_results = compute_pca(X)
    n_comp = pca_results["n_components_retained"]

    # Varimax rotation
    loadings = pca_results["eigenvectors"][:, :n_comp] * np.sqrt(pca_results["eigenvalues"][:n_comp])
    rotated_loadings = apply_varimax(loadings)

    # Composite scores
    composite = compute_composite_scores(
        pca_results["X_clean"],
        pca_results["eigenvalues"],
        pca_results["eigenvectors"],
        n_comp,
    )

    # Save PCA results
    scores_df = pd.DataFrame({
        "iso3": countries,
        "composite_score": composite,
    }).sort_values("composite_score", ascending=False).reset_index(drop=True)
    scores_df["rank"] = range(1, len(scores_df) + 1)
    scores_df.to_parquet(output_dir / "composite_scores.parquet", index=False)
    logger.info("Top 10 by composite score:\n%s", scores_df.head(10).to_string(index=False))

    # FSI sanity check (if FSI data available)
    fsi_path = clean_dir / "fsi.parquet"
    if fsi_path.exists():
        fsi_df = pd.read_parquet(fsi_path)
        fsi_total = fsi_df[fsi_df["indicator"] == "fsi_total"]
        if not fsi_total.empty:
            fsi_scores_aligned = np.full(n, np.nan)
            for i, c in enumerate(countries):
                match = fsi_total[fsi_total["iso3"] == c]
                if not match.empty:
                    fsi_scores_aligned[i] = match["value"].iloc[-1]
            sanity = sanity_check_fsi(composite, fsi_scores_aligned)
            pd.DataFrame([sanity]).to_parquet(output_dir / "fsi_sanity_check.parquet", index=False)

    # Save eigenvalues and loadings
    np.save(output_dir / "eigenvalues.npy", pca_results["eigenvalues"])
    np.save(output_dir / "eigenvectors.npy", pca_results["eigenvectors"])
    np.save(output_dir / "rotated_loadings.npy", rotated_loadings)

    # ── Step 3: Coupling Matrix ──────────────────────────────────────────
    logger.info("\n--- Step 3: Coupling Matrix ---")
    from model.coupling import (
        build_trade_matrix, build_financial_matrix,
        combine_coupling_matrices, combine_raw_matrices,
        spectral_analysis, row_normalize,
    )
    from model.geodata import build_geographic_coupling, build_gravity_trade_matrix, build_gdp_correlation_matrix
    from model.political_blocs import build_political_coupling

    coupling_weights = config["coupling"]["weights"]

    # Load dyadic data
    sub_matrices = {}

    # ── Trade: prefer DOTS data, fall back to gravity model ──
    dots_path = clean_dir / "imf_dots.parquet"
    trade_df = pd.read_parquet(dots_path) if dots_path.exists() else pd.DataFrame()
    if not trade_df.empty:
        sub_matrices["trade"] = build_trade_matrix(trade_df, countries)
        logger.info("Trade channel: using IMF DOTS bilateral data.")
    else:
        # Gravity model fallback using GDP data
        logger.info("No DOTS data — using gravity model (GDP × distance^-β).")
        gdp_map = {}
        countries_csv = PROJECT_ROOT / "data" / "countries.csv"
        if countries_csv.exists():
            _cdf = pd.read_csv(countries_csv)
            for _, row in _cdf.iterrows():
                gdp_map[row["iso3"]] = row.get("gdp_nominal_usd", row.get("value", 1e10))
        # Supplement from WDI if available
        wdi_path = clean_dir / "worldbank_wdi.parquet"
        if wdi_path.exists():
            wdi_df = pd.read_parquet(wdi_path)
            gdp_rows = wdi_df[wdi_df["indicator"].str.contains("GDP|gdp", case=False, na=False)]
            for _, row in gdp_rows.iterrows():
                if row["iso3"] not in gdp_map and pd.notna(row["value"]):
                    gdp_map[row["iso3"]] = abs(row["value"])
        sub_matrices["trade"] = build_gravity_trade_matrix(gdp_map, countries, beta=1.5)

    # ── Geographic: always use embedded centroids (no external data needed) ──
    sub_matrices["geographic"] = build_geographic_coupling(
        countries, beta=1.5, contiguity_threshold_km=100.0
    )
    logger.info("Geographic channel: embedded Haversine distances.")

    # ── Financial: prefer BIS, fall back to GDP correlation ──
    bis_path = clean_dir / "bis_banking.parquet"
    bis_df = pd.read_parquet(bis_path) if bis_path.exists() else pd.DataFrame()
    if not bis_df.empty:
        countries_csv = PROJECT_ROOT / "data" / "countries.csv"
        gdp_df = pd.read_csv(countries_csv) if countries_csv.exists() else pd.DataFrame()
        from model.coupling import build_financial_matrix
        sub_matrices["financial"] = build_financial_matrix(bis_df, gdp_df, countries)
        logger.info("Financial channel: using BIS banking data.")
    else:
        # GDP correlation proxy
        wdi_path = clean_dir / "worldbank_wdi.parquet"
        if wdi_path.exists():
            wdi_df = pd.read_parquet(wdi_path)
            gdp_growth = wdi_df[wdi_df["indicator"].isin([
                "NY.GDP.MKTP.KD.ZG", "gdp_growth", "GDP_growth"
            ])]
            if not gdp_growth.empty:
                sub_matrices["financial"] = build_gdp_correlation_matrix(gdp_growth, countries)
                logger.info("Financial channel: GDP growth correlation proxy.")
            else:
                sub_matrices["financial"] = np.zeros((n, n))
                logger.warning("No GDP growth data — financial channel zeroed.")
        else:
            sub_matrices["financial"] = np.zeros((n, n))
            logger.warning("No WDI data — financial channel zeroed.")

    # ── Political: always use embedded IGO co-membership ──
    sub_matrices["political"] = build_political_coupling(countries)
    logger.info("Political channel: IGO co-membership (30 organizations).")

    # ── Migration: if available, else redistribute weight ──
    migrant_path = clean_dir / "un_migrant.parquet"
    migrant_df = pd.read_parquet(migrant_path) if migrant_path.exists() else pd.DataFrame()
    if not migrant_df.empty:
        from model.coupling import build_migration_matrix
        countries_csv = PROJECT_ROOT / "data" / "countries.csv"
        pop_df = pd.read_csv(countries_csv) if countries_csv.exists() else pd.DataFrame()
        sub_matrices["migration"] = build_migration_matrix(migrant_df, pop_df, countries)
    else:
        logger.info("No migration data — redistributing weight to trade and geographic.")
        # Redistribute migration weight (0.15) → trade (+0.08), geographic (+0.07)
        coupling_weights = dict(coupling_weights)  # Copy
        mig_weight = coupling_weights.pop("migration", 0.15)
        coupling_weights["trade"] = coupling_weights.get("trade", 0.30) + mig_weight * 0.53
        coupling_weights["geographic"] = coupling_weights.get("geographic", 0.15) + mig_weight * 0.47

    # Combine
    W = combine_coupling_matrices(sub_matrices, coupling_weights)
    W_raw = combine_raw_matrices(sub_matrices, coupling_weights)
    np.save(output_dir / "coupling_matrix.npy", W)
    np.save(output_dir / "raw_coupling_matrix.npy", W_raw)

    # Verify non-uniformity on W_raw
    ev_test = np.abs(np.linalg.eigvals(W_raw))
    ev_test.sort()
    centrality_test = np.abs(np.linalg.eig(W_raw)[1][:, np.argmax(ev_test)].real)
    centrality_test /= centrality_test.sum()
    cent_ratio = centrality_test.max() / max(centrality_test.min(), 1e-10)
    logger.info("Coupling matrix quality: raw centrality ratio (max/min) = %.1f "
                 "(should be >> 1 for non-trivial structure)", cent_ratio)
    if cent_ratio < 2.0:
        logger.warning("⚠️ Coupling matrix may be near-uniform. Check data sources.")

    # Spectral analysis of W
    spectrum = spectral_analysis(W, countries, W_raw=W_raw)
    np.save(output_dir / "W_eigenvalues.npy", spectrum["eigenvalues"])
    np.save(output_dir / "eigenvector_centrality_out.npy", spectrum["eigenvector_centrality_out"])
    np.save(output_dir / "eigenvector_centrality_in.npy", spectrum["eigenvector_centrality_in"])

    if "centrality_out_df" in spectrum:
        spectrum["centrality_out_df"].to_parquet(output_dir / "centrality_out.parquet", index=False)
        spectrum["centrality_in_df"].to_parquet(output_dir / "centrality_in.parquet", index=False)
        logger.info("Top 10 systemic transmitters:\n%s",
                      spectrum["centrality_out_df"].head(10).to_string(index=False))

    # ── Step 3b: Laplacian Analysis ─────────────────────────────────────
    logger.info("\n--- Step 3b: Laplacian & Community Detection ---")
    from model.laplacian import fiedler_analysis, spectral_clustering, network_resilience

    fiedler = fiedler_analysis(W, countries)
    pd.DataFrame({
        "iso3": countries,
        "fiedler_vector": fiedler["fiedler_vector"],
    }).to_parquet(output_dir / "fiedler_vector.parquet", index=False)

    clustering = spectral_clustering(W, countries=countries)
    pd.DataFrame({
        "iso3": countries,
        "cluster": clustering["labels"],
    }).to_parquet(output_dir / "communities.parquet", index=False)

    # Save community membership for each cluster
    for cid, members in clustering["communities"].items():
        logger.info("Community %d (%d members): %s...",
                      cid, len(members), members[:8])

    # Resilience metrics
    resilience = network_resilience(W, countries)
    if "pagerank_df" in resilience:
        resilience["pagerank_df"].to_parquet(output_dir / "pagerank.parquet", index=False)
        logger.info("Top 10 PageRank:\n%s",
                      resilience["pagerank_df"].head(10).to_string(index=False))

    np.save(output_dir / "laplacian_eigenvalues.npy", fiedler["all_eigenvalues"])

    # ── Step 3c: Alpha Calibration ───────────────────────────────────────
    logger.info("\n--- Step 3c: Alpha Calibration ---")
    from model.calibrate import calibrate_alpha, bootstrap_alpha
    
    panel_path = clean_dir / "indicator_panel.parquet"
    if panel_path.exists():
        # Build historical scores panel
        from validation.run_all import build_historical_scores
        panel_df = pd.read_parquet(panel_path)
        indicator_matrix_df = pd.read_parquet(clean_dir / "indicator_matrix.parquet")
        
        scores_panel_df = build_historical_scores(
            panel_df, indicator_matrix_df, 
            pca_results["eigenvalues"], pca_results["eigenvectors"], n_comp
        )
        
        # Format as dicts for calibrate_alpha
        # W is static, so W_panel has same W for all years
        years = scores_panel_df["year"].unique()
        composite_scores_panel = {}
        W_panel = {}
        for y in years:
            df_y = scores_panel_df[scores_panel_df["year"] == y]
            # Ensure correct country order
            df_y = df_y.set_index("iso3").reindex(countries)
            composite_scores_panel[y] = df_y["composite_score"].values
            W_panel[y] = W
            
        cal_results = calibrate_alpha(
            composite_scores_panel, W_panel,
            grid_min=config["dynamics"]["alpha_calibration"]["grid_min"],
            grid_max=config["dynamics"]["alpha_calibration"]["grid_max"],
            grid_step=config["dynamics"]["alpha_calibration"]["grid_step"]
        )
        alpha = cal_results["best_alpha"]
        
        # Save metrics for LaTeX injection
        metrics = {
            "OptimalAlpha": f"{alpha:.3f}",
            "AlphaRMSE": f"{cal_results.get('best_rmse', 0.0):.4f}"
        }
        metrics_file = PROJECT_ROOT / "paper_draft" / "tables" / "metrics.tex"
        if metrics_file.parent.exists():
            with open(metrics_file, "w") as f:
                for key, v in metrics.items():
                    f.write(f"\\newcommand{{\\{key}}}{{{v}}}\n")
                    
        # Log to Research Log
        log_path = PROJECT_ROOT / "RESEARCH_LOG.md"
        if log_path.exists():
            with open(log_path, "a") as f:
                f.write(f"\n- **Auto-run (Calibration):** Optimal $\\alpha$ = {alpha:.3f} (RMSE = {cal_results.get('best_rmse', 0.0):.4f})\n")
    else:
        logger.warning("No indicator panel found. Using default alpha.")
        alpha = config["dynamics"]["alpha_default"]

    # ── Step 3d: Bootstrap Uncertainty Quantification ────────────────────
    logger.info("\n--- Step 3d: Bootstrap Uncertainty Quantification ---")
    
    # 1. PCA Bootstrap
    from model.pca import bootstrap_pca
    logger.info("Running PCA bootstrap...")
    pca_boot = bootstrap_pca(indicator_matrix_df.values, B=500, n_components=n_comp)
    np.save(output_dir / "pca_eigenvalues_ci.npy", pca_boot["eigenvalues_ci"])
    
    composite_ci_df = pd.DataFrame({
        "iso3": countries,
        "composite_score": composite,
        "ci_lower": pca_boot["scores_ci"][:, 0],
        "ci_upper": pca_boot["scores_ci"][:, 1]
    })
    composite_ci_df.to_parquet(output_dir / "composite_scores_ci.parquet", index=False)
    
    # 2. Alpha Bootstrap
    if panel_path.exists():
        logger.info("Running α bootstrap...")
        alpha_boot = bootstrap_alpha(
            composite_scores_panel, W_panel, B=100,
            grid_min=config["dynamics"]["alpha_calibration"]["grid_min"],
            grid_max=config["dynamics"]["alpha_calibration"]["grid_max"],
            grid_step=config["dynamics"]["alpha_calibration"]["grid_step"]
        )
        np.save(output_dir / "alpha_ci.npy", alpha_boot["alpha_ci"])
        
    # 3. Dirichlet Coupling Weight Perturbation
    logger.info("Running Dirichlet perturbation of coupling weights...")
    B_dir = min(config["uncertainty"].get("bootstrap_iterations", 500), 200)  # cap at 200 for speed
    concentration = config["uncertainty"].get("dirichlet_concentration", 50)
    
    base_keys = list(sub_matrices.keys())
    base_probs = np.array([coupling_weights.get(k, 0.0) for k in base_keys])
    base_probs = base_probs / base_probs.sum()
    alpha_dirichlet = base_probs * concentration
    
    rng = np.random.RandomState(config.get("random_seed", 42))
    perturbed_pageranks = []
    
    from model.laplacian import compute_pagerank
    from model.coupling import combine_coupling_matrices
    for _ in range(B_dir):
        w_sample = rng.dirichlet(alpha_dirichlet)
        weights_sample = dict(zip(base_keys, w_sample))
        W_sample = combine_coupling_matrices(sub_matrices, weights_sample)
        p_res = compute_pagerank(W_sample)
        perturbed_pageranks.append(p_res)
        
    perturbed_pr = np.array(perturbed_pageranks)
    pr_ci = np.percentile(perturbed_pr, [5, 95], axis=0)
    
    pr_ci_df = pd.DataFrame({
        "iso3": countries,
        "pagerank_base": resilience["pagerank"],
        "pagerank_ci_lower": pr_ci[0, :],
        "pagerank_ci_upper": pr_ci[1, :]
    })
    pr_ci_df.to_parquet(output_dir / "pagerank_ci.parquet", index=False)

    # ── Step 4: Dynamics ─────────────────────────────────────────────────
    logger.info("\n--- Step 4: Dynamic Model ---")
    from model.dynamics import check_stability, steady_state as compute_ss

    stability = check_stability(W, alpha)

    if stability["stable"]:
        ss = compute_ss(W, composite, alpha)
        ss_df = pd.DataFrame({
            "iso3": countries,
            "structural_score": composite,
            "steady_state_score": ss,
            "network_amplification": ss - composite,
        }).sort_values("steady_state_score", ascending=False).reset_index(drop=True)
        ss_df["rank"] = range(1, len(ss_df) + 1)
        ss_df.to_parquet(output_dir / "steady_state.parquet", index=False)
        logger.info("Top 10 by steady-state instability:\n%s", ss_df.head(10).to_string(index=False))

        # ── Step 4b: Systemic Risk Scoring ───────────────────────────────
        logger.info("\n--- Step 4b: Systemic Risk & Cascade Analysis ---")
        from model.cascade import systemic_risk_score

        risk = systemic_risk_score(W, composite, alpha, shock_magnitude=2.0)
        risk_df = pd.DataFrame({
            "iso3": countries,
            "systemic_risk": risk,
        }).sort_values("systemic_risk", ascending=False).reset_index(drop=True)
        risk_df.to_parquet(output_dir / "systemic_risk.parquet", index=False)
        logger.info("Top 10 systemic risk:\n%s", risk_df.head(10).to_string(index=False))
    else:
        logger.error("System is UNSTABLE with default α=%.3f. Reduce α.", alpha)

    # Save model metadata
    metadata = {
        "n_countries": n,
        "n_indicators": k,
        "n_components": n_comp,
        "alpha": alpha,
        "spectral_radius_W": float(spectrum["spectral_radius"]),
        "spectral_gap": float(spectrum["spectral_gap"]),
        "algebraic_connectivity": float(fiedler["algebraic_connectivity"]),
        "diffusion_timescale": float(fiedler["diffusion_timescale"]),
        "n_communities": int(clustering["k"]),
        "modularity": float(clustering["modularity"]),
        "cumulative_variance": float(pca_results["cumulative_variance"][n_comp - 1]),
        "stability": stability["stable"],
        "centrality_ratio": float(cent_ratio),
    }
    pd.DataFrame([metadata]).to_parquet(output_dir / "model_metadata.parquet", index=False)

    # Save countries list for cross-reference
    pd.DataFrame({"iso3": countries}).to_csv(output_dir / "countries_order.csv", index=False)

    logger.info("\n" + "=" * 60)
    logger.info("✅ SPECTRAL MODEL COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
