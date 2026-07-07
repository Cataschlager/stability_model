"""Coupling matrix construction and spectral analysis.

Builds the (125×125) asymmetric coupling matrix W from 5 channels
(trade, financial, geographic, political, migration) and performs
eigendecomposition per METHODOLOGY.md §5.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_trade_matrix(trade_df: pd.DataFrame, countries: list[str]) -> np.ndarray:
    """Build trade dependency sub-matrix.

    W_trade[i,j] = (exports_ij + imports_ij) / total_trade_i
    Asymmetric. Diagonal = 0.

    Args:
        trade_df: DataFrame with columns iso3_i, iso3_j, value (bilateral trade USD)
        countries: Ordered list of ISO-3 codes (defines matrix ordering)

    Returns:
        (N, N) matrix
    """
    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}
    W = np.zeros((n, n))

    for _, row in trade_df.iterrows():
        i = idx.get(row.get("iso3_i"))
        j = idx.get(row.get("iso3_j"))
        if i is not None and j is not None and i != j:
            W[i, j] += row["value"]

    # Normalize by total trade per country
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums

    np.fill_diagonal(W, 0)
    logger.info("Trade matrix: %d×%d, density=%.3f", n, n, (W > 0).mean())
    return W


def build_financial_matrix(bis_df: pd.DataFrame, gdp_df: pd.DataFrame,
                            countries: list[str]) -> np.ndarray:
    """Build financial exposure sub-matrix.

    W_fin[i,j] = (claims_ij + liabilities_ij) / GDP_i
    Uses mirror data where available.

    Args:
        bis_df: DataFrame with iso3_i, iso3_j, value
        gdp_df: DataFrame with iso3, gdp_nominal_usd
        countries: Ordered country list
    """
    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}
    W = np.zeros((n, n))

    gdp_map = {}
    if not gdp_df.empty:
        for _, row in gdp_df.iterrows():
            gdp_map[row["iso3"]] = row.get("gdp_nominal_usd", 1e12)  # Default high GDP to avoid division issues

    for _, row in bis_df.iterrows():
        i = idx.get(row.get("iso3_i"))
        j = idx.get(row.get("iso3_j"))
        if i is not None and j is not None and i != j:
            W[i, j] += row["value"]
            # Mirror: also record reverse direction
            W[j, i] += row["value"] * 0.5  # Discount mirror data

    # Normalize by GDP
    for i_idx, country in enumerate(countries):
        gdp = gdp_map.get(country, 1e12)
        if gdp > 0:
            W[i_idx, :] /= gdp

    np.fill_diagonal(W, 0)
    logger.info("Financial matrix: %d×%d, density=%.3f", n, n, (W > 0).mean())
    return W


def build_geographic_matrix(geo_df: pd.DataFrame, countries: list[str],
                             contiguity_bonus: float = 0.5) -> np.ndarray:
    """Build geographic proximity sub-matrix.

    W_geo[i,j] = 1/d_ij + β·contiguity_ij

    Args:
        geo_df: DataFrame with iso3_i, iso3_j, indicator, value
        countries: Ordered country list
        contiguity_bonus: β parameter
    """
    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}
    W = np.zeros((n, n))

    # Parse distance and contiguity from geo_df
    dist_df = geo_df[geo_df["indicator"] == "distance_km"]
    contig_df = geo_df[geo_df["indicator"] == "contiguity"]

    for _, row in dist_df.iterrows():
        i = idx.get(row.get("iso3_i"))
        j = idx.get(row.get("iso3_j"))
        if i is not None and j is not None and i != j and row["value"] > 0:
            W[i, j] += 1.0 / row["value"]

    for _, row in contig_df.iterrows():
        i = idx.get(row.get("iso3_i"))
        j = idx.get(row.get("iso3_j"))
        if i is not None and j is not None and i != j and row["value"] == 1:
            W[i, j] += contiguity_bonus

    np.fill_diagonal(W, 0)
    logger.info("Geographic matrix: %d×%d, density=%.3f", n, n, (W > 0).mean())
    return W


def build_political_matrix(alliance_df: pd.DataFrame, countries: list[str],
                            alliance_weights: dict | None = None,
                            mid_weight: float = 1.0) -> np.ndarray:
    """Build political ties sub-matrix from alliances and MIDs.

    Args:
        alliance_df: DataFrame with iso3_i, iso3_j, indicator, value
        countries: Ordered country list
        alliance_weights: Dict mapping alliance type → weight
        mid_weight: Weight for active MIDs
    """
    if alliance_weights is None:
        alliance_weights = {"alliance_defense": 1.0, "alliance_neutrality": 0.5, "alliance_entente": 0.3}

    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}
    W = np.zeros((n, n))

    for _, row in alliance_df.iterrows():
        i = idx.get(row.get("iso3_i"))
        j = idx.get(row.get("iso3_j"))
        ind = row.get("indicator", "")
        if i is not None and j is not None and i != j:
            weight = alliance_weights.get(ind, mid_weight if "mid" in ind.lower() else 0.5)
            W[i, j] += weight * row.get("value", 1.0)
            W[j, i] += weight * row.get("value", 1.0)  # Symmetric

    np.fill_diagonal(W, 0)
    logger.info("Political matrix: %d×%d, density=%.3f", n, n, (W > 0).mean())
    return W


def build_migration_matrix(migrant_df: pd.DataFrame, pop_df: pd.DataFrame,
                            countries: list[str]) -> np.ndarray:
    """Build migration linkage sub-matrix.

    W_mig[i,j] = migrants_from_j_in_i / population_i

    Args:
        migrant_df: DataFrame with iso3_i (destination), iso3_j (origin), value
        pop_df: DataFrame with iso3, population
        countries: Ordered country list
    """
    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}
    W = np.zeros((n, n))

    pop_map = {}
    if not pop_df.empty:
        for _, row in pop_df.iterrows():
            pop_map[row["iso3"]] = row.get("population", row.get("value", 1e7))

    for _, row in migrant_df.iterrows():
        i = idx.get(row.get("iso3_i"))
        j = idx.get(row.get("iso3_j"))
        if i is not None and j is not None and i != j:
            W[i, j] += row.get("value", 0)

    # Normalize by population
    for i_idx, country in enumerate(countries):
        pop = pop_map.get(country, 1e7)
        if pop > 0:
            W[i_idx, :] /= pop

    np.fill_diagonal(W, 0)
    logger.info("Migration matrix: %d×%d, density=%.3f", n, n, (W > 0).mean())
    return W


def row_normalize(W: np.ndarray) -> np.ndarray:
    """Row-normalize matrix so each row sums to 1.

    Handles zero rows by leaving them as zero.
    """
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero; zero rows stay zero
    return W / row_sums


def combine_coupling_matrices(matrices: dict[str, np.ndarray],
                               weights: dict[str, float]) -> np.ndarray:
    """Weighted combination of sub-matrices, then row-normalize.

    Args:
        matrices: Dict mapping channel name → (N, N) sub-matrix
        weights: Dict mapping channel name → weight

    Returns:
        (N, N) row-stochastic coupling matrix W
    """
    # Validate
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        logger.warning("Coupling weights sum to %.3f, not 1.0. Normalizing.", total_weight)
        weights = {k: v / total_weight for k, v in weights.items()}

    # Combine
    n = None
    W = None
    for name, sub_W in matrices.items():
        if name not in weights:
            continue
        w = weights[name]
        # Row-normalize each sub-matrix first
        sub_W_norm = row_normalize(sub_W)
        if W is None:
            n = sub_W.shape[0]
            W = np.zeros((n, n))
        W += w * sub_W_norm

    if W is None:
        raise ValueError("No matrices provided.")

    # Final row-normalization
    W = row_normalize(W)
    np.fill_diagonal(W, 0)

    # Re-normalize after zeroing diagonal
    W = row_normalize(W)

    logger.info("Combined coupling matrix: %d×%d, density=%.3f, sum check: rows sum to %.4f±%.4f",
                 n, n, (W > 0).mean(), W.sum(axis=1).mean(), W.sum(axis=1).std())
    return W


def combine_raw_matrices(matrices: dict[str, np.ndarray],
                         weights: dict[str, float]) -> np.ndarray:
    """Weighted combination of sub-matrices scaled by global sum, preserving heterogeneity.

    Unlike combine_coupling_matrices, this does NOT row-normalize, so countries
    with larger GDP, trade, population, etc. will have larger raw degrees.
    """
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        weights = {k: v / total_weight for k, v in weights.items()}

    n = None
    W_raw = None
    for name, sub_W in matrices.items():
        if name not in weights:
            continue
        w = weights[name]

        # Copy to avoid side effects
        sub_W_copy = sub_W.copy()
        np.fill_diagonal(sub_W_copy, 0)

        # Normalize by its global sum so that all channels are on the same scale
        total_sum = sub_W_copy.sum()
        if total_sum > 0:
            sub_W_norm = sub_W_copy / total_sum
        else:
            sub_W_norm = sub_W_copy

        if W_raw is None:
            n = sub_W.shape[0]
            W_raw = np.zeros((n, n))
        W_raw += w * sub_W_norm

    if W_raw is None:
        raise ValueError("No matrices provided.")

    logger.info("Combined raw adjacency matrix: %d×%d, density=%.3f, sum check=%.4f",
                 n, n, (W_raw > 0).mean(), W_raw.sum())
    return W_raw


def spectral_analysis(W: np.ndarray, countries: list[str] | None = None,
                       W_raw: np.ndarray | None = None) -> dict:
    """Full eigendecomposition of the coupling matrix W.

    **Important mathematical note**: W is row-stochastic (each row sums to 1).
    By the Perron-Frobenius theorem, row-stochastic matrices always have:
      - Leading eigenvalue λ₁ = 1 exactly
      - Leading right eigenvector = uniform (1/n, ..., 1/n)
    Therefore eigenvector centrality on W is ALWAYS uniform and useless.

    Corrections:
      1. PageRank (with damping) gives heterogeneous, meaningful centrality on W
      2. Eigenvector centrality on W_raw (non-normalized adjacency) is meaningful

    Args:
        W: (N, N) row-stochastic coupling matrix
        countries: Optional country labels
        W_raw: Optional non-normalized adjacency matrix (for eigenvector centrality)

    Returns:
        dict with eigenvalues, magnitudes, spectral_radius, spectral_gap,
        pagerank_centrality_out (primary), eigenvector_centrality_out (if W_raw given),
        eigenvector_centrality_in (from W.T PageRank)
    """
    n = W.shape[0]

    # Eigendecomposition of W (for spectral radius and eigenvalue spectrum)
    eigenvalues, right_evecs = np.linalg.eig(W)
    magnitudes = np.abs(eigenvalues)
    idx = np.argsort(magnitudes)[::-1]
    eigenvalues = eigenvalues[idx]
    magnitudes = magnitudes[idx]
    right_evecs = right_evecs[:, idx]

    # Left eigendecomposition
    left_eigenvalues, left_evecs = np.linalg.eig(W.T)
    left_idx = np.argsort(np.abs(left_eigenvalues))[::-1]
    left_evecs = left_evecs[:, left_idx]

    spectral_radius = float(magnitudes[0].real)
    spectral_gap = float((magnitudes[0] - magnitudes[1]).real) if len(magnitudes) > 1 else spectral_radius

    # ── PageRank centrality (primary, heterogeneous) ──
    # PageRank: r = d * W^T * r + (1-d)/n
    # This IS heterogeneous because the damping term breaks the row-stochastic symmetry
    damping = 0.85
    r_out = np.ones(n) / n
    r_in = np.ones(n) / n
    for _ in range(200):
        r_out_new = damping * W.T @ r_out + (1 - damping) / n
        r_out_new /= r_out_new.sum()
        if np.linalg.norm(r_out_new - r_out) < 1e-10:
            break
        r_out = r_out_new

    for _ in range(200):
        r_in_new = damping * W @ r_in + (1 - damping) / n
        r_in_new /= r_in_new.sum()
        if np.linalg.norm(r_in_new - r_in) < 1e-10:
            break
        r_in = r_in_new

    # ── Eigenvector centrality on raw adjacency (if provided) ──
    if W_raw is not None:
        ev_raw, evec_raw = np.linalg.eig(W_raw)
        ev_raw_idx = np.argsort(np.abs(ev_raw))[::-1]
        ev_centrality_out = np.abs(evec_raw[:, ev_raw_idx[0]].real)
        ev_centrality_out /= ev_centrality_out.sum()

        ev_raw_l, evec_raw_l = np.linalg.eig(W_raw.T)
        ev_raw_l_idx = np.argsort(np.abs(ev_raw_l))[::-1]
        ev_centrality_in = np.abs(evec_raw_l[:, ev_raw_l_idx[0]].real)
        ev_centrality_in /= ev_centrality_in.sum()
    else:
        # Fall back to PageRank as both in and out centrality
        ev_centrality_out = r_out.copy()
        ev_centrality_in = r_in.copy()

    logger.info("Spectral analysis: ρ(W)=%.4f, spectral gap=%.4f, "
                 "PageRank ratio=%.1fx, top eigenvalues: %s",
                 spectral_radius, spectral_gap,
                 r_out.max() / max(r_out.min(), 1e-10),
                 [f"{m:.4f}" for m in magnitudes[:5]])

    result = {
        "eigenvalues": eigenvalues,
        "magnitudes": magnitudes,
        "right_eigenvectors": right_evecs,
        "left_eigenvectors": left_evecs,
        "spectral_radius": spectral_radius,
        "spectral_gap": spectral_gap,
        "pagerank_out": r_out,
        "pagerank_in": r_in,
        "eigenvector_centrality_out": ev_centrality_out,
        "eigenvector_centrality_in": ev_centrality_in,
    }

    if countries:
        result["centrality_out_df"] = pd.DataFrame({
            "iso3": countries,
            "centrality_out": ev_centrality_out,
            "pagerank": r_out,
        }).sort_values("centrality_out", ascending=False).reset_index(drop=True)

        result["centrality_in_df"] = pd.DataFrame({
            "iso3": countries,
            "centrality_in": ev_centrality_in,
        }).sort_values("centrality_in", ascending=False).reset_index(drop=True)

    return result


