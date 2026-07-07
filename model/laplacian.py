"""Graph Laplacian analysis: Fiedler vector, spectral clustering, resilience.

Implements spectral graph theory tools for analyzing the coupling network
structure, including community detection and network resilience metrics.
"""

import logging
from collections import deque

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def compute_laplacian(W: np.ndarray) -> dict:
    """Compute combinatorial and normalized Laplacian matrices.

    For directed networks (asymmetric W), we first symmetrize:
        W_sym = (W + W^T) / 2

    Then:
        L = D - W_sym  (combinatorial)
        L_sym = D^{-1/2} L D^{-1/2}  (symmetric normalized)
        L_rw = D^{-1} L  (random walk normalized)

    This ensures the Laplacian is positive semi-definite with a meaningful
    spectrum, even when W is row-stochastic.

    Args:
        W: (N, N) coupling matrix (non-negative, possibly asymmetric)

    Returns:
        dict with L, L_sym, L_rw, D, d (degree vector), W_sym
    """
    n = W.shape[0]

    # Symmetrize: standard approach for directed networks
    W_sym = (W + W.T) / 2.0
    np.fill_diagonal(W_sym, 0)

    d = W_sym.sum(axis=1)  # Degree vector
    D = np.diag(d)
    L = D - W_sym

    # Symmetric normalized: D^{-1/2} L D^{-1/2}
    d_inv_sqrt = np.zeros(n)
    mask = d > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    # Random walk normalized: D^{-1} L = I - D^{-1} W_sym
    d_inv = np.zeros(n)
    d_inv[mask] = 1.0 / d[mask]
    D_inv = np.diag(d_inv)
    L_rw = D_inv @ L

    return {"L": L, "L_sym": L_sym, "L_rw": L_rw, "D": D, "d": d, "W_sym": W_sym}


def fiedler_analysis(W: np.ndarray, countries: list[str] | None = None) -> dict:
    """Compute the Fiedler vector and algebraic connectivity.

    The Fiedler value (λ₂) is the second-smallest eigenvalue of the
    combinatorial Laplacian. It measures how well-connected the network is.
    The corresponding eigenvector (Fiedler vector) naturally partitions
    the network into two halves.

    Args:
        W: (N, N) coupling matrix
        countries: Optional list of ISO-3 codes for labeling

    Returns:
        dict with:
        - algebraic_connectivity: λ₂
        - fiedler_vector: eigenvector for λ₂
        - natural_partition: (group_A, group_B) country lists
        - diffusion_timescale: 1/λ₂
        - all_eigenvalues: sorted Laplacian eigenvalues
    """
    lap = compute_laplacian(W)
    L = lap["L"]

    # Full eigendecomposition of symmetric Laplacian
    # L is real symmetric, so eigh gives real eigenvalues/vectors
    eigenvalues, eigenvectors = eigh(L)

    # Sort (eigh returns ascending, which is what we want)
    # λ₁ ≈ 0, λ₂ = Fiedler value
    fiedler_value = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    fiedler_vec = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(W.shape[0])

    # Natural partition: split by sign of Fiedler vector
    group_a_idx = np.where(fiedler_vec >= 0)[0]
    group_b_idx = np.where(fiedler_vec < 0)[0]

    result = {
        "algebraic_connectivity": fiedler_value,
        "fiedler_vector": fiedler_vec,
        "diffusion_timescale": 1.0 / fiedler_value if fiedler_value > 1e-10 else float("inf"),
        "all_eigenvalues": eigenvalues,
        "group_a_indices": group_a_idx,
        "group_b_indices": group_b_idx,
    }

    if countries:
        result["group_a"] = [countries[i] for i in group_a_idx]
        result["group_b"] = [countries[i] for i in group_b_idx]

    logger.info("Fiedler analysis: λ₂=%.6f, partition=%d/%d, τ_diff=%.1f",
                 fiedler_value, len(group_a_idx), len(group_b_idx),
                 result["diffusion_timescale"])
    return result


def spectral_clustering(W: np.ndarray, k: int | None = None,
                         countries: list[str] | None = None) -> dict:
    """Spectral clustering on the coupling matrix.

    Uses normalized Laplacian eigenvectors + k-means to detect
    communities (instability blocs).

    Args:
        W: (N, N) coupling matrix
        k: Number of clusters. If None, auto-detect via eigengap heuristic.
        countries: Optional list of ISO-3 codes

    Returns:
        dict with k, labels, eigengap, communities, modularity
    """
    n = W.shape[0]
    lap = compute_laplacian(W)
    L_sym = lap["L_sym"]

    # Eigendecomposition of normalized Laplacian
    eigenvalues, eigenvectors = eigh(L_sym)

    # Eigengap heuristic for k selection
    max_k = min(20, n // 5)
    gaps = np.diff(eigenvalues[:max_k + 1])

    if k is None:
        # Find largest gap after λ₁ (skip gap 0 which is λ₁→λ₂)
        k = int(np.argmax(gaps[1:max_k]) + 2)  # +2 because we skip index 0
        k = max(2, min(k, 15))  # Clamp to [2, 15]
        logger.info("Auto-detected k=%d from eigengap heuristic", k)

    # Use first k eigenvectors for embedding
    embedding = eigenvectors[:, :k]

    # Normalize rows (Ng-Jordan-Weiss normalization)
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    embedding = embedding / row_norms

    # K-means clustering
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = kmeans.fit_predict(embedding)

    # Build communities dict
    communities = {}
    for cluster_id in range(k):
        member_indices = np.where(labels == cluster_id)[0]
        if countries:
            communities[cluster_id] = [countries[i] for i in member_indices]
        else:
            communities[cluster_id] = member_indices.tolist()

    # Newman modularity: Q = (1/2m) Σ_ij [W_ij - d_i*d_j/(2m)] δ(c_i, c_j)
    m = W.sum() / 2.0
    if m > 0:
        d = W.sum(axis=1)
        Q = 0.0
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    Q += W[i, j] - d[i] * d[j] / (2 * m)
        Q /= (2 * m)
    else:
        Q = 0.0

    logger.info("Spectral clustering: k=%d, modularity=%.4f, sizes=%s",
                 k, Q, [int((labels == c).sum()) for c in range(k)])

    return {
        "k": k,
        "labels": labels,
        "eigengap": gaps,
        "communities": communities,
        "modularity": Q,
        "embedding": embedding,
        "eigenvalues": eigenvalues[:max_k + 1],
    }


def compute_pagerank(W: np.ndarray, damping: float = 0.85,
                      max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """Compute PageRank of the coupling network.

    Uses power iteration: r(t+1) = d * W^T * r(t) + (1-d)/n * 1

    Args:
        W: (N, N) row-stochastic coupling matrix
        damping: Damping factor (default 0.85)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        (N,) PageRank vector summing to 1
    """
    n = W.shape[0]
    r = np.ones(n) / n  # Uniform initialization

    for iteration in range(max_iter):
        r_new = damping * W.T @ r + (1 - damping) / n
        r_new /= r_new.sum()  # Normalize

        if np.linalg.norm(r_new - r) < tol:
            logger.info("PageRank converged in %d iterations", iteration + 1)
            return r_new
        r = r_new

    logger.warning("PageRank did not converge in %d iterations", max_iter)
    return r


def compute_betweenness_centrality(W: np.ndarray, threshold_quantile: float = 0.9) -> np.ndarray:
    """Approximate betweenness centrality using thresholded adjacency + BFS.

    Thresholds the coupling matrix to keep only the top edges,
    then computes betweenness via BFS shortest paths.

    Args:
        W: (N, N) coupling matrix
        threshold_quantile: Keep edges above this quantile

    Returns:
        (N,) betweenness centrality vector
    """
    n = W.shape[0]

    # Threshold to create unweighted adjacency
    threshold = np.quantile(W[W > 0], threshold_quantile) if (W > 0).any() else 0
    A = (W >= threshold).astype(int)
    np.fill_diagonal(A, 0)

    betweenness = np.zeros(n)

    for s in range(n):
        # BFS from source s
        dist = np.full(n, -1)
        dist[s] = 0
        n_paths = np.zeros(n)
        n_paths[s] = 1
        queue = deque([s])
        order = []

        while queue:
            v = queue.popleft()
            order.append(v)
            for w in range(n):
                if A[v, w] > 0:
                    if dist[w] == -1:  # First visit
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        n_paths[w] += n_paths[v]

        # Accumulate betweenness
        delta = np.zeros(n)
        for v in reversed(order):
            if v == s:
                continue
            for w in range(n):
                if A[w, v] > 0 and dist[w] == dist[v] - 1 and n_paths[v] > 0:
                    delta[w] += (n_paths[w] / n_paths[v]) * (1 + delta[v])
            betweenness[v] += delta[v]

    # Normalize
    betweenness /= max(betweenness.max(), 1e-10)

    logger.info("Betweenness centrality: top nodes=%s",
                 np.argsort(betweenness)[::-1][:5].tolist())
    return betweenness


def network_resilience(W: np.ndarray, countries: list[str] | None = None) -> dict:
    """Comprehensive network resilience metrics.

    Args:
        W: (N, N) coupling matrix
        countries: Optional country labels

    Returns:
        dict with algebraic_connectivity, pagerank, betweenness_centrality,
        effective_resistance, density
    """
    n = W.shape[0]

    fiedler = fiedler_analysis(W, countries)
    pagerank = compute_pagerank(W)
    betweenness = compute_betweenness_centrality(W)

    density = (W > 0).sum() / (n * (n - 1))

    result = {
        "algebraic_connectivity": fiedler["algebraic_connectivity"],
        "diffusion_timescale": fiedler["diffusion_timescale"],
        "pagerank": pagerank,
        "betweenness_centrality": betweenness,
        "density": density,
        "fiedler_vector": fiedler["fiedler_vector"],
    }

    if countries:
        import pandas as pd
        result["pagerank_df"] = pd.DataFrame({
            "iso3": countries,
            "pagerank": pagerank,
            "betweenness": betweenness,
        }).sort_values("pagerank", ascending=False).reset_index(drop=True)

    logger.info("Network resilience: λ₂=%.6f, density=%.3f, τ=%.1f",
                 fiedler["algebraic_connectivity"], density,
                 fiedler["diffusion_timescale"])
    return result
