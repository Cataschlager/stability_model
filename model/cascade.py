"""Cascade and contagion analysis for instability propagation.

Implements domino-effect simulation, contagion threshold analysis,
and systemic risk scoring.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cascade_analysis(W: np.ndarray, s: np.ndarray, alpha: float,
                     shock_countries: list[int], shock_magnitude: float,
                     threshold: float | None = None) -> dict:
    """Analyze contagion cascade from shock to one or more countries.

    1. Compute baseline steady state
    2. Apply shock(s) to structural scores
    3. Compute new steady state
    4. Identify countries that cross threshold
    5. Measure cascade depth and breadth

    Args:
        W: (N, N) coupling matrix
        s: (N,) structural instability scores
        alpha: Coupling strength
        shock_countries: List of country indices to shock
        shock_magnitude: Shock in standard deviations
        threshold: Score above which a country is "in crisis".
                   Default: 90th percentile of baseline.

    Returns:
        dict with baseline_ss, shocked_ss, delta, cascaded_countries,
        cascade_depth, cascade_breadth, amplification_ratio
    """
    n = len(s)
    I = np.eye(n)

    # Baseline steady state: x* = (I - αW)^{-1} (1-α) s
    leontief = np.linalg.inv(I - alpha * W)
    baseline_ss = leontief @ ((1 - alpha) * s)

    # Apply shock
    s_shocked = s.copy()
    for idx in shock_countries:
        s_shocked[idx] += shock_magnitude

    # Shocked steady state
    shocked_ss = leontief @ ((1 - alpha) * s_shocked)
    delta = shocked_ss - baseline_ss

    # Threshold
    if threshold is None:
        threshold = np.percentile(baseline_ss, 90)

    # Find cascaded countries (crossed threshold due to shock, weren't above before)
    newly_crossed = []
    for i in range(n):
        if i not in shock_countries:
            if shocked_ss[i] > threshold and baseline_ss[i] <= threshold:
                newly_crossed.append(i)

    # Cascade breadth: fraction of non-shocked countries affected
    n_non_shocked = n - len(shock_countries)
    cascade_breadth = len(newly_crossed) / max(n_non_shocked, 1)

    # Cascade depth: max hops from shocked country to affected (via W)
    cascade_depth = 0
    if newly_crossed:
        # Use BFS on thresholded W to find shortest paths
        from collections import deque
        W_binary = (W > np.median(W[W > 0])).astype(int) if (W > 0).any() else np.zeros_like(W, dtype=int)
        for target in newly_crossed:
            for source in shock_countries:
                dist = _bfs_distance(W_binary, source, target)
                if dist > cascade_depth:
                    cascade_depth = dist

    # Amplification ratio
    max_delta_non_shocked = max(
        (abs(delta[i]) for i in range(n) if i not in shock_countries),
        default=0
    )
    amplification = max_delta_non_shocked / max(shock_magnitude, 1e-10)

    logger.info("Cascade: %d countries shocked (δ=%.2f), %d newly crossed threshold, "
                 "breadth=%.2f, depth=%d, amplification=%.3f",
                 len(shock_countries), shock_magnitude, len(newly_crossed),
                 cascade_breadth, cascade_depth, amplification)

    return {
        "baseline_ss": baseline_ss,
        "shocked_ss": shocked_ss,
        "delta": delta,
        "threshold": threshold,
        "newly_crossed": newly_crossed,
        "cascade_breadth": cascade_breadth,
        "cascade_depth": cascade_depth,
        "amplification_ratio": amplification,
    }


def _bfs_distance(A: np.ndarray, source: int, target: int) -> int:
    """BFS shortest path distance in adjacency matrix."""
    from collections import deque
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[source] = True
    queue = deque([(source, 0)])

    while queue:
        node, dist = queue.popleft()
        if node == target:
            return dist
        for neighbor in range(n):
            if A[node, neighbor] > 0 and not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, dist + 1))

    return n  # Unreachable


def contagion_threshold(W: np.ndarray, s: np.ndarray, alpha: float,
                         country_idx: int, threshold: float | None = None,
                         n_magnitudes: int = 50,
                         max_magnitude: float = 5.0) -> dict:
    """Find minimum shock to trigger cascade from a country.

    Sweeps shock magnitude from 0 to max_magnitude and records how many
    countries cross the instability threshold at each level.

    Args:
        W: (N, N) coupling matrix
        s: (N,) structural scores
        alpha: Coupling strength
        country_idx: Country to shock
        threshold: Crisis threshold. Default: 90th percentile.
        n_magnitudes: Number of magnitude steps to test
        max_magnitude: Maximum shock magnitude (σ)

    Returns:
        dict with critical_magnitude, first_affected, magnitude_curve
    """
    n = len(s)
    I = np.eye(n)
    leontief = np.linalg.inv(I - alpha * W)
    baseline_ss = leontief @ ((1 - alpha) * s)

    if threshold is None:
        threshold = np.percentile(baseline_ss, 90)

    magnitudes = np.linspace(0, max_magnitude, n_magnitudes)
    curve = []
    critical_magnitude = None
    first_affected = None

    for mag in magnitudes:
        s_shocked = s.copy()
        s_shocked[country_idx] += mag
        shocked_ss = leontief @ ((1 - alpha) * s_shocked)

        # Count newly crossed
        n_affected = 0
        first = None
        for i in range(n):
            if i != country_idx and shocked_ss[i] > threshold and baseline_ss[i] <= threshold:
                n_affected += 1
                if first is None:
                    first = i

        curve.append((float(mag), n_affected))

        if n_affected > 0 and critical_magnitude is None:
            critical_magnitude = float(mag)
            first_affected = first

    logger.info("Contagion threshold for country %d: critical_magnitude=%.2f, "
                 "first_affected=%s",
                 country_idx,
                 critical_magnitude if critical_magnitude else float("inf"),
                 first_affected)

    return {
        "critical_magnitude": critical_magnitude if critical_magnitude else float("inf"),
        "first_affected": first_affected,
        "magnitude_curve": curve,
        "threshold": threshold,
    }


def systemic_risk_score(W: np.ndarray, s: np.ndarray, alpha: float,
                         shock_magnitude: float = 2.0,
                         threshold: float | None = None) -> np.ndarray:
    """Compute systemic risk for each country.

    For each country i, compute how many other countries would be
    pushed above the threshold by a shock of given magnitude.

    Args:
        W: (N, N) coupling matrix
        s: (N,) structural scores
        alpha: Coupling strength
        shock_magnitude: Size of shock to apply (σ)
        threshold: Crisis threshold

    Returns:
        (N,) array of systemic risk scores (0 to N-1)
    """
    n = len(s)
    I = np.eye(n)
    leontief = np.linalg.inv(I - alpha * W)
    baseline_ss = leontief @ ((1 - alpha) * s)

    if threshold is None:
        threshold = np.percentile(baseline_ss, 90)

    risk = np.zeros(n)
    for i in range(n):
        s_shocked = s.copy()
        s_shocked[i] += shock_magnitude
        shocked_ss = leontief @ ((1 - alpha) * s_shocked)

        for j in range(n):
            if j != i and shocked_ss[j] > threshold and baseline_ss[j] <= threshold:
                risk[i] += 1

    logger.info("Systemic risk: max=%d countries, mean=%.1f, "
                 "%d countries with risk>0",
                 int(risk.max()), risk.mean(), int((risk > 0).sum()))
    return risk


def domino_sequence(W: np.ndarray, s: np.ndarray, alpha: float,
                     shock_country: int, shock_magnitude: float,
                     threshold: float | None = None,
                     max_steps: int = 20) -> list[dict]:
    """Simulate step-by-step domino effect.

    At each step, any country crossing the threshold also becomes a
    source of instability, amplifying its effect on neighbors.

    This is different from the linear steady-state calculation — it
    models a nonlinear chain reaction.

    Args:
        W: (N, N) coupling matrix
        s: (N,) structural scores
        alpha: Coupling strength
        shock_country: Initial country index to shock
        shock_magnitude: Initial shock size (σ)
        threshold: Crisis threshold
        max_steps: Maximum domino steps

    Returns:
        Ordered list of {step, country_idx, score, triggered_by}
    """
    n = len(s)
    I = np.eye(n)
    leontief = np.linalg.inv(I - alpha * W)
    baseline_ss = leontief @ ((1 - alpha) * s)

    if threshold is None:
        threshold = np.percentile(baseline_ss, 90)

    # Track which countries have "fallen"
    fallen = set()
    sequence = []

    # Step 0: initial shock
    s_current = s.copy()
    s_current[shock_country] += shock_magnitude

    current_ss = leontief @ ((1 - alpha) * s_current)

    if current_ss[shock_country] > threshold:
        fallen.add(shock_country)
        sequence.append({
            "step": 0,
            "country_idx": shock_country,
            "score": float(current_ss[shock_country]),
            "triggered_by": None,
        })

    for step in range(1, max_steps + 1):
        new_fallen = []
        for i in range(n):
            if i not in fallen and current_ss[i] > threshold:
                # This country crossed threshold — add amplification
                s_current[i] += 0.5 * shock_magnitude  # Secondary shock
                new_fallen.append(i)

        if not new_fallen:
            break  # No more dominoes

        for idx in new_fallen:
            fallen.add(idx)
            # Find which fallen neighbor most influenced this country
            trigger = None
            max_influence = 0
            for f in fallen:
                if f != idx and W[idx, f] > max_influence:
                    max_influence = W[idx, f]
                    trigger = f
            sequence.append({
                "step": step,
                "country_idx": idx,
                "score": float(current_ss[idx]),
                "triggered_by": trigger,
            })

        # Recompute steady state with new shocks
        current_ss = leontief @ ((1 - alpha) * s_current)

    logger.info("Domino sequence: %d countries fell in %d steps",
                 len(sequence), max(seq_item["step"] for seq_item in sequence) + 1 if sequence else 0)
    return sequence
