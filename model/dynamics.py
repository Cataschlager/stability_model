"""Dynamic propagation model - Friedkin-Johnsen with stochastic perturbation.

x(t+1) = α·W·x(t) + (1-α)·s + η(t)

Implements steady-state computation, simulation, stability analysis,
factor shocks, edge shocks, and compound scenarios per METHODOLOGY.md §6-7.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def check_stability(W: np.ndarray, alpha: float) -> dict:
    """Check whether α·ρ(W) < 1 (system is damped).

    Args:
        W: (N, N) coupling matrix
        alpha: Coupling strength

    Returns:
        dict with spectral_radius, effective_radius, stable (bool), margin
    """
    eigenvalues = np.linalg.eigvals(W)
    rho = np.max(np.abs(eigenvalues))
    effective = alpha * rho
    stable = bool(effective < 1.0)
    margin = float(1.0 - effective)

    logger.info("Stability: α=%.3f, ρ(W)=%.4f, α·ρ(W)=%.4f - %s (margin=%.4f)",
                 alpha, rho, effective, "STABLE" if stable else "UNSTABLE", margin)

    return {
        "spectral_radius": rho,
        "effective_radius": effective,
        "stable": stable,
        "margin": margin,
        "alpha": alpha,
    }


def steady_state(W: np.ndarray, s: np.ndarray, alpha: float) -> np.ndarray:
    """Compute the steady-state instability vector.

    x* = (I - α·W)^(-1) · (1-α) · s

    This is the Leontief inverse applied to the structural baseline.

    Args:
        W: (N, N) row-stochastic coupling matrix
        s: (N,) structural baseline (composite scores from PCA)
        alpha: Coupling strength (must be < 1 for convergence)

    Returns:
        (N,) steady-state instability vector
    """
    n = W.shape[0]
    assert alpha < 1.0, f"α must be < 1 for stability. Got α={alpha}"

    I = np.eye(n)
    # (I - αW)^(-1) · (1-α) · s
    leontief = np.linalg.solve(I - alpha * W, (1 - alpha) * s)

    logger.info("Steady state: mean=%.3f, std=%.3f, range=[%.3f, %.3f]",
                 leontief.mean(), leontief.std(), leontief.min(), leontief.max())
    return leontief


def simulate(W: np.ndarray, s: np.ndarray, alpha: float,
             x0: np.ndarray | None = None, n_steps: int = 50,
             eta: np.ndarray | None = None) -> np.ndarray:
    """Run the dynamic propagation model forward.

    x(t+1) = α·W·x(t) + (1-α)·s + η(t)

    Args:
        W: (N, N) coupling matrix
        s: (N,) structural baseline
        alpha: Coupling strength
        x0: (N,) initial state. If None, uses s.
        n_steps: Number of time steps
        eta: (n_steps, N) noise matrix. If None, uses zero noise.

    Returns:
        (n_steps+1, N) trajectory array
    """
    n = W.shape[0]
    if x0 is None:
        x0 = s.copy()
    if eta is None:
        eta = np.zeros((n_steps, n))

    trajectory = np.zeros((n_steps + 1, n))
    trajectory[0] = x0

    for t in range(n_steps):
        trajectory[t + 1] = alpha * W @ trajectory[t] + (1 - alpha) * s + eta[t]

    return trajectory


def factor_shock(W: np.ndarray, s: np.ndarray, alpha: float,
                 country_idx: int, delta: float,
                 n_steps: int = 50) -> dict:
    """Apply a factor shock to a country and compute propagation.

    Increases country_idx's structural baseline by delta standard deviations.

    Args:
        W: Coupling matrix
        s: Structural baseline
        alpha: Coupling strength
        country_idx: Index of country to shock
        delta: Shock magnitude in standard deviations
        n_steps: Simulation steps

    Returns:
        dict with: baseline_ss, shocked_ss, delta_ss, trajectory,
                   top10_partners (by impact), convergence_step
    """
    n = W.shape[0]

    # Baseline steady state
    ss_base = steady_state(W, s, alpha)

    # Shocked baseline
    s_shocked = s.copy()
    s_shocked[country_idx] += delta

    # New steady state
    ss_shocked = steady_state(W, s_shocked, alpha)
    delta_ss = ss_shocked - ss_base

    # Trajectory from baseline to new steady state
    trajectory = simulate(W, s_shocked, alpha, x0=ss_base, n_steps=n_steps)

    # Top 10 most impacted (excluding the shocked country)
    impact = np.abs(delta_ss.copy())
    impact[country_idx] = 0  # Exclude self
    top10_idx = np.argsort(impact)[::-1][:10]

    # Convergence: when max diff from final state < 1% of total delta
    total_delta = np.linalg.norm(delta_ss)
    convergence_step = n_steps
    if total_delta > 1e-10:
        for t in range(n_steps + 1):
            diff = np.linalg.norm(trajectory[t] - ss_shocked)
            if diff < 0.01 * total_delta:
                convergence_step = t
                break

    logger.info("Factor shock: country %d, δ=%.2f, convergence at step %d",
                 country_idx, delta, convergence_step)

    return {
        "baseline_ss": ss_base,
        "shocked_ss": ss_shocked,
        "delta_ss": delta_ss,
        "trajectory": trajectory,
        "top10_partners": top10_idx,
        "convergence_step": convergence_step,
    }


def edge_shock(W: np.ndarray, s: np.ndarray, alpha: float,
               i: int, j: int, multiplier: float = 0.0) -> dict:
    """Modify an edge in the coupling matrix and recompute.

    Args:
        W: Original coupling matrix
        s: Structural baseline
        alpha: Coupling strength
        i, j: Country pair
        multiplier: 0 = sever connection, 2 = double, etc.

    Returns:
        dict with new_W, new_spectrum, new_ss, delta_ss, spectral_comparison
    """
    from model.coupling import row_normalize, spectral_analysis

    # Original spectrum
    orig_spectrum = spectral_analysis(W)
    orig_ss = steady_state(W, s, alpha)

    # Modify edge
    W_new = W.copy()
    W_new[i, j] *= multiplier
    W_new[j, i] *= multiplier

    # Re-normalize
    W_new = row_normalize(W_new)

    # New spectrum
    new_spectrum = spectral_analysis(W_new)
    new_ss = steady_state(W_new, s, alpha)
    delta_ss = new_ss - orig_ss

    logger.info("Edge shock: (%d, %d) × %.2f → Δρ=%.4f",
                 i, j, multiplier,
                 new_spectrum["spectral_radius"] - orig_spectrum["spectral_radius"])

    return {
        "new_W": W_new,
        "new_spectrum": new_spectrum,
        "orig_spectrum": orig_spectrum,
        "new_ss": new_ss,
        "orig_ss": orig_ss,
        "delta_ss": delta_ss,
    }


def compound_scenario(W: np.ndarray, s: np.ndarray, alpha: float,
                       shocks: list[dict]) -> dict:
    """Apply multiple simultaneous shocks.

    Each shock dict has:
      - type: "factor" or "edge"
      - For factor: country_idx, delta
      - For edge: i, j, multiplier

    Returns:
        dict with stability assessment, steady states, trajectories
    """
    from model.coupling import row_normalize

    W_eff = W.copy()
    s_eff = s.copy()

    for shock in shocks:
        if shock["type"] == "factor":
            s_eff[shock["country_idx"]] += shock["delta"]
        elif shock["type"] == "edge":
            W_eff[shock["i"], shock["j"]] *= shock.get("multiplier", 0.0)
            W_eff[shock["j"], shock["i"]] *= shock.get("multiplier", 0.0)

    # Re-normalize if edges were modified
    if any(sh["type"] == "edge" for sh in shocks):
        W_eff = row_normalize(W_eff)

    # Stability check
    stability = check_stability(W_eff, alpha)

    result = {
        "stability": stability,
        "W_effective": W_eff,
        "s_effective": s_eff,
    }

    if stability["stable"]:
        ss = steady_state(W_eff, s_eff, alpha)
        ss_base = steady_state(W, s, alpha)
        trajectory = simulate(W_eff, s_eff, alpha, x0=ss_base, n_steps=50)
        result["steady_state"] = ss
        result["baseline_ss"] = ss_base
        result["delta_ss"] = ss - ss_base
        result["trajectory"] = trajectory
    else:
        # Explosive: report divergence rate
        trajectory = simulate(W_eff, s_eff, alpha, n_steps=20)
        result["trajectory"] = trajectory
        result["divergence_rate"] = stability["effective_radius"] - 1.0
        logger.warning("COMPOUND SCENARIO: System is EXPLOSIVE with α·ρ(W')=%.4f",
                        stability["effective_radius"])

    return result


def sensitivity_decomposition(W: np.ndarray, alpha: float,
                               target_country: int) -> np.ndarray:
    """Extract column i of (I - αW)^(-1) - sensitivity of all countries to target.

    ∂x*_j / ∂s_i for all j given a unit increase in country i's baseline.

    Args:
        W: Coupling matrix
        alpha: Coupling strength
        target_country: Index of the country to analyze

    Returns:
        (N,) vector of sensitivities
    """
    n = W.shape[0]
    I = np.eye(n)
    leontief = np.linalg.inv(I - alpha * W)
    sensitivities = leontief[:, target_country] * (1 - alpha)

    logger.info("Sensitivity decomposition for country %d: "
                 "max sensitivity=%.4f, mean=%.4f",
                 target_country, sensitivities.max(), sensitivities.mean())
    return sensitivities
