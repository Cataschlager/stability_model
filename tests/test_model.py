"""Tests for the model core: PCA, coupling, dynamics."""

import numpy as np
import pytest


class TestPCA:
    def test_eigenvalue_sum(self, small_indicator_matrix):
        """Eigenvalues of correlation matrix should sum to K (number of indicators)."""
        from model.pca import compute_pca

        X = small_indicator_matrix.values
        result = compute_pca(X)
        k = X.shape[1]
        assert abs(result["eigenvalues"].sum() - k) < 0.01, \
            f"Eigenvalue sum {result['eigenvalues'].sum()} != {k}"

    def test_orthogonality(self, small_indicator_matrix):
        """Eigenvectors should be orthogonal."""
        from model.pca import compute_pca

        X = small_indicator_matrix.values
        result = compute_pca(X)
        V = result["eigenvectors"]
        product = V.T @ V
        np.testing.assert_array_almost_equal(product, np.eye(V.shape[1]), decimal=5)

    def test_composite_scores_shape(self, small_indicator_matrix):
        """Composite scores should have shape (n_countries,)."""
        from model.pca import compute_pca, compute_composite_scores

        X = small_indicator_matrix.values
        result = compute_pca(X)
        scores = compute_composite_scores(
            result["X_clean"], result["eigenvalues"],
            result["eigenvectors"], result["n_components_retained"]
        )
        assert scores.shape == (X.shape[0],)


class TestCoupling:
    def test_row_stochastic(self, small_coupling_matrix):
        """Each row of W should sum to 1."""
        row_sums = small_coupling_matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10), decimal=5)

    def test_non_negative(self, small_coupling_matrix):
        """All entries of W should be >= 0."""
        assert np.all(small_coupling_matrix >= 0)

    def test_zero_diagonal(self, small_coupling_matrix):
        """Diagonal of W should be 0."""
        np.testing.assert_array_almost_equal(np.diag(small_coupling_matrix), np.zeros(10))

    def test_spectral_radius(self, small_coupling_matrix):
        """Spectral radius of row-stochastic W should be 1."""
        eigenvalues = np.linalg.eigvals(small_coupling_matrix)
        rho = np.max(np.abs(eigenvalues))
        assert abs(rho - 1.0) < 0.01, f"Spectral radius {rho} != 1.0"

    def test_spectral_analysis_output(self, small_coupling_matrix):
        """Spectral analysis should return all expected keys."""
        from model.coupling import spectral_analysis

        result = spectral_analysis(small_coupling_matrix)
        expected_keys = ["eigenvalues", "magnitudes", "spectral_radius",
                          "spectral_gap", "eigenvector_centrality_out",
                          "eigenvector_centrality_in"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestDynamics:
    def test_stability_check_stable(self, small_coupling_matrix):
        """With alpha < 1, system should be stable."""
        from model.dynamics import check_stability
        result = check_stability(small_coupling_matrix, 0.5)
        assert result["stable"] is True

    def test_stability_check_unstable(self, small_coupling_matrix):
        """With alpha > 1, system should be unstable."""
        from model.dynamics import check_stability
        result = check_stability(small_coupling_matrix, 1.5)
        assert result["stable"] is False

    def test_steady_state_satisfies_equation(self, small_coupling_matrix, small_composite_scores):
        """Steady state should satisfy x* = α·W·x* + (1-α)·s."""
        from model.dynamics import steady_state

        alpha = 0.4
        W = small_coupling_matrix
        s = small_composite_scores
        x_star = steady_state(W, s, alpha)

        # Check: x* ≈ α·W·x* + (1-α)·s
        lhs = x_star
        rhs = alpha * W @ x_star + (1 - alpha) * s
        np.testing.assert_array_almost_equal(lhs, rhs, decimal=5)

    def test_simulation_convergence(self, small_coupling_matrix, small_composite_scores):
        """With alpha < 1, simulation should converge to steady state."""
        from model.dynamics import simulate, steady_state

        alpha = 0.4
        W = small_coupling_matrix
        s = small_composite_scores

        ss = steady_state(W, s, alpha)
        trajectory = simulate(W, s, alpha, n_steps=100)

        # Final state should be close to steady state
        np.testing.assert_array_almost_equal(trajectory[-1], ss, decimal=3)

    def test_factor_shock_increases_target(self, small_coupling_matrix, small_composite_scores):
        """A positive factor shock should increase the target country's instability."""
        from model.dynamics import factor_shock

        result = factor_shock(small_coupling_matrix, small_composite_scores,
                               0.4, country_idx=0, delta=2.0)
        assert result["delta_ss"][0] > 0, "Shocked country's instability should increase"

    def test_sensitivity_shape(self, small_coupling_matrix):
        """Sensitivity decomposition should return (N,) vector."""
        from model.dynamics import sensitivity_decomposition

        sens = sensitivity_decomposition(small_coupling_matrix, 0.4, 0)
        assert sens.shape == (10,)
