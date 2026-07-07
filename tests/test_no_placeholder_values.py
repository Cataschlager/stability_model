"""Tests to ensure no placeholder or synthetic data in outputs."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


@pytest.mark.skipif(not CLEAN_DIR.exists(), reason="No clean data directory")
class TestNoPlaceholders:
    """Enforce that no synthetic or placeholder data exists in outputs."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        matrix_path = CLEAN_DIR / "indicator_matrix.parquet"
        if matrix_path.exists():
            self.matrix = pd.read_parquet(matrix_path)
        else:
            pytest.skip("indicator_matrix.parquet not found")

    def test_no_constant_columns(self):
        """No indicator column should be constant (all same value)."""
        numeric = self.matrix.select_dtypes(include=[np.number])
        for col in numeric.columns:
            nunique = numeric[col].dropna().nunique()
            assert nunique > 1, f"Column '{col}' is constant (value={numeric[col].dropna().iloc[0]})"

    def test_no_all_nan_columns(self):
        """No indicator column should be all NaN."""
        for col in self.matrix.columns:
            assert not self.matrix[col].isna().all(), f"Column '{col}' is all NaN"

    def test_no_placeholder_patterns(self):
        """Check for common placeholder patterns."""
        numeric = self.matrix.select_dtypes(include=[np.number])
        for col in numeric.columns:
            vals = numeric[col].dropna()
            if len(vals) < 5:
                continue
            # Check for all-999 or all-(-999) patterns
            assert not (vals == 999).all(), f"Column '{col}' contains placeholder value 999"
            assert not (vals == -999).all(), f"Column '{col}' contains placeholder value -999"
            # Check for sequential integers (1, 2, 3, ...)
            if vals.dtype in [np.int64, np.float64]:
                diffs = np.diff(np.sort(vals.values))
                if len(diffs) > 5:
                    assert not np.all(diffs == 1), f"Column '{col}' appears to be sequential integers"

    def test_country_count(self):
        """Verify approximately 125 countries in the output."""
        n = len(self.matrix)
        assert 100 <= n <= 130, f"Expected ~125 countries, got {n}"


@pytest.mark.skipif(not OUTPUT_DIR.exists(), reason="No output directory")
class TestOutputIntegrity:
    """Check model output files for integrity."""

    def test_composite_scores_exist(self):
        path = OUTPUT_DIR / "composite_scores.parquet"
        if not path.exists():
            pytest.skip("composite_scores.parquet not found")
        df = pd.read_parquet(path)
        assert len(df) > 0
        assert "iso3" in df.columns
        assert "composite_score" in df.columns

    def test_coupling_matrix_shape(self):
        path = OUTPUT_DIR / "coupling_matrix.npy"
        if not path.exists():
            pytest.skip("coupling_matrix.npy not found")
        W = np.load(path)
        assert W.ndim == 2
        assert W.shape[0] == W.shape[1]
        assert np.all(W >= 0), "Coupling matrix has negative entries"
        np.testing.assert_array_almost_equal(W.sum(axis=1), np.ones(W.shape[0]), decimal=3)
