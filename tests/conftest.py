"""Test fixtures for the spectral instability model."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_countries():
    """10 test countries."""
    return ["USA", "CHN", "JPN", "DEU", "GBR", "FRA", "IND", "BRA", "CAN", "AUS"]


@pytest.fixture
def small_indicator_matrix(small_countries):
    """10 countries × 5 indicators, seeded."""
    rng = np.random.RandomState(42)
    X = rng.randn(10, 5)
    return pd.DataFrame(X, index=small_countries,
                         columns=["ind1", "ind2", "ind3", "ind4", "ind5"])


@pytest.fixture
def small_coupling_matrix():
    """10×10 row-stochastic coupling matrix."""
    rng = np.random.RandomState(42)
    W = rng.rand(10, 10)
    np.fill_diagonal(W, 0)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    return W


@pytest.fixture
def small_composite_scores():
    """10 composite scores."""
    rng = np.random.RandomState(42)
    return rng.randn(10)
