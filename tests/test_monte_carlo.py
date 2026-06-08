"""Tests for the Monte Carlo simulation module."""

import numpy as np
import pytest

from fifa_predictor.model import monte_carlo


def test_simulate_match_degenerate_matrix_returns_that_scoreline() -> None:
    """A matrix with a single nonzero cell always returns that scoreline."""
    matrix = np.zeros((4, 4))
    matrix[2, 1] = 1.0
    rng = np.random.default_rng(0)

    for _ in range(20):
        assert monte_carlo.simulate_match(matrix, rng) == (2, 1)


def test_simulate_match_empirical_distribution_matches_matrix() -> None:
    """Sampling many matches should reproduce the input probabilities."""
    matrix = np.array(
        [
            [0.30, 0.10, 0.05],
            [0.15, 0.20, 0.05],
            [0.05, 0.03, 0.07],
        ]
    )
    rng = np.random.default_rng(42)

    counts = np.zeros_like(matrix)
    n = 50_000
    for _ in range(n):
        h, a = monte_carlo.simulate_match(matrix, rng)
        counts[h, a] += 1

    empirical = counts / n
    np.testing.assert_allclose(empirical, matrix, atol=0.01)


def test_simulate_match_returns_python_ints() -> None:
    """Returned goals should be plain Python ints, not numpy scalars."""
    matrix = np.zeros((3, 3))
    matrix[1, 2] = 1.0
    h, a = monte_carlo.simulate_match(matrix, np.random.default_rng(1))

    assert type(h) is int and type(a) is int
