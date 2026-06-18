"""Smoke tests that verify every package module imports without error."""

import importlib

import pytest

MODULES = [
    "fifa_predictor",
    "fifa_predictor.utils.logging_config",
    "fifa_predictor.utils.display",
    "fifa_predictor.data.fetch_results",
    "fifa_predictor.data.fetch_odds",
    "fifa_predictor.data.fetch_elo",
    "fifa_predictor.model.vig_removal",
    "fifa_predictor.model.poisson_inversion",
    "fifa_predictor.model.dixon_coles",
    "fifa_predictor.model.simulate",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    """Each package module should import without raising."""
    importlib.import_module(module_name)
