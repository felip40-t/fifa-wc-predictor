# FIFA World Cup 2026 Scoreline Predictor

A Python project for predicting FIFA World Cup 2026 match scorelines from historical
results, bookmaker odds, and Elo ratings.

## Overview

The pipeline works in three stages:

1. **Data acquisition** (`fifa_predictor.data`): pulls historical match results,
   bookmaker odds, and national team Elo ratings.
2. **Modeling** (`fifa_predictor.model`): converts odds into fair probabilities
   (vig removal), derives implied goal-scoring rates (Poisson inversion), fits a
   Dixon-Coles model for scoreline probabilities, and runs Monte Carlo tournament
   simulations.
3. **Utilities** (`fifa_predictor.utils`): shared infrastructure such as logging
   configuration.

The project is currently a skeleton: module stubs are in place but the modeling
and data-fetching logic has not been implemented yet.

## Installation

```
make install
```

This installs the package in editable mode along with its dev dependencies
(pytest, ruff).

## Usage

Fetch raw data:

```
make data
```

Run the test suite:

```
make test
```

Run just the import smoke tests:

```
make smoke
```

## Development

```
make lint     # check for lint issues with ruff
make format   # auto-format source with ruff
make clean    # remove __pycache__ and .pytest_cache
```

## Project layout

```
src/fifa_predictor/
├── data/      fetch_results.py, fetch_odds.py, fetch_elo.py
├── model/     vig_removal.py, poisson_inversion.py, dixon_coles.py, monte_carlo.py
└── utils/     logging_config.py
```

See `.claude/CLAUDE.md` for project conventions and constraints.
