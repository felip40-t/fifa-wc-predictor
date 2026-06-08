# Interpreter for all targets. Override for CI, e.g. `make test PYTHON=python`.
PYTHON ?= .venv/bin/python

.PHONY: install test smoke lint format clean odds simulate

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) -m pytest tests/smoke_test.py -v

lint:
	$(PYTHON) -m ruff check src/

format:
	$(PYTHON) -m ruff format src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Fetch vig-free odds -> data/raw/odds_<competition>.csv
odds:
	$(PYTHON) -m fifa_predictor.data.fetch_odds

# Simulate every game from the odds CSV -> data/processed/simulated_outcomes_<competition>.csv
simulate:
	$(PYTHON) -m fifa_predictor.model.monte_carlo
