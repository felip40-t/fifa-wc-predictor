# Interpreter for all targets. Override for CI, e.g. `make test PYTHON=python`.
PYTHON ?= .venv/bin/python

.PHONY: install test smoke lint format clean odds simulate report predict

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

# Summarize every game from the odds CSV -> data/processed/simulated_outcomes_<competition>.csv
# Override the competition with COMP, e.g. `make simulate COMP=world_cup_2026`.
COMP ?= world_cup_2026
simulate:
	$(PYTHON) -m fifa_predictor.model.monte_carlo $(COMP)

# Pretty-print the simulated outcomes CSV as an aligned, human-readable table
report:
	$(PYTHON) -m fifa_predictor.utils.display

# Write the two-column predictions CSV -> data/processed/predictions_<comp>.csv
predict:
	$(PYTHON) -m fifa_predictor.utils.display $(COMP) --predict
