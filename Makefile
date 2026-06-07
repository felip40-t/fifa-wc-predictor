.PHONY: install test smoke lint format clean data

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

smoke:
	pytest tests/smoke_test.py -v

lint:
	ruff check src/

format:
	ruff format src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

data:
	python -m fifa_predictor.data.fetch_results
