# FIFA World Cup 2026 Scoreline Predictor

A Python project for predicting FIFA World Cup 2026 match scorelines from historical
results, bookmaker odds, and Elo ratings.

## Overview

The working pipeline turns bookmaker odds into simulated match outcomes:

1. **Fetch odds** (`fifa_predictor.data.fetch_odds`): pulls h2h (1X2) and
   over/under markets from The Odds API, preferring Pinnacle and falling back to
   a multi-book consensus, and writes vig-free-ready odds to a CSV.
2. **Invert odds to goal rates** (`fifa_predictor.model.vig_removal` +
   `poisson_inversion`): removes the bookmaker overround, then solves the 1X2 +
   over/under probabilities for implied Dixon-Coles goal rates (λ_home, λ_away).
3. **Simulate each game** (`fifa_predictor.model.monte_carlo`): builds a
   Dixon-Coles scoreline matrix from the implied rates and Monte Carlo samples it
   to produce per-game outcome probabilities and the most likely scoreline.

### Implementation status

| Area | Status |
|---|---|
| Odds fetching (`fetch_odds`) | Implemented |
| Vig removal, Poisson/Dixon-Coles inversion | Implemented |
| Per-game Monte Carlo (`simulate_games_from_odds`) | Implemented |
| Results & Elo fetching (`fetch_results`, `fetch_elo`) | Stub |
| Standalone Dixon-Coles MLE fit (`dixon_coles.py`) | Stub |
| Full tournament bracket (`simulate_tournament`) | Stub |

## Installation

```
make install
```

This installs the package in editable mode along with its dev dependencies
(pytest, ruff).

Fetching odds requires a The Odds API key. Put it in a `.env` file at the repo
root:

```
ODDS_API_KEY=your_key_here
```

## Usage

Fetch vig-free odds for the World Cup (writes `data/raw/odds_world_cup_2026.csv`):

```
make odds
```

Simulate every game in that odds file and write a per-game summary to
`data/processed/simulated_outcomes_world_cup_2026.csv`:

```
make simulate
```

Each output row carries the implied goal rates (`lh`, `la`), the fit quality
(`residual_norm`), simulated `sim_p_home/draw/away`, and the
`most_likely_scoreline` with its frequency.

For more control, call the function directly to set `n_simulations`, `rho`
(Dixon-Coles correlation), `max_goals`, or a `seed` for reproducibility:

```python
from fifa_predictor.model.monte_carlo import simulate_games_from_odds

df = simulate_games_from_odds("data/raw/odds_world_cup_2026.csv", seed=1)
```

See the `simulate_games_from_odds` docstring for the full set of options.

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
├── data/      fetch_odds.py            (fetch_results.py, fetch_elo.py: stubs)
├── model/     vig_removal.py, poisson_inversion.py, monte_carlo.py
│              (dixon_coles.py: stub; monte_carlo.simulate_tournament: stub)
└── utils/     logging_config.py
```

Raw odds land in `data/raw/`; simulation output goes to `data/processed/`.

See `.claude/CLAUDE.md` for project conventions and constraints.
