# FIFA World Cup 2026 Scoreline Predictor

A Python project for predicting FIFA World Cup 2026 match scorelines from historical
results, bookmaker odds, and Elo ratings.

## Overview

The working pipeline turns bookmaker odds into simulated match outcomes:

1. **Fetch odds** (`fifa_predictor.data.fetch_odds`): pulls h2h (1X2) and
   over/under markets from The Odds API, preferring Pinnacle and falling back to
   a multi-book consensus, and writes vig-free-ready odds to a CSV. Alongside the
   primary over/under line it also stores Pinnacle's sharpest quarter line (e.g.
   2.25) as a second total.
2. **Invert odds to goal rates** (`fifa_predictor.model.vig_removal` +
   `poisson_inversion`): removes the bookmaker overround with the power method
   (which corrects the favorite-longshot bias the proportional method leaves
   behind), then solves the 1X2 + over/under probabilities for implied
   Dixon-Coles parameters: the goal rates (λ_home, λ_away) and the low-score
   correlation ρ. Fitting ρ alongside the rates makes the core three-market
   system exactly determined, so the draw is no longer sacrificed to fit the
   home/away and over/under prices. When a Pinnacle quarter line is available it
   is added as an extra, down-weighted total constraint that sharpens the
   goal-total shape without out-voting the draw.
3. **Summarize each game** (`fifa_predictor.model.monte_carlo`): builds a
   Dixon-Coles scoreline matrix from the implied rates and reads the outcome
   probabilities and most likely scorelines directly off it (exact, no
   sampling), plus two point estimates: a result-consistent most likely score
   and an expected rounded-goal-rate score.

### Implementation status

| Area | Status |
|---|---|
| Odds fetching (`fetch_odds`) | Implemented |
| Vig removal, Poisson/Dixon-Coles inversion | Implemented |
| Per-game outcome summary (`simulate_games_from_odds`) | Implemented |
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

Summarize every game in that odds file and write a per-game table to
`data/processed/simulated_outcomes_world_cup_2026.csv` (a progress bar tracks
the games):

```
make simulate
```

Set the competition with `COMP`:

```
make simulate COMP=world_cup_2026
```

The per-game summary is read exactly off the Dixon-Coles matrix, so it is
analytic and deterministic with nothing to sample or seed.

Each output row carries the implied goal rates (`lh`, `la`), the fitted
Dixon-Coles correlation (`rho`), the fit quality (`residual_norm`), the outcome
probabilities `sim_p_home/draw/away`, two
headline point estimates, and the three most likely scorelines:

- `likely_score` — the most likely scoreline *within* the most likely result,
  so the headline never contradicts the win/draw/away call (a clear favourite
  never shows a draw).
- `expected_score` — `round(λ_home)-round(λ_away)`, where a team is credited an
  extra goal only once its rate reaches the next `.7`. Surfaces blowout
  magnitude.
- `score_1..score_3` with `score_1_freq..score_3_freq` (ranked most to least
  likely). The single most likely exact score often sits near ~10%, so the top
  three give a fuller picture.

The CSV is dense, so pretty-print it as an aligned table for reading at a glance:

```
make report
```

This drops the game-id hash, rounds the goal rates, and shows the win/draw/away
probabilities, the `LIKELY` (result-consistent) and `EXP` (expected) scores, and
the top three scorelines as percentages. Games whose odds inversion fit poorly
(`residual_norm > 0.05`) are marked with a trailing `*`. Pass a competition name
to read a different file, e.g.
`.venv/bin/python -m fifa_predictor.utils.display world_cup_2026`.

For a single clean prediction per game, write a two-column
(`match`, `score`) CSV to `data/processed/predictions_world_cup_2026.csv`:

```
make predict
```

Each game gets one headline scoreline: the result-consistent `likely_score` when
the most likely result leads the runner-up result by a clear margin, otherwise
the single most likely exact score (so a wafer-thin favourite is not forced into
a win scoreline). Set the competition with `COMP`.

For more control, call the function directly to set the `rho` starting guess
(the Dixon-Coles correlation is fitted per game, seeded from this value) or
`max_goals`:

```python
from fifa_predictor.model.monte_carlo import simulate_games_from_odds

df = simulate_games_from_odds("data/raw/odds_world_cup_2026.csv")
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
└── utils/     logging_config.py, display.py
```

Raw odds land in `data/raw/`; simulation output goes to `data/processed/`.

See `.claude/CLAUDE.md` for project conventions and constraints.

## Next steps

The per-game final-score distribution is closed form (the DC matrix), so
sampling it adds only noise, which is why the per-game summary is read straight
off the matrix. Monte Carlo earns its place only where a quantity *can't* be
written down from the matrix. Candidate uses, roughly in order of value-to-effort
for this pipeline:

1. **Parameter / odds uncertainty (highest value).** The inversion treats the
   fitted `(λ_home, λ_away, rho)` as exact point estimates. Resample the odds
   across books (or jitter within the vig band), re-invert to a *cloud* of
   `(λ_home, λ_away)` draws, and build a matrix per draw. Yields a posterior
   predictive with credible intervals on `sim_p_*`, instead of falsely precise
   point values.
2. **Time- and state-dependent scoring.** Model goals as events over 90 minutes
   with an intensity that updates with the scoreline (leaders shut down, chasers
   push). The final score is no longer a product of two Poissons and has no clean
   closed form, so simulate the match and read the score off each run. A real
   model upgrade with more realistic blowout tails.
3. **Cross-game correlation.** Slate-level quantities (accumulators,
   "all favourites win") are a closed-form product *if* games are independent.
   Add a shared latent factor (daily goals environment, weather) and the joint
   distribution needs MC over that factor, distinct from a full bracket.
4. **Contest-score optimization.** Picking the reported scoreline that maximizes
   expected points under a pool's scoring rule is closed form for one fixed game,
   but becomes MC-natural when optimizing a whole slate against a leaderboard and
   other entrants.

Trap to avoid: anything that is a region sum of the matrix (handicap cover,
over/under at any line, exact-score hit rate) is closed form. Sample only where
there is a layer the matrix cannot express.

See also the stubs in the implementation-status table: `fetch_results`,
`fetch_elo`, standalone `dixon_coles.py` MLE, and `simulate_tournament`.
