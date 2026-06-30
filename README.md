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
3. **Summarize each game** (`fifa_predictor.model.simulate`): builds a
   Dixon-Coles scoreline matrix from the implied rates and reads the outcome
   probabilities and most likely scorelines directly off it (exact, no
   sampling), plus two point estimates: a result-consistent most likely score
   and an expected rounded-goal-rate score.
4. **Resolve the knockout bracket** (`fifa_predictor.model.standings` +
   `knockout`): combines actual group-stage results with the headline
   predictions for games not yet played, builds the twelve group tables under
   FIFA tiebreakers, ranks the best eight third-placed teams, and fills the
   predetermined Round of 32 from `data/reference/knockout_bracket.json` (whose
   495-row table maps which third-placed group feeds which match). Once a Round
   of 32 game has been played, its actual score advances the winner: a decisive
   score picks the higher scorer, and a draw decided in extra time or on
   penalties reads the advancing team from the results CSV. Rounds past the
   Round of 32 are not resolved yet.

### Implementation status

| Area | Status |
|---|---|
| Odds fetching (`fetch_odds`) | Implemented |
| Vig removal, Poisson/Dixon-Coles inversion | Implemented |
| Per-game outcome summary (`simulate_games_from_odds`) | Implemented |
| Completed-game score fetching (`fetch_results.fetch_scores`) | Implemented |
| Prediction vs actual comparison (`utils.compare`) | Implemented |
| Group standings + Round of 32 allocation (`model.standings`, `model.knockout`) | Implemented |
| Round of 32 result resolution (winners advanced from actual scores) | Implemented |
| Historical results & Elo fetching (`fetch_results` year-range, `fetch_elo`) | Stub |
| Standalone Dixon-Coles MLE fit (`dixon_coles.py`) | Stub |
| Knockout resolution past the Round of 32 / match simulation (`simulate_tournament`) | Stub |

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
analytic and deterministic with nothing to sample or seed. Games the books have
not priced (e.g. knockout matchups) are skipped rather than failing the solve,
and games already played are skipped too. Each run merges into the existing
summary CSV, keeping played games and any hand-added rows that are not in the
current odds file.

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

Once games have been played, fetch their final scores from The Odds API
`/scores` endpoint into `data/raw/results_world_cup_2026.csv`:

```
make results
```

That endpoint only reports games completed in the last few days, so results are
accumulated: each run upserts freshly completed games onto the stored file
(keyed by `game_id`), and older games persist. Run it periodically through the
tournament.

The results file carries a `winner` column for knockout games. A decisive score
fills it automatically; a knockout draw decided in extra time or on penalties
leaves it blank for you to enter the advancing team by hand, and that value is
preserved across later fetches. Group games and decisive games can stay blank
(the winner is read from the score).

Then print predictions next to actual results for every played game:

```
make compare
```

This reads the published predictions CSV (so run `make predict` first) and joins
it to the stored results by the `"Home vs Away"` match string, which results
reconstructs from its `home_team`/`away_team` columns. Only games in both frames
(i.e. already played) appear. It shows each matchup with its predicted and actual
scoreline and ticks `RESULT` (the win/draw/away call matched) and `EXACT` (the
exact scoreline matched). It also joins the simulated-outcomes CSV (by the same
match string) to print each game's three most likely scorelines and tick `TOP3`
when the actual scoreline was one of them, so a near miss (the actual sat in our
top three but not our single headline pick) is visible even when `EXACT` fails.
The footer counts exact, result, and top-3 hits out of the games played. The
predicted scoreline is the same headline `make predict` publishes. `make compare`
also writes `data/processed/comparison_world_cup_2026.csv`. Set the competition
with `COMP`.

### Knockout stage

Resolve the group standings and fill the Round of 32 from the saved results and
predictions (so run `make predict` first):

```
make knockout
```

This uses actual results where games have been played and the headline
predictions for the rest, builds the twelve group tables under FIFA tiebreakers
(points, goal difference, goals for, then head-to-head, then a deterministic
fallback in place of drawing lots), ranks the best eight third-placed teams, and
fills the predetermined bracket. Played Round of 32 games are resolved to a
winner from their actual score; a draw decided in extra time or on penalties
needs its `winner` recorded in the results CSV, and `make knockout` stops with a
clear error if a played draw has none. It writes
`data/processed/group_standings_world_cup_2026.csv` and
`data/processed/knockout_bracket_resolved_world_cup_2026.json`. The group draw
lives in `data/reference/groups_world_cup_2026.json` and the bracket structure
plus its 495-row third-place allocation table in
`data/reference/knockout_bracket.json`. Re-run as results come in; a slot reads
`projected` until its group finishes, then `final`. Set the competition with
`COMP`.

Pretty-print either output:

```
make standings   # group tables, one block per group, with the qualification cut marked
make bracket     # the full knockout tree, R32 teams feeding through to the final
```

Each column header names the round its teams are in, from `R32` (the matchups)
through to `CHAMPION`. A Round of 32 match that has been played shows the team
that advanced; matches still to be decided, and every round past the Round of
32, show a `W##` (winner of match ##) placeholder.

For more control, call the function directly to set the `rho` starting guess
(the Dixon-Coles correlation is fitted per game, seeded from this value) or
`max_goals`:

```python
from fifa_predictor.model.simulate import simulate_games_from_odds

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
├── data/      fetch_odds.py, fetch_results.py  (fetch_elo.py: stub;
│              fetch_results year-range path: stub)
├── model/     vig_removal.py, poisson_inversion.py, simulate.py,
│              standings.py, knockout.py
│              (dixon_coles.py: stub; simulate.simulate_tournament: stub)
└── utils/     logging_config.py, display.py, compare.py
```

Raw odds and fetched results land in `data/raw/`; the group draw and the
predetermined bracket live in `data/reference/`; simulation, prediction,
comparison, standings, and resolved-bracket output go to `data/processed/`.

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
