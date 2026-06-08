"""Monte Carlo tournament simulation built on top of scoreline prediction models."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fifa_predictor.model.poisson_inversion import (
    implied_goal_rates_dc,
    scoreline_probabilities_dc,
)
from fifa_predictor.model.vig_removal import (
    odds_to_implied_probabilities,
    remove_vig,
)
from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)


def simulate_match(scoreline_matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a single match scoreline from a probability matrix.

    Args:
        scoreline_matrix: Matrix where entry [i, j] is the probability of
            a final score of i-j.
        rng: NumPy random generator used for sampling.

    Returns:
        A tuple of (home_goals, away_goals) sampled for the match.
    """
    flat = scoreline_matrix.ravel()
    probabilities = flat / flat.sum()
    index = rng.choice(flat.size, p=probabilities)
    home_goals, away_goals = np.unravel_index(index, scoreline_matrix.shape)
    return int(home_goals), int(away_goals)


def _implied_rates_from_odds_row(
    row, rho: float, max_goals: int
) -> tuple[float, float, float]:
    """Invert one odds-CSV row into implied Dixon-Coles goal rates.

    Vig-removes the h2h odds [home, draw, away] and the over/under odds
    [over, under], then solves for (lh, la) against those fair probabilities
    and the row's over/under line.

    Args:
        row: A mapping with the pinnacle_h2h_* and pinnacle_ou_* fields (a
            pandas Series row or a plain dict).
        rho: Dixon-Coles low-score correlation parameter.
        max_goals: Maximum number of goals to consider per team.

    Returns:
        A tuple of (lh, la, residual_norm) from implied_goal_rates_dc.
    """
    h2h_odds = np.array(
        [row["pinnacle_h2h_home"], row["pinnacle_h2h_draw"], row["pinnacle_h2h_away"]]
    )
    p_home, p_draw, p_away = remove_vig(odds_to_implied_probabilities(h2h_odds))

    ou_odds = np.array([row["pinnacle_ou_over"], row["pinnacle_ou_under"]])
    p_over, _p_under = remove_vig(odds_to_implied_probabilities(ou_odds))

    return implied_goal_rates_dc(
        p_home,
        p_draw,
        p_away,
        row["pinnacle_ou_line"],
        p_over,
        rho=rho,
        max_goals=max_goals,
    )


def _simulate_game_outcomes(
    lh: float,
    la: float,
    rho: float,
    max_goals: int,
    n_simulations: int,
    rng: np.random.Generator,
) -> dict:
    """Simulate one game many times and summarize the outcomes.

    Builds the DC scoreline matrix from (lh, la, rho) and draws n_simulations
    scorelines, tallying win/draw/loss frequencies and the single most frequent
    scoreline.

    Args:
        lh: Implied home goal rate.
        la: Implied away goal rate.
        rho: Dixon-Coles low-score correlation parameter.
        max_goals: Maximum number of goals to consider per team.
        n_simulations: Number of match simulations to run.
        rng: NumPy random generator used for sampling.

    Returns:
        A dict with sim_p_home, sim_p_draw, sim_p_away, most_likely_scoreline,
        and most_likely_scoreline_freq.
    """
    matrix = scoreline_probabilities_dc(lh, la, rho, max_goals)

    home_wins = draws = away_wins = 0
    scoreline_counts: dict[str, int] = {}
    for _ in range(n_simulations):
        home_goals, away_goals = simulate_match(matrix, rng)
        if home_goals > away_goals:
            home_wins += 1
        elif home_goals == away_goals:
            draws += 1
        else:
            away_wins += 1
        key = f"{home_goals}-{away_goals}"
        scoreline_counts[key] = scoreline_counts.get(key, 0) + 1

    most_likely_scoreline = max(scoreline_counts, key=scoreline_counts.get)
    return {
        "sim_p_home": home_wins / n_simulations,
        "sim_p_draw": draws / n_simulations,
        "sim_p_away": away_wins / n_simulations,
        "most_likely_scoreline": most_likely_scoreline,
        "most_likely_scoreline_freq": scoreline_counts[most_likely_scoreline] / n_simulations,
    }


def simulate_games_from_odds(
    odds_csv_path: str,
    output_csv_path: str | None = None,
    n_simulations: int = 10_000,
    rho: float = -0.13,
    max_goals: int = 10,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate every game in an odds CSV and summarize the outcomes.

    Reads the odds CSV, inverts each row to implied DC goal rates, Monte Carlo
    simulates the match, and assembles a one-row-per-game summary frame. Games
    with a poor inversion fit are still included; residual_norm is surfaced so
    callers can flag or filter them downstream.

    Args:
        odds_csv_path: Path to the odds CSV (e.g. data/raw/odds_world_cup_2026.csv).
        output_csv_path: Where to write the summary CSV. Defaults to
            data/processed/simulated_outcomes_world_cup_2026.csv.
        n_simulations: Number of simulations per game.
        rho: Dixon-Coles low-score correlation parameter.
        max_goals: Maximum number of goals to consider per team.
        seed: Seed for a single shared np.random.default_rng, for reproducibility.

    Returns:
        The summary DataFrame (also written to output_csv_path).
    """
    if output_csv_path is None:
        output_csv_path = "data/processed/simulated_outcomes_world_cup_2026.csv"

    odds = pd.read_csv(odds_csv_path)
    rng = np.random.default_rng(seed)

    records = []
    for _, row in odds.iterrows():
        lh, la, residual_norm = _implied_rates_from_odds_row(row, rho, max_goals)
        outcomes = _simulate_game_outcomes(lh, la, rho, max_goals, n_simulations, rng)
        records.append(
            {
                "game_id": row["game_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "lh": lh,
                "la": la,
                "residual_norm": residual_norm,
                **outcomes,
            }
        )

    summary = pd.DataFrame.from_records(records)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv_path, index=False)
    logger.info(
        "Simulated %d games (%d sims each) -> %s",
        len(summary),
        n_simulations,
        output_csv_path,
    )
    return summary


def simulate_tournament(
    fixtures: pd.DataFrame, params: dict[str, np.ndarray], n_simulations: int
) -> pd.DataFrame:
    """Run repeated Monte Carlo simulations of a tournament bracket.

    Args:
        fixtures: DataFrame describing the tournament structure and fixtures.
        params: Fitted model parameters used to generate scoreline matrices.
        n_simulations: Number of full tournament simulations to run.

    Returns:
        A DataFrame summarizing simulation outcomes, such as each team's
        probability of reaching each stage and winning the tournament.
    """
    raise NotImplementedError


def main() -> None:
    """Simulate the default competition's games and write the summary to disk.

    Entry point for `python -m fifa_predictor.model.monte_carlo` (see the Makefile
    `simulate` target). Reads `data/raw/odds_<competition>.csv` and writes
    `data/processed/simulated_outcomes_<competition>.csv`. The competition can be
    overridden with the first CLI argument.
    """
    competition = sys.argv[1] if len(sys.argv) > 1 else "world_cup_2026"
    odds_csv_path = f"data/raw/odds_{competition}.csv"
    output_csv_path = f"data/processed/simulated_outcomes_{competition}.csv"
    simulate_games_from_odds(odds_csv_path, output_csv_path=output_csv_path)


if __name__ == "__main__":
    main()
