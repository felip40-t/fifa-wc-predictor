"""Monte Carlo tournament simulation built on top of scoreline prediction models."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

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

# How many of the most frequent scorelines to record per game. The single most
# likely score often sits near ~10%, so the top few give a fuller picture.
TOP_SCORELINES = 3

# Fraction of a goal a team must be expected to add before goals_round credits
# it. Deliberately above 0.5 so sub-1.0 goal rates round down rather than being
# over-credited an extra goal.
GOALS_ROUND_UP_AT = 0.7


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


def _row_value(row, key: str) -> float:
    """Read a column from an odds row, returning NaN if the column is absent.

    Tolerates rows from older odds CSVs that predate the secondary-line columns.
    """
    try:
        return row[key]
    except (KeyError, IndexError):
        return float("nan")


def _implied_rates_from_odds_row(
    row, rho_init: float, max_goals: int
) -> tuple[float, float, float, float]:
    """Invert one odds-CSV row into implied Dixon-Coles goal rates.

    Vig-removes the h2h odds [home, draw, away] and the over/under odds
    [over, under], then solves for (lh, la, rho) against those fair
    probabilities and the row's over/under line. When a secondary over/under
    line is present (pinnacle_ou2_* columns), it is threaded into
    implied_goal_rates_dc to sharpen the goal-distribution shape.

    Args:
        row: A mapping with the pinnacle_h2h_* and pinnacle_ou_* fields (a
            pandas Series row or a plain dict). May optionally contain
            pinnacle_ou2_line, pinnacle_ou2_over, pinnacle_ou2_under.
        rho_init: Initial guess for the fitted Dixon-Coles correlation.
        max_goals: Maximum number of goals to consider per team.

    Returns:
        A tuple of (lh, la, rho, residual_norm) from implied_goal_rates_dc,
        where rho is fitted per game.
    """
    h2h_odds = np.array(
        [row["pinnacle_h2h_home"], row["pinnacle_h2h_draw"], row["pinnacle_h2h_away"]]
    )
    p_home, p_draw, p_away = remove_vig(odds_to_implied_probabilities(h2h_odds))

    ou_odds = np.array([row["pinnacle_ou_over"], row["pinnacle_ou_under"]])
    p_over, _p_under = remove_vig(odds_to_implied_probabilities(ou_odds))

    # pd.isna covers both a missing column (NaN from _row_value) and a NaN read
    # from the CSV as numpy.float64, and None, in one robust check.
    secondary_line = _row_value(row, "pinnacle_ou2_line")
    if pd.isna(secondary_line):
        secondary_line = None
        p_over_secondary = None
    else:
        ou2_odds = np.array(
            [row["pinnacle_ou2_over"], row["pinnacle_ou2_under"]]
        )
        p_over_secondary, _ = remove_vig(odds_to_implied_probabilities(ou2_odds))
        secondary_line = float(secondary_line)

    return implied_goal_rates_dc(
        p_home,
        p_draw,
        p_away,
        row["pinnacle_ou_line"],
        p_over,
        rho_init=rho_init,
        max_goals=max_goals,
        secondary_line=secondary_line,
        p_over_secondary=p_over_secondary,
    )


def goals_round(rate: float) -> int:
    """Round an expected goal rate to a whole number of goals.

    Rounds up only when the fractional part reaches GOALS_ROUND_UP_AT (0.7),
    otherwise down. More conservative than half-up so a team is credited an
    extra goal only when its rate is comfortably above the integer; sub-1.0
    rates round to 0.

    Args:
        rate: An expected (Poisson) goal rate, assumed non-negative.

    Returns:
        The rounded whole number of goals.
    """
    floor = math.floor(rate)
    return floor + (1 if rate - floor >= GOALS_ROUND_UP_AT else 0)


def _result_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return the (home, draw, away) probabilities from a scoreline matrix.

    The matrix is normalised to sum to one first, so the DC reweighting of the
    low-score cells does not bias the totals.
    """
    probabilities = matrix / matrix.sum()
    p_home = float(np.tril(probabilities, -1).sum())
    p_draw = float(np.trace(probabilities))
    p_away = float(np.triu(probabilities, 1).sum())
    return p_home, p_draw, p_away


def result_consistent_mode(matrix: np.ndarray) -> str:
    """Most likely scoreline within the most likely match result.

    Picks the most likely result (home/draw/away) from the matrix, then the
    highest-probability scoreline within that result's region. This keeps the
    headline scoreline consistent with the win/draw/away call: a clear
    favourite never gets a draw as its predicted score, even when the draw is
    the single most likely exact scoreline. Ties break toward the highest
    probability, then the fewest total goals, then the fewest home goals.

    Args:
        matrix: A scoreline probability matrix where entry [h, a] is the
            probability of a final score of h-a.

    Returns:
        The chosen scoreline as an "h-a" string.
    """
    p_home, p_draw, p_away = _result_probabilities(matrix)
    result = max(
        (("home", p_home), ("draw", p_draw), ("away", p_away)), key=lambda kv: kv[1]
    )[0]

    def in_region(h: int, a: int) -> bool:
        if result == "home":
            return h > a
        if result == "away":
            return h < a
        return h == a

    candidates = [
        (-matrix[h, a], h + a, h, a)
        for h in range(matrix.shape[0])
        for a in range(matrix.shape[1])
        if in_region(h, a)
    ]
    _, _, home_goals, away_goals = min(candidates)
    return f"{home_goals}-{away_goals}"


def _game_outcomes(lh: float, la: float, rho: float, max_goals: int) -> dict:
    """Summarize one game's outcomes analytically from its DC scoreline matrix.

    Reads win/draw/away probabilities and the most likely scorelines directly
    off the exact Dixon-Coles matrix, so there is no Monte Carlo sampling noise.
    Also derives two point estimates: a result-consistent most likely score and
    an expected (rounded goal-rate) score.

    Args:
        lh: Implied home goal rate.
        la: Implied away goal rate.
        rho: Dixon-Coles low-score correlation parameter.
        max_goals: Maximum number of goals to consider per team.

    Returns:
        A dict with sim_p_home, sim_p_draw, sim_p_away, likely_score,
        expected_score, and the three most likely scorelines as score_1..
        score_3 with their score_1_freq..score_3_freq probabilities (ranked
        most to least likely).
    """
    matrix = scoreline_probabilities_dc(lh, la, rho, max_goals)
    probabilities = matrix / matrix.sum()
    p_home, p_draw, p_away = _result_probabilities(matrix)

    outcomes = {
        "sim_p_home": p_home,
        "sim_p_draw": p_draw,
        "sim_p_away": p_away,
        "likely_score": result_consistent_mode(matrix),
        "expected_score": f"{goals_round(lh)}-{goals_round(la)}",
    }

    flat = probabilities.ravel()
    ranked = np.argsort(flat)[::-1]
    for rank in range(TOP_SCORELINES):
        index = ranked[rank]
        home_goals, away_goals = np.unravel_index(index, probabilities.shape)
        outcomes[f"score_{rank + 1}"] = f"{int(home_goals)}-{int(away_goals)}"
        outcomes[f"score_{rank + 1}_freq"] = float(flat[index])
    return outcomes


def simulate_games_from_odds(
    odds_csv_path: str,
    output_csv_path: str | None = None,
    rho: float = -0.13,
    max_goals: int = 10,
    progress: bool = False,
) -> pd.DataFrame:
    """Summarize every game in an odds CSV.

    Reads the odds CSV, inverts each row to implied DC goal rates, and reads the
    outcome summary directly off the exact DC scoreline matrix, assembling a
    one-row-per-game summary frame. The summary is analytic (no per-game
    sampling), so it is deterministic. Games with a poor inversion fit are still
    included; residual_norm is surfaced so callers can flag or filter them
    downstream.

    Args:
        odds_csv_path: Path to the odds CSV (e.g. data/raw/odds_world_cup_2026.csv).
        output_csv_path: Where to write the summary CSV. Defaults to
            data/processed/simulated_outcomes_world_cup_2026.csv.
        rho: Initial guess for the per-game fitted Dixon-Coles correlation. The
            actual rho is fitted for each game and surfaced in the output.
        max_goals: Maximum number of goals to consider per team.
        progress: When True, show a tqdm progress bar over the games (one step
            per game). Off by default so library use and tests stay quiet.

    Returns:
        The summary DataFrame (also written to output_csv_path).
    """
    if output_csv_path is None:
        output_csv_path = "data/processed/simulated_outcomes_world_cup_2026.csv"

    odds = pd.read_csv(odds_csv_path)

    rows = odds.iterrows()
    if progress:
        rows = tqdm(rows, total=len(odds), desc="Simulating", unit="game")

    records = []
    for _, row in rows:
        lh, la, fitted_rho, residual_norm = _implied_rates_from_odds_row(row, rho, max_goals)
        # Use the per-game fitted rho for the scoreline matrix so the simulated
        # outcomes stay consistent with the rates we just solved for.
        outcomes = _game_outcomes(lh, la, fitted_rho, max_goals)
        records.append(
            {
                "game_id": row["game_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "lh": lh,
                "la": la,
                "rho": fitted_rho,
                "residual_norm": residual_norm,
                **outcomes,
            }
        )

    summary = pd.DataFrame.from_records(records)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv_path, index=False)
    logger.info(
        "Summarized %d games analytically -> %s",
        len(summary),
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
    """Summarize a competition's games from its odds CSV and write the output.

    Entry point for `python -m fifa_predictor.model.monte_carlo` (see the Makefile
    `simulate` target). Reads `data/raw/odds_<competition>.csv` and writes
    `data/processed/simulated_outcomes_<competition>.csv`, showing a progress bar.
    """
    parser = argparse.ArgumentParser(
        description="Summarize every game's scoreline outcomes from an odds CSV."
    )
    parser.add_argument(
        "competition",
        nargs="?",
        default="world_cup_2026",
        help="Competition slug used to locate the odds and output CSVs.",
    )
    args = parser.parse_args()

    odds_csv_path = f"data/raw/odds_{args.competition}.csv"
    output_csv_path = f"data/processed/simulated_outcomes_{args.competition}.csv"
    simulate_games_from_odds(
        odds_csv_path,
        output_csv_path=output_csv_path,
        progress=True,
    )


if __name__ == "__main__":
    main()
