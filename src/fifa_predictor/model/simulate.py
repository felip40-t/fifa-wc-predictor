"""Scoreline prediction model."""

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

# Minimum lead (in probability) the top match result must hold over the
# runner-up result before we commit to that result's scoreline. Below this the
# race is treated as too close to call and the overall most likely exact score
# is used instead.
RESULT_MARGIN = 0.15


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


# Odds fields the inversion requires; a row missing any of them cannot be solved.
_REQUIRED_ODDS_FIELDS = (
    "pinnacle_h2h_home",
    "pinnacle_h2h_draw",
    "pinnacle_h2h_away",
    "pinnacle_ou_line",
    "pinnacle_ou_over",
    "pinnacle_ou_under",
)


def _row_is_priced(row) -> bool:
    """True if a row carries every odds field the inversion needs.

    Confirmed-but-unpriced games (e.g. knockout matchups the books have not
    posted markets for yet) arrive with NaN odds; those rows cannot be inverted
    and are skipped rather than crashing the solve.
    """
    return not any(pd.isna(_row_value(row, field)) for field in _REQUIRED_ODDS_FIELDS)


def _implied_rates_from_odds_row(
    row, rho: float, max_goals: int
) -> tuple[float, float, float, float]:
    """Invert one odds-CSV row into implied Dixon-Coles goal rates.

    Vig-removes the h2h odds [home, draw, away] and the over/under odds
    [over, under], then fits the parameters (lh, la, rho) against those fair
    probabilities and the row's over/under line.

    Args:
        row: A mapping with the pinnacle_h2h_* and pinnacle_ou_* fields (a
            pandas Series row or a plain dict).
        rho: Starting seed for the per-game fitted Dixon-Coles correlation.
        max_goals: Maximum number of goals to consider per team.

    Returns:
        A tuple of (lh, la, rho, residual_norm) from implied_goal_rates_dc.
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


def select_headline_score(
    p_home: float,
    p_draw: float,
    p_away: float,
    result_consistent: str,
    most_likely_exact: str,
    margin: float = RESULT_MARGIN,
) -> str:
    """Pick one headline scoreline for a game.

    When the most likely match result leads the runner-up result by at least
    ``margin`` (in probability), the favored result is clear, so the
    result-consistent most likely score is used. Otherwise the race is treated
    as too close to call and the overall most likely exact scoreline is used, so
    a wafer-thin favorite does not force a win scoreline.

    Args:
        p_home: Probability of a home win.
        p_draw: Probability of a draw.
        p_away: Probability of an away win.
        result_consistent: The most likely scoreline within the most likely
            result (the game's ``likely_score``), as an "h-a" string.
        most_likely_exact: The single highest-probability exact scoreline (the
            game's ``score_1``), as an "h-a" string.
        margin: Minimum top-minus-runner-up result-probability lead required to
            commit to ``result_consistent``. Defaults to ``RESULT_MARGIN``.

    Returns:
        Either ``result_consistent`` or ``most_likely_exact`` as an "h-a" string.
    """
    top, runner_up = sorted([p_home, p_draw, p_away], reverse=True)[:2]
    return result_consistent if top - runner_up >= margin else most_likely_exact


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


def _played_game_ids(results_csv_path: str | None) -> set:
    """Game ids already played, read from the results CSV.

    Returns an empty set when no path is given or the file is absent, so a fresh
    run (or one without results yet) simply treats every game as still to come.
    """
    if results_csv_path is None or not Path(results_csv_path).exists():
        return set()
    return set(pd.read_csv(results_csv_path)["game_id"])


def _merge_into_existing(
    new_records: list[dict], odds: pd.DataFrame, output_csv_path: str
) -> pd.DataFrame:
    """Merge freshly computed rows into the existing summary CSV.

    Updating rather than rewriting keeps rows we deliberately did not recompute:
    played games (carried over from a prior run) and any hand-added games whose
    game_id is absent from the odds file. The frame is assembled in odds order,
    preferring a fresh record when present, then falling back to the existing
    row; manual rows not in the odds file are appended at the end.

    Args:
        new_records: The rows computed this run, keyed downstream by game_id.
        odds: The odds frame, used to order the merged output.
        output_csv_path: Path to the existing summary CSV (may not exist yet).

    Returns:
        The merged summary DataFrame.
    """
    fresh = {record["game_id"]: record for record in new_records}

    existing_path = Path(output_csv_path)
    existing_by_id: dict = {}
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing_by_id = {row["game_id"]: row.to_dict() for _, row in existing.iterrows()}

    merged: list[dict] = []
    for game_id in odds["game_id"]:
        if game_id in fresh:
            merged.append(fresh[game_id])
        elif game_id in existing_by_id:
            merged.append(existing_by_id[game_id])

    # Hand-added rows whose game_id never appears in the odds file are preserved
    # at the end so a manual game survives a re-run.
    odds_ids = set(odds["game_id"])
    for game_id, row in existing_by_id.items():
        if game_id not in odds_ids and game_id not in fresh:
            merged.append(row)

    return pd.DataFrame.from_records(merged)


def simulate_games_from_odds(
    odds_csv_path: str,
    output_csv_path: str | None = None,
    results_csv_path: str | None = None,
    rho: float = -0.13,
    max_goals: int = 10,
    progress: bool = False,
) -> pd.DataFrame:
    """Summarize the still-to-be-played games in an odds CSV.

    Reads the odds CSV, inverts each unplayed, priced row to implied DC goal
    rates, and reads the outcome summary directly off the exact DC scoreline
    matrix. Games whose game_id is already in the results CSV are skipped (their
    result is known), as are unpriced rows. The freshly computed rows are merged
    into the existing summary CSV rather than overwriting it, so played games and
    hand-added games carry over from earlier runs. The summary is analytic (no
    per-game sampling), so it is deterministic.

    Args:
        odds_csv_path: Path to the odds CSV (e.g. data/raw/odds_world_cup_2026.csv).
        output_csv_path: Where to write the summary CSV. Defaults to
            data/processed/simulated_outcomes_world_cup_2026.csv.
        results_csv_path: Path to the results CSV whose game_ids mark played
            games to skip. When None or absent, every game is treated as still
            to be played (the original whole-slate behavior).
        rho: Starting seed for the per-game fitted Dixon-Coles correlation. The
            fitted value (one per game) is surfaced in the output.
        max_goals: Maximum number of goals to consider per team.
        progress: When True, show a tqdm progress bar over the games (one step
            per game). Off by default so library use and tests stay quiet.

    Returns:
        The merged summary DataFrame (also written to output_csv_path).
    """
    if output_csv_path is None:
        output_csv_path = "data/processed/simulated_outcomes_world_cup_2026.csv"

    odds = pd.read_csv(odds_csv_path)
    played = _played_game_ids(results_csv_path)

    rows = odds.iterrows()
    if progress:
        rows = tqdm(rows, total=len(odds), desc="Simulating", unit="game")

    records = []
    skipped_unpriced = 0
    skipped_played = 0
    for _, row in rows:
        if row["game_id"] in played:
            skipped_played += 1
            continue
        if not _row_is_priced(row):
            skipped_unpriced += 1
            logger.warning(
                "Skipping unpriced game %s (%s vs %s): no odds posted yet",
                row["game_id"],
                row["home_team"],
                row["away_team"],
            )
            continue
        lh, la, rho_used, residual_norm = _implied_rates_from_odds_row(row, rho, max_goals)
        # Build the scoreline matrix at the same fitted rho returned by the
        # inversion so the simulated outcomes stay consistent with the solve.
        outcomes = _game_outcomes(lh, la, rho_used, max_goals)
        records.append(
            {
                "game_id": row["game_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "lh": lh,
                "la": la,
                "rho": rho_used,
                "residual_norm": residual_norm,
                **outcomes,
            }
        )

    summary = _merge_into_existing(records, odds, output_csv_path)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv_path, index=False)
    logger.info(
        "Simulated %d unplayed game(s) (%d played, %d unpriced skipped); "
        "merged into %s (%d rows total)",
        len(records),
        skipped_played,
        skipped_unpriced,
        output_csv_path,
        len(summary),
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

    Entry point for `python -m fifa_predictor.model.simulate` (see the Makefile
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
