"""Human-readable rendering of the simulated_outcomes CSV.

The raw CSV is hard to scan: game-id hashes, full-precision goal rates, and
probabilities as long decimals. This module turns one of those frames (or its
CSV path) into an aligned, percentage-formatted table for reading at a glance.
"""

import argparse
from pathlib import Path

import pandas as pd

from fifa_predictor.model.simulate import select_headline_score
from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

# Above this 1X2 + over/under inversion residual, the implied goal rates fit the
# bookmaker prices poorly; such rows are flagged so readers treat them with care.
POOR_FIT_THRESHOLD = 0.05

# How many ranked scoreline columns (score_1..score_N) the CSV carries.
_TOP_SCORELINES = 3


def _pct(value: float) -> str:
    """Render a 0-1 probability as a one-decimal percentage, e.g. 0.688 -> '68.8%'."""
    return f"{value * 100:.1f}%"


def _top_scores(row: "pd.Series") -> str:
    """Join a row's ranked scorelines into 'score freq' parts, skipping blanks."""
    parts = []
    for rank in range(1, _TOP_SCORELINES + 1):
        scoreline = row.get(f"score_{rank}")
        if pd.isna(scoreline) or scoreline == "":
            continue
        parts.append(f"{scoreline} {_pct(row[f'score_{rank}_freq'])}")
    return "   ".join(parts)


def format_simulated_outcomes(source: "str | Path | pd.DataFrame") -> str:
    """Build an aligned, human-readable table of simulated match outcomes.

    Args:
        source: Either an already-loaded summary DataFrame (as returned by
            simulate_games_from_odds) or a path to the simulated_outcomes CSV.

    Returns:
        A multi-line string: one row per game with the matchup, implied goal
        rates, win/draw/away percentages, the result-consistent most likely
        score (LIKELY), the expected rounded-goal-rate score (EXP), and the
        three most likely scorelines each with their frequency. Poorly fit
        games are marked with a trailing
        '*' and explained in a footnote. The game-id hash is intentionally
        omitted.
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)

    matchups = [f"{home} vs {away}" for home, away in zip(df["home_team"], df["away_team"])]
    flagged = [residual > POOR_FIT_THRESHOLD for residual in df["residual_norm"]]
    labels = [matchup + (" *" if flag else "") for matchup, flag in zip(matchups, flagged)]

    match_w = max([len("MATCHUP"), *(len(label) for label in labels)])

    header = (
        f"{'MATCHUP':<{match_w}}  {'λH':>4}  {'λA':>4}  "
        f"{'HOME':>6}  {'DRAW':>6}  {'AWAY':>6}  {'LIKELY':>6}  {'EXP':>4}  TOP 3 SCORES"
    )
    lines = [header, "-" * len(header)]

    for label, (_, row) in zip(labels, df.iterrows()):
        lines.append(
            f"{label:<{match_w}}  {row['lh']:>4.2f}  {row['la']:>4.2f}  "
            f"{_pct(row['sim_p_home']):>6}  {_pct(row['sim_p_draw']):>6}  "
            f"{_pct(row['sim_p_away']):>6}  {row['likely_score']:>6}  "
            f"{row['expected_score']:>4}  {_top_scores(row)}"
        )

    if any(flagged):
        lines.append("")
        lines.append(
            f"* implied goal rates fit the odds poorly "
            f"(residual_norm > {POOR_FIT_THRESHOLD}); treat with caution"
        )

    return "\n".join(lines)


def export_predictions(
    source: "str | Path | pd.DataFrame",
    output_path: "str | Path",
) -> pd.DataFrame:
    """Write a two-column predictions CSV from a simulated-outcomes frame.

    Applies select_headline_score to each game and writes a CSV with exactly two
    columns: ``match`` ("Home vs Away") and ``score`` (the chosen "h-a" string).

    Args:
        source: A summary DataFrame (as returned by simulate_games_from_odds) or
            a path to the simulated_outcomes CSV.
        output_path: Where to write the predictions CSV.

    Returns:
        The two-column predictions DataFrame (also written to output_path).
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)

    matches = [f"{home} vs {away}" for home, away in zip(df["home_team"], df["away_team"])]
    scores = [
        select_headline_score(
            row["sim_p_home"],
            row["sim_p_draw"],
            row["sim_p_away"],
            row["likely_score"],
            row["score_1"],
        )
        for _, row in df.iterrows()
    ]

    predictions = pd.DataFrame({"match": matches, "score": scores})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info("Wrote %d predictions -> %s", len(predictions), output_path)
    return predictions


def display_simulated_outcomes(source: "str | Path | pd.DataFrame") -> None:
    """Log the formatted simulated-outcomes table as a single message.

    Args:
        source: A summary DataFrame or a path to the simulated_outcomes CSV.
    """
    logger.info("Simulated outcomes:\n%s", format_simulated_outcomes(source))


def main() -> None:
    """Pretty-print a competition's simulated outcomes, or export predictions.

    Entry point for `python -m fifa_predictor.utils.display`. Reads
    `data/processed/simulated_outcomes_<competition>.csv`. With `--predict`, it
    instead writes `data/processed/predictions_<competition>.csv` (match, score).
    """
    parser = argparse.ArgumentParser(
        description="Render simulated outcomes, or export a predictions CSV."
    )
    parser.add_argument("competition", nargs="?", default="world_cup_2026")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Write the two-column predictions CSV instead of printing the table.",
    )
    args = parser.parse_args()

    source = f"data/processed/simulated_outcomes_{args.competition}.csv"
    if args.predict:
        export_predictions(source, f"data/processed/predictions_{args.competition}.csv")
    else:
        display_simulated_outcomes(source)


if __name__ == "__main__":
    main()
