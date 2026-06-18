"""Side-by-side comparison of predicted scorelines against actual results.

Joins the published predictions CSV (predicted) to the stored results frame
(actual) by the ``"Home vs Away"`` match string, which results reconstructs from
its ``home_team``/``away_team`` columns. The predicted score is read straight
from the predictions CSV, so the comparison reflects exactly what was published.
Only games present in both frames (i.e. already played) appear.

When the simulated-outcomes frame is also supplied it is joined by the same match
string to surface each game's three most likely scorelines, so a near miss (the
actual score sat in our top 3 but not our single headline pick) is visible. That
frame also carries the model's outcome marginals and DC rates, which give two
probabilities per game: the model's probability for the outcome that actually
happened and for the exact actual scoreline.
"""

import argparse
import math
from pathlib import Path

import pandas as pd

from fifa_predictor.model.poisson_inversion import scoreline_probabilities_dc
from fifa_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

_COMPARISON_COLUMNS = [
    "match",
    "predicted",
    "actual",
    "result_hit",
    "exact_hit",
    "top3",
    "top3_hit",
    "result_prob",
    "score_prob",
]

# How many ranked scorelines the simulated-outcomes CSV carries per game.
_TOP_SCORELINES = 3

# Goal ceiling used to rebuild a game's DC scoreline matrix from its stored
# (lh, la, rho). Matches the default simulate_games_from_odds uses, so the
# rebuilt matrix reproduces the one the outcomes were summarised from.
_MAX_GOALS = 10

# Maps an actual outcome (H/D/A) to the marginal-probability column that holds
# the model's probability for that outcome.
_OUTCOME_PROB_COLUMN = {
    "H": "sim_p_home",
    "D": "sim_p_draw",
    "A": "sim_p_away",
}


def _score_outcome(home_score: int, away_score: int) -> str:
    """Classify a scoreline as a home win (H), draw (D), or away win (A)."""
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _outcome_of(score: str) -> str:
    """Outcome (H/D/A) of an 'h-a' scoreline string."""
    home, away = (int(part) for part in score.split("-"))
    return _score_outcome(home, away)


def build_comparison(
    predictions_source: "str | Path | pd.DataFrame",
    results_source: "str | Path | pd.DataFrame",
    outcomes_source: "str | Path | pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Join predictions to actual results and flag exact/result/top-3 hits.

    Args:
        predictions_source: A predictions DataFrame (``match``, ``score``) or a
            path to the predictions CSV.
        results_source: A results DataFrame or a path to its CSV.
        outcomes_source: Optional simulated-outcomes DataFrame (or path) carrying
            the ranked ``score_1..score_3`` scorelines. When supplied, each row
            gets the game's three most likely scorelines (``top3``) and a
            ``top3_hit`` flag set when the actual scoreline is one of them. When
            omitted, ``top3`` is blank and ``top3_hit`` is False.

    Returns:
        A DataFrame in ``_COMPARISON_COLUMNS`` shape, one row per played game
        (present in both inputs): the matchup, the published predicted score, the
        actual score, boolean result_hit / exact_hit columns, the three most
        likely scorelines, a top3_hit flag, and two probabilities read off the
        game's DC matrix: result_prob (the model's probability for the outcome
        that actually happened) and score_prob (its probability for the exact
        actual scoreline). The two probabilities are NaN when no outcomes source
        is supplied. Empty (but typed) when no games line up.
    """
    predictions = _as_frame(predictions_source)
    results = _as_frame(results_source)

    actual_by_match = {
        f"{row['home_team']} vs {row['away_team']}": (
            int(row["home_score"]),
            int(row["away_score"]),
        )
        for _, row in results.iterrows()
    }
    outcomes_by_match = _outcomes_by_match(outcomes_source)

    rows = []
    for _, game in predictions.iterrows():
        actual = actual_by_match.get(game["match"])
        if actual is None:
            continue
        home_score, away_score = actual
        predicted = game["score"]
        actual_str = f"{home_score}-{away_score}"
        outcome = _score_outcome(home_score, away_score)
        info = outcomes_by_match.get(game["match"], {})
        top3_scores = info.get("top3", [])
        rows.append(
            {
                "match": game["match"],
                "predicted": predicted,
                "actual": actual_str,
                "result_hit": _outcome_of(predicted) == outcome,
                "exact_hit": predicted == actual_str,
                "top3": " / ".join(top3_scores),
                "top3_hit": actual_str in top3_scores,
                "result_prob": _result_prob(info, outcome),
                "score_prob": _score_prob(info, home_score, away_score),
            }
        )

    return pd.DataFrame(rows, columns=_COMPARISON_COLUMNS)


def _result_prob(info: dict, outcome: str) -> float:
    """Probability the model gave the outcome (H/D/A) that actually happened.

    Reads the matching ``sim_p_home/draw/away`` marginal off the game's outcomes
    row. Returns NaN when the marginals are absent (no outcomes source).
    """
    value = info.get(_OUTCOME_PROB_COLUMN[outcome])
    return float(value) if value is not None and pd.notna(value) else math.nan


def _score_prob(info: dict, home_score: int, away_score: int) -> float:
    """Probability of the exact actual scoreline under the game's DC matrix.

    Rebuilds the Dixon-Coles scoreline matrix from the stored (lh, la, rho),
    normalises it, and reads cell [home_score, away_score]. Returns NaN when the
    rates are absent (no outcomes source) and 0.0 when the scoreline sits beyond
    the matrix's goal ceiling.
    """
    lh, la, rho = info.get("lh"), info.get("la"), info.get("rho")
    if any(v is None or pd.isna(v) for v in (lh, la, rho)):
        return math.nan
    if home_score > _MAX_GOALS or away_score > _MAX_GOALS:
        return 0.0
    matrix = scoreline_probabilities_dc(float(lh), float(la), float(rho), _MAX_GOALS)
    return float(matrix[home_score, away_score] / matrix.sum())


def _outcomes_by_match(outcomes_source: "str | Path | pd.DataFrame | None") -> dict:
    """Map each "Home vs Away" matchup to the outcome data compare reads.

    Each value is a dict carrying the ranked ``top3`` scorelines plus, when the
    columns are present, the ``sim_p_home/draw/away`` marginals and the
    ``lh``/``la``/``rho`` rates used to rebuild the scoreline matrix. Returns an
    empty mapping when no outcomes frame is supplied, which leaves every game's
    top-3 and probability columns blank.
    """
    if outcomes_source is None:
        return {}
    outcomes = _as_frame(outcomes_source)
    score_cols = [f"score_{rank}" for rank in range(1, _TOP_SCORELINES + 1)]
    prob_cols = ["sim_p_home", "sim_p_draw", "sim_p_away", "lh", "la", "rho"]
    return {
        f"{row['home_team']} vs {row['away_team']}": {
            "top3": [row[col] for col in score_cols if pd.notna(row[col])],
            **{col: row[col] for col in prob_cols if col in row},
        }
        for _, row in outcomes.iterrows()
    }


def format_comparison(df: pd.DataFrame) -> str:
    """Build an aligned table of the comparison with a hit-count summary footer.

    Args:
        df: A comparison DataFrame as returned by build_comparison.

    Returns:
        A multi-line string: one row per game showing the matchup, predicted and
        actual scores, a tick for result/exact/top-3 hits, the model's percent
        probability for the actual outcome (RES%) and the exact actual scoreline
        (SCORE%), and the three most likely scorelines, followed by a summary line
        counting exact, result, and top-3 hits out of the total. The TOP3 column
        shows which of the three most likely scorelines the actual score was
        (1, 2, or 3), or '.' when it was not among them. RES%/SCORE% show '.' when
        no outcomes frame was supplied.
    """
    match_w = max([len("MATCHUP"), *(len(m) for m in df["match"])], default=len("MATCHUP"))

    header = (
        f"{'MATCHUP':<{match_w}}  {'PRED':>5}  {'ACTUAL':>6}  {'RESULT':>6}  {'EXACT':>5}  "
        f"{'TOP3':>5}  {'RES%':>6}  {'SCORE%':>6}  TOP 3 SCORES"
    )
    lines = [header, "-" * len(header)]

    for _, row in df.iterrows():
        lines.append(
            f"{row['match']:<{match_w}}  {row['predicted']:>5}  {row['actual']:>6}  "
            f"{('OK' if row['result_hit'] else '.'):>6}  "
            f"{('OK' if row['exact_hit'] else '.'):>5}  "
            f"{_top3_rank(row['top3'], row['actual']):>5}  "
            f"{_format_pct(row.get('result_prob')):>6}  "
            f"{_format_pct(row.get('score_prob')):>6}  {row['top3']}"
        )

    total = len(df)
    exact = int(df["exact_hit"].sum()) if total else 0
    result = int(df["result_hit"].sum()) if total else 0
    top3 = int(df["top3_hit"].sum()) if total else 0
    lines.append("")
    lines.append(f"exact {exact}/{total}, result {result}/{total}, top3 {top3}/{total}")
    return "\n".join(lines)


def export_comparison(
    predictions_source: "str | Path | pd.DataFrame",
    results_source: "str | Path | pd.DataFrame",
    output_path: "str | Path",
    outcomes_source: "str | Path | pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Build the comparison frame and write it to a CSV.

    Args:
        predictions_source: A predictions DataFrame or path to its CSV.
        results_source: A results DataFrame or path to its CSV.
        output_path: Where to write the comparison CSV.
        outcomes_source: Optional simulated-outcomes frame/path for the top-3
            columns (see build_comparison).

    Returns:
        The comparison DataFrame (also written to output_path).
    """
    df = build_comparison(predictions_source, results_source, outcomes_source)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d comparison row(s) -> %s", len(df), output_path)
    return df


def display_comparison(
    predictions_source: "str | Path | pd.DataFrame",
    results_source: "str | Path | pd.DataFrame",
    outcomes_source: "str | Path | pd.DataFrame | None" = None,
) -> None:
    """Log the formatted prediction-vs-actual table as a single message."""
    df = build_comparison(predictions_source, results_source, outcomes_source)
    logger.info("Prediction vs actual:\n%s", format_comparison(df))


def _top3_rank(top3: str, actual: str) -> str:
    """Rank (1..3) of the actual scoreline within the ' / '-joined top-3, or '.'.

    Returns the 1-based position of the actual score among the three most likely
    scorelines, so a near miss shows which of the top 3 it was. '.' when the
    actual score is not among them (or no top-3 was supplied).
    """
    scores = [s.strip() for s in top3.split(" / ") if s.strip()]
    return str(scores.index(actual) + 1) if actual in scores else "."


def _format_pct(value: "float | None") -> str:
    """Render a probability in [0, 1] as a one-decimal percent, or '.' when absent."""
    if value is None or pd.isna(value):
        return "."
    return f"{value * 100:.1f}%"


def _as_frame(source: "str | Path | pd.DataFrame") -> pd.DataFrame:
    return source if isinstance(source, pd.DataFrame) else pd.read_csv(source)


def main() -> None:
    """Print the prediction-vs-actual table, optionally writing a comparison CSV.

    Entry point for ``python -m fifa_predictor.utils.compare``. Reads
    ``data/processed/predictions_<competition>.csv``,
    ``data/raw/results_<competition>.csv``, and (for the top-3 columns)
    ``data/processed/simulated_outcomes_<competition>.csv`` when present. With
    ``--export`` it also writes ``data/processed/comparison_<competition>.csv``.
    """
    parser = argparse.ArgumentParser(
        description="Compare predicted scorelines against actual results."
    )
    parser.add_argument("competition", nargs="?", default="world_cup_2026")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Also write the comparison CSV under data/processed/.",
    )
    args = parser.parse_args()

    predictions = f"data/processed/predictions_{args.competition}.csv"
    results = f"data/raw/results_{args.competition}.csv"
    outcomes_path = Path(f"data/processed/simulated_outcomes_{args.competition}.csv")
    outcomes = outcomes_path if outcomes_path.exists() else None
    if outcomes is None:
        logger.warning(
            "No simulated outcomes at %s; top-3 columns will be blank.", outcomes_path
        )
    display_comparison(predictions, results, outcomes)
    if args.export:
        export_comparison(
            predictions,
            results,
            f"data/processed/comparison_{args.competition}.csv",
            outcomes,
        )


if __name__ == "__main__":
    main()
