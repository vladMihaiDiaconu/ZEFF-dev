"""Composite injury-risk score: age, missed-games durability history,
current injury status, and on-court role. Pure DataFrame-in/out, no I/O --
same separation as `zscore/engine.py`.

There's no real historical injury-event log feeding this (see the project
plan for why: the standard source for that, Pro Sports Transactions, blocks
automated access even with a real browser engine). `avg_games_missed_pct`
is a durability *proxy* -- fraction of a team's games a player missed,
averaged over whatever seasons are on file for them -- not a count of
actual injury diagnoses.
"""
from __future__ import annotations

import pandas as pd

AGE_BASELINE = 24
AGE_RISK_PER_YEAR = 0.05
DURABILITY_RISK_SCALE = 1.5
FULL_TIME_MINUTES = 36.0

SEVERITY_KEYWORDS = (
    ("out", 1.5),
    ("doubtful", 1.3),
    ("day-to-day", 1.1),
    ("questionable", 1.1),
    ("probable", 1.02),
)


def add_games_missed_pct(multi_season_df: pd.DataFrame) -> pd.DataFrame:
    """Add `games_missed_pct` per row: 1 - games / (max games played by any
    of that player's teammates that season). Using the observed team max
    rather than a hardcoded 82 handles shortened seasons and in-progress
    ones without special-casing.
    """
    df = multi_season_df.copy()
    team_max_games = df.groupby(["season", "team"])["games"].transform("max")
    df["games_missed_pct"] = 1 - df["games"] / team_max_games
    return df


def summarize_durability(df_with_pct: pd.DataFrame) -> pd.DataFrame:
    """Per player, average `games_missed_pct` across whatever seasons are
    present -- not an error if that's just one season."""
    return df_with_pct.groupby("player_id", as_index=False).agg(
        avg_games_missed_pct=("games_missed_pct", "mean")
    )


def _severity_factor(status) -> float:
    if status is None or pd.isna(status):
        return 1.0
    status_lower = str(status).lower()
    for keyword, factor in SEVERITY_KEYWORDS:
        if keyword in status_lower:
            return factor
    return 1.0


def compute_injury_risk(
    players_df: pd.DataFrame,
    durability_df: pd.DataFrame,
    current_injuries_df: pd.DataFrame,
) -> pd.Series:
    """`players_df` needs player_id, age, minutes_per_game.
    `durability_df` is `summarize_durability`'s output.
    `current_injuries_df` needs player_id, status (e.g. `store.load_injuries`).

    Returns a Series aligned to players_df's index. Missing durability
    history or missing age/current-status default to "nothing notable"
    (factor 1.0) rather than NaN/error.
    """
    avg_missed_by_player = durability_df.set_index("player_id")["avg_games_missed_pct"]
    status_by_player = current_injuries_df.set_index("player_id")["status"]

    age = players_df["age"].fillna(AGE_BASELINE)
    avg_games_missed_pct = players_df["player_id"].map(avg_missed_by_player).fillna(0.0)
    status = players_df["player_id"].map(status_by_player)
    minutes_per_game = players_df["minutes_per_game"]

    age_factor = 1.0 + (age - AGE_BASELINE).clip(lower=0) * AGE_RISK_PER_YEAR
    durability_factor = 1.0 + avg_games_missed_pct * DURABILITY_RISK_SCALE
    severity_factor = status.apply(_severity_factor)
    impact_factor = minutes_per_game / FULL_TIME_MINUTES

    return age_factor * durability_factor * severity_factor * impact_factor
