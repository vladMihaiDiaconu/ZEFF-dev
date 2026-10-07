"""Translate a player's final college season into a rough NBA-equivalent
per-game stat line, for rookies without enough (or any) NBA history.

This is a simple, clearly-labeled heuristic -- not a rigorous statistical
projection. Constants are named and visible here so the approximation is
inspectable and tunable, same spirit as `zscore/injury_risk.py`.
"""
from __future__ import annotations

import pandas as pd

COUNTING_DISCOUNT = 0.75   # pace + level-of-competition discount
EFFICIENCY_DISCOUNT = 0.93  # shooting efficiency degrades less than volume does
TOV_INFLATION = 1.15        # rookies cough it up more against NBA speed/length
MINUTES_DISCOUNT = 0.85     # fewer rookie minutes than a college workload


def select_final_season(career_df: pd.DataFrame) -> pd.Series:
    """`career_df` is `cbb_reference.fetch_player_college_stats`'s output
    (one row per college season, oldest first). Returns the most recent one
    -- the season closest to entering the draft."""
    if career_df.empty:
        raise ValueError("no college seasons found to translate")
    return career_df.iloc[-1]


def translate_to_nba_equivalent(final_season: pd.Series, player_name: str) -> dict:
    """Returns a dict shaped for `store.upsert_season_stats`'s expected
    columns. `fg_made`/`ft_made` are derived from the translated pct x
    translated attempts, not scaled independently, so the row stays
    internally consistent.
    """
    fg_att = final_season["fg_att"] * COUNTING_DISCOUNT
    fg_pct = final_season["fg_pct"] * EFFICIENCY_DISCOUNT
    ft_att = final_season["ft_att"] * COUNTING_DISCOUNT
    ft_pct = final_season["ft_pct"] * EFFICIENCY_DISCOUNT

    return {
        "name": player_name,
        "team": final_season["school"],
        "age": None,
        "games": final_season["games"],
        "minutes_per_game": final_season["minutes_per_game"] * MINUTES_DISCOUNT,
        "pts": final_season["pts"] * COUNTING_DISCOUNT,
        "reb": final_season["reb"] * COUNTING_DISCOUNT,
        "ast": final_season["ast"] * COUNTING_DISCOUNT,
        "stl": final_season["stl"] * COUNTING_DISCOUNT,
        "blk": final_season["blk"] * COUNTING_DISCOUNT,
        "tov": final_season["tov"] * TOV_INFLATION,
        "fg_att": fg_att,
        "fg_pct": fg_pct,
        "fg_made": fg_pct * fg_att,
        "ft_att": ft_att,
        "ft_pct": ft_pct,
        "ft_made": ft_pct * ft_att,
        "threes_made": final_season["threes_made"] * COUNTING_DISCOUNT,
        "usage_pct": None,
    }
