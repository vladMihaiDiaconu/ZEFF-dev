"""Building blocks for a forward-looking projected stat line. Pure
DataFrames in/out, no I/O -- same separation as `injury_risk.py`/
`college_translation.py`, with named/tunable constants so the approximation
stays inspectable.

`cli.py`'s `_build_projected_pool` is the default pipeline:
`blend_two_seasons_by_games` -> `apply_age_curve` -> `apply_pace_adjustment`
on the latest two seasons' raw stats. Both the blend and the age curve
constants were empirically tuned and validated by backtesting across 3
season transitions (2023-24->2024-25, 2024-25->2025-26, and one more) with
proper held-out checks -- not hand-picked. Two
things were tried and dropped because they didn't hold up: an N-season
recency-weighted blend (`blend_seasons`, still here but unused by default
-- a *fixed-ratio* blend diluted healthy full seasons with stale data,
which `blend_two_seasons_by_games`'s games-weighting avoids) and a
momentum/trend-continuation signal (never made it into this file -- its
"best" rate flipped sign between training splits, a clear overfitting
tell, and didn't beat the baseline on a proper held-out test).
`regress_shooting_pct` is also here and unused by default for the same
reason. Whichever functions a pipeline uses, re-derive fg_made/ft_made
once at the end, after every stage that touches attempts, so the row stays
internally consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DECAY = 0.6  # recency weight for a season k years back is DECAY**k

RAW_STAT_COLUMNS = [
    "pts", "reb", "ast", "stl", "blk", "tov", "threes_made",
    "fg_made", "fg_att", "ft_made", "ft_att", "minutes_per_game",
]

REGRESSION_PSEUDO_ATTEMPTS = 50  # shrink FG%/FT% toward the pool average as
                                  # if every player took this many extra
                                  # shots at the pool's rate

# Tuned via hill-climbing search against 3 backtested season transitions,
# optimizing mean Spearman rank correlation with actual next-season
# results, then confirmed on a held-out transition the search never saw.
# DECLINE_PER_YEAR converged to 0 and stayed there -- every attempt to add
# an aging penalty made held-out accuracy worse. Best guess why: a
# player's actual last season already reflects any real decline (an aging
# penalty on top double-counts it), and the draft-relevant pool is already
# survivorship-biased toward players who aged gracefully (the ones who
# didn't tend to have already fallen out of the pool).
PEAK_AGE = 25
YOUTH_GROWTH_PER_YEAR = 0.025  # per year under PEAK_AGE
DECLINE_PER_YEAR = 0.0         # per year over PEAK_AGE -- see note above
AGE_SCALED_COLUMNS = [
    "pts", "reb", "ast", "stl", "blk", "threes_made",
    "fg_att", "ft_att", "minutes_per_game",
]  # not TOV (age/turnover direction is genuinely ambiguous), not FG%/FT%
   # (handled by regression instead)

PACE_ADJUSTABLE_COLUMNS = [
    "pts", "reb", "ast", "stl", "blk", "threes_made", "tov", "fg_att", "ft_att",
]  # not minutes_per_game -- pace changes possessions per minute, not minutes


def blend_seasons(multi_season_df: pd.DataFrame, decay: float = DECAY) -> pd.DataFrame:
    """Per player, a recency-weighted average of raw per-game stats across
    whichever seasons are on file. `age`, `team`, `games`, `usage_pct` are
    carried from the most recent season rather than blended -- `games` in
    particular should reflect recent availability, not history (that's
    what `injury_risk`'s durability lookback is for).
    """
    rows = []
    for player_id, group in multi_season_df.groupby("player_id"):
        group = group.sort_values("season", ascending=False).reset_index(drop=True)
        weights = decay ** group.index.to_numpy(dtype=float)
        weights = weights / weights.sum()

        blended = {col: float((group[col] * weights).sum()) for col in RAW_STAT_COLUMNS}
        blended["fg_pct"] = blended["fg_made"] / blended["fg_att"] if blended["fg_att"] else 0.0
        blended["ft_pct"] = blended["ft_made"] / blended["ft_att"] if blended["ft_att"] else 0.0

        most_recent = group.iloc[0]
        blended.update({
            "player_id": player_id,
            "name": most_recent["name"],
            "team": most_recent["team"],
            "age": most_recent["age"],
            "games": most_recent["games"],
            "usage_pct": most_recent["usage_pct"],
        })
        rows.append(blended)

    return pd.DataFrame(rows)


def blend_two_seasons_by_games(t1_df: pd.DataFrame, t2_df: pd.DataFrame) -> pd.DataFrame:
    """Per player, blend the latest season (t1) and the one before it (t2)
    by weighting each season's raw per-game rates by how many games *that
    season* is based on. A full, healthy t1 season dominates its own blend
    almost entirely; only a short/injury-limited t1 season pulls in
    meaningful weight from t2. Falls back to t1 alone for players with no
    t2 (rookies) or when t2_df is empty (e.g. that season hasn't been
    scraped yet). `age`/`team`/`usage_pct` are carried from t1; `games` is
    t1's own games (unblended -- it should reflect recent availability).

    Validated by backtest across 3 season transitions: consistently
    improved rank correlation with actual next-season results over using
    t1 alone, including on a held-out transition never used to design this
    -- unlike the fixed-ratio `blend_seasons` below, which hurt accuracy by
    diluting healthy full seasons with stale data regardless of how many
    games either season was based on.
    """
    if t2_df.empty:
        return t1_df.copy()

    merged = t1_df.merge(t2_df, on="player_id", how="left", suffixes=("", "_t2"))
    has_t2 = merged["games_t2"].notna()
    weight_t1 = merged["games"]
    weight_t2 = merged["games_t2"].fillna(0)

    out = merged[["player_id", "name", "team", "age", "games", "usage_pct"]].copy()
    for col in RAW_STAT_COLUMNS:
        blended = (merged[col] * weight_t1 + merged[f"{col}_t2"].fillna(0) * weight_t2) / (weight_t1 + weight_t2)
        out[col] = np.where(has_t2, blended, merged[col])

    out["fg_pct"] = out["fg_made"] / out["fg_att"]
    out["ft_pct"] = out["ft_made"] / out["ft_att"]
    return out


def regress_shooting_pct(
    blended_df: pd.DataFrame, pseudo_attempts: float = REGRESSION_PSEUDO_ATTEMPTS
) -> pd.DataFrame:
    """Shrinks FG%/FT% toward the pool's volume-weighted average -- pulls
    low-volume/small-sample shooters toward the mean much more than
    high-volume ones. Sets the target percentage only; fg_made/ft_made are
    re-derived once at the end of the whole pipeline, not here.

    `fg_att`/`ft_att` are per-game rates, but `pseudo_attempts` is meant as
    a season-scale sample size (e.g. "50 extra shots" only makes sense
    against a few hundred real attempts). Regressing the per-game rate
    directly would let 50 pseudo-attempts swamp every real per-game number
    and over-regress everyone toward the pool average. So the shrinkage is
    computed on games-scaled (season-equivalent) makes/attempts instead --
    the result is still a plain percentage, so nothing needs converting
    back afterward.
    """
    df = blended_df.copy()
    fg_att_scaled = df["fg_att"] * df["games"]
    fg_made_scaled = df["fg_made"] * df["games"]
    ft_att_scaled = df["ft_att"] * df["games"]
    ft_made_scaled = df["ft_made"] * df["games"]

    pool_fg_pct = fg_made_scaled.sum() / fg_att_scaled.sum()
    pool_ft_pct = ft_made_scaled.sum() / ft_att_scaled.sum()

    df["fg_pct"] = (fg_made_scaled + pool_fg_pct * pseudo_attempts) / (fg_att_scaled + pseudo_attempts)
    df["ft_pct"] = (ft_made_scaled + pool_ft_pct * pseudo_attempts) / (ft_att_scaled + pseudo_attempts)
    return df


def apply_age_curve(df: pd.DataFrame, peak_age: float = PEAK_AGE) -> pd.DataFrame:
    """Scales counting stats, FGA/FTA, and minutes by an age factor: young
    players trend up toward a bigger role. With the tuned DECLINE_PER_YEAR
    (0.0), players at or past peak_age get a factor of exactly 1.0 -- no
    additional adjustment, positive or negative. See the module docstring
    for why."""
    df = df.copy()
    age = df["age"].fillna(peak_age)
    growth = (peak_age - age).clip(lower=0) * YOUTH_GROWTH_PER_YEAR
    decline = (age - peak_age).clip(lower=0) * DECLINE_PER_YEAR
    age_factor = 1.0 + growth - decline

    for col in AGE_SCALED_COLUMNS:
        df[col] = df[col] * age_factor
    return df


def apply_pace_adjustment(
    df: pd.DataFrame, current_teams: pd.Series, team_pace: pd.Series
) -> pd.DataFrame:
    """`current_teams` is player_id -> resolved current team (already
    override-then-auto-detected-trade resolved by the caller). `team_pace`
    is team abbreviation -> PACE. Only players whose resolved team differs
    from their most-recent-season team get scaled, and only when pace data
    exists for both teams; `team` is updated to the resolved team whenever
    it's known, independent of whether pace data was available to scale by.
    """
    df = df.copy()
    resolved_team = df["player_id"].map(current_teams)
    has_new_team = resolved_team.notna() & (resolved_team != df["team"])

    old_pace = df["team"].map(team_pace)
    new_pace = resolved_team.map(team_pace)
    can_scale = has_new_team & old_pace.notna() & new_pace.notna()
    pace_ratio = new_pace / old_pace

    for col in PACE_ADJUSTABLE_COLUMNS:
        df.loc[can_scale, col] = df.loc[can_scale, col] * pace_ratio[can_scale]

    df.loc[has_new_team, "team"] = resolved_team[has_new_team]
    return df
