"""Compute per-category and total z-scores for a pool of players."""
from __future__ import annotations

import pandas as pd

from zeff.zscore.categories import NINE_CAT, Category


def compute_zscores(
    df: pd.DataFrame,
    categories: tuple[Category, ...] = NINE_CAT,
    min_games: int = 20,
) -> pd.DataFrame:
    """Return `df` filtered to players with games >= min_games, with a
    z_<category> column per category and a total_z column, sorted best-first.

    Percentage categories (FG%, FT%) use volume-weighted impact rather than
    the raw percentage, so a player who is 3/3 doesn't outrank one who is
    500/1000. Inverted categories (TOV) get their z-score negated so lower
    values rank higher.
    """
    pool = df[df["games"] >= min_games].copy()
    if pool.empty:
        raise ValueError(f"no players with games >= {min_games}")

    z_columns = []
    for cat in categories:
        z_col = f"z_{cat.name}"
        z_columns.append(z_col)

        if cat.kind == "percentage":
            league_avg = pool[cat.makes_column].sum() / pool[cat.attempts_column].sum()
            impact = (pool[cat.column] - league_avg) * pool[cat.attempts_column]
            pool[z_col] = _zscore(impact)
        elif cat.kind == "inverted":
            pool[z_col] = -_zscore(pool[cat.column])
        else:
            pool[z_col] = _zscore(pool[cat.column])

    pool["total_z"] = pool[z_columns].sum(axis=1)
    return pool.sort_values("total_z", ascending=False).reset_index(drop=True)


def rank_draft_pool(
    df: pd.DataFrame,
    pool_size: int,
    categories: tuple[Category, ...] = NINE_CAT,
    min_games: int = 20,
) -> pd.DataFrame:
    """Two-pass, draft-pool-relative ranking: a preliminary league-wide pass
    identifies the `pool_size` best players, then z-scores are recomputed
    using *only* that pool as the reference population.

    Category means/stdevs computed against the whole league (~450+ players)
    don't reflect a real draft, where only `pool_size` players (e.g. 200 for
    a 12-team, 9-cat league) actually get picked. Deep bench/waiver-wire
    players who'll never be drafted still drag the standard deviation
    around in a single-pass calculation, which understates how much real
    separation there is between stars and replacement level within the
    pool that matters. This is standard practice for fantasy value tools.
    """
    preliminary = compute_zscores(df, categories=categories, min_games=min_games)
    pool_ids = preliminary.head(pool_size)["player_id"]
    pool_df = df[df["player_id"].isin(pool_ids)]
    return compute_zscores(pool_df, categories=categories, min_games=min_games)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0:
        return series * 0
    return (series - series.mean()) / std
