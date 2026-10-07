"""Turn a z-scored player pool into a ranked, display-ready table."""
from __future__ import annotations

import pandas as pd

from zeff.zscore.categories import NINE_CAT, Category


def add_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def display_columns(categories: tuple[Category, ...] = NINE_CAT) -> list[str]:
    return ["rank", "name", "team", "games", "total_z"] + [f"z_{cat.name}" for cat in categories]
