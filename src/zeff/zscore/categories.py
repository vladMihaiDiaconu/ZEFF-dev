"""Definitions of the standard 9 roto/H2H fantasy categories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

CategoryKind = Literal["counting", "inverted", "percentage"]


@dataclass(frozen=True)
class Category:
    name: str
    column: str
    kind: CategoryKind
    attempts_column: Optional[str] = None
    makes_column: Optional[str] = None


NINE_CAT: tuple[Category, ...] = (
    Category("PTS", "pts", "counting"),
    Category("REB", "reb", "counting"),
    Category("AST", "ast", "counting"),
    Category("STL", "stl", "counting"),
    Category("BLK", "blk", "counting"),
    Category("3PM", "threes_made", "counting"),
    Category("FG%", "fg_pct", "percentage", attempts_column="fg_att", makes_column="fg_made"),
    Category("FT%", "ft_pct", "percentage", attempts_column="ft_att", makes_column="ft_made"),
    Category("TOV", "tov", "inverted"),
)
