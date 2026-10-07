"""Fetch a player's college (NCAA) per-game career stats from
sports-reference.com/cbb, for translating into an NBA-equivalent stat line
for rookies who don't have enough (or any) NBA history yet.

Same company/infrastructure as Basketball-Reference, so the existing
curl_cffi session works unchanged.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
from curl_cffi.requests import Session

from zeff.scrape import cache

SEARCH_ENDPOINT = "https://www.sports-reference.com/cbb/search/search.fcgi"
TABLE_ID = "players_per_game"

PLAYER_PAGE_RE = re.compile(r"^https://www\.sports-reference\.com/cbb/players/[^/]+\.html$")

# sports-reference.com/cbb column -> our intermediate college-stat naming
COLLEGE_COLUMN_MAP = {
    "Season": "season_label",
    "Team": "school",
    "G": "games",
    "MP": "minutes_per_game",
    "FG": "fg_made",
    "FGA": "fg_att",
    "FG%": "fg_pct",
    "3P": "threes_made",
    "FT": "ft_made",
    "FTA": "ft_att",
    "FT%": "ft_pct",
    "TRB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "tov",
    "PTS": "pts",
}

NUMERIC_COLUMNS = [
    "games", "minutes_per_game", "fg_made", "fg_att", "fg_pct", "threes_made",
    "ft_made", "ft_att", "ft_pct", "reb", "ast", "stl", "blk", "tov", "pts",
]


def _resolve_player_url(session: Session, player_name: str) -> str:
    response = session.get(SEARCH_ENDPOINT, params={"search": player_name}, timeout=60)
    response.raise_for_status()
    final_url = response.url
    if not PLAYER_PAGE_RE.match(final_url):
        raise ValueError(
            f"could not resolve a single college player page for {player_name!r} "
            f"(landed on {final_url!r}) -- try a more specific name"
        )
    return final_url


def _parse_career_table(html: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(io.StringIO(html), attrs={"id": TABLE_ID})
    except ValueError:
        uncommented = html.replace("<!--", "").replace("-->", "")
        tables = pd.read_html(io.StringIO(uncommented), attrs={"id": TABLE_ID})

    df = tables[0].rename(columns=COLLEGE_COLUMN_MAP)
    df = df[list(COLLEGE_COLUMN_MAP.values())]

    # The table ends with a "Career" summary row, which isn't one season.
    df = df[~df["season_label"].astype(str).str.contains("Career", case=False, na=False)]

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["games"])

    return df.reset_index(drop=True)


def fetch_player_college_stats(
    session: Session,
    player_name: str,
    cache_dir: Path = cache.DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Returns one row per college season (career table), oldest first."""
    url = _resolve_player_url(session, player_name)
    html = cache.get_or_fetch(session, url, cache_dir=cache_dir, force_refresh=force_refresh)
    return _parse_career_table(html)
