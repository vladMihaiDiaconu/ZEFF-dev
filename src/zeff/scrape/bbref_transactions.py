"""Fetch and parse a season's transaction log from Basketball-Reference.

`season` is the year the season ended in, e.g. 2025 for the 2024-25 season.
The page isn't a table (`pandas.read_html` doesn't apply) -- it's a dated
list, so this uses BeautifulSoup instead.

Two things are extracted from it:
- `_parse_transactions_html`: every transaction as raw text + an `is_trade`
  flag. Not parsed into structured from-team/to-team/players; multi-team,
  multi-asset trades make that not worth it here.
- `extract_simple_trade_moves`: for genuinely simple 2-team trades only
  (exactly one team on each side), *does* resolve which team each named
  player ended up on, using the `data-attr-from`/`data-attr-to` attributes
  the site puts on the team links. Used to auto-detect a player's current
  team after an offseason trade for `rank --projected`'s pace adjustment.
  Anything more complex (3+ teams) is left to a manual override.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi.requests import Session

from zeff.scrape import cache

SEASON_URL_TEMPLATE = "https://www.basketball-reference.com/leagues/NBA_{season}_transactions.html"


def _iter_dated_paragraphs(soup: BeautifulSoup):
    container = soup.find("ul", class_="page_index")
    if container is None:
        return
    for item in container.find_all("li", recursive=False):
        span = item.find("span")
        if span is None:
            continue
        date_text = span.get_text(strip=True)
        for paragraph in item.find_all("p"):
            yield date_text, paragraph


def _parse_date_to_iso(date_text: str) -> str | None:
    try:
        return datetime.strptime(date_text, "%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_transactions_html(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for date_text, paragraph in _iter_dated_paragraphs(soup):
        description = paragraph.get_text(" ", strip=True)
        rows.append({
            "date": date_text,
            "description": description,
            "is_trade": "traded" in description.lower(),
        })
    return pd.DataFrame(rows, columns=["date", "description", "is_trade"])


def extract_simple_trade_moves(html: str) -> pd.DataFrame:
    """One row per player per simple (exactly 2-team) trade: date (ISO
    yyyy-mm-dd), player_name, new_team. Paragraphs involving 3+ teams are
    skipped -- their sentence structure isn't reliably positional."""
    soup = BeautifulSoup(html, "lxml")
    rows = []

    for date_text, paragraph in _iter_dated_paragraphs(soup):
        from_links = paragraph.find_all("a", attrs={"data-attr-from": True})
        to_links = paragraph.find_all("a", attrs={"data-attr-to": True})
        if len(from_links) != 1 or len(to_links) != 1:
            continue

        iso_date = _parse_date_to_iso(date_text)
        if iso_date is None:
            continue

        from_team = from_links[0]["data-attr-from"]
        to_team = to_links[0]["data-attr-to"]

        # "The [FROM] traded [players] to the [TO] for [players]" -- players
        # named before the TO-team link are moving to it; players named
        # after are moving (back) to the FROM-team.
        seen_to_team_link = False
        for link in paragraph.find_all("a"):
            if link.has_attr("data-attr-to"):
                seen_to_team_link = True
                continue
            if link.has_attr("data-attr-from"):
                continue
            if link.get("href", "").startswith("/players/"):
                rows.append({
                    "date": iso_date,
                    "player_name": link.get_text(strip=True),
                    "new_team": from_team if seen_to_team_link else to_team,
                })

    return pd.DataFrame(rows, columns=["date", "player_name", "new_team"])


def fetch_transactions_html(
    session: Session,
    season: int,
    cache_dir: Path = cache.DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> str:
    url = SEASON_URL_TEMPLATE.format(season=season)
    return cache.get_or_fetch(session, url, cache_dir=cache_dir, force_refresh=force_refresh)
