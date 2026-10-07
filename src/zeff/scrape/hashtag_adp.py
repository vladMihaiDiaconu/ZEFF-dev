"""Fetch fantasy basketball Average Draft Position (ADP) from
hashtagbasketball.com -- real aggregate draft behavior pulled from actual
Yahoo/ESPN/Fantrax leagues, not a stats model (like ZEFF's own ranking or
this same site's separate rankings page) and not one site's curated
opinion (like FantasyPros' ECR). It shows where the market actually drafts
a player, which is what "value vs ADP" comparisons are built on.

Server-rendered (an ASP.NET GridView-style table), no auth needed -- the
real per-platform ADP numbers live in `data-yadp`/`data-eadp`/`data-fadp`/
`data-madp` attributes on each `<tr>`, not in the visible cell text (which
is populated client-side for display), so those are read directly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi.requests import Session

from zeff.scrape import cache

URL = "https://hashtagbasketball.com/fantasy-basketball-adp"

TABLE_ID = "rawDataTable"


def _parse_adp_html(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=TABLE_ID)

    rows = []
    for tr in table.find_all("tr"):
        if "data-madp" not in tr.attrs:
            continue  # header/spacer rows carry no ADP data

        cells = tr.find_all("td")
        name_link = cells[0].find("a")
        name = name_link.get_text(strip=True) if name_link else cells[0].get_text(strip=True)

        rows.append({
            "name": name,
            "team": cells[1].get_text(strip=True),
            "yahoo_adp": tr.get("data-yadp"),
            "espn_adp": tr.get("data-eadp"),
            "fantrax_adp": tr.get("data-fadp"),
            "blend_adp": tr.get("data-madp"),
        })

    df = pd.DataFrame(rows, columns=["name", "team", "yahoo_adp", "espn_adp", "fantrax_adp", "blend_adp"])
    for col in ["yahoo_adp", "espn_adp", "fantrax_adp", "blend_adp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_adp(
    session: Session,
    cache_dir: Path = cache.DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    html = cache.get_or_fetch(session, URL, cache_dir=cache_dir, force_refresh=force_refresh)
    return _parse_adp_html(html)
