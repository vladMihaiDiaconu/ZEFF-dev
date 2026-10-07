"""Disk cache of raw scraped responses, keyed by URL + params.

Keeps development iterations from re-fetching data already downloaded,
which matters for being a polite, low-volume client of stats.nba.com.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from curl_cffi.requests import Session

from zeff.scrape import http

DEFAULT_CACHE_DIR = Path("data") / "response_cache"


def _cache_path(url: str, params: dict | None, cache_dir: Path) -> Path:
    key_source = url
    if params:
        key_source += "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.txt"


def get_or_fetch(
    session: Session,
    url: str,
    params: dict | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(url, params, cache_dir)

    if path.exists() and not force_refresh:
        return path.read_text(encoding="utf-8")

    text = http.fetch(session, url, params=params)
    path.write_text(text, encoding="utf-8")
    return text
