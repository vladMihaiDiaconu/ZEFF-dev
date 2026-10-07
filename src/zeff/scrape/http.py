"""A browser-impersonating HTTP session for scraping NBA stats sites.

Both Basketball-Reference (Cloudflare) and stats.nba.com (Akamai) block
plain `requests` sessions at the TLS-fingerprint level regardless of
headers -- Basketball-Reference returns a 403, stats.nba.com just never
responds. `curl_cffi` mimics a real Chrome TLS/HTTP2 handshake, which gets
past both. Callers should prefer `cache.get_or_fetch` over calling `fetch`
directly so repeat runs during development don't re-hit the site.
"""
from __future__ import annotations

import time

from curl_cffi.requests import Session

IMPERSONATE = "chrome124"

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

DEFAULT_DELAY_SECONDS = 1.5


def create_session() -> Session:
    session = Session(impersonate=IMPERSONATE)
    session.headers.update(DEFAULT_HEADERS)
    return session


def fetch(
    session: Session,
    url: str,
    params: dict | None = None,
    delay: float = DEFAULT_DELAY_SECONDS,
) -> str:
    time.sleep(delay)
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.text
