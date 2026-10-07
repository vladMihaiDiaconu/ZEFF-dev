"""Fetch and parse the current NBA injury report from ESPN.

This always reflects *current* state (no season/history param), so unlike
the other scrape modules this bypasses the disk cache -- caching by URL
would silently serve a stale snapshot on every rerun, which defeats the
point of an injury feed.
"""
from __future__ import annotations

import json

import pandas as pd
from curl_cffi.requests import Session

from zeff.scrape import http

ENDPOINT = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def _parse_injuries_response(raw_json: str) -> pd.DataFrame:
    payload = json.loads(raw_json)
    rows = []
    for team in payload.get("injuries", []):
        team_name = team.get("displayName")
        for injury in team.get("injuries", []):
            athlete = injury.get("athlete") or {}
            rows.append({
                "name": athlete.get("displayName"),
                "team": team_name,
                "status": injury.get("status"),
                "description": injury.get("shortComment"),
                "reported_at": injury.get("date"),
            })
    return pd.DataFrame(rows, columns=["name", "team", "status", "description", "reported_at"])


def fetch_current_injuries(session: Session) -> pd.DataFrame:
    raw_json = http.fetch(session, ENDPOINT)
    return _parse_injuries_response(raw_json)
