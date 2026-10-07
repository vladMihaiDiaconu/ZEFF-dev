"""Fetch and parse season per-game player stats from stats.nba.com.

`season` is the year the season ended in, e.g. 2025 for the 2024-25 season.
Unlike Basketball-Reference, this endpoint already returns one aggregated
row per player per season (no separate per-team-stint rows to dedupe for
players who were traded).

Two calls are made per season: MeasureType=Base (counting stats, shooting)
and MeasureType=Advanced (usage%), merged on the NBA player id.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from curl_cffi.requests import Session

from zeff.scrape import cache

ENDPOINT = "https://stats.nba.com/stats/leaguedashplayerstats"
TEAM_ENDPOINT = "https://stats.nba.com/stats/leaguedashteamstats"

# leaguedashteamstats' Advanced response gives TEAM_NAME, not an abbreviation.
# Stable data (team names/abbreviations rarely change) -- not worth a scrape.
TEAM_NAME_TO_ABBREVIATION = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# stats.nba.com column -> our schema column. PLAYER_ID is kept through both
# parses only to merge Base + Advanced; it's dropped before returning.
BASE_COLUMN_MAP = {
    "PLAYER_ID": "nba_player_id",
    "PLAYER_NAME": "name",
    "TEAM_ABBREVIATION": "team",
    "AGE": "age",
    "GP": "games",
    "MIN": "minutes_per_game",
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "tov",
    "FGM": "fg_made",
    "FGA": "fg_att",
    "FG_PCT": "fg_pct",
    "FTM": "ft_made",
    "FTA": "ft_att",
    "FT_PCT": "ft_pct",
    "FG3M": "threes_made",
}

ADVANCED_COLUMN_MAP = {
    "PLAYER_ID": "nba_player_id",
    "USG_PCT": "usage_pct",
}


def _season_param(season: int) -> str:
    """2025 -> '2024-25' (stats.nba.com's season string format)."""
    return f"{season - 1}-{str(season)[-2:]}"


def _build_params(season: int, measure_type: str) -> dict:
    return {
        "College": "", "Conference": "", "Country": "", "DateFrom": "", "DateTo": "",
        "Division": "", "DraftPick": "", "DraftYear": "", "GameScope": "", "GameSegment": "",
        "Height": "", "LastNGames": "0", "LeagueID": "00", "Location": "", "MeasureType": measure_type,
        "Month": "0", "OpponentTeamID": "0", "Outcome": "", "PORound": "0", "PaceAdjust": "N",
        "PerMode": "PerGame", "Period": "0", "PlayerExperience": "", "PlayerPosition": "",
        "PlusMinus": "N", "Rank": "N", "Season": _season_param(season), "SeasonSegment": "",
        "SeasonType": "Regular Season", "ShotClockRange": "", "StarterBench": "", "TeamID": "0",
        "TwoWay": "0", "VsConference": "", "VsDivision": "", "Weight": "",
    }


def _parse_response(raw_json: str, column_map: dict) -> pd.DataFrame:
    payload = json.loads(raw_json)
    result_set = payload["resultSets"][0]
    df = pd.DataFrame(result_set["rowSet"], columns=result_set["headers"])
    df = df.rename(columns=column_map)
    return df[list(column_map.values())].reset_index(drop=True)


def fetch_season_per_game_stats(
    session: Session,
    season: int,
    cache_dir: Path = cache.DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    base_json = cache.get_or_fetch(
        session, ENDPOINT, params=_build_params(season, "Base"),
        cache_dir=cache_dir, force_refresh=force_refresh,
    )
    advanced_json = cache.get_or_fetch(
        session, ENDPOINT, params=_build_params(season, "Advanced"),
        cache_dir=cache_dir, force_refresh=force_refresh,
    )

    base_df = _parse_response(base_json, BASE_COLUMN_MAP)
    advanced_df = _parse_response(advanced_json, ADVANCED_COLUMN_MAP)
    return _merge_base_and_advanced(base_df, advanced_df)


def _merge_base_and_advanced(base_df: pd.DataFrame, advanced_df: pd.DataFrame) -> pd.DataFrame:
    merged = base_df.merge(advanced_df, on="nba_player_id", how="left")
    return merged.drop(columns="nba_player_id")


def _build_team_params(season: int) -> dict:
    return {
        "Conference": "", "Division": "", "GameScope": "", "GameSegment": "", "LastNGames": "0",
        "LeagueID": "00", "Location": "", "MeasureType": "Advanced", "Month": "0",
        "OpponentTeamID": "0", "Outcome": "", "PORound": "0", "PaceAdjust": "N", "PerMode": "PerGame",
        "Period": "0", "PlusMinus": "N", "Rank": "N", "Season": _season_param(season), "SeasonSegment": "",
        "SeasonType": "Regular Season", "ShotClockRange": "", "TeamID": "0", "VsConference": "",
        "VsDivision": "",
    }


def _parse_team_pace_response(raw_json: str) -> pd.DataFrame:
    payload = json.loads(raw_json)
    result_set = payload["resultSets"][0]
    df = pd.DataFrame(result_set["rowSet"], columns=result_set["headers"])
    df["team"] = df["TEAM_NAME"].map(TEAM_NAME_TO_ABBREVIATION)
    return df[["team", "PACE"]].rename(columns={"PACE": "pace"}).reset_index(drop=True)


def fetch_team_pace(
    session: Session,
    season: int,
    cache_dir: Path = cache.DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Returns columns: team (abbreviation), pace."""
    raw_json = cache.get_or_fetch(
        session, TEAM_ENDPOINT, params=_build_team_params(season),
        cache_dir=cache_dir, force_refresh=force_refresh,
    )
    return _parse_team_pace_response(raw_json)
