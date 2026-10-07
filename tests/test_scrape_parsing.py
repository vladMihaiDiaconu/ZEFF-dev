from pathlib import Path

import pytest

from zeff.scrape.stats_nba import (
    ADVANCED_COLUMN_MAP,
    BASE_COLUMN_MAP,
    _merge_base_and_advanced,
    _parse_response,
    _parse_team_pace_response,
)

FIXTURES = Path(__file__).parent / "fixtures"
BASE_FIXTURE = FIXTURES / "leaguedashplayerstats_sample.json"
ADVANCED_FIXTURE = FIXTURES / "leaguedashplayerstats_advanced_sample.json"
TEAM_PACE_FIXTURE = FIXTURES / "leaguedashteamstats_advanced_sample.json"


@pytest.fixture
def base_df():
    return _parse_response(BASE_FIXTURE.read_text(encoding="utf-8"), BASE_COLUMN_MAP)


@pytest.fixture
def advanced_df():
    return _parse_response(ADVANCED_FIXTURE.read_text(encoding="utf-8"), ADVANCED_COLUMN_MAP)


def test_row_count_and_columns(base_df):
    assert len(base_df) == 2
    expected_columns = {
        "name", "team", "age", "games", "minutes_per_game", "pts", "reb", "ast",
        "stl", "blk", "tov", "fg_made", "fg_att", "fg_pct", "ft_made",
        "ft_att", "ft_pct", "threes_made",
    }
    assert expected_columns.issubset(base_df.columns)


def test_values_parsed_correctly(base_df):
    player_a = base_df[base_df["name"] == "Player A"].iloc[0]
    assert player_a["team"] == "DEN"
    assert player_a["age"] == 28
    assert player_a["games"] == 70
    assert player_a["pts"] == pytest.approx(27.0)
    assert player_a["fg_pct"] == pytest.approx(0.556)


def test_advanced_parses_usage_pct(advanced_df):
    assert set(advanced_df.columns) == {"nba_player_id", "usage_pct"}
    assert len(advanced_df) == 2


def test_merge_joins_usage_pct_onto_base_stats(base_df, advanced_df):
    merged = _merge_base_and_advanced(base_df, advanced_df)
    assert "nba_player_id" not in merged.columns

    player_a = merged[merged["name"] == "Player A"].iloc[0]
    assert player_a["usage_pct"] == pytest.approx(0.301)
    assert player_a["pts"] == pytest.approx(27.0)  # base columns still present

    player_b = merged[merged["name"] == "Player B"].iloc[0]
    assert player_b["usage_pct"] == pytest.approx(0.221)


def test_team_pace_maps_team_name_to_abbreviation():
    df = _parse_team_pace_response(TEAM_PACE_FIXTURE.read_text(encoding="utf-8"))
    by_team = df.set_index("team")["pace"]
    assert by_team["ATL"] == pytest.approx(102.5)
    assert by_team["TOR"] == pytest.approx(98.1)
