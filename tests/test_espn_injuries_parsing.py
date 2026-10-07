from pathlib import Path

import pytest

from zeff.scrape.espn_injuries import _parse_injuries_response

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "espn_injuries_sample.json"


@pytest.fixture
def parsed():
    raw_json = FIXTURE_PATH.read_text(encoding="utf-8")
    return _parse_injuries_response(raw_json)


def test_flattens_all_teams_and_injuries(parsed):
    assert len(parsed) == 3
    assert set(parsed["name"]) == {"Player A", "Player B", "Player C"}


def test_columns(parsed):
    assert list(parsed.columns) == ["name", "team", "status", "description", "reported_at"]


def test_values_parsed_correctly(parsed):
    player_b = parsed[parsed["name"] == "Player B"].iloc[0]
    assert player_b["team"] == "Los Angeles Lakers"
    assert player_b["status"] == "Out"
    assert "surgery" in player_b["description"]
