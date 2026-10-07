from pathlib import Path

import pytest

from zeff.scrape.cbb_reference import _parse_career_table

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cbb_player_page_sample.html"


@pytest.fixture
def parsed():
    html = FIXTURE_PATH.read_text(encoding="utf-8")
    return _parse_career_table(html)


def test_drops_the_career_summary_row(parsed):
    assert "Career" not in parsed["season_label"].values
    assert len(parsed) == 2


def test_seasons_kept_in_order(parsed):
    assert list(parsed["season_label"]) == ["2023-24", "2024-25"]


def test_final_season_values_parsed_correctly(parsed):
    final_season = parsed.iloc[-1]
    assert final_season["school"] == "Duke"
    assert final_season["games"] == 32
    assert final_season["pts"] == pytest.approx(21.5)
    assert final_season["fg_pct"] == pytest.approx(0.536)
    assert final_season["minutes_per_game"] == pytest.approx(33.0)
