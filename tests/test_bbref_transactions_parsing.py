from pathlib import Path

import pytest

from zeff.scrape.bbref_transactions import _parse_transactions_html, extract_simple_trade_moves

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bbref_transactions_sample.html"


@pytest.fixture
def html():
    return FIXTURE_PATH.read_text(encoding="utf-8")


@pytest.fixture
def parsed(html):
    return _parse_transactions_html(html)


@pytest.fixture
def moves(html):
    return extract_simple_trade_moves(html)


def test_one_row_per_paragraph(parsed):
    assert len(parsed) == 4  # 2 on the first date, 1 on the second, 1 on the third


def test_date_shared_across_paragraphs_in_the_same_li(parsed):
    june_26_rows = parsed[parsed["date"] == "June 26, 2024"]
    assert len(june_26_rows) == 2


def test_is_trade_flag(parsed):
    trade_row = parsed[parsed["description"].str.contains("Kyshawn George")].iloc[0]
    assert bool(trade_row["is_trade"]) is True

    signing_row = parsed[parsed["description"].str.contains("Some Player")].iloc[0]
    assert bool(signing_row["is_trade"]) is False

    waived_row = parsed[parsed["description"].str.contains("Waived Player")].iloc[0]
    assert bool(waived_row["is_trade"]) is False


def test_resolves_players_in_a_simple_2team_trade(moves):
    moves_by_player = moves.set_index("player_name")["new_team"]
    assert moves_by_player["Kyshawn George"] == "WAS"
    assert moves_by_player["Dillon Jones"] == "NYK"


def test_uses_iso_date_format(moves):
    george_row = moves[moves["player_name"] == "Kyshawn George"].iloc[0]
    assert george_row["date"] == "2024-06-26"


def test_skips_trades_with_more_than_two_teams(moves):
    assert "Player A" not in moves["player_name"].values
    assert "Player B" not in moves["player_name"].values
    assert "Player C" not in moves["player_name"].values


def test_skips_non_trade_transactions(moves):
    assert "Some Player" not in moves["player_name"].values
    assert "Waived Player" not in moves["player_name"].values
