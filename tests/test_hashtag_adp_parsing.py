from pathlib import Path

import pytest

from zeff.scrape.hashtag_adp import _parse_adp_html

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hashtag_adp_sample.html"


@pytest.fixture
def parsed():
    html = FIXTURE_PATH.read_text(encoding="utf-8")
    return _parse_adp_html(html)


def test_skips_header_and_spacer_rows(parsed):
    assert len(parsed) == 2


def test_extracts_full_name_not_the_short_name_too(parsed):
    names = set(parsed["name"])
    assert names == {"Victor Wembanyama", "Nikola Jokic"}
    # the short "V.Wembanyama" mobile-only name must not be concatenated in
    assert "V.Wembanyama" not in parsed["name"].values


def test_values_parsed_correctly(parsed):
    wemby = parsed[parsed["name"] == "Victor Wembanyama"].iloc[0]
    assert wemby["team"] == "SA"
    assert wemby["yahoo_adp"] == pytest.approx(2.4)
    assert wemby["espn_adp"] == pytest.approx(2.9)
    assert wemby["fantrax_adp"] == pytest.approx(1.5)
    assert wemby["blend_adp"] == pytest.approx(2.3)
