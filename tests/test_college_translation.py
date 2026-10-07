import pandas as pd
import pytest

from zeff.zscore.college_translation import (
    COUNTING_DISCOUNT,
    EFFICIENCY_DISCOUNT,
    MINUTES_DISCOUNT,
    TOV_INFLATION,
    select_final_season,
    translate_to_nba_equivalent,
)

CAREER_DF = pd.DataFrame([
    {"season_label": "2023-24", "school": "Duke", "games": 30, "minutes_per_game": 30.0,
     "fg_made": 6.0, "fg_att": 12.0, "fg_pct": 0.500, "threes_made": 1.0,
     "ft_made": 4.0, "ft_att": 5.0, "ft_pct": 0.800,
     "reb": 8.0, "ast": 3.0, "stl": 1.0, "blk": 1.0, "tov": 2.0, "pts": 17.0},
    {"season_label": "2024-25", "school": "Duke", "games": 32, "minutes_per_game": 33.0,
     "fg_made": 7.5, "fg_att": 14.0, "fg_pct": 0.536, "threes_made": 1.5,
     "ft_made": 5.0, "ft_att": 6.0, "ft_pct": 0.833,
     "reb": 9.0, "ast": 4.0, "stl": 1.2, "blk": 1.1, "tov": 2.5, "pts": 21.5},
])


def test_select_final_season_picks_the_last_row():
    final = select_final_season(CAREER_DF)
    assert final["season_label"] == "2024-25"


def test_select_final_season_raises_on_empty_career():
    with pytest.raises(ValueError):
        select_final_season(CAREER_DF.iloc[0:0])


def test_translate_applies_each_discount_factor():
    final_season = select_final_season(CAREER_DF)
    translated = translate_to_nba_equivalent(final_season, "Test Prospect")

    assert translated["name"] == "Test Prospect"
    assert translated["team"] == "Duke"
    assert translated["games"] == 32
    assert translated["age"] is None
    assert translated["usage_pct"] is None

    assert translated["pts"] == pytest.approx(21.5 * COUNTING_DISCOUNT)
    assert translated["reb"] == pytest.approx(9.0 * COUNTING_DISCOUNT)
    assert translated["ast"] == pytest.approx(4.0 * COUNTING_DISCOUNT)
    assert translated["stl"] == pytest.approx(1.2 * COUNTING_DISCOUNT)
    assert translated["blk"] == pytest.approx(1.1 * COUNTING_DISCOUNT)
    assert translated["threes_made"] == pytest.approx(1.5 * COUNTING_DISCOUNT)
    assert translated["tov"] == pytest.approx(2.5 * TOV_INFLATION)
    assert translated["minutes_per_game"] == pytest.approx(33.0 * MINUTES_DISCOUNT)

    expected_fg_att = 14.0 * COUNTING_DISCOUNT
    expected_fg_pct = 0.536 * EFFICIENCY_DISCOUNT
    assert translated["fg_att"] == pytest.approx(expected_fg_att)
    assert translated["fg_pct"] == pytest.approx(expected_fg_pct)
    assert translated["fg_made"] == pytest.approx(expected_fg_pct * expected_fg_att)

    expected_ft_att = 6.0 * COUNTING_DISCOUNT
    expected_ft_pct = 0.833 * EFFICIENCY_DISCOUNT
    assert translated["ft_att"] == pytest.approx(expected_ft_att)
    assert translated["ft_pct"] == pytest.approx(expected_ft_pct)
    assert translated["ft_made"] == pytest.approx(expected_ft_pct * expected_ft_att)


def test_translated_row_is_internally_consistent():
    final_season = select_final_season(CAREER_DF)
    translated = translate_to_nba_equivalent(final_season, "Test Prospect")
    assert translated["fg_made"] == pytest.approx(translated["fg_pct"] * translated["fg_att"])
    assert translated["ft_made"] == pytest.approx(translated["ft_pct"] * translated["ft_att"])
