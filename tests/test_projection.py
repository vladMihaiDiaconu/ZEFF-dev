import pandas as pd
import pytest

from zeff.zscore.projection import (
    AGE_SCALED_COLUMNS,
    DECLINE_PER_YEAR,
    PEAK_AGE,
    YOUTH_GROWTH_PER_YEAR,
    apply_age_curve,
    apply_pace_adjustment,
    blend_seasons,
    blend_two_seasons_by_games,
    regress_shooting_pct,
)


def test_blend_seasons_recency_weights_and_carries_recent_context():
    multi_season = pd.DataFrame([
        {"player_id": "p1", "name": "Test Player", "season": 2026, "team": "DEN", "age": 25,
         "games": 70, "usage_pct": 0.25, "pts": 20, "reb": 5, "ast": 5, "stl": 1, "blk": 1,
         "tov": 2, "threes_made": 2, "fg_made": 8, "fg_att": 16, "ft_made": 4, "ft_att": 5,
         "minutes_per_game": 30},
        {"player_id": "p1", "name": "Test Player", "season": 2025, "team": "LAL", "age": 24,
         "games": 60, "usage_pct": 0.20, "pts": 10, "reb": 4, "ast": 3, "stl": 0.5, "blk": 0.5,
         "tov": 1.5, "threes_made": 1, "fg_made": 4, "fg_att": 10, "ft_made": 2, "ft_att": 3,
         "minutes_per_game": 20},
    ])

    blended = blend_seasons(multi_season).set_index("player_id").loc["p1"]

    # weights: most recent season ~0.625, prior season ~0.375 (1 : 0.6, normalized)
    assert blended["pts"] == pytest.approx(16.25)
    assert blended["reb"] == pytest.approx(4.625)
    assert blended["minutes_per_game"] == pytest.approx(26.25)
    assert blended["fg_made"] == pytest.approx(6.5)
    assert blended["fg_att"] == pytest.approx(13.75)
    assert blended["fg_pct"] == pytest.approx(6.5 / 13.75)

    # carried from the most recent season, not blended
    assert blended["age"] == 25
    assert blended["team"] == "DEN"
    assert blended["games"] == 70
    assert blended["usage_pct"] == pytest.approx(0.25)


T1 = pd.DataFrame([
    {"player_id": "p1", "name": "Test P1", "team": "DEN", "age": 25, "games": 50, "usage_pct": 0.25,
     "pts": 20.0, "reb": 5.0, "ast": 5.0, "stl": 1.0, "blk": 1.0, "tov": 2.0, "threes_made": 2.0,
     "fg_made": 8.0, "fg_att": 16.0, "ft_made": 4.0, "ft_att": 5.0, "minutes_per_game": 30.0},
    {"player_id": "p2", "name": "Test P2 (rookie)", "team": "BOS", "age": 20, "games": 30, "usage_pct": 0.18,
     "pts": 12.0, "reb": 3.0, "ast": 2.0, "stl": 0.5, "blk": 0.3, "tov": 1.0, "threes_made": 1.0,
     "fg_made": 5.0, "fg_att": 11.0, "ft_made": 2.0, "ft_att": 3.0, "minutes_per_game": 20.0},
])
T2 = pd.DataFrame([
    {"player_id": "p1", "name": "Test P1", "team": "DEN", "age": 24, "games": 70, "usage_pct": 0.20,
     "pts": 15.0, "reb": 4.0, "ast": 4.0, "stl": 0.8, "blk": 0.8, "tov": 1.5, "threes_made": 1.5,
     "fg_made": 6.0, "fg_att": 12.0, "ft_made": 3.0, "ft_att": 4.0, "minutes_per_game": 25.0},
    # p2 (rookie) has no prior season
])


def test_blend_two_seasons_by_games_weights_by_each_seasons_games():
    result = blend_two_seasons_by_games(T1, T2).set_index("player_id")

    # p1: weight_t1=50, weight_t2=70 -> total 120
    assert result.loc["p1", "pts"] == pytest.approx((20.0 * 50 + 15.0 * 70) / 120)
    assert result.loc["p1", "minutes_per_game"] == pytest.approx((30.0 * 50 + 25.0 * 70) / 120)
    assert result.loc["p1", "fg_made"] == pytest.approx((8.0 * 50 + 6.0 * 70) / 120)
    assert result.loc["p1", "fg_att"] == pytest.approx((16.0 * 50 + 12.0 * 70) / 120)
    assert result.loc["p1", "fg_pct"] == pytest.approx(0.5)  # (400+420)/(800+840) = 820/1640

    # carried from t1, not blended
    assert result.loc["p1", "age"] == 25
    assert result.loc["p1", "team"] == "DEN"
    assert result.loc["p1", "games"] == 50


def test_blend_two_seasons_by_games_falls_back_to_t1_with_no_prior_season():
    result = blend_two_seasons_by_games(T1, T2).set_index("player_id")
    assert result.loc["p2", "pts"] == pytest.approx(12.0)
    assert result.loc["p2", "fg_pct"] == pytest.approx(5.0 / 11.0)
    assert result.loc["p2", "games"] == 30


def test_blend_two_seasons_by_games_falls_back_to_t1_when_t2_entirely_empty():
    empty_t2 = T2.iloc[0:0]
    result = blend_two_seasons_by_games(T1, empty_t2)
    pd.testing.assert_frame_equal(result, T1)


def test_regress_shooting_pct_shrinks_toward_pool_average():
    # Per-game rates, but shrinkage is calibrated against season-scale
    # volume (games x per-game attempts) -- 50 pseudo-attempts should be a
    # mild nudge against ~1600 season attempts, not an overwhelming one.
    blended = pd.DataFrame([
        {"player_id": "x", "games": 80, "fg_made": 8.0, "fg_att": 20.0, "ft_made": 6.0, "ft_att": 8.0},
        {"player_id": "y", "games": 80, "fg_made": 12.0, "fg_att": 20.0, "ft_made": 2.0, "ft_att": 8.0},
    ])
    regressed = regress_shooting_pct(blended).set_index("player_id")

    # season-scale: x made/att = 640/1600 (.400), y = 960/1600 (.600), pool = .500
    assert regressed.loc["x", "fg_pct"] == pytest.approx(665 / 1650)
    assert regressed.loc["y", "fg_pct"] == pytest.approx(985 / 1650)
    # season-scale: x made/att = 480/640 (.750), y = 160/640 (.250), pool = .500
    assert regressed.loc["x", "ft_pct"] == pytest.approx(505 / 690)
    assert regressed.loc["y", "ft_pct"] == pytest.approx(185 / 690)

    # the shrinkage should be mild at this volume -- nowhere near the pool average
    assert regressed.loc["x", "fg_pct"] < 0.45
    assert regressed.loc["y", "fg_pct"] > 0.55


def _full_row(player_id, **overrides):
    """A row with every column apply_age_curve/apply_pace_adjustment touch,
    so tests don't accidentally rely on columns that don't exist -- the
    real pipeline always hands these functions blend_seasons's full output."""
    row = {
        "player_id": player_id, "age": PEAK_AGE, "team": "DEN",
        "pts": 20.0, "reb": 5.0, "ast": 5.0, "stl": 1.0, "blk": 1.0,
        "threes_made": 2.0, "tov": 3.0, "fg_att": 15.0, "ft_att": 5.0,
        "minutes_per_game": 30.0, "fg_pct": 0.5,
    }
    row.update(overrides)
    return row


def test_apply_age_curve_peak_age_is_unchanged():
    df = pd.DataFrame([_full_row("p", age=PEAK_AGE)])
    result = apply_age_curve(df).iloc[0]
    assert result["pts"] == pytest.approx(20.0)


def test_apply_age_curve_boosts_young_players():
    df = pd.DataFrame([_full_row("young", age=PEAK_AGE - 5)])
    result = apply_age_curve(df).iloc[0]

    young_factor = 1.0 + 5 * YOUTH_GROWTH_PER_YEAR
    assert result["pts"] == pytest.approx(20.0 * young_factor)

    # TOV and FG% are deliberately not age-scaled
    assert result["tov"] == pytest.approx(3.0)
    assert result["fg_pct"] == pytest.approx(0.5)


def test_apply_age_curve_no_penalty_past_peak():
    """DECLINE_PER_YEAR is tuned to 0.0 -- see the module docstring for why.
    Players past peak_age get factor 1.0 exactly, not a penalty."""
    df = pd.DataFrame([_full_row("veteran", age=PEAK_AGE + 15)])
    result = apply_age_curve(df).iloc[0]
    assert DECLINE_PER_YEAR == 0.0
    assert result["pts"] == pytest.approx(20.0)


def test_apply_age_curve_missing_age_defaults_to_no_change():
    df = pd.DataFrame([_full_row("p", age=None)])
    result = apply_age_curve(df).iloc[0]
    assert result["pts"] == pytest.approx(20.0)


def test_apply_pace_adjustment_scales_only_resolved_traded_players():
    df = pd.DataFrame([
        _full_row("stayed", team="DEN"),
        _full_row("traded_known_pace", team="LAL"),
        _full_row("traded_unknown_pace", team="LAL"),
    ])
    current_teams = pd.Series({
        "traded_known_pace": "BOS",
        "traded_unknown_pace": "XXX",  # no pace data for this team
    })
    team_pace = pd.Series({"DEN": 100.0, "LAL": 95.0, "BOS": 104.5})

    result = apply_pace_adjustment(df, current_teams, team_pace).set_index("player_id")

    # untouched: no resolved trade
    assert result.loc["stayed", "team"] == "DEN"
    assert result.loc["stayed", "pts"] == pytest.approx(20.0)

    # scaled: resolved trade with pace data on both sides
    expected_ratio = 104.5 / 95.0
    assert result.loc["traded_known_pace", "team"] == "BOS"
    assert result.loc["traded_known_pace", "pts"] == pytest.approx(20.0 * expected_ratio)
    assert result.loc["traded_known_pace", "fg_att"] == pytest.approx(15.0 * expected_ratio)

    # team updated, but stats NOT scaled -- no pace data for the new team
    assert result.loc["traded_unknown_pace", "team"] == "XXX"
    assert result.loc["traded_unknown_pace", "pts"] == pytest.approx(20.0)
