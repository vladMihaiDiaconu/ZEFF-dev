import pandas as pd
import pytest

from zeff.zscore.injury_risk import (
    AGE_BASELINE,
    AGE_RISK_PER_YEAR,
    DURABILITY_RISK_SCALE,
    FULL_TIME_MINUTES,
    add_games_missed_pct,
    compute_injury_risk,
    summarize_durability,
)


def test_add_games_missed_pct_uses_team_max_not_hardcoded_82():
    df = pd.DataFrame([
        {"player_id": "a", "season": 2024, "team": "DEN", "games": 70},
        {"player_id": "b", "season": 2024, "team": "DEN", "games": 56},  # missed 20% of team's 70
        {"player_id": "c", "season": 2024, "team": "LAL", "games": 82},  # different team, full season
    ])
    result = add_games_missed_pct(df)
    result = result.set_index("player_id")
    assert result.loc["a", "games_missed_pct"] == pytest.approx(0.0)
    assert result.loc["b", "games_missed_pct"] == pytest.approx(0.2)
    assert result.loc["c", "games_missed_pct"] == pytest.approx(0.0)


def test_summarize_durability_averages_across_seasons():
    df_with_pct = pd.DataFrame([
        {"player_id": "a", "games_missed_pct": 0.0},
        {"player_id": "a", "games_missed_pct": 0.4},
        {"player_id": "b", "games_missed_pct": 0.1},
    ])
    summary = summarize_durability(df_with_pct).set_index("player_id")
    assert summary.loc["a", "avg_games_missed_pct"] == pytest.approx(0.2)
    assert summary.loc["b", "avg_games_missed_pct"] == pytest.approx(0.1)


def test_compute_injury_risk_no_history_no_current_injury_is_baseline():
    players = pd.DataFrame([
        {"player_id": "clean", "age": AGE_BASELINE, "minutes_per_game": FULL_TIME_MINUTES},
    ])
    durability = pd.DataFrame(columns=["player_id", "avg_games_missed_pct"])
    injuries = pd.DataFrame(columns=["player_id", "status"])

    risk = compute_injury_risk(players, durability, injuries)
    # age at baseline, no missed-games history, no current injury, full-time minutes
    # -> every factor is 1.0 except impact_factor which is also 1.0 (36/36)
    assert risk.iloc[0] == pytest.approx(1.0)


def test_compute_injury_risk_combines_all_factors():
    players = pd.DataFrame([
        {"player_id": "risky", "age": AGE_BASELINE + 10, "minutes_per_game": FULL_TIME_MINUTES},
    ])
    durability = pd.DataFrame([{"player_id": "risky", "avg_games_missed_pct": 0.2}])
    injuries = pd.DataFrame([{"player_id": "risky", "status": "Out"}])

    risk = compute_injury_risk(players, durability, injuries)

    expected_age_factor = 1.0 + 10 * AGE_RISK_PER_YEAR
    expected_durability_factor = 1.0 + 0.2 * DURABILITY_RISK_SCALE
    expected_severity_factor = 1.5  # "Out"
    expected_impact_factor = 1.0  # 36/36
    expected = (
        expected_age_factor * expected_durability_factor
        * expected_severity_factor * expected_impact_factor
    )
    assert risk.iloc[0] == pytest.approx(expected)


def test_compute_injury_risk_low_minutes_role_player_scores_lower():
    players = pd.DataFrame([
        {"player_id": "starter", "age": AGE_BASELINE, "minutes_per_game": 36.0},
        {"player_id": "bench", "age": AGE_BASELINE, "minutes_per_game": 12.0},
    ])
    durability = pd.DataFrame(columns=["player_id", "avg_games_missed_pct"])
    injuries = pd.DataFrame(columns=["player_id", "status"])

    risk = compute_injury_risk(players, durability, injuries).set_axis(players["player_id"])
    assert risk["bench"] < risk["starter"]


def test_compute_injury_risk_missing_age_defaults_to_baseline():
    players = pd.DataFrame([
        {"player_id": "unknown_age", "age": None, "minutes_per_game": FULL_TIME_MINUTES},
    ])
    durability = pd.DataFrame(columns=["player_id", "avg_games_missed_pct"])
    injuries = pd.DataFrame(columns=["player_id", "status"])

    risk = compute_injury_risk(players, durability, injuries)
    assert risk.iloc[0] == pytest.approx(1.0)
