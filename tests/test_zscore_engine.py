import pandas as pd
import pytest

from zeff.zscore.engine import compute_zscores, rank_draft_pool

# Reb/ast/stl/blk/3PM/FT% are held constant across players so their z-score
# is exactly 0 (zero variance), and PTS/TOV use the same underlying values
# so their (positive/inverted) z-scores exactly cancel. That leaves total_z
# equal to just the FG% contribution, which makes the expected numbers easy
# to hand-verify:
#
#   league FG% = (5+8+6) / (10+10+20) = 0.475
#   impact = (player_pct - 0.475) * attempts -> [0.25, 3.25, -3.5]
#   z(impact) ~= [0.0905, 1.1770, -1.2676]
PLAYERS = pd.DataFrame([
    {"name": "P1", "games": 70, "pts": 10, "tov": 10, "reb": 5, "ast": 5, "stl": 1, "blk": 1,
     "threes_made": 2, "fg_made": 5, "fg_att": 10, "fg_pct": 0.5, "ft_made": 8, "ft_att": 10, "ft_pct": 0.8},
    {"name": "P2", "games": 65, "pts": 20, "tov": 20, "reb": 5, "ast": 5, "stl": 1, "blk": 1,
     "threes_made": 2, "fg_made": 8, "fg_att": 10, "fg_pct": 0.8, "ft_made": 8, "ft_att": 10, "ft_pct": 0.8},
    {"name": "P3", "games": 60, "pts": 30, "tov": 30, "reb": 5, "ast": 5, "stl": 1, "blk": 1,
     "threes_made": 2, "fg_made": 6, "fg_att": 20, "fg_pct": 0.3, "ft_made": 8, "ft_att": 10, "ft_pct": 0.8},
    # below the default min_games threshold -- must be excluded from the pool
    {"name": "P4 (scrub)", "games": 5, "pts": 100, "tov": 0, "reb": 20, "ast": 20, "stl": 10, "blk": 10,
     "threes_made": 10, "fg_made": 10, "fg_att": 10, "fg_pct": 1.0, "ft_made": 10, "ft_att": 10, "ft_pct": 1.0},
])


def test_excludes_players_below_min_games():
    result = compute_zscores(PLAYERS, min_games=20)
    assert "P4 (scrub)" not in result["name"].values
    assert len(result) == 3


def test_zero_variance_categories_score_zero():
    result = compute_zscores(PLAYERS, min_games=20)
    for col in ["z_REB", "z_AST", "z_STL", "z_BLK", "z_3PM", "z_FT%"]:
        assert (result[col] == 0).all()


def test_counting_and_inverted_cancel_out():
    result = compute_zscores(PLAYERS, min_games=20)
    # PTS and TOV share the same values, and TOV is inverted, so they cancel.
    combined = (result["z_PTS"] + result["z_TOV"]).to_numpy()
    assert combined == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_percentage_category_is_volume_weighted():
    result = compute_zscores(PLAYERS, min_games=20).set_index("name")
    assert result.loc["P1", "z_FG%"] == pytest.approx(0.0905, abs=1e-3)
    assert result.loc["P2", "z_FG%"] == pytest.approx(1.1770, abs=1e-3)
    assert result.loc["P3", "z_FG%"] == pytest.approx(-1.2676, abs=1e-3)


def test_total_z_matches_only_surviving_category_and_ranks_by_it():
    result = compute_zscores(PLAYERS, min_games=20).set_index("name")
    for name in ["P1", "P2", "P3"]:
        assert result.loc[name, "total_z"] == pytest.approx(result.loc[name, "z_FG%"], abs=1e-9)
    # sorted best-first
    assert list(result.index) == ["P2", "P1", "P3"]


# 6 "starters" with clearly separated PTS, 4 "scrubs" with much lower PTS.
# Every other category is held constant (zero variance -> z=0) so total_z
# is driven by PTS alone, making the draft-pool recompute easy to hand-verify.
DRAFT_POOL_PLAYERS = pd.DataFrame([
    {"player_id": f"p{i}", "name": f"P{i}", "games": 70, "pts": pts, "tov": 2,
     "reb": 5, "ast": 5, "stl": 1, "blk": 1, "threes_made": 2,
     "fg_made": 5, "fg_att": 10, "fg_pct": 0.5, "ft_made": 4, "ft_att": 5, "ft_pct": 0.8}
    for i, pts in enumerate([30, 28, 26, 24, 22, 20, 10, 8, 6, 4], start=1)
])


def test_rank_draft_pool_keeps_only_the_top_n_by_preliminary_rank():
    result = rank_draft_pool(DRAFT_POOL_PLAYERS, pool_size=5, min_games=20)
    assert len(result) == 5
    assert set(result["name"]) == {"P1", "P2", "P3", "P4", "P5"}


def test_rank_draft_pool_recomputes_zscores_against_the_smaller_pool():
    result = rank_draft_pool(DRAFT_POOL_PLAYERS, pool_size=5, min_games=20).set_index("name")

    # top-5 PTS = [30, 28, 26, 24, 22] -> mean 26, pop stdev sqrt(40/5) = 2.82843
    # (very different from the mean/stdev a naive all-10-player pass would use)
    expected_std = (40 / 5) ** 0.5
    assert result.loc["P1", "z_PTS"] == pytest.approx((30 - 26) / expected_std)
    assert result.loc["P3", "z_PTS"] == pytest.approx(0.0, abs=1e-9)
    assert result.loc["P5", "z_PTS"] == pytest.approx((22 - 26) / expected_std)

    # sorted best-first
    assert list(result.index) == ["P1", "P2", "P3", "P4", "P5"]
