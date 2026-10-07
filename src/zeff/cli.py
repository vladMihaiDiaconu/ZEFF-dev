"""Command-line entry point: scrape a season, then rank it.

    python -m zeff.cli scrape --season 2025
    python -m zeff.cli scrape-injuries
    python -m zeff.cli scrape-transactions --season 2025
    python -m zeff.cli scrape-college --player "Cooper Flagg" --season 2026
    python -m zeff.cli scrape-adp
    python -m zeff.cli set-team --player "Kawhi Leonard" --team TOR
    python -m zeff.cli rank --season 2025 --top 50
    python -m zeff.cli rank --season 2026 --projected --show-adp --top 50
"""
from __future__ import annotations

import argparse

import pandas as pd

from zeff.db import store
from zeff.scrape import bbref_transactions, cbb_reference, espn_injuries, hashtag_adp, http, stats_nba
from zeff.zscore import college_translation, engine, injury_risk, projection, rank


def _cmd_scrape(args: argparse.Namespace) -> None:
    session = http.create_session()
    df = stats_nba.fetch_season_per_game_stats(
        session, args.season, force_refresh=args.force_refresh
    )
    conn = store.connect()
    store.upsert_season_stats(conn, args.season, df)
    print(f"scraped and stored {len(df)} players for season {args.season}")


def _cmd_scrape_injuries(args: argparse.Namespace) -> None:
    session = http.create_session()
    df = espn_injuries.fetch_current_injuries(session)
    conn = store.connect()
    store.replace_injuries(conn, df)
    print(f"scraped and stored {len(df)} current injuries")


def _cmd_scrape_transactions(args: argparse.Namespace) -> None:
    session = http.create_session()
    html = bbref_transactions.fetch_transactions_html(
        session, args.season, force_refresh=args.force_refresh
    )
    df = bbref_transactions._parse_transactions_html(html)
    moves = bbref_transactions.extract_simple_trade_moves(html)

    conn = store.connect()
    store.upsert_transactions(conn, args.season, df)
    store.upsert_trade_moves(conn, args.season, moves)

    trade_count = int(df["is_trade"].sum())
    print(
        f"scraped and stored {len(df)} transactions for season {args.season} ({trade_count} trades), "
        f"auto-resolved {len(moves)} player team moves from simple 2-team trades"
    )


def _cmd_set_team(args: argparse.Namespace) -> None:
    conn = store.connect()
    store.set_team_override(conn, args.player, args.team)
    print(f"set {args.player}'s current team override to {args.team}")


def _cmd_scrape_college(args: argparse.Namespace) -> None:
    session = http.create_session()
    career_df = cbb_reference.fetch_player_college_stats(
        session, args.player, force_refresh=args.force_refresh
    )
    final_season = college_translation.select_final_season(career_df)
    translated = college_translation.translate_to_nba_equivalent(final_season, args.player)

    conn = store.connect()
    store.upsert_season_stats(
        conn, args.season, pd.DataFrame([translated]), data_source="college_translated"
    )
    print(
        f"translated {args.player}'s final college season "
        f"({final_season['season_label']}, {final_season['school']}) to an NBA-equivalent "
        f"line for season {args.season}: {translated['pts']:.1f} pts, {translated['reb']:.1f} reb, "
        f"{translated['ast']:.1f} ast, {translated['fg_pct'] * 100:.1f}% FG, "
        f"{translated['minutes_per_game']:.1f} min"
    )


def _cmd_scrape_adp(args: argparse.Namespace) -> None:
    session = http.create_session()
    df = hashtag_adp.fetch_adp(session, force_refresh=args.force_refresh)
    conn = store.connect()
    store.replace_adp(conn, df)
    print(f"scraped and stored ADP for {len(df)} players")


def _add_adp_comparison(conn, ranked: pd.DataFrame) -> pd.DataFrame:
    """`ranked` must already have a `rank` column (from `rank.add_rank`).
    ADP is itself a pick-number scale, directly comparable to `rank`.
    `vs_adp` = adp - our rank: positive means the market drafts this player
    later than we rank them (value at that pick), negative means the market
    drafts them earlier than we do (a reach relative to our numbers, or the
    market pricing in something -- role, hype, name value -- our box-score
    model doesn't).
    """
    adp = store.load_adp(conn)
    ranked = ranked.merge(
        adp[["player_id", "blend_adp"]], on="player_id", how="left"
    ).rename(columns={"blend_adp": "adp"})
    ranked["vs_adp"] = ranked["adp"] - ranked["rank"]
    return ranked


def _enrich_ranked_table(conn, ranked: pd.DataFrame, season: int, seasons_back: int) -> pd.DataFrame:
    injuries = store.load_injuries(conn)
    ranked = ranked.merge(
        injuries[["player_id", "status"]], on="player_id", how="left"
    ).rename(columns={"status": "injury_status"})

    transactions = store.load_transactions(conn, season)
    trade_descriptions = transactions.loc[transactions["is_trade"] == 1, "description"].tolist()
    ranked["recently_traded"] = ranked["name"].apply(
        lambda name: any(name in description for description in trade_descriptions)
    )

    lookback_seasons = list(range(season - seasons_back + 1, season + 1))
    multi_season = store.load_seasons_stats(conn, lookback_seasons)
    durability = injury_risk.summarize_durability(injury_risk.add_games_missed_pct(multi_season))
    ranked["injury_risk"] = injury_risk.compute_injury_risk(ranked, durability, injuries)

    return ranked


def _build_projected_pool(conn, season: int) -> pd.DataFrame:
    """Games-weighted blend of the latest two seasons, age-curved, with a
    pace adjustment for players who changed teams. The blend and the age
    curve constants (peak_age=25, no decline penalty) were both tuned and
    validated by backtest across 3 season transitions with held-out checks
    -- see `zscore/projection.py`'s module docstring for what was tried and
    why the rest (an N-season fixed-ratio blend, shooting-pct regression, a
    momentum/trend signal) got left out.
    """
    t1 = store.load_season_stats(conn, season)
    if t1.empty:
        raise SystemExit(f"no data for season {season} -- run `scrape --season {season}` first")
    t2 = store.load_season_stats(conn, season - 1)

    blended = projection.blend_two_seasons_by_games(t1, t2)
    aged = projection.apply_age_curve(blended)

    overrides = store.load_team_overrides(conn)
    auto_moves = store.load_current_teams(conn)
    current_teams = overrides.combine_first(auto_moves)  # override wins if both exist

    session = http.create_session()
    team_pace = stats_nba.fetch_team_pace(session, season).set_index("team")["pace"]

    projected = projection.apply_pace_adjustment(aged, current_teams, team_pace)
    projected["fg_made"] = projected["fg_pct"] * projected["fg_att"]
    projected["ft_made"] = projected["ft_pct"] * projected["ft_att"]
    projected["data_source"] = "projected"

    return projected


def _cmd_rank(args: argparse.Namespace) -> None:
    conn = store.connect()
    if args.projected:
        df = _build_projected_pool(conn, args.season)
    else:
        df = store.load_season_stats(conn, args.season)
        if df.empty:
            raise SystemExit(
                f"no data for season {args.season} -- run `scrape --season {args.season}` first"
            )

    ranked = engine.rank_draft_pool(df, pool_size=args.pool_size, min_games=args.min_games)
    ranked = rank.add_rank(ranked)
    ranked = _enrich_ranked_table(conn, ranked, args.season, args.seasons_back)
    ranked = ranked.rename(columns={"data_source": "source"})
    columns = rank.display_columns() + [
        "age", "usage_pct", "injury_status", "recently_traded", "injury_risk", "source",
    ]
    if args.show_adp:
        ranked = _add_adp_comparison(conn, ranked)
        columns += ["adp", "vs_adp"]

    print(ranked[columns].head(args.top).to_string(index=False))
    if args.csv:
        ranked[columns].to_csv(args.csv, index=False)
        print(f"wrote full ranked table to {args.csv}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="zeff")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_parser = subparsers.add_parser("scrape", help="scrape a season's per-game stats into the local db")
    scrape_parser.add_argument("--season", type=int, required=True, help="ending year, e.g. 2025 for 2024-25")
    scrape_parser.add_argument("--force-refresh", action="store_true", help="re-fetch even if cached")

    subparsers.add_parser("scrape-injuries", help="scrape the current NBA injury report")

    transactions_parser = subparsers.add_parser(
        "scrape-transactions", help="scrape a season's trade/transaction log"
    )
    transactions_parser.add_argument("--season", type=int, required=True, help="ending year, e.g. 2025 for 2024-25")
    transactions_parser.add_argument("--force-refresh", action="store_true", help="re-fetch even if cached")

    college_parser = subparsers.add_parser(
        "scrape-college",
        help="translate a player's final college season to an NBA-equivalent line and store it",
    )
    college_parser.add_argument("--player", type=str, required=True, help="e.g. \"Cooper Flagg\"")
    college_parser.add_argument(
        "--season", type=int, required=True,
        help="which season (ending year) this translated line should be ranked under",
    )
    college_parser.add_argument("--force-refresh", action="store_true", help="re-fetch even if cached")

    set_team_parser = subparsers.add_parser(
        "set-team",
        help="manually record a player's current team (for trades scrape-transactions couldn't auto-resolve)",
    )
    set_team_parser.add_argument("--player", type=str, required=True)
    set_team_parser.add_argument("--team", type=str, required=True, help="3-letter team abbreviation, e.g. TOR")

    adp_parser = subparsers.add_parser(
        "scrape-adp", help="scrape average draft position from hashtagbasketball.com"
    )
    adp_parser.add_argument("--force-refresh", action="store_true", help="re-fetch even if cached")

    rank_parser = subparsers.add_parser("rank", help="rank a season already scraped by 9-cat z-score")
    rank_parser.add_argument("--season", type=int, required=True)
    rank_parser.add_argument("--top", type=int, default=50)
    rank_parser.add_argument("--min-games", type=int, default=20)
    rank_parser.add_argument(
        "--pool-size", type=int, default=200,
        help="draft-pool-relative ranking: z-scores are computed against only the top N players "
             "(default 200, roughly a 12-team 9-cat league), not the whole league",
    )
    rank_parser.add_argument(
        "--projected", action="store_true",
        help="project forward instead of using raw single-season stats: an age curve applied to "
             "the latest season, plus a pace adjustment for players who changed teams (a "
             "multi-season blend and shooting-pct regression were tested and dropped -- they "
             "didn't beat a naive 'last season repeats' baseline in backtesting)",
    )
    rank_parser.add_argument(
        "--seasons-back", type=int, default=3,
        help="how many seasons (ending at --season) to average missed-games durability over",
    )
    rank_parser.add_argument("--csv", type=str, default=None, help="optional path to write the full ranked table")
    rank_parser.add_argument(
        "--show-adp", action="store_true",
        help="if `scrape-adp` has been run, join in average draft position for comparison "
             "(adp, vs_adp columns) -- real draft-market data, not another stats model",
    )

    args = parser.parse_args()
    if args.command == "scrape":
        _cmd_scrape(args)
    elif args.command == "scrape-injuries":
        _cmd_scrape_injuries(args)
    elif args.command == "scrape-transactions":
        _cmd_scrape_transactions(args)
    elif args.command == "scrape-college":
        _cmd_scrape_college(args)
    elif args.command == "set-team":
        _cmd_set_team(args)
    elif args.command == "scrape-adp":
        _cmd_scrape_adp(args)
    elif args.command == "rank":
        _cmd_rank(args)


if __name__ == "__main__":
    main()
