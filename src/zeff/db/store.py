"""SQLite storage for scraped player season stats, injuries, and transactions."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path("data") / "zeff.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text())
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """One-off column additions for local dbs created before this column existed.

    Not a general migration framework -- just enough to keep a solo dev's
    local `data/zeff.db` working across schema tweaks without deleting it.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(player_season_stats)")}
    for missing_column in {"usage_pct", "age"} - columns:
        conn.execute(f"ALTER TABLE player_season_stats ADD COLUMN {missing_column} REAL")
    if "data_source" not in columns:
        conn.execute(
            "ALTER TABLE player_season_stats ADD COLUMN data_source TEXT NOT NULL DEFAULT 'nba'"
        )
    conn.commit()


def slugify_name(name: str) -> str:
    """Turn a player name into a stable id, e.g. 'Nikola Jokić' -> 'nikola-jokic'."""
    ascii_name = name.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug


def upsert_season_stats(
    conn: sqlite3.Connection, season: int, stats: pd.DataFrame, data_source: str = "nba"
) -> None:
    """Insert or replace player_season_stats rows for a season.

    `stats` must have columns: name, team, age, games, minutes_per_game,
    pts, reb, ast, stl, blk, tov, fg_made, fg_att, fg_pct, ft_made, ft_att,
    ft_pct, threes_made, usage_pct.
    """
    rows = stats.copy()
    rows["player_id"] = rows["name"].map(slugify_name)

    conn.executemany(
        "INSERT OR REPLACE INTO players (player_id, name) VALUES (?, ?)",
        rows[["player_id", "name"]].drop_duplicates().itertuples(index=False),
    )

    columns = [
        "season", "player_id", "team", "age", "games", "minutes_per_game",
        "pts", "reb", "ast", "stl", "blk", "tov",
        "fg_made", "fg_att", "fg_pct", "ft_made", "ft_att", "ft_pct", "threes_made",
        "usage_pct", "data_source",
    ]
    rows["season"] = season
    rows["data_source"] = data_source
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT OR REPLACE INTO player_season_stats ({', '.join(columns)}) VALUES ({placeholders})",
        rows[columns].itertuples(index=False),
    )
    conn.commit()


def load_season_stats(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    query = """
        SELECT p.name, s.*
        FROM player_season_stats s
        JOIN players p ON p.player_id = s.player_id
        WHERE s.season = ?
    """
    return pd.read_sql_query(query, conn, params=(season,))


def load_seasons_stats(conn: sqlite3.Connection, seasons: list[int]) -> pd.DataFrame:
    """Like `load_season_stats`, but over several seasons at once (for
    lookback-window calculations like injury/durability history)."""
    placeholders = ", ".join("?" for _ in seasons)
    query = f"""
        SELECT p.name, s.*
        FROM player_season_stats s
        JOIN players p ON p.player_id = s.player_id
        WHERE s.season IN ({placeholders})
    """
    return pd.read_sql_query(query, conn, params=tuple(seasons))


def replace_injuries(conn: sqlite3.Connection, injuries: pd.DataFrame) -> None:
    """Replace the injuries snapshot wholesale -- ESPN's feed is current-state
    only, so a partial upsert would leave stale rows for recovered players.

    `injuries` must have columns: name, team, status, description, reported_at.
    """
    rows = injuries.copy()
    rows["player_id"] = rows["name"].map(slugify_name)

    conn.executemany(
        "INSERT OR REPLACE INTO players (player_id, name) VALUES (?, ?)",
        rows[["player_id", "name"]].drop_duplicates().itertuples(index=False),
    )

    conn.execute("DELETE FROM injuries")
    columns = ["player_id", "team", "status", "description", "reported_at"]
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO injuries ({', '.join(columns)}) VALUES ({placeholders})",
        rows[columns].itertuples(index=False),
    )
    conn.commit()


def load_injuries(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM injuries", conn)


def upsert_transactions(conn: sqlite3.Connection, season: int, transactions: pd.DataFrame) -> None:
    """`transactions` must have columns: date, description, is_trade."""
    rows = transactions.copy()
    rows["season"] = season
    columns = ["season", "date", "description", "is_trade"]
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT OR REPLACE INTO transactions ({', '.join(columns)}) VALUES ({placeholders})",
        rows[columns].itertuples(index=False),
    )
    conn.commit()


def load_transactions(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM transactions WHERE season = ?", conn, params=(season,)
    )


def upsert_trade_moves(conn: sqlite3.Connection, season: int, moves: pd.DataFrame) -> None:
    """`moves` must have columns: date (ISO yyyy-mm-dd, so it sorts correctly),
    player_name, new_team."""
    rows = moves.copy()
    rows["season"] = season
    columns = ["season", "date", "player_name", "new_team"]
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT OR REPLACE INTO trade_moves ({', '.join(columns)}) VALUES ({placeholders})",
        rows[columns].itertuples(index=False),
    )
    conn.commit()


def load_current_teams(conn: sqlite3.Connection) -> pd.Series:
    """Player's most recently auto-detected team from a simple 2-team trade,
    across every scraped season -- latest date wins. Indexed by player_id."""
    df = pd.read_sql_query("SELECT player_name, new_team, date FROM trade_moves", conn)
    if df.empty:
        return pd.Series(dtype=object)
    df["player_id"] = df["player_name"].map(slugify_name)
    latest = df.sort_values("date").groupby("player_id").tail(1)
    return latest.set_index("player_id")["new_team"]


def set_team_override(conn: sqlite3.Connection, player_name: str, team: str) -> None:
    player_id = slugify_name(player_name)
    conn.execute(
        "INSERT OR REPLACE INTO players (player_id, name) VALUES (?, ?)",
        (player_id, player_name),
    )
    conn.execute(
        "INSERT OR REPLACE INTO team_overrides (player_id, team) VALUES (?, ?)",
        (player_id, team),
    )
    conn.commit()


def load_team_overrides(conn: sqlite3.Connection) -> pd.Series:
    """Indexed by player_id."""
    df = pd.read_sql_query("SELECT player_id, team FROM team_overrides", conn)
    return df.set_index("player_id")["team"]


def replace_adp(conn: sqlite3.Connection, adp: pd.DataFrame) -> None:
    """Replace the ADP snapshot wholesale -- it reflects the current draft
    market, so a partial upsert would leave stale rows behind.

    `adp` must have columns: name, team, yahoo_adp, espn_adp, fantrax_adp, blend_adp.
    """
    rows = adp.copy()
    rows["player_id"] = rows["name"].map(slugify_name)

    conn.executemany(
        "INSERT OR REPLACE INTO players (player_id, name) VALUES (?, ?)",
        rows[["player_id", "name"]].drop_duplicates().itertuples(index=False),
    )

    conn.execute("DELETE FROM adp")
    columns = ["player_id", "team", "yahoo_adp", "espn_adp", "fantrax_adp", "blend_adp"]
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO adp ({', '.join(columns)}) VALUES ({placeholders})",
        rows[columns].itertuples(index=False),
    )
    conn.commit()


def load_adp(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM adp", conn)
