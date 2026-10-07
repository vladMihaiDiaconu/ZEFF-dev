CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    season INTEGER NOT NULL,
    player_id TEXT NOT NULL REFERENCES players(player_id),
    team TEXT,
    age REAL,
    games INTEGER,
    minutes_per_game REAL,
    pts REAL,
    reb REAL,
    ast REAL,
    stl REAL,
    blk REAL,
    tov REAL,
    fg_made REAL,
    fg_att REAL,
    fg_pct REAL,
    ft_made REAL,
    ft_att REAL,
    ft_pct REAL,
    threes_made REAL,
    usage_pct REAL,
    data_source TEXT NOT NULL DEFAULT 'nba',
    PRIMARY KEY (season, player_id)
);

CREATE TABLE IF NOT EXISTS injuries (
    player_id TEXT PRIMARY KEY REFERENCES players(player_id),
    team TEXT,
    status TEXT,
    description TEXT,
    reported_at TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    season INTEGER NOT NULL,
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    is_trade INTEGER NOT NULL,
    PRIMARY KEY (season, date, description)
);

CREATE TABLE IF NOT EXISTS trade_moves (
    season INTEGER NOT NULL,
    date TEXT NOT NULL,
    player_name TEXT NOT NULL,
    new_team TEXT NOT NULL,
    PRIMARY KEY (season, date, player_name)
);

CREATE TABLE IF NOT EXISTS team_overrides (
    player_id TEXT PRIMARY KEY REFERENCES players(player_id),
    team TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS adp (
    player_id TEXT PRIMARY KEY REFERENCES players(player_id),
    team TEXT,
    yahoo_adp REAL,
    espn_adp REAL,
    fantrax_adp REAL,
    blend_adp REAL
);
