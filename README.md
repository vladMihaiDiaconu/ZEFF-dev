# ZEFF — Z-score Estimator for Fantasy

A helper for drafting NBA fantasy teams: ranks players by z-score across the
standard 9 roto/H2H categories (PTS, REB, AST, STL, BLK, 3PM, FG%, FT%, TO),
computed from historical season stats scraped from stats.nba.com. Z-scores
are computed draft-pool-relative (top `--pool-size` players, default 200,
roughly a 12-team league), not against the whole NBA. Enriched with current
injury status, recent trades, usage%, a composite injury risk score, a
simple forward-looking projection, and real draft-market ADP for comparison.

stats.nba.com and Basketball-Reference both block plain HTTP clients at the
TLS-fingerprint level, so requests go through `curl_cffi` (Chrome
impersonation) rather than the plain `requests` library.

## Setup

```
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
```

## Usage

Scrape a season's per-game stats (cached locally, so re-running is cheap).
For `injury_risk` to have any durability history to work with, scrape a few
past seasons too:

```
python -m zeff.cli scrape --season 2026
python -m zeff.cli scrape --season 2025
python -m zeff.cli scrape --season 2024
```

Current injury snapshot, this season's trade log, and draft-market ADP:

```
python -m zeff.cli scrape-injuries
python -m zeff.cli scrape-transactions --season 2026
python -m zeff.cli scrape-adp
```

Rank players (retrospective, based on real stats already played):

```
python -m zeff.cli rank --season 2026 --top 50
```

Or projected forward (games-weighted blend of the last two seasons, age
curve, plus a pace adjustment for trades), with the ADP comparison joined
in:

```
python -m zeff.cli rank --season 2026 --projected --show-adp --top 50
```

## Columns

- `total_z` / `z_*` — the 9-cat z-score ranking (the core of the tool),
  computed against the top `--pool-size` players only (default 200).
- `usage_pct` — from stats.nba.com's Advanced stats.
- `injury_status` — current ESPN injury snapshot, if any.
- `recently_traded` — whether the player appears in an `is_trade` entry in
  this season's Basketball-Reference transaction log.
- `injury_risk` — composite score (roughly 0-3, higher = more expected
  value at risk): age, a missed-games *durability proxy* averaged over
  `--seasons-back` seasons (default 3), current injury severity, and
  on-court role (minutes/game). See `src/zeff/zscore/injury_risk.py` for
  the exact formula. **Not** a count of real injury diagnoses — the
  standard source for that (Pro Sports Transactions) blocks automated
  access even with a real headless browser, so this leans on games-missed
  history instead.
- `adp` / `vs_adp` (with `--show-adp`) — blended average draft position
  from hashtagbasketball.com (real Yahoo/ESPN/Fantrax draft data, not
  another stats model). `vs_adp = adp - our rank`: a large positive number
  is a player we value well ahead of where the market drafts them (a
  potential value pick, or a red flag the market knows about that box
  scores don't); a large negative number is the reverse.

### `--projected`

Games-weighted blend of the last two seasons, adjusted with an age curve,
plus a pace adjustment for players who changed teams. A full season
dominates its own blend almost entirely; only a short, injury-limited
season pulls in meaningful weight from the one before it.

Backtested against a naive "assume last season repeats" baseline (Spearman
rank correlation, top-200 pool) across three season transitions. The age
curve constants were chosen via a grid search over peak_age/youth_growth/
decline on the first two transitions only -- the literal optimum there was
close to (peak_age=24, growth=0.0275, decline=0.0075), and the shipped
round numbers (peak_age=25, growth=0.025, decline=0.0) score within noise
of it, so those were kept. 2024-25 → 2025-26 was held out of that search
entirely and only scored afterward:

| Transition | Naive | Age curve only | Age curve + blend (default) |
| --- | --- | --- | --- |
| 2022-23 → 2023-24 | 0.754 | 0.769 | 0.784 |
| 2023-24 → 2024-25 | 0.702 | 0.736 | 0.758 |
| 2024-25 → 2025-26 | 0.681 | 0.712 | 0.732 |

The age curve + blend combination beats both the naive baseline and the
age-curve-only version in every transition tested, including 2024-25 →
2025-26, which never factored into picking the constants. A fixed-ratio
multi-season blend (`blend_seasons`), shooting-pct
regression to the mean (`regress_shooting_pct`), and a momentum/trend
signal were also tried and left out — none beat the baseline out of
sample. The first two are still in `src/zeff/zscore/projection.py` if
that's ever worth revisiting. Trade impact is deliberately pace-only, not
a usage/role guess — see that module for why.

## Known limitations

- Player name matching across sources (stats.nba.com / ESPN / Basketball-
  Reference / hashtagbasketball.com) is exact-match after slugifying (e.g.
  "Nikola Jokić" -> "nikola-jokic"). A suffix or accent mismatch between
  sources could silently miss a join. A few notable players (e.g. Jokić,
  Dončić) were observed missing from HashtagBasketball's ADP table for
  reasons not yet diagnosed — not a join bug on our end, but worth knowing
  `adp`/`vs_adp` can be blank for real stars.
- Trades are raw transaction text + date + an `is_trade` flag; only simple
  2-team trades get auto-resolved into a structured team-change (via
  `data-attr-from`/`data-attr-to` on Basketball-Reference's transaction
  page). Complex trades need `set-team --player NAME --team ABC` manually.
- `--projected` is a simple, backtested-but-modest heuristic, not a
  statistical model with real error bars. Treat it as a tiebreaker, not a
  forecast.

## Status

Z-score engine (draft-pool-relative), injuries/trades/usage% enrichment,
composite injury risk, a backtested `--projected` mode, and an ADP
comparison are built. A live draft-day UI (track picks, re-rank by
category need) is the natural next step.
