# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                          # install/sync dependencies
uv run streamlit run src/nhl_saves/main.py       # run the app
uv run pytest                                    # run all tests
uv run pytest tests/test_foo.py::test_bar        # run a single test
uv run ruff check .                              # lint
uv run ruff format .                             # format
```

## Architecture

NHL Saves is a Streamlit data application that tracks NHL goalie save statistics historically and projects saves for upcoming games.

**Entry point:** `src/nhl_saves/main.py` — Streamlit app bootstrapped with `st.set_page_config`.

**Data flow (intended):** `requests` fetches NHL data from external APIs → `pandas` processes/stores it in `data/raw/` and `data/processed/` → `plotly` renders interactive charts inside the Streamlit UI.

**Package layout:** `src/nhl_saves/` (src layout). The `data/` directory is gitignored (only `.gitkeep` files are committed); populate it locally at runtime.

**Ruff config:** line length 88, rules `E`, `F`, `I` (errors, pyflakes, import sorting).

## NHL API (`src/nhl_saves/api.py`)

**API reference:** https://github.com/Zmalski/NHL-API-Reference/blob/main/README.md

`NHLClient` wraps two public NHL APIs (no auth required):

| Base URL | Used for |
|---|---|
| `https://api-web.nhle.com/v1` | Game logs, schedule, roster, team stats |
| `https://api.nhle.com/stats/rest/en` | Bulk goalie/skater stats with filtering |

**Key methods:**
- `get_schedule(date)` — league schedule; `date` as `"YYYY-MM-DD"` or `None` for today
- `get_team_roster(team, season)` → `{"goalies": [...], "skaters": [...]}`
- `get_goalie_leaders(season, category, limit)` — top goalies by stat category
- `get_goalie_stats(season)` — paginated bulk goalie stats via Stats REST API
- `get_player_game_log(player_id, season)` — per-game stats for any player (goalie or skater)
- `get_team_stats(team, season)` — team averages incl. `shotsForPerGame` / `shotsAgainstPerGame`

**Conventions:** Season format `YYYYYYYY` (e.g. `"20242025"`). Game type `2` = regular season, `3` = playoffs. Team codes are 3-letter abbreviations (e.g. `"TOR"`).

## Data Layer (`src/nhl_saves/store.py`)

Caches API responses as Parquet in `data/raw/` and builds analysis-ready DataFrames in `data/processed/`. Cache TTL is 1 hour (mtime-based).

**Raw cache functions** (fetch from API or return cached file):
- `fetch_schedule(date)` — flat game rows: `gameId`, `gameDate`, `homeTeam`, `awayTeam`, `venue`, `gameState`
- `fetch_goalie_game_log(player_id, season)` — per-game goalie stats; injects `player_id` column
- `fetch_skater_game_log(player_id, season)` — per-game skater stats; injects `player_id` column
- `fetch_team_stats(team_abbrev, season)` — single-row team averages incl. `shotsForPerGame`
- `fetch_goalie_stats(season)` — bulk all-goalie season summary, paginated

**Processed layer** (builds enriched DataFrames written to `data/processed/`):
- `build_goalie_game_logs(season, player_ids)` — all goalies combined into one DataFrame
- `build_goalie_features(season, player_ids, rolling_windows)` — adds `opponent_shotsForPerGame` and rolling averages (`savePctg_rolling_5g/10g`, `shotsAgainst_rolling_5g/10g`)

**File layout:**
```
data/raw/
  schedule/{date or "today"}.parquet
  game_log/goalie/{season}_{game_type}/{player_id}.parquet
  game_log/skater/{season}_{game_type}/{player_id}.parquet
  team_stats/{season}_{game_type}/{team_abbrev}.parquet
  goalie_stats/{season}_{game_type}.parquet
data/processed/
  goalie_game_logs/{season}_{game_type}.parquet
  goalie_features/{season}_{game_type}.parquet
```
