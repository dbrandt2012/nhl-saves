"""Data layer: caches NHL API responses as Parquet and builds processed DataFrames.

Raw cache layout (data/raw/):
  schedule/{date or "today"}.parquet
  game_log/goalie/{season}_{game_type}/{player_id}.parquet
  game_log/skater/{season}_{game_type}/{player_id}.parquet
  team_stats/{season}_{game_type}/{team_abbrev}.parquet
  goalie_stats/{season}_{game_type}.parquet

Processed layout (data/processed/):
  goalie_game_logs/{season}_{game_type}.parquet
  goalie_features/{season}_{game_type}.parquet
"""

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from nhl_saves.api import NHLClient

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CACHE_TTL_SECONDS = 3600

_client: NHLClient | None = None


def _get_client() -> NHLClient:
    global _client
    if _client is None:
        _client = NHLClient()
    return _client


def _is_fresh(path: Path, ttl: int = CACHE_TTL_SECONDS) -> bool:
    """Return True if path exists and was modified within ttl seconds."""
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < ttl


def _raw_path(*parts: str) -> Path:
    """Build a path under RAW_DIR, creating parent directories as needed."""
    path = RAW_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_or_fetch(
    path: Path,
    fetch_fn: Callable[[], list[dict]],
    extra_cols: dict | None = None,
    ttl: int = CACHE_TTL_SECONDS,
) -> pd.DataFrame:
    """Return cached Parquet if fresh, otherwise fetch, cache, and return."""
    if _is_fresh(path, ttl):
        return pd.read_parquet(path)

    records = fetch_fn()
    df = pd.DataFrame(records)
    if extra_cols:
        for col, val in extra_cols.items():
            df[col] = val
    df.to_parquet(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Raw layer
# ---------------------------------------------------------------------------


def fetch_schedule(
    date: str | None = None,
    *,
    client: NHLClient | None = None,
    ttl: int = CACHE_TTL_SECONDS,
) -> pd.DataFrame:
    """Return a flat DataFrame of games for a date (YYYY-MM-DD) or today.

    Schema: gameId, gameDate, homeTeam, awayTeam, venue, gameState.
    Cached at: data/raw/schedule/{date or "today"}.parquet
    """
    c = client or _get_client()
    label = date or "today"
    path = _raw_path("schedule", f"{label}.parquet")

    if _is_fresh(path, ttl):
        return pd.read_parquet(path)

    raw = c.get_schedule(date)
    rows = []
    for day in raw.get("gameWeek", []):
        for game in day.get("games", []):
            rows.append({
                "gameId": game.get("id"),
                "gameDate": day.get("date"),
                "homeTeam": game.get("homeTeam", {}).get("abbrev"),
                "awayTeam": game.get("awayTeam", {}).get("abbrev"),
                "venue": game.get("venue", {}).get("default"),
                "gameState": game.get("gameState"),
            })

    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def fetch_goalie_game_log(
    player_id: int,
    season: str,
    game_type: int = 2,
    *,
    client: NHLClient | None = None,
    ttl: int = CACHE_TTL_SECONDS,
) -> pd.DataFrame:
    """Return per-game log for one goalie.

    Schema: player_id, gameId, gameDate, teamAbbrev, opponentAbbrev,
            homeRoadFlag, decision, shotsAgainst, goalsAgainst, savePctg,
            shutouts, toi, gamesStarted.
    Cached at: data/raw/game_log/goalie/{season}_{game_type}/{player_id}.parquet
    """
    c = client or _get_client()
    path = _raw_path(
        "game_log", "goalie", f"{season}_{game_type}", f"{player_id}.parquet"
    )
    return _read_or_fetch(
        path,
        lambda: c.get_player_game_log(player_id, season, game_type),
        extra_cols={"player_id": player_id},
        ttl=ttl,
    )


def fetch_skater_game_log(
    player_id: int,
    season: str,
    game_type: int = 2,
    *,
    client: NHLClient | None = None,
    ttl: int = CACHE_TTL_SECONDS,
) -> pd.DataFrame:
    """Return per-game log for one skater.

    Schema: player_id, gameId, gameDate, teamAbbrev, opponentAbbrev,
            homeRoadFlag, shots, goals, assists, points, toi, plusMinus, pim.
    Cached at: data/raw/game_log/skater/{season}_{game_type}/{player_id}.parquet
    """
    c = client or _get_client()
    path = _raw_path(
        "game_log", "skater", f"{season}_{game_type}", f"{player_id}.parquet"
    )
    return _read_or_fetch(
        path,
        lambda: c.get_player_game_log(player_id, season, game_type),
        extra_cols={"player_id": player_id},
        ttl=ttl,
    )


def fetch_team_stats(
    team_abbrev: str,
    season: str,
    game_type: int = 2,
    *,
    client: NHLClient | None = None,
    ttl: int = CACHE_TTL_SECONDS,
) -> pd.DataFrame:
    """Return team-level stats as a single-row DataFrame.

    Schema: teamAbbrev, shotsForPerGame, shotsAgainstPerGame, goalsFor,
            goalsAgainst, wins, losses, overtimeLosses,
            powerPlayPercentage, penaltyKillPercentage.
    Cached at: data/raw/team_stats/{season}_{game_type}/{team_abbrev}.parquet
    """
    c = client or _get_client()
    path = _raw_path("team_stats", f"{season}_{game_type}", f"{team_abbrev}.parquet")

    # API returns a dict, not a list — wrap so _read_or_fetch gets list[dict]
    return _read_or_fetch(
        path,
        lambda: [c.get_team_stats(team_abbrev, season, game_type)],
        extra_cols={"teamAbbrev": team_abbrev},
        ttl=ttl,
    )


def fetch_goalie_stats(
    season: str,
    game_type: int = 2,
    *,
    client: NHLClient | None = None,
    ttl: int = CACHE_TTL_SECONDS,
) -> pd.DataFrame:
    """Return bulk season summary stats for all goalies.

    Paginates the Stats REST API in pages of 100.
    Cached at: data/raw/goalie_stats/{season}_{game_type}.parquet
    """
    c = client or _get_client()
    path = _raw_path("goalie_stats", f"{season}_{game_type}.parquet")

    if _is_fresh(path, ttl):
        return pd.read_parquet(path)

    all_records: list[dict] = []
    start = 0
    page_size = 100
    while True:
        page = c.get_goalie_stats(season, game_type, start=start, limit=page_size)
        if not page:
            break
        all_records.extend(page)
        start += page_size

    df = pd.DataFrame(all_records)
    df.to_parquet(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Processed layer
# ---------------------------------------------------------------------------


def build_goalie_game_logs(
    season: str,
    game_type: int = 2,
    player_ids: list[int] | None = None,
    *,
    client: NHLClient | None = None,
) -> pd.DataFrame:
    """Combine per-game logs for all goalies in a season into one DataFrame.

    If player_ids is None, derives them from fetch_goalie_stats() (playerId column).
    Written to: data/processed/goalie_game_logs/{season}_{game_type}.parquet
    """
    c = client or _get_client()

    if player_ids is None:
        bulk = fetch_goalie_stats(season, game_type, client=c)
        player_ids = bulk["playerId"].dropna().astype(int).tolist()

    frames = [
        fetch_goalie_game_log(pid, season, game_type, client=c)
        for pid in player_ids
    ]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    out_path = PROCESSED_DIR / "goalie_game_logs" / f"{season}_{game_type}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def build_goalie_features(
    season: str,
    game_type: int = 2,
    player_ids: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    *,
    client: NHLClient | None = None,
) -> pd.DataFrame:
    """Produce an analysis-ready feature DataFrame for goalie modelling.

    Enriches each game row with:
      - opponent_shotsForPerGame: opposing team's season avg shots per game
      - savePctg_rolling_{w}g / shotsAgainst_rolling_{w}g for each window

    Written to: data/processed/goalie_features/{season}_{game_type}.parquet
    """
    c = client or _get_client()
    windows = rolling_windows or [5, 10]

    df = build_goalie_game_logs(season, game_type, player_ids, client=c)
    if df.empty:
        return df

    # Attach opponent shot volume
    opponent_teams = df["opponentAbbrev"].dropna().unique().tolist()
    team_stats_map: dict[str, float] = {}
    for team in opponent_teams:
        ts = fetch_team_stats(team, season, game_type, client=c)
        if not ts.empty and "shotsForPerGame" in ts.columns:
            team_stats_map[team] = float(ts["shotsForPerGame"].iloc[0])

    df["opponent_shotsForPerGame"] = df["opponentAbbrev"].map(team_stats_map)

    # Rolling averages per player, sorted chronologically
    df = df.sort_values(["player_id", "gameDate"]).reset_index(drop=True)
    for w in windows:
        for stat in ("savePctg", "shotsAgainst"):
            col = f"{stat}_rolling_{w}g"
            df[col] = (
                df.groupby("player_id")[stat]
                .transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            )

    out_path = PROCESSED_DIR / "goalie_features" / f"{season}_{game_type}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
