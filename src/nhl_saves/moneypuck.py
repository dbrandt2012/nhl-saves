"""Download and cache MoneyPuck CSV data for goalie modeling.

Attribution: MoneyPuck.com — data must be credited when displaying derived results.

Base URL: https://moneypuck.com/moneypuck/playerData
Season format: starting year integer — e.g., 2024 = 2024-25 season.
"""

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

MP_BASE = "https://moneypuck.com/moneypuck/playerData"
MP_TTL = 86400  # 24h cache for historical seasons
MP_BIOS_TTL = 86400 * 7  # 7-day cache for player bios

RAW_DIR = Path("data/raw")


def _mp_year(nhl_season: str) -> int:
    """Convert NHL season format to MoneyPuck year. '20242025' → 2024"""
    return int(nhl_season[:4])


def _mp_path(*parts: str) -> Path:
    """Build a path under data/raw/moneypuck/, creating parents as needed."""
    path = RAW_DIR / "moneypuck" / Path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _is_fresh(path: Path, ttl: int) -> bool:
    """Return True if path exists and was modified within ttl seconds."""
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < ttl


def _mp_fetch(url: str, path: Path, ttl: int = MP_TTL) -> pd.DataFrame:
    """Download CSV from url if cache is stale; return cached Parquet otherwise."""
    if _is_fresh(path, ttl):
        return pd.read_parquet(path)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, timeout=30, headers=headers)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    df.to_parquet(path, index=False)
    return df


def fetch_mp_goalies(season: str, game_type: str = "regular") -> pd.DataFrame:
    """Season summary — all goalies, all situations (one row per goalie per situation).

    Args:
        season: NHL season string (e.g. "20242025") or MP year string (e.g. "2024").
        game_type: "regular" or "playoffs".

    Columns include: playerId, name, team, situation, games_played, icetime, xGoals,
        goals, ongoal, rebounds, lowDangerShots, mediumDangerShots, highDangerShots,
        blocked_shot_attempts, unblocked_shot_attempts, penalties, penalityMinutes.

    Cache: data/raw/moneypuck/goalies/{mp_year}.parquet
    """
    mp_year = _mp_year(season) if len(season) == 8 else int(season)
    path = _mp_path("goalies", f"{mp_year}.parquet")
    url = f"{MP_BASE}/seasonSummary/{mp_year}/{game_type}/goalies.csv"
    return _mp_fetch(url, path)


def fetch_mp_teams(season: str, game_type: str = "regular") -> pd.DataFrame:
    """Season summary — all teams, all situations (one row per team per situation).

    Args:
        season: NHL season string (e.g. "20242025") or MP year string (e.g. "2024").
        game_type: "regular" or "playoffs".

    Columns include: team, situation, games_played, corsiPercentage, fenwickPercentage,
        xGoalsPercentage, shotsOnGoalFor, shotsOnGoalAgainst, goalsFor, goalsAgainst,
        highDangerShotsFor, highDangerShotsAgainst.

    Cache: data/raw/moneypuck/teams/{mp_year}.parquet
    """
    mp_year = _mp_year(season) if len(season) == 8 else int(season)
    path = _mp_path("teams", f"{mp_year}.parquet")
    url = f"{MP_BASE}/seasonSummary/{mp_year}/{game_type}/teams.csv"
    return _mp_fetch(url, path)


def fetch_mp_player_bios() -> pd.DataFrame:
    """Player bios — birthDate (YYYY-MM-DD), nationality, position for all players.

    Columns: playerId, name, position, team, birthDate, weight, height,
        nationality, shootsCatches, primaryNumber, primaryPosition.

    Cache: data/raw/moneypuck/playerBios.parquet (TTL: 7 days)
    """
    path = _mp_path("playerBios.parquet")
    url = f"{MP_BASE}/playerBios/allPlayersLookup.csv"
    return _mp_fetch(url, path, ttl=MP_BIOS_TTL)


def fetch_mp_goalie_career(player_id: int, game_type: str = "regular") -> pd.DataFrame:
    """Per-game career data for one goalie across all seasons.

    gameDate column is in YYYYMMDD integer format — callers must parse with:
        pd.to_datetime(df['gameDate'].astype(str), format='%Y%m%d')

    Team columns: playerTeam, opposingTeam (not teamAbbrev/opponentAbbrev).
    Home/away: home_or_away column with values "HOME" / "AWAY".
    Season: starting year integer (e.g., 2024 for 2024-25).

    Columns include: playerId, season, name, gameId, playerTeam, opposingTeam,
        home_or_away, gameDate, position, situation, icetime, xGoals, goals,
        ongoal, rebounds, lowDangerShots, mediumDangerShots, highDangerShots,
        blocked_shot_attempts, unblocked_shot_attempts, penalties, penalityMinutes.

    Returns empty DataFrame if player not found in MoneyPuck.

    Cache: data/raw/moneypuck/careers/goalies/{player_id}.parquet
    """
    path = _mp_path("careers", "goalies", f"{player_id}.parquet")
    url = f"{MP_BASE}/careers/gameByGame/{game_type}/goalies/{player_id}.csv"
    try:
        return _mp_fetch(url, path)
    except requests.HTTPError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
