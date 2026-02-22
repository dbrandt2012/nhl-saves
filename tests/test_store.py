"""Unit tests for the data layer (store.py). No real network or filesystem I/O."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nhl_saves.store import (
    _is_fresh,
    build_goalie_features,
    build_goalie_game_logs,
    fetch_goalie_game_log,
    fetch_goalie_stats,
    fetch_team_stats,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

GOALIE_GAME = {
    "gameId": 2024020001,
    "gameDate": "2024-10-10",
    "teamAbbrev": "TOR",
    "opponentAbbrev": "BOS",
    "homeRoadFlag": "H",
    "decision": "W",
    "shotsAgainst": 30,
    "goalsAgainst": 2,
    "savePctg": 0.933,
    "shutouts": 0,
    "toi": "60:00",
    "gamesStarted": 1,
}

TEAM_STATS_DICT = {
    "shotsForPerGame": 31.5,
    "shotsAgainstPerGame": 28.0,
    "goalsFor": 150,
    "goalsAgainst": 120,
    "wins": 30,
    "losses": 15,
    "overtimeLosses": 5,
    "powerPlayPercentage": 22.5,
    "penaltyKillPercentage": 81.0,
}


def _make_goalie_df(player_id: int, n: int = 3) -> pd.DataFrame:
    rows = [
        {**GOALIE_GAME, "player_id": player_id, "gameDate": f"2024-10-{10 + i:02d}"}
        for i in range(n)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _is_fresh
# ---------------------------------------------------------------------------


def test_is_fresh_missing():
    assert _is_fresh(Path("nonexistent_file_xyz.parquet")) is False


def test_is_fresh_stale():
    mock_stat = MagicMock()
    mock_stat.st_mtime = time.time() - 7200  # 2 hours ago
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "stat", return_value=mock_stat),
    ):
        assert _is_fresh(Path("some.parquet"), ttl=3600) is False


def test_is_fresh_fresh():
    mock_stat = MagicMock()
    mock_stat.st_mtime = time.time() - 10  # 10 seconds ago
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "stat", return_value=mock_stat),
    ):
        assert _is_fresh(Path("some.parquet"), ttl=3600) is True


# ---------------------------------------------------------------------------
# fetch_goalie_game_log — cache hit and miss
# ---------------------------------------------------------------------------


def test_fetch_goalie_log_cache_hit():
    expected = _make_goalie_df(8480045)
    mock_client = MagicMock()
    with (
        patch("nhl_saves.store._is_fresh", return_value=True),
        patch("nhl_saves.store._raw_path", return_value=Path("dummy.parquet")),
        patch("pandas.read_parquet", return_value=expected) as mock_read,
    ):
        result = fetch_goalie_game_log(8480045, "20242025", client=mock_client)
        mock_read.assert_called_once()
        mock_client.get_player_game_log.assert_not_called()
    pd.testing.assert_frame_equal(result, expected)


def test_fetch_goalie_log_cache_miss():
    mock_client = MagicMock()
    mock_client.get_player_game_log.return_value = [GOALIE_GAME]

    with (
        patch("nhl_saves.store._is_fresh", return_value=False),
        patch("nhl_saves.store._raw_path", return_value=Path("dummy.parquet")),
        patch("pandas.DataFrame.to_parquet"),
    ):
        result = fetch_goalie_game_log(8480045, "20242025", client=mock_client)

    mock_client.get_player_game_log.assert_called_once_with(8480045, "20242025", 2)
    assert "player_id" in result.columns
    assert result["player_id"].iloc[0] == 8480045
    assert result["shotsAgainst"].iloc[0] == 30


# ---------------------------------------------------------------------------
# fetch_team_stats — API returns a dict, must become a single-row DataFrame
# ---------------------------------------------------------------------------


def test_fetch_team_stats_wraps_dict():
    mock_client = MagicMock()
    mock_client.get_team_stats.return_value = TEAM_STATS_DICT

    with (
        patch("nhl_saves.store._is_fresh", return_value=False),
        patch("nhl_saves.store._raw_path", return_value=Path("dummy.parquet")),
        patch("pandas.DataFrame.to_parquet"),
    ):
        result = fetch_team_stats("BOS", "20242025", client=mock_client)

    assert len(result) == 1
    assert "teamAbbrev" in result.columns
    assert result["teamAbbrev"].iloc[0] == "BOS"
    assert result["shotsForPerGame"].iloc[0] == 31.5


# ---------------------------------------------------------------------------
# fetch_goalie_stats — pagination
# ---------------------------------------------------------------------------


def test_fetch_goalie_stats_paginates():
    page1 = [{"playerId": i, "wins": 1} for i in range(100)]
    page2 = []  # signals end of pages

    mock_client = MagicMock()
    mock_client.get_goalie_stats.side_effect = [page1, page2]

    with (
        patch("nhl_saves.store._is_fresh", return_value=False),
        patch("nhl_saves.store._raw_path", return_value=Path("dummy.parquet")),
        patch("pandas.DataFrame.to_parquet"),
    ):
        result = fetch_goalie_stats("20242025", client=mock_client)

    assert mock_client.get_goalie_stats.call_count == 2
    assert len(result) == 100


# ---------------------------------------------------------------------------
# build_goalie_game_logs — combines multiple players
# ---------------------------------------------------------------------------


def test_build_game_logs_combines():
    df1 = _make_goalie_df(1001, n=3)
    df2 = _make_goalie_df(1002, n=3)

    with (
        patch("nhl_saves.store.fetch_goalie_stats") as mock_bulk,
        patch("nhl_saves.store.fetch_goalie_game_log", side_effect=[df1, df2]),
        patch("pandas.DataFrame.to_parquet"),
        patch.object(Path, "mkdir"),
    ):
        mock_bulk.return_value = pd.DataFrame({"playerId": [1001, 1002]})
        result = build_goalie_game_logs("20242025")

    assert len(result) == 6
    assert set(result["player_id"].unique()) == {1001, 1002}


# ---------------------------------------------------------------------------
# build_goalie_features — opponent context join
# ---------------------------------------------------------------------------


def test_build_features_opponent_context():
    logs = _make_goalie_df(1001, n=2)
    logs["opponentAbbrev"] = "BOS"
    team_stats_df = pd.DataFrame([{**TEAM_STATS_DICT, "teamAbbrev": "BOS"}])

    with (
        patch("nhl_saves.store.build_goalie_game_logs", return_value=logs),
        patch("nhl_saves.store.fetch_team_stats", return_value=team_stats_df),
        patch("pandas.DataFrame.to_parquet"),
        patch.object(Path, "mkdir"),
    ):
        result = build_goalie_features("20242025", player_ids=[1001])

    assert "opponent_shotsForPerGame" in result.columns
    assert (result["opponent_shotsForPerGame"] == 31.5).all()


# ---------------------------------------------------------------------------
# build_goalie_features — rolling averages
# ---------------------------------------------------------------------------


def test_build_features_rolling_avgs():
    # 6 games with known savePctg values for one player
    save_pcts = [0.900, 0.920, 0.880, 0.950, 0.910, 0.940]
    rows = [
        {
            **GOALIE_GAME,
            "player_id": 1001,
            "gameDate": f"2024-10-{10 + i:02d}",
            "savePctg": sp,
            "shotsAgainst": 30,
            "opponentAbbrev": "BOS",
        }
        for i, sp in enumerate(save_pcts)
    ]
    logs = pd.DataFrame(rows)
    team_stats_df = pd.DataFrame([{**TEAM_STATS_DICT, "teamAbbrev": "BOS"}])

    with (
        patch("nhl_saves.store.build_goalie_game_logs", return_value=logs),
        patch("nhl_saves.store.fetch_team_stats", return_value=team_stats_df),
        patch("pandas.DataFrame.to_parquet"),
        patch.object(Path, "mkdir"),
    ):
        result = build_goalie_features(
            "20242025", player_ids=[1001], rolling_windows=[5]
        )

    result = result.sort_values("gameDate").reset_index(drop=True)
    expected_rolling5_game6 = sum(save_pcts[1:6]) / 5  # last 5 of games 2–6
    assert "savePctg_rolling_5g" in result.columns
    assert result["savePctg_rolling_5g"].iloc[5] == pytest.approx(
        expected_rolling5_game6, rel=1e-5
    )
