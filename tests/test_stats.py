"""Unit tests for stats.py — pure functions, no mocking required."""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from nhl_saves.stats import (
    days_since_last_game,
    goalie_report,
    goalie_vs_opponent_sog_season,
    mean_range_summary,
    opponent_goal_rate_stats,
    opponent_sog_stats,
    percentile_summary,
    range_summary,
    save_pct_stats,
    sog_allowed_stats,
    team_save_pct_stats,
    team_sog_stats,
    vs_opponent_history,
    vs_opponent_sog_season,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_goalie_log(rows: list[dict]) -> pd.DataFrame:
    """Build a goalie game log DataFrame from a list of row dicts."""
    defaults = {
        "gameId": 1,
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
        "player_id": 1001,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _season_log() -> pd.DataFrame:
    """Ten-game log with known values for a single goalie."""
    return _make_goalie_log(
        [
            {
                "gameId": i,
                "gameDate": f"2024-10-{10 + i:02d}",
                "shotsAgainst": 20 + i,
                "goalsAgainst": i % 3,
                "savePctg": round(1 - (i % 3) / (20 + i), 4),
                "opponentAbbrev": "BOS" if i <= 5 else "MTL",
            }
            for i in range(1, 11)
        ]
    )


# ---------------------------------------------------------------------------
# percentile_summary
# ---------------------------------------------------------------------------


def test_percentile_summary_basic():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = percentile_summary(s)
    assert result["median"] == pytest.approx(30.0)
    assert result["p25"] == pytest.approx(20.0)
    assert result["p75"] == pytest.approx(40.0)
    assert result["n"] == 5


def test_percentile_summary_drops_nan():
    s = pd.Series([10.0, float("nan"), 30.0])
    result = percentile_summary(s)
    assert result["n"] == 2
    assert result["median"] == pytest.approx(20.0)


def test_percentile_summary_empty():
    result = percentile_summary(pd.Series([], dtype=float))
    assert result["n"] == 0
    assert math.isnan(result["median"])


# ---------------------------------------------------------------------------
# range_summary
# ---------------------------------------------------------------------------


def test_range_summary_basic():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = range_summary(s)
    assert result["mean"] == pytest.approx(30.0)
    assert result["min"] == pytest.approx(10.0)
    assert result["max"] == pytest.approx(50.0)
    assert result["n"] == 5


def test_range_summary_drops_nan():
    s = pd.Series([10.0, float("nan"), 30.0])
    result = range_summary(s)
    assert result["n"] == 2
    assert result["min"] == pytest.approx(10.0)
    assert result["max"] == pytest.approx(30.0)


def test_range_summary_empty():
    result = range_summary(pd.Series([], dtype=float))
    assert result["n"] == 0
    assert math.isnan(result["mean"])
    assert math.isnan(result["min"])
    assert math.isnan(result["max"])


# ---------------------------------------------------------------------------
# mean_range_summary
# ---------------------------------------------------------------------------


def test_mean_range_summary_basic():
    s = pd.Series([10.0, 20.0, 30.0])
    result = mean_range_summary(s)
    assert result["mean"] == pytest.approx(20.0)
    assert result["min"] == pytest.approx(10.0)
    assert result["max"] == pytest.approx(30.0)
    assert result["n"] == 3


def test_mean_range_summary_empty():
    result = mean_range_summary(pd.Series([], dtype=float))
    assert result["n"] == 0
    assert math.isnan(result["mean"])


# ---------------------------------------------------------------------------
# Stat 1: sog_allowed_stats
# ---------------------------------------------------------------------------


def _all_logs_for_team(team: str, shots_per_game: list[int]) -> pd.DataFrame:
    """Combined goalie logs where all games belong to `team` on defence."""
    rows = [
        {
            "gameId": 5000 + i,
            "gameDate": f"2024-10-{10 + i:02d}",
            "teamAbbrev": team,
            "opponentAbbrev": "VAN",
            "shotsAgainst": s,
            "player_id": 8000 + i,
            "gamesStarted": 1,
            "savePctg": 0.920,
            "goalsAgainst": 2,
        }
        for i, s in enumerate(shots_per_game)
    ]
    return pd.DataFrame(rows)


def test_sog_allowed_goalie_season_count():
    log = _season_log()
    all_logs = log.copy()
    result = sog_allowed_stats(log, all_logs)
    assert result["goalie"]["season"]["n"] == 10


def test_sog_allowed_goalie_last_n_count():
    log = _season_log()
    all_logs = log.copy()
    result = sog_allowed_stats(log, all_logs, last_n=5)
    assert result["goalie"]["last_n"]["n"] == 5


def test_sog_allowed_goalie_last_n_has_range_keys():
    log = _season_log()
    result = sog_allowed_stats(log, log)
    assert "min" in result["goalie"]["last_n"]
    assert "max" in result["goalie"]["last_n"]


def test_sog_allowed_goalie_excludes_non_starts():
    log = _make_goalie_log(
        [
            {
                "gameId": 1,
                "gameDate": "2024-10-10",
                "shotsAgainst": 30,
                "gamesStarted": 1,
            },
            {
                "gameId": 2,
                "gameDate": "2024-10-11",
                "shotsAgainst": 5,
                "gamesStarted": 0,
            },
        ]
    )
    result = sog_allowed_stats(log, log)
    assert result["goalie"]["season"]["n"] == 1
    assert result["goalie"]["season"]["median"] == pytest.approx(30.0)


def test_sog_allowed_goalie_last_n_uses_most_recent():
    log = _make_goalie_log(
        [
            {"gameId": i, "gameDate": f"2024-10-{10 + i:02d}", "shotsAgainst": i * 10}
            for i in range(1, 7)
        ]
    )
    result = sog_allowed_stats(log, log, last_n=3)
    # Games 4, 5, 6 → shots 40, 50, 60 → mean 50
    assert result["goalie"]["last_n"]["mean"] == pytest.approx(50.0)


def test_sog_allowed_team_aggregates_all_goalies():
    """Team split should include games started by other goalies on the team."""
    goalie_log = _make_goalie_log(
        [
            {
                "gameId": i,
                "gameDate": f"2024-10-{10 + i:02d}",
                "teamAbbrev": "TOR",
                "shotsAgainst": 30,
            }
            for i in range(1, 4)
        ]
    )
    other_goalie = _make_goalie_log(
        [
            {
                "gameId": i + 10,
                "gameDate": f"2024-10-{14 + i:02d}",
                "teamAbbrev": "TOR",
                "shotsAgainst": 25,
                "player_id": 9999,
            }
            for i in range(1, 3)
        ]
    )
    all_logs = pd.concat([goalie_log, other_goalie], ignore_index=True)
    result = sog_allowed_stats(goalie_log, all_logs)
    assert result["team"]["season"]["n"] == 5
    assert result["goalie"]["season"]["n"] == 3


def test_sog_allowed_team_sums_multi_goalie_game():
    """If two goalies played in the same game, team total should be summed."""
    goalie_log = _make_goalie_log(
        [
            {
                "gameId": 1,
                "gameDate": "2024-10-10",
                "teamAbbrev": "TOR",
                "shotsAgainst": 20,
                "gamesStarted": 1,
            },
        ]
    )
    backup = _make_goalie_log(
        [
            {
                "gameId": 1,
                "gameDate": "2024-10-10",
                "teamAbbrev": "TOR",
                "shotsAgainst": 10,
                "gamesStarted": 0,
                "player_id": 9999,
            },
        ]
    )
    all_logs = pd.concat([goalie_log, backup], ignore_index=True)
    result = sog_allowed_stats(goalie_log, all_logs)
    assert result["team"]["season"]["n"] == 1
    assert result["team"]["season"]["median"] == pytest.approx(30.0)


def test_sog_allowed_team_last_n_uses_most_recent():
    all_logs = _all_logs_for_team("TOR", [10, 20, 30, 40, 50, 60])
    result = sog_allowed_stats(all_logs.head(1), all_logs, last_n=3)
    # Most recent 3 team games: 40, 50, 60 → mean 50
    assert result["team"]["last_n"]["mean"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Stat 2: opponent_sog_stats
# ---------------------------------------------------------------------------


def _make_all_logs_for_opponent(
    opponent: str, shots_per_game: list[int]
) -> pd.DataFrame:
    """Create combined goalie logs where `opponent` generated shots in each game."""
    rows = [
        {
            "gameId": 2000 + i,
            "gameDate": f"2024-10-{10 + i:02d}",
            "opponentAbbrev": opponent,
            "shotsAgainst": s,
            "player_id": 9000 + i,
            "gamesStarted": 1,
            "savePctg": 0.920,
            "goalsAgainst": 2,
            "teamAbbrev": "TOR",
        }
        for i, s in enumerate(shots_per_game)
    ]
    return pd.DataFrame(rows)


def test_opponent_sog_season_median():
    logs = _make_all_logs_for_opponent("BOS", [28, 32, 25, 35, 30, 27, 33])
    result = opponent_sog_stats(logs, "BOS")
    expected_median = pd.Series([28, 32, 25, 35, 30, 27, 33]).median()
    assert result["season"]["median"] == pytest.approx(expected_median)


def test_opponent_sog_last_n_uses_most_recent():
    logs = _make_all_logs_for_opponent("BOS", [20, 25, 30, 35, 40])
    result = opponent_sog_stats(logs, "BOS", last_n=3)
    # Most recent 3 games: shots 30, 35, 40 → mean 35
    assert result["last_n"]["mean"] == pytest.approx(35.0)
    assert result["last_n"]["n"] == 3


def test_opponent_sog_last_n_has_range_keys():
    logs = _make_all_logs_for_opponent("BOS", [20, 25, 30])
    result = opponent_sog_stats(logs, "BOS")
    assert "min" in result["last_n"]
    assert "max" in result["last_n"]


def test_opponent_sog_sums_multi_goalie_game():
    """When two goalies played in the same game, shots should be summed."""
    logs = pd.DataFrame(
        [
            {
                "gameId": 3001,
                "gameDate": "2024-10-10",
                "opponentAbbrev": "BOS",
                "shotsAgainst": 20,
                "player_id": 1,
                "gamesStarted": 1,
                "savePctg": 0.900,
                "goalsAgainst": 2,
                "teamAbbrev": "TOR",
            },
            {
                "gameId": 3001,
                "gameDate": "2024-10-10",
                "opponentAbbrev": "BOS",
                "shotsAgainst": 15,
                "player_id": 2,
                "gamesStarted": 0,
                "savePctg": 0.867,
                "goalsAgainst": 2,
                "teamAbbrev": "TOR",
            },
        ]
    )
    result = opponent_sog_stats(logs, "BOS")
    assert result["season"]["n"] == 1
    assert result["season"]["median"] == pytest.approx(35.0)


def test_opponent_sog_unknown_team_returns_nan():
    logs = _make_all_logs_for_opponent("BOS", [30, 32])
    result = opponent_sog_stats(logs, "VAN")
    assert result["season"]["n"] == 0
    assert math.isnan(result["season"]["median"])


# ---------------------------------------------------------------------------
# Stat 3: save_pct_stats  (values are now percentages: 0–100)
# ---------------------------------------------------------------------------


def test_save_pct_season_values():
    log = _make_goalie_log(
        [
            {"gameId": i, "gameDate": f"2024-10-{10 + i:02d}", "savePctg": v}
            for i, v in enumerate([0.900, 0.920, 0.930, 0.910, 0.950])
        ]
    )
    result = save_pct_stats(log)
    # Values now ×100
    assert result["season"]["median"] == pytest.approx(92.0)
    assert result["season"]["n"] == 5


def test_save_pct_last_n():
    log = _make_goalie_log(
        [
            {
                "gameId": i,
                "gameDate": f"2024-10-{10 + i:02d}",
                "savePctg": 0.880 + i * 0.01,
            }
            for i in range(8)
        ]
    )
    result = save_pct_stats(log, last_n=3)
    assert result["last_n"]["n"] == 3
    # Most recent 3: indices 5,6,7 → 0.930, 0.940, 0.950 → mean 0.940 → ×100 = 94.0
    assert result["last_n"]["mean"] == pytest.approx(94.0)


def test_save_pct_last_n_has_range_keys():
    log = _make_goalie_log(
        [
            {"gameId": i, "gameDate": f"2024-10-{10 + i:02d}", "savePctg": 0.920}
            for i in range(5)
        ]
    )
    result = save_pct_stats(log)
    assert "min" in result["last_n"]
    assert "max" in result["last_n"]


# ---------------------------------------------------------------------------
# Stat 4: vs_opponent_history
# ---------------------------------------------------------------------------


def test_vs_opponent_filters_correctly():
    log = _season_log()  # BOS games: i=1..5, MTL games: i=6..10
    result = vs_opponent_history(log, "BOS")
    if "opponentAbbrev" in result.columns:
        assert (result["opponentAbbrev"] == "BOS").all()
    assert len(result) == 5


def test_vs_opponent_sorted_most_recent_first():
    log = _season_log()
    result = vs_opponent_history(log, "BOS")
    dates = result["gameDate"].tolist()
    assert dates == sorted(dates, reverse=True)


def test_vs_opponent_no_games_returns_empty():
    log = _season_log()
    result = vs_opponent_history(log, "VAN")
    assert len(result) == 0


def test_vs_opponent_columns():
    log = _season_log()
    result = vs_opponent_history(log, "BOS")
    expected_cols = {
        "gameDate",
        "homeRoadFlag",
        "decision",
        "shotsAgainst",
        "goalsAgainst",
        "savePctg",
        "toi",
    }
    assert expected_cols.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Stat 5: opponent_goal_rate_stats  (values are now percentages: 0–100)
# ---------------------------------------------------------------------------


def test_opponent_goal_rate_computed_from_raw_counts():
    logs = _make_all_logs_for_opponent("BOS", [30, 25])
    logs["goalsAgainst"] = [3, 2]
    result = opponent_goal_rate_stats(logs, "BOS")
    # Game 0: 3/30=10.0%, Game 1: 2/25=8.0% → median 9.0%
    assert result["season"]["median"] == pytest.approx(9.0)


def test_opponent_goal_rate_excludes_zero_shot_games():
    logs = pd.DataFrame(
        [
            {
                "gameId": 1,
                "gameDate": "2024-10-10",
                "opponentAbbrev": "BOS",
                "shotsAgainst": 0,
                "goalsAgainst": 0,
                "player_id": 1,
                "gamesStarted": 1,
                "savePctg": 1.0,
                "teamAbbrev": "TOR",
            },
            {
                "gameId": 2,
                "gameDate": "2024-10-11",
                "opponentAbbrev": "BOS",
                "shotsAgainst": 30,
                "goalsAgainst": 3,
                "player_id": 1,
                "gamesStarted": 1,
                "savePctg": 0.900,
                "teamAbbrev": "TOR",
            },
        ]
    )
    result = opponent_goal_rate_stats(logs, "BOS")
    assert result["season"]["n"] == 1
    # 3/30 = 10.0%
    assert result["season"]["median"] == pytest.approx(10.0)


def test_opponent_goal_rate_last_n():
    logs = pd.DataFrame(
        [
            {
                "gameId": i,
                "gameDate": f"2024-10-{10 + i:02d}",
                "opponentAbbrev": "BOS",
                "shotsAgainst": 30,
                "goalsAgainst": i,
                "player_id": 1,
                "gamesStarted": 1,
                "savePctg": 1 - i / 30,
                "teamAbbrev": "TOR",
            }
            for i in range(1, 7)
        ]
    )
    result = opponent_goal_rate_stats(logs, "BOS", last_n=3)
    # Last 3 games: i=4,5,6 → rates 4/30, 5/30, 6/30 → mean 5/30 → ×100 ≈ 16.667%
    assert result["last_n"]["n"] == 3
    assert result["last_n"]["mean"] == pytest.approx(5 / 30 * 100)


def test_opponent_goal_rate_last_n_has_range_keys():
    logs = _make_all_logs_for_opponent("BOS", [30, 30, 30])
    logs["goalsAgainst"] = [3, 2, 1]
    result = opponent_goal_rate_stats(logs, "BOS")
    assert "min" in result["last_n"]
    assert "max" in result["last_n"]


def test_opponent_goal_rate_unknown_team_returns_nan():
    logs = _make_all_logs_for_opponent("BOS", [30, 32])
    result = opponent_goal_rate_stats(logs, "VAN")
    assert result["season"]["n"] == 0
    assert math.isnan(result["season"]["median"])


# ---------------------------------------------------------------------------
# days_since_last_game
# ---------------------------------------------------------------------------


def test_days_since_last_game_normal():
    today = date.today()
    days_ago = 3
    past_date = (today - timedelta(days=days_ago)).isoformat()
    df = _make_goalie_log([{"gameDate": past_date}])
    result = days_since_last_game(df)
    assert result == days_ago


def test_days_since_last_game_today():
    df = _make_goalie_log([{"gameDate": date.today().isoformat()}])
    result = days_since_last_game(df)
    assert result == 0


def test_days_since_last_game_empty():
    result = days_since_last_game(pd.DataFrame())
    assert result is None


def test_days_since_last_game_uses_most_recent():
    today = date.today()
    df = _make_goalie_log(
        [
            {"gameId": 1, "gameDate": (today - timedelta(days=10)).isoformat()},
            {"gameId": 2, "gameDate": (today - timedelta(days=2)).isoformat()},
        ]
    )
    result = days_since_last_game(df)
    assert result == 2


# ---------------------------------------------------------------------------
# vs_opponent_sog_season
# ---------------------------------------------------------------------------


def _make_matchup_logs(team: str, opponent: str, shots: list[int]) -> pd.DataFrame:
    rows = [
        {
            "gameId": 7000 + i,
            "gameDate": f"2024-11-{10 + i:02d}",
            "teamAbbrev": team,
            "opponentAbbrev": opponent,
            "shotsAgainst": s,
            "goalsAgainst": 2,
            "savePctg": 0.920,
            "gamesStarted": 1,
            "player_id": 1001,
        }
        for i, s in enumerate(shots)
    ]
    return pd.DataFrame(rows)


def test_vs_opponent_sog_season_matched_games():
    logs = _make_matchup_logs("TOR", "BOS", [28, 32, 30])
    result = vs_opponent_sog_season(logs, "TOR", "BOS")
    assert result["n"] == 3
    assert result["mean"] == pytest.approx((28 + 32 + 30) / 3)
    assert result["min"] == pytest.approx(28.0)
    assert result["max"] == pytest.approx(32.0)


def test_vs_opponent_sog_season_no_meetings():
    logs = _make_matchup_logs("TOR", "BOS", [28, 32])
    result = vs_opponent_sog_season(logs, "TOR", "MTL")
    assert result["n"] == 0
    assert math.isnan(result["mean"])


def test_vs_opponent_sog_season_sums_multi_goalie_game():
    logs = pd.DataFrame(
        [
            {
                "gameId": 8001,
                "gameDate": "2024-11-10",
                "teamAbbrev": "TOR",
                "opponentAbbrev": "BOS",
                "shotsAgainst": 20,
                "goalsAgainst": 2,
                "savePctg": 0.900,
                "gamesStarted": 1,
                "player_id": 1001,
            },
            {
                "gameId": 8001,
                "gameDate": "2024-11-10",
                "teamAbbrev": "TOR",
                "opponentAbbrev": "BOS",
                "shotsAgainst": 10,
                "goalsAgainst": 1,
                "savePctg": 0.900,
                "gamesStarted": 0,
                "player_id": 9999,
            },
        ]
    )
    result = vs_opponent_sog_season(logs, "TOR", "BOS")
    assert result["n"] == 1
    assert result["mean"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# goalie_vs_opponent_sog_season
# ---------------------------------------------------------------------------


def test_goalie_vs_opponent_sog_season_matched():
    log = _make_goalie_log(
        [
            {"gameId": 1, "gameDate": "2024-11-10", "opponentAbbrev": "BOS",
             "shotsAgainst": 28},
            {"gameId": 2, "gameDate": "2024-11-15", "opponentAbbrev": "BOS",
             "shotsAgainst": 34},
            {"gameId": 3, "gameDate": "2024-11-20", "opponentAbbrev": "MTL",
             "shotsAgainst": 25},
        ]
    )
    result = goalie_vs_opponent_sog_season(log, "BOS")
    assert result["n"] == 2
    assert result["mean"] == pytest.approx((28 + 34) / 2)
    assert result["min"] == pytest.approx(28.0)
    assert result["max"] == pytest.approx(34.0)


def test_goalie_vs_opponent_sog_season_no_meetings():
    log = _season_log()
    result = goalie_vs_opponent_sog_season(log, "VAN")
    assert result["n"] == 0
    assert math.isnan(result["mean"])


# ---------------------------------------------------------------------------
# team_sog_stats
# ---------------------------------------------------------------------------


def test_team_sog_stats_season():
    all_logs = _all_logs_for_team("TOR", [28, 30, 32, 25, 35, 27])
    result = team_sog_stats(all_logs, "TOR")
    expected = pd.Series([28, 30, 32, 25, 35, 27]).median()
    assert result["season"]["median"] == pytest.approx(expected)
    assert result["season"]["n"] == 6


def test_team_sog_stats_last_n():
    all_logs = _all_logs_for_team("TOR", [10, 20, 30, 40, 50, 60])
    result = team_sog_stats(all_logs, "TOR", last_n=3)
    # Most recent 3: 40, 50, 60 → mean 50
    assert result["last_n"]["mean"] == pytest.approx(50.0)
    assert result["last_n"]["n"] == 3


def test_team_sog_stats_multi_goalie_game():
    """Two goalies in same game: shots should be summed."""
    logs = pd.DataFrame(
        [
            {
                "gameId": 1,
                "gameDate": "2024-10-10",
                "teamAbbrev": "TOR",
                "opponentAbbrev": "BOS",
                "shotsAgainst": 20,
                "goalsAgainst": 2,
                "gamesStarted": 1,
                "player_id": 1001,
                "savePctg": 0.900,
            },
            {
                "gameId": 1,
                "gameDate": "2024-10-10",
                "teamAbbrev": "TOR",
                "opponentAbbrev": "BOS",
                "shotsAgainst": 10,
                "goalsAgainst": 1,
                "gamesStarted": 0,
                "player_id": 9999,
                "savePctg": 0.900,
            },
        ]
    )
    result = team_sog_stats(logs, "TOR")
    assert result["season"]["n"] == 1
    assert result["season"]["median"] == pytest.approx(30.0)


def test_team_sog_stats_unknown_team():
    all_logs = _all_logs_for_team("TOR", [30, 32])
    result = team_sog_stats(all_logs, "VAN")
    assert result["season"]["n"] == 0
    assert math.isnan(result["season"]["median"])


# ---------------------------------------------------------------------------
# team_save_pct_stats
# ---------------------------------------------------------------------------


def test_team_save_pct_stats_season():
    """Per-game save% = (shots - goals) / shots × 100."""
    rows = [
        {
            "gameId": i,
            "gameDate": f"2024-10-{10 + i:02d}",
            "teamAbbrev": "TOR",
            "opponentAbbrev": "BOS",
            "shotsAgainst": 30,
            "goalsAgainst": 3,
            "gamesStarted": 1,
            "player_id": 1001,
            "savePctg": 0.900,
        }
        for i in range(4)
    ]
    logs = pd.DataFrame(rows)
    result = team_save_pct_stats(logs, "TOR")
    # Each game: (30-3)/30 × 100 = 90.0%
    assert result["season"]["median"] == pytest.approx(90.0)
    assert result["season"]["n"] == 4


def test_team_save_pct_stats_last_n():
    rows = [
        {
            "gameId": i,
            "gameDate": f"2024-10-{10 + i:02d}",
            "teamAbbrev": "TOR",
            "opponentAbbrev": "BOS",
            "shotsAgainst": 30,
            "goalsAgainst": i,  # varying goals → varying save%
            "gamesStarted": 1,
            "player_id": 1001,
            "savePctg": (30 - i) / 30,
        }
        for i in range(1, 7)
    ]
    logs = pd.DataFrame(rows)
    result = team_save_pct_stats(logs, "TOR", last_n=3)
    # Last 3: i=4,5,6 → (30-4)/30, (30-5)/30, (30-6)/30 × 100 = 86.67, 83.33, 80.0
    # mean = 83.33%
    assert result["last_n"]["n"] == 3
    assert result["last_n"]["mean"] == pytest.approx((30 - 5) / 30 * 100)


def test_team_save_pct_stats_unknown_team():
    all_logs = _all_logs_for_team("TOR", [30, 32])
    result = team_save_pct_stats(all_logs, "VAN")
    assert result["season"]["n"] == 0
    assert math.isnan(result["season"]["median"])


# ---------------------------------------------------------------------------
# goalie_report (master function)
# ---------------------------------------------------------------------------


def test_goalie_report_keys():
    goalie_log = _season_log()
    all_logs = pd.concat([goalie_log, _make_all_logs_for_opponent("BOS", [30, 32, 28])])
    result = goalie_report(goalie_log, all_logs, "BOS")
    assert set(result.keys()) == {
        "sog_allowed",
        "opponent_sog",
        "save_pct",
        "vs_opponent",
        "opponent_goal_rate",
        "days_since",
        "vs_opponent_sog",
    }


def test_goalie_report_vs_opponent_is_dataframe():
    goalie_log = _season_log()
    all_logs = goalie_log.copy()
    result = goalie_report(goalie_log, all_logs, "BOS")
    assert isinstance(result["vs_opponent"], pd.DataFrame)


def test_goalie_report_days_since_is_int_or_none():
    goalie_log = _season_log()
    all_logs = goalie_log.copy()
    result = goalie_report(goalie_log, all_logs, "BOS")
    assert result["days_since"] is None or isinstance(result["days_since"], int)


def test_goalie_report_vs_opponent_sog_is_dict():
    goalie_log = _season_log()
    all_logs = goalie_log.copy()
    result = goalie_report(goalie_log, all_logs, "BOS")
    sog = result["vs_opponent_sog"]
    assert isinstance(sog, dict)
    assert "mean" in sog and "min" in sog and "max" in sog and "n" in sog


def test_goalie_report_vs_opponent_sog_matched():
    goalie_log = _season_log()  # has 5 BOS games
    all_logs = goalie_log.copy()
    result = goalie_report(goalie_log, all_logs, "BOS")
    # 5 BOS games in _season_log (i=1..5 have opponentAbbrev="BOS")
    assert result["vs_opponent_sog"]["n"] == 5


def test_goalie_report_stat_dicts_have_expected_keys():
    goalie_log = _season_log()
    all_logs = goalie_log.copy()
    result = goalie_report(goalie_log, all_logs, "BOS")

    # sog_allowed has a team/goalie split; check both sub-keys
    for split in ("team", "goalie"):
        d_season = result["sog_allowed"][split]["season"]
        d_last_n = result["sog_allowed"][split]["last_n"]
        assert "median" in d_season and "p25" in d_season and "p75" in d_season
        assert "mean" in d_last_n and "min" in d_last_n and "max" in d_last_n

    # save_pct and opponent_sog/goal_rate: season has p25/p75, last_n has min/max
    for key in ("save_pct", "opponent_sog", "opponent_goal_rate"):
        assert "p25" in result[key]["season"]
        assert "min" in result[key]["last_n"]
