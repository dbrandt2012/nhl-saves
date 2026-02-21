"""Unit tests for stats.py — pure functions, no mocking required."""

import math

import pandas as pd
import pytest

from nhl_saves.stats import (
    goal_rate_stats,
    goalie_report,
    opponent_sog_stats,
    percentile_summary,
    save_pct_stats,
    sog_allowed_stats,
    vs_opponent_history,
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
    return _make_goalie_log([
        {"gameId": i, "gameDate": f"2024-10-{10 + i:02d}",
         "shotsAgainst": 20 + i, "goalsAgainst": i % 3,
         "savePctg": round(1 - (i % 3) / (20 + i), 4),
         "opponentAbbrev": "BOS" if i <= 5 else "MTL"}
        for i in range(1, 11)
    ])


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


def test_sog_allowed_goalie_excludes_non_starts():
    log = _make_goalie_log([
        {"gameId": 1, "gameDate": "2024-10-10", "shotsAgainst": 30, "gamesStarted": 1},
        {"gameId": 2, "gameDate": "2024-10-11", "shotsAgainst": 5, "gamesStarted": 0},
    ])
    result = sog_allowed_stats(log, log)
    assert result["goalie"]["season"]["n"] == 1
    assert result["goalie"]["season"]["median"] == pytest.approx(30.0)


def test_sog_allowed_goalie_last_n_uses_most_recent():
    log = _make_goalie_log([
        {"gameId": i, "gameDate": f"2024-10-{10 + i:02d}", "shotsAgainst": i * 10}
        for i in range(1, 7)
    ])
    result = sog_allowed_stats(log, log, last_n=3)
    # Games 4, 5, 6 → shots 40, 50, 60 → median 50
    assert result["goalie"]["last_n"]["median"] == pytest.approx(50.0)


def test_sog_allowed_team_aggregates_all_goalies():
    """Team split should include games started by other goalies on the team."""
    # Goalie 1001 (TOR) started 3 games
    goalie_log = _make_goalie_log([
        {"gameId": i, "gameDate": f"2024-10-{10 + i:02d}",
         "teamAbbrev": "TOR", "shotsAgainst": 30}
        for i in range(1, 4)
    ])
    # Another TOR goalie played 2 more games
    other_goalie = _make_goalie_log([
        {"gameId": i + 10, "gameDate": f"2024-10-{14 + i:02d}",
         "teamAbbrev": "TOR", "shotsAgainst": 25, "player_id": 9999}
        for i in range(1, 3)
    ])
    all_logs = pd.concat([goalie_log, other_goalie], ignore_index=True)
    result = sog_allowed_stats(goalie_log, all_logs)
    # Team sees all 5 games; goalie sees only their 3
    assert result["team"]["season"]["n"] == 5
    assert result["goalie"]["season"]["n"] == 3


def test_sog_allowed_team_sums_multi_goalie_game():
    """If two goalies played in the same game, team total should be summed."""
    goalie_log = _make_goalie_log([
        {"gameId": 1, "gameDate": "2024-10-10", "teamAbbrev": "TOR",
         "shotsAgainst": 20, "gamesStarted": 1},
    ])
    backup = _make_goalie_log([
        {"gameId": 1, "gameDate": "2024-10-10", "teamAbbrev": "TOR",
         "shotsAgainst": 10, "gamesStarted": 0, "player_id": 9999},
    ])
    all_logs = pd.concat([goalie_log, backup], ignore_index=True)
    result = sog_allowed_stats(goalie_log, all_logs)
    assert result["team"]["season"]["n"] == 1
    assert result["team"]["season"]["median"] == pytest.approx(30.0)


def test_sog_allowed_team_last_n_uses_most_recent():
    all_logs = _all_logs_for_team("TOR", [10, 20, 30, 40, 50, 60])
    goalie_log = all_logs[all_logs["player_id"] == 8000].copy()
    if goalie_log.empty:
        goalie_log = all_logs.head(1).copy()
    result = sog_allowed_stats(all_logs.head(1), all_logs, last_n=3)
    # Most recent 3 team games: 40, 50, 60 → median 50
    assert result["team"]["last_n"]["median"] == pytest.approx(50.0)


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
    # Most recent 3 games: shots 30, 35, 40 → median 35
    assert result["last_n"]["median"] == pytest.approx(35.0)
    assert result["last_n"]["n"] == 3


def test_opponent_sog_sums_multi_goalie_game():
    """When two goalies played in the same game, shots should be summed."""
    logs = pd.DataFrame([
        {"gameId": 3001, "gameDate": "2024-10-10", "opponentAbbrev": "BOS",
         "shotsAgainst": 20, "player_id": 1, "gamesStarted": 1,
         "savePctg": 0.900, "goalsAgainst": 2, "teamAbbrev": "TOR"},
        {"gameId": 3001, "gameDate": "2024-10-10", "opponentAbbrev": "BOS",
         "shotsAgainst": 15, "player_id": 2, "gamesStarted": 0,
         "savePctg": 0.867, "goalsAgainst": 2, "teamAbbrev": "TOR"},
    ])
    result = opponent_sog_stats(logs, "BOS")
    # One game, 20+15=35 total shots
    assert result["season"]["n"] == 1
    assert result["season"]["median"] == pytest.approx(35.0)


def test_opponent_sog_unknown_team_returns_nan():
    logs = _make_all_logs_for_opponent("BOS", [30, 32])
    result = opponent_sog_stats(logs, "VAN")
    assert result["season"]["n"] == 0
    assert math.isnan(result["season"]["median"])


# ---------------------------------------------------------------------------
# Stat 3: save_pct_stats
# ---------------------------------------------------------------------------


def test_save_pct_season_values():
    log = _make_goalie_log([
        {"gameId": i, "gameDate": f"2024-10-{10 + i:02d}", "savePctg": v}
        for i, v in enumerate([0.900, 0.920, 0.930, 0.910, 0.950])
    ])
    result = save_pct_stats(log)
    assert result["season"]["median"] == pytest.approx(0.920)
    assert result["season"]["n"] == 5


def test_save_pct_last_n():
    log = _make_goalie_log([
        {"gameId": i, "gameDate": f"2024-10-{10 + i:02d}", "savePctg": 0.880 + i * 0.01}
        for i in range(8)
    ])
    result = save_pct_stats(log, last_n=3)
    assert result["last_n"]["n"] == 3
    # Most recent 3: indices 5,6,7 → 0.930, 0.940, 0.950 → median 0.940
    assert result["last_n"]["median"] == pytest.approx(0.940)


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
    expected_cols = {"gameDate", "homeRoadFlag", "decision",
                     "shotsAgainst", "goalsAgainst", "savePctg", "toi"}
    assert expected_cols.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Stat 5: goal_rate_stats
# ---------------------------------------------------------------------------


def test_goal_rate_computed_from_raw_counts():
    log = _make_goalie_log([
        {"gameId": 1, "gameDate": "2024-10-10", "shotsAgainst": 30, "goalsAgainst": 3},
        {"gameId": 2, "gameDate": "2024-10-11", "shotsAgainst": 25, "goalsAgainst": 2},
    ])
    result = goal_rate_stats(log)
    # Game 1: 3/30=0.100, Game 2: 2/25=0.080 → median 0.090
    assert result["season"]["median"] == pytest.approx(0.090)


def test_goal_rate_excludes_zero_shot_games():
    log = _make_goalie_log([
        {"gameId": 1, "gameDate": "2024-10-10", "shotsAgainst": 0, "goalsAgainst": 0},
        {"gameId": 2, "gameDate": "2024-10-11", "shotsAgainst": 30, "goalsAgainst": 3},
    ])
    result = goal_rate_stats(log)
    assert result["season"]["n"] == 1
    assert result["season"]["median"] == pytest.approx(0.100)


def test_goal_rate_last_n():
    log = _make_goalie_log([
        {"gameId": i, "gameDate": f"2024-10-{10 + i:02d}",
         "shotsAgainst": 30, "goalsAgainst": i}
        for i in range(1, 7)
    ])
    result = goal_rate_stats(log, last_n=3)
    # Last 3 games: i=4,5,6 → rates 4/30, 5/30, 6/30 → median 5/30
    assert result["last_n"]["n"] == 3
    assert result["last_n"]["median"] == pytest.approx(5 / 30)


# ---------------------------------------------------------------------------
# goalie_report (master function)
# ---------------------------------------------------------------------------


def test_goalie_report_keys():
    goalie_log = _season_log()
    all_logs = pd.concat([goalie_log, _make_all_logs_for_opponent("BOS", [30, 32, 28])])
    result = goalie_report(goalie_log, all_logs, "BOS")
    assert set(result.keys()) == {
        "sog_allowed", "opponent_sog", "save_pct", "vs_opponent", "goal_rate"
    }


def test_goalie_report_vs_opponent_is_dataframe():
    goalie_log = _season_log()
    all_logs = goalie_log.copy()
    result = goalie_report(goalie_log, all_logs, "BOS")
    assert isinstance(result["vs_opponent"], pd.DataFrame)


def test_goalie_report_stat_dicts_have_expected_keys():
    goalie_log = _season_log()
    all_logs = goalie_log.copy()
    result = goalie_report(goalie_log, all_logs, "BOS")
    for key in ("sog_allowed", "opponent_sog", "save_pct", "goal_rate"):
        for window in ("season", "last_n"):
            assert "median" in result[key][window]
            assert "p25" in result[key][window]
            assert "p75" in result[key][window]
