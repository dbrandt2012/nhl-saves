"""Unit tests for NHLClient. All network calls are mocked."""

from unittest.mock import MagicMock, patch

import pytest

from nhl_saves.api import NHLClient


@pytest.fixture
def client():
    return NHLClient()


def _mock_response(payload: dict) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = payload
    mock.raise_for_status.return_value = None
    return mock


# ------------------------------------------------------------------
# Schedule
# ------------------------------------------------------------------


def test_get_schedule_today(client):
    with patch.object(client._session, "get", return_value=_mock_response({"gameWeek": []})) as mock_get:
        result = client.get_schedule()
        mock_get.assert_called_once_with(
            "https://api-web.nhle.com/v1/schedule/now", params=None
        )
        assert result == {"gameWeek": []}


def test_get_schedule_date(client):
    with patch.object(client._session, "get", return_value=_mock_response({"gameWeek": []})) as mock_get:
        client.get_schedule("2025-01-15")
        mock_get.assert_called_once_with(
            "https://api-web.nhle.com/v1/schedule/2025-01-15", params=None
        )


# ------------------------------------------------------------------
# Roster
# ------------------------------------------------------------------


def test_get_team_roster_splits_by_position(client):
    payload = {
        "goalies": [{"id": 1, "positionCode": "G"}],
        "forwards": [{"id": 2, "positionCode": "C"}, {"id": 3, "positionCode": "L"}],
        "defensemen": [{"id": 4, "positionCode": "D"}],
    }
    with patch.object(client._session, "get", return_value=_mock_response(payload)):
        result = client.get_team_roster("TOR", "20242025")

    assert len(result["goalies"]) == 1
    assert result["goalies"][0]["id"] == 1
    assert len(result["skaters"]) == 3
    assert all(p["positionCode"] != "G" for p in result["skaters"])


def test_get_team_roster_url(client):
    with patch.object(client._session, "get", return_value=_mock_response({})) as mock_get:
        client.get_team_roster("EDM", "20242025")
        mock_get.assert_called_once_with(
            "https://api-web.nhle.com/v1/roster/EDM/20242025", params=None
        )


# ------------------------------------------------------------------
# Goalie leaders
# ------------------------------------------------------------------


def test_get_goalie_leaders_returns_category_list(client):
    payload = {"savePctg": [{"id": 8476932, "value": 0.925}]}
    with patch.object(client._session, "get", return_value=_mock_response(payload)):
        result = client.get_goalie_leaders("20242025")

    assert result == [{"id": 8476932, "value": 0.925}]


def test_get_goalie_leaders_query_params(client):
    with patch.object(client._session, "get", return_value=_mock_response({"wins": []})) as mock_get:
        client.get_goalie_leaders("20242025", game_type=2, category="wins", limit=10)
        mock_get.assert_called_once_with(
            "https://api-web.nhle.com/v1/goalie-stats-leaders/20242025/2",
            params={"categories": "wins", "limit": 10},
        )


# ------------------------------------------------------------------
# Bulk goalie stats (Stats REST API)
# ------------------------------------------------------------------


def test_get_goalie_stats_url_and_params(client):
    with patch.object(client._session, "get", return_value=_mock_response({"data": []})) as mock_get:
        client.get_goalie_stats("20242025")
        mock_get.assert_called_once_with(
            "https://api.nhle.com/stats/rest/en/goalie/summary",
            params={
                "cayenneExp": "seasonId=20242025 and gameTypeId=2",
                "start": 0,
                "limit": 100,
            },
        )


def test_get_goalie_stats_returns_data_list(client):
    payload = {"data": [{"playerId": 1, "wins": 30}]}
    with patch.object(client._session, "get", return_value=_mock_response(payload)):
        result = client.get_goalie_stats("20242025")
    assert result == [{"playerId": 1, "wins": 30}]


# ------------------------------------------------------------------
# Player game log
# ------------------------------------------------------------------


def test_get_player_game_log_url(client):
    with patch.object(client._session, "get", return_value=_mock_response({"gameLog": []})) as mock_get:
        client.get_player_game_log(8480045, "20242025")
        mock_get.assert_called_once_with(
            "https://api-web.nhle.com/v1/player/8480045/game-log/20242025/2",
            params=None,
        )


def test_get_player_game_log_returns_game_list(client):
    games = [{"gameId": 1, "savePctg": 0.933}, {"gameId": 2, "savePctg": 0.900}]
    with patch.object(client._session, "get", return_value=_mock_response({"gameLog": games})):
        result = client.get_player_game_log(8480045, "20242025")
    assert len(result) == 2
    assert result[0]["savePctg"] == 0.933


# ------------------------------------------------------------------
# Team stats
# ------------------------------------------------------------------


def test_get_team_stats_url(client):
    with patch.object(client._session, "get", return_value=_mock_response({})) as mock_get:
        client.get_team_stats("BOS", "20242025")
        mock_get.assert_called_once_with(
            "https://api-web.nhle.com/v1/club-stats/BOS/20242025/2", params=None
        )


def test_get_team_stats_returns_dict(client):
    payload = {"shotsForPerGame": 32.5, "shotsAgainstPerGame": 28.1}
    with patch.object(client._session, "get", return_value=_mock_response(payload)):
        result = client.get_team_stats("BOS", "20242025")
    assert result["shotsForPerGame"] == 32.5
