"""NHL API client.

Wraps two public NHL APIs (no auth required):
  - api-web.nhle.com/v1        — game logs, schedule, roster, team stats
  - api.nhle.com/stats/rest/en — bulk goalie/skater stats with filtering
"""

import requests


class NHLClient:
    WEB_BASE = "https://api-web.nhle.com/v1"
    STATS_BASE = "https://api.nhle.com/stats/rest/en"

    def __init__(self) -> None:
        self._session = requests.Session()

    def _get(self, url: str, params: dict | None = None) -> dict:
        response = self._session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def get_schedule(self, date: str | None = None) -> dict:
        """Return league schedule for a given date (YYYY-MM-DD) or today."""
        if date:
            url = f"{self.WEB_BASE}/schedule/{date}"
        else:
            url = f"{self.WEB_BASE}/schedule/now"
        return self._get(url)

    # ------------------------------------------------------------------
    # Roster
    # ------------------------------------------------------------------

    def get_team_roster(
        self, team_abbrev: str, season: str
    ) -> dict[str, list[dict]]:
        """Return team roster split into goalies and skaters.

        Args:
            team_abbrev: Three-letter team code, e.g. "TOR".
            season: Season in YYYYYYYY format, e.g. "20242025".

        Returns:
            {"goalies": [...], "skaters": [...]}
        """
        url = f"{self.WEB_BASE}/roster/{team_abbrev}/{season}"
        data = self._get(url)

        goalies = data.get("goalies", [])
        skaters = data.get("forwards", []) + data.get("defensemen", [])
        return {"goalies": goalies, "skaters": skaters}

    # ------------------------------------------------------------------
    # Goalie stats
    # ------------------------------------------------------------------

    def get_goalie_leaders(
        self,
        season: str,
        game_type: int = 2,
        category: str = "savePctg",
        limit: int = 50,
    ) -> list[dict]:
        """Return season goalie leaders for a stat category.

        Args:
            season: Season in YYYYYYYY format, e.g. "20242025".
            game_type: 2 = regular season, 3 = playoffs.
            category: Stat category — "savePctg", "wins", "gaa", "shutouts".
            limit: Max number of results (-1 for all).

        Returns:
            List of goalie objects with id, firstName, lastName, teamAbbrev, value.
        """
        url = f"{self.WEB_BASE}/goalie-stats-leaders/{season}/{game_type}"
        data = self._get(url, params={"categories": category, "limit": limit})
        return data.get(category, [])

    def get_goalie_stats(
        self,
        season: str,
        game_type: int = 2,
        start: int = 0,
        limit: int = 100,
    ) -> list[dict]:
        """Return bulk goalie summary stats via the Stats REST API.

        Supports pagination for fetching all goalies in a season.

        Args:
            season: Season ID, e.g. "20242025".
            game_type: 2 = regular season, 3 = playoffs.
            start: Pagination offset.
            limit: Page size (-1 for all).

        Returns:
            List of goalie stat objects.
        """
        url = f"{self.STATS_BASE}/goalie/summary"
        params = {
            "cayenneExp": f"seasonId={season} and gameTypeId={game_type}",
            "start": start,
            "limit": limit,
        }
        data = self._get(url, params=params)
        return data.get("data", [])

    # ------------------------------------------------------------------
    # Player game log (goalies and skaters)
    # ------------------------------------------------------------------

    def get_player_game_log(
        self, player_id: int, season: str, game_type: int = 2
    ) -> list[dict]:
        """Return per-game stats for a player (goalie or skater).

        Goalie fields: gameId, gameDate, teamAbbrev, opponentAbbrev,
            homeRoadFlag, decision, shotsAgainst, goalsAgainst,
            savePctg, shutouts, toi, gamesStarted.

        Skater fields: gameId, gameDate, teamAbbrev, opponentAbbrev,
            homeRoadFlag, shots, goals, assists, points, toi, plusMinus, pim.

        Args:
            player_id: Numeric NHL player ID.
            season: Season in YYYYYYYY format, e.g. "20242025".
            game_type: 2 = regular season, 3 = playoffs.

        Returns:
            List of game log entries ordered most-recent first.
        """
        url = f"{self.WEB_BASE}/player/{player_id}/game-log/{season}/{game_type}"
        data = self._get(url)
        return data.get("gameLog", [])

    # ------------------------------------------------------------------
    # Team stats
    # ------------------------------------------------------------------

    def get_team_stats(
        self, team_abbrev: str, season: str, game_type: int = 2
    ) -> dict:
        """Return team-level stats including shots-for/against per game.

        Useful fields: shotsForPerGame, shotsAgainstPerGame, goalsFor,
            goalsAgainst, wins, losses, overtimeLosses, powerPlayPercentage,
            penaltyKillPercentage.

        Args:
            team_abbrev: Three-letter team code, e.g. "TOR".
            season: Season in YYYYYYYY format, e.g. "20242025".
            game_type: 2 = regular season, 3 = playoffs.

        Returns:
            Team stats dict.
        """
        url = f"{self.WEB_BASE}/club-stats/{team_abbrev}/{season}/{game_type}"
        return self._get(url)
