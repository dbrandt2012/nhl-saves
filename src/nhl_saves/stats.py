"""Pure statistics computations for goalie analysis.

All functions are side-effect free: they take DataFrames and return dicts or
DataFrames. No I/O. The caller (store.py or main.py) is responsible for
fetching and supplying the data.

Stats produced:
  1. SOG/gm the goalie's team allows, split:
       - team: all goalies on the team (full defensive picture)
       - goalie: only games this goalie started
     Both: last N + season, median, p25, p75
  2. SOG/gm the opposing team averages        (last N + season: median, p25, p75)
  3. Goalie save %                            (last N + season: median, p25, p75)
  4. SOG allowed & saves vs this opponent     (per game, most recent first)
  5. Opponent goal conversion % (goals / SOG)   (last N + season: median, p25, p75)
"""

from datetime import date

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def percentile_summary(series: pd.Series) -> dict[str, float]:
    """Return median, p25, p75, and n for a numeric series (NaNs dropped)."""
    clean = series.dropna()
    if clean.empty:
        nan = float("nan")
        return {"median": nan, "p25": nan, "p75": nan, "n": 0}
    return {
        "median": float(clean.median()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "n": int(len(clean)),
    }


def range_summary(series: pd.Series) -> dict[str, float]:
    """Return median, min, max, and n for a numeric series (NaNs dropped).

    Used for last-N game windows where the full range is more informative
    than percentiles over a small sample.
    """
    clean = series.dropna()
    if clean.empty:
        nan = float("nan")
        return {"median": nan, "min": nan, "max": nan, "n": 0}
    return {
        "median": float(clean.median()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "n": int(len(clean)),
    }


def mean_range_summary(series: pd.Series) -> dict[str, float]:
    """Return mean, min, max, and n for a numeric series (NaNs dropped).

    Used for vs-opponent historical stats where mean is preferred.
    """
    clean = series.dropna()
    if clean.empty:
        nan = float("nan")
        return {"mean": nan, "min": nan, "max": nan, "n": 0}
    return {
        "mean": float(clean.mean()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "n": int(len(clean)),
    }


def _last_n(df: pd.DataFrame, n: int, date_col: str = "gameDate") -> pd.DataFrame:
    """Return the n most recent rows sorted by date_col ascending."""
    return df.sort_values(date_col).tail(n)


# ---------------------------------------------------------------------------
# Stat 1: SOG/gm the goalie's team allows
# ---------------------------------------------------------------------------


def sog_allowed_stats(
    goalie_log: pd.DataFrame,
    all_goalie_logs: pd.DataFrame,
    last_n: int = 5,
) -> dict[str, dict[str, dict[str, float]]]:
    """SOG/gm the goalie's team allows, split into team-level and goalie-level.

    - "team": all goalies on this team, grouped by game (full defensive picture
      regardless of who was in net). Uses all_goalie_logs filtered by teamAbbrev.
    - "goalie": only games this specific goalie started.

    Args:
        goalie_log: Single goalie's game log DataFrame.
        all_goalie_logs: All goalies' combined game log for the season.
        last_n: Number of most recent games for the "last N" window.

    Returns:
        {
            "team":   {"season": {median, p25, p75, n},
                       "last_n": {median, min, max, n}},
            "goalie": {"season": {median, p25, p75, n},
                       "last_n": {median, min, max, n}},
        }
    """
    # --- Team split: all goalies on this team, one row per game ---
    team_abbrev = goalie_log["teamAbbrev"].iloc[0] if not goalie_log.empty else None
    if team_abbrev is not None:
        team_games = all_goalie_logs[
            all_goalie_logs["teamAbbrev"] == team_abbrev
        ].copy()
        per_game = (
            team_games.groupby("gameId")
            .agg(shotsAgainst=("shotsAgainst", "sum"), gameDate=("gameDate", "max"))
            .reset_index()
            .sort_values("gameDate")
        )
        team_season = percentile_summary(per_game["shotsAgainst"])
        team_last_n = range_summary(_last_n(per_game, last_n)["shotsAgainst"])
    else:
        empty_pct = percentile_summary(pd.Series([], dtype=float))
        empty_rng = range_summary(pd.Series([], dtype=float))
        team_season = empty_pct
        team_last_n = empty_rng

    # --- Goalie split: only this goalie's started games ---
    started = goalie_log[goalie_log["gamesStarted"] == 1].sort_values("gameDate")
    goalie_season = percentile_summary(started["shotsAgainst"])
    goalie_last_n = range_summary(_last_n(started, last_n)["shotsAgainst"])

    return {
        "team": {"season": team_season, "last_n": team_last_n},
        "goalie": {"season": goalie_season, "last_n": goalie_last_n},
    }


# ---------------------------------------------------------------------------
# Stat 2: SOG/gm the opposing team averages
# ---------------------------------------------------------------------------


def opponent_sog_stats(
    all_goalie_logs: pd.DataFrame,
    opponent_abbrev: str,
    last_n: int = 5,
) -> dict[str, dict[str, float]]:
    """SOG/gm the opposing team generates across the full season.

    Derives per-game shot totals from all goalie game logs: every row where
    opponentAbbrev == opponent_abbrev represents a game where that team was
    on offence. Rows are grouped by gameId and summed to handle games where
    multiple goalies played.

    Args:
        all_goalie_logs: Combined game log DataFrame for ALL goalies in
            the season (from build_goalie_game_logs).
        opponent_abbrev: Three-letter team code for the opposing team.
        last_n: Number of most recent games for the "last N" window.

    Returns:
        {"season": {median, p25, p75, n}, "last_n": {median, min, max, n}}
    """
    opponent_games = all_goalie_logs[
        all_goalie_logs["opponentAbbrev"] == opponent_abbrev
    ].copy()

    if opponent_games.empty:
        return {
            "season": percentile_summary(pd.Series([], dtype=float)),
            "last_n": range_summary(pd.Series([], dtype=float)),
        }

    # Sum shots across all goalies who played in the same game
    per_game = (
        opponent_games.groupby("gameId")
        .agg(shotsFor=("shotsAgainst", "sum"), gameDate=("gameDate", "max"))
        .reset_index()
        .sort_values("gameDate")
    )

    season_shots = per_game["shotsFor"]
    last_n_shots = _last_n(per_game, last_n)["shotsFor"]
    return {
        "season": percentile_summary(season_shots),
        "last_n": range_summary(last_n_shots),
    }


# ---------------------------------------------------------------------------
# Stat 3: Goalie save %
# ---------------------------------------------------------------------------


def save_pct_stats(
    goalie_log: pd.DataFrame,
    last_n: int = 5,
) -> dict[str, dict[str, float]]:
    """Save percentage distribution for a goalie.

    Values are returned as percentages (0–100), not proportions.

    Args:
        goalie_log: Single goalie's game log DataFrame.
        last_n: Number of most recent games for the "last N" window.

    Returns:
        {"season": {median, p25, p75, n}, "last_n": {median, min, max, n}}
    """
    started = goalie_log[goalie_log["gamesStarted"] == 1].sort_values("gameDate")
    pcts = started["savePctg"] * 100
    return {
        "season": percentile_summary(pcts),
        "last_n": range_summary(_last_n(started, last_n)["savePctg"] * 100),
    }


# ---------------------------------------------------------------------------
# Stat 4: Historical matchup vs a specific opponent
# ---------------------------------------------------------------------------


def vs_opponent_history(
    goalie_log: pd.DataFrame,
    opponent_abbrev: str,
) -> pd.DataFrame:
    """All games this goalie has played against a specific opponent.

    Returned most recent first. Includes: gameDate, homeRoadFlag, decision,
    shotsAgainst, goalsAgainst, savePctg, toi.

    Args:
        goalie_log: Single goalie's game log DataFrame.
        opponent_abbrev: Three-letter team code for the opponent.

    Returns:
        DataFrame sorted by gameDate descending.
    """
    cols = [
        "gameDate",
        "homeRoadFlag",
        "decision",
        "shotsAgainst",
        "goalsAgainst",
        "savePctg",
        "toi",
    ]
    available_cols = [c for c in cols if c in goalie_log.columns]

    history = goalie_log[goalie_log["opponentAbbrev"] == opponent_abbrev].copy()
    return (
        history[available_cols]
        .sort_values("gameDate", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Stat 5: Opponent goal conversion % (goals / SOG per game)
# ---------------------------------------------------------------------------


def opponent_goal_rate_stats(
    all_goalie_logs: pd.DataFrame,
    opponent_abbrev: str,
    last_n: int = 5,
) -> dict[str, dict[str, float]]:
    """Per-game percentage of shots that become goals for the opposing team.

    Uses the same game-log rows as opponent_sog_stats: rows where
    opponentAbbrev == opponent_abbrev. shotsAgainst and goalsAgainst in those
    rows represent the opponent's offensive output.

    Returns:
        {"season": {median, p25, p75, n}, "last_n": {median, min, max, n}}
        Values are percentages (0–100).
    """
    opponent_games = all_goalie_logs[
        all_goalie_logs["opponentAbbrev"] == opponent_abbrev
    ].copy()

    if opponent_games.empty:
        return {
            "season": percentile_summary(pd.Series([], dtype=float)),
            "last_n": range_summary(pd.Series([], dtype=float)),
        }

    per_game = (
        opponent_games.groupby("gameId")
        .agg(
            shotsFor=("shotsAgainst", "sum"),
            goalsFor=("goalsAgainst", "sum"),
            gameDate=("gameDate", "max"),
        )
        .reset_index()
        .sort_values("gameDate")
    )

    with_shots = per_game[per_game["shotsFor"] > 0].copy()
    with_shots["goal_rate"] = with_shots["goalsFor"] / with_shots["shotsFor"] * 100

    rates = with_shots["goal_rate"]
    return {
        "season": percentile_summary(rates),
        "last_n": range_summary(_last_n(with_shots, last_n)["goal_rate"]),
    }


# ---------------------------------------------------------------------------
# New helpers: days since last game, vs-opponent SOG stats
# ---------------------------------------------------------------------------


def days_since_last_game(
    df: pd.DataFrame,
    date_col: str = "gameDate",
) -> int | None:
    """Return the number of days since the most recent game.

    Args:
        df: DataFrame containing a date column.
        date_col: Name of the date column (string or datetime).

    Returns:
        Integer days since last game, or None if df is empty.
    """
    if df.empty:
        return None
    last = pd.to_datetime(df[date_col]).max()
    return (date.today() - last.date()).days


def vs_opponent_sog_season(
    all_goalie_logs: pd.DataFrame,
    team_abbrev: str,
    opponent_abbrev: str,
) -> dict[str, float]:
    """Team-level SOG allowed in prior meetings vs a specific opponent this season.

    Filters all_goalie_logs to games where teamAbbrev == team_abbrev AND
    opponentAbbrev == opponent_abbrev, groups by gameId, sums shotsAgainst,
    and returns mean_range_summary.

    Args:
        all_goalie_logs: Combined game log for all goalies this season.
        team_abbrev: The defending team's abbreviation.
        opponent_abbrev: The attacking opponent's abbreviation.

    Returns:
        {mean, min, max, n} — empty summary (NaN values) if no prior meetings.
    """
    mask = (all_goalie_logs["teamAbbrev"] == team_abbrev) & (
        all_goalie_logs["opponentAbbrev"] == opponent_abbrev
    )
    matching = all_goalie_logs[mask].copy()

    if matching.empty:
        return mean_range_summary(pd.Series([], dtype=float))

    per_game = (
        matching.groupby("gameId")
        .agg(shotsAgainst=("shotsAgainst", "sum"))
        .reset_index()
    )
    return mean_range_summary(per_game["shotsAgainst"])


def goalie_vs_opponent_sog_season(
    goalie_log: pd.DataFrame,
    opponent_abbrev: str,
) -> dict[str, float]:
    """Goalie-level SOG allowed in prior meetings vs a specific opponent.

    Filters goalie_log to games where opponentAbbrev == opponent_abbrev and
    returns mean_range_summary of per-game shotsAgainst.

    Args:
        goalie_log: Single goalie's game log DataFrame.
        opponent_abbrev: The attacking opponent's abbreviation.

    Returns:
        {mean, min, max, n} — empty summary (NaN values) if no prior meetings.
    """
    matching = goalie_log[goalie_log["opponentAbbrev"] == opponent_abbrev].copy()

    if matching.empty:
        return mean_range_summary(pd.Series([], dtype=float))

    return mean_range_summary(matching["shotsAgainst"])


def team_sog_stats(
    all_goalie_logs: pd.DataFrame,
    team_abbrev: str,
    last_n: int = 5,
) -> dict[str, dict[str, float]]:
    """Team-level SOG allowed per game across the season.

    Groups by gameId, sums shotsAgainst across all goalies in each game,
    returns percentile_summary for the full season and range_summary for the
    last N games.

    Args:
        all_goalie_logs: Combined game log for all goalies this season.
        team_abbrev: The team's abbreviation.
        last_n: Number of most recent games for the "last N" window.

    Returns:
        {"season": {median, p25, p75, n}, "last_n": {median, min, max, n}}
    """
    team_games = all_goalie_logs[
        all_goalie_logs["teamAbbrev"] == team_abbrev
    ].copy()

    if team_games.empty:
        return {
            "season": percentile_summary(pd.Series([], dtype=float)),
            "last_n": range_summary(pd.Series([], dtype=float)),
        }

    per_game = (
        team_games.groupby("gameId")
        .agg(shotsAgainst=("shotsAgainst", "sum"), gameDate=("gameDate", "max"))
        .reset_index()
        .sort_values("gameDate")
    )

    return {
        "season": percentile_summary(per_game["shotsAgainst"]),
        "last_n": range_summary(_last_n(per_game, last_n)["shotsAgainst"]),
    }


def team_save_pct_stats(
    all_goalie_logs: pd.DataFrame,
    team_abbrev: str,
    last_n: int = 5,
) -> dict[str, dict[str, float]]:
    """Team-level save percentage per game across the season.

    Groups by gameId, sums shotsAgainst and goalsAgainst, computes per-game
    save% as (shotsAgainst - goalsAgainst) / shotsAgainst * 100. Returns
    percentile_summary for the season, range_summary for last N.

    Args:
        all_goalie_logs: Combined game log for all goalies this season.
        team_abbrev: The team's abbreviation.
        last_n: Number of most recent games for the "last N" window.

    Returns:
        {"season": {median, p25, p75, n}, "last_n": {median, min, max, n}}
    """
    team_games = all_goalie_logs[
        all_goalie_logs["teamAbbrev"] == team_abbrev
    ].copy()

    if team_games.empty:
        return {
            "season": percentile_summary(pd.Series([], dtype=float)),
            "last_n": range_summary(pd.Series([], dtype=float)),
        }

    per_game = (
        team_games.groupby("gameId")
        .agg(
            shotsAgainst=("shotsAgainst", "sum"),
            goalsAgainst=("goalsAgainst", "sum"),
            gameDate=("gameDate", "max"),
        )
        .reset_index()
        .sort_values("gameDate")
    )

    with_shots = per_game[per_game["shotsAgainst"] > 0].copy()
    with_shots["save_pct"] = (
        (with_shots["shotsAgainst"] - with_shots["goalsAgainst"])
        / with_shots["shotsAgainst"]
        * 100
    )

    return {
        "season": percentile_summary(with_shots["save_pct"]),
        "last_n": range_summary(_last_n(with_shots, last_n)["save_pct"]),
    }


# ---------------------------------------------------------------------------
# Master report
# ---------------------------------------------------------------------------


def goalie_report(
    goalie_log: pd.DataFrame,
    all_goalie_logs: pd.DataFrame,
    opponent_abbrev: str,
    last_n: int = 5,
) -> dict:
    """Compute all five stats for a goalie against a given opponent.

    Args:
        goalie_log: This goalie's individual game log.
        all_goalie_logs: All goalies' combined game log for the season.
        opponent_abbrev: Three-letter team code for the upcoming opponent.
        last_n: Window size for "last N games" summaries.

    Returns:
        {
            "sog_allowed":    {"season": {...}, "last_n": {...}},  # stat 1
            "opponent_sog":   {"season": {...}, "last_n": {...}},  # stat 2
            "save_pct":       {"season": {...}, "last_n": {...}},  # stat 3
            "vs_opponent":    DataFrame,                           # stat 4
            "opponent_goal_rate": {"season": {...}, "last_n": {...}},  # stat 5
            "days_since":     int or None,
            "vs_opponent_sog": {mean, min, max, n},
        }
    """
    started = goalie_log[goalie_log["gamesStarted"] == 1].sort_values("gameDate")
    return {
        "sog_allowed": sog_allowed_stats(goalie_log, all_goalie_logs, last_n),
        "opponent_sog": opponent_sog_stats(all_goalie_logs, opponent_abbrev, last_n),
        "save_pct": save_pct_stats(goalie_log, last_n),
        "vs_opponent": vs_opponent_history(goalie_log, opponent_abbrev),
        "opponent_goal_rate": opponent_goal_rate_stats(
            all_goalie_logs, opponent_abbrev, last_n
        ),
        "days_since": days_since_last_game(started),
        "vs_opponent_sog": goalie_vs_opponent_sog_season(goalie_log, opponent_abbrev),
    }
