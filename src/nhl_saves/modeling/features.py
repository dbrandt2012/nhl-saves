"""Feature engineering for goalie saves prediction model — Phase 2.

Primary data source: MoneyPuck.com (season summary + per-game career files).
Spine: NHL API game logs (dates, homeRoadFlag, decision, gamesStarted, saves target).

Attribution: MoneyPuck.com — credit required when displaying derived results.
"""

import pandas as pd

from nhl_saves.moneypuck import (
    _mp_year,
    fetch_mp_goalie_career,
    fetch_mp_goalies,
    fetch_mp_player_bios,
    fetch_mp_teams,
)
from nhl_saves.store import build_goalie_game_logs
from nhl_saves.venues import trip_km as _trip_km

_DEFAULT_SEASONS = ["20212022", "20222023", "20232024", "20242025", "20252026"]


# ── Private helpers ───────────────────────────────────────────────────────────


def _lagged_rolling(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over previous `window` games (excludes current row)."""
    return series.shift(1).rolling(window, min_periods=window).mean()


def _parse_toi(toi_str) -> float:
    """Convert 'MM:SS' string to minutes as float. Returns NaN on failure."""
    try:
        if pd.isna(toi_str):
            return float("nan")
        parts = str(toi_str).split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except Exception:
        return float("nan")


def _team_home_streak(df: pd.DataFrame) -> pd.DataFrame:
    """Compute signed home/road streak per team per game.

    +N = N consecutive home games; -N = N consecutive road games.
    Merges onto df by (teamAbbrev, gameDate).
    """
    team_games = (
        df[["teamAbbrev", "gameDate", "homeRoadFlag", "gameId"]]
        .drop_duplicates(subset=["teamAbbrev", "gameDate"])
        .sort_values(["teamAbbrev", "gameDate"])
        .copy()
    )
    team_games["_hw"] = team_games["homeRoadFlag"].map({"H": 1, "R": -1}).fillna(0)

    streaks = []
    for team, grp in team_games.groupby("teamAbbrev"):
        grp = grp.sort_values("gameDate").copy()
        streak_vals = []
        cur_streak = 0
        for val in grp["_hw"]:
            if val == 0:
                cur_streak = 0
            elif cur_streak == 0:
                cur_streak = val
            elif (val > 0) == (cur_streak > 0):
                cur_streak += val
            else:
                cur_streak = val
            streak_vals.append(cur_streak)
        grp["team_home_streak"] = streak_vals
        streaks.append(grp[["teamAbbrev", "gameDate", "team_home_streak"]])

    streak_df = pd.concat(streaks, ignore_index=True)
    return df.merge(streak_df, on=["teamAbbrev", "gameDate"], how="left")


def _date_rolling(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    value_col: str,
    days: int,
    func: str = "sum",
) -> pd.Series:
    """Rolling window over past `days` days (excluding current row) per group.

    Returns a Series aligned to df.index.
    """
    result = pd.Series(float("nan"), index=df.index)
    for key, grp in df.groupby(group_col):
        grp = grp.sort_values(date_col)
        s = grp.set_index(date_col)[value_col]
        rolled = s.rolling(f"{days}D", closed="left").agg(func)
        result.loc[grp.index] = rolled.values
    return result


# ── Step 1: Load and combine NHL API game logs ─────────────────────────────────


def _load_all_logs(seasons: list[str], game_type: int) -> pd.DataFrame:
    """Load and concatenate game logs for all seasons.

    Adds a 'season' column to distinguish rows across seasons.
    """
    frames = []
    for s in seasons:
        df = build_goalie_game_logs(s, game_type)
        if not df.empty:
            df = df.copy()
            df["season"] = s
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["gameDate"] = pd.to_datetime(combined["gameDate"])
    return combined


# ── Step 2: Build per-team offensive + defensive time series from game logs ────


def _build_team_time_series(
    all_logs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-team offensive and defensive time series from all game logs.

    Offensive (opp perspective): rows where opponentAbbrev == X give
        shots X generated (shotsAgainst) and goals X scored (goalsAgainst).
    Defensive (team perspective): rows where teamAbbrev == X give
        shots X allowed (shotsAgainst).

    Returns (opp_ts, team_def_ts) DataFrames with rolling/YTD columns.
    """
    # ── Offensive time series ─────────────────────────────────────────────────
    opp_by_game = (
        all_logs.groupby(["opponentAbbrev", "gameDate"], as_index=False)
        .agg(shots_gen=("shotsAgainst", "first"), goals_gen=("goalsAgainst", "first"))
        .rename(columns={"opponentAbbrev": "team"})
        .sort_values(["team", "gameDate"])
    )

    opp_rows = []
    for team, grp in opp_by_game.groupby("team"):
        grp = grp.sort_values("gameDate").copy()
        grp["opp_sog_last_game"] = grp["shots_gen"].shift(1)
        grp["opp_sog_roll5"] = (
            grp["shots_gen"].shift(1).rolling(5, min_periods=1).mean()
        )
        grp["opp_sog_ytd"] = grp["shots_gen"].shift(1).expanding().mean()
        grp["opp_goals_roll5"] = (
            grp["goals_gen"].shift(1).rolling(5, min_periods=1).mean()
        )
        # Per-game conversion (goals / shots)
        game_conv = grp["goals_gen"] / grp["shots_gen"].replace(0, float("nan"))
        grp["opp_goal_conv_roll5"] = game_conv.shift(1).rolling(5, min_periods=1).mean()
        # YTD conversion using cumulative totals (avoids mean-of-means)
        cum_goals = grp["goals_gen"].shift(1).expanding().sum()
        cum_shots = grp["shots_gen"].shift(1).expanding().sum()
        grp["opp_goal_conv_ytd"] = cum_goals / cum_shots.replace(0, float("nan"))
        opp_rows.append(grp)

    opp_ts = pd.concat(opp_rows, ignore_index=True)

    # ── Defensive time series ─────────────────────────────────────────────────
    team_def_by_game = (
        all_logs.groupby(["teamAbbrev", "gameDate"], as_index=False)
        .agg(shots_allowed=("shotsAgainst", "first"))
        .rename(columns={"teamAbbrev": "team"})
        .sort_values(["team", "gameDate"])
    )

    def_rows = []
    for team, grp in team_def_by_game.groupby("team"):
        grp = grp.sort_values("gameDate").copy()
        grp["team_sog_allowed_last_game"] = grp["shots_allowed"].shift(1)
        grp["team_sog_allowed_roll5"] = (
            grp["shots_allowed"].shift(1).rolling(5, min_periods=1).mean()
        )
        grp["team_sog_allowed_ytd"] = grp["shots_allowed"].shift(1).expanding().mean()
        def_rows.append(grp)

    team_def_ts = pd.concat(def_rows, ignore_index=True)

    return opp_ts, team_def_ts


# ── Step 3: Load MoneyPuck career data for all player IDs ─────────────────────


def _load_mp_career(player_ids: list[int], seasons: list[str]) -> pd.DataFrame:
    """Load and combine MP per-game career data for all players.

    Parses gameDate to datetime, normalizes to match NHL API format.
    Filters to the requested seasons only.
    """
    mp_years = {_mp_year(s) for s in seasons}
    frames = []
    for pid in player_ids:
        try:
            career = fetch_mp_goalie_career(int(pid))
            if career.empty:
                continue
            career = career[career["season"].isin(mp_years)].copy()
            if career.empty:
                continue
            frames.append(career)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # Parse YYYYMMDD → datetime
    df["gameDate"] = pd.to_datetime(df["gameDate"].astype(str), format="%Y%m%d")
    return df


def _build_mp_career_wide(mp_career: pd.DataFrame) -> pd.DataFrame:
    """Pivot MP career from long (one row per situation) to wide.

    One row per (playerId, gameDate).
    Extracts 'all' situation aggregate stats plus situation-specific icetime.
    """
    if mp_career.empty:
        return pd.DataFrame()

    agg_cols = [
        "playerId",
        "gameDate",
        "playerTeam",
        "opposingTeam",
        "home_or_away",
        "season",
        "ongoal",
        "rebounds",
        "lowDangerShots",
        "mediumDangerShots",
        "highDangerShots",
        "lowDangerGoals",
        "mediumDangerGoals",
        "highDangerGoals",
        "blocked_shot_attempts",
        "unblocked_shot_attempts",
        "penalties",
        "penalityMinutes",
        "xGoals",
        "goals",
    ]
    existing = [c for c in agg_cols if c in mp_career.columns]
    all_sit = mp_career[mp_career["situation"] == "all"][existing].copy()

    sit_cols = [
        ("5on5", "icetime_5on5"), ("5on4", "icetime_5on4"), ("4on5", "icetime_4on5")
    ]
    for sit, col in sit_cols:
        if "icetime" not in mp_career.columns:
            break
        sit_df = mp_career[mp_career["situation"] == sit][
            ["playerId", "gameDate", "icetime"]
        ].rename(columns={"icetime": col})
        all_sit = all_sit.merge(sit_df, on=["playerId", "gameDate"], how="left")

    return all_sit


def _build_opp_hd_time_series(mp_career: pd.DataFrame) -> pd.DataFrame:
    """Build rolling high-danger shots time series for each team as offense.

    Uses MP career data: rows where opposingTeam == X.
    highDangerShots = shots X generated.
    """
    if mp_career.empty or "highDangerShots" not in mp_career.columns:
        return pd.DataFrame()

    all_sit = mp_career[mp_career["situation"] == "all"].copy()
    if all_sit.empty:
        return pd.DataFrame()

    opp_hd = (
        all_sit.groupby(["opposingTeam", "gameDate"], as_index=False)
        .agg(hd_shots_gen=("highDangerShots", "first"))
        .rename(columns={"opposingTeam": "team"})
        .sort_values(["team", "gameDate"])
    )

    rows = []
    for team, grp in opp_hd.groupby("team"):
        grp = grp.sort_values("gameDate").copy()
        grp["opp_hd_shots_roll5"] = (
            grp["hd_shots_gen"].shift(1).rolling(5, min_periods=1).mean()
        )
        grp["opp_hd_shots_ytd"] = grp["hd_shots_gen"].shift(1).expanding().mean()
        rows.append(grp)

    return pd.concat(rows, ignore_index=True)


def _build_team_hd_allowed_time_series(mp_career: pd.DataFrame) -> pd.DataFrame:
    """Build rolling high-danger shots allowed time series for each team as defense.

    Uses MP career data: rows where playerTeam == X, highDangerShots = shots X faced.
    """
    if mp_career.empty or "highDangerShots" not in mp_career.columns:
        return pd.DataFrame()

    all_sit = mp_career[mp_career["situation"] == "all"].copy()
    if all_sit.empty:
        return pd.DataFrame()

    team_hd = (
        all_sit.groupby(["playerTeam", "gameDate"], as_index=False)
        .agg(hd_shots_allowed=("highDangerShots", "first"))
        .rename(columns={"playerTeam": "team"})
        .sort_values(["team", "gameDate"])
    )

    rows = []
    for team, grp in team_hd.groupby("team"):
        grp = grp.sort_values("gameDate").copy()
        grp["team_hd_shots_allowed_roll5"] = (
            grp["hd_shots_allowed"].shift(1).rolling(5, min_periods=1).mean()
        )
        rows.append(grp)

    return pd.concat(rows, ignore_index=True)


# ── Step 4: Prior season features from MP season summaries ────────────────────


def _load_prior_season_goalie_features(season: str) -> pd.DataFrame:
    """Load MP goalie season summary for the prior season.

    Returns per-player features: sog_prior_season, save_pct_prior_season,
    hd_shots_prior_season (computed from 'all' situation rows).
    """
    prior_mp_year = _mp_year(season) - 1
    try:
        goalies = fetch_mp_goalies(str(prior_mp_year))
    except Exception:
        return pd.DataFrame()

    if goalies.empty:
        return pd.DataFrame()

    g = goalies[goalies["situation"] == "all"].copy()
    if g.empty or "games_played" not in g.columns:
        return pd.DataFrame()

    gp = g["games_played"].replace(0, float("nan"))
    g["sog_prior_season"] = g["ongoal"] / gp if "ongoal" in g.columns else float("nan")
    if "ongoal" in g.columns and "goals" in g.columns:
        g["save_pct_prior_season"] = (g["ongoal"] - g["goals"]) / g["ongoal"].replace(
            0, float("nan")
        )
    else:
        g["save_pct_prior_season"] = float("nan")
    g["hd_shots_prior_season"] = (
        g["highDangerShots"] / gp if "highDangerShots" in g.columns else float("nan")
    )

    keep = [
        "playerId", "sog_prior_season", "save_pct_prior_season", "hd_shots_prior_season"
    ]
    return g[keep]


def _load_prior_season_team_features(season: str) -> pd.DataFrame:
    """Load MP team season summary for the prior season.

    Returns per-team features for opponent and defensive context.
    """
    prior_mp_year = _mp_year(season) - 1
    try:
        teams = fetch_mp_teams(str(prior_mp_year))
    except Exception:
        return pd.DataFrame()

    if teams.empty:
        return pd.DataFrame()

    # Use first 'team' column (teams CSV has duplicate 'team' column)
    t = teams[teams["situation"] == "all"].copy()
    # Keep only the first 'team' column if duplicate
    t = t.loc[:, ~t.columns.duplicated()]

    if "games_played" not in t.columns or "team" not in t.columns:
        return pd.DataFrame()

    gp = t["games_played"].replace(0, float("nan"))

    t["opp_sog_prior_season"] = (
        t["shotsOnGoalFor"] / gp if "shotsOnGoalFor" in t.columns else float("nan")
    )
    t["opp_goal_conv_prior_season"] = (
        t["goalsFor"] / t["shotsOnGoalFor"].replace(0, float("nan"))
        if "goalsFor" in t.columns and "shotsOnGoalFor" in t.columns
        else float("nan")
    )
    t["opp_hd_shots_prior_season"] = (
        t["highDangerShotsFor"] / gp
        if "highDangerShotsFor" in t.columns
        else float("nan")
    )
    t["opp_corsi_pct_prior_season"] = (
        t["corsiPercentage"] if "corsiPercentage" in t.columns else float("nan")
    )
    t["team_sog_allowed_prior_season"] = (
        t["shotsOnGoalAgainst"] / gp
        if "shotsOnGoalAgainst" in t.columns
        else float("nan")
    )
    t["team_corsi_pct_prior_season"] = (
        t["corsiPercentage"] if "corsiPercentage" in t.columns else float("nan")
    )

    keep = [
        "team",
        "opp_sog_prior_season",
        "opp_goal_conv_prior_season",
        "opp_hd_shots_prior_season",
        "opp_corsi_pct_prior_season",
        "team_sog_allowed_prior_season",
        "team_corsi_pct_prior_season",
    ]
    return t[[c for c in keep if c in t.columns]]


def _load_current_season_team_corsi(season: str) -> pd.DataFrame:
    """Load MP team season summary for the current season (season-level Corsi%).

    Used as a proxy for YTD Corsi% (season average).
    """
    try:
        teams = fetch_mp_teams(season)
    except Exception:
        return pd.DataFrame()

    if teams.empty:
        return pd.DataFrame()

    t = teams[teams["situation"] == "all"].copy()
    t = t.loc[:, ~t.columns.duplicated()]

    if "team" not in t.columns or "corsiPercentage" not in t.columns:
        return pd.DataFrame()

    return t[["team", "corsiPercentage"]].rename(
        columns={"corsiPercentage": "season_corsi_pct"}
    )


# ── Step 5: Travel distance features ─────────────────────────────────────────


def _add_travel_features(starts: pd.DataFrame, all_logs: pd.DataFrame) -> pd.DataFrame:
    """Add travel distance and schedule density features to starts DataFrame.

    Goalie-level: trip_km, goalie_km_last_7d, goalie_starts_last_7d, days_since_home.
    Team-level: team_rest_days, team_is_back_to_back, team_km_last_7d,
        team_games_last_7d, team_cumulative_road_km.
    Opponent-level: opp_rest_days, opp_is_back_to_back, opp_km_last_7d,
        opp_games_last_7d.
    """
    starts = starts.copy()
    starts["gameDate"] = pd.to_datetime(starts["gameDate"])
    starts = starts.sort_values(["player_id", "gameDate"]).reset_index(drop=True)

    # Arena for each game = home team's arena
    starts["game_arena"] = starts["teamAbbrev"].where(
        starts["homeRoadFlag"] == "H", starts["opponentAbbrev"]
    )

    # Per-goalie: previous arena and trip distance
    grp = starts.groupby("player_id")
    starts["prev_arena"] = grp["game_arena"].transform(lambda s: s.shift(1))

    starts["trip_km"] = [
        _trip_km(r["prev_arena"], r["game_arena"])
        if pd.notna(r["prev_arena"])
        else float("nan")
        for _, r in starts.iterrows()
    ]

    # 7-day rolling travel + starts (date-based, closed='left')
    starts["_one"] = 1.0
    starts["goalie_km_last_7d"] = _date_rolling(
        starts, "player_id", "gameDate", "trip_km", 7, "sum"
    )
    starts["goalie_starts_last_7d"] = _date_rolling(
        starts, "player_id", "gameDate", "_one", 7, "sum"
    )
    starts = starts.drop(columns=["_one"])

    # days_since_home (last home start per goalie)
    def _dsince_home(grp_df: pd.DataFrame) -> pd.Series:
        grp_df = grp_df.sort_values("gameDate").copy()
        result = []
        last_home = None
        for _, row in grp_df.iterrows():
            if last_home is not None:
                result.append((row["gameDate"] - last_home).days)
            else:
                result.append(float("nan"))
            if row["homeRoadFlag"] == "H":
                last_home = row["gameDate"]
        return pd.Series(result, index=grp_df.index)

    starts["days_since_home"] = starts.groupby("player_id", group_keys=False).apply(
        _dsince_home, include_groups=False
    )

    # ── Team-level travel (from all_logs) ─────────────────────────────────────
    all_logs = all_logs.copy()
    all_logs["gameDate"] = pd.to_datetime(all_logs["gameDate"])

    # One row per team per game date
    team_games = (
        all_logs.groupby(["teamAbbrev", "gameDate"], as_index=False)
        .first()[["teamAbbrev", "gameDate", "homeRoadFlag", "opponentAbbrev"]]
        .sort_values(["teamAbbrev", "gameDate"])
        .reset_index(drop=True)
    )
    team_games["arena"] = team_games["teamAbbrev"].where(
        team_games["homeRoadFlag"] == "H", team_games["opponentAbbrev"]
    )
    team_games["prev_arena"] = team_games.groupby("teamAbbrev")["arena"].transform(
        lambda s: s.shift(1)
    )
    team_games["_trip_km"] = [
        _trip_km(r["prev_arena"], r["arena"]) if pd.notna(r["prev_arena"]) else 0.0
        for _, r in team_games.iterrows()
    ]
    team_games["_one"] = 1.0

    team_games["team_rest_days"] = team_games.groupby("teamAbbrev")[
        "gameDate"
    ].transform(lambda s: s.diff().dt.days)
    team_games["team_is_back_to_back"] = (team_games["team_rest_days"] <= 1).astype(int)

    team_games["team_km_last_7d"] = _date_rolling(
        team_games, "teamAbbrev", "gameDate", "_trip_km", 7, "sum"
    )
    team_games["team_games_last_7d"] = _date_rolling(
        team_games, "teamAbbrev", "gameDate", "_one", 7, "sum"
    )

    # Cumulative road km since last home game
    def _cum_road_km(grp_df: pd.DataFrame) -> pd.Series:
        grp_df = grp_df.sort_values("gameDate").copy()
        result = []
        cum = 0.0
        for _, row in grp_df.iterrows():
            result.append(cum)
            if row["homeRoadFlag"] == "H":
                cum = 0.0
            else:
                cum += float(row["_trip_km"]) if pd.notna(row["_trip_km"]) else 0.0
        return pd.Series(result, index=grp_df.index)

    team_games["team_cumulative_road_km"] = team_games.groupby(
        "teamAbbrev", group_keys=False
    ).apply(_cum_road_km, include_groups=False)

    team_feats = team_games[
        [
            "teamAbbrev",
            "gameDate",
            "team_rest_days",
            "team_is_back_to_back",
            "team_km_last_7d",
            "team_games_last_7d",
            "team_cumulative_road_km",
        ]
    ]
    starts = starts.merge(team_feats, on=["teamAbbrev", "gameDate"], how="left")

    # Opponent features (same team_games but keyed to opponentAbbrev)
    _opp_cols = [
        "teamAbbrev", "gameDate",
        "team_rest_days", "team_km_last_7d", "team_games_last_7d",
    ]
    opp_feats = team_games[_opp_cols].rename(
        columns={
            "teamAbbrev": "opponentAbbrev",
            "team_rest_days": "opp_rest_days",
            "team_km_last_7d": "opp_km_last_7d",
            "team_games_last_7d": "opp_games_last_7d",
        }
    )
    starts = starts.merge(opp_feats, on=["opponentAbbrev", "gameDate"], how="left")
    starts["opp_is_back_to_back"] = (starts["opp_rest_days"] <= 1).astype(int)

    return starts


# ── Main function ─────────────────────────────────────────────────────────────


def build_model_dataset(
    seasons: list[str] | str | None = None,
    game_type: int = 2,
) -> pd.DataFrame:
    """Build the full feature matrix for goalie saves prediction.

    Accepts seasons as a list of NHL season strings (e.g. ["20242025", "20252026"]),
    a single season string, or None (defaults to all 5 configured seasons).

    Only includes starts (gamesStarted == 1).
    Rolling and YTD features are computed per-goalie, per-season to prevent
    cross-season leakage. Prior-season features use MoneyPuck season summaries.

    Attribution: MoneyPuck.com
    """
    if seasons is None:
        seasons = _DEFAULT_SEASONS
    elif isinstance(seasons, str):
        seasons = [seasons]

    # ── Load all game logs (spine) ────────────────────────────────────────────
    all_logs = _load_all_logs(seasons, game_type)
    if all_logs.empty:
        return pd.DataFrame()

    # Starts only (for goalie-level features)
    starts_all = all_logs[all_logs["gamesStarted"] == 1].copy()
    if starts_all.empty:
        return pd.DataFrame()

    # ── Load MP career data for all players ───────────────────────────────────
    player_ids = starts_all["player_id"].dropna().unique().tolist()
    mp_career_raw = _load_mp_career(player_ids, seasons)
    mp_wide = _build_mp_career_wide(mp_career_raw)

    # MP high-danger time series for opponent and team contexts
    opp_hd_ts = _build_opp_hd_time_series(mp_career_raw)
    team_hd_allowed_ts = _build_team_hd_allowed_time_series(mp_career_raw)

    # ── Load bios ─────────────────────────────────────────────────────────────
    try:
        bios = fetch_mp_player_bios()
        bios = bios[["playerId", "birthDate", "nationality"]].copy()
        bios["birthDate"] = pd.to_datetime(bios["birthDate"], errors="coerce")
    except Exception:
        bios = pd.DataFrame(columns=["playerId", "birthDate", "nationality"])

    # ── Process each season independently (no cross-season rolling leakage) ───
    season_frames = []

    for season in seasons:
        # Filter all_logs and starts to this season
        logs_s = all_logs[all_logs["season"] == season].copy()
        starts_s = starts_all[starts_all["season"] == season].copy()

        if starts_s.empty:
            continue

        starts_s = (
            starts_s.sort_values(["player_id", "gameDate"]).reset_index(drop=True)
        )

        # ── Target ────────────────────────────────────────────────────────────
        starts_s["saves"] = starts_s["shotsAgainst"] - starts_s["goalsAgainst"]
        starts_s["shots_against"] = starts_s["shotsAgainst"]
        starts_s["goals_against"] = starts_s["goalsAgainst"]
        starts_s["toi_total"] = starts_s["toi"].apply(_parse_toi)

        # ── Goalie rolling features (within season only) ──────────────────────
        grp = starts_s.groupby("player_id")

        starts_s["save_pct_roll5"] = grp["savePctg"].transform(
            lambda s: _lagged_rolling(s, 5)
        )
        starts_s["save_pct_roll10"] = grp["savePctg"].transform(
            lambda s: _lagged_rolling(s, 10)
        )
        starts_s["sog_roll5"] = grp["shotsAgainst"].transform(
            lambda s: _lagged_rolling(s, 5)
        )
        starts_s["sog_roll10"] = grp["shotsAgainst"].transform(
            lambda s: _lagged_rolling(s, 10)
        )
        starts_s["sog_ytd"] = grp["shotsAgainst"].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        starts_s["save_pct_ytd"] = grp["savePctg"].transform(
            lambda s: s.shift(1).expanding().mean()
        )

        # ── Rest days, context flags ───────────────────────────────────────────
        starts_s["rest_days"] = grp["gameDate"].transform(lambda s: s.diff().dt.days)
        starts_s["is_home"] = (starts_s["homeRoadFlag"] == "H").astype(int)
        starts_s["is_back_to_back"] = (starts_s["rest_days"] <= 1).astype(int)
        starts_s["prev_opponent"] = grp["opponentAbbrev"].transform(
            lambda s: s.shift(1)
        )
        starts_s["prev_home_away"] = grp["homeRoadFlag"].transform(
            lambda s: s.shift(1)
        )
        starts_s["goalie_starts_ytd"] = grp.cumcount()

        # ── Team home streak ───────────────────────────────────────────────────
        starts_s = _team_home_streak(starts_s)

        # ── Team time series (opponent offensive + team defensive) ────────────
        opp_ts, team_def_ts = _build_team_time_series(logs_s)

        # Join opponent offensive context
        opp_merge = opp_ts.rename(columns={"team": "opponentAbbrev"})
        starts_s = starts_s.merge(
            opp_merge[
                [
                    "opponentAbbrev",
                    "gameDate",
                    "opp_sog_last_game",
                    "opp_sog_roll5",
                    "opp_sog_ytd",
                    "opp_goals_roll5",
                    "opp_goal_conv_roll5",
                    "opp_goal_conv_ytd",
                ]
            ],
            on=["opponentAbbrev", "gameDate"],
            how="left",
        )

        # Join team defensive context
        team_def_merge = team_def_ts.rename(columns={"team": "teamAbbrev"})
        starts_s = starts_s.merge(
            team_def_merge[
                [
                    "teamAbbrev",
                    "gameDate",
                    "team_sog_allowed_last_game",
                    "team_sog_allowed_roll5",
                    "team_sog_allowed_ytd",
                ]
            ],
            on=["teamAbbrev", "gameDate"],
            how="left",
        )

        # ── Join MP per-game career features ──────────────────────────────────
        mp_year = _mp_year(season)
        mp_s = (
            mp_wide[mp_wide["season"] == mp_year].copy()
            if not mp_wide.empty
            else pd.DataFrame()
        )

        if not mp_s.empty:
            # Normalize playerId to match player_id
            mp_s["player_id"] = mp_s["playerId"].astype(int)

            mp_merge_cols = [
                "player_id",
                "gameDate",
                "ongoal",
                "rebounds",
                "lowDangerShots",
                "mediumDangerShots",
                "highDangerShots",
                "blocked_shot_attempts",
                "unblocked_shot_attempts",
                "penalties",
                "penalityMinutes",
            ]
            for col in ["icetime_5on5", "icetime_5on4", "icetime_4on5"]:
                if col in mp_s.columns:
                    mp_merge_cols.append(col)

            mp_merge_cols = [c for c in mp_merge_cols if c in mp_s.columns]
            starts_s = starts_s.merge(
                mp_s[mp_merge_cols],
                on=["player_id", "gameDate"],
                how="left",
            )

            # Derived per-game MP features
            has_attempts = (
                "unblocked_shot_attempts" in starts_s.columns
                and "blocked_shot_attempts" in starts_s.columns
            )
            if has_attempts:
                starts_s["shots_all_attempts"] = (
                    starts_s["unblocked_shot_attempts"].fillna(0)
                    + starts_s["blocked_shot_attempts"].fillna(0)
                )
            if "icetime_5on5" in starts_s.columns:
                starts_s["toi_5v5"] = starts_s["icetime_5on5"] / 60
            if "icetime_5on4" in starts_s.columns:
                starts_s["toi_pp"] = starts_s["icetime_5on4"] / 60
            if "icetime_4on5" in starts_s.columns:
                starts_s["toi_pk"] = starts_s["icetime_4on5"] / 60

            # Rename MP columns to model column names
            starts_s = starts_s.rename(
                columns={
                    "penalties": "penalties_against",
                    "lowDangerShots": "low_danger_shots",
                    "mediumDangerShots": "med_danger_shots",
                    "highDangerShots": "high_danger_shots",
                }
            )

            # Rolling MP features (per goalie, within season)
            if "high_danger_shots" in starts_s.columns:
                gmp = starts_s.groupby("player_id")
                starts_s["hd_shots_roll5"] = gmp["high_danger_shots"].transform(
                    lambda s: _lagged_rolling(s, 5)
                )
                starts_s["hd_shots_roll10"] = gmp["high_danger_shots"].transform(
                    lambda s: _lagged_rolling(s, 10)
                )
                starts_s["hd_shots_ytd"] = gmp["high_danger_shots"].transform(
                    lambda s: s.shift(1).expanding().mean()
                )
            if "rebounds" in starts_s.columns:
                starts_s["rebounds_roll5"] = starts_s.groupby("player_id")[
                    "rebounds"
                ].transform(lambda s: _lagged_rolling(s, 5))

        # ── Opponent + team HD shots from MP career time series ───────────────
        if not opp_hd_ts.empty:
            opp_hd_merge = opp_hd_ts.rename(columns={"team": "opponentAbbrev"})
            _ohd_cols = [
                "opponentAbbrev", "gameDate", "opp_hd_shots_roll5", "opp_hd_shots_ytd"
            ]
            _ohd_cols = [c for c in _ohd_cols if c in opp_hd_merge.columns]
            starts_s = starts_s.merge(
                opp_hd_merge[_ohd_cols],
                on=["opponentAbbrev", "gameDate"],
                how="left",
            )

        if not team_hd_allowed_ts.empty:
            team_hd_merge = team_hd_allowed_ts.rename(columns={"team": "teamAbbrev"})
            _thd_cols = ["teamAbbrev", "gameDate", "team_hd_shots_allowed_roll5"]
            starts_s = starts_s.merge(
                team_hd_merge[_thd_cols],
                on=["teamAbbrev", "gameDate"],
                how="left",
            )

        # ── Current season team Corsi% (season-level proxy for YTD) ──────────
        corsi_ytd = _load_current_season_team_corsi(season)
        if not corsi_ytd.empty:
            starts_s = starts_s.merge(
                corsi_ytd.rename(columns={
                    "team": "opponentAbbrev",
                    "season_corsi_pct": "opp_corsi_pct_ytd",
                }),
                on="opponentAbbrev",
                how="left",
            )
            starts_s = starts_s.merge(
                corsi_ytd.rename(columns={
                    "team": "teamAbbrev",
                    "season_corsi_pct": "team_corsi_pct_ytd",
                }),
                on="teamAbbrev",
                how="left",
            )

        # ── Prior season features ─────────────────────────────────────────────
        prior_goalie = _load_prior_season_goalie_features(season)
        if not prior_goalie.empty:
            prior_goalie["player_id"] = prior_goalie["playerId"].astype(int)
            _pg_cols = [
                "player_id", "sog_prior_season",
                "save_pct_prior_season", "hd_shots_prior_season",
            ]
            starts_s = starts_s.merge(
                prior_goalie[_pg_cols],
                on="player_id",
                how="left",
            )

        prior_teams = _load_prior_season_team_features(season)
        if not prior_teams.empty:
            starts_s = starts_s.merge(
                prior_teams.rename(columns={"team": "opponentAbbrev"})[[
                    "opponentAbbrev",
                    "opp_sog_prior_season",
                    "opp_goal_conv_prior_season",
                    "opp_hd_shots_prior_season",
                    "opp_corsi_pct_prior_season",
                ]],
                on="opponentAbbrev",
                how="left",
            )
            starts_s = starts_s.merge(
                prior_teams.rename(columns={"team": "teamAbbrev"})[[
                    "teamAbbrev",
                    "team_sog_allowed_prior_season",
                    "team_corsi_pct_prior_season",
                ]],
                on="teamAbbrev",
                how="left",
            )

        # ── Biographical features ─────────────────────────────────────────────
        if not bios.empty:
            bios_merge = bios.copy()
            bios_merge["player_id"] = bios_merge["playerId"].astype(int)
            starts_s = starts_s.merge(
                bios_merge[["player_id", "birthDate", "nationality"]],
                on="player_id",
                how="left",
            )
            if "birthDate" in starts_s.columns:
                starts_s["goalie_age"] = (
                    starts_s["gameDate"] - starts_s["birthDate"]
                ).dt.days / 365.25
            starts_s = starts_s.rename(columns={"nationality": "goalie_nationality"})

        season_frames.append(starts_s)

    if not season_frames:
        return pd.DataFrame()

    df = pd.concat(season_frames, ignore_index=True)

    # ── Travel features (computed on full combined dataset) ───────────────────
    df = _add_travel_features(df, all_logs)

    # ── Drop rows missing roll10 features (first 10 starts per goalie/season) ─
    df = df.dropna(subset=["save_pct_roll10", "sog_roll10"]).reset_index(drop=True)

    # ── Final feature selection ───────────────────────────────────────────────
    feature_cols = [
        # Identifiers
        "player_id",
        "gameDate",
        "season",
        "teamAbbrev",
        "opponentAbbrev",
        # Target
        "saves",
        # Base game stats
        "shots_against",
        "goals_against",
        "toi_total",
        "toi_5v5",
        "toi_pp",
        "toi_pk",
        "shots_all_attempts",
        "rebounds",
        "low_danger_shots",
        "med_danger_shots",
        "high_danger_shots",
        "penalties_against",
        # Goalie rolling
        "save_pct_roll5",
        "save_pct_roll10",
        "sog_roll5",
        "sog_roll10",
        "hd_shots_roll5",
        "hd_shots_roll10",
        "rebounds_roll5",
        # Goalie YTD
        "sog_ytd",
        "save_pct_ytd",
        "hd_shots_ytd",
        # Prior season (goalie)
        "sog_prior_season",
        "save_pct_prior_season",
        "hd_shots_prior_season",
        # Goalie schedule
        "rest_days",
        "is_back_to_back",
        "days_since_home",
        "trip_km",
        "goalie_km_last_7d",
        "goalie_starts_last_7d",
        # Team schedule
        "team_rest_days",
        "team_is_back_to_back",
        "team_games_last_7d",
        "team_km_last_7d",
        "team_cumulative_road_km",
        "team_home_streak",
        # Game context
        "is_home",
        "prev_opponent",
        "prev_home_away",
        "goalie_starts_ytd",
        # Biographical
        "goalie_age",
        "goalie_nationality",
        # Opponent offensive context
        "opp_sog_last_game",
        "opp_sog_roll5",
        "opp_sog_ytd",
        "opp_sog_prior_season",
        "opp_goals_roll5",
        "opp_goal_conv_roll5",
        "opp_goal_conv_ytd",
        "opp_goal_conv_prior_season",
        "opp_hd_shots_roll5",
        "opp_hd_shots_ytd",
        "opp_hd_shots_prior_season",
        "opp_corsi_pct_ytd",
        "opp_corsi_pct_prior_season",
        # Team defensive context
        "team_sog_allowed_last_game",
        "team_sog_allowed_roll5",
        "team_sog_allowed_ytd",
        "team_sog_allowed_prior_season",
        "team_hd_shots_allowed_roll5",
        "team_corsi_pct_ytd",
        "team_corsi_pct_prior_season",
        # Opponent schedule
        "opp_rest_days",
        "opp_is_back_to_back",
        "opp_games_last_7d",
        "opp_km_last_7d",
    ]

    available = [c for c in feature_cols if c in df.columns]
    return df[available].reset_index(drop=True)
