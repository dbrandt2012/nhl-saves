"""NHL Saves Dashboard — Page 2: Goalie Detail."""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from nhl_saves.stats import (
    days_since_last_game,
    goalie_report,
    team_save_pct_stats,
    vs_opponent_sog_season,
)
from nhl_saves.store import (
    NHL_TEAMS,
    build_goalie_game_logs,
    fetch_schedule,
    fetch_team_goalie_ids,
)
from nhl_saves.ui import (
    box_plot,
    load_goalie_log,
    render_stats_guide_goalie,
    stat_block,
)

# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------


def current_season() -> str:
    today = date.today()
    year = today.year if today.month >= 9 else today.year - 1
    return f"{year}{year + 1}"


SEASON = current_season()

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def load_schedule() -> pd.DataFrame:
    return fetch_schedule()


@st.cache_data(ttl=3600)
def load_all_goalie_logs(season: str) -> pd.DataFrame:
    processed_path = Path("data/processed/goalie_game_logs") / f"{season}_2.parquet"
    if processed_path.exists():
        return pd.read_parquet(processed_path)
    return build_goalie_game_logs(season)


@st.cache_data(ttl=3600)
def load_team_goalie_ids(team_abbrev: str, season: str) -> list[dict]:
    return fetch_team_goalie_ids(team_abbrev, season)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _next_opponent(games_df: pd.DataFrame, team: str) -> str | None:
    upcoming = games_df[
        (games_df["homeTeam"] == team) | (games_df["awayTeam"] == team)
    ].sort_values("gameDate")
    if upcoming.empty:
        return None
    row = upcoming.iloc[0]
    return row["awayTeam"] if row["homeTeam"] == team else row["homeTeam"]


def _summary_group(
    col,
    label: str,
    stat_dict: dict,
    decimals: int = 1,
    pct: bool = False,
    sv_pct: bool = False,
) -> None:
    def _fmt(val: float) -> str:
        if pd.isna(val):
            return "N/A"
        if sv_pct:
            return f"{val / 100:.3f}"
        s = f"{val:.{decimals}f}"
        if s.endswith(".0"):
            s = s[:-2]
        return f"{s}{'%' if pct else ''}"

    last_n_median = stat_dict.get("last_n", {}).get("mean", float("nan"))
    season_median = stat_dict.get("season", {}).get("median", float("nan"))
    col.caption(label)
    col.metric("L5", _fmt(last_n_median))
    col.metric("Season", _fmt(season_median))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.caption(f"Season: {SEASON[:4]}–{SEASON[4:]}")

# Team — all 32, alphabetical
team = st.sidebar.selectbox("Team", sorted(NHL_TEAMS))

# Goalie — from bulk stats, sorted by starts desc (starter first)
goalie_ids = load_team_goalie_ids(team, SEASON)
if not goalie_ids:
    st.warning(f"No goalie data available for {team}.")
    st.stop()

names = [g["name"] for g in goalie_ids]
sel_name = st.sidebar.selectbox("Goalie", names)
player_id = goalie_ids[names.index(sel_name)]["player_id"]

# Opponent — default = team's next scheduled game
with st.spinner("Loading schedule…"):
    games_df = load_schedule()

nhl_set = set(NHL_TEAMS)
games_df = games_df[
    games_df["homeTeam"].isin(nhl_set) & games_df["awayTeam"].isin(nhl_set)
].reset_index(drop=True)

default_opp = _next_opponent(games_df, team)
opp_opts = sorted(NHL_TEAMS)
opp_idx = opp_opts.index(default_opp) if default_opp in opp_opts else 0
opponent = st.sidebar.selectbox("Opponent", opp_opts, index=opp_idx)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with st.spinner("Loading goalie game logs…"):
    goalie_log = load_goalie_log(player_id, SEASON)
    all_logs = load_all_goalie_logs(SEASON)

if goalie_log.empty:
    st.warning(f"No game log data available for {sel_name}.")
    st.stop()

report = goalie_report(goalie_log, all_logs, opponent)
started = goalie_log[goalie_log["gamesStarted"] == 1].sort_values("gameDate")

# Determine next game date for display
next_game_date = None
upcoming_team = games_df[
    (games_df["homeTeam"] == team) | (games_df["awayTeam"] == team)
].sort_values("gameDate")
if not upcoming_team.empty:
    next_game_date = upcoming_team.iloc[0]["gameDate"]

# Pre-compute team per-game data (used in L5 strip and box plots)
team_games_logs = all_logs[all_logs["teamAbbrev"] == team]
if not team_games_logs.empty:
    team_per_game = (
        team_games_logs.groupby("gameId")
        .agg(
            shotsAgainst=("shotsAgainst", "sum"),
            goalsAgainst=("goalsAgainst", "sum"),
            gameDate=("gameDate", "max"),
            opponentAbbrev=("opponentAbbrev", "first"),
        )
        .reset_index()
        .sort_values("gameDate")
    )
else:
    team_per_game = pd.DataFrame(
        columns=["shotsAgainst", "goalsAgainst", "gameDate", "opponentAbbrev"]
    )

# Pre-compute opponent per-game data (used in L5 strip and Stat 3/4)
opp_games = all_logs[all_logs["opponentAbbrev"] == opponent].copy()
if not opp_games.empty:
    opp_per_game = (
        opp_games.groupby("gameId")
        .agg(
            shotsFor=("shotsAgainst", "sum"),
            goalsFor=("goalsAgainst", "sum"),
            gameDate=("gameDate", "max"),
            playedAgainst=("teamAbbrev", "first"),
        )
        .reset_index()
        .sort_values("gameDate")
    )
else:
    opp_per_game = pd.DataFrame(
        columns=["gameId", "shotsFor", "goalsFor", "gameDate", "playedAgainst"]
    )

# Per-game GF for `team` — derived from rows where opponent was `team`
team_gf_per_game = (
    all_logs[all_logs["opponentAbbrev"] == team]
    .groupby("gameId")["goalsAgainst"]
    .sum()
)

# Merge team GF into team_per_game
if not team_per_game.empty:
    team_per_game = team_per_game.merge(
        team_gf_per_game.rename("goalsFor").reset_index(),
        on="gameId",
        how="left",
    )
else:
    team_per_game["goalsFor"] = pd.Series(dtype=float)

# Per-game GA for `opponent` — derived from rows where teamAbbrev was `opponent`
opp_ga_per_game = (
    all_logs[all_logs["teamAbbrev"] == opponent]
    .groupby("gameId")["goalsAgainst"]
    .sum()
)

# Merge opponent GA into opp_per_game
if not opp_per_game.empty:
    opp_per_game = opp_per_game.merge(
        opp_ga_per_game.rename("goalsAgainst").reset_index(),
        on="gameId",
        how="left",
    )
else:
    opp_per_game["goalsAgainst"] = pd.Series(dtype=float)


def _result_str(gf, ga, decision=None) -> str:
    """Format a game result as 'W (3-1)' or 'L (2-3)'."""
    if pd.isna(gf) or pd.isna(ga):
        return str(decision) if pd.notna(decision) else "—"
    gf_i, ga_i = int(gf), int(ga)
    if pd.notna(decision):
        outcome = str(decision)
    else:
        outcome = "W" if gf_i > ga_i else "L" if gf_i < ga_i else "?"
    return f"{outcome} ({gf_i}-{ga_i})"

# ---------------------------------------------------------------------------
# Stats guide
# ---------------------------------------------------------------------------

render_stats_guide_goalie()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.header(f"{sel_name} vs {opponent}")
if next_game_date:
    try:
        date_str = pd.to_datetime(next_game_date).strftime("%a %b %-d")
    except Exception:
        date_str = str(next_game_date)
    st.caption(f"Next game: {date_str}  ·  {team}")

# ---------------------------------------------------------------------------
# Quick Summary
# ---------------------------------------------------------------------------

st.subheader("Quick Summary")

# Goalie row
st.caption(sel_name)
summary_cols = st.columns(6)

_summary_group(summary_cols[0], "SOG Allowed", report["sog_allowed"]["goalie"])
_summary_group(summary_cols[1], "Save %", report["save_pct"], sv_pct=True)
_summary_group(summary_cols[2], "Opp SOG", report["opponent_sog"])
_summary_group(
    summary_cols[3],
    "Opp Goal Conv%",
    report["opponent_goal_rate"],
    decimals=1,
    pct=True,
)

vs_sog = report["vs_opponent_sog"]
n = int(vs_sog.get("n", 0))
mean_val = vs_sog.get("mean", float("nan"))
summary_cols[4].caption(f"vs {opponent}")
if n == 0:
    summary_cols[4].metric("Avg SOG", "No games")
else:
    s = f"{mean_val:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    summary_cols[4].metric("Avg SOG", s if not pd.isna(mean_val) else "N/A")
    summary_cols[4].caption(f"n={n}")

goalie_days = report["days_since"]
summary_cols[5].caption("Days Since")
summary_cols[5].metric(
    "Last Start", str(goalie_days) if goalie_days is not None else "—"
)

# Team row
st.caption(team)
team_summary_cols = st.columns(6)
team_svp = team_save_pct_stats(all_logs, team)
team_vs_sog = vs_opponent_sog_season(all_logs, team, opponent)
team_days = days_since_last_game(team_per_game) if not team_per_game.empty else None

_summary_group(team_summary_cols[0], "SOG Allowed", report["sog_allowed"]["team"])
_summary_group(team_summary_cols[1], "Save %", team_svp, sv_pct=True)
_summary_group(team_summary_cols[2], "Opp SOG", report["opponent_sog"])
_summary_group(
    team_summary_cols[3],
    "Opp Goal Conv%",
    report["opponent_goal_rate"],
    decimals=1,
    pct=True,
)

t_n = int(team_vs_sog.get("n", 0))
t_mean = team_vs_sog.get("mean", float("nan"))
team_summary_cols[4].caption(f"vs {opponent}")
if t_n == 0:
    team_summary_cols[4].metric("Avg SOG", "No games")
else:
    ts = f"{t_mean:.1f}"
    if ts.endswith(".0"):
        ts = ts[:-2]
    team_summary_cols[4].metric("Avg SOG", ts if not pd.isna(t_mean) else "N/A")
    team_summary_cols[4].caption(f"n={t_n}")

team_summary_cols[5].caption("Days Since")
team_summary_cols[5].metric(
    "Last Game", str(team_days) if team_days is not None else "—"
)

# ---------------------------------------------------------------------------
# Last 5 Games tables
# ---------------------------------------------------------------------------

st.subheader("Last 5 Games")

# Goalie table
st.caption(f"{sel_name} (starts)")
goalie_l5 = started.tail(5).copy()
if goalie_l5.empty:
    st.caption("No starts recorded.")
else:
    goalie_l5["Date"] = pd.to_datetime(goalie_l5["gameDate"]).dt.strftime("%b %-d")
    goalie_l5["goalsFor"] = goalie_l5["gameId"].map(team_gf_per_game)
    goalie_l5["Result"] = goalie_l5.apply(
        lambda r: _result_str(r["goalsFor"], r["goalsAgainst"], r.get("decision")),
        axis=1,
    )
    goalie_l5["Sv%"] = goalie_l5["savePctg"].apply(
        lambda v: f"{v:.3f}" if pd.notna(v) else "—"
    )
    goalie_l5_display = goalie_l5.rename(
        columns={"opponentAbbrev": "Opp", "shotsAgainst": "SOG Allowed"}
    )[["Date", "Opp", "Result", "SOG Allowed", "Sv%"]]
    st.dataframe(goalie_l5_display, hide_index=True, width="stretch")

# Team table
st.caption(f"{team} (all goalies)")
team_l5 = team_per_game.tail(5).copy()
if team_l5.empty:
    st.caption("No games recorded.")
else:
    team_l5["Date"] = pd.to_datetime(team_l5["gameDate"]).dt.strftime("%b %-d")
    team_l5["Result"] = team_l5.apply(
        lambda r: _result_str(r.get("goalsFor"), r["goalsAgainst"]), axis=1
    )
    team_l5_display = team_l5.rename(
        columns={"opponentAbbrev": "Opp", "shotsAgainst": "SOG Allowed"}
    )[["Date", "Opp", "Result", "SOG Allowed"]]
    st.dataframe(team_l5_display, hide_index=True, width="stretch")

# Opponent offense table
st.caption(f"{opponent} (offense)")
opp_l5 = opp_per_game.tail(5).copy()
if opp_l5.empty:
    st.caption("No games recorded.")
else:
    opp_l5["Date"] = pd.to_datetime(opp_l5["gameDate"]).dt.strftime("%b %-d")
    opp_l5["Result"] = opp_l5.apply(
        lambda r: _result_str(r["goalsFor"], r.get("goalsAgainst")), axis=1
    )
    opp_l5["Goal Conv%"] = opp_l5.apply(
        lambda row: (
            f"{row['goalsFor'] / row['shotsFor'] * 100:.1f}%"
            if pd.notna(row["shotsFor"]) and row["shotsFor"] > 0
            else "—"
        ),
        axis=1,
    )
    opp_l5_display = opp_l5.rename(
        columns={"playedAgainst": "Opp", "shotsFor": "SOG"}
    )[["Date", "Opp", "Result", "SOG", "Goal Conv%"]]
    st.dataframe(opp_l5_display, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Stat 1: SOG Allowed
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Shots on Goal Allowed")

# Team row
st.caption(f"Team ({team}) — all goalies")
c_left, c_right = st.columns(2)
with c_left:
    stat_block(report["sog_allowed"]["team"], label="")
with c_right:
    if not team_per_game.empty:
        hover = team_per_game["gameDate"].astype(str).tolist()
        box_plot(
            [("Season", team_per_game["shotsAgainst"], hover)],
            y_label="SOG",
        )

# Goalie row
st.caption(f"{sel_name} starts only")
c_left, c_right = st.columns(2)
with c_left:
    stat_block(report["sog_allowed"]["goalie"], label="")
with c_right:
    if not started.empty:
        hover = [
            f"{row['gameDate']} vs {row['opponentAbbrev']}"
            for _, row in started.iterrows()
        ]
        box_plot(
            [("Season", started["shotsAgainst"], hover)],
            y_label="SOG",
        )

# ---------------------------------------------------------------------------
# Stat 2: Save Percentage
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Save Percentage")
c_left, c_right = st.columns(2)
with c_left:
    stat_block(report["save_pct"], label="", sv_pct=True)
with c_right:
    if not started.empty and "savePctg" in started.columns:
        hover = [
            f"{row['gameDate']} vs {row['opponentAbbrev']}"
            for _, row in started.iterrows()
        ]
        box_plot(
            [("Season", started["savePctg"], hover)],
            y_label="Save %",
            y_fmt=".3f",
        )

# ---------------------------------------------------------------------------
# Stat 3: Opponent SOG
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Opponent Shots on Goal")
c_left, c_right = st.columns(2)
with c_left:
    stat_block(report["opponent_sog"], label="")
with c_right:
    if not opp_per_game.empty:
        hover = opp_per_game["gameDate"].astype(str).tolist()
        box_plot(
            [("Season", opp_per_game["shotsFor"], hover)],
            y_label="Opp SOG",
        )

# ---------------------------------------------------------------------------
# Stat 4: Opponent Goal Conversion %
# ---------------------------------------------------------------------------

# Compute goal_rate in 0-1 scale (pct=True will multiply by 100 in box_plot)
if not opp_per_game.empty:
    opp_with_shots = opp_per_game[opp_per_game["shotsFor"] > 0].copy()
    opp_with_shots["goal_rate"] = (
        opp_with_shots["goalsFor"] / opp_with_shots["shotsFor"]
    )
else:
    opp_with_shots = pd.DataFrame(columns=["goal_rate", "gameDate"])

st.divider()
st.subheader("Opponent Goal Conversion %")
c_left, c_right = st.columns(2)
with c_left:
    stat_block(report["opponent_goal_rate"], label="", decimals=1, pct=True)
with c_right:
    if not opp_with_shots.empty:
        hover = opp_with_shots["gameDate"].astype(str).tolist()
        box_plot(
            [("Season", opp_with_shots["goal_rate"], hover)],
            y_label="Goal Conv %",
            pct=True,
        )

# ---------------------------------------------------------------------------
# Stat 5: History vs Opponent
# ---------------------------------------------------------------------------

vs_df = report["vs_opponent"]

st.divider()
st.subheader(f"vs {opponent} This Season")

# Team-level vs-opponent aggregation
team_vs_opp_raw = all_logs[
    (all_logs["teamAbbrev"] == team) & (all_logs["opponentAbbrev"] == opponent)
]
if not team_vs_opp_raw.empty:
    team_vs_opp = (
        team_vs_opp_raw.groupby("gameId")
        .agg(
            gameDate=("gameDate", "max"),
            shotsAgainst=("shotsAgainst", "sum"),
            goalsAgainst=("goalsAgainst", "sum"),
        )
        .reset_index()
        .sort_values("gameDate", ascending=False)
    )
    team_vs_opp = team_vs_opp.merge(
        team_gf_per_game.rename("goalsFor").reset_index(),
        on="gameId",
        how="left",
    )
    team_vs_opp["Result"] = team_vs_opp.apply(
        lambda r: _result_str(r.get("goalsFor"), r["goalsAgainst"]), axis=1
    )
    team_vs_opp["Sv%"] = (
        (team_vs_opp["shotsAgainst"] - team_vs_opp["goalsAgainst"])
        / team_vs_opp["shotsAgainst"]
    ).apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    team_vs_opp_display = team_vs_opp.rename(
        columns={"gameDate": "Date", "shotsAgainst": "SOG Allowed"}
    )[["Date", "Result", "SOG Allowed", "Sv%"]]

st.caption(f"{team} (all goalies)")
if team_vs_opp_raw.empty:
    st.info(f"No {team} games vs {opponent} this season.")
else:
    st.dataframe(team_vs_opp_display, hide_index=True, width="stretch")

# Goalie-level vs-opponent table
vs_df_display = vs_df.copy()
if "gameId" in vs_df_display.columns:
    vs_df_display["goalsFor"] = vs_df_display["gameId"].map(team_gf_per_game)
else:
    vs_df_display["goalsFor"] = float("nan")
vs_df_display["Result"] = vs_df_display.apply(
    lambda r: _result_str(r["goalsFor"], r["goalsAgainst"], r.get("decision")),
    axis=1,
)
if "toi" in vs_df_display.columns:
    def _fmt_toi(v):
        if pd.isna(v):
            return "—"
        try:
            secs = int(v)
            return f"{secs // 60}:{secs % 60:02d}"
        except (ValueError, TypeError):
            return str(v)
    vs_df_display["toi"] = vs_df_display["toi"].apply(_fmt_toi)
vs_df_display = vs_df_display.rename(
    columns={
        "gameDate": "Date",
        "homeRoadFlag": "H/A",
        "shotsAgainst": "SOG Allowed",
        "savePctg": "Sv%",
        "toi": "TOI",
    }
)
if "Sv%" in vs_df_display.columns:
    vs_df_display["Sv%"] = vs_df_display["Sv%"].apply(
        lambda v: f"{v:.3f}" if pd.notna(v) else "—"
    )

st.caption(f"{sel_name}")
if vs_df.empty:
    st.info(f"No starts by {sel_name} vs {opponent} this season.")
else:
    display_cols = [c for c in ["Date", "Result", "SOG Allowed", "Sv%", "TOI"]
                    if c in vs_df_display.columns]
    st.dataframe(vs_df_display[display_cols], hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Stats guide
# ---------------------------------------------------------------------------

render_stats_guide_goalie()
