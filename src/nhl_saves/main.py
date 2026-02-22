"""NHL Goalie Save Tracker — Streamlit UI."""

import math
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nhl_saves.stats import goalie_report
from nhl_saves.store import (
    build_goalie_game_logs,
    fetch_goalie_game_log,
    fetch_goalie_stats,
    fetch_next_games,
)

st.set_page_config(page_title="NHL Goalie Save Tracker", layout="wide")


# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------


def current_season() -> str:
    today = date.today()
    year = today.year if today.month >= 9 else today.year - 1
    return f"{year}{year + 1}"


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def load_upcoming_games(season: str) -> pd.DataFrame:
    return fetch_next_games(season)


@st.cache_data(ttl=3600)
def load_goalie_stats(season: str) -> pd.DataFrame:
    return fetch_goalie_stats(season)


@st.cache_data(ttl=3600)
def load_goalie_log(player_id: int, season: str) -> pd.DataFrame:
    return fetch_goalie_game_log(player_id, season)


@st.cache_data(ttl=3600)
def load_all_goalie_logs(season: str) -> pd.DataFrame:
    """Load combined goalie game logs, reading the processed cache if available."""
    processed_path = Path("data/processed/goalie_game_logs") / f"{season}_2.parquet"
    if processed_path.exists():
        return pd.read_parquet(processed_path)
    return build_goalie_game_logs(season)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _fmt(val: float, decimals: int = 3) -> str:
    """Format a float as a fixed-decimal string, or 'N/A' for NaN."""
    if math.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def stat_block(
    stat_dict: dict,
    label_map: dict | None = None,
    decimals: int = 3,
    pct: bool = False,
) -> None:
    """Render metric cards for a stat's season and last-N windows.

    stat_dict shape: {"season": {median, p25, p75, n}, "last_n": {...}}
    label_map maps dict keys to display labels (ordered).
    pct=True multiplies values by 100 and appends a "%" suffix.
    """
    if label_map is None:
        label_map = {"last_n": "Last 5 Games", "season": "Full Season"}

    multiplier = 100 if pct else 1
    suffix = "%" if pct else ""
    display_decimals = 1 if pct else decimals

    cols = st.columns(len(label_map))
    for col, (key, window_label) in zip(cols, label_map.items()):
        d = stat_dict.get(key, {})
        median = d.get("median", float("nan"))
        n = d.get("n", 0)
        col.metric(
            window_label,
            f"{_fmt(median * multiplier, display_decimals)}{suffix}",
            f"n={n}",
        )

    pct_cols = st.columns(len(label_map))
    for col, (key, _) in zip(pct_cols, label_map.items()):
        d = stat_dict.get(key, {})
        p25 = _fmt(d.get("p25", float("nan")) * multiplier, display_decimals)
        p75 = _fmt(d.get("p75", float("nan")) * multiplier, display_decimals)
        col.caption(f"25th: {p25}{suffix} · 75th: {p75}{suffix}")


def box_plot(
    traces: list[tuple[str, pd.Series]],
    y_label: str,
    height: int = 260,
    pct: bool = False,
) -> None:
    """Compact Plotly box plot. Each trace is (name, Series)."""
    multiplier = 100 if pct else 1
    fig = go.Figure()
    for name, series in traces:
        clean = series.dropna() * multiplier
        if not clean.empty:
            fig.add_trace(go.Box(y=clean.tolist(), name=name, boxpoints="all"))
    fig.update_layout(
        yaxis_title=y_label,
        height=height,
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _game_label(row: pd.Series) -> str:
    """Format a game row as 'AWAY @ HOME — Day Mon D'."""
    try:
        d = pd.to_datetime(row["gameDate"])
        date_str = d.strftime("%a %b %-d")
    except Exception:
        date_str = str(row["gameDate"])
    return f"{row['awayTeam']} @ {row['homeTeam']} — {date_str}"


def _goalie_display_name(row: pd.Series) -> str:
    """Return a human-readable name from a goalie stats row."""
    for col in ("goalieFullName", "skaterFullName", "playerName"):
        if col in row.index and pd.notna(row.get(col)):
            return str(row[col])
    first = row.get("firstName", "")
    last = row.get("lastName", "")
    if first or last:
        return f"{first} {last}".strip()
    return f"Player {row.get('playerId', '?')}"


def _team_goalies(gs: pd.DataFrame, team_abbrev: str) -> tuple[list[str], list[int]]:
    """Return (display_names, player_ids) for a team, sorted by starts desc."""
    team = gs[gs["teamAbbrevs"].str.contains(team_abbrev, na=False)].sort_values(
        "gamesStarted", ascending=False
    )
    if team.empty:
        return [f"No data ({team_abbrev})"], [-1]
    names = [_goalie_display_name(row) for _, row in team.iterrows()]
    ids = team["playerId"].astype(int).tolist()
    return names, ids


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("NHL Goalie Save Tracker")

    season = current_season()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.caption(f"Season: **{season[:4]}–{season[4:]}**")
        st.markdown("---")
        st.subheader("Upcoming Games")

        with st.spinner("Loading schedule…"):
            games_df = load_upcoming_games(season)

        if games_df.empty:
            st.warning("No upcoming games found.")
            st.stop()

        game_labels = games_df.apply(_game_label, axis=1).tolist()
        selected_label = st.selectbox("Select game", game_labels)
        sel_idx = game_labels.index(selected_label)
        selected_game = games_df.iloc[sel_idx]

        home_team: str = selected_game["homeTeam"]
        away_team: str = selected_game["awayTeam"]
        venue: str = selected_game.get("venue") or ""
        game_date: str = selected_game["gameDate"]

        st.markdown("---")
        st.subheader("Goalie Selection")

        with st.spinner("Loading goalie stats…"):
            gs = load_goalie_stats(season)

        home_names, home_ids = _team_goalies(gs, home_team)
        away_names, away_ids = _team_goalies(gs, away_team)

        home_sel = st.selectbox(f"Home goalie ({home_team})", home_names)
        away_sel = st.selectbox(f"Away goalie ({away_team})", away_names)

        home_player_id = (
            home_ids[home_names.index(home_sel)] if home_ids[0] != -1 else None
        )
        away_player_id = (
            away_ids[away_names.index(away_sel)] if away_ids[0] != -1 else None
        )

        st.markdown("---")
        analyse_side = st.radio("Analyse which goalie?", ["Home", "Away"])

    # ── Resolve selected goalie ───────────────────────────────────────────────
    if analyse_side == "Home":
        player_id = home_player_id
        goalie_name = home_sel
        goalie_team = home_team
        opponent_team = away_team
    else:
        player_id = away_player_id
        goalie_name = away_sel
        goalie_team = away_team
        opponent_team = home_team

    if not player_id or player_id == -1:
        st.error(f"No goalie data available for {goalie_team}.")
        st.stop()

    # ── Main header ───────────────────────────────────────────────────────────
    st.header(f"{goalie_name} vs {opponent_team}")
    try:
        date_str = pd.to_datetime(game_date).strftime("%A, %B %-d, %Y")
    except Exception:
        date_str = game_date
    venue_part = f" · {venue}" if venue else ""
    st.caption(f"{date_str}{venue_part}")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading goalie game log…"):
        goalie_log = load_goalie_log(player_id, season)

    with st.spinner("Loading league-wide game logs (first run may take a minute)…"):
        all_logs = load_all_goalie_logs(season)

    if goalie_log.empty:
        st.warning("No game log data available for this goalie yet this season.")
        st.stop()

    # Compute all five stats
    report = goalie_report(goalie_log, all_logs, opponent_team)

    # ── Raw series for box plots ──────────────────────────────────────────────
    if "gamesStarted" in goalie_log.columns:
        started = goalie_log[goalie_log["gamesStarted"] == 1].sort_values("gameDate")
    else:
        started = goalie_log.sort_values("gameDate")

    _team_df = (
        all_logs[all_logs["teamAbbrev"] == goalie_team]
        .groupby("gameId")
        .agg(sog=("shotsAgainst", "sum"), gameDate=("gameDate", "max"))
        .sort_values("gameDate")
    )
    _opp_df = (
        all_logs[all_logs["opponentAbbrev"] == opponent_team]
        .groupby("gameId")
        .agg(
            sog=("shotsAgainst", "sum"),
            goals=("goalsAgainst", "sum"),
            gameDate=("gameDate", "max"),
        )
        .sort_values("gameDate")
    )
    _opp_with_shots = _opp_df[_opp_df["sog"] > 0].copy()
    _opp_with_shots["goal_rate"] = _opp_with_shots["goals"] / _opp_with_shots["sog"]

    # ── Quick Summary ─────────────────────────────────────────────────────────
    st.markdown("#### Quick Summary")

    def _summary_group(col, label, stat_dict, decimals=3, pct=False):
        mult = 100 if pct else 1
        dp = 1 if pct else decimals
        sfx = "%" if pct else ""
        col.markdown(f"**{label}**")
        c1, c2 = col.columns(2)
        c1.metric("Last 5", f"{_fmt(stat_dict['last_n']['median'] * mult, dp)}{sfx}")
        season_val = _fmt(stat_dict["season"]["median"] * mult, dp)
        c2.metric("Full Season", f"{season_val}{sfx}")

    sc = st.columns(5)
    _summary_group(sc[0], "Save %", report["save_pct"], pct=True)
    _summary_group(sc[1], "SOG Allowed", report["sog_allowed"]["goalie"], decimals=1)
    _summary_group(
        sc[2], f"Opp SOG ({opponent_team})", report["opponent_sog"], decimals=1
    )
    _summary_group(sc[3], "Opp Goal Conv%", report["opponent_goal_rate"], pct=True)

    vs_df = report["vs_opponent"]
    with sc[4]:
        st.markdown(f"**vs {opponent_team}**")
        if vs_df.empty:
            st.metric("Record (W–L–OT)", "—")
        else:
            dec = vs_df.get("decision", pd.Series(dtype=str))
            w = (dec == "W").sum()
            losses = (dec == "L").sum()
            ot = (~dec.isin(["W", "L"])).sum()
            st.metric("Record (W–L–OT)", f"{w}–{losses}–{ot}")

    # ── Stat 1: SOG Allowed ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Shots on Goal Allowed")
    st.caption("Shots on goal conceded per game")

    st.write(f"**Team ({goalie_team}) — all goalies**")
    c_met, c_box = st.columns([2, 3])
    with c_met:
        stat_block(report["sog_allowed"]["team"], decimals=1)
    with c_box:
        box_plot(
            [("Full Season", _team_df["sog"]), ("Last 5", _team_df.tail(5)["sog"])],
            "SOG/gm",
        )

    st.write(f"**{goalie_name} starts only**")
    c_met, c_box = st.columns([2, 3])
    with c_met:
        stat_block(report["sog_allowed"]["goalie"], decimals=1)
    with c_box:
        box_plot(
            [
                ("Full Season", started["shotsAgainst"]),
                ("Last 5", started.tail(5)["shotsAgainst"]),
            ],
            "SOG/gm",
        )

    # ── Stat 2: Opponent SOG ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Opponent Shots on Goal")
    st.caption(f"Shots {opponent_team} generates per game (offensive output)")
    c_met, c_box = st.columns([2, 3])
    with c_met:
        stat_block(report["opponent_sog"], decimals=1)
    with c_box:
        box_plot(
            [("Full Season", _opp_df["sog"]), ("Last 5", _opp_df.tail(5)["sog"])],
            "SOG/gm",
        )

    # ── Stat 3: Save Percentage ───────────────────────────────────────────────
    st.divider()
    st.subheader("Save Percentage")
    st.caption(f"{goalie_name}'s save % distribution")
    c_met, c_box = st.columns([2, 3])
    with c_met:
        stat_block(report["save_pct"], pct=True)
    with c_box:
        box_plot(
            [
                ("Full Season", started["savePctg"]),
                ("Last 5", started.tail(5)["savePctg"]),
            ],
            "Save %",
            pct=True,
        )

    # ── Stat 4: Opponent Goal Conversion % ───────────────────────────────────
    st.divider()
    st.subheader("Opponent Goal Conversion %")
    st.caption(
        f"% of {opponent_team}'s shots on goal that become goals (goals / SOG per game)"
    )
    c_met, c_box = st.columns([2, 3])
    with c_met:
        stat_block(report["opponent_goal_rate"], pct=True)
    with c_box:
        box_plot(
            [
                ("Full Season", _opp_with_shots["goal_rate"]),
                ("Last 5", _opp_with_shots.tail(5)["goal_rate"]),
            ],
            "Goal Conv%",
            pct=True,
        )

    # ── Stat 5: History vs Opponent ───────────────────────────────────────────
    st.divider()
    st.subheader(f"vs {opponent_team} This Season")
    if vs_df.empty:
        st.info(f"No previous meetings vs {opponent_team} this season.")
    else:
        display_df = vs_df.copy()
        if "savePctg" in display_df.columns:
            display_df["savePctg"] = display_df["savePctg"].map(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )
        st.dataframe(display_df, use_container_width=True, hide_index=True)


main()
