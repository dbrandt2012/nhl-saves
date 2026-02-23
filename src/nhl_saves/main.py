"""NHL Saves Dashboard — Streamlit UI."""

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nhl_saves.stats import (
    days_since_last_game,
    goalie_report,
    opponent_goal_rate_stats,
    opponent_sog_stats,
    team_save_pct_stats,
    team_sog_stats,
    vs_opponent_sog_season,
)
from nhl_saves.store import (
    NHL_TEAMS,
    build_goalie_game_logs,
    fetch_goalie_game_log,
    fetch_schedule,
    fetch_team_goalie_ids,
)

st.set_page_config(page_title="NHL Saves Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------


def current_season() -> str:
    today = date.today()
    year = today.year if today.month >= 9 else today.year - 1
    return f"{year}{year + 1}"


SEASON = current_season()


# ---------------------------------------------------------------------------
# Cached data loaders
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
def load_goalie_log(player_id: int, season: str) -> pd.DataFrame:
    return fetch_goalie_game_log(player_id, season)


@st.cache_data(ttl=3600)
def load_team_goalie_ids(team_abbrev: str, season: str) -> list[dict]:
    return fetch_team_goalie_ids(team_abbrev, season)


# ---------------------------------------------------------------------------
# Team ordering
# ---------------------------------------------------------------------------


def build_team_order(games_df: pd.DataFrame) -> list[dict]:
    """Return ordered list of team dicts from today's schedule.

    Order: LIVE games first (by gameDate asc), then FUT/PRE (by gameDate asc).
    For each game: home team row first, then away team row.
    """
    if games_df.empty:
        return []

    live_states = {"LIVE", "CRIT"}
    live = games_df[games_df["gameState"].isin(live_states)].sort_values("gameDate")
    other = games_df[~games_df["gameState"].isin(live_states)].sort_values(
        "gameDate"
    )
    ordered_games = pd.concat([live, other], ignore_index=True)

    result = []
    for _, row in ordered_games.iterrows():
        for role in ("homeTeam", "awayTeam"):
            team = row[role]
            opponent = row["awayTeam"] if role == "homeTeam" else row["homeTeam"]
            result.append(
                {
                    "team": team,
                    "opponent": opponent,
                    "game_state": row.get("gameState", ""),
                    "game_date": row.get("gameDate", ""),
                }
            )
    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def fmt_dist(s: dict, decimals: int = 1) -> str:
    """Percentile summary: Median [P25, P75]"""
    if pd.isna(s.get("median", float("nan"))):
        return "—"
    d = decimals
    return f"{s['median']:.{d}f} [{s['p25']:.{d}f}, {s['p75']:.{d}f}]"


def fmt_pct_dist(s: dict) -> str:
    """Percentile summary for percentages: Median% [P25%, P75%]"""
    if pd.isna(s.get("median", float("nan"))):
        return "—"
    return f"{s['median']:.1f}% [{s['p25']:.1f}%, {s['p75']:.1f}%]"


def fmt_range(s: dict, pct: bool = False) -> str:
    """Range summary (last-5): Median [Min, Max]"""
    if pd.isna(s.get("median", float("nan"))):
        return "—"
    if pct:
        return f"{s['median']:.1f}% [{s['min']:.1f}%, {s['max']:.1f}%]"
    return f"{s['median']:.1f} [{s['min']:.1f}, {s['max']:.1f}]"


def fmt_mean_range(s: dict, pct: bool = False) -> str:
    """Mean-range summary (vs-opp): Mean (Min, Max)"""
    if pd.isna(s.get("mean", float("nan"))):
        return "—"
    if pct:
        return f"{s['mean']:.1f}% ({s['min']:.1f}%, {s['max']:.1f}%)"
    return f"{s['mean']:.1f} ({s['min']:.1f}, {s['max']:.1f})"


# ---------------------------------------------------------------------------
# Build team stats table
# ---------------------------------------------------------------------------

_STAT_COLS = [
    "SOG Season",
    "SOG L5",
    "Save% Season",
    "Save% L5",
    "Opp SOG Season",
    "Opp SOG L5",
    "Opp Goal Conv% Season",
    "Opp Goal Conv% L5",
    "vs Opp SOG",
    "Days Since",
]


def _empty_stat_row() -> dict:
    return {c: "—" for c in _STAT_COLS}


def build_team_table_df(
    team_order: list[dict],
    all_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Build the master team stats DataFrame — one row per team, all stats."""
    rows = []
    for ti in team_order:
        team = ti["team"]
        opponent = ti["opponent"]
        game_state = ti["game_state"]
        game_date = ti["game_date"]

        try:
            date_str = pd.to_datetime(game_date).strftime("%a %b %-d")
        except Exception:
            date_str = str(game_date)

        sog = team_sog_stats(all_logs, team)
        svp = team_save_pct_stats(all_logs, team)
        opp_sog = opponent_sog_stats(all_logs, opponent)
        opp_gr = opponent_goal_rate_stats(all_logs, opponent)
        vs_sog = vs_opponent_sog_season(all_logs, team, opponent)

        team_games = all_logs[all_logs["teamAbbrev"] == team]
        days = days_since_last_game(team_games) if not team_games.empty else None

        rows.append(
            {
                "Team": team,
                "Opponent": opponent,
                "Date": date_str,
                "State": game_state,
                "SOG Season": fmt_dist(sog["season"]),
                "SOG L5": fmt_range(sog["last_n"]),
                "Save% Season": fmt_pct_dist(svp["season"]),
                "Save% L5": fmt_range(svp["last_n"], pct=True),
                "Opp SOG Season": fmt_dist(opp_sog["season"]),
                "Opp SOG L5": fmt_range(opp_sog["last_n"]),
                "Opp Goal Conv% Season": fmt_pct_dist(opp_gr["season"]),
                "Opp Goal Conv% L5": fmt_range(opp_gr["last_n"], pct=True),
                "vs Opp SOG": fmt_mean_range(vs_sog),
                "Days Since": str(days) if days is not None else "—",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build goalie rows DataFrame
# ---------------------------------------------------------------------------


def build_goalie_rows(
    goalie_ids: list[dict],
    all_logs: pd.DataFrame,
    opponent: str,
    season: str,
) -> pd.DataFrame:
    """Build a display DataFrame with one row per goalie."""
    rows = []
    for ginfo in goalie_ids:
        pid = ginfo["player_id"]
        name = ginfo["name"]
        log = load_goalie_log(pid, season)

        if log.empty:
            rows.append({"Goalie": name, **_empty_stat_row()})
            continue

        report = goalie_report(log, all_logs, opponent)

        rows.append(
            {
                "Goalie": name,
                "SOG Season": fmt_dist(
                    report["sog_allowed"]["goalie"]["season"]
                ),
                "SOG L5": fmt_range(report["sog_allowed"]["goalie"]["last_n"]),
                "Save% Season": fmt_pct_dist(report["save_pct"]["season"]),
                "Save% L5": fmt_range(
                    report["save_pct"]["last_n"], pct=True
                ),
                "Opp SOG Season": fmt_dist(report["opponent_sog"]["season"]),
                "Opp SOG L5": fmt_range(report["opponent_sog"]["last_n"]),
                "Opp Goal Conv% Season": fmt_pct_dist(
                    report["opponent_goal_rate"]["season"]
                ),
                "Opp Goal Conv% L5": fmt_range(
                    report["opponent_goal_rate"]["last_n"], pct=True
                ),
                "vs Opp SOG": fmt_mean_range(report["vs_opponent_sog"]),
                "Days Since": (
                    str(report["days_since"])
                    if report["days_since"] is not None
                    else "—"
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Goalie box plot panel
# ---------------------------------------------------------------------------


def _box_plot(
    season_vals: list,
    last5_vals: list,
    title: str,
    pct: bool = False,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=season_vals,
            name="Season",
            boxpoints=False,
            marker_color="#4C78A8",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=last5_vals,
            x=["L5"] * len(last5_vals),
            mode="markers",
            name="Last 5",
            marker=dict(color="orange", size=10, symbol="circle"),
        )
    )
    fig.update_layout(
        title=title,
        showlegend=True,
        height=280,
        margin={"t": 40, "b": 20, "l": 10, "r": 10},
        yaxis_ticksuffix="%" if pct else "",
    )
    st.plotly_chart(fig, width="stretch")


def render_goalie_detail(
    player_id: int,
    name: str,
    opponent: str,
    season: str,
    all_logs: pd.DataFrame,
) -> None:
    """Render box plot panel for a selected goalie."""
    st.subheader(f"{name} vs {opponent}")
    if st.button("✕ Close"):
        st.session_state.selected_goalie = None
        st.rerun()

    log = load_goalie_log(player_id, season)
    if log.empty:
        st.info("No game log data available for this goalie.")
        return

    started = log[log["gamesStarted"] == 1].sort_values("gameDate")
    if started.empty:
        st.info("No started games found.")
        return

    last5 = started.tail(5)

    for label, col, is_pct in [
        ("SOG Allowed", "shotsAgainst", False),
        ("Save %", "savePctg", True),
    ]:
        if col not in started.columns:
            continue
        mult = 100 if is_pct else 1
        _box_plot(
            season_vals=(started[col].dropna() * mult).tolist(),
            last5_vals=(last5[col].dropna() * mult).tolist(),
            title=label,
            pct=is_pct,
        )

    # Opponent SOG and goal conversion (league-wide)
    opp_games = all_logs[all_logs["opponentAbbrev"] == opponent].copy()
    if not opp_games.empty:
        per_game = (
            opp_games.groupby("gameId")
            .agg(
                sog=("shotsAgainst", "sum"),
                goals=("goalsAgainst", "sum"),
                gameDate=("gameDate", "max"),
            )
            .sort_values("gameDate")
        )
        pg_last5 = per_game.tail(5)

        _box_plot(
            season_vals=per_game["sog"].dropna().tolist(),
            last5_vals=pg_last5["sog"].dropna().tolist(),
            title="Opp SOG",
        )

        ws = per_game[per_game["sog"] > 0].copy()
        ws["goal_rate"] = ws["goals"] / ws["sog"] * 100
        ws5 = pg_last5[pg_last5["sog"] > 0].copy()
        ws5["goal_rate"] = ws5["goals"] / ws5["sog"] * 100

        _box_plot(
            season_vals=ws["goal_rate"].dropna().tolist(),
            last5_vals=ws5["goal_rate"].dropna().tolist(),
            title="Opp Goal Conv%",
            pct=True,
        )


# ---------------------------------------------------------------------------
# Stats guide
# ---------------------------------------------------------------------------


def render_stats_guide() -> None:
    with st.expander("Stats Guide & How to Use"):
        st.markdown("### Statistics Explained")
        st.markdown(
            "| Stat | Definition | Format |\n"
            "|------|-----------|--------|\n"
            "| SOG Allowed | SOG the team/goalie concedes per game"
            " | Season: Median [P25,P75]; L5: Median [Min,Max] |\n"
            "| Save % | Saves ÷ SOG × 100 "
            "| Season: Median [P25,P75]; L5: Median [Min,Max] |\n"
            "| Opp SOG | SOG the opponent generates per game"
            " | Season: Median [P25,P75]; L5: Median [Min,Max] |\n"
            "| Opp Goal Conv % | Opponent goals ÷ SOG × 100"
            " | Season: Median [P25,P75]; L5: Median [Min,Max] |\n"
            "| vs Opp SOG | SOG allowed vs this opponent this season"
            " | Mean (Min, Max) |\n"
            "| Days Since | Days since last game | Single integer |"
        )
        st.markdown(
            "> **Note:** Season stats use **P25/P75** (quartile range);"
            " Last 5 and vs-opponent stats use **Min/Max** (full range)."
        )
        st.markdown("---\n### How to Use")
        st.markdown(
            "- All teams with games today are shown in the main table"
            " — no clicking required to see team stats.\n"
            "- Teams are ordered by game time (LIVE first, then upcoming)."
            " Home team appears before away team for each game.\n"
            "- **Click a team row** to see individual goalie stats"
            " for that team below the main table.\n"
            "- **Click a goalie row** to open box plots showing the"
            " full-season distribution with last-5 points in orange.\n"
            "- Use **✕ Close** to dismiss the box plot panel."
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("NHL Saves Dashboard")

    # Initialise session state
    if "selected_team_idx" not in st.session_state:
        st.session_state.selected_team_idx = None
    if "selected_goalie" not in st.session_state:
        st.session_state.selected_goalie = None

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading today's schedule…"):
        games_df = load_schedule()

    if games_df.empty:
        st.warning("No games found for today.")
        return

    # Filter out non-NHL teams (CAN, USA, etc. appear in international games
    # that can show up in the gameWeek response)
    nhl_set = set(NHL_TEAMS)
    games_df = games_df[
        games_df["homeTeam"].isin(nhl_set) & games_df["awayTeam"].isin(nhl_set)
    ].reset_index(drop=True)

    team_order = build_team_order(games_df)
    if not team_order:
        st.warning("No games to display.")
        return

    with st.spinner("Loading goalie game logs (first run may take a minute)…"):
        all_logs = load_all_goalie_logs(SEASON)

    with st.spinner("Computing team stats…"):
        team_df = build_team_table_df(team_order, all_logs)

    # ── Master team table (always visible) ────────────────────────────────────
    n_games = len(games_df)
    plural = "s" if n_games != 1 else ""
    st.caption(f"Season {SEASON[:4]}–{SEASON[4:]} · {n_games} upcoming game{plural}")

    team_sel = st.dataframe(
        team_df,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        width="stretch",
        key="team_table",
    )

    # Track team selection; clear goalie if team changes
    new_team_idx = (
        team_sel.selection.rows[0] if team_sel.selection.rows else None
    )
    if new_team_idx != st.session_state.selected_team_idx:
        st.session_state.selected_team_idx = new_team_idx
        st.session_state.selected_goalie = None

    # ── Goalie sub-table (appears when a team row is selected) ────────────────
    team_idx = st.session_state.selected_team_idx
    if team_idx is not None and team_idx < len(team_order):
        ti = team_order[team_idx]
        team = ti["team"]
        opponent = ti["opponent"]

        st.divider()
        st.markdown(f"#### {team} goalies — vs {opponent}")

        goalie_ids = load_team_goalie_ids(team, SEASON)
        if not goalie_ids:
            st.caption("No goalie data available for this team.")
        else:
            with st.spinner(f"Loading {team} goalie stats…"):
                goalie_df = build_goalie_rows(
                    goalie_ids, all_logs, opponent, SEASON
                )

            # Layout: goalie table left, box plots right (if goalie selected)
            if st.session_state.selected_goalie:
                col_g, col_p = st.columns([3, 2])
            else:
                col_g = st.container()
                col_p = None

            with col_g:
                goalie_sel = st.dataframe(
                    goalie_df,
                    on_select="rerun",
                    selection_mode="single-row",
                    hide_index=True,
                    width="stretch",
                    key=f"goalie_table_{team}",
                )

                if goalie_sel.selection.rows:
                    g_idx = goalie_sel.selection.rows[0]
                    if g_idx < len(goalie_ids):
                        new_goalie = {
                            **goalie_ids[g_idx],
                            "team": team,
                            "opponent": opponent,
                        }
                        if new_goalie != st.session_state.selected_goalie:
                            st.session_state.selected_goalie = new_goalie
                            st.rerun()

            if col_p is not None and st.session_state.selected_goalie:
                with col_p:
                    sel_g = st.session_state.selected_goalie
                    render_goalie_detail(
                        player_id=sel_g["player_id"],
                        name=sel_g["name"],
                        opponent=sel_g["opponent"],
                        season=SEASON,
                        all_logs=all_logs,
                    )

    # ── Stats guide ───────────────────────────────────────────────────────────
    render_stats_guide()


main()
