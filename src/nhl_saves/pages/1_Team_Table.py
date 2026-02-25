"""NHL Saves Dashboard — Page 1: Team Overview."""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from nhl_saves.store import (
    NHL_TEAMS,
    build_goalie_game_logs,
    fetch_schedule,
    fetch_team_goalie_ids,
)
from nhl_saves.ui import (
    build_goalie_rows,
    build_team_table_df,
    render_stats_guide_team,
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
def load_team_goalie_ids(team_abbrev: str, season: str) -> list[dict]:
    return fetch_team_goalie_ids(team_abbrev, season)


# ---------------------------------------------------------------------------
# Alpha team ordering
# ---------------------------------------------------------------------------


def build_alpha_team_rows(games_df: pd.DataFrame) -> list[dict]:
    """One row per team, alphabetical. Opponent = team's next scheduled game."""
    team_next: dict[str, dict] = {}
    for _, row in games_df.sort_values("gameDate").iterrows():
        for team_col, opp_col in [("homeTeam", "awayTeam"), ("awayTeam", "homeTeam")]:
            t = row[team_col]
            if t not in team_next:
                team_next[t] = {
                    "team": t,
                    "opponent": row[opp_col],
                    "next_game_date": row["gameDate"],
                }
    return sorted(team_next.values(), key=lambda x: x["team"])


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("NHL Saves Dashboard")

    # Initialise session state
    if "selected_team_idx" not in st.session_state:
        st.session_state.selected_team_idx = None

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading today's schedule…"):
        games_df = load_schedule()

    if games_df.empty:
        st.warning("No games found for today.")
        return

    # Filter out non-NHL teams
    nhl_set = set(NHL_TEAMS)
    games_df = games_df[
        games_df["homeTeam"].isin(nhl_set) & games_df["awayTeam"].isin(nhl_set)
    ].reset_index(drop=True)

    team_rows = build_alpha_team_rows(games_df)
    if not team_rows:
        st.warning("No games to display.")
        return

    with st.spinner("Loading goalie game logs (first run may take a minute)…"):
        all_logs = load_all_goalie_logs(SEASON)

    with st.spinner("Computing team stats…"):
        team_df = build_team_table_df(team_rows, all_logs, simplified=True)

    # ── Stats guide ───────────────────────────────────────────────────────────
    render_stats_guide_team()

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

    # Track team selection
    new_team_idx = team_sel.selection.rows[0] if team_sel.selection.rows else None
    if new_team_idx != st.session_state.selected_team_idx:
        st.session_state.selected_team_idx = new_team_idx

    # ── Goalie sub-table (appears when a team row is selected) ────────────────
    team_idx = st.session_state.selected_team_idx
    if team_idx is not None and team_idx < len(team_rows):
        ti = team_rows[team_idx]
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
                    goalie_ids, all_logs, opponent, SEASON, simplified=True
                )

            st.dataframe(
                goalie_df,
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True,
                width="stretch",
                key=f"goalie_table_{team}",
            )


main()
