"""Shared UI components for the NHL Saves Dashboard."""

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
from nhl_saves.store import fetch_goalie_game_log

# ---------------------------------------------------------------------------
# Cached loader (shared between pages)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def load_goalie_log(player_id: int, season: str) -> pd.DataFrame:
    return fetch_goalie_game_log(player_id, season)


# ---------------------------------------------------------------------------
# Number helper
# ---------------------------------------------------------------------------


def _n(val: float, decimals: int = 1) -> str:
    """Format a number, stripping trailing '.0' for whole-number results."""
    s = f"{val:.{decimals}f}"
    if s.endswith(".0"):
        s = s[:-2]
    return s


# ---------------------------------------------------------------------------
# Bracket formatters (Page 2 full-detail rows)
# ---------------------------------------------------------------------------


def fmt_dist(s: dict, decimals: int = 1) -> str:
    """Percentile summary: Median [P25, P75]"""
    if pd.isna(s.get("median", float("nan"))):
        return "—"
    d = decimals
    return f"{_n(s['median'], d)} [{_n(s['p25'], d)}, {_n(s['p75'], d)}]"


def fmt_pct_dist(s: dict) -> str:
    """Percentile summary for percentages: Median% [P25%, P75%]"""
    if pd.isna(s.get("median", float("nan"))):
        return "—"
    return f"{_n(s['median'], 1)}% [{_n(s['p25'], 1)}%, {_n(s['p75'], 1)}%]"


def fmt_range(s: dict, pct: bool = False) -> str:
    """Range summary (last-5): Mean [Min, Max]"""
    if pd.isna(s.get("mean", float("nan"))):
        return "—"
    if pct:
        return f"{_n(s['mean'], 1)}% [{_n(s['min'], 1)}%, {_n(s['max'], 1)}%]"
    return f"{_n(s['mean'], 1)} [{_n(s['min'], 1)}, {_n(s['max'], 1)}]"


def fmt_mean_range(s: dict, pct: bool = False) -> str:
    """Mean-range summary (vs-opp): Mean (Min, Max)"""
    if pd.isna(s.get("mean", float("nan"))):
        return "—"
    if pct:
        return f"{_n(s['mean'], 1)}% ({_n(s['min'], 1)}%, {_n(s['max'], 1)}%)"
    return f"{_n(s['mean'], 1)} ({_n(s['min'], 1)}, {_n(s['max'], 1)})"


# ---------------------------------------------------------------------------
# Single-value formatters (Page 1 simplified rows)
# ---------------------------------------------------------------------------


def fmt_val(s: dict, key: str = "median", decimals: int = 1) -> str:
    """Return just s[key] as a number string, or '—'."""
    v = s.get(key, float("nan"))
    if pd.isna(v):
        return "—"
    return _n(v, decimals)


def fmt_pct_val(s: dict, key: str = "median") -> str:
    """Return just s[key] as 'X.X%', or '—'."""
    v = s.get(key, float("nan"))
    if pd.isna(v):
        return "—"
    return f"{_n(v, 1)}%"


def fmt_sv_pct_val(s: dict, key: str = "median") -> str:
    """Display save% as 0.912 — divides 0-100 stored value by 100."""
    v = s.get(key, float("nan"))
    if pd.isna(v):
        return "—"
    return f"{v / 100:.3f}"


def fmt_sv_pct_combined(season_dict: dict, lastn_dict: dict) -> str:
    """Combined Season (L5) display for save%: '0.912 (0.905)'."""
    s = fmt_sv_pct_val(season_dict)
    ln = fmt_sv_pct_val(lastn_dict, key="mean")
    return "—" if s == "—" else f"{s} ({ln})"


def fmt_combined(
    season_dict: dict, lastn_dict: dict, decimals: int = 1, pct: bool = False
) -> str:
    """Format as 'Season (L5)' for table cells."""
    if pct:
        s = fmt_pct_val(season_dict)
        ln = fmt_pct_val(lastn_dict, key="mean")
    else:
        s = fmt_val(season_dict, decimals=decimals)
        ln = fmt_val(lastn_dict, key="mean", decimals=decimals)
    return "—" if s == "—" else f"{s} ({ln})"


# ---------------------------------------------------------------------------
# stat_block
# ---------------------------------------------------------------------------


def stat_block(
    stat_dict: dict,
    label: str,
    decimals: int = 1,
    pct: bool = False,
    sv_pct: bool = False,
) -> None:
    """Render metric cards + caption row for a stat dict.

    stat_dict shape: {"season": {median, p25, p75, n}, "last_n": {median, min, max, n}}
    pct: values are already in 0-100 range (no multiplication needed).
    sv_pct: values are 0-100 range, display as 3-decimal proportion (0.912).
    """
    sfx = "%" if pct else ""
    if sv_pct:
        sfx = ""
    label_map = {"last_n": "Last 5 Games", "season": "Full Season"}
    n_map: dict[str, int] = {}
    cols = st.columns(2)
    for col, (key, window_label) in zip(cols, label_map.items()):
        d = stat_dict.get(key, {})
        median = d.get("median", d.get("mean", float("nan")))
        n = d.get("n", 0)
        n_map[key] = n
        if sv_pct:
            val_str = f"{median / 100:.3f}" if not pd.isna(median) else "N/A"
        else:
            val_str = f"{_n(median, decimals)}{sfx}" if not pd.isna(median) else "N/A"
        col.metric(window_label, val_str)
    # Caption row: n= + p25/p75 for season, n= + min/max for last_n
    cap_cols = st.columns(2)
    for cap_col, (key, _) in zip(cap_cols, label_map.items()):
        d = stat_dict.get(key, {})
        n = n_map[key]
        if "p25" in d:
            lo = d.get("p25", float("nan"))
            hi = d.get("p75", float("nan"))
            lo_lbl, hi_lbl = "25th", "75th"
        else:
            lo = d.get("min", float("nan"))
            hi = d.get("max", float("nan"))
            lo_lbl, hi_lbl = "Min", "Max"
        if sv_pct:
            lo_s = f"{lo / 100:.3f}" if not pd.isna(lo) else "N/A"
            hi_s = f"{hi / 100:.3f}" if not pd.isna(hi) else "N/A"
        else:
            lo_s = f"{_n(lo, decimals)}{sfx}" if not pd.isna(lo) else "N/A"
            hi_s = f"{_n(hi, decimals)}{sfx}" if not pd.isna(hi) else "N/A"
        cap_col.caption(f"n={n} · {lo_lbl}: {lo_s} · {hi_lbl}: {hi_s}")


# ---------------------------------------------------------------------------
# box_plot
# ---------------------------------------------------------------------------


def box_plot(
    traces: list[tuple],
    y_label: str,
    last_n: int = 5,
    height: int = 260,
    pct: bool = False,
    y_fmt: str = ".1f",
) -> None:
    """Box plots with individual data points overlaid.

    Each trace: (name, Series) or (name, Series, hover_texts).
    pct: multiply values by 100 before plotting.
    last_n: highlight the last N points in orange; earlier points in grey.
    y_fmt: format string for hover template precision (e.g. '.1f', '.3f').
    """
    pct_sfx = "%" if pct else ""
    fig = go.Figure()
    for i, trace in enumerate(traces):
        name = trace[0]
        series = trace[1]
        raw_hover = trace[2] if len(trace) > 2 else None

        mask = series.notna()
        vals = series.dropna()
        if pct:
            vals = vals * 100

        # Align hover texts to non-NaN values
        if raw_hover is not None:
            if len(raw_hover) == len(series):
                aligned_hover = [t for t, m in zip(raw_hover, mask) if m]
            elif len(raw_hover) == len(vals):
                aligned_hover = list(raw_hover)
            else:
                aligned_hover = None
        else:
            aligned_hover = None

        split = max(0, len(vals) - last_n)
        last_n_vals = vals.iloc[split:]
        last_n_hover = aligned_hover[split:] if aligned_hover is not None else None

        # Hover templates
        s_tmpl = (
            f"%{{y:{y_fmt}}}{pct_sfx}<br>%{{text}}<extra>Season</extra>"
            if aligned_hover is not None
            else f"%{{y:{y_fmt}}}{pct_sfx}<extra>Season</extra>"
        )
        l_tmpl = (
            f"%{{y:{y_fmt}}}{pct_sfx}<br>%{{text}}<extra>L5G</extra>"
            if last_n_hover is not None
            else f"%{{y:{y_fmt}}}{pct_sfx}<extra>L5G</extra>"
        )

        # Box with all points in grey — go.Box.marker.color only accepts a
        # scalar, so we colour all points grey here and overlay orange separately.
        box_kw: dict = dict(
            y=vals.tolist(),
            name=name,
            boxpoints="all",
            marker=dict(color="#888888", size=6),
            showlegend=False,
            hovertemplate=s_tmpl,
        )
        if aligned_hover is not None:
            box_kw["text"] = aligned_hover
        fig.add_trace(go.Box(**box_kw))

        # Orange scatter overlay for L5G — x=[name] places points at the same
        # categorical x-position as the box trace above.
        if len(last_n_vals) > 0:
            scatter_kw: dict = dict(
                x=[name] * len(last_n_vals),
                y=last_n_vals.tolist(),
                mode="markers",
                name="Last 5",
                marker=dict(color="#F58518", size=9, opacity=0.9),
                showlegend=(i == 0),
                hovertemplate=l_tmpl,
            )
            if last_n_hover is not None:
                scatter_kw["text"] = last_n_hover
            fig.add_trace(go.Scatter(**scatter_kw))

    fig.update_layout(
        yaxis_title=y_label,
        showlegend=True,
        height=height,
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        yaxis_ticksuffix=pct_sfx,
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Shared columns list and empty row
# ---------------------------------------------------------------------------

STAT_COLS = [
    "SOG Allowed",
    "Save%",
    "Opp SOG",
    "Opp Goal Conv%",
    "vs Opp SOG",
    "Days Since",
]


def empty_stat_row() -> dict:
    return {c: "—" for c in STAT_COLS}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_team_stat_row(
    all_logs: pd.DataFrame,
    team: str,
    opponent: str,
    simplified: bool = True,
) -> dict:
    """Compute all stat values for a team row."""
    try:
        sog = team_sog_stats(all_logs, team)
        svp = team_save_pct_stats(all_logs, team)
        opp_sog = opponent_sog_stats(all_logs, opponent)
        opp_gr = opponent_goal_rate_stats(all_logs, opponent)
        vs_sog = vs_opponent_sog_season(all_logs, team, opponent)
        team_games = all_logs[all_logs["teamAbbrev"] == team]
        days = days_since_last_game(team_games) if not team_games.empty else None

        if simplified:
            return {
                "SOG Allowed": fmt_combined(sog["season"], sog["last_n"], decimals=0),
                "Save%": fmt_sv_pct_combined(svp["season"], svp["last_n"]),
                "Opp SOG": fmt_combined(
                    opp_sog["season"], opp_sog["last_n"], decimals=0
                ),
                "Opp Goal Conv%": fmt_combined(
                    opp_gr["season"], opp_gr["last_n"], pct=True
                ),
                "vs Opp SOG": fmt_val(vs_sog, key="mean", decimals=0),
                "Days Since": str(days) if days is not None else "—",
            }
        else:
            return {
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
    except Exception:
        return empty_stat_row()


def build_goalie_stat_row(
    player_id: int,
    goalie_log: pd.DataFrame,
    all_logs: pd.DataFrame,
    opponent: str,
    simplified: bool = True,
) -> dict:
    """Compute all stat values for a goalie row."""
    if goalie_log.empty:
        return empty_stat_row()
    try:
        report = goalie_report(goalie_log, all_logs, opponent)
        if simplified:
            return {
                "SOG Allowed": fmt_combined(
                    report["sog_allowed"]["goalie"]["season"],
                    report["sog_allowed"]["goalie"]["last_n"],
                    decimals=0,
                ),
                "Save%": fmt_sv_pct_combined(
                    report["save_pct"]["season"], report["save_pct"]["last_n"]
                ),
                "Opp SOG": fmt_combined(
                    report["opponent_sog"]["season"],
                    report["opponent_sog"]["last_n"],
                    decimals=0,
                ),
                "Opp Goal Conv%": fmt_combined(
                    report["opponent_goal_rate"]["season"],
                    report["opponent_goal_rate"]["last_n"],
                    pct=True,
                ),
                "vs Opp SOG": fmt_val(
                    report["vs_opponent_sog"], key="mean", decimals=0
                ),
                "Days Since": (
                    str(report["days_since"])
                    if report["days_since"] is not None
                    else "—"
                ),
            }
        else:
            return {
                "SOG Season": fmt_dist(report["sog_allowed"]["goalie"]["season"]),
                "SOG L5": fmt_range(report["sog_allowed"]["goalie"]["last_n"]),
                "Save% Season": fmt_pct_dist(report["save_pct"]["season"]),
                "Save% L5": fmt_range(report["save_pct"]["last_n"], pct=True),
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
    except Exception:
        return empty_stat_row()


def build_team_table_df(
    team_rows: list[dict],
    all_logs: pd.DataFrame,
    simplified: bool = True,
) -> pd.DataFrame:
    """Build the master team stats DataFrame — one row per team, all stats."""
    rows = []
    for ti in team_rows:
        team = ti["team"]
        opponent = ti["opponent"]
        next_game_date = ti.get("next_game_date", "")

        try:
            date_str = pd.to_datetime(next_game_date).strftime("%a %b %-d")
        except Exception:
            date_str = str(next_game_date)

        stat_row = build_team_stat_row(all_logs, team, opponent, simplified)
        rows.append(
            {
                "Team": team,
                "Opponent": opponent,
                "Next Game": date_str,
                **stat_row,
            }
        )
    return pd.DataFrame(rows)


def build_goalie_rows(
    goalie_ids: list[dict],
    all_logs: pd.DataFrame,
    opponent: str,
    season: str,
    simplified: bool = True,
) -> pd.DataFrame:
    """Build a display DataFrame with one row per goalie."""
    rows = []
    for ginfo in goalie_ids:
        pid = ginfo["player_id"]
        name = ginfo["name"]
        log = load_goalie_log(pid, season)
        stat_row = build_goalie_stat_row(pid, log, all_logs, opponent, simplified)
        rows.append({"Goalie": name, **stat_row})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stats guide
# ---------------------------------------------------------------------------


def render_stats_guide_team() -> None:
    with st.expander("Stats Guide & How to Use"):
        st.markdown("### Statistics Explained")
        st.markdown(
            "| Stat | Definition | Format |\n"
            "|------|-----------|--------|\n"
            "| SOG Allowed | SOG the team/goalie concedes per game"
            " | Season (L5): Median (Mean) |\n"
            "| Save % | Saves ÷ SOG shown as proportion"
            " | Season (L5): Median (Mean) |\n"
            "| Opp SOG | SOG the opponent generates per game"
            " | Season (L5): Median (Mean) |\n"
            "| Opp Goal Conv % | Opponent goals ÷ SOG × 100"
            " | Season (L5): Median (Mean) |\n"
            "| vs Opp SOG | SOG allowed vs this opponent this season"
            " | Mean |\n"
            "| Days Since | Days since last game | Single integer |"
        )
        st.markdown(
            "> **Note:** Combined cells show **Season (L5)** values."
            " Season uses median; L5 uses mean of last 5 games."
        )
        st.markdown("---\n### How to Use")
        st.markdown(
            "- All teams are shown in the main table sorted alphabetically.\n"
            "- **Click a team row** to see individual goalie stats"
            " for that team below the main table.\n"
            "- Use **Goalie Detail** in the sidebar for full distributions,"
            " box plots, and history vs a specific opponent."
        )

def render_stats_guide_goalie() -> None:
    with st.expander("How to Use"):
        st.markdown("### How to Use")
        st.markdown(
            "- **Click a team** to see team season and L5 game breakdowns.\n"
            "- **Filter for a specific goalie**, sorted by the number of starts"
            " this season, to alter per goalie stats.\n"
            "- **Opponent filter** automatically selects the next opponent on the"
            " schedule, but can be overridden.\n"
            ">**Note:** Each statistic provides breakdown of Median [25th Percentile,"
            " 75th Percentile] for the season; Mean (Min, Max) for the last 5 games."
            " Boxplots display data points for each game for the team or goalie,"
            " depending on the statistic, in grey and the last 5 games in orange"
        )