"""NHL Saves — Exploratory Data Analysis (Phase 1).

Run sections interactively in Positron (select block → Run in Console).
Each section is independently runnable after the "Load data" section.
"""

# ── Section 1: Load data ──────────────────────────────────────────────────────
import sys
from pathlib import Path

# Allow running from project root (interactive) or analysis/ directory (script)
try:
    _src = Path(__file__).parent.parent / "src"
except NameError:
    _src = Path.cwd() / "src"
sys.path.insert(0, str(_src))

import pandas as pd  # noqa: E402

from nhl_saves.modeling.features import build_model_dataset  # noqa: E402
from nhl_saves.moneypuck import fetch_mp_player_bios  # noqa: E402

df = build_model_dataset(["20242025", "20252026"])
print(df.shape)
print(list(df.columns))
df.describe()


# ── Section 2: Target distribution ───────────────────────────────────────────
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram + KDE
axes[0].hist(df["saves"].dropna(), bins=30, edgecolor="white", color="#4878d0")
axes[0].set_xlabel("Saves")
axes[0].set_ylabel("Count")
axes[0].set_title("Target Distribution: Saves per Start")

sns.kdeplot(df["saves"].dropna(), ax=axes[1], fill=True, color="#4878d0")
axes[1].set_xlabel("Saves")
axes[1].set_title("Saves — KDE")

plt.tight_layout()
plt.show()

print(f"Skewness: {df['saves'].skew():.3f}")
print(f"Kurtosis: {df['saves'].kurt():.3f}")
print(df["saves"].describe())


# ── Section 3: Feature distributions ─────────────────────────────────────────
feat_cols = ["rest_days", "sog_roll5", "opp_sog_season_avg"]
fig, axes = plt.subplots(1, len(feat_cols), figsize=(14, 4))

for ax, col in zip(axes, feat_cols):
    data = df[col].dropna()
    ax.hist(data, bins=30, edgecolor="white", color="#ee854a")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(f"{col}\n(n={len(data)}, skew={data.skew():.2f})")

plt.tight_layout()
plt.show()


# ── Section 4: Bivariate exploration ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 4a — Opponent SOG season avg vs saves (scatter)
axes[0].scatter(
    df["opp_sog_season_avg"],
    df["saves"],
    alpha=0.3,
    s=12,
    color="#4878d0",
)
axes[0].set_xlabel("opp_sog_season_avg")
axes[0].set_ylabel("saves")
axes[0].set_title("Opp SOG Season Avg vs Saves")

corr = df[["opp_sog_season_avg", "saves"]].corr().iloc[0, 1]
axes[0].annotate(f"r = {corr:.3f}", xy=(0.05, 0.92), xycoords="axes fraction")

# 4b — Rest days bucket vs saves (box plot)
rest_bins = [-float("inf"), 1, 3, float("inf")]
rest_labels = ["0–1 (B2B)", "2–3", "4+"]
df["rest_bucket"] = pd.cut(df["rest_days"], bins=rest_bins, labels=rest_labels)
rest_groups = [
    df.loc[df["rest_bucket"] == lbl, "saves"].dropna() for lbl in rest_labels
]
axes[1].boxplot(rest_groups, tick_labels=rest_labels)
axes[1].set_xlabel("Rest Days")
axes[1].set_ylabel("Saves")
axes[1].set_title("Rest Days Bucket vs Saves")

# 4c — Home vs road (box plot)
home_saves = df.loc[df["is_home"] == 1, "saves"].dropna()
road_saves = df.loc[df["is_home"] == 0, "saves"].dropna()
axes[2].boxplot([home_saves, road_saves], tick_labels=["Home", "Road"])
axes[2].set_xlabel("Home / Road")
axes[2].set_ylabel("Saves")
axes[2].set_title("Home vs Road Saves")

plt.tight_layout()
plt.show()


# ── Section 5: Correlation matrix ────────────────────────────────────────────
numeric_features = [
    "saves",
    "shots_against",
    "save_pct_roll5",
    "save_pct_roll10",
    "sog_roll5",
    "sog_roll10",
    "rest_days",
    "is_home",
    "is_back_to_back",
    "team_home_streak",
    "opp_sog_season_avg",
    "team_sog_allowed_season_avg",
    "opp_goal_conv_season",
    "goalie_starts_ytd",
]
available_feats = [c for c in numeric_features if c in df.columns]
corr_matrix = df[available_feats].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 8},
)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# Highlight strongest correlations with target
target_corr = corr_matrix["saves"].drop("saves").sort_values(key=abs, ascending=False)
print("\nCorrelations with 'saves' (sorted by |r|):")
print(target_corr.to_string())


# ── Section 6: Feature importance preview (RandomForest) ─────────────────────
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

predictor_cols = [c for c in available_feats if c != "saves"]
model_df = df[available_feats].dropna()

X = model_df[predictor_cols]
y = model_df["saves"]

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=predictor_cols).sort_values(
    ascending=True
)

fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind="barh", ax=ax, color="#4878d0")
ax.set_xlabel("Feature Importance")
ax.set_title("RandomForest Feature Importances (untuned — directional only)")
plt.tight_layout()
plt.show()

print("\nFeature importances:")
print(importances.sort_values(ascending=False).to_string())


# ── Section 7: Situational breakdown ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 7a — saves by toi_pk quartile (more PK time → harder saves?)
if "toi_pk" in df.columns:
    df["toi_pk_q"] = pd.qcut(
        df["toi_pk"].dropna(),
        q=4,
        labels=["Q1\n(least PK)", "Q2", "Q3", "Q4\n(most PK)"],
        duplicates="drop",
    )
    pk_groups = [
        df.loc[df["toi_pk_q"] == q, "saves"].dropna()
        for q in df["toi_pk_q"].cat.categories
    ]
    axes[0].boxplot(pk_groups, tick_labels=df["toi_pk_q"].cat.categories)
    axes[0].set_xlabel("PK TOI Quartile")
    axes[0].set_ylabel("Saves")
    axes[0].set_title("Saves by PK TOI Quartile")
else:
    axes[0].set_title("toi_pk not available")

# 7b — high danger shots vs saves (scatter)
if "high_danger_shots" in df.columns:
    axes[1].scatter(
        df["high_danger_shots"].dropna(),
        df.loc[df["high_danger_shots"].notna(), "saves"],
        alpha=0.2,
        s=10,
        color="#6acc65",
    )
    corr_hd = df[["high_danger_shots", "saves"]].corr().iloc[0, 1]
    axes[1].set_xlabel("High Danger Shots")
    axes[1].set_ylabel("Saves")
    axes[1].set_title(f"High Danger Shots vs Saves  (r={corr_hd:.3f})")
else:
    axes[1].set_title("high_danger_shots not available")

plt.tight_layout()
plt.show()


# ── Section 8: Multi-season trends ───────────────────────────────────────────
if "season" in df.columns:
    _trend_candidates = ["saves", "high_danger_shots", "opp_corsi_pct_ytd"]
    trend_cols = [c for c in _trend_candidates if c in df.columns]
    season_medians = df.groupby("season")[trend_cols].median()

    fig, axes = plt.subplots(1, len(trend_cols), figsize=(5 * len(trend_cols), 4))
    if len(trend_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, trend_cols):
        ax.plot(
            season_medians.index.astype(str),
            season_medians[col],
            marker="o",
            color="#4878d0",
        )
        ax.set_xlabel("Season")
        ax.set_ylabel(f"Median {col}")
        ax.set_title(f"{col} — Median by Season")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Multi-Season Trend Detection", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("'season' column not available — run with multi-season dataset for Section 8")


# ── Section 9: Fatigue analysis ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 9a — 4-way B2B box plot: both fresh / goalie only / team only / both B2B
if "is_back_to_back" in df.columns and "team_is_back_to_back" in df.columns:
    b2b_df = df.dropna(
        subset=["is_back_to_back", "team_is_back_to_back", "saves"]
    ).copy()
    b2b_df["b2b_combo"] = b2b_df.apply(
        lambda r: (
            "Both Fresh"
            if r["is_back_to_back"] == 0 and r["team_is_back_to_back"] == 0
            else (
                "Goalie B2B\nTeam Fresh"
                if r["is_back_to_back"] == 1 and r["team_is_back_to_back"] == 0
                else (
                    "Goalie Fresh\nTeam B2B"
                    if r["is_back_to_back"] == 0 and r["team_is_back_to_back"] == 1
                    else "Both B2B"
                )
            )
        ),
        axis=1,
    )
    labels = [
        "Both Fresh", "Goalie B2B\nTeam Fresh", "Goalie Fresh\nTeam B2B", "Both B2B"
    ]
    groups = [b2b_df.loc[b2b_df["b2b_combo"] == lbl, "saves"] for lbl in labels]
    axes[0].boxplot(groups, tick_labels=labels)
    axes[0].set_ylabel("Saves")
    axes[0].set_title("Saves by B2B Status (4-way)")
else:
    axes[0].set_title("B2B columns not available")

# 9b — goalie_km_last_7d vs saves
if "goalie_km_last_7d" in df.columns:
    valid = df.dropna(subset=["goalie_km_last_7d", "saves"])
    axes[1].scatter(
        valid["goalie_km_last_7d"], valid["saves"], alpha=0.2, s=10, color="#ee854a"
    )
    corr_gkm = df[["goalie_km_last_7d", "saves"]].corr().iloc[0, 1]
    axes[1].set_xlabel("Goalie Travel km (last 7d)")
    axes[1].set_ylabel("Saves")
    axes[1].set_title(f"Goalie km (7d) vs Saves  (r={corr_gkm:.3f})")
else:
    axes[1].set_title("goalie_km_last_7d not available")

# 9c — team_games_last_7d bucket vs saves
if "team_games_last_7d" in df.columns:
    df["games_7d_bucket"] = pd.cut(
        df["team_games_last_7d"],
        bins=[-float("inf"), 2.5, 3.5, 4.5, float("inf")],
        labels=["≤2", "3", "4", "5+"],
    )
    gam_groups = [
        df.loc[df["games_7d_bucket"] == lbl, "saves"].dropna()
        for lbl in ["≤2", "3", "4", "5+"]
    ]
    axes[2].boxplot(gam_groups, tick_labels=["≤2", "3", "4", "5+"])
    axes[2].set_xlabel("Team Games (last 7 days)")
    axes[2].set_ylabel("Saves")
    axes[2].set_title("Team Schedule Load vs Saves")
else:
    axes[2].set_title("team_games_last_7d not available")

plt.suptitle("Section 9 — Fatigue Features", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()


# ── Section 10: Covariate analysis at 3 levels ───────────────────────────────

# 10a — Level 1: Full-dataset correlation heatmap (extended feature set)
extended_numeric = [
    c
    for c in df.columns
    if df[c].dtype in ("float64", "int64", "float32", "int32")
    and c not in ("player_id", "gameId")
]
# Keep top 20 features by correlation with saves to keep heatmap readable
if "saves" in extended_numeric:
    top_corr = (
        df[extended_numeric]
        .corr()["saves"]
        .drop("saves")
        .abs()
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )
    heatmap_cols = ["saves"] + top_corr
    corr20 = df[heatmap_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr20,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Level 1: Full-Dataset Correlation Heatmap (top 20 features by |r|)")
    plt.tight_layout()
    plt.show()

    print("\nTop 20 features correlated with saves:")
    target_corr20 = corr20["saves"].drop("saves").sort_values(key=abs, ascending=False)
    print(target_corr20.to_string())

# 10b — Level 2: Per-team saves distribution (grouped bar: median + IQR)
if "teamAbbrev" in df.columns:
    team_stats = (
        df.groupby("teamAbbrev")["saves"]
        .agg(["median", lambda x: x.quantile(0.75) - x.quantile(0.25)])
        .rename(columns={"median": "median_saves", "<lambda_0>": "iqr_saves"})
        .sort_values("median_saves", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(16, 5))
    x = range(len(team_stats))
    ax.bar(
        x, team_stats["median_saves"], color="#4878d0", alpha=0.8, label="Median saves"
    )
    ax.errorbar(
        x,
        team_stats["median_saves"],
        yerr=team_stats["iqr_saves"] / 2,
        fmt="none",
        color="#333",
        linewidth=1.2,
        capsize=3,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(team_stats.index, rotation=90)
    ax.set_ylabel("Saves")
    ax.set_title("Level 2: Saves Distribution by Team (median ± IQR/2)")
    ax.legend()
    plt.tight_layout()
    plt.show()

# 10c — Level 3: Rolling save_pct trend for top 10 goalies by starts
if "save_pct_roll5" in df.columns and "player_id" in df.columns:
    top10_pids = (
        df.groupby("player_id")["saves"].count().sort_values(ascending=False).head(10).index
    )

    # Attempt to get names from MP bios
    pid_names = {}
    try:
        bio_names = fetch_mp_player_bios()[["playerId", "name"]]
        pid_names = bio_names.set_index("playerId")["name"].to_dict()
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10.colors
    for i, pid in enumerate(top10_pids):
        player_df = df[df["player_id"] == pid].sort_values("gameDate")
        label = pid_names.get(int(pid), f"Player {pid}")
        ax.plot(
            player_df["gameDate"],
            player_df["save_pct_roll5"],
            alpha=0.75,
            linewidth=1.2,
            label=label,
            color=colors[i % 10],
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Save % (5-game rolling)")
    ax.set_title("Level 3: Rolling Save % Trend — Top 10 Goalies by Starts")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()
