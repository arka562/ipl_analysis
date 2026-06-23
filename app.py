import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Resolve the base directory dynamically relative to this script
current_file_dir = Path(__file__).resolve().parent

if (current_file_dir / "reports").exists():
    BASE = current_file_dir
elif (current_file_dir / "ipl_analytics_platform" / "reports").exists():
    BASE = current_file_dir / "ipl_analytics_platform"
else:
    BASE = Path("ipl_analytics_platform")

REPORTS   = BASE / "reports" / "tables"
ADVANCED  = BASE / "reports" / "advanced_metrics"
MODELING  = BASE / "reports" / "modeling"
MODELS    = BASE / "models"

# Handle split data directories (processed data in parsers folder)
if (current_file_dir / "parsers" / "ipl_analytics_platform" / "data" / "processed").exists():
    PROCESSED = current_file_dir / "parsers" / "ipl_analytics_platform" / "data" / "processed"
else:
    PROCESSED = BASE / "data" / "processed"

st.set_page_config(
    page_title="IPL Analytics Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────
BG        = "#0f1117"
SURFACE   = "#1a1d27"
BORDER    = "#2a2d3a"
ORANGE    = "#f97316"
BLUE      = "#38bdf8"
GREEN     = "#4ade80"
PURPLE    = "#c084fc"
YELLOW    = "#fbbf24"
TEXT      = "#f1f5f9"
MUTED     = "#94a3b8"

PLOT_LAYOUT = dict(
    plot_bgcolor  = SURFACE,
    paper_bgcolor = BG,
    font          = dict(color=TEXT, family="DM Sans, sans-serif"),
    xaxis         = dict(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED),
    yaxis         = dict(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    margin        = dict(l=10, r=10, t=40, b=10),
    hoverlabel    = dict(
        bgcolor=SURFACE,
        bordercolor=BORDER,
        font=dict(color=TEXT, family="DM Sans, sans-serif")
    ),
)

ROLE_COLORS = {
    "Batter":      BLUE,
    "Bowler":      GREEN,
    "All-rounder": ORANGE,
}

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG};
    color: {TEXT};
}}
h1, h2, h3 {{
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
    color: {TEXT};
}}
.block-container {{ padding-top: 1.5rem; max-width: 1400px; }}

/* Metric Cards Premium Styling */
div[data-testid="stMetric"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    padding: 0.8rem 1.2rem;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}
div[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    border-color: {ORANGE};
    box-shadow: 0 6px 20px rgba(249, 115, 22, 0.15);
}}

[data-testid="stMetricValue"] {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem !important;
    color: {ORANGE};
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.72rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.8rem;
    color: {BLUE};
}}
section[data-testid="stSidebar"] {{
    background-color: {SURFACE};
    border-right: 1px solid {BORDER};
}}
.sidebar-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    color: {ORANGE};
    letter-spacing: 3px;
}}
.stDataFrame {{ background-color: {SURFACE}; }}
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stRadio"] label {{
    color: {MUTED};
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
hr {{ border-color: {BORDER}; }}
.stButton > button {{
    background-color: {ORANGE};
    color: #000;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 1px;
    border: none;
    padding: 0.5rem 2rem;
    border-radius: 4px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(249, 115, 22, 0.2);
}}
.stButton > button:hover {{
    background-color: {YELLOW};
    box-shadow: 0 6px 20px rgba(251, 191, 36, 0.3);
    transform: translateY(-1px);
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_data
def load(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏏 IPL Analytics</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "Season Overview",
            "Team Performance",
            "Player Analysis",
            "Venue Insights",
            "Win Probability",
            "Score Predictor",
            "Tournament Simulator",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Data: Cricsheet · 2008–present\n294K+ deliveries · 1,239 matches")


# ─────────────────────────────────────────────
# PAGE 1 — SEASON OVERVIEW
# ─────────────────────────────────────────────
if page == "Season Overview":
    st.title("Season Overview")
    st.caption("How IPL scoring and match dynamics have evolved across seasons.")

    df = load(REPORTS / "season_summary.csv")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Seasons",        df["season"].nunique())
    c2.metric("Total Matches",  f"{df['matches'].sum():,}")
    c3.metric("Total Runs",     f"{df['total_runs'].sum():,.0f}")
    c4.metric("Total Wickets",  f"{df['total_wickets'].sum():,.0f}")

    st.markdown("### Run Rate by Season")
    fig = px.bar(
        df, x="season", y="run_rate",
        color="run_rate",
        color_continuous_scale=[[0, "#0284c7"], [0.5, BLUE], [1.0, ORANGE]],
        labels={"run_rate": "RPO", "season": "Season"},
        text="run_rate",
    )
    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        textfont=dict(color=TEXT, size=11),
        marker_line_width=0,
    )
    fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Avg Innings Score")
        fig2 = px.line(
            df, x="season", y="avg_innings_score",
            markers=True,
            labels={"avg_innings_score": "Avg Score", "season": "Season"},
            color_discrete_sequence=[BLUE],
        )
        fig2.update_traces(
            marker=dict(size=10, color="#ffffff", line=dict(color=BLUE, width=3)),
            line=dict(width=3.5),
        )
        fig2.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### Wickets per Season")
        fig3 = px.line(
            df, x="season", y="total_wickets",
            markers=True,
            labels={"total_wickets": "Wickets", "season": "Season"},
            color_discrete_sequence=[GREEN],
        )
        fig3.update_traces(
            marker=dict(size=10, color="#ffffff", line=dict(color=GREEN, width=3)),
            line=dict(width=3.5),
        )
        fig3.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Full Season Table")
    st.dataframe(df.sort_values("season", ascending=False), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE 2 — TEAM PERFORMANCE
# ─────────────────────────────────────────────
elif page == "Team Performance":
    st.title("Team Performance")
    st.caption("Win rates, toss impact, chasing trends, and phase-wise batting strength.")

    teams   = load(REPORTS / "team_summary.csv")
    toss    = load(REPORTS / "toss_impact_by_season.csv")
    chasing = load(REPORTS / "chasing_vs_defending.csv")
    phase   = load(ADVANCED / "team_phase_strength.csv")

    best  = teams.loc[teams["win_pct"].idxmax()]
    most  = teams.loc[teams["matches"].idxmax()]
    avg_toss = toss["toss_winner_win_pct"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Highest Win %",  f"{best['win_pct']}%",  best["team"])
    c2.metric("Most Matches",   f"{most['matches']}",   most["team"])
    c3.metric("Avg Toss Win %", f"{avg_toss:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Win % by Team")
        fig = px.bar(
            teams.sort_values("win_pct", ascending=True),
            x="win_pct", y="team", orientation="h",
            color="win_pct",
            color_continuous_scale=[[0, "#0284c7"], [0.5, BLUE], [1.0, ORANGE]],
            labels={"win_pct": "Win %", "team": ""},
            text="win_pct",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            textfont=dict(color=TEXT, size=11),
            marker_line_width=0,
        )
        fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Toss Winner Win % by Season")
        fig2 = px.line(
            toss, x="season", y="toss_winner_win_pct",
            markers=True,
            labels={"toss_winner_win_pct": "Win %", "season": "Season"},
            color_discrete_sequence=[ORANGE],
        )
        fig2.update_traces(
            marker=dict(size=11, color="#ffffff", line=dict(color=ORANGE, width=3)),
            line=dict(width=3.5),
        )
        fig2.add_hline(y=50, line_dash="dash", line_color=MUTED,
                       annotation_text="50%", annotation_font_color=MUTED)
        fig2.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Chasing vs Defending by Season")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=chasing["season"], y=chasing["chasing_win_pct"],
        name="Chasing Win %", mode="lines+markers",
        line=dict(color=BLUE, width=3.5),
        marker=dict(size=11, color="#ffffff", line=dict(color=BLUE, width=3)),
    ))
    fig3.add_trace(go.Scatter(
        x=chasing["season"], y=chasing["defending_wins"] / chasing["completed_matches"] * 100,
        name="Defending Win %", mode="lines+markers",
        line=dict(color=ORANGE, width=3.5),
        marker=dict(size=11, color="#ffffff", line=dict(color=ORANGE, width=3)),
    ))
    fig3.add_hline(y=50, line_dash="dash", line_color=MUTED,
                   annotation_text="50%", annotation_font_color=MUTED)
    fig3.update_layout(**PLOT_LAYOUT, xaxis_title="Season", yaxis_title="Win %")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Team Phase Strength Score")
    phase_pivot = phase.pivot_table(
        index="team", columns="phase", values="team_phase_strength_score"
    ).reset_index()
    st.dataframe(phase_pivot, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE 3 — PLAYER ANALYSIS
# ─────────────────────────────────────────────
elif page == "Player Analysis":
    st.title("Player Analysis")
    st.caption("Impact scores, phase performance, and death-over specialists.")

    impact    = load(ADVANCED / "player_impact_score.csv")
    death_bat = load(ADVANCED / "death_over_batting_index.csv")
    death_bowl= load(ADVANCED / "death_over_bowling_index.csv")
    phase_bat = load(REPORTS  / "top_batters_by_phase.csv")

    eligible = impact[impact["impact_eligible"] == 1].copy()
    top      = eligible.iloc[0]
    batters  = eligible[eligible["role_signal"] == "Batter"]
    bowlers  = eligible[eligible["role_signal"] == "Bowler"]
    top_bat  = batters.iloc[0]  if not batters.empty  else eligible.iloc[0]
    top_bowl = bowlers.iloc[0]  if not bowlers.empty  else eligible.iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Top Impact Player", top["player"],     f"Score: {top['player_impact_score']}")
    c2.metric("Top Batter",        top_bat["player"], f"SR: {top_bat['batting_strike_rate']}")
    c3.metric("Top Bowler",        top_bowl["player"],f"Eco: {top_bowl['bowling_economy']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top 20 by Impact Score")
        top20 = eligible.head(20)
        fig = px.bar(
            top20.sort_values("player_impact_score"),
            x="player_impact_score", y="player", orientation="h",
            color="role_signal",
            color_discrete_map=ROLE_COLORS,
            labels={"player_impact_score": "Impact Score", "player": ""},
            text="player_impact_score",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
            textfont=dict(color=TEXT, size=11),
            marker_line_width=0,
        )
        fig.update_layout(**PLOT_LAYOUT, legend_title="Role")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Strike Rate vs Runs")
        top_bat_df = eligible[eligible["batting_balls"] >= 600].head(40)
        fig2 = px.scatter(
            top_bat_df, x="batting_runs", y="batting_strike_rate",
            hover_name="player", color="role_signal",
            color_discrete_map=ROLE_COLORS,
            size="batting_balls",
            size_max=20,
            labels={"batting_runs": "Runs", "batting_strike_rate": "Strike Rate"},
        )
        fig2.update_traces(
            marker=dict(
                line=dict(color="#ffffff", width=1.5),
                sizemin=8,
            ),
            opacity=0.9,
        )
        fig2.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Phase-wise Top Batters")
    phase_sel  = st.selectbox("Phase", ["Powerplay", "Middle", "Death"])
    phase_data = phase_bat[phase_bat["phase"] == phase_sel].copy()
    fig3 = px.bar(
        phase_data.sort_values("runs", ascending=True),
        x="runs", y="batter", orientation="h",
        color="strike_rate",
        color_continuous_scale=[[0, "#0284c7"], [0.5, BLUE], [1.0, ORANGE]],
        labels={"runs": "Runs", "batter": "", "strike_rate": "SR"},
        text="runs",
    )
    fig3.update_traces(
        texttemplate="%{text}",
        textposition="outside",
        textfont=dict(color=TEXT, size=11),
        marker_line_width=0,
    )
    fig3.update_layout(**PLOT_LAYOUT, coloraxis_showscale=True,
                       coloraxis_colorbar=dict(title="SR", tickfont=dict(color=MUTED)))
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Death Batting Index — Top 15")
        st.dataframe(
            death_bat.head(15)[[
                "player", "death_batting_index",
                "death_runs", "death_strike_rate", "death_boundary_pct"
            ]],
            use_container_width=True, hide_index=True,
        )
    with col4:
        st.markdown("### Death Bowling Index — Top 15")
        st.dataframe(
            death_bowl.head(15)[[
                "player", "death_bowling_index",
                "death_wickets", "death_economy", "death_dot_ball_pct"
            ]],
            use_container_width=True, hide_index=True,
        )


# ─────────────────────────────────────────────
# PAGE 4 — VENUE INSIGHTS
# ─────────────────────────────────────────────
elif page == "Venue Insights":
    st.title("Venue Insights")
    st.caption("Par scores, chasing success rates, and ground characteristics.")

    venues    = load(REPORTS  / "venue_summary.csv")
    venue_par = load(ADVANCED / "venue_par_score.csv")

    highest    = venues.loc[venues["avg_first_innings_score"].idxmax()]
    best_chase = venue_par.loc[venue_par["chasing_win_pct"].idxmax()]

    c1, c2, c3 = st.columns(3)
    c1.metric("Highest Scoring Ground", highest["venue"].split(",")[0],
              f"{highest['avg_first_innings_score']} avg")
    c2.metric("Best for Chasing", best_chase["venue"].split(",")[0],
              f"{best_chase['chasing_win_pct']}%")
    c3.metric("Venues Tracked", len(venue_par))

    st.markdown("### Avg First Innings Score by Venue (Top 20)")
    fig = px.bar(
        venues.sort_values("avg_first_innings_score", ascending=True).tail(20),
        x="avg_first_innings_score", y="venue", orientation="h",
        color="avg_first_innings_score",
        color_continuous_scale=[[0, "#0284c7"], [0.5, BLUE], [1.0, ORANGE]],
        labels={"avg_first_innings_score": "Avg First Innings", "venue": ""},
        text="avg_first_innings_score",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        textfont=dict(color=TEXT, size=11),
        marker_line_width=0,
    )
    fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Chasing Win % by Venue (Top 15)")
        fig2 = px.bar(
            venue_par.sort_values("chasing_win_pct", ascending=True).tail(15),
            x="chasing_win_pct", y="venue", orientation="h",
            color="chasing_win_pct",
            color_continuous_scale=[[0, "#047857"], [0.5, GREEN], [1.0, YELLOW]],
            labels={"chasing_win_pct": "Chasing Win %", "venue": ""},
            text="chasing_win_pct",
        )
        fig2.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            textfont=dict(color=TEXT, size=11),
            marker_line_width=0,
        )
        fig2.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### Par Score vs Median Score")
        fig3 = px.scatter(
            venue_par,
            x="venue_par_score", y="median_first_innings_score",
            hover_name="venue", size="matches",
            color="chasing_win_pct",
            color_continuous_scale=[[0, "#ef4444"], [0.5, YELLOW], [1.0, GREEN]],
            size_max=22,
            labels={
                "venue_par_score":           "Avg First Innings (Par)",
                "median_first_innings_score":"Median First Innings",
                "chasing_win_pct":           "Chase Win %",
            },
        )
        fig3.update_traces(
            marker=dict(
                line=dict(color="#ffffff", width=1.5),
                sizemin=8,
            ),
            opacity=0.9,
        )
        fig3.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Full Venue Table")
    st.dataframe(venue_par, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE 5 — WIN PROBABILITY
# ─────────────────────────────────────────────
elif page == "Win Probability":
    st.title("Win Probability Curve")
    st.caption("Over-by-over win probability for any match in the dataset.")

    training     = load(MODELING / "win_probability_training_data.csv")
    model        = load_model(MODELS / "win_probability_model.joblib")
    metrics_data = load_json(MODELING / "win_probability_metrics.json")
    features     = metrics_data["features"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Accuracy", f"{metrics_data['accuracy']*100:.1f}%")
    c2.metric("ROC AUC",        metrics_data["roc_auc"])
    c3.metric("Log Loss",       metrics_data["log_loss"])

    all_matches = sorted(training["match_id"].unique())
    selected    = st.selectbox("Select Match ID", all_matches, index=0)

    match_df = training[training["match_id"] == selected].copy()
    match_df["win_probability"] = (
        model.predict_proba(match_df[features])[:, 1] * 100
    ).round(1)

    colors_wp = [BLUE, ORANGE]
    fig = go.Figure()
    for i, (key, grp) in enumerate(match_df.groupby(["innings", "batting_team"])):
        innings_no, team = key
        grp = grp.sort_values("over_number")
        col = colors_wp[i % 2]
        fig.add_trace(go.Scatter(
            x=grp["over_number"],
            y=grp["win_probability"],
            mode="lines+markers",
            name=f"Inn {int(innings_no)}: {team}",
            line=dict(color=col, width=3.5),
            marker=dict(
                size=11,
                color="#ffffff",
                line=dict(color=col, width=3),
                symbol="circle",
            ),
            hovertemplate=(
                f"<b>{team}</b><br>"
                "Over %{x}<br>"
                "Win Prob: <b>%{y:.1f}%</b><br>"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=50, line_dash="dash", line_color=MUTED,
                  annotation_text="50%", annotation_font_color=MUTED)
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_layout(
        xaxis_title="Over",
        yaxis_title="Win Probability (%)",
        yaxis=dict(range=[0, 100], gridcolor=BORDER, color=MUTED),
        xaxis=dict(gridcolor=BORDER, color=MUTED, dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Match data table"):
        st.dataframe(
            match_df[[
                "innings", "batting_team", "over_number",
                "current_score", "wickets_lost", "target",
                "runs_required", "win_probability", "batting_team_won",
            ]],
            use_container_width=True, hide_index=True,
        )


# ─────────────────────────────────────────────
# PAGE 6 — SCORE PREDICTOR
# ─────────────────────────────────────────────
elif page == "Score Predictor":
    st.title("Score Predictor")
    st.caption("Enter first 10 overs stats to predict final innings score.")

    score_model   = load_model(MODELS / "final_score_predictor_v3.joblib")
    score_metrics = load_json(MODELING / "score_model3_metrics.json")
    features      = score_metrics["features"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Model MAE",  f"±{score_metrics['mae']} runs")
    c2.metric("R² Score",   score_metrics["r2"])
    c3.metric("RMSE",       f"{score_metrics['rmse']} runs")

    st.markdown("### Match Inputs")
    teams_df  = load(PROCESSED / "matches.csv")
    team_list = sorted(teams_df["team_1"].dropna().unique().tolist())
    venue_list= sorted(teams_df["venue"].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        season      = st.number_input("Season",      min_value=2008, max_value=2025, value=2024)
        innings     = st.selectbox("Innings",        [1, 2])
        batting_team= st.selectbox("Batting Team",   team_list)
        venue       = st.selectbox("Venue",          venue_list)

    with col2:
        runs_so_far  = st.number_input("Runs after 10 overs", min_value=0,  max_value=200, value=65)
        wickets_so_far=st.number_input("Wickets lost",        min_value=0,  max_value=10,  value=2)
        balls_so_far = st.number_input("Balls bowled",        min_value=30, max_value=60,  value=60)
        boundaries   = st.number_input("Boundaries hit",      min_value=0,  max_value=40,  value=8)

    with col3:
        sixes        = st.number_input("Sixes hit",           min_value=0,  max_value=20,  value=2)
        dot_balls    = st.number_input("Dot balls",           min_value=0,  max_value=60,  value=20)
        target_score = st.number_input("Target (2nd inn only, else 0)", min_value=0, max_value=300, value=0)

    b = max(balls_so_far, 1)
    run_rate_so_far  = runs_so_far * 6 / b
    wickets_remaining= 10 - wickets_so_far
    boundary_pct     = 100 * boundaries / b
    dot_ball_pct     = 100 * dot_balls / b
    six_pct          = 100 * sixes / b
    is_first_innings = int(innings == 1)
    balls_remaining  = 120 - balls_so_far
    runs_needed      = max(target_score - runs_so_far, 0) if innings == 2 else 0
    required_run_rate= (runs_needed * 6 / max(balls_remaining, 1)) if innings == 2 else 0

    input_df = pd.DataFrame([{
        "season": season, "innings": innings, "batting_team": batting_team, "venue": venue,
        "runs_so_far": runs_so_far, "wickets_so_far": wickets_so_far,
        "balls_so_far": balls_so_far, "boundaries": boundaries, "sixes": sixes,
        "dot_balls": dot_balls, "run_rate_so_far": run_rate_so_far,
        "wickets_remaining": wickets_remaining, "boundary_pct": boundary_pct,
        "dot_ball_pct": dot_ball_pct, "six_pct": six_pct,
        "is_first_innings": is_first_innings, "balls_remaining": balls_remaining,
        "target_score": target_score, "runs_needed": runs_needed,
        "required_run_rate": required_run_rate,
    }])

    if st.button("Predict Final Score", type="primary"):
        prediction = int(score_model.predict(input_df[features])[0])
        low  = prediction - int(score_metrics["mae"])
        high = prediction + int(score_metrics["mae"])

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Score", prediction)
        r2.metric("Lower Bound",     low)
        r3.metric("Upper Bound",     high)

        # mini gauge bar
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            gauge=dict(
                axis=dict(range=[0, 280], tickcolor=MUTED),
                bar=dict(color=ORANGE),
                bgcolor=SURFACE,
                bordercolor=BORDER,
                steps=[
                    dict(range=[0,   140], color="#1e293b"),
                    dict(range=[140, 180], color="#3b82f6"),
                    dict(range=[180, 220], color=ORANGE),
                    dict(range=[220, 280], color="#ef4444"),
                ],
                threshold=dict(line=dict(color=YELLOW, width=4), thickness=0.75, value=prediction),
            ),
            number=dict(font=dict(color=TEXT, size=48, family="Bebas Neue")),
            title=dict(text="Predicted Final Score", font=dict(color=MUTED, size=14)),
        ))
        fig_g.update_layout(
            paper_bgcolor=BG,
            font=dict(color=TEXT),
            height=300,
            margin=dict(l=30, r=30, t=30, b=10),
        )
        st.plotly_chart(fig_g, use_container_width=True)

        st.info(
            f"Based on **{runs_so_far}/{wickets_so_far}** after **{balls_so_far}** balls, "
            f"the model predicts a final score of **{prediction}** runs "
            f"(confidence range: **{low}–{high}**)."
        )

# ─────────────────────────────────────────────
# PAGE 7 — TOURNAMENT SIMULATOR
# ─────────────────────────────────────────────
elif page == "Tournament Simulator":
    st.title("Tournament Simulator")
    st.caption("Monte Carlo simulation results for an upcoming IPL season.")

    try:
        sim_results = load(MODELING / "tournament_simulation_results.csv")
        
        top_team = sim_results.iloc[0]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Highest Champ Prob", f"{top_team['Championship Probability (%)']}%", top_team["Team"])
        c2.metric("Highest Playoff Prob", f"{sim_results['Playoff Probability (%)'].max()}%")
        c3.metric("Simulations Run", "1,000")
        
        st.markdown("### Championship Probability by Team")
        fig = px.bar(
            sim_results.sort_values("Championship Probability (%)", ascending=True),
            x="Championship Probability (%)", y="Team", orientation="h",
            color="Championship Probability (%)",
            color_continuous_scale=[[0, "#0284c7"], [0.5, BLUE], [1.0, ORANGE]],
            labels={"Championship Probability (%)": "Win Probability (%)", "Team": ""},
            text="Championship Probability (%)",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            textfont=dict(color=TEXT, size=11),
            marker_line_width=0,
        )
        fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Full Simulation Results")
        st.dataframe(sim_results, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.warning("Tournament Simulation results not found. Please run `python tournament_simulator.py` first.")
