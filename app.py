"""
IPL Win Probability & Player Stats Dashboard
Run locally:  streamlit run app.py
Deploy:       push this folder + requirements.txt to Render (see README.md)
"""

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# PAGE CONFIG + GLASSMORPHISM CSS
# ============================================================================

st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background gradient */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 10%, #1e2757 0%, #0d1224 45%, #090c1a 100%);
}
[data-testid="stHeader"] { background: transparent; }

/* Glass cards -- applied to bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.055);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 20px;
    padding: 4px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    padding: 16px 18px;
    backdrop-filter: blur(10px);
}
[data-testid="stMetricValue"] { color: #ffffff; font-weight: 800; }
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.65); }

/* Titles */
h1 {
    background: linear-gradient(120deg, #7C9CFF 0%, #C792EA 55%, #FF9E80 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: -0.5px;
}
h2, h3 { color: #eef0ff; font-weight: 700; }
p, label, .stMarkdown { color: rgba(255,255,255,0.85); }

/* Buttons */
.stButton > button {
    background: linear-gradient(120deg, rgba(124,156,255,0.35), rgba(199,146,234,0.35));
    backdrop-filter: blur(10px);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 14px;
    font-weight: 600;
    padding: 10px 22px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124,156,255,0.35);
    border-color: rgba(255,255,255,0.4);
}

/* Inputs / selects / multiselect */
[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #fff !important;
}
[data-baseweb="tag"] {
    background: rgba(124,156,255,0.35) !important;
    border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px 12px 0 0;
    color: rgba(255,255,255,0.7);
    padding: 10px 18px;
}
.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,0.12) !important;
    color: #fff !important;
}

/* Progress bar (win probability) */
.stProgress > div > div > div { background: linear-gradient(90deg, #7C9CFF, #C792EA); }

/* Disclaimer chip */
.disclaimer {
    background: rgba(255, 158, 128, 0.12);
    border: 1px solid rgba(255, 158, 128, 0.35);
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.8);
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA / MODELS
# ============================================================================

DATA_DIR = Path(__file__).parent / "data"

@st.cache_resource
def load_pickle(name):
    with open(DATA_DIR / name, "rb") as f:
        return pickle.load(f)

team_pipe = joblib.load(DATA_DIR / "pipe.pkl")                    # team-level model (compressed via joblib)
player_pipe = load_pickle("player_model_pipe.pkl")                # player-level model
player_features = load_pickle("player_model_features.pkl")        # feature column order
player_stats_lookup = load_pickle("player_stats_lookup.pkl")      # career stats for Stats tab
player_to_teams = load_pickle("player_to_teams.pkl")              # for Stats tab dropdown
team_to_players = load_pickle("team_to_players.pkl")              # active squads, Win Prob tab
player_current_form = load_pickle("player_current_form.pkl")      # rolling stats for live inference

# Canonical (deduped) team names used everywhere in the UI
ACTIVE_TEAMS = sorted(team_to_players.keys())

# The OLD team-level pipe.pkl still has duplicate/defunct categories baked into its
# encoder (never retrained). Map each canonical UI name -> the exact string that
# model expects, so predictions stay correct without retraining pipe.pkl.
TEAM_NAME_TO_PIPE_CATEGORY = {
    "Chennai Super Kings": "Chennai Super Kings",
    "Delhi Capitals": "Delhi Capitals",
    "Gujarat Titans": "Gujarat Titans",
    "Kolkata Knight Riders": "Kolkata Knight Riders",
    "Lucknow Super Giants": "Lucknow Super Giants",
    "Mumbai Indians": "Mumbai Indians",
    "Punjab Kings": "Punjab Kings",
    "Rajasthan Royals": "Rajasthan Royals",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad": "Sunrisers Hyderabad",
}

CITIES = sorted(team_pipe.named_steps["preprocessing"]
                 .named_transformers_["ohe"].categories_[2].tolist())

ROLE_LOOKUP = (
    player_stats_lookup.dropna(subset=["role"])
    .drop_duplicates("player")
    .set_index("player")["role"]
    .to_dict()
)

STAT_COLS = ["career_batting_avg", "career_strike_rate", "recent_strike_rate",
             "career_bowling_avg", "career_economy", "recent_economy"]

# ============================================================================
# HELPERS
# ============================================================================

def team_xi_features(players):
    """Average a list of players' current rolling stats + count all-rounders."""
    rows = player_current_form.reindex(players)
    avgs = {c: np.nanmean(rows[c].values) if rows[c].notna().any() else np.nan for c in STAT_COLS}
    n_all_rounders = sum(ROLE_LOOKUP.get(p) == "all-rounder" for p in players)
    return avgs, n_all_rounders


def predict_player_based(team_a_players, team_b_players, toss_winner_is_a):
    a_avgs, a_ar = team_xi_features(team_a_players)
    b_avgs, b_ar = team_xi_features(team_b_players)

    row = {"toss_winner_is_a": int(toss_winner_is_a)}
    stat_to_diffcol = {
        "career_batting_avg": "diff_avg_batting_avg",
        "career_strike_rate": "diff_avg_strike_rate",
        "recent_strike_rate": "diff_avg_recent_strike_rate",
        "career_bowling_avg": "diff_avg_bowling_avg",
        "career_economy": "diff_avg_economy",
        "recent_economy": "diff_avg_recent_economy",
    }
    for stat, col in stat_to_diffcol.items():
        diff = a_avgs[stat] - b_avgs[stat]
        row[col] = 0.0 if np.isnan(diff) else diff
    row["diff_n_all_rounders"] = a_ar - b_ar

    X = pd.DataFrame([row])[player_features]
    probs = player_pipe.predict_proba(X)[0]
    return probs[1], probs[0]  # (team_a_win_prob, team_b_win_prob)


def predict_team_based(team_a, team_b, city, target, current_score, overs, balls_in_over, wickets_down):
    balls_bowled = overs * 6 + balls_in_over
    runs_left = max(target - current_score, 0)
    balls_left = max(120 - balls_bowled, 0)
    wickets_left = max(10 - wickets_down, 0)
    crr = (current_score / (balls_bowled / 6)) if balls_bowled > 0 else 0.0
    rrr = (runs_left / (balls_left / 6)) if balls_left > 0 else runs_left * 6

    X = pd.DataFrame([{
        "batting_team": TEAM_NAME_TO_PIPE_CATEGORY[team_a],
        "bowling_team": TEAM_NAME_TO_PIPE_CATEGORY[team_b],
        "city": city,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "total_runs_x": target,
        "crr": crr,
        "rrr": rrr,
    }])
    probs = team_pipe.predict_proba(X)[0]
    return probs[1], probs[0]  # (batting_team_win_prob, bowling_team_win_prob)


def show_probability_result(name_a, prob_a, name_b, prob_b, calibration_note=False):
    c1, c2 = st.columns(2)
    with c1:
        st.metric(name_a, f"{prob_a*100:.1f}%")
        st.progress(min(max(prob_a, 0.0), 1.0))
    with c2:
        st.metric(name_b, f"{prob_b*100:.1f}%")
        st.progress(min(max(prob_b, 0.0), 1.0))
    if calibration_note:
        st.markdown(
            '<div class="disclaimer">⚠️ This model\'s probabilities run slightly '
            'overconfident at the extremes (validated Brier score: 0.196). Treat numbers '
            'above ~85% or below ~15% as directionally right, not exact.</div>',
            unsafe_allow_html=True,
        )

# ============================================================================
# HEADER
# ============================================================================

st.title("🏏 IPL Win Predictor")
st.caption("Pre-match and live win probability, plus player career stats — built on Cricsheet ball-by-ball data (2008–present).")

tab1, tab2 = st.tabs(["🎯 Win Probability", "📊 Player Stats"])

# ============================================================================
# TAB 1 -- WIN PROBABILITY
# ============================================================================

with tab1:
    mode = st.radio(
        "Prediction mode",
        ["Player-Based (pre-match, pick 11 vs 11)", "Team-Based (live match state)"],
        horizontal=True,
    )

    st.divider()

    if mode.startswith("Player-Based"):
        with st.container(border=True):
            st.subheader("Team A")
            team_a = st.selectbox("Team A", ACTIVE_TEAMS, key="pa_team_a")
            squad_a = team_to_players.get(team_a, [])
            players_a = st.multiselect(
                f"Select Team A's XI ({len(squad_a)} players available)",
                options=squad_a, max_selections=11, key="pa_players_a",
            )

        with st.container(border=True):
            st.subheader("Team B")
            remaining_teams = [t for t in ACTIVE_TEAMS if t != team_a]
            team_b = st.selectbox("Team B", remaining_teams, key="pa_team_b")
            squad_b = team_to_players.get(team_b, [])
            players_b = st.multiselect(
                f"Select Team B's XI ({len(squad_b)} players available)",
                options=squad_b, max_selections=11, key="pa_players_b",
            )

        with st.container(border=True):
            st.subheader("Toss")
            toss_winner = st.radio("Toss winner", [team_a, team_b], horizontal=True, key="pa_toss")

        if len(players_a) != 11 or len(players_b) != 11:
            st.info(f"Select exactly 11 players per team to predict. Currently: Team A = {len(players_a)}, Team B = {len(players_b)}.")
        else:
            if st.button("Predict Win Probability", key="pa_predict"):
                prob_a, prob_b = predict_player_based(players_a, players_b, toss_winner == team_a)
                show_probability_result(team_a, prob_a, team_b, prob_b, calibration_note=True)

    else:  # Team-Based (live match state)
        with st.container(border=True):
            st.subheader("Match Setup")
            col1, col2, col3 = st.columns(3)
            with col1:
                team_a = st.selectbox("Batting team (chasing)", ACTIVE_TEAMS, key="tb_team_a")
            with col2:
                remaining_teams = [t for t in ACTIVE_TEAMS if t != team_a]
                team_b = st.selectbox("Bowling team", remaining_teams, key="tb_team_b")
            with col3:
                city = st.selectbox("City", CITIES, key="tb_city")

        with st.container(border=True):
            st.subheader("Current Match State")
            col1, col2 = st.columns(2)
            with col1:
                target = st.number_input("Target score", min_value=1, value=180, key="tb_target")
                current_score = st.number_input("Current score", min_value=0, value=90, key="tb_current")
                wickets_down = st.number_input("Wickets down", min_value=0, max_value=10, value=3, key="tb_wickets")
            with col2:
                overs = st.number_input("Overs completed", min_value=0, max_value=19, value=10, key="tb_overs")
                balls_in_over = st.number_input("Balls into current over", min_value=0, max_value=5, value=0, key="tb_balls")

        if st.button("Predict Win Probability", key="tb_predict"):
            prob_a, prob_b = predict_team_based(team_a, team_b, city, target, current_score, overs, balls_in_over, wickets_down)
            show_probability_result(f"{team_a} (chasing)", prob_a, f"{team_b} (defending)", prob_b)

# ============================================================================
# TAB 2 -- PLAYER STATS
# ============================================================================

with tab2:
    with st.container(border=True):
        all_players = sorted(player_to_teams.keys())
        selected_player = st.selectbox("Search for a player", all_players, key="ps_player")

        teams_played_for = player_to_teams.get(selected_player, [])
        selected_team = st.selectbox("Team stint", teams_played_for, key="ps_team")

    row = player_stats_lookup[
        (player_stats_lookup["player"] == selected_player) &
        (player_stats_lookup["team"] == selected_team)
    ]

    if row.empty:
        st.warning("No stats found for this player/team combination.")
    else:
        r = row.iloc[0]
        st.subheader(f"{selected_player} — {selected_team}")
        st.caption(f"Role: {r.get('role', 'unknown')} · {int(r['matches']) if pd.notna(r.get('matches')) else '—'} matches")

        st.markdown("##### Batting")
        c1, c2, c3 = st.columns(3)
        c1.metric("Batting Average", f"{r['batting_average']:.1f}" if pd.notna(r.get("batting_average")) else "—")
        c2.metric("Strike Rate", f"{r['strike_rate']:.1f}" if pd.notna(r.get("strike_rate")) else "—")
        c3.metric("Highest Score", f"{int(r['highest_score'])}" if pd.notna(r.get("highest_score")) else "—")

        st.markdown("##### Bowling")
        c4, c5 = st.columns(2)
        c4.metric("Bowling Average", f"{r['bowling_average']:.1f}" if pd.notna(r.get("bowling_average")) else "—")
        c5.metric("Economy", f"{r['economy']:.2f}" if pd.notna(r.get("economy")) else "—")
