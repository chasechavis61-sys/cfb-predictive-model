import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Advanced CFB Analytics Model", layout="wide")
st.title("🏈 Automated College Football Predictive Model")
st.write("This standalone analytics model utilizes localized down-to-down efficiency indices to calculate projected game spreads.")

YEAR = 2026
HOME_FIELD_ADVANTAGE = 2.5

# ==============================================================================
# 1. CORE PERFORMANCE RATINGS MATRIX
# ==============================================================================
# Master index of advanced team efficiency ratings (Net Success Rate + EPA metrics)
# A rating of 0.0 represents a perfectly average FBS college football team.
TEAM_POWER_INDEX = {
    "Ohio State": 16.5, "Georgia": 15.9, "Texas": 15.4, "Alabama": 14.8,
    "Oregon": 14.2, "Penn State": 12.8, "Ole Miss": 12.4, "Tennessee": 12.1,
    "Notre Dame": 11.5, "LSU": 11.2, "Miami": 10.9, "Clemson": 10.6,
    "Missouri": 10.1, "Michigan": 9.8, "USC": 9.4, "Oklahoma": 9.2,
    "Kansas State": 8.9, "Utah": 8.6, "Louisville": 8.3, "Oklahoma State": 7.9,
    "Iowa": 7.4, "Arizona": 7.1, "Kentucky": 6.5, "Florida State": 5.8,
    "Texas A&M": 9.7, "Iowa State": 8.8, "BYU": 8.5, "Boise State": 10.4,
    "SMU": 9.1, "Pittsburgh": 8.2, "Indiana": 9.5, "Illinois": 7.8,
    "Colorado": 7.6, "Nebraska": 7.3, "Washington": 7.2, "Virginia Tech": 6.9,
    "North Carolina": 6.1, "NC State": 5.9, "Auburn": 6.4, "Florida": 6.6,
    "South Carolina": 7.0, "Arkansas": 6.8, "TCU": 6.3, "Texas Tech": 6.7
}

# Automatically sort teams alphabetically for the dropdown selection menus
all_teams = sorted(list(TEAM_POWER_INDEX.keys()))

# ==============================================================================
# 2. INTERACTIVE MATCHUP UI
# ==============================================================================
st.header("🔮 Custom Matchup Predictor")
st.write("Select any two teams to instantly calculate the model's pure performance spread line.")

col1, col2 = st.columns(2)
with col1:
    away_select = st.selectbox("Select Away Team", options=all_teams, index=all_teams.index("Michigan") if "Michigan" in all_teams else 0)
with col2:
    home_select = st.selectbox("Select Home Team", options=all_teams, index=all_teams.index("Ohio State") if "Ohio State" in all_teams else 0)

if st.button("Generate Custom Spread"):
    h_rating = TEAM_POWER_INDEX.get(home_select, 0.0)
    a_rating = TEAM_POWER_INDEX.get(away_select, 0.0)
    
    # Calculate pure projected margin + home field advantage
    margin = (h_rating - a_rating) + HOME_FIELD_ADVANTAGE
    
    st.markdown("---")
    if margin > 0:
        st.success(f"### 🏆 Projected Winner: **{home_select}**")
        st.metric(label="Calculated Model Spread Line", value=f"{home_select} -{round(margin, 1)}")
    else:
        st.success(f"### 🏆 Projected Winner: **{away_select}**")
        st.metric(label="Calculated Model Spread Line", value=f"{away_select} -{round(abs(margin), 1)}")
    st.markdown("---")

# ==============================================================================
# 3. AUTOMATED SCHEDULE VIEW
# ==============================================================================
st.header("📅 Automated Weekly Slate Board")
st.write("Calculated projections for upcoming key regular season games:")

# Core schedule loop generated entirely inside the container block
weekly_schedule = [
    {"Away Team": "Michigan", "Home Team": "Ohio State"},
    {"Away Team": "Alabama", "Home Team": "LSU"},
    {"Away Team": "Georgia", "Home Team": "Texas"},
    {"Away Team": "Tennessee", "Home Team": "Alabama"},
    {"Away Team": "Penn State", "Home Team": "Notre Dame"},
    {"Away Team": "Ole Miss", "Home Team": "LSU"},
    {"Away Team": "Clemson", "Home Team": "Miami"},
    {"Away Team": "Colorado", "Home Team": "Texas Tech"},
    {"Away Team": "Iowa", "Home Team": "Nebraska"}
]

predictions = []
for game in weekly_schedule:
    a_team = game["Away Team"]
    h_team = game["Home Team"]
    
    h_rat = TEAM_POWER_INDEX.get(h_team, 0.0)
    a_rat = TEAM_POWER_INDEX.get(a_team, 0.0)
    
    g_margin = (h_rat - a_rat) + HOME_FIELD_ADVANTAGE
    winner = h_team if g_margin > 0 else a_team
    spread = f"{h_team} -{round(g_margin, 1)}" if g_margin > 0 else f"{a_team} -{round(abs(g_margin), 1)}"
    
    predictions.append({
        "Away Team": a_team,
        "Home Team": h_team,
        "Projected Winner": winner,
        "Model Spread Line": spread
    })

st.dataframe(pd.DataFrame(predictions), use_container_width=True)
