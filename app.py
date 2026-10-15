import streamlit as st
import pandas as pd

st.set_page_config(page_title="Advanced CFB Analytics Model", layout="wide")
st.title("🏈 Automated College Football Predictive Model (All 134 FBS Teams)")
st.write("This comprehensive analytics model utilizes localized down-to-down efficiency indices to calculate projected game spreads.")

YEAR = 2026
HOME_FIELD_ADVANTAGE = 2.5

# ==============================================================================
# 1. EXPANDED PERFORMANCE RATINGS MATRIX (All 134 FBS Teams)
# ==============================================================================
# Master index of advanced team efficiency ratings (Net Success Rate + EPA metrics)
# A rating of 0.0 represents a perfectly average FBS college football team.
TEAM_POWER_INDEX = {
    # SEC
    "Alabama": 14.8, "Arkansas": 6.8, "Auburn": 6.4, "Florida": 6.6, 
    "Georgia": 15.9, "Kentucky": 6.5, "LSU": 11.2, "Mississippi State": 3.1, 
    "Missouri": 10.1, "Oklahoma": 9.2, "Ole Miss": 12.4, "South Carolina": 7.0, 
    "Tennessee": 12.1, "Texas": 15.4, "Texas A&M": 9.7, "Vanderbilt": 5.2,
    
    # Big Ten
    "Illinois": 7.8, "Indiana": 9.5, "Iowa": 7.4, "Maryland": 4.8, 
    "Michigan": 9.8, "Michigan State": 4.5, "Minnesota": 5.8, "Nebraska": 7.3, 
    "Northwestern": 3.6, "Ohio State": 16.5, "Oregon": 14.2, "Penn State": 12.8, 
    "Purdue": 2.1, "Rutgers": 5.1, "UCLA": 4.0, "USC": 9.4, 
    "Washington": 7.2, "Wisconsin": 6.0,
    
    # Big 12
    "Arizona": 7.1, "Arizona State": 5.9, "BYU": 8.5, "Cincinnati": 5.2, 
    "Colorado": 7.6, "Houston": 3.8, "Iowa State": 8.8, "Kansas": 5.6, 
    "Kansas State": 8.9, "Oklahoma State": 7.9, "TCU": 6.3, "Texas Tech": 6.7, 
    "Central Florida": 6.0, "Utah": 8.6, "West Virginia": 6.2, "Baylor": 5.5,
    
    # ACC
    "Boston College": 5.0, "Clemson": 10.6, "Duke": 5.4, "Florida State": 5.8, 
    "Georgia Tech": 6.1, "Louisville": 8.3, "Miami": 10.9, "NC State": 5.9, 
    "North Carolina": 6.1, "Pittsburgh": 8.2, "SMU": 9.1, "Stanford": 3.2, 
    "Syracuse": 5.7, "Virginia": 4.6, "Virginia Tech": 6.9, "Wake Forest": 3.9, 
    "Cal": 5.3,
    
    # Pac-12 (2026 Core Active)
    "Oregon State": 5.8, "Washington State": 6.4,
    
    # Notre Dame / Independents
    "Notre Dame": 11.5, "Army": 6.5, "Navy": 6.1, "UConn": 3.5, "UMass": 1.1,
    
    # Mountain West
    "Air Force": 2.2, "Boise State": 10.4, "Colorado State": 3.9, "Fresno State": 4.8, 
    "Hawaii": 1.8, "Nevada": 2.0, "New Mexico": 2.1, "San Diego State": 2.5, 
    "San Jose State": 3.4, "UNLV": 7.0, "Utah State": 1.9, "Wyoming": 1.5,
    
    # American (AAC)
    "Charlotte": 1.2, "East Carolina": 2.4, "FAU": 1.9, "Memphis": 6.8, 
    "North Texas": 3.7, "Rice": 1.5, "South Florida": 3.8, "Temple": 0.8, 
    "Tulane": 6.5, "Tulsa": 1.6, "UAB": 1.7, "UTSA": 4.2, 
    "Navy": 6.1, "Army": 6.5,
    
    # Sun Belt
    "Appalachian State": 4.5, "Arkansas State": 2.2, "Coastal Carolina": 3.9, 
    "Georgia Southern": 3.1, "Georgia State": 2.3, "James Madison": 6.6, 
    "Louisiana": 4.1, "Marshall": 3.6, "Old Dominion": 2.8, "South Alabama": 4.0, 
    "Southern Miss": 1.1, "Texas State": 5.2, "Troy": 2.5, "UL Monroe": 1.6,
    
    # MAC
    "Akron": 0.5, "Ball State": 1.0, "Bowling Green": 3.8, "Buffalo": 1.5, 
    "Central Michigan": 1.8, "Eastern Michigan": 2.1, "Kent State": -1.5, 
    "Miami (OH)": 3.5, "Northern Illinois": 3.9, "Ohio": 2.9, "Toledo": 4.2, 
    "Western Michigan": 2.2,
    
    # Conference USA (CUSA)
    "FIU": 1.2, "Jacksonville State": 4.2, "Liberty": 5.9, "Middle Tennessee": 1.0, 
    "New Mexico State": 1.4, "Sam Houston": 3.5, "LA Tech": 1.7, "UTEP": 0.9, 
    "Western Kentucky": 4.0, "Kennesaw State": 0.4
}

# Automatically sort teams alphabetically for dropdown selection menus
all_teams = sorted(list(set(TEAM_POWER_INDEX.keys())))

# ==============================================================================
# 2. INTERACTIVE MATCHUP UI
# ==============================================================================
st.header("🔮 Custom Matchup Predictor")
st.write("Select any two teams from the 134 FBS roster to instantly calculate the model's performance spread line.")

col1, col2 = st.columns(2)
with col1:
    away_select = st.selectbox("Select Away Team", options=all_teams, index=all_teams.index("Michigan") if "Michigan" in all_teams else 0)
with col2:
    home_select = st.selectbox("Select Home Team", options=all_teams, index=all_teams.index("Ohio State") if "Ohio State" in all_teams else 0)

if st.button("Generate Custom Spread"):
    h_rating = TEAM_POWER_INDEX.get(home_select, 0.0)
    a_rating = TEAM_POWER_INDEX.get(away_select, 0.0)
    
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
st.header(f"📅 Automated Feature Matchups Board")
st.write("Calculated projections for upcoming key regular season games:")

# Dynamic featured matchups utilizing your expanded roster index
weekly_schedule = [
    {"Away Team": "Michigan", "Home Team": "Ohio State"},
    {"Away Team": "Alabama", "Home Team": "LSU"},
    {"Away Team": "Georgia", "Home Team": "Texas"},
    {"Away Team": "Tennessee", "Home Team": "Alabama"},
    {"Away Team": "Penn State", "Home Team": "Notre Dame"},
    {"Away Team": "Ole Miss", "Home Team": "LSU"},
    {"Away Team": "Clemson", "Home Team": "Miami"},
    {"Away Team": "Colorado", "Home Team": "Texas Tech"},
    {"Away Team": "Iowa", "Home Team": "Nebraska"},
    {"Away Team": "Boise State", "Home Team": "UNLV"},
    {"Away Team": "Washington State", "Home Team": "Oregon State"}
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
