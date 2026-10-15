import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

st.set_page_config(page_title="CFB Fully Automated Spread Model", layout="wide")
st.title("🏈 Fully Automated College Football Predictive Model")
st.write("This model automatically streams live game data to compute performance-based power ratings and projected week spreads.")

YEAR = 2026
HOME_FIELD_ADVANTAGE = 2.5

# ==============================================================================
# 1. LIVE AUTOMATED DATA PIPELINE & MODEL ENGINE
# ==============================================================================
@st.cache_data(ttl=1800) # Auto-refreshes data every 30 minutes
def load_live_cfb_engine():
    # Stable repository source link for real-time schedule and score assets
    url_data = "https://githubusercontent.com"
    
    try:
        df_all = pd.read_csv(url_data, low_memory=False)
    except Exception as e:
        st.error(f"❌ Error connecting to the live data server: {e}")
        st.stop()

    # Filter out current season tracking and prior season anchor data
    df_curr_raw = df_all[df_all['season'] == YEAR].copy()
    df_prior_raw = df_all[df_all['season'] == (YEAR - 1)].copy()

    # Process current season schedule mapping
    games_curr = []
    for row in df_curr_raw.itertuples():
        home_pts = getattr(row, 'home_score', None)
        away_pts = getattr(row, 'away_score', None)
        
        completed = pd.notna(home_pts) and pd.notna(away_pts)
        margin = (int(float(home_pts)) - int(float(away_pts))) if completed else None
        
        games_curr.append({
            'week': int(getattr(row, 'week', 1)), 
            'home_team': str(getattr(row, 'home_team', 'Unknown')), 
            'away_team': str(getattr(row, 'away_team', 'Unknown')),
            'margin': margin, 'completed': completed
        })
    df_2026_mapped = pd.DataFrame(games_curr)

    # Automatically identify the active calendar week right now
    df_uncompleted = df_2026_mapped[df_2026_mapped['completed'] == False]
    current_wk = int(df_uncompleted['week'].min()) if not df_uncompleted.empty else int(df_2026_mapped['week'].max())

    # Map prior year history to anchor early-season ratings
    games_prior = []
    for row in df_prior_raw.itertuples():
        home_pts = getattr(row, 'home_score', None)
        away_pts = getattr(row, 'away_score', None)
        if pd.notna(home_pts) and pd.notna(away_pts):
            games_prior.append({
                'home_team': str(getattr(row, 'home_team', 'Unknown')), 
                'away_team': str(getattr(row, 'away_team', 'Unknown')), 
                'margin': int(float(home_pts)) - int(float(away_pts))
            })
    df_prior_mapped = pd.DataFrame(games_prior)

    # Double-weight current season games to adapt quickly to recent form
    df_curr_past = df_2026_mapped[df_2026_mapped['completed'] == True].copy()
    df_past = pd.concat([df_prior_mapped, df_curr_past, df_curr_past], ignore_index=True)

    # Generate distinct master list of all valid FBS teams
    all_t = sorted(list(set(df_past['home_team'].unique()) | set(df_past['away_team'].unique()) | set(df_2026_mapped['home_team'].unique())))
    team_to_idx = {team: i for i, team in enumerate(all_t)}

    # Build the design matrix for linear equations
    X = np.zeros((len(df_past), len(all_t)))
    Y = np.zeros(len(df_past))

    for idx, row in enumerate(df_past.itertuples()):
        X[idx, team_to_idx[row.home_team]] = 1
        X[idx, team_to_idx[row.away_team]] = -1
        Y[idx] = row.margin - HOME_FIELD_ADVANTAGE

    # Fit Ridge Regression to calculate dynamic mathematical power ratings
    model = Ridge(alpha=2.0, fit_intercept=False)
    model.fit(X, Y)

    power_ratings = pd.DataFrame({'Team': all_t, 'Rating': model.coef_})
    power_ratings['Rating'] = power_ratings['Rating'] - power_ratings['Rating'].mean()
    
    return df_2026_mapped, current_wk, power_ratings

with st.spinner("Streaming live football matrix statistics and training model..."):
    df_2026, current_week, power_ratings = load_live_cfb_engine()

# Clean up dropdown team sorting
all_available_teams = sorted(power_ratings['Team'].unique())

# ==============================================================================
# 2. INTERACTIVE MATCHUP UI
# ==============================================================================
st.header("🔮 Custom Matchup Predictor")
col1, col2 = st.columns(2)
with col1:
    away_select = st.selectbox("Select Away Team", options=all_available_teams, index=all_available_teams.index("Michigan") if "Michigan" in all_available_teams else 0)
with col2:
    home_select = st.selectbox("Select Home Team", options=all_available_teams, index=all_available_teams.index("Ohio State") if "Ohio State" in all_available_teams else 0)

if st.button("Generate Custom Spread"):
    h_rating_arr = power_ratings.loc[power_ratings['Team'] == home_select, 'Rating'].values
    a_rating_arr = power_ratings.loc[power_ratings['Team'] == away_select, 'Rating'].values
    
    h_rating = float(h_rating_arr[0]) if len(h_rating_arr) > 0 else 0.0
    away_rating = float(a_rating_arr[0]) if len(a_rating_arr) > 0 else 0.0
    
    margin = (h_rating - away_rating) + HOME_FIELD_ADVANTAGE
    
    if margin > 0:
        st.success(f"**Projected Winner:** {home_select} | **Model Line:** {home_select} -{round(margin, 1)}")
    else:
        st.success(f"**Projected Winner:** {away_select} | **Model Line:** {away_select} -{round(abs(margin), 1)}")

# ==============================================================================
# 3. AUTOMATED CURRENT WEEK SCHEDULE VIEW
# ==============================================================================
st.header(f"📅 Schedule & Predicted Spreads for Week {current_week}")
current_games = df_2026[(df_2026['completed'] == False) & (df_2026['week'] == current_week)]

if current_games.empty:
    st.info("No unplayed regular season games left for this week's active window.")
else:
    predictions = []
    for row in current_games.itertuples():
        h_rating_arr = power_ratings.loc[power_ratings['Team'] == row.home_team, 'Rating'].values
        a_rating_arr = power_ratings.loc[power_ratings['Team'] == row.away_team, 'Rating'].values
        
        h_rating = float(h_rating_arr[0]) if len(h_rating_arr) > 0 else 0.0
        a_rating = float(a_rating_arr[0]) if len(a_rating_arr) > 0 else 0.0
        
        margin = (h_rating - a_rating) + HOME_FIELD_ADVANTAGE
        winner = row.home_team if margin > 0 else row.away_team
        spread = f"{row.home_team} -{round(margin, 1)}" if margin > 0 else f"{row.away_team} -{round(abs(margin), 1)}"
        
        predictions.append({
            "Away Team": row.away_team, 
            "Home Team": row.home_team, 
            "Projected Winner": winner, 
            "Model Spread Line": spread
        })
        
    st.dataframe(pd.DataFrame(predictions), use_container_width=True)
