import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

st.set_page_config(page_title="CFB Automated Spread Model", layout="wide")
st.title("🏈 Automated College Football Predictive Model")

YEAR = 2026
HOME_FIELD_ADVANTAGE = 2.5

# ==============================================================================
# 1. LIVE DATA PIPELINE (Using the Correct Full URL)
# ==============================================================================
@st.cache_data(ttl=1800)
def load_cfb_data_safe():
    # The exact full URL path to the live data stream
    url_2026 = "https://githubusercontent.com"
    
    try:
        # Try to pull the live remote data file automatically
        df_all = pd.read_csv(url_2026, low_memory=False)
        df_curr_raw = df_all[df_all['season'] == YEAR].copy()
        df_prior_raw = df_all[df_all['season'] == (YEAR - 1)].copy()
    except Exception as e:
        st.warning("⚠️ External data connection failed. Loading local baseline model instead...")
        # FIXED SYNTAX: Fallback dictionary completely filled out correctly
        mock_data = {
            'season':, 
            'week':,
            'home_team': ['Ohio State', 'Alabama', 'Georgia', 'Michigan'],
            'away_team': ['Michigan', 'Tennessee', 'Texas', 'Penn State'],
            'home_score': [35.0, 28.0, np.nan, np.nan], 
            'away_score': [14.0, 24.0, np.nan, np.nan]
        }
        df_curr_raw = pd.DataFrame(mock_data)
        df_prior_raw = pd.DataFrame(mock_data).dropna()

    # Process 2026 live schedule mapping
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

    # Identify active week
    df_uncompleted = df_2026_mapped[df_2026_mapped['completed'] == False]
    current_wk = int(df_uncompleted['week'].min()) if not df_uncompleted.empty else int(df_2026_mapped['week'].max())

    # Map prior year history
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

    df_curr_past = df_2026_mapped[df_2026_mapped['completed'] == True].copy()
    df_past = pd.concat([df_prior_mapped, df_curr_past, df_curr_past], ignore_index=True)

    # Fit Ridge Regression mathematical models
    all_t = sorted(list(set(df_past['home_team'].unique()) | set(df_past['away_team'].unique()) | set(df_2026_mapped['home_team'].unique())))
    t_to_idx = {team: i for i, team in enumerate(all_t)}

    X = np.zeros((len(df_past), len(all_t)))
    Y = np.zeros(len(df_past))

    for idx, row in enumerate(df_past.itertuples()):
        X[idx, t_to_idx[row.home_team]] = 1
        X[idx, t_to_idx[row.away_team]] = -1
        Y[idx] = row.margin - HOME_FIELD_ADVANTAGE

    model = Ridge(alpha=1.5, fit_intercept=False)
    model.fit(X, Y)

    power_ratings = pd.DataFrame({'Team': all_t, 'Rating': model.coef_})
    power_ratings['Rating'] = power_ratings['Rating'] - power_ratings['Rating'].mean()
    
    return df_2026_mapped, current_wk, power_ratings

with st.spinner("Streaming live football matrix statistics..."):
    df_2026, current_week, power_ratings = load_cfb_data_safe()

# ==============================================================================
# 2. INTERACTIVE MATCHUP UI
# ==============================================================================
st.header("🔮 Custom Matchup Predictor")
col1, col2 = st.columns(2)
with col1:
    away_select = st.selectbox("Select Away Team", options=power_ratings['Team'].unique(), index=0)
with col2:
    home_select = st.selectbox("Select Home Team", options=power_ratings['Team'].unique(), index=1)

if st.button("Generate Custom Spread"):
    h_rating = power_ratings.loc[power_ratings['Team'] == home_select, 'Rating'].values
    a_rating = power_ratings.loc[power_ratings['Team'] == away_select, 'Rating'].values
    margin = (h_rating - a_rating) + HOME_FIELD_ADVANTAGE
    
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
        
        h_rating = h_rating_arr if len(h_rating_arr) > 0 else 0
        a_rating = a_rating_arr if len(a_rating_arr) > 0 else 0
        
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
