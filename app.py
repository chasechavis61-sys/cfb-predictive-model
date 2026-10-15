import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

st.set_page_config(page_title="CFB Live Automated Spread Model", layout="wide")
st.title("🏈 Keyless Automated College Football Predictive Model")

YEAR = 2026
HOME_FIELD_ADVANTAGE = 2.5

# ==============================================================================
# 1. LIVE DATA PIPELINE (Bypassing API Key Restrictions)
# ==============================================================================
@st.cache_data(ttl=1800) # Refreshes live data every 30 minutes
def load_mirror_data():
    # Stream live 2026 and historical 2025 data directly from open data storage
    url_2026 = f"https://githubusercontent.com"
    url_2025 = f"https://githubusercontent.com"
    
    try:
        df_curr = pd.read_csv(url_2026)
        df_prior = pd.read_csv(url_2025)
    except Exception as e:
        st.error("Could not stream live schedules. The data mirror might be down.")
        st.stop()
        
    # Map current 2026 games
    games_curr = []
    for row in df_curr.itertuples():
        # Handle variations in open-source mirror column names safely
        home_pts = getattr(row, 'home_points', None) if pd.notna(getattr(row, 'home_points', None)) else getattr(row, 'home_score', None)
        away_pts = getattr(row, 'away_points', None) if pd.notna(getattr(row, 'away_points', None)) else getattr(row, 'away_score', None)
        
        completed = pd.notna(home_pts) and pd.notna(away_pts)
        margin = (int(home_pts) - int(away_pts)) if completed else None
        
        games_curr.append({
            'week': int(row.week), 'home_team': str(row.home_team), 'away_team': str(row.away_team),
            'margin': margin, 'completed': completed
        })
df_2026_mapped = pd.DataFrame(games_curr)

# Automatically identify the active week we are playing right now
df_uncompleted = df_2026_mapped[df_2026_mapped['completed'] == False]
current_wk = int(df_uncompleted['week'].min()) if not df_uncompleted.empty else int(df_2026_mapped['week'].max())

# Map 2025 history to anchor power ratings
games_prior = []
for row in df_prior.itertuples():
    home_pts = getattr(row, 'home_points', None) if pd.notna(getattr(row, 'home_points', None)) else getattr(row, 'home_score', None)
    away_pts = getattr(row, 'away_points', None) if pd.notna(getattr(row, 'away_points', None)) else getattr(row, 'away_score', None)
    if pd.notna(home_pts) and pd.notna(away_pts):
        games_prior.append({'home_team': str(row.home_team), 'away_team': str(row.away_team), 'margin': int(home_pts) - int(away_pts)})
df_prior_mapped = pd.DataFrame(games_prior)

# Combine past datasets (weighting current season games higher)
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

with st.spinner("Streaming live mirror stats and rendering analytics model..."):
df_2026, current_week, power_ratings = load_mirror_data()

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
    st.success(f"**Projected Winner:** {home_select} | **Model Line:** {home_select} -{round(margin[0], 1)}")
else:
    st.success(f"**Projected Winner:** {away_select} | **Model Line:** {away_select} -{round(abs(margin[0]), 1)}")

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
        
        h_rating = h_rating_arr[0] if len(h_rating_arr) > 0 else 0
        a_rating = a_rating_arr[0] if len(a_rating_arr) > 0 else 0
        
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
