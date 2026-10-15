import streamlit as st
import cfbd
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

st.set_page_config(page_title="CFB Automated Spread Model", layout="wide")
st.title("🏈 Automated College Football Predictive Model")

# ==============================================================================
# 1. API INITIALIZATION
# ==============================================================================
# In Streamlit, your API key should be added securely via "Secrets" settings
api_key = st.sidebar.text_input("Enter CFBD API Key", type="password", value="")

if not api_key:
    st.warning("Please enter your CollegeFootballData API key in the sidebar to load the model.")
    st.stop()

configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = f"Bearer {api_key}"
api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))

YEAR = 2026
HOME_FIELD_ADVANTAGE = 2.5

# ==============================================================================
# 2. AUTOMATIC WEEK DETECTION & PIPELINE
# ==============================================================================
@st.cache_data(ttl=3600) # Caches data for 1 hour so the app loads instantly
def load_and_model_data():
    # Fetch 2026 Schedule
    all_games_2026 = api_instance.get_games(year=YEAR, season_type='regular')
    games_2026 = []
    for g in all_games_2026:
        games_2026.append({
            'week': g.week, 'home_team': g.home_team, 'away_team': g.away_team,
            'home_points': g.home_points, 'away_points': g.away_points,
            'margin': g.home_points - g.away_points if g.home_points is not None else None,
            'completed': g.home_points is not None
        })
    df_2026 = pd.DataFrame(games_2026)

    # Detect the current active week
    df_uncompleted = df_2026[df_2026['completed'] == False]
    current_wk = df_uncompleted['week'].min() if not df_uncompleted.empty else df_2026['week'].max()

    # Fetch 2025 data as baseline priors
    fallback_games = api_instance.get_games(year=2025, season_type='regular')
    games_2025 = []
    for g in fallback_games:
        if g.home_points is not None:
            games_2025.append({'home_team': g.home_team, 'away_team': g.away_team, 'margin': g.home_points - g.away_points})
    df_2025 = pd.DataFrame(games_2025)

    # Train on 2025 + weighted completed 2026 games
    df_2026_past = df_2026[df_2026['completed'] == True].copy()
    df_past = pd.concat([df_2025, df_2026_past, df_2026_past], ignore_index=True)

    # Ridge Regression modeling
    all_t = sorted(list(set(df_past['home_team'].unique()) | set(df_past['away_team'].unique()) | set(df_2026['home_team'].unique())))
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

    return df_2026, current_wk, power_ratings

with st.spinner("Analyzing matchup histories and training model..."):
    df_2026, current_week, power_ratings = load_and_model_data()

# ==============================================================================
# 3. INTERACTIVE MATCHUP UI
# ==============================================================================
st.header("🔮 Custom Matchup Predictor")
col1, col2 = st.columns(2)
with col1:
    away_select = st.selectbox("Select Away Team", options=power_ratings['Team'].unique(), index=0)
with col2:
    home_select = st.selectbox("Select Home Team", options=power_ratings['Team'].unique(), index=1)

if st.button("Generate Custom Spread"):
    h_rating = power_ratings.loc[power_ratings['Team'] == home_select, 'Rating'].values[0]
    a_rating = power_ratings.loc[power_ratings['Team'] == away_select, 'Rating'].values[0]
    margin = (h_rating - a_rating) + HOME_FIELD_ADVANTAGE

    if margin > 0:
        st.success(f"**Projected Winner:** {home_select} | **Model Line:** {home_select} -{round(margin, 1)}")
    else:
        st.success(f"**Projected Winner:** {away_select} | **Model Line:** {away_select} -{round(abs(margin), 1)}")

# ==============================================================================
# 4. FULL CURRENT WEEK SCHEDULE DUMP
# ==============================================================================
st.header(f"📅 Schedule & Spreads for Week {current_week}")

current_games = df_2026[(df_2026['completed'] == False) & (df_2026['week'] == current_week)]

if current_games.empty:
    st.info("No unplayed regular season games left for this week.")
else:
    predictions = []
    for row in current_games.itertuples():
        h_rating_arr = power_ratings.loc[power_ratings['Team'] == row.home_team, 'Rating'].values
        a_rating_arr = power_ratings.loc[power_ratings['Team'] == row.away_team, 'Rating'].values

        h_rating = h_rating_arr[0] if len(h_rating_arr) > 0 else 0
        a_rating = a_rating_arr[0] if len(a_rating_arr) > 0 else 0

        margin = (h_rating - a_rating) + HOME_FIELD_ADVANTAGE
        winner = row.home_team if margin > 0 else row.away_team
        spread = f" {row.home_team} -{round(margin, 1)}" if margin > 0 else f" {row.away_team} -{round(abs(margin), 1)}"

        predictions.append({
            "Away Team": row.away_team,
            "Home Team": row.home_team,
            "Projected Winner": winner,
            "Model Spread Line": spread
        })

    st.dataframe(pd.DataFrame(predictions), use_container_width=True)