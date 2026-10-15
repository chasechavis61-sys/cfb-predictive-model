import streamlit as st
import cfbd
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

st.set_page_config(page_title="CFB Automated Spread Model", layout="wide")
st.title("🏈 Automated College Football Predictive Model")

# ==============================================================================
# 1. API INITIALIZATION (The 401 Fix)
# ==============================================================================
# Paste your raw alphanumeric key string between the quotes below.
# Do NOT include the word "Bearer " inside these quotes.
RAW_API_KEY = "Lx5SW7fjPsgpYp84gv245NpLzpUAW3ZuJVttG785YZKkqI3wDqgWUpqCqnY0YxXN"

# Clean out any accidental whitespace
clean_key = RAW_API_KEY.replace("Bearer ", "").strip()

# THE EXACT SDK FIX: Pass your key directly as the access_token parameter
configuration = cfbd.Configuration(access_token=clean_key)
api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))

YEAR = 2025
HOME_FIELD_ADVANTAGE = 2.5

# ==============================================================================
# 2. AUTOMATIC DATA PIPELINE
# ==============================================================================
@st.cache_data(ttl=3600)
def load_and_model_data():
    all_games_curr = api_instance.get_games(year=YEAR, season_type='regular')
    games_curr = []
    for g in all_games_curr:
        games_curr.append({
            'week': g.week, 'home_team': g.home_team, 'away_team': g.away_team,
            'home_points': g.home_points, 'away_points': g.away_points,
            'margin': g.home_points - g.away_points if g.home_points is not None else None,
            'completed': g.home_points is not None
        })
    df_curr = pd.DataFrame(games_curr)

    df_uncompleted = df_curr[df_curr['completed'] == False]
    current_wk = df_uncompleted['week'].min() if not df_uncompleted.empty else df_curr['week'].max()

    fallback_games = api_instance.get_games(year=YEAR-1, season_type='regular')
    games_prior = []
    for g in fallback_games:
        if g.home_points is not None:
            games_prior.append({'home_team': g.home_team, 'away_team': g.away_team, 'margin': g.home_points - g.away_points})
    df_prior = pd.DataFrame(games_prior)

    df_curr_past = df_curr[df_curr['completed'] == True].copy()
    df_past = pd.concat([df_prior, df_curr_past, df_curr_past], ignore_index=True)

    all_t = sorted(list(set(df_past['home_team'].unique()) | set(df_past['away_team'].unique()) | set(df_curr['home_team'].unique())))
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

    return df_curr, current_wk, power_ratings

with st.spinner("Analyzing matchup histories and training model..."):
    df_curr, current_week, power_ratings = load_and_model_data()

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
    h_rating = power_ratings.loc[power_ratings['Team'] == home_select, 'Rating'].values
    a_rating = power_ratings.loc[power_ratings['Team'] == away_select, 'Rating'].values
    margin = (h_rating - a_rating) + HOME_FIELD_ADVANTAGE

    if margin > 0:
        st.success(f"**Projected Winner:** {home_select} | **Model Line:** {home_select} -{round(margin[0], 1)}")
    else:
        st.success(f"**Projected Winner:** {away_select} | **Model Line:** {away_select} -{round(abs(margin[0]), 1)}")

# ==============================================================================
# 4. FULL WEEK SCHEDULE VIEW
# ==============================================================================
st.header(f"📅 Schedule & Spreads for Week {current_week}")

# Initialize the lines API client
lines_api = cfbd.LinesApi(cfbd.ApiClient(configuration))

current_games = df_curr[(df_curr['completed'] == False) & (df_curr['week'] == current_week)]

if current_games.empty:
    st.info("No unplayed regular season games left for this week.")
else:
    # Fetch all Vegas lines for the current week in one call
    try:
        vegas_data = lines_api.get_lines(year=YEAR, week=current_week, season_type='regular')
        vegas_dict = {g.id: g.lines[0].text for g in vegas_data if g.lines}
    except Exception:
        vegas_dict = {}

    predictions = []
    for row in current_games.itertuples():
        h_rating_arr = power_ratings.loc[power_ratings['Team'] == row.home_team, 'Rating'].values
        a_rating_arr = power_ratings.loc[power_ratings['Team'] == row.away_team, 'Rating'].values

        h_rating = h_rating_arr if len(h_rating_arr) > 0 else 0
        a_rating = a_rating_arr if len(a_rating_arr) > 0 else 0

        margin = (h_rating - a_rating) + HOME_FIELD_ADVANTAGE
        winner = row.home_team if margin > 0 else row.away_team
        spread = f"{row.home_team} -{round(margin, 1)}" if margin > 0 else f"{row.away_team} -{round(abs(margin), 1)}"

        # Grab Vegas line if available, otherwise mark as N/A
        vegas_line = vegas_dict.get(row.id, "N/A")

        predictions.append({
            "Away Team": row.away_team,
            "Home Team": row.home_team,
            "Projected Winner": winner,
            "Your Model Line": spread,
            "Vegas Consensus Line": vegas_line
        })

    st.dataframe(pd.DataFrame(predictions), use_container_width=True)
