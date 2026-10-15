import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

st.set_page_config(page_title="CFB Automated Spread Model", layout="wide")
st.title("🏈 Keyless Automated College Football Predictive Model")

YEAR = 2025
HOME_FIELD_ADVANTAGE = 2.5

# ==============================================================================
# 1. API-FREE DATA PIPELINE (Public GitHub Mirror)
# ==============================================================================
@st.cache_data(ttl=3600)
def load_mirror_data():
# Load 2025 games directly from the public cfbfastR parquet/csv mirror
url_2025 = f"https://githubusercontent.com_{YEAR}.parquet"
url_2024 = f"https://githubusercontent.com_{YEAR-1}.parquet"

try:
df_curr = pd.read_parquet(url_2025)
df_prior = pd.read_parquet(url_2024)
except Exception:
# Fallback to CSV variants if parquet mirror shifts structures
url_2025_csv = f"https://githubusercontent.com_{YEAR}.csv"
url_2024_csv = f"https://githubusercontent.com_{YEAR-1}.csv"
df_curr = pd.read_csv(url_2025_csv)
df_prior = pd.read_csv(url_2024_csv)

# Standardize data framing
games_curr = []
for row in df_curr.itertuples():
# Handle varying mirror column naming strategies
home_pts = getattr(row, 'home_points', None) if pd.notna(getattr(row, 'home_points', None)) else getattr(row, 'home_score', None)
away_pts = getattr(row, 'away_points', None) if pd.notna(getattr(row, 'away_points', None)) else getattr(row, 'away_score', None)

completed = pd.notna(home_pts) and pd.notna(away_pts)
margin = (home_pts - away_pts) if completed else None

games_curr.append({
'week': row.week, 'home_team': row.home_team, 'away_team': row.away_team,
'margin': margin, 'completed': completed
})
df_2026_mapped = pd.DataFrame(games_curr)

# Detect current target slate week
df_uncompleted = df_2026_mapped[df_2026_mapped['completed'] == False]
current_wk = df_uncompleted['week'].min() if not df_uncompleted.empty else df_2026_mapped['week'].max()

games_prior = []
for row in df_prior.itertuples():
home_pts = getattr(row, 'home_points', None) if pd.notna(getattr(row, 'home_points', None)) else getattr(row, 'home_score', None)
away_pts = getattr(row, 'away_points', None) if pd.notna(getattr(row, 'away_score', None)) else getattr(row, 'away_score', None)
if pd.notna(home_pts) and pd.notna(away_pts):
games_prior.append({'home_team': row.home_team, 'away_team': row.away_team, 'margin': home_pts - away_pts})
df_prior_mapped = pd.DataFrame(games_prior)

df_curr_past = df_2026_mapped[df_2026_mapped['completed'] == True].copy()
df_past = pd.concat([df_prior_mapped, df_curr_past, df_curr_past], ignore_index=True)

# Core Mathematical Linear Regression Model setup
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
st.success(f"**Projected Winner:** {home_select} | **Model Line:** {home_select} -{round(margin, 1)}")
else:
st.success(f"**Projected Winner:** {away_select} | **Model Line:** {away_select} -{round(abs(margin), 1)}")

# ==============================================================================
# 3. AUTOMATED SCHEDULE VIEW
# ==============================================================================
st.header(f"📅 Schedule & Spreads for Week {current_week}")
current_games = df_2026[(df_2026['completed'] == False) & (df_2026['week'] == current_week)]

if current_games.empty:
st.info("No unplayed regular season games left for this active slate window.")
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

predictions.append({"Away Team": row.away_team, "Home Team": row.home_team, "Projected Winner": winner, "Model Spread Line": spread})

st.dataframe(pd.DataFrame(predictions), use_container_width=True)
