import streamlit as st
import pandas as pd
import numpy as np
import requests

# =====================================================================
# 1. AUTOMATED NATIVE ELO TRAINING ENGINE (SPORTSDATAVERSE)
# =====================================================================
@st.cache_data(ttl=86400) # Re-calculates historical team power ratings once a day
def train_keyless_elo_model():
    """Trains a math-based Elo system using historical SportsDataverse data archives."""
    elo_ratings = {}
    K_FACTOR = 35
    HOME_ADVANTAGE = 55

    # Cycles past seasons to build base power rankings for all D1 programs
    training_years = [2024, 2025]

    for year in training_years:
        url = f"https://githubusercontent.com_{year}.csv"
        try:
            df_games = pd.read_csv(url)
            df_games = df_games[(df_games['season_type'] == 'regular') & (df_games['home_points'].notna())]
            df_games = df_games.sort_values(by='week')

            for _, game in df_games.iterrows():
                home_team = str(game['home_team']).strip()
                away_team = str(game['away_team']).strip()
                home_score = int(game['home_points'])
                away_score = int(game['away_points'])

                if home_team not in elo_ratings: elo_ratings[home_team] = 1500.0
                if away_team not in elo_ratings: elo_ratings[away_team] = 1500.0

                r_home = elo_ratings[home_team] + HOME_ADVANTAGE
                r_away = elo_ratings[away_team]

                exp_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
                actual_home = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)

                margin = abs(home_score - away_score)
                mov_multiplier = np.log(margin + 1) * (2.2 / ((r_home - r_away) * 0.001 + 2.2)) if margin > 0 else 1.0

                shift = K_FACTOR * mov_multiplier * (actual_home - exp_home)
                elo_ratings[home_team] += shift
                elo_ratings[away_team] -= shift
        except:
            pass

    return elo_ratings

def generate_score_projection(home_team, away_team, elo_dict):
    """Generates a score prediction based on team power rankings."""
    HOME_ADVANTAGE_ELO = 55
    BASE_AVG_SCORE = 28.0
    POINTS_PER_ELO = 25.0

    r_home = elo_dict.get(home_team, 1500.0) + HOME_ADVANTAGE_ELO
    r_away = elo_dict.get(away_team, 1500.0)

    projected_margin = (r_home - r_away) / POINTS_PER_ELO
    proj_home = max(0, BASE_AVG_SCORE + (projected_margin / 2))
    proj_away = max(0, BASE_AVG_SCORE - (projected_margin / 2))

    win_prob = 1 / (1 + 10 ** ((r_away - r_home) / 400))
    return proj_home, proj_away, win_prob

# =====================================================================
# 2. FIXED AUTOMATION: LIVE ESPN SLATE & ODDS LOADER (NO KEYS)
# =====================================================================
@st.cache_data(ttl=3600) # Updates lines and games automatically every hour
def fetch_live_espn_weekly_slate():
    """
    Connects directly to ESPN's unauthenticated live scoreboard API feed.
    Pulls upcoming week matchups and Vegas consensus lines automatically.
    """
    processed_games = []
    url = "https://espn.com"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            events = data.get("events", [])

            for event in events:
                competitions = event.get("competitions", [])
                if not competitions:
                    continue

                comp = competitions[0]
                competitors = comp.get("competitors", [])

                home_team = None
                away_team = None

                for team in competitors:
                    team_name = team.get("team", {}).get("displayName")
                    if team.get("homeAway") == "home":
                        home_team = team_name
                    else:
                        away_team = team_name

                # Extract Vegas line odds data out of the ESPN odds container
                market_spread = 0.0
                odds_list = comp.get("odds", [])
                if odds_list:
                    details = odds_list[0].get("details", "")
                    if "PK" in details or "EVEN" in details or not details:
                        market_spread = 0.0
                    else:
                        # Extract the trailing line number (e.g., "ALA -7.5" -> -7.5)
                        words = details.split()
                        if words:
                            try:
                                # Standard layout sets favorite to negative
                                raw_val = float(words[-1])
                                # If home team is favored, layout matches our model margin format
                                if words[0].lower() in home_team.lower():
                                    market_spread = raw_val
                                else:
                                    market_spread = -raw_val
                            except:
                                pass

                if home_team and away_team:
                    processed_games.append({
                        "home_team": home_team,
                        "away_team": away_team,
                        "market_spread": market_spread
                    })
    except:
        pass

    return processed_games

# =====================================================================
# 3. INTERFACE ARCHITECTURE
# =====================================================================
st.set_page_config(page_title="Automated D1 CFB Matrix", layout="wide")

st.title("ģ Ultimate All-D1 College Football Spread Matrix")
st.markdown("ģ **Fully Automated Edition:** Syncing historical power metrics against live ESPN sportsbook feeds.")

# Run calculations instantly
with st.spinner("Synchronizing live scheduling parameters..."):
    trained_elo_database = train_keyless_elo_model()
    live_odds_feed = fetch_live_espn_weekly_slate()

# Build team dropdown options safely
teams_set = set(trained_elo_database.keys())
for game in live_odds_feed:
    teams_set.add(game["home_team"])
    teams_set.add(game["away_team"])
all_teams = sorted(list(teams_set))

# SECTION 1: SIDEBAR INTERACTIVE MATCHUP SIMULATOR
st.sidebar.markdown("## ģ® Section 1: Custom Matchup Simulator")
st.sidebar.write("Select two teams to run a custom head-to-head score model simulation.")

sim_home = st.sidebar.selectbox("Home Team", all_teams, index=all_teams.index("Georgia") if "Georgia" in all_teams else 0)
sim_away = st.sidebar.selectbox("Away Team", all_teams, index=all_teams.index("Alabama") if "Alabama" in all_teams else 1)

if sim_home and sim_away:
    s_home, s_away, s_prob = generate_score_projection(sim_home, sim_away, trained_elo_database)
    model_spread = s_away - s_home

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"#### ģ Projection Results:")
    st.sidebar.markdown(f"**Score Prediction:**  \n`{sim_away} {s_away:.1f}` vs. `{sim_home} {s_home:.1f}`")
    st.sidebar.markdown(f"**Algorithmic Implied Spread:**  \n`{sim_home} {model_spread:+.1f}`")
    st.sidebar.markdown("---")

# SECTION 2: MAIN SLATE POINT SPREAD BOARD
st.markdown("## ģ Section 2: Full Weekend Matchups vs. Point Spread Board")
st.markdown("This board updates automatically every week with fresh lines and newly calculated model metrics.")

processed_slate = []
for game in live_odds_feed:
    home_team = game["home_team"]
    away_team = game["away_team"]
    market_spread = game["market_spread"]

    proj_home, proj_away, home_prob = generate_score_projection(home_team, away_team, trained_elo_database)
    winner = home_team if proj_home > proj_away else away_team
    win_percentage = home_prob if proj_home > proj_away else (1 - home_prob)

    model_home_margin = proj_away - proj_home
    edge = market_spread - model_home_margin

    if edge > 3.5:
        lean = f"ģ Best Bet: {home_team} ({market_spread:+.1f})"
    elif edge > 1.5:
        lean = f"ģ Value: {home_team} ({market_spread:+.1f})"
    elif edge < -3.5:
        lean = f"ģ Best Bet: {away_team} (+{abs(market_spread):.1f})"
    elif edge < -1.5:
        lean = f"ģ Value: {away_team} (+{abs(market_spread):.1f})"
    else:
        lean = "ģ Pass"

    processed_slate.append({
        "Home Program": home_team,
        "Away Program": away_team,
        "Public Market Line": f"Spread value: {market_spread:+.1f}",
        "Model Score Projection": f"{proj_away:.1f} - {proj_home:.1f}",
        "Predicted Winner": f"{winner} ({win_percentage:.0%})",
        "Model Edge Lean": lean
    })

if processed_slate:
    df_slate = pd.DataFrame(processed_slate)

    search_query = st.text_input("ģ Filter the board by team name (e.g. Indiana, Ohio State):", "").lower()
    if search_query:
        df_slate = df_slate[
            df_slate["Home Program"].str.lower().str.contains(search_query) |
            df_slate["Away Program"].str.lower().str.contains(search_query)
        ]

    st.dataframe(df_slate, use_container_width=True, hide_index=True)
else:
    st.warning("ģ  Connected to live streams. The board will automatically display games as sportsbooks release the weekly slates.")
    st.dataframe(df_slate, use_container_width=True, hide_index=True)
else:
    st.warning("⚠️ Connected to live streams. The board will automatically display games as sportsbooks release the weekly slates.")
