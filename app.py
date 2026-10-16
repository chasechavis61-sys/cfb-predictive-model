import streamlit as st
import requests
import pandas as pd
import numpy as np
import difflib

# =====================================================================
# 1. API CONFIGURATION
# =====================================================================
#  PASTE YOUR ACTIVE API KEYS HERE:
THE_ODDS_API_KEY = "875f2f3bcab472c2046d7efb6659a7c3"
CFBD_API_KEY = "Lx5SW7fjPsgpYp84gv245NpLzpUAW3ZuJVttG785YZKkqI3wDqgWUpqCqnY0YxXN"

# =====================================================================
# 2. THE EMBEDDED ELO PREDICTIVE MODEL ENGINE (ADJUSTED)
# =====================================================================

def get_closest_elo(team_name, elo_dict):
    """Helper function to map mismatched team names from different APIs."""
    if team_name in elo_dict:
        return elo_dict[team_name]
    
    # Find closest match in the Elo database if exact match fails
    closest_matches = difflib.get_close_matches(team_name, elo_dict.keys(), n=1, cutoff=0.4)
    if closest_matches:
        return elo_dict[closest_matches[0]]
        
    return 1500.0  # Fallback if absolutely no match is found

def generate_score_projection(home_team, away_team, elo_dict):
    HOME_ADVANTAGE_ELO = 55
    BASE_AVG_SCORE = 28.0
    POINTS_PER_ELO = 25.0

    # FIXED: Using the fuzzy mapping function instead of direct .get()

    r_home = get_closest_elo(home_team, elo_dict) + HOME_ADVANTAGE_ELO
    r_away = get_closest_elo(away_team, elo_dict)

    projected_margin = (r_home - r_away) / POINTS_PER_ELO
    proj_home = max(0, BASE_AVG_SCORE + (projected_margin / 2))
    proj_away = max(0, BASE_AVG_SCORE - (projected_margin / 2))

    win_prob = 1 / (1 + 10 ** ((r_away - r_home) / 400))
    return proj_home, proj_away, win_prob

# =====================================================================
# 3. LIVE MARKET SYNCHRONIZATION PIPELINE
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_live_market_spreads(api_key):
    combined_games = []
    sport_keys = ["americanfootball_ncaaf", "americanfootball_ncaaf_fcs"]

    for key in sport_keys:
        url = f"https://api.the-odds-api.com/v4/sports/{key}/odds/"
        params = {"apiKey": api_key, "regions": "us", "markets": "spreads", "oddsFormat": "american"}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                combined_games.extend(response.json())
            else:
                st.error(f"The Odds API Error ({key}): Status Code {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Network error linking to The Odds API: {e}")
    return combined_games

# =====================================================================
# 4. EDGE MATRIX COMPILATION ENGINE
# =====================================================================
def compile_betting_slate(games, elo_dict):
    processed_slate = []

    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]

        market_spread = None
        bookmaker_used = "Consensus"

        if game.get("bookmakers") and len(game["bookmakers"]) > 0:
            primary_book = game["bookmakers"][0]
            bookmaker_used = primary_book.get("title", "Unknown")

            markets = primary_book.get("markets", [])
            if markets and len(markets) > 0:
                outcomes = markets[0].get("outcomes", [])
                for outcome in outcomes:
                    if outcome["name"] == home_team:
                        market_spread = float(outcome["point"])

        if market_spread is None:
            continue

        proj_home, proj_away, home_prob = generate_score_projection(home_team, away_team, elo_dict)
        winner = home_team if proj_home > proj_away else away_team
        win_percentage = home_prob if proj_home > proj_away else (1 - home_prob)

        model_home_margin = proj_away - proj_home
        edge = market_spread - model_home_margin

        if edge > 3.5:
            lean = f"🔥 {home_team} ({market_spread}) Strong Play"
        elif edge > 1.5:
            lean = f"👌 {home_team} ({market_spread}) Slight Lean"
        elif edge < -3.5:
            lean = f"🔥 {away_team} (+{abs(market_spread)}) Strong Play"
        elif edge < -1.5:
            lean = f"👌 {away_team} (+{abs(market_spread)}) Slight Lean"
        else:
            lean = "➡️ Pass / No Value"

        processed_slate.append({
            "Home Program": home_team,
            "Away Program": away_team,
            "Live Vegas Spread": f"{home_team} {market_spread} ({bookmaker_used})",
            "Model Score Projection": f"{proj_away:.1f} - {proj_home:.1f}",
            "Outright Winner Projection": f"{winner} ({win_percentage:.0%})",
            "Calculated Algorithmic Lean": lean
        })

    return pd.DataFrame(processed_slate)

# =====================================================================
# 5. STREAMLIT VISUAL FRONTEND UI
# =====================================================================
st.set_page_config(page_title="All-D1 Predictive Matrix", layout="wide")

st.title("🏈 Ultimate All-D1 CFB Algorithmic Betting Matrix")
st.markdown("Automated edge mapping syncing custom historical data against live odds feeds.")

if THE_ODDS_API_KEY == "YOUR_THE_ODDS_API_KEY_HERE" or CFBD_API_KEY == "YOUR_CFBD_API_KEY_HERE":
    st.error("ģ Key Insertion Required: Update lines 10 & 11 inside the app.py script file.")
else:
    status_box = st.info("Step 1: Running Elo calculations across thousands of historical D1 matches...")
    trained_elo_database = (CFBD_API_KEY)

    status_box.info(
"Step 2: Connecting to sportsbooks to collect live games and spreads...")
    live_odds_feed = fetch_live_market_spreads(THE_ODDS_API_KEY)

    status_box.empty()

    if live_odds_feed:
        df_slate = compile_betting_slate(live_odds_feed, trained_elo_database)

        if not df_slate.empty:
            search_query = st.text_input("ģ Search for a team:", "").lower()
            if search_query:
                df_slate = df_slate[
                    df_slate["Home Program"].str.lower().str.contains(search_query) |
                    df_slate["Away Program"].str.lower().str.contains(search_query)
                ]
            st.dataframe(df_slate, use_container_width=True, hide_index=True)
        else:
            st.warning("ģ  Connected successfully, but no games with active point spread markets are open right now.")
    else:
        st.error("ģ Failed to download any upcoming odds. Check the error notifications above for structural details.")
