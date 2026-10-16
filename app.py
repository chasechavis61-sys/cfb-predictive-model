%%writefile app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="CFB Predictive Model", layout="wide")
st.title("🏈 College Football Predictive Tool")

# Securely load API Keys from Streamlit Secrets
# (Make sure to define these in your Streamlit dashboard secrets or .streamlit/secrets.toml)
CFBD_API_KEY = st.secrets.get("CFBD_API_KEY", "")
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "")

if not CFBD_API_KEY or not ODDS_API_KEY:
    st.warning("⚠️ Please configure both CFBD_API_KEY and ODDS_API_KEY in your Streamlit secrets.")

# Helper function to query CFBD API
def get_cfbd_data(endpoint, params={}):
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    url = f"https://api.collegefootballdata.com{endpoint}"
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    return []

# Helper function to query The Odds API
def get_odds_data(sport="americanfootball_ncaaf", regions="us", markets="spreads,totals"):
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return []

# Basic Predictive Model Placeholder (e.g., Elo or Simple Rating System comparison)
def predict_game_outcome(home_team, away_team):
    # Replace this simple logic with your own model's prediction engine
    # e.g., loads a trained model, calculates team ratings, etc.
    predicted_home_score = 31.5
    predicted_away_score = 24.0
    return predicted_home_score, predicted_away_score

# Fetch and Process Data
if st.sidebar.button("Fetch Latest Games & Predict"):
    if CFBD_API_KEY and ODDS_API_KEY:
        with st.spinner("Fetching market lines and generating predictions..."):
            odds_json = get_odds_data()
            
            if not odds_json:
                st.error("Failed to retrieve odds data.")
            else:
                compiled_picks = []
                for game in odds_json:
                    home_team = game.get("home_team")
                    away_team = game.get("away_team")
                    
                    # Generate prediction
                    pred_home, pred_away = predict_game_outcome(home_team, away_team)
                    pred_spread = pred_away - pred_home # negative means home is favored
                    pred_total = pred_home + pred_away
                    
                    # Extract bookmaker line (using first available bookmaker)
                    bookmakers = game.get("bookmakers", [])
                    market_spread = None
                    market_total = None
                    
                    if bookmakers:
                        markets = bookmakers[0].get("markets", [])
                        for market in markets:
                            if market["key"] == "spreads":
                                # Find home team spread
                                outcomes = market.get("outcomes", [])
                                for outcome in outcomes:
                                    if outcome["name"] == home_team:
                                        market_spread = outcome["point"]
                            elif market["key"] == "totals":
                                outcomes = market.get("outcomes", [])
                                if outcomes:
                                    market_total = outcomes[0]["point"]
                    
                    # Evaluate Picks
                    spread_pick = "No value"
                    total_pick = "No value"
                    
                    if market_spread is not None:
                        # If model projects home to win by more than market spread
                        if pred_spread < market_spread:
                            spread_pick = f"{home_team} {market_spread}"
                        else:
                            spread_pick = f"{away_team} {+market_spread if market_spread < 0 else -market_spread}"
                            
                    if market_total is not None:
                        if pred_total > market_total:
                            total_pick = f"Over {market_total}"
                        else:
                            total_pick = f"Under {market_total}"
                            
                    compiled_picks.append({
                        "Game": f"{away_team} @ {home_team}",
                        "Model Pred (Away-Home)": f"{pred_away:.1f} - {pred_home:.1f}",
                        "Market Spread": market_spread,
                        "Market Total": market_total,
                        "Best Spread Pick": spread_pick,
                        "Best Totals Pick": total_pick
                    })
                
                df_picks = pd.DataFrame(compiled_picks)
                st.dataframe(df_picks, use_container_width=True)
    else:
        st.error("Please enter valid API keys.")
