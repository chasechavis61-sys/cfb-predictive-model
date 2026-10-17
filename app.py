import streamlit as st
import requests
import pandas as pd
import numpy as np
import scipy.stats as stats

st.set_page_config(page_title="CFB Predictive Model", layout="wide")
st.title("🏈 College Football Predictive Tool")

# Securely load API Keys from Streamlit Secrets
CFBD_API_KEY = st.secrets.get("CFBD_API_KEY", "")
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "")

if not CFBD_API_KEY or not ODDS_API_KEY:
    st.warning("⚠️ Please configure both CFBD_API_KEY and ODDS_API_KEY in your Streamlit secrets.")

# Helper function to query CFBD API
@st.cache_data(ttl=3600)
def get_cfbd_data(endpoint, params={}):
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    url = f"https://api.collegefootballdata.com{endpoint}"
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    return []

# Helper function to query The Odds API
@st.cache_data(ttl=600)
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

# Fetch Team Records and Stats from CFBD to power the model
@st.cache_data(ttl=86400)
def get_team_stats_dict(year=2024):
    # Fetch team seasonal stats from CFBD
    stats_data = get_cfbd_data("/stats/season", {"year": year})
    teams_dict = {}
    for entry in stats_data:
        team = entry.get("team")
        stat_name = entry.get("statName")
        stat_val = entry.get("statValue")
        if team not in teams_dict:
            teams_dict[team] = {}
        teams_dict[team][stat_name] = stat_val
    return teams_dict

# Predictive Model utilizing real seasonal stats
def predict_game_outcome(home_team, away_team, team_stats, default_avg=28.0):
    # Default statistics fallback if team is not found in CFBD
    home_offense = team_stats.get(home_team, {}).get("totalYards", 400.0)
    away_offense = team_stats.get(away_team, {}).get("totalYards", 400.0)

    # We can use points-based indicators or yardages to build an index.
    # Let's derive a simple Power Index using yards gained as a proxy for offensive/defensive capability
    # Average CFB team gains ~400 yards/game and scores ~28 points (~1 point per 14.3 yards)
    home_pred_score = (home_offense / 14.3)
    away_pred_score = (away_offense / 14.3)

    # Add a standard home field advantage (typically +3 points in CFB)
    home_pred_score += 1.5
    away_pred_score -= 1.5

    # Clamp values to realistic scores
    home_pred_score = max(10.0, round(home_pred_score, 1))
    away_pred_score = max(10.0, round(away_pred_score, 1))

    return home_pred_score, away_pred_score

# Helper to calculate probability and confidence level
def calculate_bet_confidence(model_projected, market_line, is_total=False):
    # Spreads have a lower standard deviation (~13.5) compared to Totals (~18.5) due to high variability
    std_dev = 18.5 if is_total else 13.5
    
    edge = abs(model_projected - market_line)
    prob = stats.norm.cdf(edge / std_dev) if edge > 0 else 0.5

    # Realistically scale probabilities around typical sports betting outcomes
    # Spreads get scaled slightly differently than Totals
    if is_total:
        final_prob = 0.5 + (prob - 0.5) * 0.4  # more conservative scaling for totals
    else:
        final_prob = 0.5 + (prob - 0.5) * 0.5

    if final_prob >= 0.55:
        confidence = "🟢 Strong Play"
    elif final_prob >= 0.515:
        confidence = "🟡 Slight Lean"
    else:
        confidence = "🔴 Hold Off"

    return final_prob, confidence

# Fetch and Process Data
if st.sidebar.button("Fetch Latest Games & Predict"):
    if CFBD_API_KEY and ODDS_API_KEY:
        with st.spinner("Fetching team stats, market lines, and generating predictions..."):
            odds_json = get_odds_data()
            team_stats = get_team_stats_dict()

            if not odds_json:
                st.error("Failed to retrieve odds data.")
            else:
                compiled_picks = []
                for game in odds_json:
                    home_team = game.get("home_team")
                    away_team = game.get("away_team")

                    # Generate prediction dynamically
                    pred_home, pred_away = predict_game_outcome(home_team, away_team, team_stats)
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
                    spread_prob_str = "N/A"
                    spread_confidence = "🔴 Hold Off"

                    total_pick = "No value"
                    total_prob_str = "N/A"
                    total_confidence = "🔴 Hold Off"

                    if market_spread is not None:
                        prob, confidence = calculate_bet_confidence(pred_spread, market_spread, is_total=False)
                        spread_prob_str = f"{prob * 100:.1f}%"
                        spread_confidence = confidence

                        if pred_spread < market_spread:
                            spread_pick = f"{home_team} {market_spread}"
                        else:
                            opp_spread = +market_spread if market_spread < 0 else -market_spread
                            spread_pick = f"{away_team} {opp_spread}"

                    if market_total is not None:
                        prob, confidence = calculate_bet_confidence(pred_total, market_total, is_total=True)
                        total_prob_str = f"{prob * 100:.1f}%"
                        total_confidence = confidence

                        if pred_total > market_total:
                            total_pick = f
