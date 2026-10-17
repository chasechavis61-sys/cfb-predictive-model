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
@st.cache_data(ttl=600)
def get_cfbd_data(endpoint, params={}):
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    url = f"https://api.collegefootballdata.com{endpoint}"
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    return []

# Helper function to query The Odds API
@st.cache_data(ttl=300)
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
@st.cache_data(ttl=1800)
def get_team_stats_dict(year=2024):
    # Fetch team seasonal stats from CFBD
    stats_data = get_cfbd_data("/stats/season", {"year": year})
    teams_dict = {}
    for entry in stats_data:
        team = entry.get("team")
        stat_name = entry.get("statName")
        stat_val = entry.get("statValue")
        if not team:
            continue
        if team not in teams_dict:
            teams_dict[team] = {}
        teams_dict[team][stat_name] = stat_val
    return teams_dict

# Compute dynamic FBS average PPG from the team stats dictionary
def calculate_fbs_baseline_averages(team_stats, default_value=28.0):
    total_off_sum = 0.0
    total_def_sum = 0.0
    team_count = 0
    
    for team, stats_entry in team_stats.items():
        try:
            games = float(stats_entry.get("games", 12))
            if games <= 0:
                games = 12.0
            
            points = float(stats_entry.get("points", 0))
            points_allowed = float(stats_entry.get("pointsAllowed", 0))
            
            if points > 0 and points_allowed > 0:
                total_off_sum += (points / games)
                total_def_sum += (points_allowed / games)
                team_count += 1
        except (ValueError, TypeError):
            continue

    if team_count > 0:
        return total_off_sum / team_count, total_def_sum / team_count
    return default_value, default_value

# Standardize team names to fix API mismatches
def clean_team_name(name):
    if not name:
        return ""
    # Common suffixes or words to discard for matching purposes
    removals = ["buckeyes", "tigers", "longhorns", "crimson", "tide", "aggies", "gators",
                "seminoles", "wolverines", "badgers", "bruins", "trojans", "sooner",
                "sooners", "railriders", "state", "st", "univ", "university", "&", "and"]
    cleaned = name.lower()
    for r in removals:
        cleaned = cleaned.replace(f" {r}", " ").replace(f"{r} ", " ")
    return "".join(c for c in cleaned if c.isalnum()).strip()

# Find the best matching team key from our stats dataset
def find_matching_team(target_name, team_stats):
    target_cleaned = clean_team_name(target_name)
    if not target_cleaned:
        return None

    # Direct check
    if target_name in team_stats:
        return target_name

    # Fuzzy match using cleaned keys
    for real_name in team_stats.keys():
        if clean_team_name(real_name) == target_cleaned:
            return real_name
        if target_cleaned in clean_team_name(real_name) or clean_team_name(real_name) in target_cleaned:
            return real_name
    return None

# Helper to calculate and retrieve a team's per-game stats
def get_team_averages(team_name, team_stats, default_off=28.0, default_def=28.0):
    matched_name = find_matching_team(team_name, team_stats)
    if not matched_name:
        return default_off, default_def

    stats_entry = team_stats.get(matched_name, {})

    # Parse games, points, and pointsAllowed safely with numeric conversion
    try:
        games = float(stats_entry.get("games", 12))
    except (ValueError, TypeError):
        games = 12.0
    if games <= 0:
        games = 12.0

    try:
        total_pts = float(stats_entry.get("points", default_off * games))
    except (ValueError, TypeError):
        total_pts = default_off * games

    try:
        total_pts_allowed = float(stats_entry.get("pointsAllowed", default_def * games))
    except (ValueError, TypeError):
        total_pts_allowed = default_def * games

    off_ppg = total_pts / games
    def_ppg = total_pts_allowed / games
    return off_ppg, def_ppg

# Predictive Model utilizing real seasonal stats and dynamic baseline averages
def predict_game_outcome(home_team, away_team, team_stats, baseline_off=28.0, baseline_def=28.0):
    home_off, home_def = get_team_averages(home_team, team_stats, baseline_off, baseline_def)
    away_off, away_def = get_team_averages(away_team, team_stats, baseline_off, baseline_def)

    # Project score using Offensive performance vs Opponent Defense relative to baseline
    # Home score based on home offense vs away defense normalized by baselines
    home_pred_score = (home_off * away_def) / ((baseline_off + baseline_def) / 2)
    away_pred_score = (away_off * home_def) / ((baseline_off + baseline_def) / 2)

    # Add standard home field advantage (+2.5 points)
    home_pred_score += 2.5
    away_pred_score -= 2.5

    home_pred_score = max(7.0, round(home_pred_score, 1))
    away_pred_score = max(7.0, round(away_pred_score, 1))

    return home_pred_score, away_pred_score

# Helper to calculate probability and confidence level
def calculate_bet_confidence(model_projected, market_line, is_total=False):
    std_dev = 18.5 if is_total else 13.5
    edge = abs(model_projected - market_line)
    prob = stats.norm.cdf(edge / std_dev) if edge > 0 else 0.5

    if is_total:
        final_prob = 0.5 + (prob - 0.5) * 0.4
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
        with st.spinner("Fetching team stats, market lines, and generating predictions... "):
            odds_json = get_odds_data()
            team_stats = get_team_stats_dict()

            if not odds_json:
                st.error("Failed to retrieve odds data.")
            else:
                # Dynamically calculate baseline averages based on fetched data
                baseline_off, baseline_def = calculate_fbs_baseline_averages(team_stats)
                st.sidebar.metric(label="Dynamic CFB Offense Baseline", value=f"{baseline_off:.2f} PPG")
                st.sidebar.metric(label="Dynamic CFB Defense Baseline", value=f"{baseline_def:.2f} PPG")

                spread_picks = []
                totals_picks = []

                for game in odds_json:
                    home_team = game.get("home_team")
                    away_team = game.get("away_team")

                    # Generate prediction dynamically
                    pred_home, pred_away = predict_game_outcome(home_team, away_team, team_stats, baseline_off, baseline_def)
                    pred_spread = pred_away - pred_home
                    pred_total = pred_home + pred_away

                    # Retrieve team stats for transparent displaying
                    home_off_ppg, home_def_ppg = get_team_averages(home_team, team_stats, baseline_off, baseline_def)
                    away_off_ppg, away_def_ppg = get_team_averages(away_team, team_stats, baseline_off, baseline_def)

                    # Extract bookmaker lines
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

                    # 1. Evaluate Spread Play
                    if market_spread is not None:
                        prob_spread, conf_spread = calculate_bet_confidence(pred_spread, market_spread, is_total=False)
                        if pred_spread < market_spread:
                            spread_pick = f"{home_team} {market_spread}"
                        else:
                            opp_spread = +market_spread if market_spread < 0 else -market_spread
                            spread_pick = f"{away_team} {opp_spread}"

                        spread_picks.append({
                            "Game": f"{away_team} @ {home_team}",
                            "Model Pred (Away-Home)": f"{pred_away:.1f} - {pred_home:.1f}",
                            "Market Spread": market_spread,
                            "Best Spread Pick": spread_pick,
                            "Spread Prob": f"{prob_spread * 100:.1f}%",
                            "Spread Tier": conf_spread
                        })

                    # 2. Evaluate Totals Play
                    if market_total is not None:
                        prob_total, conf_total = calculate_bet_confidence(pred_total, market_total, is_total=True)
                        if pred_total > market_total:
                            total_pick = f"Over {market_total}"
                        else:
                            total_pick = f"Under {market_total}"

                        totals_picks.append({
                            "Game": f"{away_team} @ {home_team}",
                            "Away Off PPG / Def PPG": f"{away_off_ppg:.1f} / {away_def_ppg:.1f}",
                            "Home Off PPG / Def PPG": f"{home_off_ppg:.1f} / {home_def_ppg:.1f}",
                            "Model Projected Total": f"{pred_total:.1f}",
                            "Market Total": market_total,
                            "Best Totals Pick": total_pick,
                            "Totals Prob": f"{(prob_total * 100):.1f}%",
                            "Totals Tier": conf_total
                        })

                # Render UI Tables
                st.subheader("📊 Model Picks Against the Spread")
                if spread_picks:
                    st.dataframe(pd.DataFrame(spread_picks), use_container_width=True)
                else:
                    st.info("No active market spreads found.")

                st.write("---")

                st.subheader("🔮 Model Picks Against the Total (Over / Under)")
                if totals_picks:
                    st.dataframe(pd.DataFrame(totals_picks), use_container_width=True)
                else:
                    st.info("No active market totals found.")
    else:
        st.error("Please enter valid API keys.")
