import streamlit as st
import requests
import pandas as pd
import numpy as np

# =====================================================================
# 1. API CONFIGURATION
# =====================================================================
# 🛑 PASTE YOUR ACTIVE API KEYS HERE:
THE_ODDS_API_KEY = "875f2f3bcab472c2046d7efb6659a7c3"
CFBD_API_KEY = "Lx5SW7fjPsgpYp84gv245NpLzpUAW3ZuJVttG785YZKkqI3wDqgWUpqCqnY0YxXN"

CURRENT_YEAR = 2026  # Calibrated for the active season

# =====================================================================
# 2. THE EMBEDDED ELO PREDICTIVE MODEL ENGINE
# =====================================================================
@st.cache_data(ttl=86400) # Cache calculations for 24 hours to save API load
def train_cfb_elo_model(api_key, start_year=2024, end_year=2026):
    """
    Downloads historical box scores and builds a custom Elo algorithm 
    to rank all Division 1 teams based on strength of schedule and performance.
    """
    # Baseline initialization: All D1 teams start at a baseline 1500 rating
    elo_ratings = {}
    
    # Core Elo Constants
    K_FACTOR = 35       # Maximum rating points a team can gain/lose in a game
    HOME_ADVANTAGE = 55 # Home-field advantage expressed in Elo rating points
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Step A: Loop through historical seasons to train the mathematical weights
    for year in range(start_year, end_year + 1):
        url = f"https://collegefootballdata.com{year}&seasonType=regular"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                continue
            games = response.json()
            
            # Sort chronologically by week to process progression correctly
            games = sorted(games, key=lambda x: x.get('week', 0))
            
            for game in games:
                # Ensure the game has already concluded with a valid box score
                if game.get('home_points') is None or game.get('away_points') is None:
                    continue
                    
                home_team = game['home_team']
                away_team = game['away_team']
                home_score = int(game['home_points'])
                away_score = int(game['away_points'])
                
                # Initialize teams if they don't exist in our dictionary
                if home_team not in elo_ratings: elo_ratings[home_team] = 1500.0
                if away_team not in elo_ratings: elo_ratings[away_team] = 1500.0
                
                # Current ratings prior to kickoff
                r_home = elo_ratings[home_team] + HOME_ADVANTAGE
                r_away = elo_ratings[away_team]
                
                # Calculate Win Expectations based on rating gaps
                exp_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
                exp_away = 1 - exp_home
                
                # Actual Outcomes
                actual_home = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)
                actual_away = 1.0 - actual_home
                
                # Margin of Victory Multiplier (prevents ranking deflation on blowout wins)
                margin = abs(home_score - away_score)
                mov_multiplier = np.log(margin + 1) * (2.2 / ((r_home - r_away) * 0.001 + 2.2)) if margin > 0 else 1.0
                
                # Shifting Rating Adjustments
                elo_ratings[home_team] += K_FACTOR * mov_multiplier * (actual_home - exp_home)
                elo_ratings[away_team] += K_FACTOR * mov_multiplier * (actual_away - exp_away)
                
        except Exception as e:
            pass
            
    return elo_ratings

def generate_score_projection(home_team, away_team, elo_dict):
    """Uses calculated Elo metrics to generate final projected point values."""
    HOME_ADVANTAGE_ELO = 55
    BASE_AVG_SCORE = 28.0 # Typical average points scored by a CFB team
    POINTS_PER_ELO = 25.0 # How many Elo points map to a 1-point scoring margin
    
    # Use baseline default if a lower-tier team isn't caught in historical tracking
    r_home = elo_dict.get(home_team, 1400.0) + HOME_ADVANTAGE_ELO
    r_away = elo_dict.get(away_team, 1400.0)
    
    # Calculate point margin difference
    projected_margin = (r_home - r_away) / POINTS_PER_ELO
    
    # Map out scores centered around average college football baselines
    proj_home = max(0, BASE_AVG_SCORE + (projected_margin / 2))
    proj_away = max(0, BASE_AVG_SCORE - (projected_margin / 2))
    
    # Outright Win Probability via standard logistic function
    win_prob = 1 / (1 + 10 ** ((r_away - r_home) / 400))
    
    return proj_home, proj_away, win_prob

# =====================================================================
# 3. LIVE MARKET SYNCHRONIZATION PIPELINE
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_live_market_spreads(api_key):
    """Pulls live betting spreads across both FBS and FCS division structures."""
    combined_games = []
    sport_keys = ["americanfootball_ncaaf", "americanfootball_ncaaf_fcs"]
    
    for key in sport_keys:
        url = f"https://the-odds-api.com{key}/odds/"
        params = {"apiKey": api_key, "regions": "us", "markets": "spreads", "oddsFormat": "american"}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                combined_games.extend(response.json())
        except:
            pass
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
        
        if game.get("bookmakers"):
            primary_book = game["bookmakers"][0]
            bookmaker_used = primary_book["title"]
            markets = primary_book.get("markets", [])
            if markets:
                outcomes = markets[0].get("outcomes", [])
                for outcome in outcomes:
                    if outcome["name"] == home_team:
                        market_spread = float(outcome["point"])
                        
        if market_spread is None:
            continue
            
        # Run our dynamically calculated algorithmic model
        proj_home, proj_away, home_prob = generate_score_projection(home_team, away_team, elo_dict)
        
        # Determine Predicted Winner Outright
        winner = home_team if proj_home > proj_away else away_team
        win_percentage = home_prob if proj_home > proj_away else (1 - home_prob)
        
        # Calculate Vegas Spread vs. Algorithmic Margin Edge
        # Vegas spread is home-centric (e.g. -7 means home team must win by 7+)
        model_home_margin = proj_away - proj_home  
        edge = market_spread - model_home_margin
        
        # Separate normal market variance from structural mathematical advantages
        if edge > 3.5:
            lean = f"🔥 {home_team} ({market_spread}) Strong Play"
        elif edge > 1.5:
            lean = f"➡️ {home_team} ({market_spread}) Slight Lean"
        elif edge < -3.5:
            lean = f"🔥 {away_team} (+{abs(market_spread) if market_spread < 0 else -market_spread}) Strong Play"
        elif edge < -1.5:
            lean = f"➡️ {away_team} (+{abs(market_spread) if market_spread < 0 else -market_spread}) Slight Lean"
        else:
            lean = "❌ No Clear Value / Pass"
            
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
st.markdown("This dashboard extracts years of game records to run a native **Elo Power Ranking System**, identifying value gaps against live Las Vegas spreads in real time.")

if THE_ODDS_API_KEY == "YOUR_THE_ODDS_API_KEY_HERE" or CFBD_API_KEY == "YOUR_CFBD_API_KEY_HERE":
    st.error("🚨 Key Insertion Required: Update lines 10 & 11 with your specific API key tokens.")
else:
    with st.spinner("Training mathematical Elo weights from college football history and updating live lines..."):
        # Train model over multiple historic cycles up to current date
        trained_elo_database = train_cfb_elo_model(CFBD_API_KEY)
        live_odds_feed = fetch_live_market_spreads(THE_ODDS_API_KEY)
        
        if live_odds_feed:
            df_slate = compile_betting_slate(live_odds_feed, trained_elo_database)
            
            # Global Text Query Filtering
            search_query = st.text_input("🔍 Search for a team:", "").lower()
            if search_query:
                df_slate = df_slate[
                    df_slate["Home Program"].str.lower().str.contains(search_query) | 
                    df_slate["Away Program"].str.lower().str.contains(search_query)
                ]
            
            # Display interactive dataframe matrix
            st.dataframe(df_slate, use_container_width=True, hide_index=True)
