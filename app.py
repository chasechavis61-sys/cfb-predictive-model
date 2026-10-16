import streamlit as st
import requests
import pandas as pd
import numpy as np

# =====================================================================
# 1. API CONFIGURATION
# =====================================================================
# 🛑 PASTE YOUR ACTIVE API KEYS HERE:
THE_ODDS_API_KEY = "Lx5SW7fjPsgpYp84gv245NpLzpUAW3ZuJVttG785YZKkqI3wDqgWUpqCqnY0YxXN"
CFBD_API_KEY = "875f2f3bcab472c2046d7efb6659a7c3"

# =====================================================================
# 2. THE EMBEDDED ELO PREDICTIVE MODEL ENGINE
# =====================================================================
@st.cache_data(ttl=86400)
def train_cfb_elo_model(api_key):
    """Trains a math-based Elo system by processing past games chronologically."""
    elo_ratings = {}
    K_FACTOR = 35       
    HOME_ADVANTAGE = 55 
    
    headers = {"Authorization": f"Bearer {api_key}"}

    
for year in training_years:
        url = f"https://collegefootballdata.com{year}&seasonType=regular"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                games = response.json()
                games = sorted(games, key=lambda x: x.get('week', 0))
                
                for game in games:
                    if game.get('home_points') is None or game.get('away_points') is None:
                        continue
                        
                    home_team = game['home_team']
                    away_team = game['away_team']
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
            else:
                st.warning(f"CFBD API responded with status {response.status_code} for season {year}")
        except Exception as e:
            st.error(f"Error compiling history for {year}: {e}")
        return elo_ratings

def generate_score_projection(home_team, away_team, elo_dict):
    """Calculates final scores and cover logic using the trained Elo ratings."""
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
# 3. LIVE MARKET SYNCHRONIZATION PIPELINE (ISOLATING FANDUEL)
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_live_market_spreads(api_key):
    """Pulls betting spreads and prioritizes FanDuel data."""
    combined_games = []
    sport_keys = ["americanfootball_ncaaf", "americanfootball_ncaaf_fcs"]
    
    for key in sport_keys:
        url = f"https://the-odds-api.com{key}/odds/"
        params = {"apiKey": api_key, "regions": "us", "markets": "spreads", "oddsFormat": "american"}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                combined_games.extend(response.json())
        except Exception as e:
            st.error(f"Network error linking to Odds feed: {e}")
    return combined_games

def extract_fanduel_spread(game):
    """Safely searches the odds list to isolate FanDuel's specific numbers."""
    if game.get("bookmakers"):
        # Loop to specifically find FanDuel inside the sportsbook payload
        for bookmaker in game["bookmakers"]:
            if bookmaker.get("key", "").lower() == "fanduel":
                markets = bookmaker.get("markets", [])
                if markets:
                    outcomes = markets[0].get("outcomes", [])
                    for outcome in outcomes:
                        if outcome["name"] == game["home_team"]:
                            return float(outcome["point"]), "FanDuel"
                            
        # Fallback to consensus bookmaker if FanDuel hasn't posted a line yet
        primary_book = game["bookmakers"][0]
        markets = primary_book.get("markets", [])
        if markets:
            outcomes = markets[0].get("outcomes", [])
            for outcome in outcomes:
                if outcome["name"] == game["home_team"]:
                    return float(outcome["point"]), f"{primary_book.get('title')} (FD Pending)"
                    
    return None, "No Line"

# =====================================================================
# 4. EDGE MATRIX COMPILATION ENGINE
# =====================================================================
def compile_betting_slate(games, elo_dict):
    processed_slate = []
    
    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        
        market_spread, book_source = extract_fanduel_spread(game)
                        
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
            lean = f"➡️ {home_team} ({market_spread}) Slight Lean"
        elif edge < -3.5:
            lean = f"🔥 {away_team} (+{abs(market_spread)}) Strong Play"
        elif edge < -1.5:
            lean = f"➡️ {away_team} (+{abs(market_spread)}) Slight Lean"
        else:
            lean = "❌ Pass / No Value"
            
        processed_slate.append({
            "Home Program": home_team,
            "Away Program": away_team,
            "FanDuel Market Spread": f"{home_team} {market_spread} ({book_source})",
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

if THE_ODDS_API_KEY == "YOUR_THE_ODDS_API_KEY_HERE" or CFBD_API_KEY == "YOUR_CFBD_API_KEY_HERE":
    st.error("🚨 Key Insertion Required: Please update lines 10 & 11 with your API credentials.")
else:
    # 1. Initialize and build the model database
    trained_elo_database = train_cfb_elo_model(CFBD_API_KEY)
    live_odds_feed = fetch_live_market_spreads(THE_ODDS_API_KEY)
    
    # Create clean list of all unique teams detected by the model
    all_teams = sorted(list(trained_elo_database.keys()))
    
    # =================================================================
    # NEW FEATURE: CUSTOM MANUAL MATCHUP SIMULATOR WIDGET
    # =================================================================
    st.markdown("### 🧮 Manual Custom Matchup Simulator")
    st.write("Pick any two D1 programs to generate an instant algorithmic projection.")
    
    col1, col2 = st.columns(2)
    with col1:
        sim_home = st.selectbox("Select Home Team:", all_teams, index=all_teams.index("Georgia") if "Georgia" in all_teams else 0)
    with col2:
        sim_away = st.selectbox("Select Away Team:", all_teams, index=all_teams.index("Alabama") if "Alabama" in all_teams else 1)
        
    if sim_home and sim_away:
        s_home, s_away, s_prob = generate_score_projection(sim_home, sim_away, trained_elo_database)
        model_spread = s_away - s_home # Negative means Home is favored
        
        # Look for an active live spread for this simulated game in our odds feed
        found_fd_line = "No Live Line Posted"
        for g in live_odds_feed:
            if g["home_team"] == sim_home and g["away_team"] == sim_away:
                line, source = extract_fanduel_spread(g)
                if line is not None:
                    found_fd_line = f"{sim_home} {line} ({source})"
                    break
        
        # Display custom metrics boxes
        m1, m2, m3 = st.columns(3)
        m1.metric("Projected Score", f"{sim_away} {s_away:.1f} - {sim_home} {s_home:.1f}")
        m2.metric("Model Implied Spread", f"{sim_home} {model_spread:+.1f}")
        m3.metric("FanDuel Market Line", found_fd_line)
    
    st.divider()
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

# =====================================================================
# 2. THE EMBEDDED ELO PREDICTIVE MODEL ENGINE
# =====================================================================
@st.cache_data(ttl=86400)
def train_cfb_elo_model(api_key):
    """Trains a math-based Elo system by processing past games chronologically."""
    elo_ratings = {}
    K_FACTOR = 35       
    HOME_ADVANTAGE = 55 
    
    headers = {"Authorization": f"Bearer {api_key}"}
    training_years = [2024, 2025, 2026]
    
    for year in training_years:
        url = f"https://collegefootballdata.com{year}&seasonType=regular"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                games = response.json()
                games = sorted(games, key=lambda x: x.get('week', 0))
                
                for game in games:
                    if game.get('home_points') is None or game.get('away_points') is None:
                        continue
                        
                    home_team = game['home_team']
                    away_team = game['away_team']
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
    """Calculates final scores using the trained Elo ratings database."""
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
# 3. LIVE MARKET SYNCHRONIZATION PIPELINE (ISOLATING FANDUEL)
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_live_market_spreads(api_key):
    """Pulls betting spreads across both FBS and FCS division channels."""
    combined_games = []
    sport_keys = ["americanfootball_ncaaf", "americanfootball_ncaaf_fcs"]
    
    for key in sport_keys:
        url = f"https://api.the-odds-api.com/v4/sports/{key}/odds/"
        params = {"apiKey": api_key, "regions": "us", "markets": "spreads", "oddsFormat": "american"}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                combined_games.extend(response.json())
        except:
            pass
    return combined_games

def extract_fanduel_spread(game):
    """Isolates FanDuel odds safely, falling back accurately to alternative sportsbooks."""
    bookmakers = game.get("bookmakers", [])
    if not bookmakers:
        return 0.0, "Opening Line / PK"
        
    # Step A: Target FanDuel explicitly
    for bookmaker in bookmakers:
        if bookmaker.get("key", "").lower() == "fanduel":
            for market in bookmaker.get("markets", []):
                if market.get("key") == "spreads":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == game["home_team"]:
                            return float(outcome["point"]), "FanDuel"
                            
    # Step B: ✅ FIXED: Explicitly targeting index 0 of bookmakers list to prevent crashing
    first_book = bookmakers[0]
    for market in first_book.get("markets", []):
        if market.get("key") == "spreads":
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == game["home_team"]:
                    return float(outcome["point"]), f"{first_book.get('title', 'Book')} (Alt)"
                    
    return 0.0, "Market Line"

# =====================================================================
# 4. EDGE MATRIX COMPILATION ENGINE
# =====================================================================
def compile_betting_slate(games, elo_dict):
    processed_slate = []
    
    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        
        market_spread, book_source = extract_fanduel_spread(game)
            
        proj_home, proj_away, home_prob = generate_score_projection(home_team, away_team, elo_dict)
        winner = home_team if proj_home > proj_away else away_team
        win_percentage = home_prob if proj_home > proj_away else (1 - home_prob)
        
        model_home_margin = proj_away - proj_home  
        edge = market_spread - model_home_margin
        
        if edge > 3.5:
            lean = f"🔥 Best Bet: {home_team} ({market_spread:+.1f})"
        elif edge > 1.5:
            lean = f"➡️ Value: {home_team} ({market_spread:+.1f})"
        elif edge < -3.5:
            lean = f"🔥 Best Bet: {away_team} (+{abs(market_spread):.1f})"
        elif edge < -1.5:
            lean = f"➡️ Value: {away_team} (+{abs(market_spread):.1f})"
        else:
            lean = "❌ Pass"
            
        processed_slate.append({
            "Home Program": home_team,
            "Away Program": away_team,
            "Live Market Spread": f"{home_team} {market_spread:+.1f} ({book_source})",
            "Model Score Projection": f"{proj_away:.1f} - {proj_home:.1f}",
            "Outright Winner Prediction": f"{winner} ({win_percentage:.0%})",
            "Calculated Algorithmic Lean": lean
        })
        
    return pd.DataFrame(processed_slate)

# =====================================================================
# 5. STREAMLIT VISUAL FRONTEND UI
# =====================================================================
st.set_page_config(page_title="All-D1 Predictive Matrix", layout="wide")

st.title("🏈 Ultimate All-D1 CFB Algorithmic Betting Matrix")

if THE_ODDS_API_KEY == "YOUR_THE_ODDS_API_KEY_HERE" or CFBD_API_KEY == "YOUR_CFBD_API_KEY_HERE":
    st.error("🚨 Key Insertion Required: Please update lines 10 & 11 with your API credentials.")
else:
    trained_elo_database = train_cfb_elo_model(CFBD_API_KEY)
    live_odds_feed = fetch_live_market_spreads(THE_ODDS_API_KEY)
    
    teams_set = set(trained_elo_database.keys())
    if live_odds_feed:
        for g in live_odds_feed:
            teams_set.add(g["home_team"])
            teams_set.add(g["away_team"])
            
    all_teams = sorted(list(teams_set)) if teams_set else ["Georgia", "Alabama", "Ohio State", "Clemson"]

    # =================================================================
    # ✅ FIXED SECTION 1: SIDEBAR INTERACTIVE MATCHUP SIMULATOR W/ TEXT WRAPPING
    # =================================================================
    st.sidebar.markdown("## 🧮 Section 1: Custom Matchup Simulator")
    st.sidebar.write("Select two teams to run a custom head-to-head score model simulation.")
    
    sim_home = st.sidebar.selectbox("Home Team", all_teams, index=all_teams.index("Georgia") if "Georgia" in all_teams else 0)
    sim_away = st.sidebar.selectbox("Away Team", all_teams, index=all_teams.index("Alabama") if "Alabama" in all_teams else 1)
    
    if sim_home and sim_away:
        s_home, s_away, s_prob = generate_score_projection(sim_home, sim_away, trained_elo_database)
        model_spread = s_away - s_home
        
        # Using markdown blocks instead of rigid metric boxes ensures text wrapping is flawless
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"#### 📊 Projection Results:")
        st.sidebar.markdown(f"**Score Prediction:**  \n`{sim_away} {s_away:.1f}` vs. `{sim_home} {s_home:.1f}`")
        st.sidebar.markdown(f"**Algorithmic Implied Spread:**  \n`{sim_home} {model_spread:+.1f}`")
        st.sidebar.markdown("---")
