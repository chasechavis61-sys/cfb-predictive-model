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

@st.cache_data(ttl=86400)
def train_cfb_elo_model(api_key):
    elo_ratings = {}
    K_FACTOR = 35
    HOME_ADVANTAGE = 55

    headers = {"Authorization": f"Bearer {api_key}"}

    # FIXED: Using a range function to pull 2023, 2024, 2025, and 2026 securely
    for year in range(2023, 2027):
        url = f"https://collegefootballdata.com{year}&seasonType=regular"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                games = response.json()
                if not games:
                    continue
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
                st.sidebar.error(
f"CFBD API error for {year}: Status {response.status_code}")
        except Exception as e:
            st.sidebar.warning(f"Error compiling history for {year}: {e}")

    return elo_ratings

def get_closest_elo(team_name, elo_dict):
    """
    Directly translates commercial betting names (The Odds API)
    to official statistical names (CFBD API) to guarantee unique projections.
    """
    if not team_name:
        return 1500.0

    # 1. Manual Translation Map for major structural differences
    translation_map = {
        # SEC
        "Mississippi Rebels": "Ole Miss",
        "Mississippi St Bulldogs": "Mississippi State",
        "Texas A&M Aggies": "Texas A&M",
        "LSU Tigers": "LSU",
        "Alabama Crimson Tide": "Alabama",
        "Georgia Bulldogs": "Georgia",
        "Texas Longhorns": "Texas",
        "Tennessee Volunteers": "Tennessee",
        "Missouri Tigers": "Missouri",
        "Oklahoma Sooners": "Oklahoma",
        "Kentucky Wildcats": "Kentucky",
        "Auburn Tigers": "Auburn",
        "Florida Gators": "Florida",
        "South Carolina Gamecocks": "South Carolina",
        "Arkansas Razorbacks": "Arkansas",
        "Vanderbilt Commodores": "Vanderbilt",
        
        # Big Ten
        "Ohio State Buckeyes": "Ohio State",
        "Oregon Ducks": "Oregon",
        "Penn State Nittany Lions": "Penn State",
        "Michigan Wolverines": "Michigan",
        "USC Trojans": "USC",
        "Southern California Trojans": "USC",
        "Washington Huskies": "Washington",
        "Iowa Hawkeyes": "Iowa",
        "Wisconsin Badgers": "Wisconsin",
        "Nebraska Cornhuskers": "Nebraska",
        "Maryland Terrapins": "Maryland",
        "Minnesota Golden Gophers": "Minnesota",
        "Rutgers Scarlet Knights": "Rutgers",
        "Northwestern Wildcats": "Northwestern",
        "Illinois Fighting Illini": "Illinois",
        "Michigan State Spartans": "Michigan State",
        "UCLA Bruins": "UCLA",
        "Indiana Hoosiers": "Indiana",
        "Purdue Boilermakers": "Purdue",
        
        # Big 12
        "Utah Utes": "Utah",
        "Kansas State Wildcats": "Kansas State",
        "Oklahoma State Cowboys": "Oklahoma State",
        "Arizona Wildcats": "Arizona",
        "Kansas Jayhawks": "Kansas",
        "Iowa State Cyclones": "Iowa State",
        "West Virginia Mountaineers": "West Virginia",
        "TCU Horned Frogs": "TCU",
        "Texas Tech Red Raiders": "Texas Tech",
        "UCF Knights": "UCF",
        "BYU Cougars": "BYU",
        "Colorado Buffaloes": "Colorado",
        "Baylor Bears": "Baylor",
        "Houston Cougars": "Houston",
        "Arizona State Sun Devils": "Arizona State",
        "Cincinnati Bearcats": "Cincinnati",
        
        # ACC
        "Florida State Seminoles": "Florida State",
        "Clemson Tigers": "Clemson",
        "Miami Hurricanes": "Miami",
        "Louisville Cardinals": "Louisville",
        "NC State Wolfpack": "NC State",
        "SMU Mustangs": "SMU",
        "Virginia Tech Hokies": "Virginia Tech",
        "North Carolina Tar Heels": "North Carolina",
        "Georgia Tech Yellow Jackets": "Georgia Tech",
        "California Golden Bears": "California",
        "Duke Blue Devils": "Duke",
        "Syracuse Orange": "Syracuse",
        "Pittsburgh Panthers": "Pittsburgh",
        "Boston College Eagles": "Boston College",
        "Wake Forest Demon Deacons": "Wake Forest",
        "Virginia Cavaliers": "Virginia",
        "Stanford Cardinal": "Stanford",
        
        # Independents / Noteworthy Others
        "Notre Dame Fighting Irish": "Notre Dame",
        "Boise State Broncos": "Boise State",
        "Liberty Flames": "Liberty",
        "James Madison Dukes": "James Madison",
        "Memphis Tigers": "Memphis",
        "Tulane Green Wave": "Tulane",
        "UNLV Rebels": "UNLV"
    }

    # 2. Check the manual map first
    if team_name in translation_map:
        target_name = translation_map[team_name]
        return elo_dict.get(target_name, 1500.0)

    # 3. Check for exact match in case the API provides the short name
    if team_name in elo_dict:
        return elo_dict[team_name]

    # 4. Strip common mascots as a dynamic safety fallback
    clean_name = team_name
    mascots = ["Buckeyes", "Trojans", "Rebels", "Tigers", "Bulldogs", "Crimson Tide", "Longhorns", "Volunteers", "Sooners"]
    for mascot in mascots:
        clean_name = clean_name.replace(mascots, "").strip()

    if clean_name in elo_dict:
        return elo_dict[clean_name]

    # 5. Fuzzy matching fallback with list formatting correction
    closest_matches = difflib.get_close_matches(team_name, list(elo_dict.keys()), n=1, cutoff=0.25)
    if closest_matches:
        return elo_dict[closest_matches]

    return 1500.0

def generate_score_projection(home_team, away_team, elo_dict):
    HOME_ADVANTAGE_ELO = 55
    BASE_AVG_SCORE = 28.0
    POINTS_PER_ELO = 25.0

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
# 4. EDGE MATRIX COMPILATION ENGINE (FIXED EXTRACTION)
# =====================================================================
def compile_betting_slate(games, elo_dict):
    processed_slate = []

    for game in games:
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        market_spread = None
        bookmaker_used = "Consensus"

        # FIXED: Corrected deep list parsing for The Odds API structure
        bookmakers = game.get("bookmakers", [])
        if bookmakers and len(bookmakers) > 0:
            primary_book = bookmakers[0]
            bookmaker_used = primary_book.get("title", "Unknown")

            markets = primary_book.get("markets", [])
            if markets and len(markets) > 0:
                # Target the spreads market safely
                spread_market = next((m for m in markets if m.get("key") == "spreads"), markets[0])
                outcomes = spread_market.get("outcomes", [])
                
                for outcome in outcomes:
                    # We match the spread relative to the home team
                    if outcome.get("name") == home_team:
                        market_spread = float(outcome.get("point", 0))

        # If no active spread exists, use a default 0 spread to ensure the game still processes
        if market_spread is None:
            market_spread = 0.0
            bookmaker_used = "No Live Spread"

        proj_home, proj_away, home_prob = generate_score_projection(home_team, away_team, elo_dict)
        
        # Determine the projected winner based on unique scores
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
    trained_elo_database = train_cfb_elo_model(CFBD_API_KEY)

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

