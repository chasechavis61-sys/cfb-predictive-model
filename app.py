import streamlit as st
import requests
import pandas as pd
import numpy as np
import difflib

# =====================================================================
# 1. API CONFIGURATION
# =====================================================================
THE_ODDS_API_KEY = "875f2f3bcab472c2046d7efb6659a7c3"
CFBD_API_KEY = "Lx5SW7fjPsgpYp84gv245NpLzpUAW3ZuJVttG785YZKkqI3wDqgWUpqCqnY0YxXN"

# =====================================================================
# 2. THE EMBEDDED ELO PREDICTIVE MODEL ENGINE
# =====================================================================

@st.cache_data(ttl=86400)
def train_cfb_elo_model(api_key):
    elo_ratings = {}
    K_FACTOR = 35
    HOME_ADVANTAGE = 55

    headers = {"Authorization": f"Bearer {api_key}"}

    # Pulls data up to the current 2026 season dynamically
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
        except Exception as e:
            st.sidebar.warning(f"Error compiling history for {year}: {e}")

    return elo_ratings

def get_closest_elo(team_name, elo_dict):
    """Translates commercial betting names to match official statistical databases."""
    if not team_name:
        return 1500.0

    translation_map = {
        "Mississippi Rebels": "Ole Miss", "Mississippi St Bulldogs": "Mississippi State",
        "Texas A&M Aggies": "Texas A&M", "LSU Tigers": "LSU", "Alabama Crimson Tide": "Alabama",
        "Georgia Bulldogs": "Georgia", "Texas Longhorns": "Texas", "Tennessee Volunteers": "Tennessee",
        "Missouri Tigers": "Missouri", "Oklahoma Sooners": "Oklahoma", "Kentucky Wildcats": "Kentucky",
        "Auburn Tigers": "Auburn", "Florida Gators": "Florida", "South Carolina Gamecocks": "South Carolina",
        "Arkansas Razorbacks": "Arkansas", "Vanderbilt Commodores": "Vanderbilt",
        "Ohio State Buckeyes": "Ohio State", "Oregon Ducks": "Oregon", "Penn State Nittany Lions": "Penn State",
        "Michigan Wolverines": "Michigan", "USC Trojans": "USC", "Southern California Trojans": "USC",
        "Washington Huskies": "Washington", "Iowa Hawkeyes": "Iowa", "Wisconsin Badgers": "Wisconsin",
        "Nebraska Cornhuskers": "Nebraska", "Maryland Terrapins": "Maryland", "Minnesota Golden Gophers": "Minnesota",
        "Rutgers Scarlet Knights": "Rutgers", "Northwestern Wildcats": "Northwestern", "Illinois Fighting Illini": "Illinois",
        "Michigan State Spartans": "Michigan State", "UCLA Bruins": "UCLA", "Indiana Hoosiers": "Indiana", "Purdue Boilermakers": "Purdue",
        "Utah Utes": "Utah", "Kansas State Wildcats": "Kansas State", "Oklahoma State Cowboys": "Oklahoma State",
        "Arizona Wildcats": "Arizona", "Kansas Jayhawks": "Kansas", "Iowa State Cyclones": "Iowa State",
        "West Virginia Mountaineers": "West Virginia", "TCU Horned Frogs": "TCU", "Texas Tech Red Raiders": "Texas Tech",
        "UCF Knights": "UCF", "BYU Cougars": "BYU", "Colorado Buffaloes": "Colorado", "Baylor Bears": "Baylor",
        "Houston Cougars": "Houston", "Arizona State Sun Devils": "Arizona State", "Cincinnati Bearcats": "Cincinnati",
        "Florida State Seminoles": "Florida State", "Clemson Tigers": "Clemson", "Miami Hurricanes": "Miami",
        "Louisville Cardinals": "Louisville", "NC State Wolfpack": "NC State", "SMU Mustangs": "SMU",
        "Virginia Tech Hokies": "Virginia Tech", "North Carolina Tar Heels": "North Carolina", "Georgia Tech Yellow Jackets": "Georgia Tech",
        "California Golden Bears": "California", "Duke Blue Devils": "Duke", "Syracuse Orange": "Syracuse",
        "Pittsburgh Panthers": "Pittsburgh", "Boston College Eagles": "Boston College", "Wake Forest Demon Deacons": "Wake Forest",
        "Virginia Cavaliers": "Virginia", "Stanford Cardinal": "Stanford", "Notre Dame Fighting Irish": "Notre Dame",
        "Boise State Broncos": "Boise State", "Liberty Flames": "Liberty", "James Madison Dukes": "James Madison",
        "Memphis Tigers": "Memphis", "Tulane Green Wave": "Tulane", "UNLV Rebels": "UNLV"
    }

    if team_name in translation_map:
        return elo_dict.get(translation_map[team_name], 1500.0)
    if team_name in elo_dict:
        return elo_dict[team_name]

    # Clean up standard suffix formatting gaps
    clean_name = team_name
    mascots = ["Buckeyes", "Trojans", "Rebels", "Tigers", "Bulldogs", "Crimson Tide", "Longhorns", "Volunteers", "Sooners", "Gators", "Wildcats", "Razorbacks", "Wolverines", "Ducks", "Huskies", "Hawkeyes", "Badgers"]
    for mascot in mascots:
        clean_name = clean_name.replace(mascot, "").strip()

    if clean_name in elo_dict:
        return elo_dict[clean_name]

    closest_matches = difflib.get_close_matches(team_name, list(elo_dict.keys()), n=1, cutoff=0.25)
    if closest_matches:
        return elo_dict.get(closest_matches[0], 1500.0)

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
    sport_keys = ["americanfootball_ncaaf"]

    for key in sport_keys:
        url = f"https://the-odds-api.com{key}/odds/"
        params = {"apiKey": api_key, "regions": "us", "markets": "spreads", "oddsFormat": "american"}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                combined_games.extend(response.json())
        except Exception as e:
            st.error(f"Network error linking to The Odds API: {e}")
    return combined_games

# =====================================================================
# 4. EDGE MATRIX COMPILATION ENGINE
# =====================================================================
def compile_betting_slate(games, elo_dict):
    processed_slate = []

    for game in games:
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        if not home_team or not away_team:
            continue

        market_spread = None
        bookmaker_used = "Consensus"

        bookmakers = game.get("bookmakers", [])
        if bookmakers and len(bookmakers) > 0:
            primary_book = bookmakers[0]
            bookmaker_used = primary_book.get("title", "Unknown")

            markets = primary_book.get("markets", [])
            spread_market = next((m for m in markets if m.get("key") == "spreads"), None)
            
            if spread_market:
                outcomes = spread_market.get("outcomes", [])
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        market_spread = float(outcome.get("point", 0.0))

        if market_spread is None:
            market_spread = 0.0
            bookmaker_used = "No Live Line"

        proj_home, proj_away, home_prob = generate_score_projection(home_team, away_team, elo_dict)
        
        if proj_home > proj_away:
            winner = home_team
            win_percentage = home_prob
        else:
            winner = away_team
            win_percentage = 1.0 - home_prob

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
