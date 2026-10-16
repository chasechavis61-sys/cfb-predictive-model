import streamlit as st
import requests
import pandas as pd
import numpy as np
import difflib

# =====================================================================
# 1. API CONFIGURATION
# =====================================================================
THE_ODDS_API_KEY = "875f2f3bcab472c2046d7efb6659a7c3"

# =====================================================================
# 2. SEEDED POWER INDEX MATRIX (Replaces Broken CFBD Key Loop)
# =====================================================================
def get_compiled_elo_ratings():
    """
    Seeded baseline Elo power distribution matrix. Bypasses CFBD v2 
    API key limit failures while preserving real-world program variance.
    
"""
    return {
        # SEC
        "Georgia": 1780.0, "Alabama": 1755.0, "Texas": 1745.0, "Ole Miss": 1695.0,
        "Tennessee": 1685.0, "LSU": 1665.0, "Missouri": 1635.0, "Oklahoma": 1625.0,
        "Texas A&M": 1615.0, "Kentucky": 1565.0, "Auburn": 1545.0, "Florida": 1535.0,
        "South Carolina": 1525.0, "Arkansas": 1515.0, "Mississippi State": 1455.0, "Vanderbilt": 1415.0,
        # Big Ten
        "Ohio State": 1775.0, "Oregon": 1735.0, "Penn State": 1715.0, "Michigan": 1685.0,
        "USC": 1645.0, "Washington": 1625.0, "Iowa": 1595.0, "Wisconsin": 1575.0,
        "Nebraska": 1555.0, "Maryland": 1535.0, "Minnesota": 1525.0, "Rutgers": 1515.0,
        "Northwestern": 1495.0, "Illinois": 1485.0, "Michigan State": 1475.0, "UCLA": 1465.0,
        "Indiana": 1455.0, "Purdue": 1425.0,
        # Big 12
        "Utah": 1655.0, "Kansas State": 1635.0, "Oklahoma State": 1625.0, "Arizona": 1605.0,
        "Kansas": 1585.0, "Iowa State": 1575.0, "West Virginia": 1555.0, "TCU": 1545.0,
        "Texas Tech": 1535.0, "UCF": 1525.0, "BYU": 1515.0, "Colorado": 1495.0,
        "Baylor": 1485.0, "Houston": 1445.0, "Arizona State": 1435.0, "Cincinnati": 1425.0,
        # ACC
        "Florida State": 1665.0, "Clemson": 1655.0, "Miami": 1645.0, "Louisville": 1615.0,
        "NC State": 1585.0, "SMU": 1575.0, "Virginia Tech": 1565.0, "North Carolina": 1545.0,
        "Georgia Tech": 1535.0, "California": 1515.0, "Duke": 1505.0, "Syracuse": 1495.0,
        "Pittsburgh": 1485.0, "Boston College": 1475.0, "Wake Forest": 1465.0, "Virginia": 1435.0,
        "Stanford": 1415.0,
        # Group of 5 Standouts
        "Notre Dame": 1695.0, "Liberty": 1565.0, "Boise State": 1555.0, "James Madison": 1545.0,
        "Memphis": 1535.0, "Tulane": 1525.0, "Fresno State": 1505.0, "Appalachian State": 1495.0
    }

def get_closest_elo(team_name, elo_dict):
    """Maps commercial betting nicknames cleanly into structural statistical records."""
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

    # Clean mascot strings dynamically
    clean_name = team_name
    mascots = ["Buckeyes", "Trojans", "Rebels", "Tigers", "Bulldogs", "Crimson Tide", "Longhorns", "Volunteers", "Sooners", "Gators", "Wildcats", "Razorbacks", "Wolverines", "Ducks", "Huskies", "Hawkeyes", "Badgers"]
    for msc in mascots:
        clean_name = clean_name.replace(msc, "").strip()

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
@st.cache_data(ttl=1800)
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
            st.error(f"Live Market Connection Exception: {e}")
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
            "Live Vegas Spread": f"{home_team} {market_spread}
