import streamlit as st
import requests
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime

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
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.sidebar.error(f"CFBD API Error: Status {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"CFBD Connection Error: {str(e)}")
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
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.sidebar.error(f"Odds API Error: Status {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"Odds Connection Error: {str(e)}")
    return []

# Fetch Team Records and Stats from CFBD to power the model
@st.cache_data(ttl=1800)
def get_team_stats_dict(year=2024):
    stats_data = get_cfbd_data("/stats/season", {"year": year})
    if not stats_data:
        st.sidebar.warning("⚠️ No seasonal stats retrieved from CFBD API. Using fallback model baselines.")
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

# Standardize team names to fix API mismatches
def clean_team_name(name):
    if not name:
        return ""
    removals = ["buckeyes", "tigers", "longhorns", "crimson", "tide", "aggies", "gators",
                "seminoles", "wolverines", "badgers", "bruins", "trojans", "sooner",
                "sooners", "railriders", "state", "st", "univ", "university", "&", "and"]
    cleaned = name.lower()
    for r in removals:
        cleaned = cleaned.replace(f" {r}", " ").replace(f"{r} ", " ")
    return "".join(c for c in cleaned if c.isalnum()).strip()

# Step 3: Advanced token-based team matching to avoid wrong identity mappings
def find_matching_team(target_name, team_stats):
    target_cleaned = clean_team_name(target_name)
    if not target_cleaned:
        return None

    if target_name in team_stats:
        return target_name

    best_match = None
    best_score = 0.0
    target_tokens = set(target_cleaned.split())

    for real_name in team_stats.keys():
        real_cleaned = clean_team_name(real_name)
        if real_cleaned == target_cleaned:
            return real_name

        real_tokens = set(real_cleaned.split())
        if target_tokens and real_tokens:
            intersection = target_tokens.intersection(real_tokens)
            score = len(intersection) / float(max(len(target_tokens), len(real_tokens)))
            if score > best_score and (target_cleaned in real_cleaned or real_cleaned in target_cleaned):
                best_score = score
                best_match = real_name

    return best_match if best_score > 0.4 else None

# Calculate dynamic FBS average stats from the team stats dictionary
def calculate_fbs_baseline_averages(team_stats, default_off=28.3, default_def=28.3):
    total_off_sum = 0.0
    total_def_sum = 0.0
    total_off_yards = 0.0
    total_def_yards = 0.0
    total_off_plays = 0.0
    total_def_plays = 0.0
    team_count = 0

    for team, stats_entry in team_stats.items():
        try:
            games = float(stats_entry.get("games", 12))
            if games <= 0:
                games = 12.0

            points = float(stats_entry.get("points", 0))
            points_allowed = float(stats_entry.get("pointsAllowed", 0))
            off_yds = float(stats_entry.get("totalYards", 0))
            def_yds = float(stats_entry.get("opponentYards", 0))
            off_plays = float(stats_entry.get("offensePlays", 70 * games))
            def_plays = float(stats_entry.get("defensePlays", 70 * games))

            if points > 0 or points_allowed > 0:
                total_off_sum += (points / games)
                total_def_sum += (points_allowed / games)
                total_off_yards += (off_yds / games)
                total_def_yards += (def_yds / games)
                total_off_plays += (off_plays / games)
                total_def_plays += (def_plays / games)
                team_count += 1
        except (ValueError, TypeError):
            continue

    if team_count > 0:
        return (
            total_off_sum / team_count,
            total_def_sum / team_count,
            total_off_yards / team_count,
            total_def_yards / team_count,
            total_off_plays / team_count,
            total_def_plays / team_count
        )
    return default_off, default_def, 401.5, 398.2, 72.0, 71.5

# Helper to calculate and retrieve a team's per-game stats
def get_team_averages(team_name, team_stats, baseline_off, baseline_def, baseline_oyg, baseline_oplays):
    matched_name = find_matching_team(team_name, team_stats)
    if not matched_name:
        hash_offset = (hash(team_name) % 10) - 5
        temp_off = max(14.0, baseline_off + (hash_offset * 1.5))
        temp_def = max(14.0, baseline_def - (hash_offset * 1.2))
        return temp_off, temp_def, baseline_oyg, baseline_oyg, 5.5, 5.5

    stats_entry = team_stats.get(matched_name, {})
    try:
        games = float(stats_entry.get("games", 12))
    except (ValueError, TypeError):
        games = 12.0
    if games <= 0:
        games = 12.0

    off_ppg = float(stats_entry.get("points", baseline_off * games)) / games
    def_ppg = float(stats_entry.get("pointsAllowed", baseline_def * games)) / games
    off_ypg = float(stats_entry.get("totalYards", baseline_oyg * games)) / games
    def_ypg = float(stats_entry.get("opponentYards", baseline_oyg * games)) / games

    off_plays = float(stats_entry.get("offensePlays", baseline_oplays * games)) / games
    def_plays = float(stats_entry.get("defensePlays", baseline_oplays * games)) / games

    if off_plays > 0:
        off_ypp = off_ypg / off_plays
        off_ypp = off_ypp * 0.745
    else:
        off_ypp = 5.5

    if def_plays > 0:
        def_ypp = def_ypg / def_plays
        def_ypp = def_ypp * 0.745
    else:
        def_ypp = 5.5

    return off_ppg, def_ppg, off_ypg, def_ypg, off_ypp, def_ypp

# Fetch game logs
@st.cache_data(ttl=1800)
def fetch_recent_game_logs(year=2024):
    return get_cfbd_data("/games", {
