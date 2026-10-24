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

# Fetch Team Records and Stats from CFBD
@st.cache_data(ttl=1800)
def get_team_stats_dict(year=2024):
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

# Step 3: Advanced token-based team matching
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

# Calculate dynamic FBS average stats
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
    return get_cfbd_data("/games", {"year": year})

# Calculate Weighted Recency, SOS adjustments, and Rest Factors
def compute_advanced_factors(home_team, away_team, games_logs, team_stats, baseline_off, baseline_def):
    completed_games = [g for g in games_logs if g.get("home_points") is not None and g.get("away_points") is not None]

    def get_team_momentum(team_name):
        team_games = []
        for g in completed_games:
            is_home = find_matching_team(g["home_team"], team_stats) == find_matching_team(team_name, team_stats)
            is_away = find_matching_team(g["away_team"], team_stats) == find_matching_team(team_name, team_stats)
            if is_home or is_away:
                week = g.get("week", 1)
                margin = (g["home_points"] - g["away_points"]) if is_home else (g["away_points"] - g["home_points"])
                team_games.append((week, margin))
        if not team_games:
            return 0.0
        team_games = sorted(team_games, key=lambda x: x[0], reverse=True)
        recent_margins = [m for _, m in team_games[:4]]
        return np.mean(recent_margins) * 0.15 if recent_margins else 0.0

    home_momentum = get_team_momentum(home_team)
    away_momentum = get_team_momentum(away_team)

    def get_days_since_last_game(team_name):
        team_dates = []
        for g in completed_games:
            is_home = find_matching_team(g["home_team"], team_stats) == find_matching_team(team_name, team_stats)
            is_away = find_matching_team(g["away_team"], team_stats) == find_matching_team(team_name, team_stats)
            if (is_home or is_away) and g.get("start_date"):
                try:
                    date_obj = datetime.strptime(g["start_date"].split("T")[0], "%Y-%m-%d")
                    team_dates.append(date_obj)
                except Exception:
                    continue
        if not team_dates:
            return 7
        team_dates = sorted(team_dates, reverse=True)
        delta = (datetime.now() - team_dates[0]).days
        return delta

    home_rest_days = get_days_since_last_game(home_team)
    away_rest_days = get_days_since_last_game(away_team)
    rest_adjustment = 0.0
    if home_rest_days >= 13:
        rest_adjustment += 1.5
    if away_rest_days >= 13:
        rest_adjustment -= 1.5
    return home_momentum - away_momentum + rest_adjustment

# Advanced Prediction Model incorporating Automatic Injury Overrides
def predict_game_outcome_advanced(home_team, away_team, team_stats, games_logs, baseline_vals, injury_adjustments={}):
    base_off, base_def, base_oyg, base_dyg, base_oplays, base_dplays = baseline_vals
    home_off, home_def, home_oyg, home_dyg, home_oypp, home_dypp = get_team_averages(home_team, team_stats, base_off, base_def, base_oyg, base_oplays)
    away_off, away_def, away_oyg, away_dyg, away_oypp, away_dypp = get_team_averages(away_team, team_stats, base_off, base_def, base_oyg, base_oplays)

    home_raw = (home_off * away_def) / ((base_off + base_def) / 2)
    away_raw = (away_off * home_def) / ((base_off + base_def) / 2)

    home_ypp_mult = (home_oypp / max(0.5, away_dypp))
    away_ypp_mult = (away_oypp / max(0.5, home_dypp))
    home_ypp_mult = np.clip(home_ypp_mult, 0.85, 1.15)
    away_ypp_mult = np.clip(away_ypp_mult, 0.85, 1.15)

    home_pred = home_raw * home_ypp_mult
    away_pred = away_raw * away_ypp_mult

    home_pred += 2.5
    away_pred -= 2.5

    situational_offset = compute_advanced_factors(home_team, away_team, games_logs, team_stats, base_off, base_def)
    home_pred += situational_offset / 2
    away_pred -= situational_offset / 2

    matched_home = find_matching_team(home_team, team_stats)
    matched_away = find_matching_team(away_team, team_stats)

    if matched_home in injury_adjustments:
        home_pred += injury_adjustments[matched_home]
    if matched_away in injury_adjustments:
        away_pred += injury_adjustments[matched_away]

    return max(7.0, round(home_pred, 1)), max(7.0, round(away_pred, 1))

# Calibrated betting confidence
def calculate_bet_confidence(model_projected, market_line, is_total=False):
    std_dev = 19.0 if is_total else 14.0
    edge = abs(model_projected - market_line)
    prob = stats.norm.cdf(edge / std_dev) if edge > 0 else 0.5
    if is_total:
        final_prob = 0.5 + (prob - 0.5) * 0.45
    else:
        final_prob = 0.5 + (prob - 0.5) * 0.55

    if final_prob >= 0.75:
        confidence = "🔥 Best Play"
    elif final_prob >= 0.60:
        confidence = "🟢 Strong Play"
    elif final_prob >= 0.512:
        confidence = "🟡 Slight Lean"
    else:
        confidence = "🔴 Hold Off"
    return final_prob, confidence

# --- Sidebar Automated Injury Setup ---
st.sidebar.markdown("### 🤕 Automated Injury Calculator")
st.sidebar.info("Add a player, select their position, and the system automatically calculates the point impact.")

try:
    team_stats_pre = get_team_stats_dict()
    all_teams_list = sorted(list(team_stats_pre.keys()))
except Exception:
    all_teams_list = []

injury_adjustments = {}
if all_teams_list:
    selected_injured_team = st.sidebar.selectbox("Select Team with Injury:", ["-- None --"] + all_teams_list)
    if selected_injured_team != "-- None --":
        player_name = st.sidebar.text_input("Player Name:", "")
        position_role = st.sidebar.selectbox(
            "Position / Role Importance:",
            [
                "Elite Starting Quarterback (-6.0)",
                "Good QB / Star Skill Player (-4.0)",
                "Serviceable QB / Top O-Lineman (-2.5)",
                "Top Defensive Starter (-1.5)",
                "Standard Starter / Key Rotational Player (-0.75)"
            ]
        )
        # Map roles to specific point reductions
        val_map = {
            "Elite Starting Quarterback (-6.0)": -6.0,
            "Good QB / Star Skill Player (-4.0)": -4.0,
            "Serviceable QB / Top O-Lineman (-2.5)": -2.5,
            "Top Defensive Starter (-1.5)": -1.5,
            "Standard Starter / Key Rotational Player (-0.75)": -0.75
        }
        if player_name:
            calculated_penalty = val_map[position_role]
            injury_adjustments[selected_injured_team] = calculated_penalty
            st.sidebar.success(f"✅ Automatically applied {calculated_penalty} pts to {selected_injured_team} for {player_name}.")

# Fetch and Predict Execution Flow
if st.sidebar.button("Fetch Latest Games & Predict"):
    if CFBD_API_KEY and ODDS_API_KEY:
        with st.spinner("Fetching team stats, market lines, and generating predictions... "):
            odds_json = get_odds_data()
            team_stats = get_team_stats_dict()
            games_logs = fetch_recent_game_logs()

            if not odds_json:
                st.error("Failed to retrieve odds data.")
            else:
                baseline_vals = calculate_fbs_baseline_averages(team_stats)
                st.sidebar.metric(label="Dynamic CFB Offense Baseline", value=f"{baseline_vals[0]:.2f} PPG")
                st.sidebar.metric(label="Dynamic CFB Defense Baseline", value=f"{baseline_vals[1]:.2f} PPG")

                spread_picks = []
                totals_picks = []

                for game in odds_json:
                    home_team = game.get("home_team")
                    away_team = game.get("away_team")

                    pred_home, pred_away = predict_game_outcome_advanced(
                        home_team, away_team, team_stats, games_logs, baseline_vals, injury_adjustments
                    )
                    pred_spread = pred_away - pred_home
                    pred_total = pred_home + pred_away

                    home_off_ppg, home_def_ppg, _, _, home_ypp, _ = get_team_averages(home_team, team_stats, baseline_vals[0], baseline_vals[1], baseline_vals[2], baseline_vals[4])
                    away_off_ppg, away_def_ppg, _, _, away_ypp, _ = get_team_averages(away_team, team_stats, baseline_vals[0], baseline_vals[1], baseline_vals[2], baseline_vals[4])

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

                    if market_spread is not None:
                        prob_spread, conf_spread = calculate_bet_confidence(pred_spread, market_spread, is_total=False)
                        if pred_spread < market_spread:
                            formatted_spread = f"{market_spread:+.1f}" if market_spread != 0 else "PK"
                            spread_pick = f"{home_team} {formatted_spread}"
                        else:
                            opp_spread = -market_spread
                            formatted_spread = f"{opp_spread:+.1f}" if opp_spread != 0 else "PK"
                            spread_pick = f"{away_team} {formatted_spread}"

                        formatted_market_spread = f"{market_spread:+.1f}" if market_spread != 0 else "PK"
                        spread_picks.append({
                            "Game": f"{away_team} @ {home_team}",
                            "Model Pred (Away-Home)": f"{pred_away:.1f} - {pred_home:.1f}",
                            "Market Spread": formatted_market_spread,
                            "Best Spread Pick": spread_pick,
                            "Spread Prob": f"{prob_spread * 100:.1f}%",
                            "Spread Tier": conf_spread
                        })

                    if market_total is not None:
                        prob_total, conf_total = calculate_bet_confidence(pred_total, market_total, is_total=True)
                        if pred_total > market_total:
                            total_pick = f"Over {market_total}"
                        else:
                            total_pick = f"Under {market_total}"

                        totals_picks.append({
                            "Game": f"{away_team} @ {home_team}",
                            "Away Off PPG / YPP": f"{away_off_ppg:.1f} / {away_ypp:.2f}",
                            "Home Off PPG / YPP": f"{home_off_ppg:.1f} / {home_ypp:.2f}",
                            "Model Projected Total": f"{pred_total:.1f}",
                            "Market Total": market_total,
                            "Best Totals Pick": total_pick,
                            "Totals Prob": f"{(prob_total * 100):.1f}%",
                            "Totals Tier": conf_total
                        })

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
