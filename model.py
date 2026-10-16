import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

# --------------------------------------------------
# 1. Load your data (replace with CFBD API pulls)
# --------------------------------------------------
def load_data():
    # Expected columns:
    # team, opponent, team_score, opp_score, spread, week, season, <stats...>
    df = pd.read_csv("cfb_games.csv")
    df = df.sort_values(["season", "week"])
    return df

# --------------------------------------------------
# 2. Build rolling features
# --------------------------------------------------
def build_features(df, stats, window=8):
    for stat in stats:
        df[f"{stat}_roll"] = (
            df.groupby("team")[stat]
            .transform(lambda x: x.shift(1).rolling(window).mean())
        )
    return df

# --------------------------------------------------
# 3. Train regression model to predict score differential
# --------------------------------------------------
def train_model(df, feature_cols):
    df = df.dropna(subset=feature_cols + ["score_diff"])

    X = df[feature_cols]
    y = df["score_diff"]

    model = Pipeline([
        ("scale", StandardScaler()),
        ("bag", BaggingRegressor(
            base_estimator=Lasso(alpha=0.01),
            n_estimators=50,
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model

# --------------------------------------------------
# 4. Predict margin for a matchup
# --------------------------------------------------
def predict_margin(model, team_row):
    X = team_row.values.reshape(1, -1)
    return float(model.predict(X)[0])

# --------------------------------------------------
# 5. Convert prediction to a pick
# --------------------------------------------------
def pick_against_spread(pred_margin, vegas_spread):
    diff = pred_margin - vegas_spread

    if diff > 3:
        return f"✅ Pick FAVORITE (model margin {pred_margin:.1f} vs spread {vegas_spread})"
    elif diff < -3:
        return f"✅ Pick UNDERDOG (model margin {pred_margin:.1f} vs spread {vegas_spread})"
    else:
        return f"⚠️ No edge (model margin {pred_margin:.1f} vs spread {vegas_spread})"

# --------------------------------------------------
# 6. Full pipeline
# --------------------------------------------------
def run_pipeline():
    df = load_data()

    # Example stats — replace with your real features
    stats = ["yards_per_play", "success_rate", "epa"]

    df = build_features(df, stats)

    df["score_diff"] = df["team_score"] - df["opp_score"]

    feature_cols = [f"{s}_roll" for s in stats]

    model = train_model(df, feature_cols)

    return model, feature_cols
