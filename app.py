import streamlit as st
import pandas as pd
from model import run_pipeline, predict_margin, pick_against_spread

st.title("CFB Predictive Spread Model")

model, feature_cols = run_pipeline()

uploaded = st.file_uploader("Upload matchup CSV", type="csv")

if uploaded:
    games = pd.read_csv(uploaded)

    st.write("### Predictions")
    for _, row in games.iterrows():
        pred = predict_margin(model, row[feature_cols])
        pick = pick_against_spread(pred, row["spread"])

        st.write(f"**{row['team']} vs {row['opponent']}**")
        st.write(f"Predicted margin: {pred:.1f}")
        st.write(pick)
        st.write("---")
