import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.title("📊 SPM Top 5 Over 2.5 Picks (Liquidity ≥ 500 in Column AB)")

uploaded_file = st.file_uploader("Upload your SPM Excel file", type=["xlsx"])

def poisson_o25_prob_from_lambda(lam):
    p0 = np.exp(-lam)
    p1 = p0 * lam
    p2 = p1 * lam / 2.0
    return 1.0 - (p0 + p1 + p2)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="Matches")

        # Column mappings
        col_home = "Home"
        col_away = "Away"
        col_country = "Country"
        col_date = "Date"
        col_time = "Time"
        col_odds = "O2.5 Back(T0)"
        col_lam = "Total Goals"
        col_liq = df.columns[27]  # Column AB

        # Process
        df["o25_odds"] = pd.to_numeric(df[col_odds], errors="coerce")
        df["lam"] = pd.to_numeric(df[col_lam], errors="coerce")
        df["liq"] = pd.to_numeric(df[col_liq], errors="coerce")

        df = df.dropna(subset=["o25_odds", "lam", "liq"])
        df = df[(df["o25_odds"] >= 1.60) & (df["o25_odds"] <= 2.60)]
        df = df[df["liq"] >= 500]

        # Calculate probabilities & edge
        df["p_model"] = poisson_o25_prob_from_lambda(df["lam"])
        df["p_market"] = 1.0 / df["o25_odds"]
        df["edge"] = df["p_model"] - df["p_market"]
        df["Edge_%"] = (df["edge"] * 100).round(1)

        # Create kickoff datetime
        df["Kickoff"] = pd.to_datetime(
            df[col_date].astype(str) + " " + df[col_time].astype(str),
            errors="coerce"
        )

        # Get top 5
        result = df.sort_values(by="edge", ascending=False).head(5)
        result = result[
            [col_country, "Kickoff", col_home, col_away, "o25_odds",
             "p_model", "p_market", "Edge_%", "liq"]
        ]

        st.success(f"Found {len(result)} picks")
        st.dataframe(result)

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Load and display logo
logo = Image.open("spm_logo.png")  # Make sure this file is in your GitHub repo
st.image(logo, width=200)
st.markdown("<h1 style='text-align: center; color: green;'>SPM Soccer Price Monitor – AI Agent</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Top 5 Over 2.5 Picks</h4>", unsafe_allow_html=True)

