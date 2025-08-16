import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
st.set_page_config(layout="wide")   # ✅ Add this line here

# =========================
# Branding
# =========================
try:
    st.image(Image.open("spm_logo.png"), width=180)
except Exception:
    pass

st.markdown(
    "<h1 style='text-align:center;color:green;'>SPM Tips – Soccer Price Monitor • AI Agent</h1>",
    unsafe_allow_html=True,
)
st.write("Upload your SPM Excel and pick a strategy to generate tips.")

# =========================
# Helpers
# =========================
def excel_col_to_idx(col_letters: str) -> int:
    """Excel letters -> 0-based index (e.g., AB -> 27)."""
    s = 0
    for ch in col_letters.upper():
        s = s * 26 + (ord(ch) - 64)
    return s - 1

def pick_col(df, candidates):
    """Find first matching column name (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def to_num(series):
    return pd.to_numeric(series, errors="coerce")

def add_kickoff(frame, col_date, col_time):
    if col_date and col_time:
        frame["Kickoff"] = pd.to_datetime(
            frame[col_date].astype(str) + " " + frame[col_time].astype(str),
            errors="coerce",
        )
    else:
        frame["Kickoff"] = pd.NaT
    return frame

# =========================
# Upload
# =========================
uploaded = st.file_uploader("Upload your SPM Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.stop()

try:
    df = pd.read_excel(uploaded, sheet_name="Matches")
except Exception as e:
    st.error(f"Could not open sheet **Matches**: {e}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

with st.expander("Detected columns in your sheet"):
    st.write(df.columns.tolist())

# Common display columns (optional)
col_home    = pick_col(df, ["Home", "Home Team"])
col_away    = pick_col(df, ["Away", "Away Team"])
col_country = pick_col(df, ["Country", "League", "Competition"])
col_date    = pick_col(df, ["Date"])
col_time    = pick_col(df, ["Time"])

# =========================
# Tabs (Strategies)
# =========================
tab1, tab2 = st.tabs(["⚽ Over 2.5 Tips", "🏠 Home Fav Tips"])

# --------------------------------------------------------------------
# TAB 1: Over 2.5 Tips
# Rules: BQ ≥ 60, AB ≥ 500 (AUD), BP ≥ 2.5
# Columns by Excel letter:
#   AB = Volume, BP = Combined GS+GC, BQ = Combined 2.5
# --------------------------------------------------------------------
with tab1:
    st.subheader("Over 2.5 Tips (BQ ≥ 60, AB ≥ 500, BP ≥ 2.5)")

    IDX_AB = excel_col_to_idx("AB")
    IDX_BP = excel_col_to_idx("BP")
    IDX_BQ = excel_col_to_idx("BQ")

    max_needed = max(IDX_AB, IDX_BP, IDX_BQ)
    if len(df.columns) <= max_needed:
        st.error("Not enough columns for AB / BP / BQ. Please export the full SPM file.")
    else:
        col_AB = df.columns[IDX_AB]  # Volume (AUD)
        col_BP = df.columns[IDX_BP]  # Combined GS+GC
        col_BQ = df.columns[IDX_BQ]  # Combined 2.5

        work = df.copy()
        work["Volume_AB"]     = to_num(work[col_AB])
        work["GS_GC_BP"]      = to_num(work[col_BP])
        work["Combined25_BQ"] = to_num(work[col_BQ])

        # If BQ looks like 0–1, convert to %
        if work["Combined25_BQ"].median(skipna=True) <= 1:
            work["Combined25_BQ"] *= 100.0

        # Apply your rules
        filt = (
            (work["Combined25_BQ"] >= 60.0) &
            (work["Volume_AB"] >= 500.0) &
            (work["GS_GC_BP"] >= 2.5)
        )
        tips = work.loc[filt].copy()
        tips = add_kickoff(tips, col_date, col_time)

        # Optional odds column to show if present
        col_odds_o25 = pick_col(df, ["O2.5 Back(T0)", "Over 2.5 Back", "Over2.5 Back"])

        show_cols = []
        if col_country: show_cols.append(col_country)
        show_cols += ["Kickoff"]
        if col_home: show_cols.append(col_home)
        if col_away: show_cols.append(col_away)
        if col_odds_o25: show_cols.append(col_odds_o25)
        show_cols += ["Combined25_BQ", "GS_GC_BP", "Volume_AB"]

        if tips.empty:
            st.warning("No matches met the Over 2.5 rules.")
        else:
            # --- Top‑N slider + clean sort ---
            tips = tips.sort_values("Combined25_BQ", ascending=False).reset_index(drop=True)
            top_n = st.slider("How many tips to show?", 5, 50, 10)  # default 10
            top = tips.head(top_n)

            st.success(f"SPM Tips (Over 2.5) — Top {len(top)}")
            st.dataframe(top[show_cols], use_container_width=True, height=500)

            # Per‑tab CSV
            csv1 = top[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Over 2.5 SPM Tips (CSV)",
                data=csv1,
                file_name="SPM_Tips_Over25.csv",
                mime="text/csv",
                key="dl_over25_csv",
            )

            # Save for combined download
            st.session_state["tips_over25"] = top[show_cols].assign(Strategy="Over 2.5")
# --------------------------------------------------------------------
# TAB 2: Home Fav Tips
# Your rules (AUD only for W):
# 1) Home Fav Odds (home odds) between 2.00 and 5.00
# 2) Number of home games (BU) > 5
# 3) Match Volume (W) ≥ 5,000 AUD
# 4) Predictions: BY ≥ 45%, BZ ≥ 45%, CA ≥ 10%
# 5) Attacking Potential (CE) ≥ 60
# 6) Wins The Game (CO) ≥ 60
# --------------------------------------------------------------------
with tab2:
    st.subheader("Home Fav Tips (6 Rules • AUD)")

    col_home_odds = pick_col(df, ["Home Back(T0)", "Home Odds", "Home Back(TO)", "Home Back"])

    IDX_W  = excel_col_to_idx("W")   # Match Volume (AUD)
    IDX_BU = excel_col_to_idx("BU")  # # home games
    IDX_BY = excel_col_to_idx("BY")  # Prediction %
    IDX_BZ = excel_col_to_idx("BZ")  # Prediction %
    IDX_CA = excel_col_to_idx("CA")  # Prediction %
    IDX_CE = excel_col_to_idx("CE")  # Attacking potential
    IDX_CO = excel_col_to_idx("CO")  # Wins The Game

    need = max(IDX_W, IDX_BU, IDX_BY, IDX_BZ, IDX_CA, IDX_CE, IDX_CO)
    if len(df.columns) <= need:
        st.error("Not enough columns for W / BU / BY / BZ / CA / CE / CO.")
    else:
        col_W  = df.columns[IDX_W]
        col_BU = df.columns[IDX_BU]
        col_BY = df.columns[IDX_BY]
        col_BZ = df.columns[IDX_BZ]
        col_CA = df.columns[IDX_CA]
        col_CE = df.columns[IDX_CE]
        col_CO = df.columns[IDX_CO]

        work2 = df.copy()
        work2["HomeOdds"]     = to_num(work2[col_home_odds]) if col_home_odds else np.nan
        work2["HomeGames_BU"] = to_num(work2[col_BU])
        work2["MatchVol_W"]   = to_num(work2[col_W])
        work2["Pred_BY"]      = to_num(work2[col_BY])
        work2["Pred_BZ"]      = to_num(work2[col_BZ])
        work2["Pred_CA"]      = to_num(work2[col_CA])
        work2["Attack_CE"]    = to_num(work2[col_CE])
        work2["WinsGame_CO"]  = to_num(work2[col_CO])

        # Normalize predictions to % if 0–1
        for pcol in ["Pred_BY", "Pred_BZ", "Pred_CA"]:
            if work2[pcol].median(skipna=True) <= 1:
                work2[pcol] *= 100.0

        work2 = add_kickoff(work2, col_date, col_time)

        mask = (
            (work2["HomeOdds"].between(2.00, 5.00, inclusive="both") if col_home_odds else True) &
            (work2["HomeGames_BU"] > 5) &
            (work2["MatchVol_W"] >= 5000.0) &
            (work2["Pred_BY"] >= 45.0) &
            (work2["Pred_BZ"] >= 45.0) &
            (work2["Pred_CA"] >= 10.0) &
            (work2["Attack_CE"] >= 60.0) &
            (work2["WinsGame_CO"] >= 60.0)
        )

        tips2 = work2.loc[mask].copy()

        show2 = []
        if col_country: show2.append(col_country)
        show2 += ["Kickoff"]
        if col_home: show2.append(col_home)
        if col_away: show2.append(col_away)
        if col_home_odds: show2.append("HomeOdds")
        show2 += [
            "HomeGames_BU", "MatchVol_W", "Pred_BY", "Pred_BZ", "Pred_CA",
            "Attack_CE", "WinsGame_CO",
        ]

        if tips2.empty:
            st.warning("No matches met the Home Fav rules.")
        else:
            # Sort by strongest signal first (BY, BZ, then volume)
            tips2 = tips2.sort_values(["Pred_BY", "Pred_BZ", "MatchVol_W"], ascending=False).reset_index(drop=True)
            top2 = tips2.head(10)
            st.success(f"SPM Tips (Home Fav) — Top {len(top2)}")
            st.dataframe(top2[show2])

            # Per‑tab CSV
            csv2 = top2[show2].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Home Fav SPM Tips (CSV)",
                data=csv2,
                file_name="SPM_Tips_HomeFav.csv",
                mime="text/csv",
                key="dl_homefav_csv",
            )

            # Save for combined download
            st.session_state["tips_homefav"] = top2[show2].assign(Strategy="Home Fav")

# =========================
# Combined Download
# =========================
st.markdown("---")
st.subheader("📦 Download All SPM Tips (Combined)")

dfs = []
if "tips_over25" in st.session_state:
    dfs.append(st.session_state["tips_over25"])
if "tips_homefav" in st.session_state:
    dfs.append(st.session_state["tips_homefav"])

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    csv_all = combined.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download SPM Tips – All Strategies (CSV)",
        data=csv_all,
        file_name="SPM_Tips_All_Strategies.csv",
        mime="text/csv",
        key="dl_all_csv",
    )
    st.dataframe(combined)  # optional preview
else:
    st.info("Generate tips in the tabs above to enable the combined download.")
