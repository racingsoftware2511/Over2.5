import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

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

def normalize_pct(series: pd.Series) -> pd.Series:
    """
    Force values to % in [0,100].
    Accepts inputs like 0.75 -> 75, 75.5 -> 75.5, 7550 -> 75.5, 10000 -> 100.
    """
    s = pd.to_numeric(series, errors="coerce")

    def fix(v):
        if pd.isna(v):
            return v
        v = float(v)
        if v <= 1.0:            # proportions
            v *= 100.0
        while v > 100.0:        # scale down 7550 -> 755 -> 75.5
            v /= 10.0
        return v

    return s.map(fix)

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
# TAB 1: Over 2.5 Tips  (Z odds 1.40–3.00, BP ≥ 3.0, BQ ≥ 60, min(CE,CF) ≥ 35)
# Columns by Excel letter:
#   Z  = Over 2.5 Odds
#   BP = Combined GS (GS+GC proxy)
#   BQ = Combined 2.5 (percentage-like; we normalize to 0–100)
#   CE = Attacking Potential (Home)  [share; 0–100]
#   CF = Attacking Potential (Away)  [share; 0–100]
# --------------------------------------------------------------------
with tab1:
    st.subheader("Over 2.5 Tips (Strategy1)")

    IDX_Z  = excel_col_to_idx("Z")
    IDX_BP = excel_col_to_idx("BP")
    IDX_BQ = excel_col_to_idx("BQ")
    IDX_CE = excel_col_to_idx("CE")
    IDX_CF = excel_col_to_idx("CF")

    # sanity check: do we have the needed columns?
    needed_max = max(IDX_Z, IDX_BP, IDX_BQ, IDX_CE, IDX_CF)
    if len(df.columns) <= needed_max:
        st.error("Not enough columns for Z / BP / BQ / CE / CF. Please export the full SPM file.")
    else:
        col_Z  = df.columns[IDX_Z]     # Over 2.5 Odds
        col_BP = df.columns[IDX_BP]    # Combined GS
        col_BQ = df.columns[IDX_BQ]    # Combined 2.5
        col_CE = df.columns[IDX_CE]    # Attacking Potential (H) share
        col_CF = df.columns[IDX_CF]    # Attacking Potential (A) share

        work = df.copy()

        # Numbers
        work["O25_odds_Z"]    = to_num(work[col_Z])
        work["CombinedGS_BP"] = to_num(work[col_BP])
        work["Combined25_BQ"] = normalize_pct(work[col_BQ])     # robust to 0–1, 0–10000, etc.

        # Attacking shares (leave as 0–100 shares; DO NOT normalize to % again)
        work["Attack_H_CE"]   = to_num(work[col_CE])
        work["Attack_A_CF"]   = to_num(work[col_CF])
        work["Attack_min"]    = pd.concat([work["Attack_H_CE"], work["Attack_A_CF"]], axis=1).min(axis=1)

        # Build Kickoff timestamp if the date/time columns exist
        work = add_kickoff(work, col_date, col_time)

        # ---- Apply your four rules ----
        filt = (
            work["O25_odds_Z"].between(1.40, 3.00, inclusive="both") &
            (work["CombinedGS_BP"] >= 3.0) &
            (work["Combined25_BQ"] >= 60.0) &
            (work["Attack_min"] >= 35.0)
        )

        tips = work.loc[filt].copy()

        # Columns to show
        show_cols = []
        if col_country: show_cols.append(col_country)
        show_cols += ["Kickoff"]
        if col_home: show_cols.append(col_home)
        if col_away: show_cols.append(col_away)
        show_cols += ["O25_odds_Z", "CombinedGS_BP", "Combined25_BQ", "Attack_H_CE", "Attack_A_CF"]

        if tips.empty:
            st.warning("No matches met the Over 2.5 rules.")
        else:
            # Sort by strongest signals first (BQ then CombinedGS)
            tips = tips.sort_values(["Combined25_BQ", "CombinedGS_BP"], ascending=False).reset_index(drop=True)
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
        work2["Pred_BY"]      = normalize_pct(to_num(work2[col_BY]))  # normalized
        work2["Pred_BZ"]      = normalize_pct(to_num(work2[col_BZ]))  # normalized
        work2["Pred_CA"]      = normalize_pct(to_num(work2[col_CA]))  # normalized
        work2["Attack_CE"]    = to_num(work2[col_CE])
        work2["WinsGame_CO"]  = to_num(work2[col_CO])

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
            tips2 = tips2.sort_values(["Pred_BY", "Pred_BZ", "MatchVol_W"], ascending=False).reset_index(drop=True)
            top2 = tips2.head(10)
            st.success(f"SPM Tips (Home Fav) — Top {len(top2)}")
            st.dataframe(top2[show2], use_container_width=True, height=500)

            # Per-tab CSV
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
    st.dataframe(combined, use_container_width=True, height=500)  # optional preview
else:
    st.info("Generate tips in the tabs above to enable the combined download.")
