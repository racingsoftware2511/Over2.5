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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚽ Over 2.5 Tips",
    "🏠 Home Fav Tips",
    "🔥 Over 2.5 (Based on Poisson)",
    "🚫 Lay the Draw",
    "✅ Back the Away"
])
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
                key="dl_over25_csv_t1",
            )

            # Save for combined download
            st.session_state["tips_over25"] = top[show_cols].assign(Strategy="Over 2.5")
# --------------------------------------------------------------------
# TAB 2: Home Fav Tips (Final rules)
# Rules:
# - Home Fav Odds between 2.00 and 4.00
# - Home games (BU) > 10
# - Attacking Potential (CE) ≥ 60
# - Wins The Game (CO) ≥ 60
# --------------------------------------------------------------------
with tab2:
    st.subheader("Home Fav Tips (Strategy2)")

    # Home odds column (by name; keep flexible)
    col_home_odds = pick_col(df, ["Home Back(T0)", "Home Odds", "Home Back(TO)", "Home Back"])

    # Excel letters we need
    IDX_BU = excel_col_to_idx("BU")  # # home games
    IDX_CE = excel_col_to_idx("CE")  # Attacking potential (share 0–100)
    IDX_CO = excel_col_to_idx("CO")  # Wins The Game

    needed_max = max(IDX_BU, IDX_CE, IDX_CO)
    if len(df.columns) <= needed_max:
        st.error("Not enough columns for BU / CE / CO. Please export the full SPM file.")
    else:
        col_BU = df.columns[IDX_BU]
        col_CE = df.columns[IDX_CE]
        col_CO = df.columns[IDX_CO]

        work2 = df.copy()
        work2["HomeOdds"]     = to_num(work2[col_home_odds]) if col_home_odds else np.nan
        work2["HomeGames_BU"] = to_num(work2[col_BU])
        work2["Attack_CE"]    = to_num(work2[col_CE])   # CE is already a 0–100 share
        work2["WinsGame_CO"]  = to_num(work2[col_CO])

        work2 = add_kickoff(work2, col_date, col_time)

        mask = (
            (work2["HomeOdds"].between(2.00, 4.00, inclusive="both") if col_home_odds else True) &
            (work2["HomeGames_BU"] > 10) &
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
        show2 += ["HomeGames_BU", "Attack_CE", "WinsGame_CO"]

        if tips2.empty:
            st.warning("No matches met the Home Fav rules.")
        else:
            # Sort strongest first
            tips2 = tips2.sort_values(
                ["WinsGame_CO", "Attack_CE", "HomeGames_BU"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            top2 = tips2.head(10)
            st.success(f"SPM Tips (Home Fav) — Top {len(top2)}")
            st.dataframe(top2[show2], use_container_width=True, height=500)

            # CSV download
            csv2 = top2[show2].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Home Fav SPM Tips (CSV)",
                data=csv2,
                file_name="SPM_Tips_HomeFav.csv",
                mime="text/csv",
                key="dl_homefav_csv_t2",
            )

            # Save for combined download
            st.session_state["tips_homefav"] = top2[show2].assign(Strategy="Home Fav")
            # --------------------------------------------------------------------
# TAB 3: Over 2.5 (new rules with SIGNED Poisson gap)
# Rules:
#   Z  (Over25 odds) between 1.40 and 3.00
#   (CJ - CI) >= 20  OR  (CJ - CI) <= -20    # signed gap
#   BP (Combined GS) >= 3.0
# --------------------------------------------------------------------
with tab3:
    st.subheader("Over 2.5 (Strategy3)")

    IDX_Z  = excel_col_to_idx("Z")
    IDX_BP = excel_col_to_idx("BP")
    IDX_CI = excel_col_to_idx("CI")
    IDX_CJ = excel_col_to_idx("CJ")

    needed_max = max(IDX_Z, IDX_BP, IDX_CI, IDX_CJ)
    if len(df.columns) <= needed_max:
        st.error("Not enough columns for Z / BP / CI / CJ.")
    else:
        col_Z  = df.columns[IDX_Z]
        col_BP = df.columns[IDX_BP]
        col_CI = df.columns[IDX_CI]
        col_CJ = df.columns[IDX_CJ]

        w = df.copy()
        w["O25_odds_Z"]     = to_num(w[col_Z])
        w["CombinedGS_BP"]  = to_num(w[col_BP])
        w["poi_h_CI"]       = to_num(w[col_CI])
        w["poi_a_CJ"]       = to_num(w[col_CJ])

        # SIGNED gap: CJ - CI (can be positive or negative)
        w["poi_gap_signed"] = w["poi_a_CJ"] - w["poi_h_CI"]

        w = add_kickoff(w, col_date, col_time)

        filt = (
            w["O25_odds_Z"].between(1.40, 3.00, inclusive="both") &
            (
                (w["poi_gap_signed"] >= 20.0) |
                (w["poi_gap_signed"] <= -20.0)
            ) &
            (w["CombinedGS_BP"] >= 3.0)
        )
        tips3 = w.loc[filt].copy()

        show3 = []
        if col_country: show3.append(col_country)
        show3 += ["Kickoff"]
        if col_home: show3.append(col_home)
        if col_away: show3.append(col_away)
        show3 += ["O25_odds_Z", "CombinedGS_BP", "poi_h_CI", "poi_a_CJ", "poi_gap_signed"]

        if tips3.empty:
            st.warning("No matches met the Over 2.5 (signed Poisson gap) rules.")
        else:
            # Sort by goals signal then magnitude of gap (largest first)
            tips3 = tips3.sort_values(
                ["CombinedGS_BP", "poi_gap_signed"],
                ascending=[False, False]
            ).reset_index(drop=True)

            top3 = tips3.head(20)
            st.success(f"Over 2.5 (signed gap) — {len(top3)} picks")
            st.dataframe(top3[show3], use_container_width=True, height=500)

            csv3 = top3[show3].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Over 2.5 (signed Poisson gap) CSV",
                data=csv3,
                file_name="SPM_Over25_signed_gap.csv",
                mime="text/csv",
                key="dl_over25_signed"
            )
            csv3 = top3[show3].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Over 2.5 (signed Poisson gap) CSV",
                data=csv3,
                file_name="SPM_Over25_signed_gap.csv",
                mime="text/csv",
                key="dl_over25_signed_t3"
            )

            # Save for combined download (INSIDE else:)
            st.session_state["tips_over25_gap"] = top3[show3].assign(
                Strategy="Over 2.5 (Z/BP/signed gap)"
            )
# --------------------------------------------------------------------
# TAB 4: Lay the Draw
# Rules:
#   CE (Attacking H) ≥ 70 AND CF (Attacking A) ≥ 70
#   P  (Draw odds)   < 4.0
# --------------------------------------------------------------------
with tab4:
    st.subheader("Lay the Draw (Strategy4)")

    IDX_CE = excel_col_to_idx("CE")
    IDX_CF = excel_col_to_idx("CF")
    IDX_P  = excel_col_to_idx("P")    # Draw odds

    needed_max = max(IDX_CE, IDX_CF, IDX_P)
    if len(df.columns) <= needed_max:
        st.error("Not enough columns for CE / CF / P.")
    else:
        col_CE = df.columns[IDX_CE]
        col_CF = df.columns[IDX_CF]
        col_P  = df.columns[IDX_P]

        w = df.copy()
        w["Attack_H_CE"] = to_num(w[col_CE])
        w["Attack_A_CF"] = to_num(w[col_CF])
        w["DrawOdds_P"]  = to_num(w[col_P])

        w = add_kickoff(w, col_date, col_time)

        filt = (
            (w["Attack_H_CE"] >= 70.0) &
            (w["Attack_A_CF"] >= 70.0) &
            (w["DrawOdds_P"] < 4.0)
        )
        ltd = w.loc[filt].copy()

        show4 = []
        if col_country: show4.append(col_country)
        show4 += ["Kickoff"]
        if col_home: show4.append(col_home)
        if col_away: show4.append(col_away)
        show4 += ["DrawOdds_P", "Attack_H_CE", "Attack_A_CF"]

        if ltd.empty:
            st.warning("No matches met the Lay the Draw rules.")
        else:
            ltd = ltd.sort_values(["DrawOdds_P", "Attack_H_CE", "Attack_A_CF"],
                                  ascending=[True, False, False]).reset_index(drop=True)
            top4 = ltd.head(20)
            st.success(f"Lay the Draw — {len(top4)} picks")
            st.dataframe(top4[show4], use_container_width=True, height=500)

            csv4 = top4[show4].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Lay the Draw CSV", data=csv4,
                               file_name="SPM_LayDraw.csv", mime="text/csv", key="dl_ltd")
            csv4 = top4[show4].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Lay the Draw CSV",
                data=csv4,
                file_name="SPM_LayDraw.csv",
                mime="text/csv",
                key="dl_ltd_csv_t4"
            )

            # Save for combined download (INSIDE else:)
            st.session_state["tips_lay_draw"] = top4[show4].assign(Strategy="Lay the Draw")
# --------------------------------------------------------------------
# TAB 5: Back the Away
# Rules:
#   CF (Attacking Away) ≥ 60
#   CG (Defensive Home) ≤ 40
#   CP (Wins The Game Away) ≥ 70
# --------------------------------------------------------------------
with tab5:
    st.subheader("Back the Away (Strategy5)")

    IDX_CF = excel_col_to_idx("CF")
    IDX_CG = excel_col_to_idx("CG")
    IDX_CP = excel_col_to_idx("CP")

    needed_max = max(IDX_CF, IDX_CG, IDX_CP)
    if len(df.columns) <= needed_max:
        st.error("Not enough columns for CF / CG / CP.")
    else:
        col_CF = df.columns[IDX_CF]
        col_CG = df.columns[IDX_CG]
        col_CP = df.columns[IDX_CP]

        w = df.copy()
        w["Attack_A_CF"] = to_num(w[col_CF])
        w["Def_H_CG"]    = to_num(w[col_CG])
        w["Wins_A_CP"]   = to_num(w[col_CP])

        w = add_kickoff(w, col_date, col_time)

        filt = (
            (w["Attack_A_CF"] >= 60.0) &
            (w["Def_H_CG"] <= 40.0) &
            (w["Wins_A_CP"]  >= 70.0)
        )
        bta = w.loc[filt].copy()

        show5 = []
        if col_country: show5.append(col_country)
        show5 += ["Kickoff"]
        if col_home: show5.append(col_home)
        if col_away: show5.append(col_away)
        show5 += ["Attack_A_CF", "Def_H_CG", "Wins_A_CP"]

        if bta.empty:
            st.warning("No matches met the Back the Away rules.")
        else:
            bta = bta.sort_values(["Wins_A_CP", "Attack_A_CF", "Def_H_CG"],
                                  ascending=[False, False, True]).reset_index(drop=True)
            top5 = bta.head(20)
            st.success(f"Back the Away — {len(top5)} picks")
            st.dataframe(top5[show5], use_container_width=True, height=500)

            csv5 = top5[show5].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Back the Away CSV", data=csv5,
                               file_name="SPM_BackAway.csv", mime="text/csv", key="dl_bta")
            csv5 = top5[show5].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Back the Away CSV",
                data=csv5,
                file_name="SPM_BackAway.csv",
                mime="text/csv",
                key="dl_bta_csv_t5"
            )

            # Save for combined download (INSIDE else:)
            st.session_state["tips_back_away"] = top5[show5].assign(Strategy="Back the Away")
# =========================
# Combined Download
# =========================
st.markdown("---")
st.subheader("📦 Download All SPM Tips (Combined)")

# Collect whatever strategies are available
keys = [
    "tips_over25",        # Tab 1
    "tips_homefav",       # Tab 2
    "tips_over25_gap",    # Tab 3
    "tips_lay_draw",      # Tab 4
    "tips_back_away",     # Tab 5
]

pieces = [st.session_state[k] for k in keys if k in st.session_state]

if pieces:
    # Unify columns across strategies
    all_cols = sorted(set().union(*[p.columns for p in pieces]))
    combined = pd.concat([p.reindex(columns=all_cols) for p in pieces], ignore_index=True)

    # Put Strategy first if it exists
    if "Strategy" in combined.columns:
        ordered_cols = (["Strategy"] + [c for c in combined.columns if c != "Strategy"])
        combined = combined[ordered_cols]

    if combined.empty:
        st.info("No rows to download yet — generate tips in any tab above.")
    else:
        csv_all = combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download SPM Tips – All Strategies (CSV)",
            data=csv_all,
            file_name="SPM_Tips_All_Strategies.csv",
            mime="text/csv",
            key="dl_all_csv_main",
        )
        st.dataframe(combined, use_container_width=True, height=500)
else:
    st.info("Generate tips in any tab above to enable the combined download.")
