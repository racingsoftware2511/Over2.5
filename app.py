import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import re
from io import BytesIO

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
    s = 0
    for ch in col_letters.upper():
        s = s * 26 + (ord(ch) - 64)
    return s - 1

def pick_col(df, candidates):
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
    s = pd.to_numeric(series, errors="coerce")
    def fix(v):
        if pd.isna(v): return v
        v = float(v)
        if v <= 1.0: v *= 100.0   # proportions → %
        while v > 100.0: v /= 10.0
        return v
    return s.map(fix)

# --- Pretty number formats for UI tables (no effect on CSV files) ---
FORMAT_MAP = {
    # Odds
    "O25_odds_Z": "{:.2f}",
    "HomeOdds": "{:.2f}",
    "AwayOdds_M": "{:.2f}",
    "DrawOdds_P": "{:.2f}",
    # Model/goal metrics
    "CombinedGS_BP": "{:.1f}",
    "Combined25_BQ": "{:.1f}",
    "poi_h_CI": "{:.0f}",
    "poi_a_CJ": "{:.0f}",
    "poi_gap_signed": "{:.0f}",
    # Shares / ratings
    "Attack_H_CE": "{:.0f}",
    "Attack_A_CF": "{:.0f}",
    "Strength_H_CC": "{:.0f}",
    "WinsGame_CO": "{:.0f}",
    "Wins_A_CP": "{:.0f}",
    "Defense_H_CG": "{:.0f}",
}

# --- Outcome helpers (for UI coloring / combined XLSX) ---
def parse_ft_to_goals(ft_str):
    """Parse '2-1' or '2:1' -> (2,1)."""
    if not isinstance(ft_str, str):
        return (None, None)
    m = re.match(r"\s*(\d+)\s*[-:]\s*(\d+)\s*$", ft_str)
    if not m:
        return (None, None)
    return int(m.group(1)), int(m.group(2))

def decide_outcome(strategy, ft_str):
    """Return 'WIN', 'LOSE' or None based on strategy and full-time score."""
    hg, ag = parse_ft_to_goals(ft_str)
    if hg is None or ag is None:
        return None
    s = (strategy or "").strip()
    if s.startswith("Over 2.5"):
        return "WIN" if (hg + ag) >= 3 else "LOSE"
    if s == "Home Fav":
        return "WIN" if hg > ag else "LOSE"
    if s == "Lay the Draw":
        return "WIN" if hg != ag else "LOSE"
    if s == "Back the Away":
        return "WIN" if ag > hg else "LOSE"
    return None

def xl_col_letter(idx):
    """0-based index -> Excel column letters."""
    n = idx + 1
    letters = ""
    while n:
        n, r = divmod(n - 1, 26)
        letters = chr(65 + r) + letters
    return letters

# --- UI row coloring helper (tabs 1–5 only; CSVs remain plain) ---
def make_outcome_row_colorizer(strategy: str, ft_colname: str):
    """Return a function(row) -> list[css] that colors a row WIN=green, LOSE=red."""
    def _f(row):
        ft = row.get(ft_colname)
        outcome = decide_outcome(strategy, ft) if ft_colname else None
        if outcome == "WIN":
            return ["background-color: #e6ffea"] * len(row)  # soft green
        if outcome == "LOSE":
            return ["background-color: #ffecec"] * len(row)  # soft red
        return [""] * len(row)
    return _f
# Give every row a stable id so we can join picks back to the original rows
# (put this right after you read the Excel, see step 2 below)
# df["__row_id__"] = np.arange(len(df))

def get_ft_columns(df):
    """Try to locate final result columns (goals and winner) with flexible names."""
    ft_goals = pick_col(df, [
        "FT Goals", "Total Goals FT", "Goals FT", "Final Goals", "Full Time Goals"
    ])
    ft_winner = pick_col(df, [
        "Winner FT", "FT Winner", "Full Time Result", "Result FT", "Result"
    ])
    return ft_goals, ft_winner

def summarize_picks(picks: pd.DataFrame, df: pd.DataFrame, strategy: str):
    """
    Return (tips, wins, losses, strike%) for a given set of picks.
    Strategy is used to evaluate the outcome if we only have FT winner.
    """
    n = len(picks)
    if n == 0:
        return 0, 0, 0, 0.0

    ft_goals_col, ft_winner_col = get_ft_columns(df)
    if ft_goals_col is None and ft_winner_col is None:
        # No results available yet
        return n, None, None, None

    # Join picks back to the original results using the row id
    cols_to_pull = ["__row_id__"]
    if ft_goals_col:  cols_to_pull.append(ft_goals_col)
    if ft_winner_col: cols_to_pull.append(ft_winner_col)

    base = df[cols_to_pull].copy()
    j = picks.merge(base, on="__row_id__", how="left")

    # Compute a boolean "win" per row depending on the strategy
    win = pd.Series(False, index=j.index)

    # If we have total-goals, that's the most direct for Over 2.5
    if strategy in {"Over 2.5", "Over 2.5 (Z/BP/signed gap)"} and ft_goals_col:
        win = pd.to_numeric(j[ft_goals_col], errors="coerce").fillna(-1) >= 3

    # If we only have FT winner, fall back to labels
    if ft_winner_col:
        w = j[ft_winner_col].astype(str).str.strip().str.lower()
        if strategy == "Lay the Draw":
            win = win | (w != "draw")
        elif strategy == "Back the Away":
            win = win | w.isin({"away", "a", "2"})
        elif strategy in {"Home Fav"}:
            win = win | w.isin({"home", "h", "1"})
        # For Over 2.5 with only FT winner we can't infer goals—leave as-is

    wins   = int(win.sum())
    losses = int(n - wins) if win.notna().all() else None
    strike = (wins / n * 100.0) if wins is not None and n > 0 else None
    return n, wins, losses, strike

def show_summary(n, wins, losses, strike):
    """Compact 4-metric bar."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tips", n if n is not None else "N/A")
    c2.metric("Wins", wins if wins is not None else "N/A")
    c3.metric("Losses", losses if losses is not None else "N/A")
    c4.metric("Win %", f"{strike:.1f}%" if strike is not None else "N/A")
# =========================
# Upload
# =========================
uploaded = st.file_uploader("Upload your SPM Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.stop()

try:
    df = pd.read_excel(uploaded, sheet_name="Matches")
    df["__row_id__"] = np.arange(len(df))
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

# Detect HT and FT once (support common aliases)
col_ht = pick_col(df, ["HT","Half-Time","Half Time","Half-Time Score","Half Time Score","HT Score"])
col_ft = pick_col(df, ["Final Score","FT","Full-Time","Full Time","Full-Time Score","Full Time Score","FT Score"])

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
# TAB 1 — Over 2.5
# --------------------------------------------------------------------
with tab1:
    st.subheader("Over 2.5 Tips (Strategy1)")

    IDX_Z  = excel_col_to_idx("Z")
    IDX_BP = excel_col_to_idx("BP")
    IDX_BQ = excel_col_to_idx("BQ")
    IDX_CE = excel_col_to_idx("CE")
    IDX_CF = excel_col_to_idx("CF")

    needed_max = max(IDX_Z, IDX_BP, IDX_BQ, IDX_CE, IDX_CF)
    if len(df.columns) <= needed_max:
        st.error("Not enough columns for Z / BP / BQ / CE / CF. Please export the full SPM file.")
    else:
        col_Z  = df.columns[IDX_Z]
        col_BP = df.columns[IDX_BP]
        col_BQ = df.columns[IDX_BQ]
        col_CE = df.columns[IDX_CE]
        col_CF = df.columns[IDX_CF]

        work = df.copy()
        work["O25_odds_Z"]    = to_num(work[col_Z])
        work["CombinedGS_BP"] = to_num(work[col_BP])
        work["Combined25_BQ"] = normalize_pct(work[col_BQ])
        work["Attack_H_CE"]   = to_num(work[col_CE])
        work["Attack_A_CF"]   = to_num(work[col_CF])
        work["Attack_min"]    = pd.concat([work["Attack_H_CE"], work["Attack_A_CF"]], axis=1).min(axis=1)
        work = add_kickoff(work, col_date, col_time)

        filt = (
            work["O25_odds_Z"].between(1.40, 3.00, inclusive="both") &
            (work["CombinedGS_BP"] >= 3.0) &
            (work["Combined25_BQ"] >= 60.0) &
            (work["Attack_min"] >= 35.0)
        )
        tips = work.loc[filt].copy()

        show_cols = []
        if col_country: show_cols.append(col_country)
        show_cols += ["Kickoff"]
        if col_ht: show_cols.append(col_ht)   # HT after Kickoff
        if col_ft: show_cols.append(col_ft)   # FT after HT
        if col_home: show_cols.append(col_home)
        if col_away: show_cols.append(col_away)
        show_cols += ["O25_odds_Z", "CombinedGS_BP", "Combined25_BQ", "Attack_H_CE", "Attack_A_CF"]

        if tips.empty:
            st.warning("No matches met the Over 2.5 rules.")
        else:
            tips = tips.sort_values(["Combined25_BQ", "CombinedGS_BP"], ascending=False).reset_index(drop=True)
            top_n = st.slider("How many tips to show?", 5, 50, 10)
            top = tips.head(top_n)

            st.success(f"SPM Tips (Over 2.5) — Top {len(top)}")
            display1 = top[show_cols]
            if col_ft:
                styler1 = display1.style.apply(
                    make_outcome_row_colorizer("Over 2.5", col_ft), axis=1
                ).format(FORMAT_MAP, na_rep="")
                st.dataframe(styler1, use_container_width=True, height=500)
            else:
                st.dataframe(display1.style.format(FORMAT_MAP, na_rep=""), use_container_width=True, height=500)

            csv1 = top[show_cols].to_csv(index=False).encode("utf-8")
            # after you create `top` (and before/after the download button, either is fine)
            n, wins, losses, strike = summarize_picks(
            top.assign(Strategy="Over 2.5"), df, "Over 2.5"
            )
            show_summary(n, wins, losses, strike)
            st.download_button(
                "📥 Download Over 2.5 SPM Tips (CSV)",
                data=csv1,
                file_name="SPM_Tips_Over25.csv",
                mime="text/csv",
                key="dl_over25_csv_t1",
            )

            st.session_state["tips_over25"] = top[show_cols].assign(Strategy="Over 2.5")

# --------------------------------------------------------------------
# TAB 2 — Home Fav
# --------------------------------------------------------------------
with tab2:
    st.subheader("Home Fav Tips (Strategy2)")

    col_home_odds = pick_col(df, ["Home Back(T0)", "Home Odds", "Home Back(TO)", "Home Back"])

    IDX_BU = excel_col_to_idx("BU")
    IDX_CE = excel_col_to_idx("CE")
    IDX_CO = excel_col_to_idx("CO")

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
        work2["Attack_CE"]    = to_num(work2[col_CE])
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
        if col_ht: show2.append(col_ht)
        if col_ft: show2.append(col_ft)
        if col_home: show2.append(col_home)
        if col_away: show2.append(col_away)
        if col_home_odds: show2.append("HomeOdds")
        show2 += ["HomeGames_BU", "Attack_CE", "WinsGame_CO"]

        if tips2.empty:
            st.warning("No matches met the Home Fav rules.")
        else:
            tips2 = tips2.sort_values(["WinsGame_CO", "Attack_CE", "HomeGames_BU"], ascending=[False, False, False]).reset_index(drop=True)
            top2 = tips2.head(10)

            st.success(f"SPM Tips (Home Fav) — Top {len(top2)}")
            display2 = top2[show2]
            if col_ft:
                styler2 = display2.style.apply(
                    make_outcome_row_colorizer("Home Fav", col_ft), axis=1
                ).format(FORMAT_MAP, na_rep="")
                st.dataframe(styler2, use_container_width=True, height=500)
            else:
                st.dataframe(display2.style.format(FORMAT_MAP, na_rep=""), use_container_width=True, height=500)

            csv2 = top2[show2].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Home Fav SPM Tips (CSV)",
                data=csv2,
                file_name="SPM_Tips_HomeFav.csv",
                mime="text/csv",
                key="dl_homefav_csv_t2",
            )

            st.session_state["tips_homefav"] = top2[show2].assign(Strategy="Home Fav")

# --------------------------------------------------------------------
# TAB 3 — Over 2.5 (signed Poisson gap)
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
        w["poi_gap_signed"] = w["poi_a_CJ"] - w["poi_h_CI"]
        w = add_kickoff(w, col_date, col_time)

        filt = (
            w["O25_odds_Z"].between(1.40, 3.00, inclusive="both") &
            ((w["poi_gap_signed"] >= 20.0) | (w["poi_gap_signed"] <= -20.0)) &
            (w["CombinedGS_BP"] >= 3.0)
        )
        tips3 = w.loc[filt].copy()

        show3 = []
        if col_country: show3.append(col_country)
        show3 += ["Kickoff"]
        if col_ht: show3.append(col_ht)
        if col_ft: show3.append(col_ft)
        if col_home: show3.append(col_home)
        if col_away: show3.append(col_away)
        show3 += ["O25_odds_Z", "CombinedGS_BP", "poi_h_CI", "poi_a_CJ", "poi_gap_signed"]

        if tips3.empty:
            st.warning("No matches met the Over 2.5 (signed Poisson gap) rules.")
        else:
            tips3 = tips3.sort_values(["CombinedGS_BP", "poi_gap_signed"], ascending=[False, False]).reset_index(drop=True)
            top3 = tips3.head(20)

            st.success(f"Over 2.5 (signed gap) — {len(top3)} picks")
            display3 = top3[show3]
            if col_ft:
                styler3 = display3.style.apply(
                    make_outcome_row_colorizer("Over 2.5", col_ft), axis=1
                ).format(FORMAT_MAP, na_rep="")
                st.dataframe(styler3, use_container_width=True, height=500)
            else:
                st.dataframe(display3.style.format(FORMAT_MAP, na_rep=""), use_container_width=True, height=500)

            csv3 = top3[show3].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Over 2.5 (signed Poisson gap) CSV",
                data=csv3,
                file_name="SPM_Over25_signed_gap.csv",
                mime="text/csv",
                key="dl_over25_signed_t3"
            )

            st.session_state["tips_over25_gap"] = top3[show3].assign(
                Strategy="Over 2.5 (Z/BP/signed gap)"
            )

# --------------------------------------------------------------------
# TAB 4 — Lay the Draw
# --------------------------------------------------------------------
with tab4:
    st.subheader("Lay the Draw (Strategy4)")

    IDX_CE = excel_col_to_idx("CE")
    IDX_CC = excel_col_to_idx("CC")
    IDX_P  = excel_col_to_idx("P")

    needed_max = max(IDX_CE, IDX_CC, IDX_P)
    if len(df.columns) <= needed_max:
        st.error("Not enough columns for CE / CC / P.")
    else:
        col_CE = df.columns[IDX_CE]
        col_CC = df.columns[IDX_CC]
        col_P  = df.columns[IDX_P]

        w = df.copy()
        w["Attack_H_CE"]   = to_num(w[col_CE])
        w["Strength_H_CC"] = to_num(w[col_CC])
        w["DrawOdds_P"]    = to_num(w[col_P])
        w = add_kickoff(w, col_date, col_time)

        filt = (
            (w["Attack_H_CE"] >= 70.0) &
            (w["Strength_H_CC"] >= 70.0) &
            (w["DrawOdds_P"] < 4.0)
        )
        ltd = w.loc[filt].copy()

        show4 = []
        if col_country: show4.append(col_country)
        show4 += ["Kickoff"]
        if col_ht: show4.append(col_ht)
        if col_ft: show4.append(col_ft)
        if col_home: show4.append(col_home)
        if col_away: show4.append(col_away)
        show4 += ["DrawOdds_P", "Attack_H_CE", "Strength_H_CC"]

        if ltd.empty:
            st.warning("No matches met the Lay the Draw rules.")
        else:
            ltd = ltd.sort_values(["DrawOdds_P", "Attack_H_CE", "Strength_H_CC"], ascending=[True, False, False]).reset_index(drop=True)
            top4 = ltd.head(20)

            st.success(f"Lay the Draw — {len(top4)} picks")
            display4 = top4[show4]
            if col_ft:
                styler4 = display4.style.apply(
                    make_outcome_row_colorizer("Lay the Draw", col_ft), axis=1
                ).format(FORMAT_MAP, na_rep="")
                st.dataframe(styler4, use_container_width=True, height=500)
            else:
                st.dataframe(display4.style.format(FORMAT_MAP, na_rep=""), use_container_width=True, height=500)

            csv4 = top4[show4].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Lay the Draw CSV",
                data=csv4,
                file_name="SPM_LayDraw.csv",
                mime="text/csv",
                key="dl_ltd_csv_t4",
            )

            st.session_state["tips_lay_draw"] = top4[show4].assign(Strategy="Lay the Draw")

# --------------------------------------------------------------------
# TAB 5 — Back the Away
# --------------------------------------------------------------------
with tab5:
    st.subheader("Back the Away (Strategy5)")

    IDX_CF = excel_col_to_idx("CF")
    IDX_CG = excel_col_to_idx("CG")
    IDX_CP = excel_col_to_idx("CP")
    IDX_M  = excel_col_to_idx("M")

    needed_max = max(IDX_CF, IDX_CG, IDX_CP, IDX_M)
    if len(df.columns) <= needed_max:
        st.error("Not enough columns for CF / CG / CP / M.")
    else:
        col_CF = df.columns[IDX_CF]
        col_CG = df.columns[IDX_CG]
        col_CP = df.columns[IDX_CP]
        col_M  = df.columns[IDX_M]

        w = df.copy()
        w["Attack_A_CF"]  = to_num(w[col_CF])
        w["Defense_H_CG"] = to_num(w[col_CG])
        w["Wins_A_CP"]    = to_num(w[col_CP])
        w["AwayOdds_M"]   = to_num(w[col_M])
        w = add_kickoff(w, col_date, col_time)

        filt = (
            (w["Attack_A_CF"] >= 60.0) &
            (w["Defense_H_CG"] <= 40.0) &
            (w["Wins_A_CP"] >= 70.0)
        )
        backaway = w.loc[filt].copy()

        show5 = []
        if col_country: show5.append(col_country)
        show5 += ["Kickoff"]
        if col_ht: show5.append(col_ht)
        if col_ft: show5.append(col_ft)
        if col_home: show5.append(col_home)
        if col_away: show5.append(col_away)
        show5 += ["AwayOdds_M", "Attack_A_CF", "Defense_H_CG", "Wins_A_CP"]

        if backaway.empty:
            st.warning("No matches met the Back the Away rules.")
        else:
            backaway = backaway.sort_values(["Wins_A_CP", "Attack_A_CF", "Defense_H_CG"], ascending=[False, False, True]).reset_index(drop=True)
            top5 = backaway.head(20)

            st.success(f"Back the Away — {len(top5)} picks")
            display5 = top5[show5]
            if col_ft:
                styler5 = display5.style.apply(
                    make_outcome_row_colorizer("Back the Away", col_ft), axis=1
                ).format(FORMAT_MAP, na_rep="")
                st.dataframe(styler5, use_container_width=True, height=500)
            else:
                st.dataframe(display5.style.format(FORMAT_MAP, na_rep=""), use_container_width=True, height=500)

            csv5 = top5[show5].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Back the Away CSV",
                data=csv5,
                file_name="SPM_BackAway.csv",
                mime="text/csv",
                key="dl_backaway_csv_t5",
            )

            st.session_state["tips_back_away"] = top5[show5].assign(Strategy="Back the Away")

# =========================
# Combined Download
# =========================
st.markdown("---")
st.subheader("📦 Download All SPM Tips (Combined)")

keys = ["tips_over25","tips_homefav","tips_over25_gap","tips_lay_draw","tips_back_away"]
pieces = [st.session_state[k].copy() for k in keys if k in st.session_state]

def normalize_for_combined(df_: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Time": "Kickoff",
        "HT": "Half-Time Score",
        "Half-Time": "Half-Time Score",
        "Half Time": "Half-Time Score",
        "Half-Time Score": "Half-Time Score",
        "Half Time Score": "Half-Time Score",
        "Final Score": "Full-Time Score",
        "FT": "Full-Time Score",
        "Full-Time": "Full-Time Score",
        "Full Time": "Full-Time Score",
        "Full-Time Score": "Full-Time Score",
        "Full Time Score": "Full-Time Score",
    }
    df_ = df_.rename(columns={k: v for k, v in rename_map.items() if k in df_.columns})
    if "Home" in df_.columns and "Away" in df_.columns:
        df_["Match"] = df_["Home"].astype(str) + " vs " + df_["Away"].astype(str)
    return df_

pieces = [normalize_for_combined(p) for p in pieces]

if pieces:
    all_cols = sorted(set().union(*[p.columns for p in pieces]))
    combined = pd.concat([p.reindex(columns=all_cols) for p in pieces], ignore_index=True)

    # Front order & drop raw Home/Away if Match exists
    front = [c for c in ["Strategy","Match","Kickoff","Half-Time Score","Full-Time Score"] if c in combined.columns]
    drop_raw = {"Home","Away"} if "Match" in front else set()
    rest = [c for c in combined.columns if c not in front and c not in drop_raw]
    combined = combined[front + rest]

    # Outcome
    if "Full-Time Score" in combined.columns and "Strategy" in combined.columns:
        combined["Outcome"] = combined.apply(
            lambda r: decide_outcome(r.get("Strategy"), r.get("Full-Time Score")),
            axis=1
        )
        desired = [c for c in ["Strategy","Match","Kickoff","Half-Time Score","Full-Time Score","Outcome"] if c in combined.columns]
        the_rest = [c for c in combined.columns if c not in desired]
        combined = combined[desired + the_rest]
    else:
        combined["Outcome"] = None

    # UI colored preview + number formatting
    def _row_colorizer(row):
        color = ""
        if row.get("Outcome") == "WIN":
            color = "background-color: #e6ffea"
        elif row.get("Outcome") == "LOSE":
            color = "background-color: #ffecec"
        return [color] * len(row)

    styled = combined.style.apply(_row_colorizer, axis=1).format(FORMAT_MAP, na_rep="")

    if combined.empty:
        st.info("No rows to download yet — generate tips in any tab above.")
    else:
        # CSV (plain)
        csv_all = combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download SPM Tips – All Strategies (CSV)",
            data=csv_all,
            file_name="SPM_Tips_All_Strategies.csv",
            mime="text/csv",
            key="dl_all_csv_main",
        )

        # Colored XLSX
        out = BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            combined.to_excel(writer, index=False, sheet_name="SPM Tips")
            wb  = writer.book
            ws  = writer.sheets["SPM Tips"]
            nrows, ncols = combined.shape
            green = wb.add_format({"bg_color": "#C6EFCE"})
            red   = wb.add_format({"bg_color": "#FFC7CE"})

            out_idx = combined.columns.get_loc("Outcome")
            out_col_letter = xl_col_letter(out_idx)
            start_row = 1  # first data row
            ws.conditional_format(start_row, 0, nrows, ncols-1, {
                "type": "formula",
                "criteria": f"=${out_col_letter}{start_row+1}=\"WIN\"",
                "format": green
            })
            ws.conditional_format(start_row, 0, nrows, ncols-1, {
                "type": "formula",
                "criteria": f"=${out_col_letter}{start_row+1}=\"LOSE\"",
                "format": red
            })

        st.download_button(
            "📗 Download Colored Excel (XLSX)",
            data=out.getvalue(),
            file_name="SPM_Tips_All_Strategies_colored.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_all_xlsx_colored",
        )

        st.dataframe(styled, use_container_width=True, height=500)
else:
    st.info("Generate tips in any tab above to enable the combined download.")
