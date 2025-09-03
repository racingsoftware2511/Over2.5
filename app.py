# app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import re
from io import BytesIO
import base64

st.set_page_config(layout="wide")

# =========================
# Banner with hyperlink
# =========================
BANNER_LOCAL  = "spmlogo_main.png"
BANNER_REMOTE = "https://raw.githubusercontent.com/racingsoftware2511/Over2.5/main/spmlogo_main.png"
BANNER_LINK   = "https://soccerpricemonitor.com/"

def image_to_data_uri(path: str) -> str:
    """Return a data: URI for a local image (falls back to remote URL if missing)."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# Prefer local image -> data URI; fall back to GitHub raw link
try:
    banner_src = image_to_data_uri(BANNER_LOCAL)
except Exception:
    banner_src = BANNER_REMOTE

st.markdown(
    f"""
    <div style="text-align:center; margin-bottom:20px;">
        <a href="{BANNER_LINK}" target="_blank">
            <img src="{banner_src}" style="width:40%; max-width:800px; height:auto;">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Branding (below banner)
# =========================
try:
    st.image(Image.open("spm_logo.png"), width=140)
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
    """Force values to [0,100] as % (accept 0–1, 0–100, or 0–10000)."""
    s = pd.to_numeric(series, errors="coerce")
    def fix(v):
        if pd.isna(v): return v
        v = float(v)
        if v <= 1.0: v *= 100.0
        while v > 100.0: v /= 10.0
        return v
    return s.map(fix)

# --- Pretty number formats for UI tables (no effect on CSV files) ---
FORMAT_MAP = {
    "O25_odds_Z": "{:.2f}",
    "HomeOdds": "{:.2f}",
    "AwayOdds_M": "{:.2f}",
    "DrawOdds_P": "{:.2f}",
    "CombinedGS_BP": "{:.1f}",
    "Combined25_BQ": "{:.1f}",
    "poi_h_CI": "{:.0f}",
    "poi_a_CJ": "{:.0f}",
    "poi_gap_signed": "{:.0f}",
    "Attack_H_CE": "{:.0f}",
    "Attack_A_CF": "{:.0f}",
    "Strength_H_CC": "{:.0f}",
    "WinsGame_CO": "{:.0f}",
    "Wins_A_CP": "{:.0f}",
    "Defense_H_CG": "{:.0f}",
}

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

def make_outcome_row_colorizer(strategy: str, ft_colname: str):
    """Row styler coloring by outcome."""
    def _f(row):
        ft = row.get(ft_colname)
        outcome = decide_outcome(strategy, ft) if ft_colname else None
        if outcome == "WIN":
            return ["background-color: #e6ffea"] * len(row)
        if outcome == "LOSE":
            return ["background-color: #ffecec"] * len(row)
        return [""] * len(row)
    return _f

def get_ft_columns(df):
    """Try to locate result columns (goals and winner) with flexible names."""
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
    Works with a single strategy (e.g. "Over 2.5") or "mixed" where
    picks has a 'Strategy' column. Uses the FT score string (2-1) if present.
    """
    n = len(picks)
    if n == 0:
        return 0, 0, 0, 0.0

    # We need row ids to join picks back to the original DF
    if "__row_id__" not in picks.columns or "__row_id__" not in df.columns:
        return n, None, None, None

    # Try to find a *score* column (2-1 style) in the source DF
    ft_score_col = pick_col(df, [
        "Final Score", "FT", "Full-Time", "Full Time",
        "Full-Time Score", "Full Time Score", "FT Score"
    ])
    if ft_score_col is None:
        # Try the "goals total" column as a fallback (less ideal for Over 2.5 only)
        ft_goals_col = pick_col(df, [
            "FT Goals", "Total Goals FT", "Goals FT", "Final Goals", "Full Time Goals"
        ])
        if ft_goals_col is None:
            return n, None, None, None
        base = df[["__row_id__", ft_goals_col]].copy()
        j = picks.merge(base, on="__row_id__", how="left", suffixes=("", "_res"))
        # Only Over 2.5 can be evaluated from total goals
        if strategy.startswith("Over 2.5"):
            win_mask = pd.to_numeric(j[ft_goals_col], errors="coerce").fillna(-1) >= 3
            wins = int(win_mask.sum())
            losses = int(n - wins)
            strike = wins / n * 100.0 if n else 0.0
            return n, wins, losses, strike
        else:
            return n, None, None, None

    # Preferred path: use FT *score string* and evaluate with decide_outcome
    base = df[["__row_id__", ft_score_col]].copy()
    j = picks.merge(base, on="__row_id__", how="left", suffixes=("", "_res"))

    # Per-row strategy when summarizing the *combined* table
    if strategy == "mixed" and "Strategy" in j.columns:
        strat_series = j["Strategy"].astype(str)
    else:
        strat_series = pd.Series([strategy] * len(j), index=j.index)

    score_series = j[ft_score_col].astype(str)

    outcomes = []
    for s, ft in zip(strat_series, score_series):
        outcomes.append(decide_outcome(s, ft))

    outcomes = pd.Series(outcomes, index=j.index)
    wins = int((outcomes == "WIN").sum())
    losses = int((outcomes == "LOSE").sum())
    strike = wins / n * 100.0 if n else 0.0
    return n, wins, losses, strike
    # --- Combined metrics: count only explicit WIN/LOSE, ignore unknowns ---
tips_all = len(combined)
wins_all = int((combined["Outcome"] == "WIN").sum())
losses_all = int((combined["Outcome"] == "LOSE").sum())
decided = wins_all + losses_all

# Show the four counters
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tips", tips_all)
c2.metric("Wins", wins_all)
c3.metric("Losses", losses_all)
c4.metric("Win %", f"{(wins_all/decided*100):.1f}%" if decided > 0 else "N/A")
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

# Common display columns
col_home    = pick_col(df, ["Home", "Home Team"])
col_away    = pick_col(df, ["Away", "Away Team"])
col_country = pick_col(df, ["Country", "League", "Competition"])
col_date    = pick_col(df, ["Date"])
col_time    = pick_col(df, ["Time"])

# HT / FT columns
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
# TAB 1 — Over 2.5  (Strategy 1)
# --------------------------------------------------------------------
with tab1:
    st.subheader("Over 2.5 Tips (Strategy 1)")

    # AI engine upgraded note
    st.caption("🤖 AI engine upgraded on 1st September 2025 for more accurate results.")

    # Excel column indices we need
    IDX_Z  = excel_col_to_idx("Z")   # Over 2.5 odds
    IDX_BP = excel_col_to_idx("BP")  # Combined GS
    IDX_BQ = excel_col_to_idx("BQ")  # Combined 2.5 (%-like)
    IDX_CE = excel_col_to_idx("CE")  # Attacking (Home)
    IDX_CF = excel_col_to_idx("CF")  # Attacking (Away)
    IDX_K  = excel_col_to_idx("K")   # Home odds (Home Back(T0))

    needed_max = max(IDX_Z, IDX_BP, IDX_BQ, IDX_CE, IDX_CF, IDX_K)
    if len(df.columns) <= needed_max:
        st.error("Not enough columns for Z / BP / BQ / CE / CF / K.")
    else:
        # Resolve column names from indices
        col_Z  = df.columns[IDX_Z]
        col_BP = df.columns[IDX_BP]
        col_BQ = df.columns[IDX_BQ]
        col_CE = df.columns[IDX_CE]
        col_CF = df.columns[IDX_CF]
        col_K  = df.columns[IDX_K]

        # Build working frame
        work = df.copy()
        work["O25_odds_Z"]     = to_num(work[col_Z])
        work["CombinedGS_BP"]  = to_num(work[col_BP])
        work["Combined25_BQ"]  = normalize_pct(work[col_BQ])
        work["Attack_H_CE"]    = to_num(work[col_CE])
        work["Attack_A_CF"]    = to_num(work[col_CF])
        work["HomeOdds_K"]     = to_num(work[col_K])

        work = add_kickoff(work, col_date, col_time)

        # -------- Rules --------
        filt = (
            work["O25_odds_Z"].between(1.40, 3.00, inclusive="both")
            & (work["CombinedGS_BP"] >= 3.5)
            & (work["Combined25_BQ"] >= 70.0)
            & (work["Attack_H_CE"] >= 35.0)
            & (work["Attack_A_CF"] >= 35.0)
            & (work["HomeOdds_K"] <= 2.00)
        )
        tips = work.loc[filt].copy()

        # Summary / UI controls
        total_qualified = int(len(tips))
        st.caption(f"✅ {total_qualified} matches meet Strategy 1 rules.")
        show_all = st.toggle("Show all qualified matches", value=False, key="t1_show_all")

        # Sort strongest first
        tips = tips.sort_values(
            ["Combined25_BQ", "CombinedGS_BP", "HomeOdds_K"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        # Decide how many to show
        if show_all or total_qualified <= 5:
            top = tips
        else:
            top_n = st.slider(
                "How many tips to show?",
                min_value=5,
                max_value=max(5, total_qualified),
                value=min(10, total_qualified),
                key="t1_topn",
            )
            top = tips.head(top_n)

        # Columns to show (keep __row_id__ for combined later)
        show_cols = []
        if col_country: show_cols.append(col_country)
        show_cols += ["Kickoff"]
        if col_ht: show_cols.append(col_ht)
        if col_ft: show_cols.append(col_ft)
        if col_home: show_cols.append(col_home)
        if col_away: show_cols.append(col_away)
        show_cols += [
            "O25_odds_Z", "HomeOdds_K",
            "CombinedGS_BP", "Combined25_BQ",
            "Attack_H_CE", "Attack_A_CF",
        ]

        if top.empty:
            st.warning("No matches met the Strategy 1 rules.")
        else:
            st.success(f"SPM Tips (Over 2.5) — Showing {len(top)}")

            # Keep row id internally; hide it in the UI table
            display1 = top[["__row_id__", *show_cols]].copy() if "__row_id__" in top.columns else top[show_cols].copy()
            out1 = display1.drop(columns=["__row_id__"], errors="ignore")

            # Styled table (WIN/LOSE coloring if FT exists)
            if col_ft:
                st.dataframe(
                    out1.style.apply(
                        make_outcome_row_colorizer("Over 2.5", col_ft), axis=1
                    ).format(FORMAT_MAP, na_rep=""),
                    use_container_width=True, height=500
                )
            else:
                st.dataframe(out1.style.format(FORMAT_MAP, na_rep=""), use_container_width=True, height=500)

            # Performance summary
            n, wins, losses, strike = summarize_picks(display1, df, "Over 2.5")
            show_summary(n, wins, losses, strike)

            # Per-tab CSV
            csv1 = out1.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Over 2.5 SPM Tips (CSV)",
                data=csv1,
                file_name="SPM_Tips_Over25.csv",
                mime="text/csv",
                key="dl_over25_csv_t1",
            )

            # Save for combined download later
            st.session_state["tips_over25"] = display1.assign(Strategy="Over 2.5")
# --------------------------------------------------------------------
# TAB 2 — Home Fav
# --------------------------------------------------------------------
with tab2:
    st.subheader("Home Fav Tips (Strategy2)")
    # AI engine upgraded note
    st.caption("🤖 AI engine upgraded on 1st September 2025 for more accurate results.")
    col_home_odds = pick_col(df, ["Home Back(T0)", "Home Odds", "Home Back(TO)", "Home Back"])
    IDX_BU = excel_col_to_idx("BU")
    IDX_CE = excel_col_to_idx("CE")
    IDX_CO = excel_col_to_idx("CO")

    needed_max = max(IDX_BU, IDX_CE, IDX_CO)
    if len(df.columns) <= needed_max:
        st.error("Not enough columns for BU / CE / CO.")
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
            (work2["Attack_CE"] >= 70.0) &
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
            top2 = tips2.head(20)

            st.success(f"SPM Tips (Home Fav) — Top {len(top2)}")
            display2 = top2[["__row_id__", *show2]]
            out2 = display2.drop(columns=["__row_id__"], errors="ignore")
            if col_ft:
                styler2 = out2.style.apply(make_outcome_row_colorizer("Home Fav", col_ft), axis=1).format(FORMAT_MAP, na_rep="")
                st.dataframe(styler2, use_container_width=True, height=500)
            else:
                st.dataframe(out2.style.format(FORMAT_MAP, na_rep=""), use_container_width=True, height=500)

            n, wins, losses, strike = summarize_picks(display2, df, "Home Fav")
            show_summary(n, wins, losses, strike, key_prefix="t2")

            csv2 = out2.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Home Fav SPM Tips (CSV)", data=csv2,
                               file_name="SPM_Tips_HomeFav.csv", mime="text/csv",
                               key="dl_homefav_csv_t2")

            st.session_state["tips_homefav"] = display2.assign(Strategy="Home Fav")
# --------------------------------------------------------------------
# TAB 3 — Over 2.5 (signed Poisson gap) — UPDATED RULES
# --------------------------------------------------------------------
with tab3:
    st.subheader("Over 2.5 (Strategy3)")
    # AI engine upgraded note
    st.caption("🤖 AI engine upgraded on 1st September 2025 for more accurate results.")
    # Needed Excel columns
    IDX_Z  = excel_col_to_idx("Z")   # O2.5 odds
    IDX_BP = excel_col_to_idx("BP")  # Combined GS
    IDX_CI = excel_col_to_idx("CI")  # Poisson H share
    IDX_CJ = excel_col_to_idx("CJ")  # Poisson A share
    IDX_K  = excel_col_to_idx("K")   # Home Back(T0) odds (column K)

    needed_max = max(IDX_Z, IDX_BP, IDX_CI, IDX_CJ, IDX_K)
    if len(df.columns) <= needed_max:
        st.error("Not enough columns for Z / BP / CI / CJ / K.")
    else:
        col_Z  = df.columns[IDX_Z]
        col_BP = df.columns[IDX_BP]
        col_CI = df.columns[IDX_CI]
        col_CJ = df.columns[IDX_CJ]
        col_K  = df.columns[IDX_K]   # Home Back(T0)

        w = df.copy()
        w["O25_odds_Z"]     = to_num(w[col_Z])
        w["CombinedGS_BP"]  = to_num(w[col_BP])
        w["poi_h_CI"]       = to_num(w[col_CI])
        w["poi_a_CJ"]       = to_num(w[col_CJ])
        w["HomeBack_K"]     = to_num(w[col_K])     # home odds from column K

        # SIGNED gap: CJ - CI (can be positive or negative)
        w["poi_gap_signed"] = w["poi_a_CJ"] - w["poi_h_CI"]

        w = add_kickoff(w, col_date, col_time)

        # ===== RULES (updated) =====
        filt = (
            w["O25_odds_Z"].between(1.40, 3.00, inclusive="both") &
            ((w["poi_gap_signed"] >= 20.0) | (w["poi_gap_signed"] <= -20.0)) &
            (w["CombinedGS_BP"] >= 3.5) &
            (w["HomeBack_K"] <= 2.00)
        )
        tips3 = w.loc[filt].copy()

        # Columns to show
        show3 = []
        if col_country: show3.append(col_country)
        show3 += ["Kickoff"]
        if col_ht: show3.append(col_ht)
        if col_ft: show3.append(col_ft)
        if col_home: show3.append(col_home)
        if col_away: show3.append(col_away)
        show3 += [
            "O25_odds_Z", "CombinedGS_BP", "HomeBack_K",
            "poi_h_CI", "poi_a_CJ", "poi_gap_signed"
        ]

        if tips3.empty:
            st.warning("No matches met the Over 2.5 (signed Poisson gap) rules.")
        else:
            # Sort strongest first: higher CombinedGS, then larger absolute gap
            tips3["abs_gap"] = tips3["poi_gap_signed"].abs()
            tips3 = tips3.sort_values(
                ["CombinedGS_BP", "abs_gap", "poi_gap_signed"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            top3 = tips3.head(20)  # keep as-is; raise/remove if you want more

            st.success(f"Over 2.5 (signed gap) — {len(top3)} picks")
            display3 = top3[["__row_id__", *show3]] if "__row_id__" in top3.columns else top3[show3]
            out3 = display3.drop(columns=["__row_id__"], errors="ignore")

            if col_ft:
                styler3 = out3.style.apply(
                    make_outcome_row_colorizer("Over 2.5", col_ft), axis=1
                ).format(FORMAT_MAP, na_rep="")
                st.dataframe(styler3, use_container_width=True, height=500)
            else:
                st.dataframe(out3.style.format(FORMAT_MAP, na_rep=""),
                             use_container_width=True, height=500)

            n, wins, losses, strike = summarize_picks(display3, df, "Over 2.5")
            show_summary(n, wins, losses, strike, key_prefix="t3")

            csv3 = out3.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Over 2.5 (signed Poisson gap) CSV", data=csv3,
                               file_name="SPM_Over25_signed_gap.csv", mime="text/csv",
                               key="dl_over25_signed_t3")

            st.session_state["tips_over25_gap"] = display3.assign(
                Strategy="Over 2.5 (Z/BP/signed gap)"
            )

# --------------------------------------------------------------------
# TAB 4 — Lay the Draw (updated rules)
# --------------------------------------------------------------------
with tab4:
    st.subheader("Lay the Draw (Strategy4)")
    # AI engine upgraded note
    st.caption("🤖 AI engine upgraded on 1st August 2025 for more accurate results.")
    IDX_CE = excel_col_to_idx("CE")  # Attack H
    IDX_CC = excel_col_to_idx("CC")  # Strength H
    IDX_P  = excel_col_to_idx("P")   # Draw odds

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
            display4 = top4[["__row_id__", *show4]]
            out4 = display4.drop(columns=["__row_id__"], errors="ignore")
            if col_ft:
                styler4 = out4.style.apply(make_outcome_row_colorizer("Lay the Draw", col_ft), axis=1).format(FORMAT_MAP, na_rep="")
                st.dataframe(styler4, use_container_width=True, height=500)
            else:
                st.dataframe(out4.style.format(FORMAT_MAP, na_rep=""), use_container_width=True, height=500)

            n, wins, losses, strike = summarize_picks(display4, df, "Lay the Draw")
            show_summary(n, wins, losses, strike, key_prefix="t4")

            csv4 = out4.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Lay the Draw CSV", data=csv4,
                               file_name="SPM_LayDraw.csv", mime="text/csv",
                               key="dl_ltd_csv_t4")

            st.session_state["tips_lay_draw"] = display4.assign(Strategy="Lay the Draw")

# --------------------------------------------------------------------
# TAB 5 — Back the Away (updated rules incl. away odds any)
# --------------------------------------------------------------------
with tab5:
    st.subheader("Back the Away (Strategy5)")
    # AI engine upgraded note
    st.caption("🤖 AI engine upgraded on 1st September 2025 for more accurate results.")
    IDX_CF = excel_col_to_idx("CF")  # Attack A
    IDX_CG = excel_col_to_idx("CG")  # Defense H
    IDX_CP = excel_col_to_idx("CP")  # Wins A
    IDX_M  = excel_col_to_idx("M")   # Away odds (can be any number)

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
            (w["Wins_A_CP"] >= 70.0) &
            (w["AwayOdds_M"] <= 4.0)
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
            display5 = top5[["__row_id__", *show5]]
            out5 = display5.drop(columns=["__row_id__"], errors="ignore")
            if col_ft:
                styler5 = out5.style.apply(make_outcome_row_colorizer("Back the Away", col_ft), axis=1).format(FORMAT_MAP, na_rep="")
                st.dataframe(styler5, use_container_width=True, height=500)
            else:
                st.dataframe(out5.style.format(FORMAT_MAP, na_rep=""), use_container_width=True, height=500)

            n, wins, losses, strike = summarize_picks(display5, df, "Back the Away")
            show_summary(n, wins, losses, strike, key_prefix="t5")

            csv5 = out5.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Back the Away CSV", data=csv5,
                               file_name="SPM_BackAway.csv", mime="text/csv",
                               key="dl_backaway_csv_t5")

            st.session_state["tips_back_away"] = display5.assign(Strategy="Back the Away")

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

    # Reorder: Strategy, Match, Kickoff, HT, FT, Outcome, then rest
    front = [c for c in ["Strategy","Match","Kickoff","Half-Time Score","Full-Time Score"] if c in combined.columns]
    drop_raw = {"Home","Away"} if "Match" in front else set()
    rest = [c for c in combined.columns if c not in front and c not in drop_raw]
    combined = combined[front + rest]

    # Outcome column
    if "Full-Time Score" in combined.columns and "Strategy" in combined.columns:
        combined["Outcome"] = combined.apply(lambda r: decide_outcome(r.get("Strategy"), r.get("Full-Time Score")), axis=1)
        desired = [c for c in ["Strategy","Match","Kickoff","Half-Time Score","Full-Time Score","Outcome"] if c in combined.columns]
        the_rest = [c for c in combined.columns if c not in desired]
        combined = combined[desired + the_rest]
    else:
        combined["Outcome"] = None

    # Overall summary (only if row ids survived)
    if "__row_id__" in combined.columns:
        n_all, w_all, l_all, sr_all = summarize_picks(combined, df, "mixed")
    else:
        n_all, w_all, l_all, sr_all = len(combined), None, None, None
    show_summary(n_all, w_all, l_all, sr_all, key_prefix="all")

    # CSV
    csv_all = combined.drop(columns=["__row_id__"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download SPM Tips – All Strategies (CSV)", data=csv_all,
                       file_name="SPM_Tips_All_Strategies.csv", mime="text/csv",
                       key="dl_all_csv_main")

    # Colored XLSX
    out = BytesIO()
    export_df = combined.drop(columns=["__row_id__"], errors="ignore")
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, index=False, sheet_name="SPM Tips")
        wb  = writer.book
        ws  = writer.sheets["SPM Tips"]
        nrows, ncols = export_df.shape
        green = wb.add_format({"bg_color": "#C6EFCE"})
        red   = wb.add_format({"bg_color": "#FFC7CE"})
        if "Outcome" in export_df.columns:
            out_idx = export_df.columns.get_loc("Outcome")
            out_col_letter = xl_col_letter(out_idx)
            start_row = 1
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

    st.download_button("📗 Download Colored Excel (XLSX)", data=out.getvalue(),
                       file_name="SPM_Tips_All_Strategies_colored.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="dl_all_xlsx_colored")

    # UI preview
    def _row_colorizer(row):
        color = ""
        if row.get("Outcome") == "WIN":
            color = "background-color: #e6ffea"
        elif row.get("Outcome") == "LOSE":
            color = "background-color: #ffecec"
        return [color] * len(row)
    st.dataframe(export_df.style.apply(_row_colorizer, axis=1).format(FORMAT_MAP, na_rep=""),
                 use_container_width=True, height=500)
else:
    st.info("Generate tips in any tab above to enable the combined download.")
