#!/usr/bin/env python3
"""Football betting UI with match intelligence and league/player analytics."""

from __future__ import annotations

import itertools
import math
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None

try:
    from sports_betting.team_names import TEAM_NAME_MAP as _TEAM_MAP
except Exception:  # noqa: BLE001
    _TEAM_MAP: dict[str, str] = {}

from sports_betting.generate_bet_combinations import (
    RESULT_VALUES,
    _read_optional_csv,
    build_team_snapshot,
    h2h_features_for_match,
    load_data,
    parse_date,
    player_match_insights,
)
from sports_betting.fetch_top6_data import (
    DEFAULT_TOP6,
    PORTUGAL,
    build_dataset,
    infer_default_start_season,
    infer_latest_season_start,
    normalize_clean,
    save_outputs,
    _update_metadata,
)
from sports_betting.xgboost_models import (
    CLASS_TO_RESULT,
    MATCH_FEATURE_COLS,
    player_probabilities_for_team,
    predict_match_proba,
    train_match_model,
    train_player_models,
)

# Anchor every data path to the directory that contains app.py so Streamlit can
# be launched from any working directory (e.g.  streamlit run /full/path/app.py)
_APP_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_FILE = _APP_DIR / "data/sports/processed/top6_plus_portugal_matches_odds_since2022.csv"
TOP6_DATA_FILE    = _APP_DIR / "data/sports/processed/top6_matches_odds_since2022.csv"
PLAYER_STATS_FILE = _APP_DIR / "data/sports/processed/player_stats.csv"

# Background-refresh log / metadata files
MATCHES_LOG_FILE = _APP_DIR / "data/sports/processed/refresh_matches.log"
PLAYERS_LOG_FILE = _APP_DIR / "data/sports/processed/refresh_players.log"
METADATA_FILE    = _APP_DIR / "data/sports/processed/refresh_metadata.json"


def _fetch_data_sync(output_dir: Path, status_fn=None) -> str:
    """Download match data synchronously from football-data.co.uk.

    Called when the processed CSV is missing (e.g. fresh Streamlit Cloud deploy).
    Returns an empty string on success or an error message on failure.
    """
    today = date.today()
    start_season = infer_default_start_season(today)  # ~2006
    end_season   = infer_latest_season_start(today)
    leagues = DEFAULT_TOP6 + (PORTUGAL,)
    prefix  = "top6_plus_portugal"

    try:
        if status_fn:
            status_fn(f"Downloading {len(leagues)} leagues × {end_season - start_season + 1} seasons…")
        raw_df = build_dataset(leagues=leagues, start_season=start_season, end_season=end_season)
        min_date = pd.Timestamp(f"{start_season}-01-01")
        clean_df = normalize_clean(raw_df, min_date)
        save_outputs(raw_df, clean_df, output_dir, prefix)
        _update_metadata("matches", len(clean_df), "football-data.co.uk")
        return ""
    except Exception as exc:  # noqa: BLE001
        return str(exc)

# Leagues we support — filters out stale Eredivisie rows still present in old CSVs
SUPPORTED_LEAGUES: frozenset[str] = frozenset({
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    "Primeira Liga",
})

# ESPN unofficial API slugs (free, no auth required)
ESPN_LEAGUE_SLUGS: dict[str, str] = {
    "Premier League": "eng.1",
    "La Liga":        "esp.1",
    "Serie A":        "ita.1",
    "Bundesliga":     "ger.1",
    "Ligue 1":        "fra.1",
    "Primeira Liga":  "por.1",
}

_STALE_HOURS = 2.0   # auto-refresh threshold

MARKET_OPTIONS = [
    "1X2",
    "Goals O/U 1.5",
    "Goals O/U 2.5",
    "Goals O/U 3.5",
    "Corners O/U 8.5",
    "Corners O/U 9.5",
    "Corners O/U 10.5",
    "Cards O/U 2.5",
    "Cards O/U 3.5",
    "Cards O/U 4.5",
    "BTTS",
    "Score First",
    "1st Half Result",
    "Win Both Halves",
    "1st Half Goals O/U 0.5",
    "1st Half Goals O/U 1.5",
    "2nd Half Goals O/U 0.5",
    "2nd Half Goals O/U 1.5",
    "Player to Score",
]

# API-Football league IDs (v3.football.api-sports.io)
LEAGUE_API_IDS: dict[str, int] = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
    "Primeira Liga": 94,
}


def apply_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

        :root {
          --bg:        #0d1117;
          --bg-2:      #161b27;
          --card:      #1c2333;
          --card-2:    #212840;
          --border:    rgba(148, 163, 184, 0.13);
          --border-2:  rgba(148, 163, 184, 0.26);
          --ink:       #e2e8f4;
          --ink-2:     #94a3b8;
          --accent:    #38bdf8;
          --accent-2:  #818cf8;
          --green:     #34d399;
          --red:       #f87171;
          --shadow:    0 8px 32px rgba(0, 0, 0, 0.5);
        }

        /* ── Base layout ── */
        html, body { font-family: 'Inter', sans-serif !important; }

        [data-testid="stAppViewContainer"] {
          background: var(--bg) !important;
        }
        [data-testid="stHeader"] {
          background: rgba(13, 17, 23, 0.85) !important;
          backdrop-filter: blur(12px);
          border-bottom: 1px solid var(--border);
        }
        .main .block-container {
          max-width: 1320px;
          padding-top: 2rem;
          padding-bottom: 3rem;
        }

        /* ── Global text ── */
        .stApp, .stApp p, .stApp label,
        .stApp span, .stApp div,
        .stApp li, .stApp td, .stApp th {
          color: var(--ink) !important;
          font-family: 'Inter', sans-serif !important;
        }
        h1 {
          font-size: 2rem !important;
          font-weight: 700 !important;
          letter-spacing: -0.03em !important;
          color: #ffffff !important;
        }
        h2, h3 {
          font-weight: 600 !important;
          letter-spacing: -0.02em !important;
          color: #ffffff !important;
        }
        .stApp [data-testid="stCaptionContainer"] p {
          color: var(--ink-2) !important;
          font-size: 0.9rem !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
          background: var(--bg-2) !important;
          border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
          color: var(--ink) !important;
        }

        /* ── Inputs / selects / sliders ── */
        [data-testid="stSelectbox"] > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stDateInput"] input {
          background: var(--card-2) !important;
          border: 1px solid var(--border-2) !important;
          color: var(--ink) !important;
          border-radius: 8px !important;
        }
        [data-testid="stSelectbox"] svg { color: var(--ink-2) !important; }

        /* ── Buttons ── */
        [data-testid="stButton"] > button {
          background: linear-gradient(135deg, #1e3a5f 0%, #1a3050 100%) !important;
          border: 1px solid var(--accent) !important;
          color: var(--accent) !important;
          border-radius: 8px !important;
          font-weight: 600 !important;
          transition: all 0.2s ease;
        }
        [data-testid="stButton"] > button:hover {
          background: var(--accent) !important;
          color: #000 !important;
        }

        /* ── Tabs ── */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
          background: var(--card) !important;
          border: 1px solid var(--border) !important;
          border-radius: 12px !important;
          padding: 4px !important;
          gap: 4px !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
          background: transparent !important;
          color: var(--ink-2) !important;
          border-radius: 9px !important;
          font-weight: 600 !important;
          font-size: 0.9rem !important;
          padding: 0 1.25rem !important;
          height: 2.4rem !important;
          border: none !important;
          transition: all 0.2s ease;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
          background: var(--card-2) !important;
          color: var(--accent) !important;
          border: 1px solid var(--border-2) !important;
        }

        /* ── DataFrames ── */
        [data-testid="stDataFrame"] {
          border: 1px solid var(--border) !important;
          border-radius: 12px !important;
          overflow: hidden !important;
          background: var(--card) !important;
        }

        /* ── Metric widgets ── */
        [data-testid="stMetric"] {
          background: var(--card) !important;
          border: 1px solid var(--border) !important;
          border-radius: 12px !important;
          padding: 1rem 1.2rem !important;
        }
        [data-testid="stMetricLabel"] p { color: var(--ink-2) !important; font-size: 0.8rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
        [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.15rem !important; font-weight: 700 !important; }
        [data-testid="stMetricDelta"] { color: var(--green) !important; font-size: 0.82rem !important; font-weight: 600 !important; }

        /* ── Expander ── */
        [data-testid="stExpander"] {
          background: var(--card) !important;
          border: 1px solid var(--border) !important;
          border-radius: 10px !important;
        }
        [data-testid="stExpander"] summary p { color: var(--ink) !important; font-weight: 600 !important; }

        /* ── Alerts ── */
        [data-testid="stAlert"] {
          border-radius: 10px !important;
          border: 1px solid var(--border-2) !important;
        }

        /* ── Custom panel & metric divs ── */
        .panel {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 16px 20px;
          box-shadow: var(--shadow);
          margin-bottom: 1rem;
        }
        .metric {
          background: var(--card-2);
          border-left: 3px solid var(--accent);
          border-radius: 12px;
          padding: 12px 16px;
          font-weight: 700;
          color: var(--ink) !important;
          box-shadow: var(--shadow);
        }
        .metric p, .metric span, .metric div { color: var(--ink) !important; }

        /* ── Subheader accent lines — main content only ── */
        /* Streamlit 1.30-1.45: main area is .main > .block-container          or [data-testid="stMain"] > .block-container */
        .main .block-container h2::after,
        .main .block-container h3::after,
        [data-testid="stMain"] h2::after,
        [data-testid="stMain"] h3::after {
          content: '';
          display: block;
          margin-top: 6px;
          height: 2px;
          width: 40px;
          background: var(--accent);
          border-radius: 2px;
        }
        /* Nuke all pseudo-elements inside the sidebar — belt AND suspenders */
        [data-testid="stSidebar"] *::before,
        [data-testid="stSidebar"] *::after,
        [data-testid="stSidebarContent"] *::before,
        [data-testid="stSidebarContent"] *::after {
          display: none !important;
          content: none !important;
        }

        /* ── Widget labels: wrap gracefully in narrow columns ── */
        /* Streamlit 1.35+ uses stWidgetLabel; older uses direct <label> */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        .stSlider     label,
        .stNumberInput label,
        .stTextInput  label,
        .stDateInput  label,
        .stSelectbox  label,
        .stMultiSelect label {
          white-space: normal !important;
          overflow: visible !important;
          line-height: 1.4 !important;
        }

        /* ── Sliders: separate label from thumb-value tooltip ── */
        /* The thumb tooltip is positioned absolute ~20 px above the track,
           inside [data-baseweb="slider"].  Adding padding-bottom to the
           label container opens a gap so the tooltip never covers the text. */
        [data-testid="stSlider"],
        .stSlider {
          margin-bottom: 0.75rem !important;
        }
        [data-testid="stSlider"] [data-testid="stWidgetLabel"],
        .stSlider [data-testid="stWidgetLabel"] {
          padding-bottom: 1.6rem !important; /* room for floating tooltip */
        }
        /* Streamlit 1.45 renders the track wrapper directly under stSlider;
           push it down so the tooltip clears the label on any viewport */
        [data-testid="stSlider"] > div:last-child,
        .stSlider > div:last-child {
          margin-top: 0.2rem !important;
        }

        /* ── Sidebar vertical breathing room ── */
        [data-testid="stSidebar"] hr,
        [data-testid="stSidebarContent"] hr {
          margin: 0.6rem 0 !important;
        }
        /* Stack every widget block with a small gap */
        [data-testid="stSidebar"] .stVerticalBlock > *,
        [data-testid="stSidebarContent"] .stVerticalBlock > * {
          margin-bottom: 0.15rem !important;
        }

        /* ── Tabs: flexible height — never clip long labels ── */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
          flex-wrap: wrap !important;   /* allow tabs to wrap on small screens */
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
          height: auto !important;
          min-height: 2.4rem !important;
          padding: 0.35rem 1rem !important;
          white-space: nowrap !important;
        }

        /* ── Multiselect: clip tags cleanly inside narrow columns ── */
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {
          max-width: 100% !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] span:first-child {
          overflow: hidden !important;
          text-overflow: ellipsis !important;
          max-width: calc(100% - 1.5rem) !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-2); }
        ::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--ink-2); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _start_background(cmd: list[str], log_file: Path) -> int:
    """Launch cmd as a detached background process, piping stdout+stderr to log_file.

    Returns the PID of the spawned process.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as fh:
        fh.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting: {' '.join(cmd)}\n\n")
    with open(log_file, "a") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc.pid


def run_refresh(start_season: int, end_season: int, min_date: date) -> int:
    """Start match-data refresh in background. Returns PID."""
    script = Path(__file__).resolve().parent / "sports_betting" / "fetch_top6_data.py"
    cmd = [
        sys.executable, str(script),
        "--start-season", str(start_season),
        "--end-season",   str(end_season),
        "--min-date",     min_date.isoformat(),
    ]
    return _start_background(cmd, MATCHES_LOG_FILE)


def run_player_stats_refresh(api_key: str = "", season: str = "2526") -> int:
    """Start player-stats refresh in background. Returns PID."""
    script = Path(__file__).resolve().parent / "sports_betting" / "fetch_player_stats.py"
    cmd = [sys.executable, str(script), "--season", season]
    if api_key.strip():
        cmd += ["--api-key", api_key.strip()]
    return _start_background(cmd, PLAYERS_LOG_FILE)


@st.cache_data(ttl=3600, show_spinner=False)
def load_player_stats(path: str) -> pd.DataFrame:
    """Load cached Understat player stats CSV. Returns empty DataFrame if not found."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        # Normalise team name capitalisation just in case
        if "team" in df.columns:
            df["team"] = df["team"].str.strip()
        if "player" in df.columns:
            df["player"] = df["player"].str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def parse_lineup_text(text: str) -> list[str]:
    if not text.strip():
        return []
    return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]


def fetch_probable_xi_api_football(api_key: str, home_team: str, away_team: str) -> tuple[list[str], list[str], str]:
    if requests is None:
        return [], [], "`requests` is not available in this environment."
    if not api_key.strip():
        return [], [], "Add API key to fetch lineups online."

    base = "https://v3.football.api-sports.io"
    headers = {"x-apisports-key": api_key.strip()}

    def _team_id(team_name: str) -> int | None:
        resp = requests.get(f"{base}/teams", params={"search": team_name}, headers=headers, timeout=20)
        data = resp.json().get("response", []) if resp.ok else []
        if not data:
            return None
        return int(data[0]["team"]["id"])

    try:
        hid = _team_id(home_team)
        aid = _team_id(away_team)
        if hid is None or aid is None:
            return [], [], "Could not resolve team IDs from API-Football."

        fx = requests.get(f"{base}/fixtures", params={"team": hid, "next": 20}, headers=headers, timeout=20)
        fixtures = fx.json().get("response", []) if fx.ok else []
        target = None
        for item in fixtures:
            teams = item.get("teams", {})
            if int(teams.get("home", {}).get("id", -1)) == hid and int(teams.get("away", {}).get("id", -1)) == aid:
                target = item
                break
            if int(teams.get("home", {}).get("id", -1)) == aid and int(teams.get("away", {}).get("id", -1)) == hid:
                target = item
                break
        if target is None:
            return [], [], "No upcoming fixture found between the selected teams."

        fixture_id = int(target["fixture"]["id"])
        lx = requests.get(f"{base}/fixtures/lineups", params={"fixture": fixture_id}, headers=headers, timeout=20)
        lineups = lx.json().get("response", []) if lx.ok else []
        if not lineups:
            return [], [], "Lineups not published yet for this fixture."

        home_xi: list[str] = []
        away_xi: list[str] = []
        for item in lineups:
            team_name = str(item.get("team", {}).get("name", ""))
            starters = [str(p.get("player", {}).get("name", "")).strip() for p in item.get("startXI", [])]
            starters = [p for p in starters if p]
            if not starters:
                continue
            if home_team.lower() in team_name.lower():
                home_xi = starters
            elif away_team.lower() in team_name.lower():
                away_xi = starters
            elif not home_xi:
                home_xi = starters
            else:
                away_xi = starters

        return home_xi, away_xi, "Fetched probable XI from API-Football."
    except Exception as exc:  # noqa: BLE001
        return [], [], f"Online lineup fetch failed: {exc}"


def _player_columns(contrib_df: pd.DataFrame) -> tuple[str | None, str | None]:
    team_col = "team" if "team" in contrib_df.columns else "team_name" if "team_name" in contrib_df.columns else None
    player_col = "player" if "player" in contrib_df.columns else "player_name" if "player_name" in contrib_df.columns else None
    return team_col, player_col


def lineup_strength(team: str, lineup: list[str], contrib_df: pd.DataFrame, as_of_ts: pd.Timestamp) -> float:
    if contrib_df.empty:
        return 0.0
    team_col, player_col = _player_columns(contrib_df)
    if team_col is None or player_col is None or "match_date" not in contrib_df.columns:
        return 0.0

    df = contrib_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.loc[df["match_date"].notna() & (df["match_date"] <= as_of_ts) & (df[team_col] == team)].copy()
    if df.empty:
        return 0.0

    for c in ("goals", "assists", "xg", "xa", "key_passes", "rating"):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["impact"] = (
        1.5 * df["goals"]
        + 1.1 * df["assists"]
        + 0.8 * df["xg"]
        + 0.6 * df["xa"]
        + 0.10 * df["key_passes"]
        + 0.2 * df["rating"]
    )

    by_player = df.groupby(player_col, dropna=False)["impact"].mean().sort_values(ascending=False)
    if by_player.empty:
        return 0.0

    if lineup:
        sel = by_player.loc[by_player.index.astype(str).isin(set(map(str, lineup)))]
        if sel.empty:
            sel = by_player.head(11)
        return float(sel.mean())
    return float(by_player.head(11).mean())


def build_context(
    data_path: Path,
    as_of_date: date,
    momentum_window: int,
    injuries_file: Path,
    player_contrib_file: Path,
    other_comp_file: Path,
) -> tuple[dict[str, object], str]:
    if not data_path.exists():
        return {}, f"Data file not found: {data_path}"

    df = load_data(data_path)
    known = df["result_ft"].isin(RESULT_VALUES)
    as_of_ts = parse_date(as_of_date.isoformat())
    # Include matches played on as_of_date itself (e.g. a 3pm kick-off on today's
    # date would be stored as midnight of that date, which the strict < would miss).
    _hist_cutoff = as_of_ts + pd.Timedelta(days=1)
    historical = df.loc[known & (df["match_date"] < _hist_cutoff)].copy()
    if historical.empty:
        return {}, "No historical matches available before selected as-of date."

    injuries_df = _read_optional_csv(injuries_file, required_cols=[])
    contrib_df = _read_optional_csv(player_contrib_file, required_cols=[])
    other_df = _read_optional_csv(other_comp_file, required_cols=[])

    snapshot = build_team_snapshot(
        historical=historical,
        as_of_date=as_of_ts,
        momentum_window=max(momentum_window, 1),
        injuries_df=injuries_df,
        player_contrib_df=contrib_df,
        other_comp_df=other_df,
    )
    league_lookup = historical[["league_code", "league_name"]].drop_duplicates()
    snapshot = snapshot.merge(league_lookup, on="league_code", how="left")

    # Current-season snapshot: used only for standings display so points/matches
    # reflect the ongoing season rather than cumulative multi-season totals.
    latest_season = historical["season_label"].dropna().max()
    current_hist = historical.loc[historical["season_label"] == latest_season].copy()
    if current_hist.empty:
        current_hist = historical
    current_snapshot = build_team_snapshot(
        historical=current_hist,
        as_of_date=as_of_ts,
        momentum_window=max(momentum_window, 1),
        injuries_df=injuries_df,
        player_contrib_df=contrib_df,
        other_comp_df=other_df,
    )
    current_snapshot = current_snapshot.merge(league_lookup, on="league_code", how="left")

    return {
        "historical": historical,
        "all_matches": df,          # full dataset incl. scheduled (no result) fixtures
        "snapshot": snapshot,
        "current_snapshot": current_snapshot,
        "as_of_ts": as_of_ts,
        "injuries_df": injuries_df,
        "contrib_df": contrib_df,
        "other_df": other_df,
        "current_season": latest_season,
    }, ""


def _team_row(snapshot: pd.DataFrame, league_name: str, team: str) -> pd.Series:
    row = snapshot.loc[(snapshot["league_name"] == league_name) & (snapshot["team"] == team)]
    return row.iloc[0] if not row.empty else pd.Series(dtype=float)


def build_feature_vector(
    context: dict[str, object],
    league_name: str,
    home_team: str,
    away_team: str,
    h2h_years: int,
    home_lineup_strength: float,
    away_lineup_strength: float,
    home_big_games_8d: float,
    away_big_games_8d: float,
) -> tuple[dict[str, float], dict[str, float]]:
    snapshot = context["snapshot"]
    historical = context["historical"]
    as_of_ts = context["as_of_ts"]

    home = _team_row(snapshot, league_name, home_team)
    away = _team_row(snapshot, league_name, away_team)
    if home.empty or away.empty:
        raise ValueError("Could not build team snapshot for selected teams.")

    h2h = h2h_features_for_match(
        historical=historical,
        league_code=str(home["league_code"]),
        home_team=home_team,
        away_team=away_team,
        as_of_date=as_of_ts,
        years=max(h2h_years, 1),
    )

    league_codes = sorted(snapshot["league_code"].astype(str).dropna().unique())
    league_idx = float({c: i for i, c in enumerate(league_codes)}.get(str(home["league_code"]), 0))

    features = {
        "form_points_gap": float(home.get("last_points_pg", 0.0) - away.get("last_points_pg", 0.0)),
        "forward_goals_gap": float(home.get("last_goals_for_pg", 0.0) - away.get("last_goals_for_pg", 0.0)),
        "defense_gap": float(away.get("last_goals_against_pg", 0.0) - home.get("last_goals_against_pg", 0.0)),
        "cards_gap": float(away.get("last_cards_pg", 0.0) - home.get("last_cards_pg", 0.0)),
        "corners_gap": float(home.get("last_corners_diff_pg", 0.0) - away.get("last_corners_diff_pg", 0.0)),
        "rest_gap": float(home.get("days_rest_effective", 7.0) - away.get("days_rest_effective", 7.0)),
        "fatigue_gap": float((away.get("total_matches_last7", 0.0) + away_big_games_8d) - (home.get("total_matches_last7", 0.0) + home_big_games_8d)),
        "season_points_gap": float(home.get("points", 0.0) / max(float(home.get("matches", 1.0)), 1.0) - away.get("points", 0.0) / max(float(away.get("matches", 1.0)), 1.0)),
        "h2h_gap": float(h2h.get("h2h_gap", 0.0)),
        "h2h_goal_diff": float(h2h.get("h2h_goal_diff_pg", 0.0)),
        "injury_gap": float(away.get("injury_impact", 0.0) - home.get("injury_impact", 0.0)),
        "lineup_strength_gap": float(home_lineup_strength - away_lineup_strength),
        "league_idx": league_idx,
    }
    return features, h2h


def outcome_name(code: str) -> str:
    return {"H": "Home Win (1)", "D": "Draw (X)", "A": "Away Win (2)"}.get(code, code)


def choose_risk_bets(probs: dict[str, float], odds: dict[str, float], reasons: str) -> list[dict[str, str | float]]:
    ev = {k: probs[k] * odds[k] - 1.0 for k in probs}
    conservative = max(probs.items(), key=lambda x: x[1])[0]

    moderate_candidates = [k for k in probs if probs[k] >= 0.25]
    if moderate_candidates:
        moderate = max(moderate_candidates, key=lambda k: ev[k])
    else:
        moderate = max(ev, key=ev.get)

    high_candidates = [k for k in probs if odds[k] >= np.median(list(odds.values()))]
    if high_candidates:
        high = max(high_candidates, key=lambda k: ev[k])
    else:
        high = max(ev, key=ev.get)

    return [
        {
            "tier": "Conservative",
            "pick": conservative,
            "prob": probs[conservative],
            "ev": ev[conservative],
            "tip": f"Highest hit probability. {reasons}",
        },
        {
            "tier": "Moderate",
            "pick": moderate,
            "prob": probs[moderate],
            "ev": ev[moderate],
            "tip": f"Balance between value and probability. {reasons}",
        },
        {
            "tier": "High Risk",
            "pick": high,
            "prob": probs[high],
            "ev": ev[high],
            "tip": f"Higher variance with stronger payout profile. {reasons}",
        },
    ]


def explain_factors(features: dict[str, float], home_team: str, away_team: str) -> str:
    msgs: list[str] = []
    if features["fatigue_gap"] > 0.5:
        msgs.append(f"{away_team} had heavier load recently, increasing fatigue risk")
    if features["fatigue_gap"] < -0.5:
        msgs.append(f"{home_team} had heavier recent load")
    if features["injury_gap"] > 0.4:
        msgs.append(f"{away_team} has higher key injury impact")
    if features["injury_gap"] < -0.4:
        msgs.append(f"{home_team} has higher key injury impact")
    if features["forward_goals_gap"] > 0.2:
        msgs.append(f"{home_team} has stronger recent attacking output")
    if features["forward_goals_gap"] < -0.2:
        msgs.append(f"{away_team} has stronger recent attacking output")
    if features["h2h_gap"] > 0.1:
        msgs.append(f"Head-to-head trend slightly favors {home_team}")
    if features["h2h_gap"] < -0.1:
        msgs.append(f"Head-to-head trend slightly favors {away_team}")
    return "; ".join(msgs) if msgs else "Model sees balanced conditions with no dominant contextual edge"


def team_last5_form(
    historical: pd.DataFrame,
    team: str,
    league_name: str,
    as_of_ts: pd.Timestamp,
    n: int = 5,
) -> str:
    """Return a W/D/L form string for the last N matches across all competitions.

    The `historical` DataFrame is already date-bounded by build_context so no
    extra date filter is needed here.  The league filter is intentionally dropped
    so a loss in the cup or Champions League is reflected in the form string.
    """
    rows = historical.loc[
        (historical["result_ft"].isin(RESULT_VALUES))
        & ((historical["home_team"] == team) | (historical["away_team"] == team))
    ].sort_values("match_date", ascending=True).tail(n)

    labels = []
    for _, row in rows.iterrows():
        r = row["result_ft"]
        result = {"H": "W", "D": "D", "A": "L"}[r] if row["home_team"] == team else {"A": "W", "D": "D", "H": "L"}[r]
        labels.append(result)
    return " ".join(labels) if labels else "–"


def _auto_suggest_odds(
    context: dict,
    match_model: object,
    league: str,
    home_team: str,
    away_team: str,
) -> tuple[float, float, float]:
    """Compute model-implied odds (with 5% bookmaker margin) for current matchup."""
    feats, _ = build_feature_vector(
        context=context,
        league_name=league,
        home_team=home_team,
        away_team=away_team,
        h2h_years=5,
        home_lineup_strength=0.0,
        away_lineup_strength=0.0,
        home_big_games_8d=0.0,
        away_big_games_8d=0.0,
    )
    p = predict_match_proba(match_model, feats)
    m = 0.05  # 5% margin
    return (
        round(max(1.01, (1 / max(p["H"], 0.01)) * (1 - m)), 2),
        round(max(1.01, (1 / max(p["D"], 0.01)) * (1 - m)), 2),
        round(max(1.01, (1 / max(p["A"], 0.01)) * (1 - m)), 2),
    )


def fetch_upcoming_fixtures_api(
    api_key: str,
    league_names: list[str],
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, str]:
    """Fetch upcoming fixtures from API-Football (v3.football.api-sports.io).

    Returns a DataFrame with match_date, league_name, home_team, away_team
    and result_ft = pd.NA (since the games haven't been played).
    Falls back to an empty DataFrame on any error.
    """
    if requests is None:
        return pd.DataFrame(), "The `requests` library is not installed."
    if not api_key.strip():
        return pd.DataFrame(), "No API key provided."

    base = "https://v3.football.api-sports.io"
    headers = {"x-apisports-key": api_key.strip()}

    # European season starts in July/August
    season = start_date.year if start_date.month >= 7 else start_date.year - 1

    rows: list[dict] = []
    errors: list[str] = []

    for league_name in league_names:
        league_id = LEAGUE_API_IDS.get(league_name)
        if league_id is None:
            errors.append(f"No API-Football ID mapped for '{league_name}'")
            continue
        try:
            resp = requests.get(
                f"{base}/fixtures",
                params={
                    "league": league_id,
                    "season": season,
                    "from": start_date.isoformat(),
                    "to": end_date.isoformat(),
                },
                headers=headers,
                timeout=20,
            )
            if not resp.ok:
                errors.append(f"{league_name}: HTTP {resp.status_code}")
                continue
            for item in resp.json().get("response", []):
                teams = item.get("teams", {})
                fixture_meta = item.get("fixture", {})
                date_str = fixture_meta.get("date", "")
                try:
                    md = pd.Timestamp(date_str)
                except Exception:
                    continue
                home_raw = str(teams.get("home", {}).get("name", ""))
                away_raw = str(teams.get("away", {}).get("name", ""))
                # Best-effort normalization to match local dataset names
                home_norm = _TEAM_MAP.get(home_raw, home_raw)
                away_norm = _TEAM_MAP.get(away_raw, away_raw)
                rows.append(
                    {
                        "match_date": md,
                        "league_name": league_name,
                        "home_team": home_norm,
                        "away_team": away_norm,
                        "result_ft": pd.NA,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{league_name}: {exc}")

    if not rows:
        msg = "API-Football returned no fixtures for the selected range."
        if errors:
            msg += " Details: " + "; ".join(errors)
        return pd.DataFrame(), msg

    df = (
        pd.DataFrame(rows)
        .sort_values("match_date")
        .reset_index(drop=True)
    )
    msg = f"✅ Fetched {len(df)} fixture(s) from API-Football."
    if errors:
        msg += f"  ⚠️ Warnings: {'; '.join(errors)}"
    return df, msg


def fetch_upcoming_fixtures_espn(
    league_names: list[str],
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, str]:
    """Fetch upcoming fixtures from the ESPN unofficial API (free, no key required).

    Endpoint: site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard
    Supports date ranges and covers all 6 leagues including Primeira Liga.
    """
    if requests is None:
        return pd.DataFrame(), "The `requests` library is not installed."

    base = "https://site.api.espn.com/apis/site/v2/sports/soccer"
    # ESPN date range format: YYYYMMDD-YYYYMMDD
    date_param = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

    rows: list[dict] = []
    errors: list[str] = []

    for league_name in league_names:
        slug = ESPN_LEAGUE_SLUGS.get(league_name)
        if slug is None:
            errors.append(f"No ESPN slug mapped for '{league_name}'")
            continue
        try:
            resp = requests.get(
                f"{base}/{slug}/scoreboard",
                params={"dates": date_param},
                timeout=15,
            )
            if not resp.ok:
                errors.append(f"{league_name}: HTTP {resp.status_code}")
                continue
            for ev in resp.json().get("events", []):
                comp = (ev.get("competitions") or [{}])[0]
                competitors = comp.get("competitors", [])
                home_team = next(
                    (c["team"]["displayName"] for c in competitors if c.get("homeAway") == "home"),
                    "",
                )
                away_team = next(
                    (c["team"]["displayName"] for c in competitors if c.get("homeAway") == "away"),
                    "",
                )
                try:
                    md = pd.Timestamp(ev.get("date", ""))
                except Exception:
                    continue
                # Only include matches in the requested window
                if not (start_date <= md.date() <= end_date):
                    continue
                rows.append({
                    "match_date":  md,
                    "league_name": league_name,
                    "home_team":   _TEAM_MAP.get(home_team, home_team),
                    "away_team":   _TEAM_MAP.get(away_team, away_team),
                    "result_ft":   pd.NA,
                })
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{league_name}: {exc}")

    if not rows:
        msg = "ESPN returned no fixtures for the selected range."
        if errors:
            msg += " Details: " + "; ".join(errors)
        return pd.DataFrame(), msg

    df = pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
    msg = f"✅ Fetched {len(df)} fixture(s) from ESPN (free)."
    if errors:
        msg += f"  ⚠️ {'; '.join(errors)}"
    return df, msg


# ── Refresh metadata helpers ─────────────────────────────────────────────────

def load_refresh_metadata() -> dict:
    """Load refresh_metadata.json; returns empty dict if missing/corrupt."""
    if METADATA_FILE.exists():
        try:
            import json as _json  # noqa: PLC0415
            return _json.loads(METADATA_FILE.read_text())
        except Exception:
            pass
    return {}


def _is_stale(ts_str: str | None, hours: float = _STALE_HOURS) -> bool:
    """Return True if ts_str is absent or older than `hours` hours."""
    if not ts_str:
        return True
    try:
        last = datetime.fromisoformat(ts_str)
        return (datetime.now() - last).total_seconds() > hours * 3600
    except Exception:
        return True


# ── Half-time / special market probability helpers ────────────────────────────

def _compute_ht_result_proba(
    historical: pd.DataFrame,
    home_team: str,
    away_team: str,
    league_name: str,
    as_of_ts: pd.Timestamp,
    seasons: int = 3,
) -> tuple[float, float, float]:
    """Return (home_ht_win, draw_ht, away_ht_win) probabilities."""
    min_date = as_of_ts - pd.Timedelta(days=seasons * 365)
    hist = historical.loc[
        (historical["league_name"] == league_name)
        & (historical["match_date"] >= min_date)
        & (historical["match_date"] < as_of_ts)
    ]
    if "home_goals_ht" not in hist.columns or hist.empty:
        return 0.40, 0.27, 0.33

    ht_h = pd.to_numeric(hist["home_goals_ht"], errors="coerce").fillna(0)
    ht_a = pd.to_numeric(hist["away_goals_ht"], errors="coerce").fillna(0)
    lg_h = float((ht_h > ht_a).mean())
    lg_d = float((ht_h == ht_a).mean())
    lg_a = float((ht_h < ht_a).mean())

    def _win_rate(df: pd.DataFrame, team: str) -> float:
        if df.empty:
            return lg_h
        r: list[int] = []
        for _, row in df.iterrows():
            hh = float(pd.to_numeric(row.get("home_goals_ht", 0), errors="coerce") or 0)
            aa = float(pd.to_numeric(row.get("away_goals_ht", 0), errors="coerce") or 0)
            r.append(
                1 if (row["home_team"] == team and hh > aa)
                or (row["away_team"] == team and aa > hh)
                else 0
            )
        return float(np.mean(r)) if r else lg_h

    h_df = hist.loc[(hist["home_team"] == home_team) | (hist["away_team"] == home_team)]
    a_df = hist.loc[(hist["home_team"] == away_team) | (hist["away_team"] == away_team)]
    p_h = float(np.clip(
        (_win_rate(h_df, home_team) + (1 - _win_rate(a_df, away_team))) / 2 * 0.7 + lg_h * 0.3,
        0.10, 0.70,
    ))
    p_a = float(np.clip(
        (_win_rate(a_df, away_team) + (1 - _win_rate(h_df, home_team))) / 2 * 0.7 + lg_a * 0.3,
        0.10, 0.70,
    ))
    p_d = max(0.05, 1.0 - p_h - p_a)
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total


def _compute_score_first_proba(
    historical: pd.DataFrame,
    home_team: str,
    away_team: str,
    league_name: str,
    as_of_ts: pd.Timestamp,
    seasons: int = 3,
) -> tuple[float, float]:
    """Return (home_scores_first_prob, away_scores_first_prob). Proxy via HT goals."""
    min_date = as_of_ts - pd.Timedelta(days=seasons * 365)
    hist = historical.loc[
        (historical["league_name"] == league_name)
        & (historical["match_date"] >= min_date)
        & (historical["match_date"] < as_of_ts)
    ]
    if "home_goals_ht" not in hist.columns or hist.empty:
        return 0.55, 0.45

    def _sf_rate(df: pd.DataFrame, team: str) -> float:
        if df.empty:
            return 0.5
        total, team_first = 0, 0.0
        for _, row in df.iterrows():
            hh = float(pd.to_numeric(row.get("home_goals_ht", 0), errors="coerce") or 0)
            aa = float(pd.to_numeric(row.get("away_goals_ht", 0), errors="coerce") or 0)
            if hh + aa <= 0:
                continue
            total += 1
            if row["home_team"] == team:
                team_first += hh / (hh + aa)
            else:
                team_first += aa / (hh + aa)
        return team_first / max(total, 1)

    h_df = hist.loc[(hist["home_team"] == home_team) | (hist["away_team"] == home_team)]
    a_df = hist.loc[(hist["home_team"] == away_team) | (hist["away_team"] == away_team)]
    combined = float(np.clip(
        (_sf_rate(h_df, home_team) + (1.0 - _sf_rate(a_df, away_team))) / 2,
        0.15, 0.85,
    ))
    return combined, 1.0 - combined


def _compute_win_both_halves_proba(
    historical: pd.DataFrame,
    home_team: str,
    away_team: str,
    league_name: str,
    as_of_ts: pd.Timestamp,
    seasons: int = 3,
) -> tuple[float, float]:
    """Return (home_wins_both_halves, away_wins_both_halves)."""
    min_date = as_of_ts - pd.Timedelta(days=seasons * 365)
    hist = historical.loc[
        (historical["league_name"] == league_name)
        & (historical["match_date"] >= min_date)
        & (historical["match_date"] < as_of_ts)
    ]
    if "home_goals_ht" not in hist.columns or hist.empty:
        return 0.22, 0.14

    ht_h = pd.to_numeric(hist["home_goals_ht"], errors="coerce").fillna(0)
    ht_a = pd.to_numeric(hist["away_goals_ht"], errors="coerce").fillna(0)
    ft_h = pd.to_numeric(hist["home_goals_ft"], errors="coerce").fillna(0)
    ft_a = pd.to_numeric(hist["away_goals_ft"], errors="coerce").fillna(0)
    sh_h = (ft_h - ht_h).clip(lower=0)
    sh_a = (ft_a - ht_a).clip(lower=0)
    lg_home_both = float(((ht_h > ht_a) & (sh_h > sh_a)).mean())
    lg_away_both = float(((ht_a > ht_h) & (sh_a > sh_h)).mean())

    def _wbh_rate(df: pd.DataFrame, team: str, default: float) -> float:
        if df.empty:
            return default
        r: list[int] = []
        for _, row in df.iterrows():
            hth = float(pd.to_numeric(row.get("home_goals_ht", 0), errors="coerce") or 0)
            ath = float(pd.to_numeric(row.get("away_goals_ht", 0), errors="coerce") or 0)
            htf = float(pd.to_numeric(row.get("home_goals_ft", 0), errors="coerce") or 0)
            atf = float(pd.to_numeric(row.get("away_goals_ft", 0), errors="coerce") or 0)
            sh2_h = max(0.0, htf - hth)
            sh2_a = max(0.0, atf - ath)
            if row["home_team"] == team:
                r.append(int(hth > ath and sh2_h > sh2_a))
            else:
                r.append(int(ath > hth and sh2_a > sh2_h))
        return float(np.mean(r)) if r else default

    h_df = hist.loc[(hist["home_team"] == home_team) | (hist["away_team"] == home_team)]
    a_df = hist.loc[(hist["home_team"] == away_team) | (hist["away_team"] == away_team)]
    home_p = float(np.clip(
        (_wbh_rate(h_df, home_team, lg_home_both) + lg_home_both) / 2, 0.03, 0.50
    ))
    away_p = float(np.clip(
        (_wbh_rate(a_df, away_team, lg_away_both) + lg_away_both) / 2, 0.03, 0.40
    ))
    return home_p, away_p


def _player_score_prob(goals: float, matches: float) -> float:
    """Poisson probability of a player scoring at least once in a match."""
    if matches <= 0:
        return 0.0
    return 1.0 - math.exp(-goals / matches)


def _get_player_score_picks(
    player_stats: pd.DataFrame,
    home_team: str,
    away_team: str,
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Return list of (pick_label, prob) for top scorers from both teams."""
    if player_stats.empty:
        return []
    picks: list[tuple[str, float]] = []
    for team_name in [home_team, away_team]:
        team_df = player_stats.loc[player_stats["team"] == team_name]
        if team_df.empty:
            team_df = player_stats.loc[
                player_stats["team"].str.lower().str.contains(
                    team_name.lower()[:7], na=False
                )
            ]
        if (
            team_df.empty
            or "goals" not in team_df.columns
            or "matches" not in team_df.columns
        ):
            continue
        team_df = team_df.copy()
        team_df["goals"] = pd.to_numeric(team_df["goals"], errors="coerce").fillna(0)
        team_df["matches"] = pd.to_numeric(team_df["matches"], errors="coerce").fillna(0)
        team_df = team_df.loc[(team_df["goals"] > 0) & (team_df["matches"] > 0)]
        if team_df.empty:
            continue
        for _, pr in team_df.sort_values("goals", ascending=False).head(top_n).iterrows():
            prob = _player_score_prob(float(pr["goals"]), float(pr["matches"]))
            if prob >= 0.10:
                name = str(pr.get("player", "Unknown"))
                picks.append((f"Score: {name}", prob))
    return picks


def _pick_context(
    historical: pd.DataFrame,
    home_team: str,
    away_team: str,
    market: str,
    pick_label: str,
    league_name: str,
    as_of_ts: pd.Timestamp,
    n: int = 5,
) -> str:
    """Return a short, human-readable stats sentence for a (match, market, pick) leg.

    Used to populate the 'Context' column in the ticket table so users can
    understand *why* a pick was suggested.
    """
    min_date = as_of_ts - pd.Timedelta(days=3 * 365)
    hist = historical.loc[
        (historical["league_name"] == league_name)
        & (historical["match_date"] >= min_date)
        & (historical["match_date"] < as_of_ts)
    ]
    m = market.lower()

    # Recent form helpers
    home_h = hist.loc[hist["home_team"] == home_team].tail(n)   # home team at home
    away_a = hist.loc[hist["away_team"] == away_team].tail(n)   # away team away

    def _avg(df: pd.DataFrame, col: str) -> float | None:
        if df.empty or col not in df.columns:
            return None
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())

    # ── 1X2 / 1st Half Result ────────────────────────────────────────────────
    if "1x2" in m or "1st half result" in m:
        use_ft = "1x2" in m
        res_col = "result_ft"
        w_h, d_h, l_h = "H", "D", "A"

        if not home_h.empty and res_col in home_h.columns:
            hw = int((home_h[res_col] == w_h).sum())
            hd = int((home_h[res_col] == d_h).sum())
            hl = int((home_h[res_col] == l_h).sum())
            h_str = f"{home_team} home (L{len(home_h)}): {hw}W {hd}D {hl}L"
        else:
            h_str = f"{home_team}: no recent data"

        if not away_a.empty and res_col in away_a.columns:
            aw = int((away_a[res_col] == l_h).sum())  # away win = "A"
            ad = int((away_a[res_col] == d_h).sum())
            al = int((away_a[res_col] == w_h).sum())
            a_str = f"{away_team} away (L{len(away_a)}): {aw}W {ad}D {al}L"
        else:
            a_str = f"{away_team}: no recent data"

        return f"{h_str}  ·  {a_str}"

    # ── Corners ──────────────────────────────────────────────────────────────
    if "corners" in m:
        hc = _avg(home_h, "home_corners")
        ac_h = _avg(home_h, "away_corners")
        hc_a = _avg(away_a, "home_corners")
        ac = _avg(away_a, "away_corners")
        parts: list[str] = []
        if hc is not None and ac_h is not None and len(home_h) > 0:
            parts.append(f"Avg {hc + ac_h:.1f} total corners in {home_team}'s home games (L{len(home_h)})")
        if hc_a is not None and ac is not None and len(away_a) > 0:
            parts.append(f"{hc_a + ac:.1f} in {away_team}'s away games (L{len(away_a)})")
        return "  ·  ".join(parts) if parts else "No corner data available"

    # ── Cards ────────────────────────────────────────────────────────────────
    if "cards" in m:
        def _cards_avg(df: pd.DataFrame) -> float | None:
            if df.empty:
                return None
            yh = pd.to_numeric(df.get("home_yellow_cards", pd.Series(dtype=float)), errors="coerce").fillna(0)
            ya = pd.to_numeric(df.get("away_yellow_cards", pd.Series(dtype=float)), errors="coerce").fillna(0)
            rh = pd.to_numeric(df.get("home_red_cards",    pd.Series(dtype=float)), errors="coerce").fillna(0)
            ra = pd.to_numeric(df.get("away_red_cards",    pd.Series(dtype=float)), errors="coerce").fillna(0)
            return float((yh + ya + rh + ra).mean())
        h_avg = _cards_avg(home_h)
        a_avg = _cards_avg(away_a)
        parts = []
        if h_avg is not None and len(home_h) > 0:
            parts.append(f"Avg {h_avg:.1f} cards in {home_team}'s home games (L{len(home_h)})")
        if a_avg is not None and len(away_a) > 0:
            parts.append(f"{a_avg:.1f} in {away_team}'s away games (L{len(away_a)})")
        return "  ·  ".join(parts) if parts else "No card data available"

    # ── Goals O/U ────────────────────────────────────────────────────────────
    if "goals" in m and "half" not in m:
        hg_h = _avg(home_h, "home_goals_ft")
        ag_h = _avg(home_h, "away_goals_ft")
        hg_a = _avg(away_a, "home_goals_ft")
        ag_a = _avg(away_a, "away_goals_ft")
        parts = []
        if hg_h is not None and ag_h is not None and len(home_h) > 0:
            parts.append(f"Avg {hg_h + ag_h:.1f} goals in {home_team}'s home games (L{len(home_h)})")
        if hg_a is not None and ag_a is not None and len(away_a) > 0:
            parts.append(f"{hg_a + ag_a:.1f} in {away_team}'s away games (L{len(away_a)})")
        return "  ·  ".join(parts) if parts else "No goal data available"

    # ── 1st Half Goals ────────────────────────────────────────────────────────
    if "1st half goals" in m:
        hh = _avg(home_h, "home_goals_ht")
        ah = _avg(home_h, "away_goals_ht")
        parts = []
        if hh is not None and ah is not None and len(home_h) > 0:
            parts.append(f"Avg {hh + ah:.1f} HT goals in {home_team}'s home games (L{len(home_h)})")
        if not parts:
            return "No HT data yet — run Refresh Data to fetch HTHG/HTAG"
        return "  ·  ".join(parts)

    # ── 2nd Half Goals ────────────────────────────────────────────────────────
    if "2nd half goals" in m:
        if "home_goals_ht" in home_h.columns and "home_goals_ft" in home_h.columns and len(home_h) > 0:
            sh = (
                (pd.to_numeric(home_h["home_goals_ft"], errors="coerce").fillna(0)
                 - pd.to_numeric(home_h["home_goals_ht"], errors="coerce").fillna(0)).clip(lower=0)
                + (pd.to_numeric(home_h["away_goals_ft"], errors="coerce").fillna(0)
                   - pd.to_numeric(home_h["away_goals_ht"], errors="coerce").fillna(0)).clip(lower=0)
            ).mean()
            return f"Avg {sh:.1f} 2nd-half goals in {home_team}'s home games (L{len(home_h)})"
        return "No HT data yet — run Refresh Data"

    # ── BTTS ─────────────────────────────────────────────────────────────────
    if "btts" in m:
        def _btts(df: pd.DataFrame) -> float | None:
            if df.empty or "home_goals_ft" not in df.columns:
                return None
            hg = pd.to_numeric(df["home_goals_ft"], errors="coerce").fillna(0)
            ag = pd.to_numeric(df["away_goals_ft"], errors="coerce").fillna(0)
            return float(((hg > 0) & (ag > 0)).mean())
        all_h = hist.loc[(hist["home_team"] == home_team) | (hist["away_team"] == home_team)].tail(n)
        all_a = hist.loc[(hist["home_team"] == away_team) | (hist["away_team"] == away_team)].tail(n)
        parts = []
        r = _btts(all_h)
        if r is not None and len(all_h) > 0:
            parts.append(f"BTTS in {r:.0%} of {home_team}'s matches (L{len(all_h)})")
        r = _btts(all_a)
        if r is not None and len(all_a) > 0:
            parts.append(f"{r:.0%} of {away_team}'s (L{len(all_a)})")
        return "  ·  ".join(parts) if parts else "No BTTS data"

    # ── Score First ───────────────────────────────────────────────────────────
    if "score first" in m:
        if "home_goals_ht" in home_h.columns and len(home_h) > 0:
            hth = pd.to_numeric(home_h["home_goals_ht"], errors="coerce").fillna(0)
            ath = pd.to_numeric(home_h["away_goals_ht"], errors="coerce").fillna(0)
            total = hth + ath
            valid = total > 0
            if valid.any():
                home_first_rate = float((hth[valid] / total[valid]).mean())
                return (f"{home_team} home HT scoring share: {home_first_rate:.0%} "
                        f"(proxy for scoring first, L{valid.sum()})")
        return "Score First proxy via HT goal share — no recent data"

    # ── Win Both Halves ───────────────────────────────────────────────────────
    if "win both" in m:
        if "home_goals_ht" in home_h.columns and "home_goals_ft" in home_h.columns and len(home_h) > 0:
            wins = 0
            for _, r in home_h.iterrows():
                hth = float(pd.to_numeric(r.get("home_goals_ht", 0), errors="coerce") or 0)
                ath = float(pd.to_numeric(r.get("away_goals_ht", 0), errors="coerce") or 0)
                htf = float(pd.to_numeric(r.get("home_goals_ft", 0), errors="coerce") or 0)
                atf = float(pd.to_numeric(r.get("away_goals_ft", 0), errors="coerce") or 0)
                wins += int(hth > ath and max(0, htf - hth) > max(0, atf - ath))
            return f"{home_team} won both halves in {wins}/{len(home_h)} recent home games"
        return "Win Both Halves — no HT data yet (run Refresh Data)"

    # ── Player to Score ───────────────────────────────────────────────────────
    if "score:" in pick_label.lower():
        return f"Poisson model · season goals/matches rate → P(scores) = 1−e^(−rate)"

    return ""


def estimate_market_proba(
    historical: pd.DataFrame,
    home_team: str,
    away_team: str,
    market: str,
    league_name: str,
    as_of_ts: pd.Timestamp,
    seasons: int = 3,
) -> tuple[float, float]:
    """Return (over_prob, under_prob) using empirical rates from team history.

    Works for Goals O/U, Corners O/U, Cards O/U and BTTS markets.
    Averages the rate from home-team matches and away-team matches.
    """
    min_date = as_of_ts - pd.Timedelta(days=seasons * 365)
    league_hist = historical.loc[
        (historical["league_name"] == league_name)
        & (historical["match_date"] >= min_date)
        & (historical["match_date"] < as_of_ts)
    ].copy()

    if league_hist.empty:
        return 0.5, 0.5

    m = market.lower()

    # ── BTTS ────────────────────────────────────────────────────────────────
    if m == "btts":
        def _btts_rate(df: pd.DataFrame) -> float:
            if df.empty:
                return 0.5
            hg = pd.to_numeric(df["home_goals_ft"], errors="coerce")
            ag = pd.to_numeric(df["away_goals_ft"], errors="coerce")
            valid = hg.notna() & ag.notna()
            return float(((hg[valid] > 0) & (ag[valid] > 0)).mean()) if valid.any() else 0.5

        h_df = league_hist.loc[
            (league_hist["home_team"] == home_team) | (league_hist["away_team"] == home_team)
        ]
        a_df = league_hist.loc[
            (league_hist["home_team"] == away_team) | (league_hist["away_team"] == away_team)
        ]
        rate = float(np.clip((_btts_rate(h_df) + _btts_rate(a_df)) / 2, 0.05, 0.95))
        return rate, 1.0 - rate

    # ── 1st Half Goals O/U ───────────────────────────────────────────────────
    if "1st half goals" in m:
        thr = 0.5 if "0.5" in market else 1.5
        if "home_goals_ht" in league_hist.columns:
            def _ht_total(df: pd.DataFrame) -> pd.Series:
                return (
                    pd.to_numeric(df["home_goals_ht"], errors="coerce").fillna(0)
                    + pd.to_numeric(df["away_goals_ht"], errors="coerce").fillna(0)
                )
            def _ht_over(df: pd.DataFrame) -> float:
                if df.empty:
                    return 0.5
                return float((_ht_total(df) > thr).mean())
            h_df = league_hist.loc[
                (league_hist["home_team"] == home_team) | (league_hist["away_team"] == home_team)
            ]
            a_df = league_hist.loc[
                (league_hist["home_team"] == away_team) | (league_hist["away_team"] == away_team)
            ]
            rate = float(np.clip((_ht_over(h_df) + _ht_over(a_df)) / 2, 0.05, 0.95))
            return rate, 1.0 - rate
        return (0.70, 0.30) if thr == 0.5 else (0.35, 0.65)

    # ── 2nd Half Goals O/U ───────────────────────────────────────────────────
    if "2nd half goals" in m:
        thr = 0.5 if "0.5" in market else 1.5
        if "home_goals_ht" in league_hist.columns and "home_goals_ft" in league_hist.columns:
            def _2h_total(df: pd.DataFrame) -> pd.Series:
                h2 = (
                    pd.to_numeric(df["home_goals_ft"], errors="coerce").fillna(0)
                    - pd.to_numeric(df["home_goals_ht"], errors="coerce").fillna(0)
                ).clip(lower=0)
                a2 = (
                    pd.to_numeric(df["away_goals_ft"], errors="coerce").fillna(0)
                    - pd.to_numeric(df["away_goals_ht"], errors="coerce").fillna(0)
                ).clip(lower=0)
                return h2 + a2
            def _2h_over(df: pd.DataFrame) -> float:
                if df.empty:
                    return 0.5
                return float((_2h_total(df) > thr).mean())
            h_df = league_hist.loc[
                (league_hist["home_team"] == home_team) | (league_hist["away_team"] == home_team)
            ]
            a_df = league_hist.loc[
                (league_hist["home_team"] == away_team) | (league_hist["away_team"] == away_team)
            ]
            rate = float(np.clip((_2h_over(h_df) + _2h_over(a_df)) / 2, 0.05, 0.95))
            return rate, 1.0 - rate
        return (0.75, 0.25) if thr == 0.5 else (0.45, 0.55)

    # ── Goals / Corners / Cards O/U ─────────────────────────────────────────
    m_match = re.search(r"(\d+\.?\d*)", m.split("o/u")[-1].strip())
    if m_match is None:
        return 0.5, 0.5
    threshold = float(m_match.group(1))

    if "goals" in m:
        def _total(df: pd.DataFrame) -> pd.Series:
            return (
                pd.to_numeric(df["home_goals_ft"], errors="coerce").fillna(0)
                + pd.to_numeric(df["away_goals_ft"], errors="coerce").fillna(0)
            )
    elif "corners" in m:
        def _total(df: pd.DataFrame) -> pd.Series:
            return (
                pd.to_numeric(df["home_corners"], errors="coerce").fillna(0)
                + pd.to_numeric(df["away_corners"], errors="coerce").fillna(0)
            )
    elif "cards" in m:
        def _total(df: pd.DataFrame) -> pd.Series:
            return (
                pd.to_numeric(df["home_yellow_cards"], errors="coerce").fillna(0)
                + pd.to_numeric(df["away_yellow_cards"], errors="coerce").fillna(0)
                + pd.to_numeric(df.get("home_red_cards", 0), errors="coerce").fillna(0)
                + pd.to_numeric(df.get("away_red_cards", 0), errors="coerce").fillna(0)
            )
    else:
        return 0.5, 0.5

    def _over_rate(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.5
        totals = _total(df)
        valid = totals > 0
        return float((totals[valid] > threshold).mean()) if valid.any() else 0.5

    h_df = league_hist.loc[
        (league_hist["home_team"] == home_team) | (league_hist["away_team"] == home_team)
    ]
    a_df = league_hist.loc[
        (league_hist["home_team"] == away_team) | (league_hist["away_team"] == away_team)
    ]
    rate = float(np.clip((_over_rate(h_df) + _over_rate(a_df)) / 2, 0.05, 0.95))
    return rate, 1.0 - rate


def _build_tickets(
    picks_df: pd.DataFrame,
    legs: int,
    n_tickets: int,
) -> dict[str, pd.DataFrame]:
    """Build N tickets per tier from picks_df.

    A ticket = up to ``legs`` picks, one pick per match (keyed by match_id).
    Within each match the available picks are rotated across tickets to create
    variety.  Returns a dict with keys "conservative", "moderate", "high_risk",
    each a DataFrame with columns:
    ticket_num, leg_num, match, market, pick_label,
    model_prob, odds, combined_odds, hit_probability, expected_roi.
    """
    _COLS = [
        "ticket_num", "leg_num", "match", "market", "pick_label",
        "model_prob", "odds", "combined_odds", "hit_probability", "expected_roi",
        "context",
    ]
    _EMPTY = pd.DataFrame(columns=_COLS)

    if picks_df.empty:
        return {"conservative": _EMPTY, "moderate": _EMPTY, "high_risk": _EMPTY}

    # Group picks by match_id; sort each group by model_prob descending
    groups: dict[str, list[dict]] = {}
    for _, row in picks_df.iterrows():
        mid = str(row["match_id"])
        groups.setdefault(mid, []).append(row.to_dict())
    for mid in groups:
        groups[mid].sort(key=lambda r: float(r.get("model_prob", 0)), reverse=True)

    all_mids = list(groups.keys())
    actual_legs = min(legs, len(all_mids))

    def _best(mid: str, key: str) -> float:
        picks = groups[mid]
        return float(picks[0].get(key, 0.0)) if picks else 0.0

    def _build_tier(sort_key: str) -> pd.DataFrame:
        sorted_mids = sorted(all_mids, key=lambda m: _best(m, sort_key), reverse=True)
        top_mids = sorted_mids[:actual_legs]
        rows: list[dict] = []
        seen: set[str] = set()

        for t in range(1, n_tickets + 1):
            # Rotate picks: ticket t, leg k → index (t-1+k) % len(options)
            ticket_picks: list[dict] = [
                groups[mid][(t - 1 + k) % len(groups[mid])]
                for k, mid in enumerate(top_mids)
            ]
            fp = "|".join(f"{p['match_id']}:{p['pick_label']}" for p in ticket_picks)
            # De-duplicate: try alternate offsets
            if fp in seen:
                for alt in range(1, 12):
                    alt_picks = [
                        groups[mid][(t - 1 + k + alt) % len(groups[mid])]
                        for k, mid in enumerate(top_mids)
                    ]
                    fp2 = "|".join(f"{p['match_id']}:{p['pick_label']}" for p in alt_picks)
                    if fp2 not in seen:
                        ticket_picks, fp = alt_picks, fp2
                        break
            seen.add(fp)

            combined_odds = float(np.prod([p["odds"] for p in ticket_picks]))
            hit_prob = float(np.prod([p["model_prob"] for p in ticket_picks]))
            ev = hit_prob * combined_odds - 1.0

            for leg, pick in enumerate(ticket_picks, 1):
                rows.append({
                    "ticket_num":      t,
                    "leg_num":         leg,
                    "match":           pick["match"],
                    "market":          pick["market"],
                    "pick_label":      pick["pick_label"],
                    "model_prob":      round(float(pick["model_prob"]), 4),
                    "odds":            round(float(pick["odds"]), 2),
                    "combined_odds":   round(combined_odds, 2),
                    "hit_probability": round(hit_prob, 4),
                    "expected_roi":    round(ev, 4),
                    "context":         str(pick.get("context", "")),
                })

        return pd.DataFrame(rows, columns=_COLS) if rows else _EMPTY.copy()

    return {
        "conservative": _build_tier("model_prob"),
        "moderate":     _build_tier("expected_roi"),
        "high_risk":    _build_tier("odds"),
    }


def _render_ticket_table(tier_df: pd.DataFrame) -> pd.DataFrame:
    """Expand ticket DataFrame into a display-ready table.

    Each leg gets its own row.  Ticket-level stats (Combo Odds, Hit %, xROI)
    appear only on the last leg of every ticket.  A blank separator row is
    inserted between tickets for visual clarity.
    """
    _DISPLAY_COLS = [
        "Ticket", "Match", "Market", "Pick", "Prob", "Odds",
        "Combo Odds", "Hit %", "xROI", "📋 Context",
    ]
    if tier_df.empty:
        return pd.DataFrame(columns=_DISPLAY_COLS)

    rows: list[dict] = []
    n_tickets = int(tier_df["ticket_num"].max())

    for t_num in range(1, n_tickets + 1):
        tkt = tier_df[tier_df["ticket_num"] == t_num].sort_values("leg_num")
        n_legs = len(tkt)
        for leg_i, (_, leg) in enumerate(tkt.iterrows(), 1):
            is_last = leg_i == n_legs
            rows.append({
                "Ticket":     f"#{t_num}",
                "Match":      leg["match"],
                "Market":     leg["market"],
                "Pick":       leg["pick_label"],
                "Prob":       f"{leg['model_prob']:.0%}",
                "Odds":       f"{leg['odds']:.2f}",
                "Combo Odds": f"{leg['combined_odds']:.2f}" if is_last else "",
                "Hit %":      f"{leg['hit_probability']:.1%}" if is_last else "",
                "xROI":       f"{leg['expected_roi']:+.1%}" if is_last else "",
                "📋 Context": str(leg.get("context", "")),
            })
        if t_num < n_tickets:
            rows.append({col: "" for col in _DISPLAY_COLS})

    return pd.DataFrame(rows, columns=_DISPLAY_COLS)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_context(
    data_path_str: str,
    as_of_date: date,
    momentum_window: int,
    injuries_str: str,
    contrib_str: str,
    other_str: str,
) -> tuple[dict, str]:
    return build_context(
        Path(data_path_str),
        as_of_date,
        momentum_window,
        Path(injuries_str),
        Path(contrib_str),
        Path(other_str),
    )


@st.cache_resource(show_spinner=False)
def _cached_models(
    _historical: pd.DataFrame,
    _injuries_df: pd.DataFrame,
    _contrib_df: pd.DataFrame,
):
    """Train and cache XGBoost models in-memory.

    Uses cache_resource instead of cache_data so the XGBClassifier objects are
    stored as live Python objects rather than being pickled/unpickled.  Pickling
    XGBClassifier (xgboost ≥ 3.x) inside Streamlit's cache layer triggers a
    sklearn import check that can fail even when scikit-learn is installed.
    """
    match_model = train_match_model(_historical, injuries_df=_injuries_df)
    player_models = train_player_models(_contrib_df)
    return match_model, player_models


def main() -> None:
    st.set_page_config(page_title="Football Bets Tool", page_icon=":soccer:", layout="wide")
    apply_style()

    st.title("Football Bets Tool")
    st.caption("Match intelligence · League standings · Player probabilities")

    tab_bets, tab_match, tab_league = st.tabs(
        ["🎯 Bet Builder", "⚽ Match Center", "📊 League & Players"]
    )

    with st.sidebar:
        st.header("Settings")
        as_of = st.date_input("As-of date", value=date.today())
        momentum_window = st.slider("Momentum window", 3, 12, 5)
        with st.expander("External data files"):
            injuries_file = Path(st.text_input("Injuries CSV", "data/sports/external/injuries.csv"))
            contrib_file = Path(st.text_input("Player contributions CSV", "data/sports/external/player_contributions.csv"))
            other_file = Path(st.text_input("Other competitions CSV", "data/sports/external/other_competitions_matches.csv"))

        st.divider()
        st.subheader("API-Football")
        api_key = st.text_input(
            "API key",
            type="password",
            help=(
                "Optional — used to auto-fetch Starting XI from API-Football.\n\n"
                "How to get your free key:\n"
                "1. Go to api-sports.io\n"
                "2. Click Sign Up → create a free account\n"
                "3. Open Dashboard → copy your API Key\n"
                "Free plan: 100 requests/day (no credit card needed)"
            ),
        )

        st.divider()
        st.subheader("Refresh Data")
        st.caption(
            "Runs in the background — you can keep using the app. "
            "Reload the page once the jobs finish to see updated data."
        )
        refresh_start = st.number_input(
            "Start season", min_value=1995, max_value=2100, value=date.today().year - 20
        )
        refresh_end = st.number_input(
            "End season", min_value=1995, max_value=2100, value=date.today().year
        )
        refresh_min_date = st.date_input(
            "Min match date", value=date(date.today().year - 20, 1, 1)
        )

        if st.button("🔄 Refresh All Data", use_container_width=True):
            pid_m = run_refresh(int(refresh_start), int(refresh_end), refresh_min_date)
            pid_p = run_player_stats_refresh(api_key=api_key, season="2526")
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state["_refresh_ts"] = ts
            st.session_state["_refresh_pids"] = (pid_m, pid_p)
            st.toast(f"Refresh started at {ts} (match PID {pid_m} · player PID {pid_p})", icon="🚀")

        if "_refresh_ts" in st.session_state:
            st.info(
                f"Last refresh started at **{st.session_state['_refresh_ts']}**. "
                "Reload the page when jobs complete to see new data.",
                icon="ℹ️",
            )

        with st.expander("📋 View Refresh Logs"):
            log_tab_m, log_tab_p = st.tabs(["Match Data", "Player Stats"])
            with log_tab_m:
                if MATCHES_LOG_FILE.exists():
                    st.code(MATCHES_LOG_FILE.read_text()[-3000:] or "Empty log.")
                else:
                    st.caption("No match-data log yet.")
            with log_tab_p:
                if PLAYERS_LOG_FILE.exists():
                    st.code(PLAYERS_LOG_FILE.read_text()[-3000:] or "Empty log.")
                else:
                    st.caption("No player-stats log yet.")

        # ── Last-updated metadata display ─────────────────────────────────────
        _meta = load_refresh_metadata()
        if _meta:
            st.divider()
            st.caption("**Last successful refresh**")
            _m_ts = _meta.get("matches_last_fetch", "–")
            _p_ts = _meta.get("players_last_fetch", "–")
            _p_src = _meta.get("players_source", "")
            st.caption(
                f"📊 Matches: `{_m_ts[:16]}`  \n"
                f"👤 Players: `{_p_ts[:16]}` ({_p_src})"
            )

    # ── Auto-refresh if data is stale (once per browser session) ─────────────
    if not st.session_state.get("_auto_refresh_done"):
        st.session_state["_auto_refresh_done"] = True
        _meta = load_refresh_metadata()
        _m_stale = _is_stale(_meta.get("matches_last_fetch"))
        _p_stale = _is_stale(_meta.get("players_last_fetch"))
        if _m_stale or _p_stale:
            _ar_start = date.today().year - 5   # quick 5-year window for auto-refresh
            _ar_end   = date.today().year
            _ar_min   = date(_ar_start, 1, 1)
            if _m_stale:
                run_refresh(_ar_start, _ar_end, _ar_min)
            if _p_stale:
                run_player_stats_refresh(api_key=api_key, season="2526")
            st.toast(
                "Data was stale (>2 h) — background refresh started automatically.",
                icon="🔄",
            )

    data_path = DEFAULT_DATA_FILE
    with st.spinner("Loading data…"):
        context, err = _cached_context(
            str(data_path), as_of, momentum_window,
            str(injuries_file), str(contrib_file), str(other_file),
        )
    if err:
        # ── First-run setup: offer to download data from football-data.co.uk ──
        is_missing = "not found" in err.lower() or "no such file" in err.lower()
        if is_missing:
            st.title("⚽ Football Bets — First-time Setup")
            st.info(
                "The match dataset is not present yet. "
                "Click the button below to download it from "
                "[football-data.co.uk](https://www.football-data.co.uk) "
                "(free, no login required). This takes about 30–60 seconds.",
                icon="📥",
            )
            if st.button("⬇️ Download match data now", type="primary"):
                _out_dir = DEFAULT_DATA_FILE.parent.parent  # data/sports/
                with st.spinner("Downloading match data from football-data.co.uk…"):
                    fetch_err = _fetch_data_sync(
                        _out_dir,
                        status_fn=lambda msg: st.toast(msg),
                    )
                if fetch_err:
                    st.error(f"Download failed: {fetch_err}")
                else:
                    st.success("Data downloaded successfully! Reloading…")
                    st.rerun()
        else:
            st.error(f"Could not load data: {err}")
        st.stop()

    try:
        with st.spinner("Training models…"):
            match_model, player_models = _cached_models(
                context["historical"], context["injuries_df"], context["contrib_df"]
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"XGBoost training failed. Install xgboost and retry. Details: {exc}")
        st.stop()

    with tab_match:
        st.subheader("Match Center")

        # ── 1. League + team selection ───────────────────────────────────────
        snapshot = context["snapshot"]
        league_names = sorted(
            l for l in snapshot["league_name"].dropna().unique()
            if l in SUPPORTED_LEAGUES
        )
        league = st.selectbox("Select League", league_names)
        teams = sorted(snapshot.loc[snapshot["league_name"] == league, "team"].dropna().unique())

        c1, c2 = st.columns(2)
        with c1:
            home_team = st.selectbox("Home team", teams)
        with c2:
            away_candidates = [t for t in teams if t != home_team]
            away_team = st.selectbox("Away team", away_candidates)

        # ── 2. Always-visible match preview ─────────────────────────────────
        current_snap = context["current_snapshot"]
        h_row = _team_row(current_snap, league, home_team)
        a_row = _team_row(current_snap, league, away_team)
        h_form = team_last5_form(context["historical"], home_team, league, context["as_of_ts"])
        a_form = team_last5_form(context["historical"], away_team, league, context["as_of_ts"])

        st.markdown("---")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"### 🏠 {home_team}")
            st.metric("Position", f"#{int(h_row.get('position', '–'))}" if not h_row.empty else "–")
            st.metric("Points (this season)", int(h_row.get("points", 0)) if not h_row.empty else "–")
            st.metric("Goals scored / conceded (season)", f"{int(h_row.get('goals_for',0))} / {int(h_row.get('goals_against',0))}" if not h_row.empty else "–")
            st.metric("Form (last 5)", h_form)
            st.metric("Avg pts last 5", f"{h_row.get('last_points_pg', 0.0):.2f}" if not h_row.empty else "–")
            st.metric("Goals scored last 5 (pg)", f"{h_row.get('last_goals_for_pg', 0.0):.2f}" if not h_row.empty else "–")
            st.metric("Goals conceded last 5 (pg)", f"{h_row.get('last_goals_against_pg', 0.0):.2f}" if not h_row.empty else "–")
        with mc2:
            st.markdown(f"### ✈️ {away_team}")
            st.metric("Position", f"#{int(a_row.get('position', '–'))}" if not a_row.empty else "–")
            st.metric("Points (this season)", int(a_row.get("points", 0)) if not a_row.empty else "–")
            st.metric("Goals scored / conceded (season)", f"{int(a_row.get('goals_for',0))} / {int(a_row.get('goals_against',0))}" if not a_row.empty else "–")
            st.metric("Form (last 5)", a_form)
            st.metric("Avg pts last 5", f"{a_row.get('last_points_pg', 0.0):.2f}" if not a_row.empty else "–")
            st.metric("Goals scored last 5 (pg)", f"{a_row.get('last_goals_for_pg', 0.0):.2f}" if not a_row.empty else "–")
            st.metric("Goals conceded last 5 (pg)", f"{a_row.get('last_goals_against_pg', 0.0):.2f}" if not a_row.empty else "–")

        # ── 3. Always-visible player intelligence ────────────────────────────
        st.markdown("---")
        st.markdown("#### Player Intelligence")
        insights = player_match_insights(
            home_team=home_team,
            away_team=away_team,
            as_of_date=context["as_of_ts"],
            injuries_df=context["injuries_df"],
            contrib_df=context["contrib_df"],
            top_n=8,
        )
        pi1, pi2, pi3 = st.columns(3)
        with pi1:
            st.markdown("🚑 **Important injuries**")
            if insights["injured_players"].empty:
                st.caption("No injury data available.")
            else:
                st.dataframe(insights["injured_players"], use_container_width=True, hide_index=True)
        with pi2:
            st.markdown("⚽ **Likely scorers**")
            if insights["likely_scorers"].empty:
                st.caption("No contribution data available.")
            else:
                st.dataframe(insights["likely_scorers"], use_container_width=True, hide_index=True)
        with pi3:
            st.markdown("🟨 **Likely cards**")
            if insights["likely_cards"].empty:
                st.caption("No contribution data available.")
            else:
                st.dataframe(insights["likely_cards"], use_container_width=True, hide_index=True)

        # ── 4. Auto-suggested odds ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Odds")
        st.caption("Auto-calculated from model (position · form · H2H). Edit freely.")

        combo_key = f"{league}|{home_team}|{away_team}"
        if st.session_state.get("_odds_combo") != combo_key and match_model is not None:
            st.session_state["_odds_combo"] = combo_key
            try:
                oh, od, oa = _auto_suggest_odds(context, match_model, league, home_team, away_team)
                st.session_state["_odd_h"] = oh
                st.session_state["_odd_d"] = od
                st.session_state["_odd_a"] = oa
            except Exception:
                st.session_state.setdefault("_odd_h", 2.10)
                st.session_state.setdefault("_odd_d", 3.20)
                st.session_state.setdefault("_odd_a", 3.10)

        odds_cols = st.columns(3)
        with odds_cols[0]:
            odd_h = st.number_input("Home odd (1)", min_value=1.01, step=0.05, key="_odd_h")
        with odds_cols[1]:
            odd_d = st.number_input("Draw odd (X)", min_value=1.01, step=0.05, key="_odd_d")
        with odds_cols[2]:
            odd_a = st.number_input("Away odd (2)", min_value=1.01, step=0.05, key="_odd_a")

        # ── 5. Starting XI ───────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("Starting XI (optional — fetch online or enter manually)", expanded=False):
            xi_key = f"xi_{league}_{home_team}_{away_team}"
            if xi_key not in st.session_state:
                st.session_state[xi_key] = {"home": [], "away": []}

            if st.button("Fetch probable XI online"):
                home_xi, away_xi, msg = fetch_probable_xi_api_football(api_key, home_team, away_team)
                if home_xi:
                    st.session_state[xi_key]["home"] = home_xi
                if away_xi:
                    st.session_state[xi_key]["away"] = away_xi
                st.info(msg)

            d1, d2 = st.columns(2)
            with d1:
                home_xi_text = st.text_area(
                    f"{home_team} XI (comma-separated)",
                    value=", ".join(st.session_state[xi_key]["home"]),
                    height=110,
                )
            with d2:
                away_xi_text = st.text_area(
                    f"{away_team} XI (comma-separated)",
                    value=", ".join(st.session_state[xi_key]["away"]),
                    height=110,
                )
        home_xi = parse_lineup_text(home_xi_text if "home_xi_text" in dir() else "")
        away_xi = parse_lineup_text(away_xi_text if "away_xi_text" in dir() else "")

        # ── 6. H2H + fatigue controls ────────────────────────────────────────
        h2h_years = st.slider("H2H years look-back", min_value=1, max_value=20, value=5)
        fatigue_cols = st.columns(2)
        with fatigue_cols[0]:
            home_big_games = st.number_input(f"{home_team} big games last 8 days", min_value=0, max_value=6, value=0)
        with fatigue_cols[1]:
            away_big_games = st.number_input(f"{away_team} big games last 8 days", min_value=0, max_value=6, value=0)

        # ── 7. Run prediction ─────────────────────────────────────────────────
        st.markdown("---")
        if st.button("▶ Run full prediction", use_container_width=True):
            if match_model is None:
                st.warning("Not enough data to train match XGBoost model.")
            else:
                home_strength = lineup_strength(home_team, home_xi, context["contrib_df"], context["as_of_ts"])
                away_strength = lineup_strength(away_team, away_xi, context["contrib_df"], context["as_of_ts"])
                features, h2h = build_feature_vector(
                    context=context,
                    league_name=league,
                    home_team=home_team,
                    away_team=away_team,
                    h2h_years=h2h_years,
                    home_lineup_strength=home_strength,
                    away_lineup_strength=away_strength,
                    home_big_games_8d=float(home_big_games),
                    away_big_games_8d=float(away_big_games),
                )

                probs = predict_match_proba(match_model, features)
                pred_label = max(probs, key=probs.get)
                reasons = explain_factors(features, home_team, away_team)

                st.markdown('<div class="metric">', unsafe_allow_html=True)
                st.write(f"Predicted outcome: **{outcome_name(pred_label)}**")
                st.write(f"Probabilities → 1: {probs['H']:.2%} | X: {probs['D']:.2%} | 2: {probs['A']:.2%}")
                st.write(f"Key factors: {reasons}")
                st.markdown("</div>", unsafe_allow_html=True)

                risk = choose_risk_bets(
                    probs=probs,
                    odds={"H": float(odd_h), "D": float(odd_d), "A": float(odd_a)},
                    reasons=reasons,
                )
                cols = st.columns(3)
                for col, item in zip(cols, risk):
                    with col:
                        st.metric(
                            label=item["tier"],
                            value=outcome_name(str(item["pick"])),
                            delta=f"P={float(item['prob']):.1%} | EV={float(item['ev']):+.2f}",
                            help=str(item["tip"]),
                        )

                st.markdown(f"#### Past H2H — last {h2h_years} years")
                hist = context["historical"]
                min_date = context["as_of_ts"] - pd.Timedelta(days=365 * h2h_years)
                h2h_rows = hist.loc[
                    (hist["league_name"] == league)
                    & (hist["match_date"] >= min_date)
                    & (
                        ((hist["home_team"] == home_team) & (hist["away_team"] == away_team))
                        | ((hist["home_team"] == away_team) & (hist["away_team"] == home_team))
                    )
                ].sort_values("match_date", ascending=False)

                st.caption(
                    f"H2H: {int(h2h['h2h_matches'])} matches · "
                    f"Home win {h2h['h2h_home_win_rate']:.1%} · "
                    f"Draw {h2h['h2h_draw_rate']:.1%} · "
                    f"Away win {h2h['h2h_away_win_rate']:.1%}"
                )
                safe_h2h_cols = [c for c in [
                    "match_date", "season_label", "home_team", "away_team",
                    "home_goals_ft", "away_goals_ft", "result_ft",
                    "home_corners", "away_corners",
                    "home_yellow_cards", "away_yellow_cards",
                    "home_fouls", "away_fouls",
                ] if c in h2h_rows.columns]
                st.dataframe(h2h_rows[safe_h2h_cols], use_container_width=True, hide_index=True)

    with tab_league:
        st.subheader("League & Players")
        current_snapshot = context["current_snapshot"].copy()
        leagues = sorted(
            l for l in current_snapshot["league_name"].dropna().unique()
            if l in SUPPORTED_LEAGUES
        )
        league = st.selectbox("Select league", leagues, key="page2_league")
        league_table = current_snapshot.loc[current_snapshot["league_name"] == league].sort_values(
            ["position", "points", "goal_diff"], ascending=[True, False, False]
        )

        st.markdown(f"**Standings — {context['current_season']} season**")

        # Add last-5 form string for each team
        league_table = league_table.copy()
        league_table["form_last5"] = league_table["team"].apply(
            lambda t: team_last5_form(context["historical"], t, league, context["as_of_ts"])
        )

        standings_cols = [
            "position",
            "team",
            "matches",
            "points",
            "goals_for",
            "goals_against",
            "goal_diff",
            "form_last5",
            "home_ppg",
            "away_ppg",
            "last_points_pg",
            "last_goals_for_pg",
            "last_goals_against_pg",
            "injury_count",
            "injury_impact",
        ]
        st.dataframe(league_table[standings_cols], use_container_width=True, hide_index=True)

        teams = sorted(current_snapshot.loc[current_snapshot["league_name"] == league, "team"].dropna().unique())
        team = st.selectbox("Check players for team", teams, key="page2_team")

        st.markdown("**Player info and likelihood (XGBoost)**")
        if player_models is None:
            st.info("Not enough player contribution data to train player XGBoost models.")
        else:
            player_probs = player_probabilities_for_team(
                team=team,
                contrib_df=context["contrib_df"],
                bundle=player_models,
                as_of_date=context["as_of_ts"],
                top_n=20,
            )
            if player_probs.empty:
                st.info("No player contribution records available for selected team.")
            else:
                st.dataframe(player_probs, use_container_width=True, hide_index=True)

        team_insights = player_match_insights(
            home_team=team,
            away_team=team,
            as_of_date=context["as_of_ts"],
            injuries_df=context["injuries_df"],
            contrib_df=context["contrib_df"],
            top_n=20,
        )
        injured = team_insights["injured_players"]
        if not injured.empty:
            injured = injured.loc[injured["team"] == team]

        st.markdown("**Important injured players**")
        st.dataframe(injured, use_container_width=True, hide_index=True)

        # ── Player Season Stats (Understat) ───────────────────────────────────
        st.markdown("---")
        st.markdown("#### Player Season Stats · 2025-26")

        _player_stats_df = load_player_stats(str(PLAYER_STATS_FILE))

        if _player_stats_df.empty:
            st.info(
                "No player stats cached yet. Click **🔄 Refresh All Data** in the sidebar. "
                "Uses API-Football if a key is set (all 6 leagues), otherwise falls back to "
                "Understat (Big 5 only — Primeira Liga not available without an API key)."
            )
        else:
            # Filter to selected team; try exact match first, then case-insensitive
            _ps_team = _player_stats_df.loc[_player_stats_df["team"] == team]
            if _ps_team.empty:
                _ps_team = _player_stats_df.loc[
                    _player_stats_df["team"].str.lower() == team.lower()
                ]

            if _ps_team.empty:
                st.info(
                    f"No Understat data for **{team}**. "
                    "This team may not be in the Big 5 leagues or the name differs slightly."
                )
            else:
                _ps_display = (
                    _ps_team[
                        [
                            "player",
                            "position",
                            "matches",
                            "minutes",
                            "goals",
                            "xg",
                            "assists",
                            "xa",
                            "shots",
                            "key_passes",
                            "yellow_cards",
                            "red_cards",
                        ]
                    ]
                    .rename(
                        columns={
                            "player": "Player",
                            "position": "Pos",
                            "matches": "MP",
                            "minutes": "Min",
                            "goals": "Goals",
                            "xg": "xG",
                            "assists": "Ast",
                            "xa": "xA",
                            "shots": "Shots",
                            "key_passes": "Chances Created",
                            "yellow_cards": "YC",
                            "red_cards": "RC",
                        }
                    )
                    .sort_values("Goals", ascending=False)
                    .reset_index(drop=True)
                )

                # Round floats for readability
                for _col in ["xG", "xA"]:
                    if _col in _ps_display.columns:
                        _ps_display[_col] = _ps_display[_col].round(2)

                st.dataframe(
                    _ps_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Goals": st.column_config.NumberColumn(format="%d"),
                        "Ast": st.column_config.NumberColumn(format="%d"),
                        "Shots": st.column_config.NumberColumn(format="%d"),
                        "Chances Created": st.column_config.NumberColumn(format="%d"),
                        "YC": st.column_config.NumberColumn(format="%d"),
                        "RC": st.column_config.NumberColumn(format="%d"),
                        "MP": st.column_config.NumberColumn(format="%d"),
                        "Min": st.column_config.NumberColumn(format="%d"),
                        "xG": st.column_config.NumberColumn(format="%.2f"),
                        "xA": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

        # ── Team Season Stats Panel ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Season Stats — Corners · Fouls · Cards")
        rank_cutoff = st.slider(
            "Rank cutoff (top N vs the rest)", 4, 12, 8, key="page2_rank_cutoff"
        )

        # Current-season matches for the selected team in the selected league
        cur_hist = context["historical"].loc[
            context["historical"]["season_label"] == context["current_season"]
        ]
        team_m = cur_hist.loc[
            (cur_hist["league_name"] == league)
            & (
                (cur_hist["home_team"] == team)
                | (cur_hist["away_team"] == team)
            )
        ].copy()

        if team_m.empty:
            st.info("No current-season match data found for this team.")
        else:
            stat_rows: list[dict] = []
            for _, row in team_m.iterrows():
                is_home = row["home_team"] == team
                opponent = row["away_team"] if is_home else row["home_team"]

                opp_snap = _team_row(current_snapshot, league, opponent)
                opp_pos = int(opp_snap.get("position", 99)) if not opp_snap.empty else 99

                r = row["result_ft"]
                if is_home:
                    pts = 3 if r == "H" else (1 if r == "D" else 0)
                    side = "H"
                else:
                    pts = 3 if r == "A" else (1 if r == "D" else 0)
                    side = "A"

                stat_rows.append(
                    {
                        "is_home": is_home,
                        "opponent": opponent,
                        "opp_position": opp_pos,
                        "points": pts,
                        "corners": pd.to_numeric(
                            row.get("home_corners" if is_home else "away_corners"),
                            errors="coerce",
                        ),
                        "fouls": pd.to_numeric(
                            row.get("home_fouls" if is_home else "away_fouls"),
                            errors="coerce",
                        ),
                        "yellows": pd.to_numeric(
                            row.get(
                                "home_yellow_cards" if is_home else "away_yellow_cards"
                            ),
                            errors="coerce",
                        ),
                        "reds": pd.to_numeric(
                            row.get(
                                "home_red_cards" if is_home else "away_red_cards"
                            ),
                            errors="coerce",
                        ),
                    }
                )

            stats_df = pd.DataFrame(stat_rows)
            for c in ["corners", "fouls", "yellows", "reds"]:
                stats_df[c] = pd.to_numeric(stats_df[c], errors="coerce")

            def _agg_seg(df: pd.DataFrame, label: str) -> dict:
                if df.empty:
                    return {
                        "Segment": label,
                        "Matches": 0,
                        "PPG": "–",
                        "W-D-L": "–",
                        "Corners/g": "–",
                        "Fouls/g": "–",
                        "Yellows/g": "–",
                        "Reds/g": "–",
                    }
                w = int((df["points"] == 3).sum())
                d = int((df["points"] == 1).sum())
                l_ = int((df["points"] == 0).sum())

                def _fmt(col: str, dec: int = 1) -> str:
                    return (
                        f"{df[col].mean():.{dec}f}"
                        if df[col].notna().any()
                        else "–"
                    )

                return {
                    "Segment": label,
                    "Matches": len(df),
                    "PPG": f"{df['points'].mean():.2f}",
                    "W-D-L": f"{w}-{d}-{l_}",
                    "Corners/g": _fmt("corners"),
                    "Fouls/g": _fmt("fouls"),
                    "Yellows/g": _fmt("yellows"),
                    "Reds/g": _fmt("reds", dec=2),
                }

            segs = [
                _agg_seg(stats_df, "All matches"),
                _agg_seg(stats_df.loc[stats_df["is_home"]], "Home"),
                _agg_seg(stats_df.loc[~stats_df["is_home"]], "Away"),
                _agg_seg(
                    stats_df.loc[stats_df["opp_position"] <= rank_cutoff],
                    f"vs Top {rank_cutoff}",
                ),
                _agg_seg(
                    stats_df.loc[stats_df["opp_position"] > rank_cutoff],
                    f"vs Below {rank_cutoff}th",
                ),
            ]
            st.dataframe(pd.DataFrame(segs), use_container_width=True, hide_index=True)

            with st.expander("Full match log — corners, fouls, cards"):
                log = stats_df[
                    [
                        "opponent",
                        "opp_position",
                        "is_home",
                        "points",
                        "corners",
                        "fouls",
                        "yellows",
                        "reds",
                    ]
                ].rename(
                    columns={
                        "opp_position": "opp_pos",
                        "is_home": "home?",
                        "yellows": "yellow_cards",
                        "reds": "red_cards",
                    }
                )
                st.dataframe(log, use_container_width=True, hide_index=True)

    # =========================================================================
    # PAGE 3 — BET BUILDER
    # =========================================================================
    with tab_bets:
        # ── Guard: reset slider session-state values that fall outside new ranges ─
        if st.session_state.get("bb_n", 3) > 10:
            st.session_state["bb_n"] = 3
        if st.session_state.get("bb_legs", 8) > 12:
            st.session_state["bb_legs"] = 8

        st.subheader("🎫 Bet Builder — Tickets")
        st.caption(
            "Select leagues + date range → fetch upcoming games → configure "
            "markets per match → generate Conservative / Moderate / High-Risk **tickets**. "
            "Each ticket contains one pick per match (one pick from each of up to 8 matches)."
        )

        # ── Row 1: date range + multi-select leagues ──────────────────────────
        bb_c1, bb_c2, bb_c3 = st.columns([1, 1, 2])
        with bb_c1:
            bb_start = st.date_input("From", value=date.today(), key="bb_start")
        with bb_c2:
            bb_end = st.date_input(
                "To", value=date.today() + timedelta(days=7), key="bb_end"
            )
        with bb_c3:
            all_league_names = sorted(
                l for l in context["historical"]["league_name"].dropna().unique()
                if l in SUPPORTED_LEAGUES
            )
            bb_leagues = st.multiselect(
                "Leagues",
                options=all_league_names,
                default=all_league_names,
                key="bb_leagues",
                help="Select any number of leagues. All 6 are selected by default.",
            )

        # ── Row 2: ticket settings ────────────────────────────────────────────
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            bb_legs = st.slider(
                "Legs per ticket", 2, 12, 8, key="bb_legs",
                help=(
                    "How many picks form one ticket (one pick per match).\n\n"
                    "• 8 legs (default) → one pick from each of 8 different matches\n"
                    "• More legs = higher combined payout but harder to win.\n"
                    "• Each leg must come from a different match."
                ),
            )
        with sc2:
            bb_n = st.slider(
                "Tickets per tier", 1, 10, 3, key="bb_n",
                help=(
                    "Number of tickets shown per tier "
                    "(Conservative / Moderate / High Risk).\n\n"
                    "Each ticket rotates picks for variety across the same matches."
                ),
            )
        with sc3:
            bb_min_prob = st.slider(
                "Min single-pick probability",
                0.15,
                0.75,
                0.35,
                step=0.01,
                key="bb_minp",
                help=(
                    "Minimum model confidence required for a pick to qualify.\n\n"
                    "• 0.35 → picks the model rates ≥ 35% likely\n"
                    "• Higher = fewer but more reliable picks\n"
                    "• Lower = more picks and market variety\n\n"
                    "For new markets (Score First, Player to Score) try 0.20–0.35."
                ),
            )

        # ── Row 3: markets to consider ────────────────────────────────────────
        bb_markets = st.multiselect(
            "Markets to consider",
            options=MARKET_OPTIONS,
            default=["1X2", "Goals O/U 2.5", "Corners O/U 9.5", "Cards O/U 3.5"],
            key="bb_markets",
            help=(
                "The app generates picks for **every selected market** across **all matches**.\n\n"
                "Each ticket leg can freely mix markets — e.g. Ticket 1 might have:\n"
                "• Match A → Corners O/U 9.5 Over\n"
                "• Match B → Score: Harry Kane\n"
                "• Match C → Home (1X2)\n\n"
                "More markets = more rotation variety across tickets."
            ),
        )

        st.markdown("---")

        if not bb_leagues:
            st.info("👆 Select at least one league above to continue.")
        elif not bb_markets:
            st.info("👆 Select at least one market above to continue.")
        else:
            # ── Session-state key — resets stored fixtures when config changes ─
            fetch_key = f"{bb_start}|{bb_end}|{'|'.join(sorted(bb_leagues))}"
            if st.session_state.get("_bb_fetch_key") != fetch_key:
                st.session_state["_bb_fetch_key"] = fetch_key
                st.session_state["_bb_fixtures"] = None
                st.session_state["_bb_fetch_msg"] = None

            # ── Fetch button + status ─────────────────────────────────────────
            fc1, fc2 = st.columns([1, 3])
            with fc1:
                fetch_btn = st.button(
                    "🔍 Fetch Upcoming Games",
                    key="bb_fetch",
                    use_container_width=True,
                    help=(
                        "With API key: fetches real upcoming fixtures from API-Football.\n"
                        "Without API key: loads matches from the local dataset."
                    ),
                )
            with fc2:
                stored_msg = st.session_state.get("_bb_fetch_msg")
                if stored_msg:
                    st.info(stored_msg)
                elif st.session_state.get("_bb_fixtures") is None:
                    st.caption(
                        "Add an **API-Football key** in the sidebar for live upcoming "
                        "fixtures. Without it, the local dataset is used as fallback."
                    )

            # ── Execute fetch ─────────────────────────────────────────────────
            if fetch_btn:
                with st.spinner("Fetching fixtures…"):
                    fetched_df = pd.DataFrame()
                    fetch_msg = ""
                    source_used = ""

                    # 1️⃣  ESPN — free, no key needed, covers all 6 leagues
                    fetched_df, fetch_msg = fetch_upcoming_fixtures_espn(
                        bb_leagues, bb_start, bb_end
                    )
                    if not fetched_df.empty:
                        source_used = "ESPN"

                    # 2️⃣  API-Football — richer data, needs key
                    if fetched_df.empty and api_key.strip():
                        fetched_df, fetch_msg = fetch_upcoming_fixtures_api(
                            api_key, bb_leagues, bb_start, bb_end
                        )
                        if not fetched_df.empty:
                            source_used = "API-Football"

                    # 3️⃣  Local dataset fallback
                    if fetched_df.empty:
                        all_m = context.get("all_matches", context["historical"])
                        mask = (
                            (all_m["match_date"].dt.date >= bb_start)
                            & (all_m["match_date"].dt.date <= bb_end)
                            & (all_m["league_name"].isin(bb_leagues))
                        )
                        fallback = all_m.loc[mask].copy()
                        n_fb = len(fallback)
                        fetch_msg = (
                            f"📂 Loaded {n_fb} match(es) from local dataset "
                            "(ESPN returned nothing for this date range — "
                            "try dates within the next 2 weeks for live fixtures)."
                        )
                        fetched_df = fallback
                        source_used = "local dataset"

                    if source_used:
                        fetch_msg = f"[{source_used}] " + fetch_msg

                    st.session_state["_bb_fixtures"] = fetched_df
                    st.session_state["_bb_fetch_msg"] = fetch_msg
                    st.rerun()

            # ── Render match table (if fixtures loaded) ───────────────────────
            fixtures_df: pd.DataFrame | None = st.session_state.get("_bb_fixtures")

            if fixtures_df is not None:
                if fixtures_df.empty:
                    st.warning(
                        "No matches found for the selected leagues / date range. "
                        "Try different dates or refresh data from the sidebar."
                    )
                else:
                    # Tag upcoming vs completed
                    rf_col = fixtures_df.get("result_ft", pd.Series(dtype=object))
                    upcoming_mask = ~fixtures_df["result_ft"].isin(RESULT_VALUES) if "result_ft" in fixtures_df.columns else pd.Series([True] * len(fixtures_df), index=fixtures_df.index)
                    n_up = int(upcoming_mask.sum())
                    n_comp = len(fixtures_df) - n_up
                    stat_parts = []
                    if n_up:
                        stat_parts.append(f"✅ **{n_up}** upcoming")
                    if n_comp:
                        stat_parts.append(f"📋 **{n_comp}** completed (historical)")
                    st.markdown("  ·  ".join(stat_parts))

                    # Build editable match table (market/pick selection is now global)
                    pick_rows: list[dict] = []
                    for _, row in fixtures_df.iterrows():
                        rf = row.get("result_ft", None)
                        is_upcoming = (rf not in RESULT_VALUES) if pd.notna(rf) else True
                        status_lbl = "Upcoming" if is_upcoming else f"Played ({rf})"
                        pick_rows.append({
                            "include": True,
                            "date":    str(row["match_date"].date()),
                            "league":  str(row.get("league_name", "")),
                            "home":    str(row.get("home_team", "")),
                            "away":    str(row.get("away_team", "")),
                            "status":  status_lbl,
                        })

                    pick_df = pd.DataFrame(pick_rows)
                    tbl_key = f"bb_tbl_{fetch_key}"

                    st.caption(
                        f"**{len(pick_df)} match(es) loaded** — "
                        f"tick the matches to include, then click **🎫 Generate Tickets**. "
                        f"Picks will be generated for all **{len(bb_markets)} selected market(s)**."
                    )
                    edited_picks = st.data_editor(
                        pick_df,
                        column_config={
                            "include": st.column_config.CheckboxColumn(
                                "✓", default=True, width="small"
                            ),
                            "date":   st.column_config.TextColumn("Date",   disabled=True, width="small"),
                            "league": st.column_config.TextColumn("League", disabled=True),
                            "home":   st.column_config.TextColumn("Home",   disabled=True),
                            "away":   st.column_config.TextColumn("Away",   disabled=True),
                            "status": st.column_config.TextColumn("Status", disabled=True, width="small"),
                        },
                        hide_index=True,
                        use_container_width=True,
                        num_rows="fixed",
                        key=tbl_key,
                    )

                    st.markdown("---")
                    if st.button(
                        "🎫 Generate Tickets",
                        use_container_width=True,
                        key="bb_gen",
                    ):
                        included = edited_picks.loc[
                            edited_picks["include"]
                        ].reset_index(drop=True)

                        if included.empty:
                            st.warning("Please include at least **1** match to generate tickets.")
                        else:
                            pick_records: list[dict] = []
                            margin = 0.05
                            _pstats = load_player_stats(str(PLAYER_STATS_FILE))

                            with st.spinner("Computing model probabilities…"):
                                for _, irow in included.iterrows():
                                    home_t   = str(irow["home"])
                                    away_t   = str(irow["away"])
                                    league_n = str(irow["league"])
                                    mid      = f"{irow['date']}|{league_n}|{home_t}|{away_t}"
                                    match_lbl = f"{home_t} vs {away_t}"

                                    # Cache 1X2 proba once per match (used by multiple markets)
                                    _p1x2_cache: dict[str, float] | None = None

                                    def _get_1x2() -> dict[str, float]:
                                        nonlocal _p1x2_cache
                                        if _p1x2_cache is None:
                                            try:
                                                feats, _ = build_feature_vector(
                                                    context=context,
                                                    league_name=league_n,
                                                    home_team=home_t,
                                                    away_team=away_t,
                                                    h2h_years=5,
                                                    home_lineup_strength=0.0,
                                                    away_lineup_strength=0.0,
                                                    home_big_games_8d=0.0,
                                                    away_big_games_8d=0.0,
                                                )
                                                _p1x2_cache = predict_match_proba(match_model, feats)
                                            except Exception:
                                                _p1x2_cache = {"H": 0.40, "D": 0.25, "A": 0.35}
                                        return _p1x2_cache

                                    # Generate picks for EVERY selected market
                                    for market in bb_markets:

                                        def _add_pick(label: str, prob: float, _mkt: str = market) -> None:
                                            p = float(np.clip(prob * 0.85, 0.01, 0.99))
                                            oddsv = round(max(1.01, (1 / p) * (1 - margin)), 2)
                                            try:
                                                ctx = _pick_context(
                                                    context["historical"],
                                                    home_t,
                                                    away_t,
                                                    _mkt,
                                                    label,
                                                    league_n,
                                                    context["as_of_ts"],
                                                )
                                            except Exception:
                                                ctx = ""
                                            pick_records.append({
                                                "match_id":     mid,
                                                "match":        match_lbl,
                                                "league":       league_n,
                                                "market":       _mkt,
                                                "pick_label":   label,
                                                "model_prob":   p,
                                                "odds":         oddsv,
                                                "edge":         p - 1.0 / oddsv,
                                                "expected_roi": p * oddsv - 1.0,
                                                "context":      ctx,
                                            })

                                        # ── 1X2 via XGBoost ──────────────────
                                        if market == "1X2":
                                            p1x2 = _get_1x2()
                                            lbl_map = {"H": "Home (1)", "D": "Draw (X)", "A": "Away (2)"}
                                            for k, lbl in lbl_map.items():
                                                _add_pick(lbl, p1x2[k])

                                        # ── 1st Half Result ───────────────────
                                        elif market == "1st Half Result":
                                            ph, pd_, pa = _compute_ht_result_proba(
                                                context["historical"], home_t, away_t,
                                                league_n, context["as_of_ts"],
                                            )
                                            _add_pick("HT Home (1)", ph)
                                            _add_pick("HT Draw (X)", pd_)
                                            _add_pick("HT Away (2)", pa)

                                        # ── Score First ───────────────────────
                                        elif market == "Score First":
                                            home_sf, away_sf = _compute_score_first_proba(
                                                context["historical"], home_t, away_t,
                                                league_n, context["as_of_ts"],
                                            )
                                            _add_pick(f"Score First — {home_t}", home_sf)
                                            _add_pick(f"Score First — {away_t}", away_sf)

                                        # ── Win Both Halves ───────────────────
                                        elif market == "Win Both Halves":
                                            home_wbh, away_wbh = _compute_win_both_halves_proba(
                                                context["historical"], home_t, away_t,
                                                league_n, context["as_of_ts"],
                                            )
                                            _add_pick(f"Win Both — {home_t}", home_wbh)
                                            _add_pick(f"Win Both — {away_t}", away_wbh)

                                        # ── Player to Score ───────────────────
                                        elif market == "Player to Score":
                                            for plbl, pprob in _get_player_score_picks(_pstats, home_t, away_t):
                                                _add_pick(plbl, pprob)
                                            if not _get_player_score_picks(_pstats, home_t, away_t):
                                                _add_pick("Player to Score (Top Scorer)", 0.45)

                                        # ── Empirical O/U + BTTS + HT Goals ──
                                        else:
                                            over_p, under_p = estimate_market_proba(
                                                context["historical"], home_t, away_t,
                                                market, league_n, context["as_of_ts"],
                                            )
                                            _add_pick(f"{market} — Over",  over_p)
                                            _add_pick(f"{market} — Under", under_p)

                            if not pick_records:
                                st.warning(
                                    "No picks could be generated. "
                                    "Check that the team names exist in the dataset."
                                )
                            else:
                                all_picks = pd.DataFrame(pick_records)
                                all_picks = all_picks.loc[
                                    all_picks["model_prob"] >= bb_min_prob
                                ].reset_index(drop=True)

                                if all_picks.empty:
                                    st.warning(
                                        f"No picks pass the {bb_min_prob:.0%} probability "
                                        "threshold. Lower the "
                                        "'Min single-pick probability' slider."
                                    )
                                else:
                                    n_matches = all_picks["match_id"].nunique()
                                    st.success(
                                        f"**{len(all_picks)} pick(s)** across "
                                        f"**{n_matches} match(es)** qualify "
                                        f"(prob ≥ {bb_min_prob:.0%}) — "
                                        f"building **{bb_n}** ticket(s) per tier "
                                        f"with up to **{bb_legs}** legs each."
                                    )
                                    with st.expander("Qualifying picks"):
                                        st.dataframe(
                                            all_picks[[
                                                "match", "market", "pick_label",
                                                "model_prob", "odds", "expected_roi",
                                            ]],
                                            use_container_width=True,
                                            hide_index=True,
                                        )

                                    tickets = _build_tickets(all_picks, bb_legs, bb_n)

                                    if all(df.empty for df in tickets.values()):
                                        st.warning(
                                            "No valid tickets could be built — "
                                            "all picks may be from the same match."
                                        )
                                    else:
                                        tier_tabs = st.tabs([
                                            "🟢 Conservative",
                                            "🟡 Moderate",
                                            "🔴 High Risk",
                                        ])
                                        with tier_tabs[0]:
                                            st.caption(
                                                "Matches sorted by highest hit-probability "
                                                "(safest legs first)"
                                            )
                                            st.dataframe(
                                                _render_ticket_table(tickets["conservative"]),
                                                use_container_width=True,
                                                hide_index=True,
                                            )
                                        with tier_tabs[1]:
                                            st.caption(
                                                "Matches sorted by best expected ROI "
                                                "(balanced value + probability)"
                                            )
                                            st.dataframe(
                                                _render_ticket_table(tickets["moderate"]),
                                                use_container_width=True,
                                                hide_index=True,
                                            )
                                        with tier_tabs[2]:
                                            st.caption(
                                                "Matches sorted by highest individual odds "
                                                "(maximum payout, higher variance)"
                                            )
                                            st.dataframe(
                                                _render_ticket_table(tickets["high_risk"]),
                                                use_container_width=True,
                                                hide_index=True,
                                            )

    st.caption("Decision support only. Betting carries financial risk.")


if __name__ == "__main__":
    main()
