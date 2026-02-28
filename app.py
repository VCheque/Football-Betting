#!/usr/bin/env python3
"""Football betting UI with match intelligence and league/player analytics."""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
from datetime import date, timedelta
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
from sports_betting.xgboost_models import (
    CLASS_TO_RESULT,
    MATCH_FEATURE_COLS,
    player_probabilities_for_team,
    predict_match_proba,
    train_match_model,
    train_player_models,
)

DEFAULT_DATA_FILE = Path("data/sports/processed/top6_plus_portugal_matches_odds_since2022.csv")
TOP6_DATA_FILE = Path("data/sports/processed/top6_matches_odds_since2022.csv")

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
]

# API-Football league IDs (v3.football.api-sports.io)
LEAGUE_API_IDS: dict[str, int] = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
    "Eredivisie": 88,
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

        /* ── Subheader accent line ── */
        h2::after {
          content: '';
          display: block;
          margin-top: 6px;
          height: 2px;
          width: 40px;
          background: var(--accent);
          border-radius: 2px;
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


def pick_data_path(option: str) -> Path:
    return TOP6_DATA_FILE if option == "Top 6" else DEFAULT_DATA_FILE


def run_refresh(refresh_dataset: str, start_season: int, end_season: int, min_date: date) -> tuple[int, str]:
    fetch_script = Path(__file__).resolve().parent / "sports_betting" / "fetch_top6_data.py"
    cmd = [
        sys.executable,
        str(fetch_script),
        "--start-season",
        str(start_season),
        "--end-season",
        str(end_season),
        "--min-date",
        min_date.isoformat(),
    ]
    if refresh_dataset == "Top 6":
        cmd.append("--exclude-portugal")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")


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
    historical = df.loc[known & (df["match_date"] < as_of_ts)].copy()
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
    """Return a coloured W/D/L form string for the last N league matches."""
    rows = historical.loc[
        (historical["league_name"] == league_name)
        & (historical["result_ft"].isin(RESULT_VALUES))
        & (historical["match_date"] < as_of_ts)
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


def _build_combos(picks_df: pd.DataFrame, legs: int) -> pd.DataFrame:
    """Build N-leg accumulator combinations from a picks DataFrame.

    Each combination must use a different match (no duplicate match_id legs).
    Returns a DataFrame with legs text, combined_odds, hit_probability, expected_roi.
    """
    records = picks_df.sort_values("expected_roi", ascending=False).to_dict("records")
    results: list[dict] = []

    for combo in itertools.combinations(records, legs):
        match_ids = {r["match_id"] for r in combo}
        if len(match_ids) < legs:
            continue  # two legs from same match — skip

        combined_odds = float(np.prod([r["odds"] for r in combo]))
        hit_prob = float(np.prod([r["model_prob"] for r in combo]))
        ev = hit_prob * combined_odds - 1.0

        legs_text = " | ".join(f"{r['match']} → {r['pick_label']}" for r in combo)
        results.append(
            {
                "legs": legs_text,
                "combined_odds": round(combined_odds, 2),
                "hit_probability": round(hit_prob, 4),
                "expected_roi": round(ev, 4),
            }
        )

    return pd.DataFrame(results)


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


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_models(historical: pd.DataFrame, injuries_df: pd.DataFrame, contrib_df: pd.DataFrame):
    match_model = train_match_model(historical, injuries_df=injuries_df)
    player_models = train_player_models(contrib_df)
    return match_model, player_models


def main() -> None:
    st.set_page_config(page_title="Football Bets Tool", page_icon=":soccer:", layout="wide")
    apply_style()

    st.title("Football Bets Tool")
    st.caption("Match intelligence · League standings · Player probabilities")

    page1, page2, page3 = st.tabs(["Match Center", "League & Players", "Bet Builder"])

    with st.sidebar:
        st.header("Settings")
        dataset = st.selectbox("Dataset", ["Top 6 + Portugal", "Top 6"])
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
        refresh_dataset = st.selectbox("Dataset to refresh", ["Top 6 + Portugal", "Top 6"])
        refresh_start = st.number_input("Start season", min_value=1995, max_value=2100, value=date.today().year - 20)
        refresh_end = st.number_input("End season", min_value=1995, max_value=2100, value=date.today().year)
        refresh_min_date = st.date_input("Min match date", value=date(date.today().year - 20, 1, 1))
        if st.button("Refresh Data", use_container_width=True):
            with st.spinner("Refreshing..."):
                code, logs = run_refresh(refresh_dataset, int(refresh_start), int(refresh_end), refresh_min_date)
            if code == 0:
                st.success("Refresh complete")
                st.cache_data.clear()
            else:
                st.error("Refresh failed")
            st.code(logs[-3500:] if logs else "No logs")

    data_path = pick_data_path(dataset)
    with st.spinner("Loading data…"):
        context, err = _cached_context(
            str(data_path), as_of, momentum_window,
            str(injuries_file), str(contrib_file), str(other_file),
        )
    if err:
        st.warning(err)
        st.stop()

    try:
        with st.spinner("Training models…"):
            match_model, player_models = _cached_models(
                context["historical"], context["injuries_df"], context["contrib_df"]
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"XGBoost training failed. Install xgboost and retry. Details: {exc}")
        st.stop()

    with page1:
        st.subheader("Match Center")

        # ── 1. League + team selection ───────────────────────────────────────
        snapshot = context["snapshot"]
        league_names = sorted(snapshot["league_name"].dropna().unique())
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

    with page2:
        st.subheader("League & Players")
        current_snapshot = context["current_snapshot"].copy()
        leagues = sorted(current_snapshot["league_name"].dropna().unique())
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
    with page3:
        st.subheader("Bet Builder")
        st.caption(
            "Select leagues + date range → fetch upcoming games → configure "
            "markets per match → generate Conservative / Moderate / High-Risk combinations."
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
                context["historical"]["league_name"].dropna().unique()
            )
            bb_leagues = st.multiselect(
                "Leagues (select up to 3)",
                options=all_league_names,
                default=all_league_names[:2] if len(all_league_names) >= 2 else all_league_names,
                max_selections=3,
                key="bb_leagues",
                help="Hold Ctrl/Cmd to pick multiple. Max 3 leagues at once.",
            )

        # ── Row 2: combination settings ───────────────────────────────────────
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            bb_legs = st.slider("Legs per combination", 2, 5, 3, key="bb_legs")
        with sc2:
            bb_n = st.slider("Combinations per tier", 3, 20, 8, key="bb_n")
        with sc3:
            bb_min_prob = st.slider(
                "Min single-pick probability",
                0.30,
                0.75,
                0.45,
                step=0.01,
                key="bb_minp",
            )

        st.markdown("---")

        if not bb_leagues:
            st.info("👆 Select at least one league above to continue.")
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

                    if api_key.strip():
                        fetched_df, fetch_msg = fetch_upcoming_fixtures_api(
                            api_key, bb_leagues, bb_start, bb_end
                        )

                    if fetched_df.empty:
                        # Fall back to local dataset
                        all_m = context.get("all_matches", context["historical"])
                        mask = (
                            (all_m["match_date"].dt.date >= bb_start)
                            & (all_m["match_date"].dt.date <= bb_end)
                            & (all_m["league_name"].isin(bb_leagues))
                        )
                        fallback = all_m.loc[mask].copy()
                        n_fb = len(fallback)
                        if fetch_msg:
                            fetch_msg += (
                                f"  Showing {n_fb} match(es) from local dataset instead."
                            )
                        else:
                            fetch_msg = (
                                f"📂 Loaded {n_fb} match(es) from the local dataset. "
                                "Add an API-Football key in the sidebar to fetch real "
                                "upcoming fixtures."
                            )
                        fetched_df = fallback

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

                    # Build editable pick table
                    pick_rows: list[dict] = []
                    for _, row in fixtures_df.iterrows():
                        rf = row.get("result_ft", None)
                        is_upcoming = (rf not in RESULT_VALUES) if pd.notna(rf) else True
                        status_lbl = (
                            "Upcoming"
                            if is_upcoming
                            else f"Played ({rf})"
                        )
                        pick_rows.append(
                            {
                                "include": True,
                                "date": str(row["match_date"].date()),
                                "league": str(row.get("league_name", "")),
                                "home": str(row.get("home_team", "")),
                                "away": str(row.get("away_team", "")),
                                "status": status_lbl,
                                "market": "1X2",
                                "pick": "Auto best",
                            }
                        )

                    pick_df = pd.DataFrame(pick_rows)
                    tbl_key = f"bb_tbl_{fetch_key}"

                    st.caption(
                        f"**{len(pick_df)} match(es) loaded** — "
                        "configure market + pick direction per row, then generate:"
                    )
                    edited_picks = st.data_editor(
                        pick_df,
                        column_config={
                            "include": st.column_config.CheckboxColumn(
                                "✓", default=True, width="small"
                            ),
                            "date": st.column_config.TextColumn(
                                "Date", disabled=True, width="small"
                            ),
                            "league": st.column_config.TextColumn(
                                "League", disabled=True
                            ),
                            "home": st.column_config.TextColumn(
                                "Home", disabled=True
                            ),
                            "away": st.column_config.TextColumn(
                                "Away", disabled=True
                            ),
                            "status": st.column_config.TextColumn(
                                "Status", disabled=True, width="small"
                            ),
                            "market": st.column_config.SelectboxColumn(
                                "Market",
                                options=MARKET_OPTIONS,
                                required=True,
                            ),
                            "pick": st.column_config.SelectboxColumn(
                                "Pick",
                                options=[
                                    "Auto best",
                                    "Home / Over",
                                    "Draw",
                                    "Away / Under",
                                ],
                                required=True,
                            ),
                        },
                        hide_index=True,
                        use_container_width=True,
                        num_rows="fixed",
                        key=tbl_key,
                    )

                    st.markdown("---")
                    if st.button(
                        "⚡ Generate Combinations",
                        use_container_width=True,
                        key="bb_gen",
                    ):
                        included = edited_picks.loc[
                            edited_picks["include"]
                        ].reset_index(drop=True)

                        if len(included) < bb_legs:
                            st.warning(
                                f"Please include at least **{bb_legs}** matches "
                                f"to build {bb_legs}-leg combinations."
                            )
                        else:
                            pick_records: list[dict] = []
                            margin = 0.05

                            with st.spinner("Computing model probabilities…"):
                                for _, irow in included.iterrows():
                                    home_t = str(irow["home"])
                                    away_t = str(irow["away"])
                                    league_n = str(irow["league"])
                                    market = str(irow["market"])
                                    pick_sel = str(irow["pick"])
                                    mid = (
                                        f"{irow['date']}|{league_n}|{home_t}|{away_t}"
                                    )

                                    # ── 1X2 via XGBoost ──────────────────────
                                    if market == "1X2":
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
                                            p1x2 = predict_match_proba(
                                                match_model, feats
                                            )
                                        except Exception:
                                            p1x2 = {"H": 0.40, "D": 0.25, "A": 0.35}

                                        if pick_sel == "Auto best":
                                            ev_map = {
                                                k: p1x2[k]
                                                * max(
                                                    1.01,
                                                    (1 / max(p1x2[k], 0.01))
                                                    * (1 - margin),
                                                )
                                                - 1.0
                                                for k in p1x2
                                            }
                                            best_k = max(ev_map, key=ev_map.get)
                                            candidates = [(best_k, p1x2[best_k])]
                                        elif pick_sel == "Home / Over":
                                            candidates = [("H", p1x2["H"])]
                                        elif pick_sel == "Draw":
                                            candidates = [("D", p1x2["D"])]
                                        else:
                                            candidates = [("A", p1x2["A"])]

                                        lbl_map = {
                                            "H": "Home (1)",
                                            "D": "Draw (X)",
                                            "A": "Away (2)",
                                        }
                                        for outcome_k, mp in candidates:
                                            odds_v = round(
                                                max(
                                                    1.01,
                                                    (1 / max(mp, 0.01)) * (1 - margin),
                                                ),
                                                2,
                                            )
                                            pick_records.append(
                                                {
                                                    "match_id": mid,
                                                    "match": f"{home_t} vs {away_t}",
                                                    "league": league_n,
                                                    "market": market,
                                                    "pick_label": lbl_map[outcome_k],
                                                    "model_prob": float(mp),
                                                    "odds": odds_v,
                                                    "edge": float(mp) - 1.0 / odds_v,
                                                    "expected_roi": float(mp) * odds_v
                                                    - 1.0,
                                                }
                                            )

                                    # ── Empirical O/U + BTTS ─────────────────
                                    else:
                                        over_p, under_p = estimate_market_proba(
                                            context["historical"],
                                            home_t,
                                            away_t,
                                            market,
                                            league_n,
                                            context["as_of_ts"],
                                        )
                                        if pick_sel in ("Home / Over", "Auto best"):
                                            mp = over_p
                                            lbl = f"{market} — Over"
                                        elif pick_sel == "Away / Under":
                                            mp = under_p
                                            lbl = f"{market} — Under"
                                        else:
                                            mp = over_p
                                            lbl = f"{market} — Over"

                                        odds_v = round(
                                            max(
                                                1.01,
                                                (1 / max(mp, 0.01)) * (1 - margin),
                                            ),
                                            2,
                                        )
                                        pick_records.append(
                                            {
                                                "match_id": mid,
                                                "match": f"{home_t} vs {away_t}",
                                                "league": league_n,
                                                "market": market,
                                                "pick_label": lbl,
                                                "model_prob": float(mp),
                                                "odds": odds_v,
                                                "edge": float(mp) - 1.0 / odds_v,
                                                "expected_roi": float(mp) * odds_v
                                                - 1.0,
                                            }
                                        )

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
                                    st.markdown(
                                        f"**{len(all_picks)} pick(s)** qualify "
                                        f"(prob ≥ {bb_min_prob:.0%}). "
                                        f"Building {bb_legs}-leg combinations…"
                                    )
                                    with st.expander("Qualifying picks"):
                                        st.dataframe(
                                            all_picks[
                                                [
                                                    "match",
                                                    "market",
                                                    "pick_label",
                                                    "model_prob",
                                                    "odds",
                                                    "expected_roi",
                                                ]
                                            ],
                                            use_container_width=True,
                                            hide_index=True,
                                        )

                                    combos_all = _build_combos(all_picks, bb_legs)

                                    if combos_all.empty:
                                        st.warning(
                                            "No valid combinations found — all picks "
                                            "may be from the same match."
                                        )
                                    else:
                                        conservative = combos_all.sort_values(
                                            "hit_probability", ascending=False
                                        ).head(bb_n)
                                        moderate = combos_all.sort_values(
                                            "expected_roi", ascending=False
                                        ).head(bb_n)
                                        risk = combos_all.sort_values(
                                            "combined_odds", ascending=False
                                        ).head(bb_n)

                                        tier_tabs = st.tabs(
                                            [
                                                "🟢 Conservative",
                                                "🟡 Moderate",
                                                "🔴 High Risk",
                                            ]
                                        )
                                        cols_show = [
                                            "legs",
                                            "combined_odds",
                                            "hit_probability",
                                            "expected_roi",
                                        ]
                                        with tier_tabs[0]:
                                            st.caption(
                                                "Sorted by highest hit-probability "
                                                "(all legs most likely to win)"
                                            )
                                            st.dataframe(
                                                conservative[cols_show],
                                                use_container_width=True,
                                                hide_index=True,
                                            )
                                        with tier_tabs[1]:
                                            st.caption(
                                                "Sorted by best expected ROI "
                                                "(balanced value + probability)"
                                            )
                                            st.dataframe(
                                                moderate[cols_show],
                                                use_container_width=True,
                                                hide_index=True,
                                            )
                                        with tier_tabs[2]:
                                            st.caption(
                                                "Sorted by highest combined odds "
                                                "(maximum payout, higher variance)"
                                            )
                                            st.dataframe(
                                                risk[cols_show],
                                                use_container_width=True,
                                                hide_index=True,
                                            )

    st.caption("Decision support only. Betting carries financial risk.")


if __name__ == "__main__":
    main()
