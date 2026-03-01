"""Fetch player season stats and cache to CSV.

Primary source  : API-Football v3 (when --api-key is supplied)
                  Covers all 6 leagues incl. Primeira Liga.
Fallback source : Understat (free, no key needed)
                  Covers Big 5 only — Primeira Liga NOT available.

The two sources are harmonised to the same output schema so the rest of
the app never needs to know which was used.

Usage:
    python sports_betting/fetch_player_stats.py                      # Understat only
    python sports_betting/fetch_player_stats.py --api-key <KEY>      # API-Football → Understat fallback
    python sports_betting/fetch_player_stats.py --season 2526        # specific season
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent.parent  # repo root
METADATA_FILE = _HERE / "data/sports/processed/refresh_metadata.json"


def _update_metadata(key: str, records: int, source: str, meta_path: Path = METADATA_FILE) -> None:
    """Upsert one section in the shared refresh_metadata.json."""
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    meta[f"{key}_last_fetch"] = datetime.now().isoformat(timespec="seconds")
    meta[f"{key}_records"] = records
    meta[f"{key}_source"] = source
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))

# ── League mappings ──────────────────────────────────────────────────────────
UNDERSTAT_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

# API-Football league IDs — includes Portugal which Understat doesn't have
API_FOOTBALL_LEAGUES: dict[str, int] = {
    "Premier League":  39,
    "La Liga":        140,
    "Serie A":        135,
    "Bundesliga":      78,
    "Ligue 1":         61,
    "Primeira Liga":   94,
}

DEFAULT_SEASON = "2526"          # 2025-26
OUT_PATH = _HERE / "data/sports/processed/player_stats.csv"

# Endpoints that return top-20 players per category — cheap (1 req each)
_TOP_ENDPOINTS = ["topscorers", "topassists", "topyellowcards", "topredcards"]


# ── API-Football source ──────────────────────────────────────────────────────

def _api_season_year(season_code: str) -> int:
    """Convert '2526' → 2025 (API-Football uses the starting calendar year)."""
    return int(season_code[:2]) + 2000


def fetch_from_api_football(
    api_key: str,
    season_code: str = DEFAULT_SEASON,
    out_path: Path = OUT_PATH,
) -> tuple[int, str]:
    """Fetch top-player stats from API-Football and write to CSV.

    Uses /players/topscorers, /topassists, /topyellowcards, /topredcards
    per league — 4 × 6 = 24 requests total (well within the 100 req/day free plan).

    Returns (returncode, message).
    """
    try:
        import requests as req  # noqa: PLC0415
    except ImportError:
        return 1, "requests library not installed."

    if not api_key.strip():
        return 1, "Empty API key."

    api_season = _api_season_year(season_code)
    base = "https://v3.football.api-sports.io"
    headers = {"x-apisports-key": api_key.strip()}

    # player_key → merged row dict  (key = f"{league_name}|{player_id}")
    merged: dict[str, dict] = {}
    errors: list[str] = []

    for league_name, league_id in API_FOOTBALL_LEAGUES.items():
        for endpoint in _TOP_ENDPOINTS:
            time.sleep(0.25)   # gentle pacing — free plan has no burst limit stated
            try:
                resp = req.get(
                    f"{base}/players/{endpoint}",
                    params={"league": league_id, "season": api_season},
                    headers=headers,
                    timeout=20,
                )
                if not resp.ok:
                    errors.append(f"{league_name}/{endpoint}: HTTP {resp.status_code}")
                    continue

                data = resp.json()
                api_errors = data.get("errors")
                if api_errors:
                    errors.append(f"{league_name}/{endpoint}: {api_errors}")
                    continue

                for item in data.get("response", []):
                    p_info = item.get("player", {})
                    pid = p_info.get("id")
                    if pid is None:
                        continue

                    stats = (item.get("statistics") or [{}])[0]
                    games   = stats.get("games", {})
                    goals   = stats.get("goals", {})
                    shots   = stats.get("shots", {})
                    passes  = stats.get("passes", {})
                    cards   = stats.get("cards", {})

                    key = f"{league_name}|{pid}"
                    if key not in merged:
                        merged[key] = {
                            "league_name":   league_name,
                            "season":        season_code,
                            "team":          (stats.get("team") or {}).get("name", ""),
                            "player":        p_info.get("name", ""),
                            "position":      games.get("position", ""),
                            "matches":       games.get("appearences") or 0,
                            "minutes":       games.get("minutes") or 0,
                            "goals":         goals.get("total") or 0,
                            "xg":            None,    # not available on free tier
                            "np_goals":      None,
                            "np_xg":         None,
                            "assists":       goals.get("assists") or 0,
                            "xa":            None,    # not available on free tier
                            "shots":         shots.get("total") or 0,
                            "shots_on":      shots.get("on") or 0,
                            "key_passes":    passes.get("key") or 0,
                            "yellow_cards":  cards.get("yellow") or 0,
                            "red_cards":     cards.get("red") or 0,
                            "xg_chain":      None,
                            "xg_buildup":    None,
                        }
                    else:
                        # Merge: keep the highest value we've seen per stat
                        row = merged[key]
                        row["goals"]        = max(row["goals"],        goals.get("total")    or 0)
                        row["assists"]      = max(row["assists"],       goals.get("assists")  or 0)
                        row["shots"]        = max(row["shots"],         shots.get("total")    or 0)
                        row["shots_on"]     = max(row["shots_on"],      shots.get("on")       or 0)
                        row["key_passes"]   = max(row["key_passes"],    passes.get("key")     or 0)
                        row["yellow_cards"] = max(row["yellow_cards"],  cards.get("yellow")   or 0)
                        row["red_cards"]    = max(row["red_cards"],     cards.get("red")      or 0)
                        # Use most complete game count
                        apps = games.get("appearences") or 0
                        if apps > row["matches"]:
                            row["matches"] = apps
                            row["minutes"] = games.get("minutes") or row["minutes"]

            except Exception as exc:
                errors.append(f"{league_name}/{endpoint}: {exc}")

    if not merged:
        err_str = "; ".join(errors[:6])
        return 1, f"API-Football returned no player data. Errors: {err_str}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(merged.values()))
    df.to_csv(out_path, index=False)
    _update_metadata("players", len(df), "API-Football")

    msg = (
        f"[API-Football] Saved {len(df)} player records "
        f"({df['league_name'].nunique()} leagues, {df['team'].nunique()} clubs) → {out_path}"
    )
    if errors:
        msg += f"\n  Warnings ({len(errors)}): {'; '.join(errors[:4])}"
    return 0, msg


# ── Understat source ─────────────────────────────────────────────────────────

def fetch_from_understat(
    season_code: str = DEFAULT_SEASON,
    out_path: Path = OUT_PATH,
) -> tuple[int, str]:
    """Fetch player stats from Understat (Big 5 only, no Portugal).

    Returns (returncode, message).
    """
    try:
        import soccerdata as sd  # noqa: PLC0415
    except ImportError:
        return 1, "soccerdata not installed. Run: pip install soccerdata"

    try:
        print(
            f"Fetching Understat player stats — season {season_code} — "
            f"leagues: {', '.join(UNDERSTAT_LEAGUES)}"
        )
        us = sd.Understat(leagues=UNDERSTAT_LEAGUES, seasons=season_code)
        df = us.read_player_season_stats().reset_index()

        # Normalise column names to match API-Football output schema
        rename = {
            "league":       "league_name",
            "season":       "season",
            "team":         "team",
            "player":       "player",
            "position":     "position",
            "matches":      "matches",
            "minutes":      "minutes",
            "goals":        "goals",
            "xg":           "xg",
            "np_goals":     "np_goals",
            "np_xg":        "np_xg",
            "assists":      "assists",
            "xa":           "xa",
            "shots":        "shots",
            "key_passes":   "key_passes",
            "yellow_cards": "yellow_cards",
            "red_cards":    "red_cards",
            "xg_chain":     "xg_chain",
            "xg_buildup":   "xg_buildup",
        }
        keep = [c for c in rename if c in df.columns]
        df = df[keep].rename(columns=rename)
        df["shots_on"] = None   # Understat doesn't expose shots on target separately

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        _update_metadata("players", len(df), "Understat")

        msg = (
            f"[Understat] Saved {len(df)} player records "
            f"({df['league_name'].nunique()} leagues, {df['team'].nunique()} clubs) → {out_path}"
        )
        return 0, msg

    except Exception as exc:
        return 1, f"Understat fetch error: {exc}"


# ── Public entry point ───────────────────────────────────────────────────────

def fetch_and_save(
    api_key: str = "",
    season: str = DEFAULT_SEASON,
    out_path: Path = OUT_PATH,
) -> tuple[int, str]:
    """Try API-Football first; fall back to Understat on failure or missing key."""
    if api_key.strip():
        print("API key provided — trying API-Football first…")
        code, msg = fetch_from_api_football(api_key, season, out_path)
        if code == 0:
            print(msg)
            return code, msg
        print(f"API-Football failed: {msg}\nFalling back to Understat…")

    code, msg = fetch_from_understat(season, out_path)
    print(msg)
    return code, msg


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch player season stats")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season code e.g. 2526")
    parser.add_argument("--api-key", default="", help="API-Football key (optional)")
    parser.add_argument("--out", default=str(OUT_PATH), help="Output CSV path")
    args = parser.parse_args()

    code, _ = fetch_and_save(
        api_key=args.api_key,
        season=args.season,
        out_path=Path(args.out),
    )
    raise SystemExit(code)
