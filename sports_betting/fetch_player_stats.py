"""Fetch player season stats from Understat and cache to CSV.

Covers the Big 5 leagues: Premier League, La Liga, Serie A, Bundesliga, Ligue 1.
Note: Primeira Liga (Portugal) is not available on Understat.

Usage:
    python sports_betting/fetch_player_stats.py              # current season
    python sports_betting/fetch_player_stats.py --season 2526
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

UNDERSTAT_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

# 2025-26 season code for soccerdata
DEFAULT_SEASON = "2526"

OUT_PATH = Path("data/sports/processed/player_stats.csv")

RENAME_MAP = {
    "league": "league_name",
    "season": "season",
    "team": "team",
    "player": "player",
    "position": "position",
    "matches": "matches",
    "minutes": "minutes",
    "goals": "goals",
    "xg": "xg",
    "np_goals": "np_goals",
    "np_xg": "np_xg",
    "assists": "assists",
    "xa": "xa",
    "shots": "shots",
    "key_passes": "key_passes",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "xg_chain": "xg_chain",
    "xg_buildup": "xg_buildup",
}


def fetch_and_save(season: str = DEFAULT_SEASON, out_path: Path = OUT_PATH) -> tuple[int, str]:
    """Fetch player stats from Understat and write to CSV.

    Returns (returncode, message).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soccerdata as sd  # noqa: PLC0415
    except ImportError:
        return 1, "soccerdata is not installed. Run: pip install soccerdata"

    try:
        print(f"Fetching Understat player stats — season {season} — leagues: {', '.join(UNDERSTAT_LEAGUES)}")
        us = sd.Understat(leagues=UNDERSTAT_LEAGUES, seasons=season)
        df = us.read_player_season_stats()
        df = df.reset_index()

        # Keep only the columns we actually need (drop internal IDs)
        keep = [c for c in RENAME_MAP.keys() if c in df.columns]
        df = df[keep].rename(columns=RENAME_MAP)

        df.to_csv(out_path, index=False)
        msg = (
            f"✅ Saved {len(df)} player records ({df['league_name'].nunique()} leagues, "
            f"{df['team'].nunique()} clubs) to {out_path}"
        )
        print(msg)
        return 0, msg

    except Exception as exc:
        msg = f"❌ Error fetching player stats: {exc}"
        print(msg)
        return 1, msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Understat player season stats")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season code e.g. 2526 for 2025-26")
    parser.add_argument("--out", default=str(OUT_PATH), help="Output CSV path")
    args = parser.parse_args()

    code, _ = fetch_and_save(season=args.season, out_path=Path(args.out))
    raise SystemExit(code)
