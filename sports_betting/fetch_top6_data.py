#!/usr/bin/env python3
"""Download football (soccer) match + odds data for top European leagues."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
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
try:
    from sports_betting.team_names import TEAM_NAME_MAP
except ModuleNotFoundError:
    from team_names import TEAM_NAME_MAP

BASE_URL = "https://www.football-data.co.uk/mmz4281"


@dataclass(frozen=True)
class League:
    code: str
    name: str


DEFAULT_TOP6 = (
    League("E0", "Premier League"),
    League("SP1", "La Liga"),
    League("I1", "Serie A"),
    League("D1", "Bundesliga"),
    League("F1", "Ligue 1"),
)
PORTUGAL = League("P1", "Primeira Liga")


def infer_latest_season_start(today: date) -> int:
    # European seasons mostly start between July and August.
    return today.year if today.month >= 7 else today.year - 1


def infer_default_start_season(today: date) -> int:
    return today.year - 20


def season_code(season_start: int) -> str:
    yy1 = str(season_start % 100).zfill(2)
    yy2 = str((season_start + 1) % 100).zfill(2)
    return f"{yy1}{yy2}"


def load_season(league: League, season_start: int) -> pd.DataFrame:
    code = season_code(season_start)
    url = f"{BASE_URL}/{code}/{league.code}.csv"
    df = pd.read_csv(url)
    df["league_code"] = league.code
    df["league_name"] = league.name
    df["season_start"] = season_start
    df["season_label"] = f"{season_start}/{season_start + 1}"
    df["source_url"] = url
    return df


def normalize_clean(df: pd.DataFrame, min_date: pd.Timestamp) -> pd.DataFrame:
    rename_map = {
        "Date": "match_date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals_ft",
        "FTAG": "away_goals_ft",
        "HTHG": "home_goals_ht",
        "HTAG": "away_goals_ht",
        "FTR": "result_ft",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_shots_on_target",
        "AST": "away_shots_on_target",
        "HF": "home_fouls",
        "AF": "away_fouls",
        "HC": "home_corners",
        "AC": "away_corners",
        "HY": "home_yellow_cards",
        "AY": "away_yellow_cards",
        "HR": "home_red_cards",
        "AR": "away_red_cards",
        "B365H": "odds_b365_home",
        "B365D": "odds_b365_draw",
        "B365A": "odds_b365_away",
        "PSH": "odds_pinnacle_home",
        "PSD": "odds_pinnacle_draw",
        "PSA": "odds_pinnacle_away",
        "MaxH": "odds_max_home",
        "MaxD": "odds_max_draw",
        "MaxA": "odds_max_away",
        "AvgH": "odds_avg_home",
        "AvgD": "odds_avg_draw",
        "AvgA": "odds_avg_away",
    }

    clean = df.rename(columns=rename_map).copy()
    clean["home_team"] = clean["home_team"].replace(TEAM_NAME_MAP)
    clean["away_team"] = clean["away_team"].replace(TEAM_NAME_MAP)
    clean["match_date"] = pd.to_datetime(
        clean["match_date"], dayfirst=True, errors="coerce"
    )
    clean = clean.loc[clean["match_date"].notna()].copy()
    clean = clean.loc[clean["match_date"] >= min_date].copy()

    desired_cols = [
        "match_date",
        "league_code",
        "league_name",
        "season_start",
        "season_label",
        "home_team",
        "away_team",
        "home_goals_ft",
        "away_goals_ft",
        "home_goals_ht",
        "away_goals_ht",
        "result_ft",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_fouls",
        "away_fouls",
        "home_corners",
        "away_corners",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_red_cards",
        "away_red_cards",
        "odds_b365_home",
        "odds_b365_draw",
        "odds_b365_away",
        "odds_pinnacle_home",
        "odds_pinnacle_draw",
        "odds_pinnacle_away",
        "odds_max_home",
        "odds_max_draw",
        "odds_max_away",
        "odds_avg_home",
        "odds_avg_draw",
        "odds_avg_away",
        "source_url",
    ]

    for col in desired_cols:
        if col not in clean.columns:
            clean[col] = pd.NA

    clean = clean[desired_cols].sort_values(
        by=["match_date", "league_code", "home_team", "away_team"]
    )
    clean.reset_index(drop=True, inplace=True)
    return clean


def build_dataset(
    leagues: tuple[League, ...], start_season: int, end_season: int
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for league in leagues:
        for season_start in range(start_season, end_season + 1):
            try:
                frames.append(load_season(league, season_start))
                print(f"Loaded {league.code} {season_start}/{season_start + 1}")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Skipped {league.code} {season_start}/{season_start + 1}: {exc}"
                )
    if not frames:
        raise RuntimeError("No data files could be loaded.")
    return pd.concat(frames, ignore_index=True)


def save_outputs(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str,
) -> None:
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{filename_prefix}_matches_odds_since2022_raw.csv"
    clean_path = processed_dir / f"{filename_prefix}_matches_odds_since2022.csv"
    summary_path = processed_dir / f"{filename_prefix}_matches_odds_since2022_summary.csv"

    df_raw.to_csv(raw_path, index=False)
    df_clean.to_csv(clean_path, index=False)

    summary = (
        df_clean.groupby(["league_name", "season_label"], dropna=False)
        .size()
        .reset_index(name="matches")
        .sort_values(["league_name", "season_label"])
    )
    summary.to_csv(summary_path, index=False)

    print("\nSaved files:")
    print(f"- {raw_path}")
    print(f"- {clean_path}")
    print(f"- {summary_path}")
    print(f"\nTotal cleaned matches: {len(df_clean):,}")
    print("\nMatches by league and season:")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch top European football leagues data (supports 20-year history)."
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=infer_default_start_season(date.today()),
        help="First season start year to fetch (default: current year - 20).",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=infer_latest_season_start(date.today()),
        help="Last season start year to fetch (default: inferred from current date).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sports"),
        help="Directory where output files are stored.",
    )
    parser.add_argument(
        "--exclude-portugal",
        action="store_true",
        help="Use only top-6 leagues (exclude Primeira Liga).",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default=f"{date.today().year - 20}-01-01",
        help="Keep matches from this date onward (YYYY-MM-DD).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_season < args.start_season:
        raise ValueError("end-season must be >= start-season")

    leagues = DEFAULT_TOP6 if args.exclude_portugal else DEFAULT_TOP6 + (PORTUGAL,)
    filename_prefix = "top6" if args.exclude_portugal else "top6_plus_portugal"

    print("Selected leagues:")
    for league in leagues:
        print(f"- {league.name} ({league.code})")

    raw_df = build_dataset(
        leagues=leagues,
        start_season=args.start_season,
        end_season=args.end_season,
    )
    min_date = pd.to_datetime(args.min_date, errors="coerce")
    if pd.isna(min_date):
        raise ValueError("Invalid --min-date. Use YYYY-MM-DD format.")
    clean_df = normalize_clean(raw_df, pd.Timestamp(min_date.date()))
    save_outputs(raw_df, clean_df, args.output_dir, filename_prefix)
    _update_metadata("matches", len(clean_df), "football-data.co.uk")
    print(f"Metadata written to {METADATA_FILE}")


if __name__ == "__main__":
    main()
