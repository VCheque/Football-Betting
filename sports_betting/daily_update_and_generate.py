#!/usr/bin/env python3
"""Daily pipeline: refresh football data and generate betting combinations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run daily refresh and generate betting picks/combinations."
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=date.today().year - 20,
        help="First season start year for fetch step (default: current year - 20).",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help="Last season start year for fetch step (default uses current season).",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default=f"{date.today().year - 20}-01-01",
        help="Keep matches from this date onward during normalization.",
    )
    parser.add_argument(
        "--exclude-portugal",
        action="store_true",
        help="Use only top-6 leagues for data/picks.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=date.today().isoformat(),
        help="Training cut-off date for pick generation.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=date.today().isoformat(),
        help="First date for suggested bets.",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=3,
        help="Number of days after start-date for fixtures.",
    )
    parser.add_argument(
        "--include-completed-fixtures",
        action="store_true",
        help="Use completed fixtures in the generation range (backtest/demo mode).",
    )
    parser.add_argument(
        "--combo-sizes",
        type=str,
        default="2,3",
        help="Comma-separated combo sizes (example: 2,3,4).",
    )
    parser.add_argument(
        "--num-combos",
        type=int,
        default=20,
        help="Number of combos returned per combo size.",
    )
    parser.add_argument(
        "--min-pick-prob",
        type=float,
        default=0.28,
        help="Minimum probability filter for single picks.",
    )
    parser.add_argument(
        "--min-pick-ev",
        type=float,
        default=0.03,
        help="Minimum expected ROI filter for single picks.",
    )
    parser.add_argument(
        "--min-combo-ev",
        type=float,
        default=0.05,
        help="Minimum expected ROI filter for combinations.",
    )
    parser.add_argument(
        "--max-picks-pool",
        type=int,
        default=24,
        help="Top N picks used while building combinations.",
    )
    parser.add_argument(
        "--odds-source",
        choices=("auto", "avg", "max", "b365", "pinnacle"),
        default="auto",
        help="Odds source used by the generation step.",
    )
    parser.add_argument(
        "--momentum-window",
        type=int,
        default=5,
        help="Number of recent matches used in momentum features.",
    )
    parser.add_argument(
        "--h2h-years",
        type=int,
        default=20,
        help="Head-to-head lookback window in years.",
    )
    parser.add_argument(
        "--injuries-file",
        type=Path,
        default=Path("data/sports/external/injuries.csv"),
        help="Optional injuries CSV path.",
    )
    parser.add_argument(
        "--player-contrib-file",
        type=Path,
        default=Path("data/sports/external/player_contributions.csv"),
        help="Optional player contributions CSV path.",
    )
    parser.add_argument(
        "--other-competitions-file",
        type=Path,
        default=Path("data/sports/external/other_competitions_matches.csv"),
        help="Optional other competitions CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sports/outputs"),
        help="Output directory for generated picks/combos.",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("\nRunning:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    fetch_script = script_dir / "fetch_top6_data.py"
    generate_script = script_dir / "generate_bet_combinations.py"

    fetch_cmd = [
        sys.executable,
        str(fetch_script),
        "--start-season",
        str(args.start_season),
    ]
    if args.end_season is not None:
        fetch_cmd.extend(["--end-season", str(args.end_season)])
    fetch_cmd.extend(["--min-date", args.min_date])
    if args.exclude_portugal:
        fetch_cmd.append("--exclude-portugal")

    run_command(fetch_cmd)

    data_file = (
        Path("data/sports/processed/top6_matches_odds_since2022.csv")
        if args.exclude_portugal
        else Path("data/sports/processed/top6_plus_portugal_matches_odds_since2022.csv")
    )

    generate_cmd = [
        sys.executable,
        str(generate_script),
        "--data-file",
        str(data_file),
        "--as-of-date",
        args.as_of_date,
        "--start-date",
        args.start_date,
        "--days-ahead",
        str(args.days_ahead),
        "--combo-sizes",
        args.combo_sizes,
        "--num-combos",
        str(args.num_combos),
        "--min-pick-prob",
        str(args.min_pick_prob),
        "--min-pick-ev",
        str(args.min_pick_ev),
        "--min-combo-ev",
        str(args.min_combo_ev),
        "--max-picks-pool",
        str(args.max_picks_pool),
        "--odds-source",
        args.odds_source,
        "--momentum-window",
        str(args.momentum_window),
        "--h2h-years",
        str(args.h2h_years),
        "--injuries-file",
        str(args.injuries_file),
        "--player-contrib-file",
        str(args.player_contrib_file),
        "--other-competitions-file",
        str(args.other_competitions_file),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.include_completed_fixtures:
        generate_cmd.append("--include-completed-fixtures")

    run_command(generate_cmd)
    print("\nDaily pipeline completed.")


if __name__ == "__main__":
    main()
