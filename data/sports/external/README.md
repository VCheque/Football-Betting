# External Signals

These files are optional but improve prediction quality:

- `injuries.csv`
  - Required columns: `date`, `team`
  - Optional: `player`, `status`, `importance_score`, `expected_return`
- `player_contributions.csv`
  - Required columns: `match_date`, `team`
  - Optional: `player`, `minutes`, `goals`, `assists`, `key_passes`, `xg`, `xa`, `rating`, `shots_on_target`, `yellow_cards`, `red_cards`, `fouls`
- `other_competitions_matches.csv`
  - Required columns: `match_date`, `team`
  - Optional: `competition`

Team names should match your main dataset team names exactly.
