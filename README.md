# Football Bets Tool

This project builds football betting intelligence for top European leagues with:
- match-level predictions using XGBoost
- standings and team context
- player likelihoods (score/assist/cards)
- H2H history (recency weighted, slicer up to 20 years)

## Main UI

Run:

```bash
streamlit run app.py
```

### Page 1 - Match Center

Flow:
1. Select league.
2. Select match (home vs away).
3. Fetch probable starting XI online (API-Football key, optional) and/or edit manually.
4. Run ML prediction (XGBoost).
5. See conservative/moderate/high-risk betting options with tooltips explaining why.
6. View important injured players, likely scorers, and likely yellow/red card players.
7. Use H2H year slicer and inspect past H2H matches.

### Page 2 - League & Players

- League standings table for selected league.
- Team player panel with XGBoost likelihoods:
  - `prob_score`
  - `prob_assist`
  - `prob_card`
- Important injured players for selected team.

## Global Sidebar

Sidebar keeps only global actions:
- refresh data (start season, end season, min date)

## Install

```bash
python -m pip install -r requirements.txt
```

If `xgboost` is missing, the app will show an explicit error and stop until installed.

## Data Scripts

- `sports_betting/fetch_top6_data.py`
  - Downloads and normalizes league data (shots, corners, fouls, cards, odds).
  - Supports long history via `--start-season` and `--min-date`.
- `sports_betting/generate_bet_combinations.py`
  - CLI picks/combos generator (still available).
- `sports_betting/daily_update_and_generate.py`
  - Daily refresh + generator pipeline.
- `sports_betting/xgboost_models.py`
  - XGBoost training/inference helpers for match and player models.

## Optional External Signals

Default folder: `data/sports/external/`

- `injuries.csv`
  - Required: `date`, `team`
  - Optional: `player`, `status`, `importance_score`, `expected_return`
- `player_contributions.csv`
  - Required: `match_date`, `team`
  - Optional: `player`, `minutes`, `goals`, `assists`, `key_passes`, `xg`, `xa`, `rating`, `shots_on_target`, `yellow_cards`, `red_cards`, `fouls`
- `other_competitions_matches.csv`
  - Required: `match_date`, `team`
  - Optional: `competition`

## Notes

- This is decision support, not guaranteed profit.
- Betting carries financial risk.
