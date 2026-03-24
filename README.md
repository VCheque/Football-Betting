# Football Bets

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FE4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat-square&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit_Cloud-live-34d399?style=flat-square" />
</p>

Football match prediction and betting analysis app for the top 6 European leagues, powered by a calibrated XGBoost model with 17 engineered features and automated data refresh via GitHub Actions.

**Live app:** [gestao-tickets.streamlit.app](https://gestao-tickets.streamlit.app)

---

## What It Does

| Feature | Description |
|---|---|
| Match prediction | XGBoost classifier — Home / Draw / Away probabilities |
| Bet builder | Generates conservative, moderate, and high-risk ticket combinations across multiple markets |
| Team context | Rolling form (last 5), standings, goals, shots-on-target, rest days |
| Player intelligence | Likely scorers, assisters, and card candidates from player stats |
| H2H analysis | Recency-weighted head-to-head history with up to 20-year slicer |
| Injury awareness | Active injuries surfaced in match preview and player panels |
| Derby detection | 33 same-city rivalries flagged as a model feature across 6 leagues |
| Language toggle | English / Portuguese (Mozambique) |

---

## Architecture

```
GitHub Actions (4×/day + weekly)
        │
        ▼
football-data.co.uk ──► data/sports/processed/*.csv ──► Streamlit app
Understat                (committed to repo)              │
                                                          ├── XGBoost model (train on load)
                                                          ├── Team snapshots
                                                          └── Bet combinations
```

Data refresh is fully automated — no manual steps needed:

| Pipeline | Schedule | Source | Output |
|---|---|---|---|
| Match & odds | Every 6 hours | football-data.co.uk | `processed/*.csv` |
| Player stats | Weekly (Sunday 02:00 UTC) | Understat | `processed/player_stats.csv` |

---

## Model

The XGBoost match outcome classifier uses **17 engineered features**, calibrated with isotonic regression:

| Feature group | Features |
|---|---|
| Rolling form | `goals_for_pg`, `goals_against_pg`, `last_points_pg`, `corners_for_pg`, `corners_against_pg`, `cards_pg` |
| Head-to-head | `h2h_home_win_rate`, `h2h_away_win_rate`, `h2h_draw_rate`, `h2h_home_goals_pg`, `h2h_away_goals_pg` |
| Context | `rest_days`, `league_idx` |
| v2 features | `home_role_gap`, `momentum_gap` (OLS slope), `derby_flag`, `sot_gap` |

The model trains at app load time from the processed CSVs — no pre-trained file needed.

---

## Data Sources

| Source | Data | Leagues |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/) | Match results, odds, corners, cards | Premier League · La Liga · Serie A · Bundesliga · Ligue 1 · Primeira Liga |
| [Understat](https://understat.com/) | Player xG, xA, goals, assists, minutes | Premier League · La Liga · Serie A · Bundesliga · Ligue 1 |

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Optional external signals

Place CSV files in `data/sports/external/` to enrich predictions:

| File | Required columns | Purpose |
|---|---|---|
| `injuries.csv` | `date`, `team` | Active injuries in match preview |
| `player_contributions.csv` | `match_date`, `team` | Player scoring/assist likelihoods |
| `other_competitions_matches.csv` | `match_date`, `team` | Fatigue from non-league games |

---

## Leagues Covered

Premier League · La Liga · Serie A · Bundesliga · Ligue 1 · Primeira Liga

---

> Decision support tool. Betting carries financial risk.
