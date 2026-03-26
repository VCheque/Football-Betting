# Football Bets

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FE4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat-square&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit_Cloud-live-34d399?style=flat-square" />
</p>

Football match prediction and betting analysis app for the top 6 European leagues. Powered by a calibrated XGBoost classifier with 20 engineered features — including ELO ratings and live league position — and fully automated data refresh via GitHub Actions.

**Live app:** [gestao-tickets.streamlit.app](https://gestao-tickets.streamlit.app)

---

## What It Does

| Feature | Description |
|---|---|
| Match prediction | XGBoost classifier — Home / Draw / Away probabilities |
| Bet builder | Conservative, moderate, and high-risk ticket combinations across multiple markets |
| Team context | Rolling form (last 5), standings, goals, shots-on-target, rest days |
| ELO ratings | Elo-style strength ratings updated after every match; `elo_gap` used as a model feature |
| League position | Live rank for each team in their division; `home_rank` and `away_rank` as model features |
| Player intelligence | Likely scorers, assisters, and card candidates from player xG/xA stats |
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
                                                          ├── ELO ratings (computed on load)
                                                          ├── Team snapshots + standings
                                                          └── Bet combinations
```

Data refresh is fully automated — no manual steps needed:

| Pipeline | Schedule | Source | Output |
|---|---|---|---|
| Match & odds | Every 6 hours | football-data.co.uk | `processed/*.csv` |
| Player stats | Weekly (Sunday 02:00 UTC) | Understat | `processed/player_stats.csv` |

---

## Model

### Algorithm

XGBoost multi-class classifier (`multi:softprob`, 3 classes: Home / Draw / Away), calibrated post-hoc with **isotonic regression** for well-calibrated probabilities.

### Loss function

**Multiclass log-loss** (`mlogloss`). Chosen because:
- Penalises confident wrong predictions heavily, rewarding well-calibrated probability distributions
- Works directly with soft probabilities rather than hard class labels
- Consistent with the isotonic calibration step applied after training

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 300 | More trees with lower learning rate → better generalisation |
| `max_depth` | 4 | Captures ELO/rank interactions without overfitting |
| `learning_rate` | 0.05 | Conservative shrinkage with 300 trees |
| `subsample` | 0.8 | Row sampling reduces variance |
| `colsample_bytree` | 0.8 | Feature sampling per tree |
| `min_child_weight` | 5 | Regularises the underrepresented Draw class |
| `reg_alpha` | 0.1 | L1 sparsity — prunes irrelevant features |
| `reg_lambda` | 1.0 | L2 ridge regularisation |
| `tree_method` | hist | Histogram-based — fast on Streamlit Cloud's single CPU |

Training uses **exponential sample weighting** (`exp(−age/500 days)`) so recent matches dominate. The model trains fresh at app load from the processed CSVs — no pre-trained file needed.

### Features (20)

| Group | Features |
|---|---|
| Rolling form (last 5) | `form_points_gap`, `forward_goals_gap`, `defense_gap`, `cards_gap`, `corners_gap`, `sot_gap` |
| Season context | `season_points_gap`, `rest_gap`, `fatigue_gap`, `league_idx` |
| H2H | `h2h_gap`, `h2h_goal_diff` |
| Enrichment | `injury_gap`, `lineup_strength_gap` |
| Role split | `home_role_gap` (home-only PPG vs away-only PPG) |
| Trend | `momentum_gap` (OLS slope of last-5 points, home minus away) |
| Rivalry | `derby_flag` (1.0 if same-city derby) |
| ELO strength | `elo_gap` (home ELO minus away ELO — long-term strength signal) |
| League position | `home_rank`, `away_rank` (current division standing, 1 = top) |

### ELO ratings

Computed chronologically from full match history at app load. Home-advantage offset of +100 ELO points is applied when computing expected scores (K = 20). Final ratings are stored in the app context and used both as a model feature and surfaced in the match preview.

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
