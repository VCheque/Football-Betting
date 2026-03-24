#!/usr/bin/env python3
"""XGBoost models for match outcomes and player events."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

RESULT_TO_CLASS = {"H": 0, "D": 1, "A": 2}
CLASS_TO_RESULT = {0: "H", 1: "D", 2: "A"}

# ── Derby / same-city rivalry lookup ─────────────────────────────────────────
# Team names as they appear in football-data.co.uk exports.
# frozenset allows bidirectional matching regardless of home/away assignment.
_DERBY_PAIRS: frozenset[frozenset[str]] = frozenset({
    # England – Premier League (E0)
    frozenset({"Man City", "Man United"}),
    frozenset({"Arsenal", "Tottenham"}),
    frozenset({"Chelsea", "Tottenham"}),
    frozenset({"Chelsea", "Arsenal"}),
    frozenset({"Chelsea", "Fulham"}),
    frozenset({"Liverpool", "Everton"}),
    frozenset({"Leeds", "Man United"}),
    frozenset({"Newcastle", "Sunderland"}),
    frozenset({"West Ham", "Millwall"}),
    frozenset({"West Ham", "Tottenham"}),
    # Spain – La Liga (SP1)
    frozenset({"Real Madrid", "Atletico Madrid"}),
    frozenset({"Barcelona", "Espanyol"}),
    frozenset({"Sevilla", "Betis"}),
    frozenset({"Athletic Club", "Sociedad"}),
    frozenset({"Valencia", "Villarreal"}),
    frozenset({"Real Madrid", "Getafe"}),
    # Italy – Serie A (I1)
    frozenset({"Juventus", "Torino"}),
    frozenset({"Inter", "AC Milan"}),
    frozenset({"Roma", "Lazio"}),
    # Germany – Bundesliga (D1)
    frozenset({"Dortmund", "Schalke 04"}),
    frozenset({"Hamburg", "St Pauli"}),
    frozenset({"Cologne", "Leverkusen"}),
    frozenset({"Dortmund", "Cologne"}),
    # France – Ligue 1 (F1)
    frozenset({"Paris SG", "Lens"}),
    frozenset({"Marseille", "Nice"}),
    frozenset({"Marseille", "Lyon"}),
    frozenset({"Lille", "Lens"}),
    # Portugal – Primeira Liga (P1)
    frozenset({"Benfica", "Sporting CP"}),
    frozenset({"Benfica", "Porto"}),
    frozenset({"Sporting CP", "Porto"}),
    frozenset({"Porto", "Boavista"}),
    frozenset({"Benfica", "Belenenses"}),
})

# Runtime override: populated from external source via set_derby_pairs().
# Falls back to the hardcoded _DERBY_PAIRS when None (offline / dev mode).
_derby_pairs_override: frozenset[frozenset[str]] | None = None


def set_derby_pairs(pairs: set[frozenset[str]]) -> None:
    """Override the derby-pairs lookup with pairs loaded from an external source."""
    global _derby_pairs_override
    if pairs:
        _derby_pairs_override = frozenset(pairs)


def is_derby(home: str, away: str) -> bool:
    """Return True when the pair is a known same-city / local rivalry."""
    pool = _derby_pairs_override if _derby_pairs_override is not None else _DERBY_PAIRS
    return frozenset({home, away}) in pool


MATCH_FEATURE_COLS = [
    "form_points_gap",
    "forward_goals_gap",
    "defense_gap",
    "cards_gap",
    "corners_gap",
    "rest_gap",
    "fatigue_gap",
    "season_points_gap",
    "h2h_gap",
    "h2h_goal_diff",
    "injury_gap",
    "lineup_strength_gap",
    "league_idx",
    # ── NEW v2 features ──────────────────────────────────────────────────────
    "home_role_gap",   # Home team home-only PPG minus away team away-only PPG
    "momentum_gap",    # OLS slope of last-5 points (home minus away) — trend
    "derby_flag",      # 1.0 if local/same-city rivalry
    "sot_gap",         # Shots-on-target differential per game last-5
]

PLAYER_FEATURE_COLS = [
    "minutes",
    "xg",
    "xa",
    "key_passes",
    "shots_on_target",
    "rating",
    "fouls",
    "yellow_cards",
    "red_cards",
]


@dataclass
class MatchModelBundle:
    model: Any
    feature_cols: list[str]


@dataclass
class PlayerModelBundle:
    goal_model: Any
    assist_model: Any
    card_model: Any
    feature_cols: list[str]


@dataclass
class TeamState:
    recent_points: deque
    recent_gf: deque
    recent_ga: deque
    recent_cards: deque
    recent_corners_diff: deque
    recent_sot_diff: deque
    dates: deque
    last_date: pd.Timestamp | None
    total_points: float
    total_matches: int
    home_points: float
    home_matches: int
    away_points: float
    away_matches: int


def _new_state(window: int) -> TeamState:
    return TeamState(
        recent_points=deque(maxlen=window),
        recent_gf=deque(maxlen=window),
        recent_ga=deque(maxlen=window),
        recent_cards=deque(maxlen=window),
        recent_corners_diff=deque(maxlen=window),
        recent_sot_diff=deque(maxlen=window),
        dates=deque(maxlen=40),
        last_date=None,
        total_points=0.0,
        total_matches=0,
        home_points=0.0,
        home_matches=0,
        away_points=0.0,
        away_matches=0,
    )


def _momentum_slope(points_deque: deque) -> float:
    """OLS slope of a rolling points sequence, normalised to ±0.9 range."""
    vals = list(points_deque)
    n = len(vals)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.array(vals, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean))) / denom / 3.0


def _mean_or_default(values: deque, default: float) -> float:
    return float(np.mean(values)) if values else default


def _summarize_state(state: TeamState, current_date: pd.Timestamp) -> dict[str, float]:
    rest_days = 7.0
    if state.last_date is not None:
        rest_days = float(max((current_date - state.last_date).days, 0))
    matches_last8 = float(sum((current_date - d).days <= 8 for d in state.dates))
    season_ppg = state.total_points / max(state.total_matches, 1)
    return {
        "form_points": _mean_or_default(state.recent_points, 1.35),
        "forward_goals": _mean_or_default(state.recent_gf, 1.25),
        "defense_ga": _mean_or_default(state.recent_ga, 1.25),
        "cards": _mean_or_default(state.recent_cards, 1.8),
        "corners_diff": _mean_or_default(state.recent_corners_diff, 0.0),
        "rest_days": rest_days,
        "matches_last8": matches_last8,
        "season_ppg": season_ppg,
        # ── NEW v2 ──────────────────────────────────────────────────────────
        "home_ppg": state.home_points / max(state.home_matches, 1),
        "away_ppg": state.away_points / max(state.away_matches, 1),
        "sot_diff": _mean_or_default(state.recent_sot_diff, 0.0),
        "momentum_slope": _momentum_slope(state.recent_points),
    }


def _h2h_summary(
    records: list[dict[str, Any]],
    current_home: str,
    current_away: str,
    current_date: pd.Timestamp,
    half_life_days: float = 900.0,
) -> tuple[float, float]:
    if not records:
        return 0.0, 0.0
    w_sum = 0.0
    win_sum = 0.0
    gd_sum = 0.0

    for rec in records:
        age = max((current_date - rec["date"]).days, 0)
        w = float(np.exp(-age / max(half_life_days, 1.0)))
        if rec["home"] == current_home and rec["away"] == current_away:
            perspective_result = rec["result"]
            perspective_gd = rec["goal_diff_home"]
        else:
            if rec["result"] == "H":
                perspective_result = "A"
            elif rec["result"] == "A":
                perspective_result = "H"
            else:
                perspective_result = "D"
            perspective_gd = -rec["goal_diff_home"]

        if perspective_result == "H":
            win_val = 1.0
        elif perspective_result == "D":
            win_val = 0.5
        else:
            win_val = 0.0

        win_sum += w * win_val
        gd_sum += w * perspective_gd
        w_sum += w

    if w_sum <= 0:
        return 0.0, 0.0
    return float((win_sum / w_sum) - 0.5), float(gd_sum / w_sum)


def build_match_training_data(
    historical: pd.DataFrame,
    injuries_df: pd.DataFrame | None = None,
    lineup_strength_map: dict[str, float] | None = None,
    window: int = 5,
    max_training_years: int = 5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = historical.copy()
    df = df.loc[df["result_ft"].isin(["H", "D", "A"])].copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.loc[df["match_date"].notna()].sort_values(["match_date", "league_code"]).reset_index(drop=True)

    # ── Speed optimisation: cap training rows to recent seasons ─────────────
    # Sample-weights already decay with exp(-age/500 days); data >5 years old
    # contributes <3% weight.  Slicing to 5 years cuts the itertuples loop from
    # ~41 K rows to ~10 K rows with no meaningful accuracy loss.
    if max_training_years > 0 and not df.empty:
        cutoff = df["match_date"].max() - pd.DateOffset(years=max_training_years)
        df = df.loc[df["match_date"] >= cutoff].reset_index(drop=True)

    for col in (
        "home_goals_ft",
        "away_goals_ft",
        "home_corners",
        "away_corners",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_red_cards",
        "away_red_cards",
        "home_shots_on_target",
        "away_shots_on_target",
    ):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    injuries_df = injuries_df.copy() if injuries_df is not None else pd.DataFrame()
    # ── Pre-index injuries by team so the inner loop is O(1) ────────────────
    _injury_by_team: dict[str, pd.DataFrame] = {}
    if not injuries_df.empty:
        date_col = "date" if "date" in injuries_df.columns else "report_date" if "report_date" in injuries_df.columns else None
        team_col = "team" if "team" in injuries_df.columns else "team_name" if "team_name" in injuries_df.columns else None
        if date_col and team_col:
            injuries_df[date_col] = pd.to_datetime(injuries_df[date_col], errors="coerce")
            if "importance_score" not in injuries_df.columns:
                injuries_df["importance_score"] = 1.0
            injuries_df["importance_score"] = pd.to_numeric(injuries_df["importance_score"], errors="coerce").fillna(1.0)
            injuries_df = injuries_df.loc[injuries_df[date_col].notna()].copy()
            injuries_df = injuries_df.rename(columns={date_col: "injury_date", team_col: "team"})
            for _team, _grp in injuries_df.groupby("team"):
                _injury_by_team[str(_team)] = _grp.reset_index(drop=True)
        else:
            injuries_df = pd.DataFrame()

    state: dict[tuple[str, str], TeamState] = defaultdict(lambda: _new_state(window))
    h2h_state: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    league_codes = sorted(df["league_code"].astype(str).dropna().unique())
    league_map = {code: idx for idx, code in enumerate(league_codes)}

    feats: list[dict[str, float]] = []
    targets: list[int] = []
    dates: list[pd.Timestamp] = []

    for row in df.itertuples(index=False):
        mdate = pd.Timestamp(row.match_date)
        league = str(row.league_code)
        home = str(row.home_team)
        away = str(row.away_team)

        hkey = (league, home)
        akey = (league, away)
        hs = _summarize_state(state[hkey], mdate)
        aw = _summarize_state(state[akey], mdate)

        pair_key = (league, min(home, away), max(home, away))
        h2h_gap, h2h_gd = _h2h_summary(h2h_state[pair_key], home, away, mdate)

        injury_gap = 0.0
        if _injury_by_team:
            one_week = mdate - pd.Timedelta(days=7)
            def _inj_score(team: str) -> float:
                grp = _injury_by_team.get(team)
                if grp is None:
                    return 0.0
                mask = (grp["injury_date"] >= one_week) & (grp["injury_date"] <= mdate)
                return float(grp.loc[mask, "importance_score"].sum())
            injury_gap = float(_inj_score(away) - _inj_score(home))

        lineup_strength_map = lineup_strength_map or {}
        lineup_strength_gap = float(lineup_strength_map.get(home, 0.0) - lineup_strength_map.get(away, 0.0))

        feats.append(
            {
                "form_points_gap": hs["form_points"] - aw["form_points"],
                "forward_goals_gap": hs["forward_goals"] - aw["forward_goals"],
                "defense_gap": aw["defense_ga"] - hs["defense_ga"],
                "cards_gap": aw["cards"] - hs["cards"],
                "corners_gap": hs["corners_diff"] - aw["corners_diff"],
                "rest_gap": hs["rest_days"] - aw["rest_days"],
                "fatigue_gap": aw["matches_last8"] - hs["matches_last8"],
                "season_points_gap": hs["season_ppg"] - aw["season_ppg"],
                "h2h_gap": h2h_gap,
                "h2h_goal_diff": h2h_gd,
                "injury_gap": injury_gap,
                "lineup_strength_gap": lineup_strength_gap,
                "league_idx": float(league_map.get(league, 0)),
                # ── NEW v2 features ──────────────────────────────────────────
                "home_role_gap": hs["home_ppg"] - aw["away_ppg"],
                "momentum_gap":  hs["momentum_slope"] - aw["momentum_slope"],
                "derby_flag":    float(is_derby(home, away)),
                "sot_gap":       hs["sot_diff"] - aw["sot_diff"],
            }
        )

        targets.append(RESULT_TO_CLASS[str(row.result_ft)])
        dates.append(mdate)

        if str(row.result_ft) == "H":
            hp = 3.0
            ap = 0.0
        elif str(row.result_ft) == "A":
            hp = 0.0
            ap = 3.0
        else:
            hp = 1.0
            ap = 1.0

        home_cards = float(row.home_yellow_cards) + 2.0 * float(row.home_red_cards)
        away_cards = float(row.away_yellow_cards) + 2.0 * float(row.away_red_cards)
        home_cd = float(row.home_corners) - float(row.away_corners)
        away_cd = -home_cd
        home_sot_d = float(row.home_shots_on_target) - float(row.away_shots_on_target)
        away_sot_d = -home_sot_d

        for key, pts, gf, ga, cards, cd, sot_d, is_home in (
            (hkey, hp, float(row.home_goals_ft), float(row.away_goals_ft), home_cards, home_cd, home_sot_d, True),
            (akey, ap, float(row.away_goals_ft), float(row.home_goals_ft), away_cards, away_cd, away_sot_d, False),
        ):
            s = state[key]
            s.recent_points.append(pts)
            s.recent_gf.append(gf)
            s.recent_ga.append(ga)
            s.recent_cards.append(cards)
            s.recent_corners_diff.append(cd)
            s.recent_sot_diff.append(sot_d)
            s.dates.append(mdate)
            s.last_date = mdate
            s.total_points += pts
            s.total_matches += 1
            if is_home:
                s.home_points += pts
                s.home_matches += 1
            else:
                s.away_points += pts
                s.away_matches += 1

        h2h_state[pair_key].append(
            {
                "date": mdate,
                "home": home,
                "away": away,
                "result": str(row.result_ft),
                "goal_diff_home": float(row.home_goals_ft) - float(row.away_goals_ft),
            }
        )

    X = pd.DataFrame(feats)
    y = pd.Series(targets)

    max_date = max(dates) if dates else pd.Timestamp.today()
    age = np.array([(max_date - d).days for d in dates], dtype=float)
    sample_weight = np.exp(-age / 500.0)
    return X, y, pd.Series(sample_weight)


def train_match_model(
    historical: pd.DataFrame,
    injuries_df: pd.DataFrame | None = None,
    lineup_strength_map: dict[str, float] | None = None,
) -> MatchModelBundle | None:
    X, y, sample_w = build_match_training_data(
        historical=historical,
        injuries_df=injuries_df,
        lineup_strength_map=lineup_strength_map,
        window=5,
        max_training_years=3,   # 3 yrs ≈ 6 K rows; sample-weight gives <1% to older data
    )

    if len(X) < 300 or y.nunique() < 3:
        return None

    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=80,        # 17 features; 80 trees sufficient, faster on cloud
        max_depth=3,            # shallower → faster; enough for this feature set
        learning_rate=0.10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=1,               # single thread avoids overhead on shared cloud CPU
        random_state=42,
    )
    model.fit(X[MATCH_FEATURE_COLS], y, sample_weight=sample_w)
    return MatchModelBundle(model=model, feature_cols=MATCH_FEATURE_COLS)


def predict_match_proba(bundle: MatchModelBundle, features: dict[str, float]) -> dict[str, float]:
    X = pd.DataFrame([{k: float(features.get(k, 0.0)) for k in bundle.feature_cols}])
    probs = bundle.model.predict_proba(X)[0]
    return {
        "H": float(probs[0]),
        "D": float(probs[1]),
        "A": float(probs[2]),
    }


def build_player_training_data(contrib_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    df = contrib_df.copy()
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int), pd.Series(dtype=int), pd.Series(dtype=float)

    if "match_date" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int), pd.Series(dtype=int), pd.Series(dtype=float)

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.loc[df["match_date"].notna()].copy()

    for c in PLAYER_FEATURE_COLS + ["goals", "assists"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[PLAYER_FEATURE_COLS].copy()
    y_goal = (df["goals"] > 0).astype(int)
    y_assist = (df["assists"] > 0).astype(int)
    cards_total = df.get("yellow_cards", 0.0) + df.get("red_cards", 0.0)
    y_card = (cards_total > 0).astype(int)

    max_date = df["match_date"].max()
    age = (max_date - df["match_date"]).dt.days.fillna(0).astype(float)
    sample_weight = np.exp(-age / 300.0)
    return X, y_goal, y_assist, y_card, sample_weight


def train_player_models(contrib_df: pd.DataFrame) -> PlayerModelBundle | None:
    X, y_goal, y_assist, y_card, sample_w = build_player_training_data(contrib_df)
    if X.empty:
        return None
    if y_goal.nunique() < 2 or y_assist.nunique() < 2 or y_card.nunique() < 2:
        return None

    from xgboost import XGBClassifier

    def _train(y: pd.Series) -> Any:
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=40,        # 9 features; 40 trees fast on cloud
            max_depth=3,
            learning_rate=0.10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,               # single thread on shared cloud CPU
            random_state=42,
        )
        model.fit(X[PLAYER_FEATURE_COLS], y, sample_weight=sample_w)
        return model

    return PlayerModelBundle(
        goal_model=_train(y_goal),
        assist_model=_train(y_assist),
        card_model=_train(y_card),
        feature_cols=PLAYER_FEATURE_COLS,
    )


def player_probabilities_for_team(
    team: str,
    contrib_df: pd.DataFrame,
    bundle: PlayerModelBundle,
    as_of_date: pd.Timestamp,
    top_n: int = 15,
) -> pd.DataFrame:
    if contrib_df.empty:
        return pd.DataFrame()

    team_col = "team" if "team" in contrib_df.columns else "team_name" if "team_name" in contrib_df.columns else None
    player_col = "player" if "player" in contrib_df.columns else "player_name" if "player_name" in contrib_df.columns else None
    if team_col is None or player_col is None or "match_date" not in contrib_df.columns:
        return pd.DataFrame()

    df = contrib_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.loc[df["match_date"].notna() & (df["match_date"] <= as_of_date) & (df[team_col] == team)].copy()
    if df.empty:
        return pd.DataFrame()

    for c in bundle.feature_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    grouped = (
        df.sort_values("match_date")
        .groupby([team_col, player_col], dropna=False)
        .tail(5)
        .groupby([team_col, player_col], dropna=False)[bundle.feature_cols]
        .mean()
        .reset_index()
        .rename(columns={team_col: "team", player_col: "player"})
    )
    if grouped.empty:
        return pd.DataFrame()

    X = grouped[bundle.feature_cols]
    grouped["prob_score"] = bundle.goal_model.predict_proba(X)[:, 1]
    grouped["prob_assist"] = bundle.assist_model.predict_proba(X)[:, 1]
    grouped["prob_card"] = bundle.card_model.predict_proba(X)[:, 1]
    grouped = grouped.sort_values(["prob_score", "prob_assist"], ascending=False).head(top_n)
    return grouped[["team", "player", "prob_score", "prob_assist", "prob_card"]]
