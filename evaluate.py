"""Chronological holdout evaluation for the goalie-saves model.

Trains on earlier games and tests on later ones (no shuffling), then compares the
model against two naive baselines on the same test set:

  1. Train-mean      - predict every goalie-game as the mean saves in the train set.
  2. Own rolling-10   - predict each goalie-game as that goalie's own average saves
                        over their previous 10 games (prior games only; the first
                        game with no history falls back to the train mean).

Run feature_engineering.py first to produce data/model_features.csv.

Usage:
    python evaluate.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = os.path.join("data", "model_features.csv")

# Fraction of games (chronological) used for training; the rest is the holdout.
TRAIN_FRAC = 0.85

VOL_FEATURES = ["blender_val", "team_pace_val", "opp_dist_val",
                "team_defense_val", "is_home"]
EFF_FEATURES = ["roll_10_gsax", "team_defense_val"]


def main():
    if not os.path.exists(DATA_PATH):
        print("Error: data/model_features.csv not found. Run feature_engineering.py first.")
        return

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Defensive: the merge fix in feature_engineering.py should already guarantee this.
    df = df.drop_duplicates(subset=["season", "game_id", "goalie_name"])
    df = df[df["opp_offense_val"] > 0].copy()

    # Chronological order by real calendar date (game_id is not monotonic with
    # date). game_id breaks ties within a date.
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

    df["sv_pct"] = df["saves"] / df["shots_faced"].replace(0, 1)

    # Own rolling-10 baseline, computed on the full ordered frame using only prior
    # games (shift(1)), so a game never sees itself or anything after it.
    df["own_roll10_saves"] = df.groupby("goalie_name")["saves"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )

    # Split on a date boundary: every game played on or after the boundary date is
    # test, everything earlier is train. Because a game has a single date, no game
    # straddles the split and no training game is dated after the test period.
    split_pos = int(len(df) * TRAIN_FRAC)
    boundary_date = df.loc[split_pos, "date"]
    is_train = df["date"] < boundary_date

    train_df = df[is_train].copy()
    test_df = df[~is_train].copy()
    test_df = test_df[test_df["shots_faced"] >= 13].copy()

    train_mean_saves = train_df["saves"].mean()

    # --- Model: shot-volume model x clipped save-percentage model ---
    vol_model = GradientBoostingRegressor(
        loss="squared_error", n_estimators=500, max_depth=3,
        learning_rate=0.02, subsample=0.8, random_state=42)
    vol_model.fit(train_df[VOL_FEATURES], train_df["shots_faced"])

    eff_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=2, learning_rate=0.02,
        subsample=0.8, random_state=42)
    eff_model.fit(train_df[EFF_FEATURES], train_df["sv_pct"])

    pred_shots = vol_model.predict(test_df[VOL_FEATURES])
    pred_sv = np.clip(eff_model.predict(test_df[EFF_FEATURES]), 0.86, 0.96)
    model_pred = pred_shots * pred_sv

    # --- Baselines on the same test set ---
    baseline_mean = np.full(len(test_df), train_mean_saves)
    # First-ever game for a goalie has no rolling history -> fall back to train mean.
    own_roll = test_df["own_roll10_saves"].fillna(train_mean_saves).to_numpy()

    mae_model = mean_absolute_error(test_df["saves"], model_pred)
    mae_mean = mean_absolute_error(test_df["saves"], baseline_mean)
    mae_own = mean_absolute_error(test_df["saves"], own_roll)

    def span(frame):
        lo = frame["date"].min().date()
        hi = frame["date"].max().date()
        return f"{lo} -> {hi}"

    print("=" * 56)
    print("NHL goalie-saves model: chronological holdout evaluation")
    print("=" * 56)
    print(f"Train goalie-games: {len(train_df)}  ({span(train_df)})")
    print(f"Test goalie-games (n_test): {len(test_df)}  ({span(test_df)})")
    print("(Chronological order and train/test split are by real game date.)")
    print("-" * 56)
    print(f"{'Predictor':<34}{'Test MAE (saves)':>20}")
    print("-" * 56)
    print(f"{'Train-mean baseline (' + format(train_mean_saves, '.2f') + ')':<34}{mae_mean:>20.3f}")
    print(f"{'Own rolling-10 baseline':<34}{mae_own:>20.3f}")
    print(f"{'Model':<34}{mae_model:>20.3f}")
    print("-" * 56)
    print(f"Model vs train-mean:     {mae_mean - mae_model:+.3f} saves")
    print(f"Model vs own rolling-10: {mae_own - mae_model:+.3f} saves")
    print("=" * 56)


if __name__ == "__main__":
    main()
