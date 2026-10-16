# NHL Goalie Saves Model

A forecasting model that predicts how many saves an NHL goalie will make in a
game, trained on public shot-level data. The honest result: the model beats
naive baselines by a small margin on one chronological holdout (about 0.21
saves of MAE against the train mean).

## What it predicts

For a goalie's start, the number of saves (shots on goal stopped). The model
is the product of two sub-models:

- a **shot-volume model** that predicts how many shots the goalie will face, and
- a **save-percentage model** that predicts the share they stop (clipped to a
  plausible 0.86–0.96 range).

Their product is the projected saves.

## Data

[MoneyPuck](https://moneypuck.com) public shot-level data for the 2024-25 and
2025-26 seasons (labelled `season` 2024 and 2025 in the files, by start year).
Each shot row carries the game, teams, shooter/goalie, shot distance, and
expected-goal value. The raw files are large (~68 MB per season uncompressed)
and are **not committed**; `download_nhl_data.py` fetches them.

**Regular season only.** The files also contain the playoffs, but playoff hockey
differs systematically from the regular season, so the pipeline keeps only
regular-season games (`feature_engineering.py` filters on the game-type digit of
`game_id`) for a single, homogeneous game population in both train and test.

MoneyPuck shot files carry no calendar date, and NHL `game_id` is not monotonic
with date (a rescheduled game keeps its original id). So `scripts/fetch_game_dates.py`
pulls each game's real date from the NHL public schedule API into a small,
committed `data/game_dates.csv`, and the pipeline orders everything by that date.

## Features

Built per goalie-game in `feature_engineering.py`, all rolling features use only
prior games (`shift(1)`), so no game sees its own or future data:

- `team_defense_val` — the goalie's team's rolling-10 shots allowed
- `opp_offense_val` / `opp_dist_val` — the opponent's rolling-10 shots-for and
  average shot distance
- `team_pace_val` — the team's rolling-10 game scoring pace
- `blender_val` — an interaction term (opponent offense × team defense / 30)
- `roll_10_gsax` — the goalie's rolling-10 goals-saved-above-expected
- `is_home` — home/away flag

Only goalie-games with at least 13 shots faced are kept. Rolling features need
at least 3 prior games, so rows without that history are dropped: 322 of 5,257
goalie-games (each goalie's and team's first few games, including the season
openers). `feature_engineering.py` prints this count.

## Method

A chronological holdout: the goalie-games are sorted by real game date, the
first ~85% are used for training and the rest for testing, with the split placed
on a date boundary so that no single game straddles train and test and no
training game is dated after the test period begins. The model is always tested
on games that happened after the ones it learned from. Both sub-models are
gradient-boosted regressors (scikit-learn `GradientBoostingRegressor`).

The model is compared against two naive baselines on the same test set:

1. **Train-mean** — predict the training set's mean saves for every game.
2. **Own rolling-10** — predict each goalie's own average saves over their
   previous 10 games (prior games only).

## Results

Produced by `python evaluate.py` on the two-season dataset:

```
========================================================
NHL goalie-saves model: chronological holdout evaluation
========================================================
Train goalie-games: 4191  (2024-10-17 -> 2026-02-27)
Test goalie-games (n_test): 744  (2026-02-28 -> 2026-04-16)
(Chronological order and train/test split are by real game date.)
--------------------------------------------------------
Predictor                             Test MAE (saves)
--------------------------------------------------------
Train-mean baseline (24.69)                      5.320
Own rolling-10 baseline                          5.381
Model                                            5.108
--------------------------------------------------------
Model vs train-mean:     +0.212 saves
Model vs own rolling-10: +0.273 saves
========================================================
```

So the model's mean absolute error is about **5.1 saves**, roughly **0.21–0.27
saves (about 4–5%) better** than either naive baseline.

## Limitations

- **The improvement is small.** Beating a goalie's own 10-game average by ~0.3
  saves is a thin gain. Game-to-game save totals are noisy (a goalie
  facing 30 shots can make anywhere from the low 20s to 30 saves), and no set of
  pre-game features captures most of that variance.
- It is one train/test split with no confidence interval, so the 0.21–0.27
  save gain is a point estimate, not a measured range.
- The window is two seasons of regular-season games only.
- A goalie's first 3 starts are dropped (see above), so the test set leaves out
  newly called-up goalies, which are among the hardest cases to predict.
- Features are team- and schedule-level; the model has no information on
  in-game events, lineups, or injuries.
- The 13-shot minimum filters out relief appearances, but it also drops starters
  who were pulled early, which conditions mildly on the game's outcome.

## How to run

```bash
pip install -r requirements.txt

python download_nhl_data.py     # downloads two ~20 MB zips into data/ (~136 MB uncompressed)
python feature_engineering.py   # builds data/model_features.csv (uses data/game_dates.csv)
python evaluate.py              # prints the table above
```

`data/game_dates.csv` is committed; rerun `scripts/fetch_game_dates.py` only if
the season window changes.

Or run all three steps at once:

```bash
python master_pipeline.py
```

## License

[MIT](LICENSE)
