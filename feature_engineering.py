import pandas as pd
import numpy as np
import os

# Configuration
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "model_features.csv")
# Real calendar date per game, built by scripts/fetch_game_dates.py. game_id is
# not monotonic with date (rescheduled games keep their id), so rolling windows
# and the train/test split are ordered by this date, not by game_id.
DATES_FILE = os.path.join(DATA_DIR, "game_dates.csv")

def load_shot_data(year):
    filename = os.path.join(DATA_DIR, f"shots_{year}.csv")
    if not os.path.exists(filename): return None
    df = pd.read_csv(filename)
    df['season'] = year
    return df

def process_data():
    print("--- Processing Features: Pace + Distance ---")

    shots_24 = load_shot_data(2024)
    shots_25 = load_shot_data(2025)
    if shots_24 is None:
        print("Error: Missing shot data.")
        return

    full_shots = pd.concat([shots_24, shots_25], ignore_index=True)

    # Regular season only. NHL game_id is season + game-type + number; the
    # game-type digit is 2 for regular season and 3 for playoffs. Playoff hockey
    # differs systematically, so we keep a single, homogeneous game population.
    full_shots = full_shots[(full_shots['game_id'] // 10000) == 2].copy()

    # Mappings
    full_shots['team'] = np.where(full_shots['isHomeTeam'] == 1, full_shots['homeTeamCode'], full_shots['awayTeamCode'])
    full_shots['opponent'] = np.where(full_shots['isHomeTeam'] == 1, full_shots['awayTeamCode'], full_shots['homeTeamCode'])
    full_shots['is_sog'] = (full_shots['goal'] == 1) | (full_shots['event'] == 'SHOT')

    # Distance Fill
    if 'arenaAdjustedShotDistance' in full_shots.columns:
        full_shots['dist'] = full_shots['arenaAdjustedShotDistance'].fillna(35.0)
    else:
        full_shots['dist'] = 35.0

    # --- 1. GAME LEVEL STATS ---
    # We aggregate per game per team to get "Pace" (Total Goals)
    game_stats = full_shots.groupby(['season', 'game_id', 'team', 'opponent']).agg({
        'is_sog': 'sum',
        'goal': 'sum',
        'dist': 'mean'
    }).reset_index().rename(columns={
        'is_sog': 'shots',
        'goal': 'goals',
        'dist': 'avg_dist'
    })

    # Calculate "Game Pace" (Total Goals in the game by both teams)
    game_totals = game_stats.groupby(['season', 'game_id'])['goals'].sum().reset_index(name='total_game_goals')
    game_stats = game_stats.merge(game_totals, on=['season', 'game_id'])

    # Attach the real calendar date to each game.
    if not os.path.exists(DATES_FILE):
        print(f"Error: {DATES_FILE} not found. Run scripts/fetch_game_dates.py first.")
        return
    game_dates = pd.read_csv(DATES_FILE, parse_dates=['date'])
    game_stats = game_stats.merge(game_dates, on=['season', 'game_id'], how='left')
    missing = game_stats['date'].isna().sum()
    assert missing == 0, f"{missing} team-games have no calendar date"

    # --- 2. ROLLING FEATURES (ordered by real date) ---
    game_stats = game_stats.sort_values(['team', 'date', 'game_id'])

    # Team Pace (Do they play high scoring games?)
    game_stats['roll_10_pace'] = game_stats.groupby('team')['total_game_goals'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )

    # Team Defense (Shots Allowed - derived from Opponent's Shots)
    # We need to know what the Team ALLOWED. We map the opponent's shot count.
    defensive_lookup = game_stats[['season', 'game_id', 'team', 'shots']].rename(
        columns={'team': 'opponent', 'shots': 'shots_against'}
    )
    game_stats = game_stats.merge(defensive_lookup, on=['season', 'game_id', 'opponent'], how='left')

    # Roll the stats
    game_stats = game_stats.sort_values(['team', 'date', 'game_id']) # Re-sort needed after merge

    game_stats['roll_10_allowed'] = game_stats.groupby('team')['shots_against'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    game_stats['roll_10_shots_for'] = game_stats.groupby('team')['shots'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    game_stats['roll_10_dist'] = game_stats.groupby('team')['avg_dist'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )

    # --- 3. GOALIE LOGS ---
    sog_data = full_shots[full_shots['is_sog']]
    goalie_starts = sog_data.groupby(['season', 'game_id', 'goalieNameForShot']).agg({
        'team': 'first', # Shooter (Opponent)
        'is_sog': 'count',
        'goal': 'sum',
        'xGoal': 'sum'
    }).reset_index().rename(columns={
        'goalieNameForShot': 'goalie_name',
        'team': 'opponent',
        'is_sog': 'shots_faced',
        'goal': 'goals_allowed',
        'xGoal': 'xg_faced'
    })

    goalie_starts['saves'] = goalie_starts['shots_faced'] - goalie_starts['goals_allowed']
    goalie_starts['GSAx'] = goalie_starts['xg_faced'] - goalie_starts['goals_allowed']

    # Filter Partials (The "13 Shot Rule")
    goalie_starts = goalie_starts[goalie_starts['shots_faced'] >= 13].copy()

    # Map Goalie's Team
    matchup_map = game_stats[['season', 'game_id', 'team', 'opponent']].rename(columns={'team': 'my_team'})
    goalie_starts = goalie_starts.merge(matchup_map, on=['season', 'game_id', 'opponent'], how='left')

    # Attach the real calendar date to each goalie-game; every row must have one.
    goalie_starts = goalie_starts.merge(game_dates, on=['season', 'game_id'], how='left')
    missing = goalie_starts['date'].isna().sum()
    assert missing == 0, f"{missing} goalie-games have no calendar date"

    # --- 4. MERGE EVERYTHING ---

    # A. My Team's Stats (Defense, Pace)
    goalie_starts = goalie_starts.merge(
        game_stats[['season', 'game_id', 'team', 'roll_10_allowed', 'roll_10_pace']].rename(
            columns={'team': 'my_team', 'roll_10_allowed': 'team_defense_val', 'roll_10_pace': 'team_pace_val'}
        ),
        on=['season', 'game_id', 'my_team'], how='left'
    )

    # B. Opponent's Stats (Offense, Distance)
    goalie_starts = goalie_starts.merge(
        game_stats[['season', 'game_id', 'team', 'roll_10_shots_for', 'roll_10_dist']].rename(
            columns={'team': 'opponent', 'roll_10_shots_for': 'opp_offense_val', 'roll_10_dist': 'opp_dist_val'}
        ),
        on=['season', 'game_id', 'opponent'], how='left'
    )

    # C. Goalie Skill Rolling (ordered by real date)
    goalie_starts = goalie_starts.sort_values(['date', 'game_id'])
    goalie_starts['roll_10_gsax'] = goalie_starts.groupby('goalie_name')['GSAx'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).sum()
    )

    # D. The Blender (Final Version)
    # (Opp Offense * Team Defense) / 30.0
    goalie_starts['blender_val'] = (goalie_starts['opp_offense_val'] * goalie_starts['team_defense_val']) / 30.0

    # Home/Away
    # Include season in the key: game_id is only unique within a season, so a
    # game_id-only merge would match each goalie-game to both seasons' home teams
    # and duplicate the row.
    home_map = full_shots[['season', 'game_id', 'homeTeamCode']].drop_duplicates()
    goalie_starts = goalie_starts.merge(home_map, on=['season', 'game_id'], how='left')
    goalie_starts['is_home'] = (goalie_starts['my_team'] == goalie_starts['homeTeamCode']).astype(int)

    # Save. Rolling features need 3 prior games, so each goalie's and team's
    # first few games (including the season openers) have no value and are dropped.
    n_before = len(goalie_starts)
    goalie_starts.dropna(inplace=True)
    print(f"Dropped {n_before - len(goalie_starts)} of {n_before} goalie-games "
          f"without enough prior games for the rolling features")

    # No row should appear more than once per (season, game_id, goalie).
    dupes = goalie_starts.duplicated(subset=['season', 'game_id', 'goalie_name']).sum()
    assert dupes == 0, f"Found {dupes} duplicate (season, game_id, goalie_name) rows"

    print(f"Saved {len(goalie_starts)} goalie-games to {OUTPUT_FILE}")
    goalie_starts.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    process_data()