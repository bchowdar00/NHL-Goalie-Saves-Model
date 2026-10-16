"""Run the full pipeline end to end: download -> features -> evaluate.

Equivalent to running, in order:
    python download_nhl_data.py
    python feature_engineering.py
    python evaluate.py

(data/game_dates.csv is committed; rerun scripts/fetch_game_dates.py only if the
season window changes.)
"""

import sys

try:
    import download_nhl_data
    import feature_engineering
    import evaluate
except ImportError as e:
    print("CRITICAL ERROR: Missing required scripts.")
    print(f"Details: {e}")
    sys.exit(1)


def main():
    print("\n[STEP 1/3] Downloading shot data from MoneyPuck...")
    download_nhl_data.main()

    print("\n[STEP 2/3] Engineering features...")
    feature_engineering.process_data()

    print("\n[STEP 3/3] Evaluating model against baselines...")
    evaluate.main()


if __name__ == "__main__":
    main()
