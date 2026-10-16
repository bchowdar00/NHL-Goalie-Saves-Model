"""Fetch the official calendar date for every regular-season game.

NHL game_id is not monotonic with the date a game was actually played
(rescheduled games keep their original id), so the rolling features and the
train/test split need each game's real date rather than its id order.

This script walks the NHL public schedule API week by week across the seasons we
use and writes ``data/game_dates.csv`` with columns ``season,game_id,date`` for
every regular-season game (gameType == 2). ``season`` and ``game_id`` match the
columns MoneyPuck uses (``season`` is the start year; ``game_id`` is the last
six digits of the NHL gamePk).

The output CSV is small and committed so the pipeline reproduces offline; rerun
this only if the season window changes. Uses the standard library only.

    python scripts/fetch_game_dates.py
"""

import csv
import datetime as dt
import json
import os
import time
import urllib.request

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "data", "game_dates.csv")
ENDPOINT = "https://api-web.nhle.com/v1/schedule/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Walk every week whose Monday falls in this range; covers the 2024-25 and
# 2025-26 regular seasons with margin.
START = "2024-09-15"
STOP = dt.date(2026, 7, 1)
REGULAR_SEASON = 2  # NHL gameType digit


def fetch_week(start_date):
    req = urllib.request.Request(ENDPOINT + start_date, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main():
    games = {}  # (season, game_id) -> date
    cursor = START
    while True:
        data = fetch_week(cursor)
        for day in data.get("gameWeek", []):
            date = day.get("date")
            for g in day.get("games", []):
                if g.get("gameType") != REGULAR_SEASON:
                    continue
                gid = int(g["id"])
                games[(gid // 1000000, gid % 1000000)] = date
        nxt = data.get("nextStartDate")
        if not nxt or dt.date.fromisoformat(nxt) >= STOP:
            break
        cursor = nxt
        time.sleep(0.3)

    rows = sorted(games.items())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["season", "game_id", "date"])
        for (season, game_id), date in rows:
            w.writerow([season, game_id, date])
    print(f"wrote {len(rows)} regular-season game dates to {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
