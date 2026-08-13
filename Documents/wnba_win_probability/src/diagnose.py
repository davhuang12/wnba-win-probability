"""
Diagnostic: checks whether load_latest_team_stats() in predict_today.py
is handing back stats that are current, or stale by one (or more) games.

For each team, compares:
  - LATEST_STATS_ROW_DATE: the GAME_DATE of the row picked by tail(1)
    in load_latest_team_stats() (i.e. what predict_today.py would use today)
  - TRUE_LAST_GAME_DATE: the team's actual most recent game date, pulled
    directly from the raw 2026 game log (data/raw/game_logs_2026.csv)

If TRUE_LAST_GAME_DATE > LATEST_STATS_ROW_DATE for a team, that team's
most recent result is not reflected in predict_today.py's features yet.

Run this from the project root (same folder as build_dataset.py / predict_today.py).
"""

import pandas as pd
from build_dataset import PROJECT_ROOT
from predict_today import load_latest_team_stats

RAW_2026_PATH = PROJECT_ROOT / "data" / "raw" / "game_logs_2026.csv"


def get_true_last_game_dates() -> pd.DataFrame:
    """
    Directly from the raw game log (no shift, no rolling logic) --
    ground truth for each team's actual most recent game date.
    """
    df = pd.read_csv(RAW_2026_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    latest = (
        df.sort_values("GAME_DATE")
        .groupby("TEAM_ID")
        .tail(1)[["TEAM_ID", "TEAM_NAME", "GAME_DATE"]]
        .rename(columns={"GAME_DATE": "TRUE_LAST_GAME_DATE"})
        .set_index("TEAM_ID")
    )
    return latest


def main():
    stats_used = load_latest_team_stats()  # what predict_today.py actually uses
    true_latest = get_true_last_game_dates()  # ground truth

    comparison = stats_used[["TEAM_NAME", "GAME_DATE"]].rename(
        columns={"GAME_DATE": "LATEST_STATS_ROW_DATE"}
    ).join(true_latest[["TRUE_LAST_GAME_DATE"]], how="left")

    comparison["GAMES_BEHIND"] = comparison["TRUE_LAST_GAME_DATE"] > comparison["LATEST_STATS_ROW_DATE"]

    comparison = comparison.sort_values("TEAM_NAME")

    pd.set_option("display.width", 120)
    print(comparison.to_string())

    stale_teams = comparison[comparison["GAMES_BEHIND"]]
    print()
    if stale_teams.empty:
        print("No stale teams found -- every team's stats reflect their most recent game.")
    else:
        print(f"{len(stale_teams)} team(s) have stats that predate their most recent played game:")
        print(stale_teams[["TEAM_NAME", "LATEST_STATS_ROW_DATE", "TRUE_LAST_GAME_DATE"]].to_string(index=False))


if __name__ == "__main__":
    main()