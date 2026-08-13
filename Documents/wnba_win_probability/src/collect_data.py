import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\davhu\Documents\wnba_win_probability")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SEASONS = ["2019","2020","2021","2022","2023","2024","2025","2026"]

def fetch_season_game_logs(season: str) -> pd.DataFrame:
    """
    Downloads team-level WNBA game logs for one regular season.

    Example season format:
    "2024" --> unlike NBA's 2023-24

    Important:
    Each WNBA game appears twice in this dataset:
    - one row for the home team
    - one row for the away team
    """

    print(f"Downloading WNBA game logs for {season}...")

    game_log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
        league_id = "10"
    )

    df = game_log.get_data_frames()[0]
    print(df.columns.tolist())

    print(f"Downloaded {len(df)} rows for {season}")

    return df

def save_raw_data(df: pd.DataFrame, season: str) -> None:
    """
    Saves the downloaded season data as a CSV file.
    """

    RAW_DATA_DIR.mkdir(parents = True, exist_ok=True)

    file_path = f"{RAW_DATA_DIR}/game_logs_{season}.csv"

    df.to_csv(file_path, index=False)

    print(f"Saved data to {file_path}")


def main():
    for season in SEASONS:
        df = fetch_season_game_logs(season)
        save_raw_data(df, season)



if __name__ == "__main__":
    main()