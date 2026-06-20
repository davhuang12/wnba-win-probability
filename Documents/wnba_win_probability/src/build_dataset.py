from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\davhu\Documents\wnba_win_probability")

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_OUTPUT_PATH = PROCESSED_DATA_DIR / f"model_dataset.csv"
TEST_OUTPUT_PATH = PROCESSED_DATA_DIR / "model_dataset_2026.csv"


SEASONS = [
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026"
]

def load_raw_data(seasons: list[str]) -> pd.DataFrame:
    all_dfs = []
    for season in seasons:
        file_path =  PROJECT_ROOT / "data" / "raw" / f"game_logs_{season}.csv"
        
        df = pd.read_csv(file_path)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]) #turn it into a date, so we can do math
        df["SEASON"] = season

        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index = True)

def create_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert W/L text into numbers.
    df["WIN"] = df["WL"].map({
        "W": 1,
        "L": 0
    })

    # vs. : home, @ : away
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.")

    return df

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates pre-game rolling statistics. This runs on every row (every game, both teams)
    stats tracked : win %, pts average, +/-, rest days
    """

    df = df.copy()

    df = df.sort_values(["SEASON", "TEAM_ID", "GAME_DATE"]) #sort by team, then by game # chronologically
    group_cols = ["SEASON", "TEAM_ID"] #what we have to consider together later

    df["SEASON_GAMES_PLAYED"] = (df.groupby(group_cols).cumcount())
    df["SEASON_WINS"] = (df.groupby(group_cols)["WIN"].transform(lambda x: x.shift(1).fillna(0).cumsum()))
    df["SEASON_LOSSES"] = (df["SEASON_GAMES_PLAYED"] - df["SEASON_WINS"])

    #season total features
    df["SEASON_WIN_PCT"] = (
        df.groupby(group_cols)["WIN"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean()) #expanding -> loop through every game
    )

    df["SEASON_PTS_AVG"] = (
        df.groupby(group_cols)["PTS"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
    )

    df["SEASON_PLUS_MINUS_AVG"] = (
        df.groupby(group_cols)["PLUS_MINUS"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
    )

    #rolling features
    df["ROLL_WIN_PCT_10"] = (
        df.groupby(group_cols)["WIN"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean()) #rolling -> loop through last x games
    )

    df["ROLL_PTS_10"] = (
        df.groupby(group_cols)["PTS"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

    df["ROLL_PLUS_MINUS_10"] = (
        df.groupby(group_cols)["PLUS_MINUS"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

    previous_game_date = df.groupby(group_cols)["GAME_DATE"].shift(1)

    df["REST_DAYS"] = (df["GAME_DATE"] - previous_game_date).dt.days
    
    return df

def create_game_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Currently, one game has 2 rows, one for away, one for home. This function merges them together
    """

    df = df.copy()

    home = df[df["IS_HOME"] == True].copy()
    away = df[df["IS_HOME"] == False].copy()

    columns_to_keep = [
        "SEASON",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "TEAM_NAME",
        "WIN",
        "PTS",
        "SEASON_WIN_PCT",
        "SEASON_PTS_AVG",
        "SEASON_PLUS_MINUS_AVG",
        "ROLL_WIN_PCT_10",
        "ROLL_PTS_10",
        "ROLL_PLUS_MINUS_10",
        "REST_DAYS",
        "SEASON_WINS",
        "SEASON_LOSSES"
    ]

    #only keep these columns
    home = home[columns_to_keep]
    away = away[columns_to_keep]

    home = home.rename(columns={
        "TEAM_ID": "HOME_TEAM_ID",
        "TEAM_NAME": "HOME_TEAM",
        "WIN": "HOME_WIN",
        "PTS": "HOME_PTS",
        "SEASON_WIN_PCT": "HOME_WIN_PCT",
        "SEASON_PTS_AVG" : "HOME_PTS_AVG",
        "SEASON_PLUS_MINUS_AVG": "HOME_PLUS_MINUS_AVG",
        "ROLL_WIN_PCT_10": "HOME_ROLL_WIN_PCT_10",
        "ROLL_PTS_10": "HOME_ROLL_PTS_10",
        "ROLL_PLUS_MINUS_10": "HOME_ROLL_PLUS_MINUS_10",
        "REST_DAYS": "HOME_REST_DAYS",
        "SEASON_WINS" : "HOME_SEASON_WINS",
        "SEASON_LOSSES" : "HOME_SEASON_LOSSES"
    })
    

    away = away.rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID",
        "TEAM_NAME": "AWAY_TEAM",
        "WIN": "AWAY_WIN",
        "PTS" : "AWAY_PTS",
        "SEASON_WIN_PCT": "AWAY_WIN_PCT",
        "SEASON_PTS_AVG" : "AWAY_PTS_AVG",
        "SEASON_PLUS_MINUS_AVG": "AWAY_PLUS_MINUS_AVG",
        "ROLL_WIN_PCT_10": "AWAY_ROLL_WIN_PCT_10",
        "ROLL_PTS_10": "AWAY_ROLL_PTS_10",
        "ROLL_PLUS_MINUS_10": "AWAY_ROLL_PLUS_MINUS_10",
        "REST_DAYS": "AWAY_REST_DAYS",
        "SEASON_WINS" : "AWAY_SEASON_WINS",
        "SEASON_LOSSES" : "AWAY_SEASON_LOSSES"
    })
    

    games = pd.merge(
        home,
        away,
        on=["SEASON", "GAME_ID", "GAME_DATE"],
        how="inner"
    )

    return games

def create_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn all the rolling features into a difference.
    """
    df = df.copy()
    
    df["WIN_PCT_DIFF"] = (df["HOME_WIN_PCT"] - df["AWAY_WIN_PCT"])
    df["WIN_PCT_DIFF_10"] = (df["HOME_ROLL_WIN_PCT_10"] - df["AWAY_ROLL_WIN_PCT_10"])

    df["PTS_DIFF"] = (df["HOME_PTS_AVG"] - df["AWAY_PTS_AVG"])
    df["PTS_DIFF_10"] = (df["HOME_ROLL_PTS_10"] - df["AWAY_ROLL_PTS_10"])

    df["PLUS_MINUS_DIFF"] = (df["HOME_PLUS_MINUS_AVG"] - df["AWAY_PLUS_MINUS_AVG"])
    df["PLUS_MINUS_DIFF_10"] = (df["HOME_ROLL_PLUS_MINUS_10"] - df["AWAY_ROLL_PLUS_MINUS_10"])

    df["REST_DIFF"] = (df["HOME_REST_DAYS"] - df["AWAY_REST_DAYS"])

    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    #remove NaN rpws
    return df.dropna().copy()

def save_datasets(df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents = True, exist_ok = True)

    train_df = df[df["SEASON"] != "2026"]
    train_df.to_csv(TRAIN_OUTPUT_PATH, index = False)

    print(f"Saved model dataset to: {TRAIN_OUTPUT_PATH}")

    test_df = df[df["SEASON"] == "2026"]
    test_df.to_csv(TEST_OUTPUT_PATH, index = False)
    print(f"Saved model dataset to: {TEST_OUTPUT_PATH}")


def main():
    df = load_raw_data(SEASONS)

    df = create_basic_columns(df)
    df = create_rolling_features(df)
    df = create_game_level_dataset(df)
    df = create_model_features(df)
    df = clean_dataset(df)

    save_datasets(df)


if __name__ == "__main__":
    main()

