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
        file_path = PROJECT_ROOT / "data" / "raw" / f"game_logs_{season}.csv"

        df = pd.read_csv(file_path)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])  # turn it into a date, so we can do math
        df["SEASON"] = season

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def create_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert W/L text into numbers.
    df["WIN"] = df["WL"].map({
        "W": 1,
        "L": 0
    })

    # vs. : home, @ : away
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.")

    # possessions for offensive/defense rating
    df["POSS"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]

    return df


def create_opponent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attaches each row's opponent PTS and POSS (from the same GAME_ID).
    This does NOT compute a rating here — ratings are a rate stat, so we need
    to sum PTS and POSS across a window of games first (in create_rolling_features)
    and divide once, rather than averaging one-game rates together.
    """
    df = df.copy()

    opp = df[["GAME_ID", "TEAM_ID", "PTS", "POSS"]].rename(columns={
        "TEAM_ID": "OPP_TEAM_ID",
        "PTS": "OPP_PTS",
        "POSS": "OPP_POSS"
    })

    df = df.merge(opp, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]].copy()  # drop self-matches from the join

    # shared possession estimate for the game (reduces noise vs. using one team's POSS alone)
    df["GAME_POSS"] = (df["POSS"] + df["OPP_POSS"]) / 2

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates pre-game rolling statistics. This runs on every row (every game, both teams)
    stats tracked : win %, pts average, +/-, rest days, offensive/defensive/net rating
    """

    df = df.copy()

    df = df.sort_values(["SEASON", "TEAM_ID", "GAME_DATE"])  # sort by team, then by game # chronologically
    group_cols = ["SEASON", "TEAM_ID"]  # what we have to consider together later

    df["SEASON_GAMES_PLAYED"] = (df.groupby(group_cols).cumcount())
    df["SEASON_WINS"] = (df.groupby(group_cols)["WIN"].transform(lambda x: x.shift(1).fillna(0).cumsum()))
    df["SEASON_LOSSES"] = (df["SEASON_GAMES_PLAYED"] - df["SEASON_WINS"])

    # season total features
    df["SEASON_WIN_PCT"] = (
        df.groupby(group_cols)["WIN"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())  # expanding -> loop through every game
    )

    df["SEASON_PTS_AVG"] = (
        df.groupby(group_cols)["PTS"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
    )

    df["SEASON_PLUS_MINUS_AVG"] = (
        df.groupby(group_cols)["PLUS_MINUS"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
    )

    # Season OREB rate: cumulative OREB / cumulative possessions (pre-game)
    df["SEASON_OREB"] = df.groupby(group_cols)["OREB"].transform(
        lambda x: x.shift(1).expanding(min_periods=3).sum()
    )
    df["SEASON_POSS"] = df.groupby(group_cols)["GAME_POSS"].transform(
        lambda x: x.shift(1).expanding(min_periods=3).sum()
    )
    df["SEASON_OREB_RATE"] = df["SEASON_OREB"] / df["SEASON_POSS"]

    # Rolling 10-game OREB rate
    df["ROLL_OREB_10"] = df.groupby(group_cols)["OREB"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).sum()
    )
    df["ROLL_POSS_10"] = df.groupby(group_cols)["GAME_POSS"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).sum()
    )
    df["ROLL_OREB_RATE_10"] = df["ROLL_OREB_10"] / df["ROLL_POSS_10"]

    # season offensive / defensive / net rating features
    # (sum of points over sum of possessions across the window — NOT an average of per-game rates)
    season_pts_sum = (
        df.groupby(group_cols)["PTS"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).sum())
    )
    season_opp_pts_sum = (
        df.groupby(group_cols)["OPP_PTS"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).sum())
    )
    season_poss_sum = (
        df.groupby(group_cols)["GAME_POSS"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).sum())
    )

    df["SEASON_ORTG_AVG"] = 100 * season_pts_sum / season_poss_sum
    df["SEASON_DRTG_AVG"] = 100 * season_opp_pts_sum / season_poss_sum
    df["SEASON_NET_RTG_AVG"] = df["SEASON_ORTG_AVG"] - df["SEASON_DRTG_AVG"]

    # rolling features
    df["ROLL_WIN_PCT_10"] = (
        df.groupby(group_cols)["WIN"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())  # rolling -> loop through last x games
    )

    df["ROLL_PTS_10"] = (
        df.groupby(group_cols)["PTS"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

    df["ROLL_PLUS_MINUS_10"] = (
        df.groupby(group_cols)["PLUS_MINUS"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

    # rolling offensive / defensive / net rating features (last 10 games)
    # (sum of points over sum of possessions across the window — NOT an average of per-game rates)
    roll_pts_sum = (
        df.groupby(group_cols)["PTS"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).sum())
    )
    roll_opp_pts_sum = (
        df.groupby(group_cols)["OPP_PTS"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).sum())
    )
    roll_poss_sum = (
        df.groupby(group_cols)["GAME_POSS"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).sum())
    )

    df["ROLL_ORTG_10"] = 100 * roll_pts_sum / roll_poss_sum
    df["ROLL_DRTG_10"] = 100 * roll_opp_pts_sum / roll_poss_sum
    df["ROLL_NET_RTG_10"] = df["ROLL_ORTG_10"] - df["ROLL_DRTG_10"]

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
        "SEASON_LOSSES",
        "SEASON_ORTG_AVG",
        "SEASON_DRTG_AVG",
        "SEASON_NET_RTG_AVG",
        "ROLL_ORTG_10",
        "ROLL_DRTG_10",
        "ROLL_NET_RTG_10",
        "SEASON_OREB_RATE",
        "ROLL_OREB_RATE_10"
    ]

    # only keep these columns
    home = home[columns_to_keep]
    away = away[columns_to_keep]

    home = home.rename(columns={
        "TEAM_ID": "HOME_TEAM_ID",
        "TEAM_NAME": "HOME_TEAM",
        "WIN": "HOME_WIN",
        "PTS": "HOME_PTS",
        "SEASON_WIN_PCT": "HOME_WIN_PCT",
        "SEASON_PTS_AVG": "HOME_PTS_AVG",
        "SEASON_PLUS_MINUS_AVG": "HOME_PLUS_MINUS_AVG",
        "ROLL_WIN_PCT_10": "HOME_ROLL_WIN_PCT_10",
        "ROLL_PTS_10": "HOME_ROLL_PTS_10",
        "ROLL_PLUS_MINUS_10": "HOME_ROLL_PLUS_MINUS_10",
        "REST_DAYS": "HOME_REST_DAYS",
        "SEASON_WINS": "HOME_SEASON_WINS",
        "SEASON_LOSSES": "HOME_SEASON_LOSSES",
        "SEASON_ORTG_AVG": "HOME_ORTG_AVG",
        "SEASON_DRTG_AVG": "HOME_DRTG_AVG",
        "SEASON_NET_RTG_AVG": "HOME_NET_RTG_AVG",
        "ROLL_ORTG_10": "HOME_ROLL_ORTG_10",
        "ROLL_DRTG_10": "HOME_ROLL_DRTG_10",
        "ROLL_NET_RTG_10": "HOME_ROLL_NET_RTG_10",
        "SEASON_OREB_RATE" : "HOME_SEASON_OREB_RATE",
        "ROLL_OREB_RATE_10": "HOME_ROLL_OREB_RATE_10"
    })

    away = away.rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID",
        "TEAM_NAME": "AWAY_TEAM",
        "WIN": "AWAY_WIN",
        "PTS": "AWAY_PTS",
        "SEASON_WIN_PCT": "AWAY_WIN_PCT",
        "SEASON_PTS_AVG": "AWAY_PTS_AVG",
        "SEASON_PLUS_MINUS_AVG": "AWAY_PLUS_MINUS_AVG",
        "ROLL_WIN_PCT_10": "AWAY_ROLL_WIN_PCT_10",
        "ROLL_PTS_10": "AWAY_ROLL_PTS_10",
        "ROLL_PLUS_MINUS_10": "AWAY_ROLL_PLUS_MINUS_10",
        "REST_DAYS": "AWAY_REST_DAYS",
        "SEASON_WINS": "AWAY_SEASON_WINS",
        "SEASON_LOSSES": "AWAY_SEASON_LOSSES",
        "SEASON_ORTG_AVG": "AWAY_ORTG_AVG",
        "SEASON_DRTG_AVG": "AWAY_DRTG_AVG",
        "SEASON_NET_RTG_AVG": "AWAY_NET_RTG_AVG",
        "ROLL_ORTG_10": "AWAY_ROLL_ORTG_10",
        "ROLL_DRTG_10": "AWAY_ROLL_DRTG_10",
        "ROLL_NET_RTG_10": "AWAY_ROLL_NET_RTG_10",
        "SEASON_OREB_RATE": "AWAY_SEASON_OREB_RATE",  # in away rename
        "ROLL_OREB_RATE_10": "AWAY_ROLL_OREB_RATE_10",
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

    # offensive / defensive / net rating diffs
    df["ORTG_DIFF"] = (df["HOME_ORTG_AVG"] - df["AWAY_ORTG_AVG"])
    df["ORTG_DIFF_10"] = (df["HOME_ROLL_ORTG_10"] - df["AWAY_ROLL_ORTG_10"])

    df["DRTG_DIFF"] = (df["HOME_DRTG_AVG"] - df["AWAY_DRTG_AVG"])
    df["DRTG_DIFF_10"] = (df["HOME_ROLL_DRTG_10"] - df["AWAY_ROLL_DRTG_10"])

    df["NET_RTG_DIFF"] = (df["HOME_NET_RTG_AVG"] - df["AWAY_NET_RTG_AVG"])
    df["NET_RTG_DIFF_10"] = (df["HOME_ROLL_NET_RTG_10"] - df["AWAY_ROLL_NET_RTG_10"])

    df["OREB_RATE_DIFF"] = df["HOME_SEASON_OREB_RATE"] - df["AWAY_SEASON_OREB_RATE"]
    df["OREB_RATE_DIFF_10"] = df["HOME_ROLL_OREB_RATE_10"] - df["AWAY_ROLL_OREB_RATE_10"]

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # remove NaN rpws
    return df.dropna().copy()


def save_datasets(df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df = df[df["SEASON"] != "2026"]
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)

    print(f"Saved model dataset to: {TRAIN_OUTPUT_PATH}")

    test_df = df[df["SEASON"] == "2026"]
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)
    print(f"Saved model dataset to: {TEST_OUTPUT_PATH}")


def main():
    df = load_raw_data(SEASONS)

    df = create_basic_columns(df)
    df = create_opponent_columns(df)
    df = create_rolling_features(df)
    df = create_game_level_dataset(df)
    df = create_model_features(df)
    df = clean_dataset(df)

    save_datasets(df)


if __name__ == "__main__":
    main()