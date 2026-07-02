from pathlib import Path
import pandas as pd
from collect_data import fetch_season_game_logs, save_raw_data
from build_dataset import (
    PROJECT_ROOT,
    load_raw_data,
    create_basic_columns,
    create_opponent_columns,
    create_rolling_features,
    create_game_level_dataset,
    create_model_features,
    clean_dataset
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DATA_DIR / "model_dataset_2026.csv"

SEASON = "2026"

def save_2026_dataset(df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved 2026 dataset to: {OUTPUT_PATH}")


def main():
    df = fetch_season_game_logs("2026")
    save_raw_data(df, "2026")

    df = load_raw_data([SEASON])
    df = create_basic_columns(df)
    df = create_opponent_columns(df)
    df = create_rolling_features(df)
    df = create_game_level_dataset(df)
    df = create_model_features(df)
    df = clean_dataset(df)

    save_2026_dataset(df)


if __name__ == "__main__":
    main()