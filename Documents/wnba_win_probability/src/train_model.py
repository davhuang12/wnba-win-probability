from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(r"C:\Users\davhu\Documents\wnba_win_probability")

DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "model_dataset.csv")

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "win_probability_model.joblib"

def load_dataset() -> pd.DataFrame:
    """
    Load the processed game-level dataset.
    """

    df = pd.read_csv(DATA_PATH)

    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"]
    )

    return df


def train_model(df: pd.DataFrame) -> None:

    features = [
        "WIN_PCT_DIFF",
        "WIN_PCT_DIFF_10",
        "PTS_DIFF",
        "PTS_DIFF_10",
        "PLUS_MINUS_DIFF",
        "PLUS_MINUS_DIFF_10",
        "REST_DIFF"
    ]

    #train on all but 2026
    train_df = df[
        df["SEASON"] != "2026"
    ].copy()

    if train_df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    # Sort test games before creating X_test and results.
    # This keeps every output column aligned.

    X_train = train_df[features]
    y_train = train_df["HOME_WIN"]

    model = Pipeline([
        ("scaler",StandardScaler()),
        ("log_reg",LogisticRegression())
    ])

    model.fit(
        X_train,
        y_train
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Saved model to: {MODEL_PATH}"
    )

def main() -> None:
    df = load_dataset()

    train_model(df)


if __name__ == "__main__":
    main()