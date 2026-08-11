from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(r"C:\Users\davhu\Documents\wnba_win_probability")

DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "model_dataset.csv")

MODELS_DIR = PROJECT_ROOT / "models"
WIN_MODEL_PATH = MODELS_DIR / "win_probability_model.joblib"
SPREAD_MODEL_PATH = MODELS_DIR / "spread_model.joblib"

FEATURES = [
    "WIN_PCT_DIFF", "WIN_PCT_DIFF_10",
    "PTS_DIFF", "PTS_DIFF_10",
    "PLUS_MINUS_DIFF","PLUS_MINUS_DIFF_10",
    "REST_DIFF",
    "ORTG_DIFF", "ORTG_DIFF_10",
    "DRTG_DIFF", "DRTG_DIFF_10",
    "OREB_RATE_DIFF", "OREB_RATE_DIFF_10",
    "FG3M_DIFF", "FG3M_DIFF_10",
    "TOV_RATE_DIFF","TOV_RATE_DIFF_10"
]

def load_dataset() -> pd.DataFrame:
    """
    Load the processed game-level dataset.
    """

    df = pd.read_csv(DATA_PATH)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    return df


def train_wl_model(df: pd.DataFrame) -> None:
    train_df = df.copy()

    if train_df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    # Sort test games before creating X_test and results.
    # This keeps every output column aligned.

    X_train = train_df[FEATURES]
    y_train = train_df["HOME_WIN"]

    model = Pipeline([
        ("scaler",StandardScaler()),
        ("log_reg",LogisticRegression())
    ])

    model.fit(X_train,y_train)

    MODELS_DIR.mkdir(parents=True,exist_ok=True)

    joblib.dump(model,WIN_MODEL_PATH)

    print(
        f"Saved model to: {WIN_MODEL_PATH}"
    )


def train_spread_model(df: pd.DataFrame) -> None:
    train_df = df.copy()
    """
    Trains a regressor that predicts the home team's margin of victory
    (HOME_PTS - AWAY_PTS). Positive = home favored, negative = away favored.
    Uses the same pre-game features as the win probability model.
    """

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    train_df["HOME_MARGIN"] = train_df["HOME_PTS"] - train_df["AWAY_PTS"]

    X_train = train_df[FEATURES]
    y_train = train_df["HOME_MARGIN"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SPREAD_MODEL_PATH)

    print(f"Saved spread model to: {SPREAD_MODEL_PATH}")

def main() -> None:
    df = load_dataset()

    train_wl_model(df)
    train_spread_model(df)


if __name__ == "__main__":
    main()