from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

PROJECT_ROOT = Path(r"C:\Users\davhu\Documents\wnba_win_probability")

DATA_PATH = (PROJECT_ROOT/ "data" / "processed" / "model_dataset_2026.csv")

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "win_probability_model.joblib"

PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
PREDICTIONS_PATH = PREDICTIONS_DIR / "2026_predictions.csv"

FEATURES = [
    "WIN_PCT_DIFF",
    "WIN_PCT_DIFF_10",
    "PTS_DIFF",
    "PTS_DIFF_10",
    "PLUS_MINUS_DIFF",
    "PLUS_MINUS_DIFF_10",
    "REST_DIFF"
]

def load_dataset() -> pd.DataFrame():
    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

def evaluate_model(df : pd.DataFrame) -> None:
    df = df.copy()
    if df.empty:
        raise ValueError("No 2026 games found.")

    # Sort test games before creating X_test and results.
    # This keeps every output column aligned.
    test_df = df.sort_values(by=["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    X_test = test_df[FEATURES] #features
    y_test = test_df["HOME_WIN"] #results

    model = joblib.load(MODEL_PATH)

    predicted_home_win = model.predict(X_test)
    predicted_home_win_probability = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, predicted_home_win))
    print("ROC-AUC:", roc_auc_score(y_test, predicted_home_win_probability))
    print("Log loss:", log_loss(y_test, predicted_home_win_probability))

    # Build results directly from the sorted test_df.
    results = test_df[[
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM",
        "AWAY_TEAM",
        "HOME_WIN",
        "HOME_SEASON_WINS",
        "HOME_SEASON_LOSSES",
        "AWAY_SEASON_WINS",
        "AWAY_SEASON_LOSSES"
    ]].copy()

    results["PREDICTED_HOME_WIN"] = predicted_home_win
    results["PREDICTED_HOME_WIN_PROBABILITY"] = predicted_home_win_probability

    results["ACTUAL_WINNER"] = results.apply(
        lambda row: row["HOME_TEAM"] if row["HOME_WIN"] == 1 else row["AWAY_TEAM"],
        axis=1
    )

    results["PREDICTED_WINNER"] = results.apply(
        lambda row: row["HOME_TEAM"] if row["PREDICTED_HOME_WIN"] == 1 else row["AWAY_TEAM"],
        axis=1
    )

    # Convert probability to a readable percentage.
    results["PREDICTED_HOME_WIN_PROBABILITY"] = (
        results["PREDICTED_HOME_WIN_PROBABILITY"] * 100
    ).round(2)

    # records (W-/L suffix avoids Excel auto-converting "1-2" into a date)
    results["HOME_RECORD"] = (
        results["HOME_SEASON_WINS"].astype(int).astype(str) + "W-"
        + results["HOME_SEASON_LOSSES"].astype(int).astype(str) + "L"
    )

    results["AWAY_RECORD"] = (
        results["AWAY_SEASON_WINS"].astype(int).astype(str) + "W-"
        + results["AWAY_SEASON_LOSSES"].astype(int).astype(str) + "L"
    )

    # Keep only the useful output columns.
    results = results[[
        "GAME_DATE",
        "HOME_TEAM",
        "HOME_RECORD",
        "AWAY_TEAM",
        "AWAY_RECORD",
        "PREDICTED_HOME_WIN_PROBABILITY",
        "PREDICTED_WINNER",
        "ACTUAL_WINNER"
    ]]

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Saved predictions to: {PREDICTIONS_PATH}")

def main():
    df = load_dataset()
    evaluate_model(df)


if __name__ == "__main__":
    main()
