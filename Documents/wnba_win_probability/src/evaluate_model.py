from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    mean_absolute_error,
    root_mean_squared_error,
)

PROJECT_ROOT = Path(r"C:\Users\davhu\Documents\wnba_win_probability")

DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "model_dataset_2026.csv")

MODELS_DIR = PROJECT_ROOT / "models"
WIN_MODEL_PATH = MODELS_DIR / "win_probability_model.joblib"
SPREAD_MODEL_PATH = MODELS_DIR / "spread_model.joblib"

PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
PREDICTIONS_PATH = PREDICTIONS_DIR / "2026_predictions.csv"

FEATURES = [
    "WIN_PCT_DIFF", "WIN_PCT_DIFF_10",
    "PTS_DIFF", "PTS_DIFF_10",
    "PLUS_MINUS_DIFF","PLUS_MINUS_DIFF_10",
    "REST_DIFF",
    "ORTG_DIFF", "ORTG_DIFF_10",
    "DRTG_DIFF", "DRTG_DIFF_10",
    "OREB_RATE_DIFF", "OREB_RATE_DIFF_10",
    "FG3M_DIFF", "FG3M_DIFF_10",
    "TOV_RATE_DIFF","TOV_RATE_DIFF_10",
    "EFG_DIFF", "EFG_DIFF_10"
]

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def evaluate_win_model(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates the win probability classifier and returns predicted
    win/loss + probability columns aligned to test_df's row order.
    """

    X_test = test_df[FEATURES]
    y_test = test_df["HOME_WIN"]

    model = joblib.load(WIN_MODEL_PATH)

    predicted_home_win = model.predict(X_test)
    predicted_home_win_probability = model.predict_proba(X_test)[:, 1]

    print("--- Win Probability Model ---")
    print("Accuracy:", accuracy_score(y_test, predicted_home_win))
    print("ROC-AUC:", roc_auc_score(y_test, predicted_home_win_probability))
    print("Log loss:", log_loss(y_test, predicted_home_win_probability))
    print()

    return pd.DataFrame({
        "PREDICTED_HOME_WIN": predicted_home_win,
        "PREDICTED_HOME_WIN_PROBABILITY": predicted_home_win_probability,
    })


def evaluate_spread_model(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates the spread (margin) regressor and returns predicted
    and actual margin columns aligned to test_df's row order.
    """

    X_test = test_df[FEATURES]
    y_test = test_df["HOME_PTS"] - test_df["AWAY_PTS"]

    model = joblib.load(SPREAD_MODEL_PATH)

    predicted_margin = model.predict(X_test)

    mae = mean_absolute_error(y_test, predicted_margin)
    rmse = root_mean_squared_error(y_test, predicted_margin)

    print("--- Spread Model ---")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print()

    return pd.DataFrame({
        "ACTUAL_MARGIN": y_test.values,
        "PREDICTED_MARGIN": predicted_margin.round(1),
    })


def build_results(test_df: pd.DataFrame, win_preds: pd.DataFrame, spread_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Combines the base game info with both models' predictions into one output table.
    """

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

    results = pd.concat([results, win_preds, spread_preds], axis=1)

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
        "ACTUAL_WINNER",
        "PREDICTED_MARGIN",
        "ACTUAL_MARGIN",
    ]]

    return results


def main():
    df = load_dataset()

    if df.empty:
        raise ValueError("No 2026 games found.")

    # Sort test games before creating X_test and results.
    # This keeps every output column aligned.
    test_df = df.sort_values(by=["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    win_preds = evaluate_win_model(test_df)
    spread_preds = evaluate_spread_model(test_df)

    results = build_results(test_df, win_preds, spread_preds)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Saved predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()