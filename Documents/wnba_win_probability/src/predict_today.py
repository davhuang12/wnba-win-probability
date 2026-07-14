from datetime import date, timedelta

import joblib
import pandas as pd

from nba_api.stats.endpoints import scoreboardv3

from build_dataset import PROJECT_ROOT

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DATA_DIR / "model_dataset_2026.csv"

MODELS_DIR = PROJECT_ROOT / "models"
WIN_MODEL_PATH = MODELS_DIR / "win_probability_model.joblib"
SPREAD_MODEL_PATH = MODELS_DIR / "spread_model.joblib"

WNBA_LEAGUE_ID = "10"

FEATURES = [
    "WIN_PCT_DIFF",
    "WIN_PCT_DIFF_10",
    "PTS_DIFF",
    "PTS_DIFF_10",
    "PLUS_MINUS_DIFF",
    "PLUS_MINUS_DIFF_10",
    "REST_DIFF",
    "ORTG_DIFF", "ORTG_DIFF_10",
    "DRTG_DIFF", "DRTG_DIFF_10",
    "OREB_RATE_DIFF", "OREB_RATE_DIFF_10",
    "FG3M_DIFF", "FG3M_DIFF_10"
]


def fetch_todays_games(game_date: str) -> pd.DataFrame:

    print(f"Fetching WNBA schedule for {game_date}...")

    scoreboard = scoreboardv3.ScoreboardV3(
        game_date=game_date,
        league_id=WNBA_LEAGUE_ID,
    )

    game_header = scoreboard.game_header.get_data_frame()
    line_score = scoreboard.line_score.get_data_frame()

    if game_header.empty:
        print("No games found for this date.")
        return game_header

    print(f"Found {len(game_header)} game(s).")

    rows = []

    for _, game in game_header.iterrows():
        game_id = game["gameId"]

        matchup_code = game["gameCode"].split("/")[-1]
        away_tricode = matchup_code[:3]
        home_tricode = matchup_code[3:]

        teams_in_game = line_score[line_score["gameId"] == game_id]

        home_team = teams_in_game[teams_in_game["teamTricode"] == home_tricode]
        away_team = teams_in_game[teams_in_game["teamTricode"] == away_tricode]

        if home_team.empty or away_team.empty:
            print(f"Could not match tricodes for game {game_id} "
                  f"(code {matchup_code}); skipping.")
            continue

        rows.append({
            "gameId": game_id,
            "home_team_id": home_team.iloc[0]["teamId"],
            "away_team_id": away_team.iloc[0]["teamId"],
            "home_team_name": home_team.iloc[0]["teamCity"] + " " + home_team.iloc[0]["teamName"],
            "away_team_name": away_team.iloc[0]["teamCity"] + " " + away_team.iloc[0]["teamName"],
        })

    return pd.DataFrame(rows)


def load_latest_team_stats() -> pd.DataFrame:
    """
    Loads model_dataset_2026.csv and returns, for each team, their most
    recent pre-game stats plus the date of that last game.
    """

    df = pd.read_csv(DATASET_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    home_rows = df[[
        "GAME_DATE", "HOME_TEAM_ID", "HOME_TEAM",
        "HOME_WIN_PCT", "HOME_PTS_AVG", "HOME_PLUS_MINUS_AVG",
        "HOME_ROLL_WIN_PCT_10", "HOME_ROLL_PTS_10", "HOME_ROLL_PLUS_MINUS_10",
        "HOME_ORTG_AVG", "HOME_DRTG_AVG",
        "HOME_ROLL_ORTG_10", "HOME_ROLL_DRTG_10",
        "HOME_SEASON_OREB_RATE", "HOME_ROLL_OREB_RATE_10",
        "HOME_SEASON_FG3M_AVG", "HOME_ROLL_FG3M_10"
    ]].rename(columns={
        "HOME_TEAM_ID": "TEAM_ID",
        "HOME_TEAM": "TEAM_NAME",
        "HOME_WIN_PCT": "WIN_PCT",
        "HOME_PTS_AVG": "PTS_AVG",
        "HOME_PLUS_MINUS_AVG": "PLUS_MINUS_AVG",
        "HOME_ROLL_WIN_PCT_10": "ROLL_WIN_PCT_10",
        "HOME_ROLL_PTS_10": "ROLL_PTS_10",
        "HOME_ROLL_PLUS_MINUS_10": "ROLL_PLUS_MINUS_10",
        "HOME_ORTG_AVG": "ORTG_AVG",
        "HOME_DRTG_AVG": "DRTG_AVG",
        "HOME_ROLL_ORTG_10": "ROLL_ORTG_10",
        "HOME_ROLL_DRTG_10": "ROLL_DRTG_10",
        "HOME_SEASON_OREB_RATE": "SEASON_OREB_RATE",
        "HOME_ROLL_OREB_RATE_10": "ROLL_OREB_RATE_10",
        "HOME_SEASON_FG3M_AVG": "SEASON_FG3M_AVG",
        "HOME_ROLL_FG3M_10": "ROLL_FG3M_10"
    })

    away_rows = df[[
        "GAME_DATE", "AWAY_TEAM_ID", "AWAY_TEAM",
        "AWAY_WIN_PCT", "AWAY_PTS_AVG", "AWAY_PLUS_MINUS_AVG",
        "AWAY_ROLL_WIN_PCT_10", "AWAY_ROLL_PTS_10", "AWAY_ROLL_PLUS_MINUS_10",
        "AWAY_ORTG_AVG", "AWAY_DRTG_AVG",
        "AWAY_ROLL_ORTG_10", "AWAY_ROLL_DRTG_10",
        "AWAY_SEASON_OREB_RATE", "AWAY_ROLL_OREB_RATE_10",
        "AWAY_SEASON_FG3M_AVG", "AWAY_ROLL_FG3M_10"
    ]].rename(columns={
        "AWAY_TEAM_ID": "TEAM_ID",
        "AWAY_TEAM": "TEAM_NAME",
        "AWAY_WIN_PCT": "WIN_PCT",
        "AWAY_PTS_AVG": "PTS_AVG",
        "AWAY_PLUS_MINUS_AVG": "PLUS_MINUS_AVG",
        "AWAY_ROLL_WIN_PCT_10": "ROLL_WIN_PCT_10",
        "AWAY_ROLL_PTS_10": "ROLL_PTS_10",
        "AWAY_ROLL_PLUS_MINUS_10": "ROLL_PLUS_MINUS_10",
        "AWAY_ORTG_AVG": "ORTG_AVG",
        "AWAY_DRTG_AVG": "DRTG_AVG",
        "AWAY_ROLL_ORTG_10": "ROLL_ORTG_10",
        "AWAY_ROLL_DRTG_10": "ROLL_DRTG_10",
        "AWAY_SEASON_OREB_RATE": "SEASON_OREB_RATE",
        "AWAY_ROLL_OREB_RATE_10": "ROLL_OREB_RATE_10",
        "AWAY_SEASON_FG3M_AVG": "SEASON_FG3M_AVG",
        "AWAY_ROLL_FG3M_10": "ROLL_FG3M_10"
    })

    all_team_rows = pd.concat([home_rows, away_rows], ignore_index=True)

    all_team_rows = all_team_rows.sort_values("GAME_DATE")
    latest_per_team = all_team_rows.groupby("TEAM_ID").tail(1).set_index("TEAM_ID")

    return latest_per_team


def build_features_for_game(
    home_team_id: int,
    away_team_id: int,
    team_stats: pd.DataFrame,
    game_date: pd.Timestamp,
) -> dict:

    if home_team_id not in team_stats.index or away_team_id not in team_stats.index:
        return None

    home = team_stats.loc[home_team_id]
    away = team_stats.loc[away_team_id]

    home_rest_days = (game_date - home["GAME_DATE"]).days
    away_rest_days = (game_date - away["GAME_DATE"]).days

    return {
        "WIN_PCT_DIFF": home["WIN_PCT"] - away["WIN_PCT"],
        "WIN_PCT_DIFF_10": home["ROLL_WIN_PCT_10"] - away["ROLL_WIN_PCT_10"],
        "PTS_DIFF": home["PTS_AVG"] - away["PTS_AVG"],
        "PTS_DIFF_10": home["ROLL_PTS_10"] - away["ROLL_PTS_10"],
        "PLUS_MINUS_DIFF": home["PLUS_MINUS_AVG"] - away["PLUS_MINUS_AVG"],
        "PLUS_MINUS_DIFF_10": home["ROLL_PLUS_MINUS_10"] - away["ROLL_PLUS_MINUS_10"],
        "REST_DIFF": home_rest_days - away_rest_days,
        "ORTG_DIFF": home["ORTG_AVG"] - away["ORTG_AVG"],
        "ORTG_DIFF_10": home["ROLL_ORTG_10"] - away["ROLL_ORTG_10"],
        "DRTG_DIFF": home["DRTG_AVG"] - away["DRTG_AVG"],
        "DRTG_DIFF_10": home["ROLL_DRTG_10"] - away["ROLL_DRTG_10"],
        "OREB_RATE_DIFF": home["SEASON_OREB_RATE"] - away["SEASON_OREB_RATE"],
        "OREB_RATE_DIFF_10": home["ROLL_OREB_RATE_10"] - away["ROLL_OREB_RATE_10"],
        "FG3M_DIFF": home["SEASON_FG3M_AVG"] - away["SEASON_FG3M_AVG"],
        "FG3M_DIFF_10": home["ROLL_FG3M_10"] - away["ROLL_FG3M_10"],
        "HOME_TEAM": home["TEAM_NAME"],
        "AWAY_TEAM": away["TEAM_NAME"],
    }


def predict_games(games_df: pd.DataFrame, game_date: pd.Timestamp) -> pd.DataFrame:
    team_stats = load_latest_team_stats()
    win_model = joblib.load(WIN_MODEL_PATH)
    spread_model = joblib.load(SPREAD_MODEL_PATH)

    results = []

    for _, game in games_df.iterrows():
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        feature_row = build_features_for_game(home_id, away_id, team_stats, game_date)

        if feature_row is None:
            print(f"Skipping game {home_id} vs {away_id}: no prior 2026 stats found.")
            continue

        X = pd.DataFrame([{k: feature_row[k] for k in FEATURES}])

        home_win_prob = win_model.predict_proba(X)[0][1]
        predicted_margin = spread_model.predict(X)[0]

        if predicted_margin >= 0:
            spread_label = f"{feature_row['HOME_TEAM']} -{predicted_margin:.1f}"
        else:
            spread_label = f"{feature_row['AWAY_TEAM']} -{abs(predicted_margin):.1f}"

        results.append({
            "HOME_TEAM": feature_row["HOME_TEAM"],
            "AWAY_TEAM": feature_row["AWAY_TEAM"],
            "HOME_WIN_PROBABILITY": round(home_win_prob, 4),
            "AWAY_WIN_PROBABILITY": round(1 - home_win_prob, 4),
            "PREDICTED_SPREAD": spread_label,
        })

    return pd.DataFrame(results)


def main():
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    today_ts = pd.Timestamp(today)

    games_df = fetch_todays_games(today_str)

    if games_df.empty:
        return

    predictions = predict_games(games_df, today_ts)

    if predictions.empty:
        print("No predictions generated (likely missing prior-game stats for these teams).")
        return

    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()