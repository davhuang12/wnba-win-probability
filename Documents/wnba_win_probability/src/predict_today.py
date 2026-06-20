from datetime import date

import joblib
import pandas as pd

from nba_api.stats.endpoints import scoreboardv3

from build_dataset import PROJECT_ROOT

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DATA_DIR / "model_dataset_2026.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "win_probability_model.joblib"

WNBA_LEAGUE_ID = "10"

FEATURES = [
    "WIN_PCT_DIFF",
    "WIN_PCT_DIFF_10",
    "PTS_DIFF",
    "PTS_DIFF_10",
    "PLUS_MINUS_DIFF",
    "PLUS_MINUS_DIFF_10",
    "REST_DIFF",
]


def fetch_todays_games(game_date: str) -> pd.DataFrame:
    """
    Pulls today's WNBA schedule (home/away team IDs) from the
    ScoreboardV3 endpoint.

    ScoreboardV3's game_header dataset has no team ID columns at all -
    only gameId/gameCode/status fields. Team IDs live in line_score,
    with two rows per gameId (one per team) and no explicit home/away
    flag. To recover home/away, we parse gameCode, which encodes the
    matchup as "YYYYMMDD/AWAYHOME" (e.g. "20260617/WASCON" = Washington
    @ Connecticut), and match those tricodes against line_score's
    teamTricode column.

    game_date format: "YYYY-MM-DD"
    """

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

        # gameCode looks like "20260617/WASCON" -> tricodes are the
        # last 6 characters, first 3 = away, last 3 = home.
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
    recent pre-game stats (whichever side - home or away - their last
    played game was on) plus the date of that last game.
    """

    df = pd.read_csv(DATASET_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    home_rows = df[[
        "GAME_DATE", "HOME_TEAM_ID", "HOME_TEAM",
        "HOME_WIN_PCT", "HOME_PTS_AVG", "HOME_PLUS_MINUS_AVG",
        "HOME_ROLL_WIN_PCT_10", "HOME_ROLL_PTS_10", "HOME_ROLL_PLUS_MINUS_10",
    ]].rename(columns={
        "HOME_TEAM_ID": "TEAM_ID",
        "HOME_TEAM": "TEAM_NAME",
        "HOME_WIN_PCT": "WIN_PCT",
        "HOME_PTS_AVG": "PTS_AVG",
        "HOME_PLUS_MINUS_AVG": "PLUS_MINUS_AVG",
        "HOME_ROLL_WIN_PCT_10": "ROLL_WIN_PCT_10",
        "HOME_ROLL_PTS_10": "ROLL_PTS_10",
        "HOME_ROLL_PLUS_MINUS_10": "ROLL_PLUS_MINUS_10",
    })

    away_rows = df[[
        "GAME_DATE", "AWAY_TEAM_ID", "AWAY_TEAM",
        "AWAY_WIN_PCT", "AWAY_PTS_AVG", "AWAY_PLUS_MINUS_AVG",
        "AWAY_ROLL_WIN_PCT_10", "AWAY_ROLL_PTS_10", "AWAY_ROLL_PLUS_MINUS_10",
    ]].rename(columns={
        "AWAY_TEAM_ID": "TEAM_ID",
        "AWAY_TEAM": "TEAM_NAME",
        "AWAY_WIN_PCT": "WIN_PCT",
        "AWAY_PTS_AVG": "PTS_AVG",
        "AWAY_PLUS_MINUS_AVG": "PLUS_MINUS_AVG",
        "AWAY_ROLL_WIN_PCT_10": "ROLL_WIN_PCT_10",
        "AWAY_ROLL_PTS_10": "ROLL_PTS_10",
        "AWAY_ROLL_PLUS_MINUS_10": "ROLL_PLUS_MINUS_10",
    })

    all_team_rows = pd.concat([home_rows, away_rows], ignore_index=True)

    # Keep only each team's single most recent game.
    all_team_rows = all_team_rows.sort_values("GAME_DATE")
    latest_per_team = all_team_rows.groupby("TEAM_ID").tail(1).set_index("TEAM_ID")

    return latest_per_team


def build_features_for_game(
    home_team_id: int,
    away_team_id: int,
    team_stats: pd.DataFrame,
    game_date: pd.Timestamp,
) -> dict:
    """
    Builds the 7 model features for one upcoming game, using each team's
    most recent pre-game stats. Returns None if either team has no
    history yet in model_dataset_2026.csv (e.g. very first games of the
    season, or a TEAM_ID mismatch).
    """

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
        "HOME_TEAM": home["TEAM_NAME"],
        "AWAY_TEAM": away["TEAM_NAME"],
    }


def predict_games(games_df: pd.DataFrame, game_date: pd.Timestamp) -> pd.DataFrame:
    team_stats = load_latest_team_stats()
    model = joblib.load(MODEL_PATH)

    results = []

    for _, game in games_df.iterrows():
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        feature_row = build_features_for_game(home_id, away_id, team_stats, game_date)

        if feature_row is None:
            print(f"Skipping game {home_id} vs {away_id}: no prior 2026 stats found.")
            continue

        X = pd.DataFrame([{k: feature_row[k] for k in FEATURES}])
        home_win_prob = model.predict_proba(X)[0][1]

        results.append({
            "HOME_TEAM": feature_row["HOME_TEAM"],
            "AWAY_TEAM": feature_row["AWAY_TEAM"],
            "HOME_WIN_PROBABILITY": round(home_win_prob, 4),
            "AWAY_WIN_PROBABILITY": round(1 - home_win_prob, 4),
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