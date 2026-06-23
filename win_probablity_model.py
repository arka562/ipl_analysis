"""
Win Probability Model
Predicts batting-team win probability at over-by-over checkpoints.
Model: RandomForestClassifier
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42


def build_checkpoint_frame(deliveries_path, innings_path):
    deliveries = pd.read_csv(deliveries_path)
    innings    = pd.read_csv(innings_path)

    completed = innings[innings["winner"].notna() & (innings["winner"] != "")].copy()

    innings_scores = completed[["match_id", "innings", "runs"]].rename(
        columns={"runs": "final_innings_score"}
    )

    first_inn_scores = (
        completed[completed["innings"] == 1][["match_id", "runs"]]
        .copy()
        .rename(columns={"runs": "target_minus_one"})
    )
    first_inn_scores["target"] = first_inn_scores["target_minus_one"] + 1

    rows = []
    grouped = deliveries[
        deliveries["winner"].notna() & (deliveries["winner"] != "")
    ].groupby(
        ["match_id", "season", "innings", "batting_team", "winner"],
        sort=False,
    )

    for (match_id, season, innings_no, batting_team, winner), group in grouped:
        group = group.sort_values(["over", "ball_in_over"])

        cum_runs    = 0
        cum_wickets = 0
        cum_balls   = 0
        ov_runs     = 0
        ov_wickets  = 0
        ov_balls    = 0

        for _, ball in group.iterrows():
            cum_runs    += int(ball["total_runs"])
            cum_wickets += int(ball["wicket_count"])
            cum_balls   += int(ball["is_legal_ball"])
            ov_runs     += int(ball["total_runs"])
            ov_wickets  += int(ball["wicket_count"])
            ov_balls    += int(ball["is_legal_ball"])

            # Snapshot at end of each complete over
            if cum_balls > 0 and cum_balls % 6 == 0:
                over_number = cum_balls // 6
                rows.append(
                    {
                        "match_id":         match_id,
                        "season":           int(season),
                        "innings":          int(innings_no),
                        "batting_team":     batting_team,
                        "over_number":      over_number,
                        "current_score":    cum_runs,
                        "wickets_lost":     cum_wickets,
                        "balls_bowled":     cum_balls,
                        "overs_remaining":  max(0, 20 - over_number),
                        "current_run_rate": cum_runs * 6 / cum_balls,
                        "last_over_runs":   ov_runs,
                        "last_over_wickets": ov_wickets,
                        "batting_team_won": int(batting_team == winner),
                    }
                )
                ov_runs = ov_wickets = ov_balls = 0

    frame = pd.DataFrame(rows)
    frame = frame.merge(innings_scores, on=["match_id", "innings"], how="left")
    frame = frame.merge(first_inn_scores[["match_id", "target"]], on="match_id", how="left")

    frame["target"]       = frame["target"].where(frame["innings"] == 2, np.nan)
    frame["runs_required"] = frame["target"] - frame["current_score"]

    balls_remaining = (120 - frame["balls_bowled"]).replace(0, np.nan)
    frame["required_run_rate"] = frame["runs_required"] * 6 / balls_remaining

    frame["score_pct_of_final"] = frame["current_score"] / frame["final_innings_score"].replace(0, np.nan)
    frame["is_chase"] = (frame["innings"] == 2).astype(int)
    frame = frame[frame["over_number"].between(1, 20)].copy()
    return frame


def train_model(frame):
    feature_cols = [
        "season", "innings", "batting_team",
        "over_number", "current_score", "wickets_lost", "balls_bowled",
        "overs_remaining", "current_run_rate",
        "last_over_runs", "last_over_wickets",
        "target", "runs_required", "required_run_rate", "is_chase",
    ]
    target_col = "batting_team_won"

    train_df, test_df = train_test_split(
        frame, test_size=0.2, random_state=RANDOM_STATE, stratify=frame[target_col]
    )

    numeric_features     = [c for c in feature_cols if c != "batting_team"]
    categorical_features = ["batting_team"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(train_df[feature_cols], train_df[target_col])
    probabilities = model.predict_proba(test_df[feature_cols])[:, 1]
    predictions   = (probabilities >= 0.5).astype(int)

    metrics = {
        "model_type":     "RandomForestClassifier",
        "prediction_task": "Predict batting team win probability at over checkpoints",
        "rows_total":  int(len(frame)),
        "rows_train":  int(len(train_df)),
        "rows_test":   int(len(test_df)),
        "accuracy":  round(accuracy_score(test_df[target_col], predictions), 3),
        "roc_auc":   round(roc_auc_score(test_df[target_col], probabilities), 3),
        "log_loss":  round(log_loss(test_df[target_col], probabilities), 3),
    }

    sample = test_df[
        [
            "match_id", "season", "innings", "batting_team",
            "over_number", "current_score", "wickets_lost",
            "target", "runs_required", "batting_team_won",
        ]
    ].copy()
    sample["win_probability"] = (probabilities * 100).round(1)
    sample = sample.sort_values(["match_id", "innings", "over_number"]).head(50)

    return model, metrics, sample, feature_cols

