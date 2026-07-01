"""
Win Probability Model
Predicts batting-team win probability at over-by-over checkpoints.
Model: RandomForestClassifier
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from shared.artifacts import create_model_dirs, save_model_artifacts
from shared.evaluation import evaluate_classification
from shared.preprocessing import build_preprocessor


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

    preprocessor = build_preprocessor(numeric_features, categorical_features)

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

    metrics = {
        "model_type":     "RandomForestClassifier",
        "prediction_task": "Predict batting team win probability at over checkpoints",
        "rows_total":  int(len(frame)),
        "rows_train":  int(len(train_df)),
        "rows_test":   int(len(test_df)),
        **evaluate_classification(test_df[target_col], probabilities),
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

def main():
    parser = argparse.ArgumentParser(description="Train IPL win probability model.")
    parser.add_argument("--processed-dir", default="ipl_analytics_platform/data/processed")
    parser.add_argument("--model-dir",     default="ipl_analytics_platform/models")
    parser.add_argument("--reports-dir",   default="ipl_analytics_platform/reports/modeling")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    model_dir, reports_dir = create_model_dirs(args.model_dir, args.reports_dir)

    print("Building checkpoint frame...")
    frame = build_checkpoint_frame(
        processed_dir / "deliveries.csv",
        processed_dir / "innings.csv",
    )

    print("Training model...")
    model, metrics, sample, feature_cols = train_model(frame)

    summary = [
        "# Win Probability Model",
        "",
        "## Task",
        "",
        "Predict whether the batting team will win from over-by-over match checkpoints.",
        "",
        "## Metrics",
        "",
        f"- Rows used  : {metrics['rows_total']:,}",
        f"- Train rows : {metrics['rows_train']:,}",
        f"- Test rows  : {metrics['rows_test']:,}",
        f"- Accuracy   : {metrics['accuracy']}",
        f"- ROC AUC    : {metrics['roc_auc']}",
        f"- Log loss   : {metrics['log_loss']}",
        "",
        "## Features",
        "",
    ]
    summary.extend(f"- `{f}`" for f in feature_cols)
    summary.extend(
        [
            "",
            "## Interpretation",
            "",
            "This model powers live-style win probability curves. It shows how a team's "
            "chances evolved over each over based on score, wickets, innings context, "
            "target pressure, and recent scoring.",
        ]
    )

    save_model_artifacts(
        model=model,
        model_path=model_dir / "win_probability_model.joblib",
        training_frame=frame,
        training_data_path=reports_dir / "win_probability_training_data.csv",
        sample_df=sample,
        sample_path=reports_dir / "win_probability_sample_predictions.csv",
        metrics=metrics,
        metrics_path=reports_dir / "win_probability_metrics.json",
        feature_cols=feature_cols,
        summary_lines=summary,
        summary_path=reports_dir / "win_probability_summary.md",
    )

    print(f"Model saved : {(model_dir / 'win_probability_model.joblib').resolve()}")
    print(f"Accuracy    : {metrics['accuracy']}")
    print(f"ROC AUC     : {metrics['roc_auc']}")
    print(f"Log loss    : {metrics['log_loss']}")


if __name__ == "__main__":
    main()
