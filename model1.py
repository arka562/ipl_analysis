"""
Score Prediction Model — v1
Predicts final innings score using first-6-over (powerplay) features.
Model: RandomForestRegressor
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from shared.artifacts import create_model_dirs, save_model_artifacts
from shared.evaluation import evaluate_regression, build_regression_sample
from shared.preprocessing import build_preprocessor


FEATURE_OVER_LIMIT = 6
RANDOM_STATE = 42


def build_training_frame(deliveries_path, innings_path):
    deliveries = pd.read_csv(deliveries_path)
    innings    = pd.read_csv(innings_path)

    early = deliveries[deliveries["over"] < FEATURE_OVER_LIMIT].copy()
    early["boundary"] = early["batter_runs"].isin([4, 6]).astype(int)
    early["dot_ball"] = (early["total_runs"] == 0).astype(int)

    features = (
        early.groupby(["match_id", "season", "innings", "batting_team"], as_index=False)
        .agg(
            powerplay_runs      =("total_runs",   "sum"),
            powerplay_wickets   =("wicket_count",  "sum"),
            powerplay_balls     =("is_legal_ball", "sum"),
            powerplay_boundaries=("boundary",      "sum"),
            powerplay_dot_balls =("dot_ball",      "sum"),
        )
    )

    features["powerplay_run_rate"]     = features["powerplay_runs"] * 6 / features["powerplay_balls"].replace(0, pd.NA)
    features["powerplay_wickets_left"] = 10 - features["powerplay_wickets"]
    features["boundary_pct"]           = 100 * features["powerplay_boundaries"] / features["powerplay_balls"].replace(0, pd.NA)
    features["dot_ball_pct"]           = 100 * features["powerplay_dot_balls"]  / features["powerplay_balls"].replace(0, pd.NA)
    features["is_first_innings"]       = (features["innings"] == 1).astype(int)

    target = innings[["match_id", "innings", "runs"]].rename(columns={"runs": "final_score"})
    frame  = features.merge(target, on=["match_id", "innings"], how="inner")
    frame  = frame[frame["powerplay_balls"] >= 30].copy()
    frame  = frame.dropna(subset=["final_score"])
    return frame


def train_model(frame):
    feature_cols = [
        "season", "innings", "batting_team",
        "powerplay_runs", "powerplay_wickets", "powerplay_balls",
        "powerplay_boundaries", "powerplay_dot_balls",
        "powerplay_run_rate", "powerplay_wickets_left",
        "boundary_pct", "dot_ball_pct", "is_first_innings",
    ]
    target_col = "final_score"

    train_df, test_df = train_test_split(
        frame, test_size=0.2, random_state=RANDOM_STATE, stratify=frame["innings"]
    )

    numeric_features     = [c for c in feature_cols if c != "batting_team"]
    categorical_features = ["batting_team"]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(train_df[feature_cols], train_df[target_col])
    predictions = model.predict(test_df[feature_cols])

    metrics = {
        "model_type":       "RandomForestRegressor",
        "prediction_task":  "Predict final innings score using first 6 overs",
        "feature_over_limit": FEATURE_OVER_LIMIT,
        "rows_total":  int(len(frame)),
        "rows_train":  int(len(train_df)),
        "rows_test":   int(len(test_df)),
        **evaluate_regression(test_df[target_col], predictions),
    }

    sample = build_regression_sample(
        test_df, predictions,
        id_cols=["match_id", "season", "innings", "batting_team",
                 "powerplay_runs", "powerplay_wickets"],
        target_col=target_col,
    )

    return model, metrics, sample, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train IPL score prediction model v1.")
    parser.add_argument("--processed-dir", default="ipl_analytics_platform/data/processed")
    parser.add_argument("--model-dir",     default="ipl_analytics_platform/models")
    parser.add_argument("--reports-dir",   default="ipl_analytics_platform/reports/modeling")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    model_dir, reports_dir = create_model_dirs(args.model_dir, args.reports_dir)

    print("Building training frame...")
    frame = build_training_frame(
        processed_dir / "deliveries.csv",
        processed_dir / "innings.csv",
    )

    print("Training model...")
    model, metrics, sample, feature_cols = train_model(frame)

    summary = [
        "# Score Prediction Model — v1",
        "",
        "## Task",
        "",
        "Predict final innings score using match context and first-six-over features.",
        "",
        "## Metrics",
        "",
        f"- Rows used  : {metrics['rows_total']:,}",
        f"- Train rows : {metrics['rows_train']:,}",
        f"- Test rows  : {metrics['rows_test']:,}",
        f"- MAE        : {metrics['mae']} runs",
        f"- RMSE       : {metrics['rmse']} runs",
        f"- R²         : {metrics['r2']}",
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
            "This model estimates where an innings is likely to finish after the powerplay. "
            "Useful for score projection, par-score analysis, and live match storytelling.",
        ]
    )

    save_model_artifacts(
        model=model,
        model_path=model_dir / "final_score_predictor.joblib",
        training_frame=frame,
        training_data_path=reports_dir / "score_model_training_data.csv",
        sample_df=sample,
        sample_path=reports_dir / "score_model_sample_predictions.csv",
        metrics=metrics,
        metrics_path=reports_dir / "score_model_metrics.json",
        feature_cols=feature_cols,
        summary_lines=summary,
        summary_path=reports_dir / "score_model_summary.md",
    )

    print(f"Model saved : {(model_dir / 'final_score_predictor.joblib').resolve()}")
    print(f"MAE         : {metrics['mae']} runs")
    print(f"RMSE        : {metrics['rmse']} runs")
    print(f"R²          : {metrics['r2']}")


if __name__ == "__main__":
    main()
