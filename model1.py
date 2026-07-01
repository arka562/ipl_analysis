"""
Score Prediction Model — v1
Predicts final innings score using first-6-over (powerplay) features.
Model: RandomForestRegressor
"""
import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
        "mae":  round(mean_absolute_error(test_df[target_col], predictions), 2),
        "rmse": round(math.sqrt(mean_squared_error(test_df[target_col], predictions)), 2),
        "r2":   round(r2_score(test_df[target_col], predictions), 3),
    }

    sample = test_df[
        ["match_id", "season", "innings", "batting_team",
         "powerplay_runs", "powerplay_wickets", "final_score"]
    ].copy()
    sample["predicted_final_score"] = predictions.round(0).astype(int)
    sample["absolute_error"] = (sample["predicted_final_score"] - sample["final_score"]).abs()
    sample = sample.sort_values("absolute_error").head(30)

    return model, metrics, sample, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train IPL score prediction model v1.")
    parser.add_argument("--processed-dir", default="ipl_analytics_platform/data/processed")
    parser.add_argument("--model-dir",     default="ipl_analytics_platform/models")
    parser.add_argument("--reports-dir",   default="ipl_analytics_platform/reports/modeling")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    model_dir     = Path(args.model_dir)
    reports_dir   = Path(args.reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    required = ["deliveries.csv", "innings.csv"]
    missing = [f for f in required if not (processed_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"Missing required input files in {processed_dir}: {', '.join(missing)}\n"
            f"Run the parser first: python parsers/match_parser.py"
        )

    print("Building training frame...")
    frame = build_training_frame(
        processed_dir / "deliveries.csv",
        processed_dir / "innings.csv",
    )

    if frame.empty:
        raise SystemExit("Training frame is empty — no valid data to train on.")

    print("Training model...")
    model, metrics, sample, feature_cols = train_model(frame)

    joblib.dump(model, model_dir / "final_score_predictor.joblib")
    frame.to_csv(reports_dir / "score_model_training_data.csv", index=False)
    sample.to_csv(reports_dir / "score_model_sample_predictions.csv", index=False)

    metrics["features"] = feature_cols
    (reports_dir / "score_model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

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
    (reports_dir / "score_model_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Model saved : {(model_dir / 'final_score_predictor.joblib').resolve()}")
    print(f"MAE         : {metrics['mae']} runs")
    print(f"RMSE        : {metrics['rmse']} runs")
    print(f"R²          : {metrics['r2']}")


if __name__ == "__main__":
    main()
