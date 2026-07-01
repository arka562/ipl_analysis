"""
Score Prediction Model — v2
Predicts final innings score using first-10-over features + venue + chase context.
Model: GradientBoostingRegressor (improved over v1 RandomForest)
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from shared.artifacts import create_model_dirs, save_model_artifacts
from shared.evaluation import evaluate_regression, build_regression_sample
from shared.preprocessing import build_preprocessor


FEATURE_OVER_LIMIT = 10
RANDOM_STATE = 42


def build_training_frame(deliveries_path, innings_path, matches_path):
    deliveries = pd.read_csv(deliveries_path)
    innings    = pd.read_csv(innings_path)
    matches    = pd.read_csv(matches_path)

    early = deliveries[deliveries["over"] < FEATURE_OVER_LIMIT].copy()
    early["boundary"] = early["batter_runs"].isin([4, 6]).astype(int)
    early["dot_ball"] = (early["total_runs"] == 0).astype(int)
    early["is_six"]   = (early["batter_runs"] == 6).astype(int)

    features = (
        early.groupby(["match_id", "season", "innings", "batting_team"], as_index=False)
        .agg(
            runs_so_far    =("total_runs",   "sum"),
            wickets_so_far =("wicket_count",  "sum"),
            balls_so_far   =("is_legal_ball", "sum"),
            boundaries     =("boundary",      "sum"),
            sixes          =("is_six",        "sum"),
            dot_balls      =("dot_ball",      "sum"),
        )
    )

    features["run_rate_so_far"]  = features["runs_so_far"] * 6 / features["balls_so_far"].replace(0, pd.NA)
    features["wickets_remaining"] = 10 - features["wickets_so_far"]
    features["boundary_pct"]     = 100 * features["boundaries"]  / features["balls_so_far"].replace(0, pd.NA)
    features["dot_ball_pct"]     = 100 * features["dot_balls"]   / features["balls_so_far"].replace(0, pd.NA)
    features["six_pct"]          = 100 * features["sixes"]       / features["balls_so_far"].replace(0, pd.NA)
    features["is_first_innings"] = (features["innings"] == 1).astype(int)
    features["balls_remaining"]  = 120 - features["balls_so_far"]

    # Attach venue
    match_meta = matches[["match_id", "venue", "city"]].copy()
    features   = features.merge(match_meta, on="match_id", how="left")

    # Attach target score (2nd innings only)
    first_inn_scores = (
        innings[innings["innings"] == 1][["match_id", "runs"]]
        .rename(columns={"runs": "target_score"})
    )
    features = features.merge(first_inn_scores, on="match_id", how="left")
    features["target_score"] = features["target_score"].where(features["innings"] == 2, 0)
    features["runs_needed"]  = (features["target_score"] - features["runs_so_far"]).clip(lower=0)
    features["runs_needed"]  = features["runs_needed"].where(features["innings"] == 2, 0)
    features["required_run_rate"] = (
        features["runs_needed"] * 6 / features["balls_remaining"].replace(0, pd.NA)
    ).where(features["innings"] == 2, 0)

    target = innings[["match_id", "innings", "runs"]].rename(columns={"runs": "final_score"})
    frame  = features.merge(target, on=["match_id", "innings"], how="inner")
    frame  = frame[frame["balls_so_far"] >= 55].copy()
    frame  = frame.dropna(subset=["final_score"])
    return frame


def train_model(frame):
    feature_cols = [
        "season", "innings", "batting_team", "venue",
        "runs_so_far", "wickets_so_far", "balls_so_far",
        "boundaries", "sixes", "dot_balls",
        "run_rate_so_far", "wickets_remaining",
        "boundary_pct", "dot_ball_pct", "six_pct",
        "is_first_innings", "balls_remaining",
        "target_score", "runs_needed", "required_run_rate",
    ]
    target_col = "final_score"

    train_df, test_df = train_test_split(
        frame, test_size=0.2, random_state=RANDOM_STATE, stratify=frame["innings"]
    )

    numeric_features     = [c for c in feature_cols if c not in ("batting_team", "venue")]
    categorical_features = ["batting_team", "venue"]

    preprocessor = build_preprocessor(
        numeric_features, categorical_features, sparse_output=False
    )

    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_leaf=5,
                    subsample=0.8,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(train_df[feature_cols], train_df[target_col])
    predictions = model.predict(test_df[feature_cols])

    metrics = {
        "model_type":        "GradientBoostingRegressor",
        "prediction_task":   f"Predict final innings score using first {FEATURE_OVER_LIMIT} overs",
        "feature_over_limit": FEATURE_OVER_LIMIT,
        "rows_total":  int(len(frame)),
        "rows_train":  int(len(train_df)),
        "rows_test":   int(len(test_df)),
        **evaluate_regression(test_df[target_col], predictions),
        "innings_1_rows": int(len(frame[frame["innings"] == 1])),
        "innings_2_rows": int(len(frame[frame["innings"] == 2])),
    }

    # Per-innings breakdown
    for inn_no in (1, 2):
        sub = test_df[test_df["innings"] == inn_no]
        if len(sub):
            preds = model.predict(sub[feature_cols])
            inn_metrics = evaluate_regression(sub[target_col], preds)
            metrics[f"mae_innings_{inn_no}"] = inn_metrics["mae"]
            metrics[f"r2_innings_{inn_no}"]  = inn_metrics["r2"]

    sample = build_regression_sample(
        test_df, predictions,
        id_cols=["match_id", "season", "innings", "batting_team",
                 "runs_so_far", "wickets_so_far"],
        target_col=target_col,
    )

    return model, metrics, sample, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train IPL score prediction model v2.")
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
        processed_dir / "matches.csv",
    )

    print("Training model...")
    model, metrics, sample, feature_cols = train_model(frame)

    summary = [
        "# Score Prediction Model — v2",
        "",
        "## Task",
        "",
        f"Predict final innings score using match context and first {FEATURE_OVER_LIMIT} overs.",
        "",
        "## Metrics",
        "",
        f"- Rows used       : {metrics['rows_total']:,}",
        f"- Train rows      : {metrics['rows_train']:,}",
        f"- Test rows       : {metrics['rows_test']:,}",
        f"- MAE             : {metrics['mae']} runs",
        f"- RMSE            : {metrics['rmse']} runs",
        f"- R²              : {metrics['r2']}",
        f"- MAE (1st inn)   : {metrics.get('mae_innings_1', 'N/A')} runs",
        f"- MAE (2nd inn)   : {metrics.get('mae_innings_2', 'N/A')} runs",
        f"- R² (1st inn)    : {metrics.get('r2_innings_1', 'N/A')}",
        f"- R² (2nd inn)    : {metrics.get('r2_innings_2', 'N/A')}",
        "",
        "## Key Improvements Over v1",
        "",
        "- Extended feature window from 6 to 10 overs",
        "- Added venue as a categorical feature",
        "- Added target score, runs needed, and required run rate for 2nd innings",
        "- Added sixes count and six percentage",
        "- Added balls remaining",
        "- Switched from Random Forest to Gradient Boosting",
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
            "This model estimates where an innings is likely to finish after 10 overs. "
            "Second-innings predictions are stronger because the model knows the target. "
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

    print(f"Model saved    : {(model_dir / 'final_score_predictor.joblib').resolve()}")
    print(f"MAE            : {metrics['mae']} runs")
    print(f"RMSE           : {metrics['rmse']} runs")
    print(f"R²             : {metrics['r2']}")
    print(f"MAE 1st inn    : {metrics.get('mae_innings_1', 'N/A')} runs")
    print(f"MAE 2nd inn    : {metrics.get('mae_innings_2', 'N/A')} runs")


if __name__ == "__main__":
    main()
