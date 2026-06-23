"""
Score Prediction Model — v3 (Time Decay)
Predicts final innings score using first-10-over features + venue + chase context.
Model: GradientBoostingRegressor with Exponential Time Decay for sample weights.
"""
import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_OVER_LIMIT = 10
RANDOM_STATE = 42
DECAY_FACTOR = 0.15


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
                        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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

    # Time Decay Sample Weights
    max_season = train_df["season"].max()
    years_ago = max_season - train_df["season"]
    sample_weights = np.exp(-DECAY_FACTOR * years_ago)

    model.fit(
        train_df[feature_cols], 
        train_df[target_col], 
        model__sample_weight=sample_weights
    )
    
    predictions = model.predict(test_df[feature_cols])

    metrics = {
        "model_type":        "GradientBoostingRegressor (Time Decayed)",
        "prediction_task":   f"Predict final innings score using first {FEATURE_OVER_LIMIT} overs with decay factor {DECAY_FACTOR}",
        "feature_over_limit": FEATURE_OVER_LIMIT,
        "decay_factor": DECAY_FACTOR,
        "rows_total":  int(len(frame)),
        "rows_train":  int(len(train_df)),
        "rows_test":   int(len(test_df)),
        "mae":  round(mean_absolute_error(test_df[target_col], predictions), 2),
        "rmse": round(math.sqrt(mean_squared_error(test_df[target_col], predictions)), 2),
        "r2":   round(r2_score(test_df[target_col], predictions), 3),
        "innings_1_rows": int(len(frame[frame["innings"] == 1])),
        "innings_2_rows": int(len(frame[frame["innings"] == 2])),
    }

    # Per-innings breakdown
    for inn_no in (1, 2):
        sub = test_df[test_df["innings"] == inn_no]
        if len(sub):
            preds = model.predict(sub[feature_cols])
            metrics[f"mae_innings_{inn_no}"] = round(mean_absolute_error(sub[target_col], preds), 2)
            metrics[f"r2_innings_{inn_no}"]  = round(r2_score(sub[target_col], preds), 3)

    sample = test_df[
        ["match_id", "season", "innings", "batting_team",
         "runs_so_far", "wickets_so_far", "final_score"]
    ].copy()
    sample["predicted_final_score"] = predictions.round(0).astype(int)
    sample["absolute_error"] = (sample["predicted_final_score"] - sample["final_score"]).abs()
    sample = sample.sort_values("absolute_error").head(30)

    return model, metrics, sample, feature_cols


def main():
    # Automatically detect if processed data is stuck in the parsers folder
    default_processed = Path("ipl_analytics_platform/data/processed")
    if Path("parsers/ipl_analytics_platform/data/processed").exists():
        default_processed = Path("parsers/ipl_analytics_platform/data/processed")

    parser = argparse.ArgumentParser(description="Train IPL score prediction model v3 (Time Decay).")
    parser.add_argument("--processed-dir", default=str(default_processed))
    parser.add_argument("--model-dir",     default="ipl_analytics_platform/models")
    parser.add_argument("--reports-dir",   default="ipl_analytics_platform/reports/modeling")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    model_dir     = Path(args.model_dir)
    reports_dir   = Path(args.reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Building training frame...")
    frame = build_training_frame(
        processed_dir / "deliveries.csv",
        processed_dir / "innings.csv",
        processed_dir / "matches.csv",
    )

    print("Training time-decay model...")
    model, metrics, sample, feature_cols = train_model(frame)

    joblib.dump(model, model_dir / "final_score_predictor_v3.joblib")
    frame.to_csv(reports_dir / "score_model3_training_data.csv", index=False)
    sample.to_csv(reports_dir / "score_model3_sample_predictions.csv", index=False)

    metrics["features"] = feature_cols
    (reports_dir / "score_model3_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    summary = [
        "# Score Prediction Model — v3 (Time Decay)",
        "",
        "## Task",
        "",
        f"Predict final innings score using match context and first {FEATURE_OVER_LIMIT} overs. "
        f"Recent seasons are given exponentially more weight (decay factor = {DECAY_FACTOR}).",
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
        "## Key Improvements Over v2",
        "",
        "- Added exponential time decay to sample weights",
        f"- Decay factor of {DECAY_FACTOR} heavily prioritizes data from the last 3-5 years",
        "- Better reflects modern scoring trends",
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
            "By weighting recent data higher, it avoids under-predicting modern IPL scores "
            "where 200+ run totals are significantly more common than in 2008-2015.",
        ]
    )
    (reports_dir / "score_model3_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Model saved    : {(model_dir / 'final_score_predictor_v3.joblib').resolve()}")
    print(f"MAE            : {metrics['mae']} runs")
    print(f"RMSE           : {metrics['rmse']} runs")
    print(f"R²             : {metrics['r2']}")
    print(f"MAE 1st inn    : {metrics.get('mae_innings_1', 'N/A')} runs")
    print(f"MAE 2nd inn    : {metrics.get('mae_innings_2', 'N/A')} runs")


if __name__ == "__main__":
    main()
