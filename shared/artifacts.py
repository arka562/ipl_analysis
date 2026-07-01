"""Shared routines for saving model artifacts (model file, CSVs, JSON, markdown)."""

import json
from pathlib import Path

import joblib
import pandas as pd


def save_model_artifacts(
    *,
    model,
    model_path,
    training_frame,
    training_data_path,
    sample_df,
    sample_path,
    metrics,
    metrics_path,
    feature_cols,
    summary_lines,
    summary_path,
):
    """Persist all standard model artifacts to disk.

    Parameters
    ----------
    model : sklearn Pipeline or estimator
        Trained model to serialize.
    model_path : str | Path
        Destination for the joblib file.
    training_frame : DataFrame
        Full training data to save as CSV.
    training_data_path : str | Path
        Destination for training data CSV.
    sample_df : DataFrame
        Sample predictions to save as CSV.
    sample_path : str | Path
        Destination for sample predictions CSV.
    metrics : dict
        Metrics dictionary; ``feature_cols`` is added in-place.
    metrics_path : str | Path
        Destination for the JSON metrics file.
    feature_cols : list[str]
        Feature column names (appended to metrics).
    summary_lines : list[str]
        Lines of the markdown summary report.
    summary_path : str | Path
        Destination for the markdown file.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    training_data_path = Path(training_data_path)
    training_data_path.parent.mkdir(parents=True, exist_ok=True)
    training_frame.to_csv(training_data_path, index=False)

    sample_path = Path(sample_path)
    sample_df.to_csv(sample_path, index=False)

    metrics["features"] = feature_cols
    metrics_path = Path(metrics_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_path = Path(summary_path)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def create_model_dirs(model_dir, reports_dir):
    """Ensure model and reports directories exist, returning Path objects."""
    model_dir = Path(model_dir)
    reports_dir = Path(reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, reports_dir
