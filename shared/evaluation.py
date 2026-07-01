"""Shared model evaluation helpers."""

import math

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def evaluate_regression(y_true, y_pred):
    """Compute standard regression metrics.

    Returns
    -------
    dict with keys: mae, rmse, r2
    """
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 2),
        "rmse": round(math.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "r2": round(r2_score(y_true, y_pred), 3),
    }


def evaluate_classification(y_true, y_prob, threshold=0.5):
    """Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (0/1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    threshold : float
        Decision threshold for accuracy calculation.

    Returns
    -------
    dict with keys: accuracy, roc_auc, log_loss
    """
    predictions = (y_prob >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, predictions), 3),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 3),
        "log_loss": round(log_loss(y_true, y_prob), 3),
    }


def build_regression_sample(test_df, predictions, id_cols, target_col, n=30):
    """Build a sample-predictions DataFrame sorted by lowest absolute error.

    Parameters
    ----------
    test_df : DataFrame
        Test split (must contain *id_cols* and *target_col*).
    predictions : array-like
        Model predictions aligned with test_df.
    id_cols : list[str]
        Columns to keep for identification/context.
    target_col : str
        Name of the ground-truth target column.
    n : int
        Number of rows to return.

    Returns
    -------
    DataFrame
    """
    sample = test_df[id_cols + [target_col]].copy()
    sample["predicted_final_score"] = predictions.round(0).astype(int)
    sample["absolute_error"] = (sample["predicted_final_score"] - sample[target_col]).abs()
    return sample.sort_values("absolute_error").head(n)
