"""Shared path-resolution utilities."""

import argparse
from pathlib import Path


def resolve_processed_dir(default="ipl_analytics_platform/data/processed"):
    """Return the processed-data directory, checking the parsers fallback location."""
    default_path = Path(default)
    fallback = Path("parsers/ipl_analytics_platform/data/processed")
    if fallback.exists():
        return fallback
    return default_path


def add_common_model_args(parser, description=None):
    """Add standard --processed-dir, --model-dir, --reports-dir arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.
    description : str | None
        If provided, set as parser description.

    Returns
    -------
    argparse.ArgumentParser
    """
    if description:
        parser.description = description
    parser.add_argument(
        "--processed-dir",
        default=str(resolve_processed_dir()),
    )
    parser.add_argument(
        "--model-dir",
        default="ipl_analytics_platform/models",
    )
    parser.add_argument(
        "--reports-dir",
        default="ipl_analytics_platform/reports/modeling",
    )
    return parser
