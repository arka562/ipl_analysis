"""Common I/O helpers used across the analytics pipeline."""

import csv
from pathlib import Path


def read_csv(path):
    """Read a CSV file and return a list of dictionaries (one per row)."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, headers):
    """Write a list of dictionaries to a CSV file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def as_int(value):
    """Safely convert a value to int, returning 0 on failure."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def as_float(value):
    """Safely convert a value to float, returning 0.0 on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_rate(numerator, denominator, multiplier=1.0):
    """Compute a rate safely, returning 0.0 when the denominator is zero."""
    return multiplier * numerator / denominator if denominator else 0.0
