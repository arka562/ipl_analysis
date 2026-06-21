import argparse
import csv
from pathlib import Path


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def top_row(rows, key):
    return max(rows, key=lambda row: as_float(row[key]))


def bottom_row(rows, key):
    return min(rows, key=lambda row: as_float(row[key]))


def fmt_pct(value):
    return f"{as_float(value):.1f}%"


def fmt_num(value):
    return f"{as_float(value):,.2f}".rstrip("0").rstrip(".")