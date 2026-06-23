"""
Win Probability Curve
Loads the trained win-probability model, selects the most exciting match
(highest probability swing in 2nd innings), and writes a CSV + SVG curve.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


COLORS = ["#2563eb", "#f97316", "#16a34a", "#9333ea"]


# ---------------------------------------------------------------------------
# Match selection
# ---------------------------------------------------------------------------

def choose_match(frame):
    """Return the match_id with the highest win-probability swing in innings 2."""
    scored = []
    for match_id, group in frame.groupby("match_id"):
        if group["innings"].nunique() < 2:
            continue
        inn2 = group[group["innings"] == 2]
        if inn2.empty:
            continue
        swing = inn2["win_probability"].max() - inn2["win_probability"].min()
        scored.append((swing, match_id))
    if not scored:
        return str(frame["match_id"].iloc[0])
    return str(max(scored)[1])


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def scale_x(over, left, chart_width):
    return left + (over - 1) * chart_width / 19


def scale_y(probability, top, chart_height):
    return top + (100 - probability) * chart_height / 100


def make_svg(match_frame, output_path):
    # Canvas
    width       = 1050
    height      = 620
    left        = 90
    right       = 40
    top         = 90
    bottom      = 90
    chart_width  = width  - left - right
    chart_height = height - top  - bottom

    match_id = match_frame["match_id"].iloc[0]
    season   = int(match_frame["season"].iloc[0])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',

        # Title
        f'<text x="{left}" y="42" font-family="Arial" font-size="28" '
        f'font-weight="700" fill="#111827">Win Probability Curve</text>',
        f'<text x="{left}" y="70" font-family="Arial" font-size="15" fill="#4b5563">'
        f'Match {match_id} · Season {season} · batting-team win probability after each over'
        f'</text>',

        # Axes
        f'<line x1="{left}" y1="{top + chart_height}" '
        f'x2="{left + chart_width}" y2="{top + chart_height}" '
        f'stroke="#374151" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" '
        f'stroke="#374151" stroke-width="1.5"/>',
    ]

    # Y-axis gridlines + labels (0 % – 100 %)
    for tick in range(0, 101, 20):
        y = scale_y(tick, top, chart_height)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" '
            f'stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" '
            f'font-family="Arial" font-size="13" fill="#4b5563">{tick}%</text>'
        )

    # X-axis tick labels at key overs
    for over in (1, 5, 10, 15, 20):
        x = scale_x(over, left, chart_width)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_height}" '
            f'stroke="#f3f4f6"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + chart_height + 28}" text-anchor="middle" '
            f'font-family="Arial" font-size="13" fill="#4b5563">{over}</text>'
        )

    # One line per innings / team
    legend_y = top + chart_height + 58
    for idx, ((innings, team), group) in enumerate(
        match_frame.groupby(["innings", "batting_team"])
    ):
        group = group.sort_values("over_number")
        color = COLORS[idx % len(COLORS)]

        points = [
            f'{scale_x(int(row.over_number), left, chart_width):.1f},'
            f'{scale_y(float(row.win_probability), top, chart_height):.1f}'
            for row in group.itertuples()
        ]

        if len(points) >= 2:
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{color}" stroke-width="3"/>'
            )

        for row in group.itertuples():
            cx = scale_x(int(row.over_number), left, chart_width)
            cy = scale_y(float(row.win_probability), top, chart_height)
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="{color}"/>')

        # Legend entry
        lx    = left + idx * 260
        label = f"Innings {int(innings)}: {team}"
        parts.append(f'<rect x="{lx}" y="{legend_y}" width="16" height="16" fill="{color}"/>')
        parts.append(
            f'<text x="{lx + 24}" y="{legend_y + 13}" '
            f'font-family="Arial" font-size="14" fill="#111827">{label}</text>'
        )

    # Axis labels
    parts.append(
        f'<text x="{left + chart_width / 2}" y="{height - 18}" '
        f'text-anchor="middle" font-family="Arial" font-size="13" fill="#4b5563">'
        f'Over number</text>'
    )
    parts.append(
        f'<text x="22" y="{top + chart_height / 2}" '
        f'transform="rotate(-90 22 {top + chart_height / 2})" '
        f'text-anchor="middle" font-family="Arial" font-size="13" fill="#4b5563">'
        f'Win probability</text>'
    )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
