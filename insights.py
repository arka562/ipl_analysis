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

def _require_rows(rows, name):
    if not rows:
        raise ValueError(f"No data found in {name} — cannot generate insights.")
    return rows


def build_insights(reports_dir, processed_dir):
    required_files = {
        "season_summary.csv": reports_dir,
        "toss_impact_by_season.csv": reports_dir,
        "chasing_vs_defending.csv": reports_dir,
        "venue_summary.csv": reports_dir,
        "team_summary.csv": reports_dir,
        "top_batters_by_phase.csv": reports_dir,
        "top_death_bowlers.csv": reports_dir,
        "batting_summary.csv": processed_dir,
        "bowling_summary.csv": processed_dir,
        "phase_summary.csv": processed_dir,
    }
    missing = [f"{d / f}" for f, d in required_files.items() if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input files for insight generation:\n" + "\n".join(f"  - {p}" for p in missing)
        )

    season       = _require_rows(read_csv(reports_dir / "season_summary.csv"), "season_summary")
    toss         = _require_rows(read_csv(reports_dir / "toss_impact_by_season.csv"), "toss_impact_by_season")
    chasing      = _require_rows(read_csv(reports_dir / "chasing_vs_defending.csv"), "chasing_vs_defending")
    venues       = _require_rows(read_csv(reports_dir / "venue_summary.csv"), "venue_summary")
    teams        = _require_rows(read_csv(reports_dir / "team_summary.csv"), "team_summary")
    phase_batters = _require_rows(read_csv(reports_dir / "top_batters_by_phase.csv"), "top_batters_by_phase")
    death_bowlers = _require_rows(read_csv(reports_dir / "top_death_bowlers.csv"), "top_death_bowlers")
    batting      = _require_rows(read_csv(processed_dir / "batting_summary.csv"), "batting_summary")
    bowling      = _require_rows(read_csv(processed_dir / "bowling_summary.csv"), "bowling_summary")
    phase_summary = _require_rows(read_csv(processed_dir / "phase_summary.csv"), "phase_summary")

    season_start   = min(as_int(row["season"]) for row in season)
    season_end     = max(as_int(row["season"]) for row in season)
    total_matches  = sum(as_int(row["matches"])       for row in season)
    total_runs     = sum(as_int(row["total_runs"])    for row in season)
    total_wickets  = sum(as_int(row["total_wickets"]) for row in season)

    highest_rr    = top_row(season,  "run_rate")
    lowest_rr     = bottom_row(season, "run_rate")
    best_toss     = top_row(toss,    "toss_winner_win_pct")
    worst_toss    = bottom_row(toss, "toss_winner_win_pct")
    best_chasing  = top_row(chasing,  "chasing_win_pct")
    worst_chasing = bottom_row(chasing, "chasing_win_pct")
    qualified_teams = [row for row in teams if as_int(row["matches"]) >= 50]
    if not qualified_teams:
        raise ValueError("No teams with 50+ matches found in team_summary.")
    best_team     = top_row(qualified_teams, "win_pct")
    highest_venue = top_row(venues, "avg_first_innings_score")
    toss_venue    = top_row(venues, "toss_winner_win_pct")
    top_batter    = batting[0]
    top_bowler    = bowling[0]

    death_phase_batters = [row for row in phase_batters if row["phase"] == "Death"]
    if not death_phase_batters:
        raise ValueError("No Death-phase batters found in top_batters_by_phase.")
    death_batter  = top_row(death_phase_batters, "strike_rate")

    powerplay_batters = [row for row in phase_batters if row["phase"] == "Powerplay"]
    if not powerplay_batters:
        raise ValueError("No Powerplay-phase batters found in top_batters_by_phase.")
    powerplay_batter = top_row(powerplay_batters, "runs")
    death_bowler  = death_bowlers[0]

    # Phase winner-loser gap
    phase_edges = []
    for phase in ("Powerplay", "Middle", "Death"):
        winner = next(
            (row for row in phase_summary
             if row["batting_result"] == "Winner" and row["phase"] == phase),
            None,
        )
        loser = next(
            (row for row in phase_summary
             if row["batting_result"] == "Loser" and row["phase"] == phase),
            None,
        )
        if winner is None or loser is None:
            raise ValueError(f"Missing Winner/Loser rows for {phase} phase in phase_summary.")
        edge = as_float(winner["avg_runs"]) - as_float(loser["avg_runs"])
        phase_edges.append((phase, edge, winner["avg_runs"], loser["avg_runs"]))

    strongest_phase = max(phase_edges, key=lambda item: item[1])

    lines = [
        "# IPL Intelligence: Key Insights",
        "",
        "## Dataset",
        "",
        f"- Seasons covered: {season_start}–{season_end}.",
        f"- Matches analysed: {total_matches:,}.",
        f"- Total runs: {total_runs:,}; total wickets: {total_wickets:,}.",
        "",
        "## Executive Summary",
        "",
        f"1. IPL scoring has not been static: the highest run-rate season was "
        f"{highest_rr['season']} at {highest_rr['run_rate']} runs per over, "
        f"compared with the lowest season, {lowest_rr['season']}, at {lowest_rr['run_rate']}.",

        f"2. The strongest winner-loser phase gap is the {strongest_phase[0].lower()} phase: "
        f"winners averaged {strongest_phase[2]} runs and losers averaged {strongest_phase[3]}, "
        f"a gap of {strongest_phase[1]:.2f} runs.",

        f"3. Toss advantage varies heavily by season: toss winners peaked at "
        f"{fmt_pct(best_toss['toss_winner_win_pct'])} in {best_toss['season']} "
        f"and fell as low as {fmt_pct(worst_toss['toss_winner_win_pct'])} in {worst_toss['season']}.",

        f"4. Chasing was strongest in {best_chasing['season']}, with chasing teams winning "
        f"{fmt_pct(best_chasing['chasing_win_pct'])} of completed matches; "
        f"it was weakest in {worst_chasing['season']} at {fmt_pct(worst_chasing['chasing_win_pct'])}.",

        f"5. Among teams with at least 50 matches, {best_team['team']} has the highest "
        f"win percentage at {fmt_pct(best_team['win_pct'])}.",

        f"6. {highest_venue['venue']} has the highest average first-innings score among "
        f"venues with at least five matches: {fmt_num(highest_venue['avg_first_innings_score'])}.",

        f"7. {toss_venue['venue']} is the venue where toss winners had the strongest match "
        f"win percentage: {fmt_pct(toss_venue['toss_winner_win_pct'])}.",

        f"8. The all-time top run scorer is {top_batter['player']} "
        f"with {int(as_float(top_batter['runs'])):,} runs.",

        f"9. The all-time top wicket taker is {top_bowler['player']} "
        f"with {int(as_float(top_bowler['wickets'])):,} wickets.",

        f"10. In death overs, {death_batter['batter']} stands out for a "
        f"{death_batter['strike_rate']} strike rate among the top phase run scorers, "
        f"while {death_bowler['bowler']} leads death-over wickets with {death_bowler['wickets']}.",

        "",
        "## Why This Matters",
        "",
        "This project is not just a dashboard. It is an analytics pipeline that starts with "
        "raw nested JSON, creates normalized ball-by-ball tables, loads them into a SQL "
        "warehouse, exports reusable report tables, and turns those reports into "
        "portfolio-ready findings.",
        "",
        "## Resume Framing",
        "",
        "Built an end-to-end IPL analytics platform using 1,200+ raw Cricsheet JSON files "
        "from 2008-present, transforming 294K+ ball-by-ball deliveries into normalized "
        "datasets, a DuckDB analytics warehouse, SQL report tables, and insight reports "
        "covering toss impact, phase-wise scoring, venue behavior, team performance, "
        "and player value.",
    ]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate recruiter-readable IPL analytics insights."
    )
    parser.add_argument("--reports-dir",   default="ipl_analytics_platform/reports/tables")
    parser.add_argument("--processed-dir", default="ipl_analytics_platform/data/processed")
    parser.add_argument("--output",        default="ipl_analytics_platform/reports/insights.md")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_insights(Path(args.reports_dir), Path(args.processed_dir)),
        encoding="utf-8",
    )
    print(f"Insights written: {output_path.resolve()}")


if __name__ == "__main__":
    main()
