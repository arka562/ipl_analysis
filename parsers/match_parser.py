import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


BOWLER_CREDITED_WICKETS = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
    "hit the ball twice",
}


def first_or_blank(values):
    return values[0] if values else ""


def season_from_info(info):
    season = info.get("season")

    if isinstance(season, int):
        return season

    if isinstance(season, str) and season[:4].isdigit():
        return int(season[:4])

    date_text = first_or_blank(info.get("dates", []))
    return int(date_text[:4]) if date_text[:4].isdigit() else None


def phase_for_over(over):
    if 0 <= over <= 5:
        return "Powerplay"
    if 6 <= over <= 15:
        return "Middle"
    if 16 <= over <= 19:
        return "Death"
    return "Other"


def write_csv(path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def parse_match(path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    info = data.get("info", {})
    outcome = info.get("outcome", {})
    toss = info.get("toss", {})
    teams = info.get("teams", [])

    winner = outcome.get("winner", "")
    match_id = path.stem
    season = season_from_info(info)
    date_text = first_or_blank(info.get("dates", []))

    match = {
        "match_id": match_id,
        "season": season,
        "date": date_text,
        "event_name": info.get("event", {}).get("name", ""),
        "match_number": info.get("event", {}).get("match_number", ""),
        "city": info.get("city", ""),
        "venue": info.get("venue", ""),
        "team_1": teams[0] if len(teams) > 0 else "",
        "team_2": teams[1] if len(teams) > 1 else "",
        "winner": winner,
        "win_by_runs": outcome.get("by", {}).get("runs", ""),
        "win_by_wickets": outcome.get("by", {}).get("wickets", ""),
        "result_method": outcome.get("method", ""),
        "toss_winner": toss.get("winner", ""),
        "toss_decision": toss.get("decision", ""),
        "player_of_match": "; ".join(info.get("player_of_match", [])),
        "balls_per_over": info.get("balls_per_over", 6),
    }

    innings_rows = []
    delivery_rows = []
    player_rows = []

    for team, players in info.get("players", {}).items():
        for player in players:
            player_rows.append(
                {
                    "match_id": match_id,
                    "season": season,
                    "team": team,
                    "player": player,
                }
            )

    for innings_number, inning in enumerate(data.get("innings", []), start=1):
        batting_team = inning.get("team", "")

        if winner and batting_team == winner:
            batting_result = "Winner"
        elif winner:
            batting_result = "Loser"
        else:
            batting_result = "No result"

        total_runs = 0
        total_wickets = 0
        legal_balls = 0

        for over_obj in inning.get("overs", []):
            over = int(over_obj.get("over", 0))

            for ball_index, delivery in enumerate(over_obj.get("deliveries", []), start=1):
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                is_legal_ball = "wides" not in extras and "noballs" not in extras

                bowler_wickets = sum(
                    1
                    for wicket in wickets
                    if wicket.get("kind", "") in BOWLER_CREDITED_WICKETS
                )

                total_runs += int(runs.get("total", 0))
                total_wickets += len(wickets)
                legal_balls += int(is_legal_ball)

                delivery_rows.append(
                    {
                        "match_id": match_id,
                        "season": season,
                        "date": date_text,
                        "innings": innings_number,
                        "batting_team": batting_team,
                        "winner": winner,
                        "batting_result": batting_result,
                        "over": over,
                        "ball_in_over": ball_index,
                        "phase": phase_for_over(over),
                        "batter": delivery.get("batter", ""),
                        "non_striker": delivery.get("non_striker", ""),
                        "bowler": delivery.get("bowler", ""),
                        "batter_runs": int(runs.get("batter", 0)),
                        "extras": int(runs.get("extras", 0)),
                        "total_runs": int(runs.get("total", 0)),
                        "is_legal_ball": int(is_legal_ball),
                        "wides": int(extras.get("wides", 0)),
                        "noballs": int(extras.get("noballs", 0)),
                        "byes": int(extras.get("byes", 0)),
                        "legbyes": int(extras.get("legbyes", 0)),
                        "penalty": int(extras.get("penalty", 0)),
                        "wicket_count": len(wickets),
                        "bowler_wickets": bowler_wickets,
                        "wicket_kinds": "; ".join(
                            wicket.get("kind", "") for wicket in wickets
                        ),
                        "players_out": "; ".join(
                            wicket.get("player_out", "") for wicket in wickets
                        ),
                    }
                )

        innings_rows.append(
            {
                "match_id": match_id,
                "season": season,
                "innings": innings_number,
                "batting_team": batting_team,
                "winner": winner,
                "batting_result": batting_result,
                "runs": total_runs,
                "wickets": total_wickets,
                "legal_balls": legal_balls,
                "overs": f"{legal_balls // 6}.{legal_balls % 6}",
            }
        )

    return match, innings_rows, delivery_rows, player_rows


def build_summary_tables(deliveries):
    batter_runs = Counter()
    bowler_wickets = Counter()
    bowler_balls = Counter()
    bowler_runs = Counter()
    phase_runs = defaultdict(int)
    phase_innings = set()

    for row in deliveries:
        batter_runs[row["batter"]] += row["batter_runs"]
        bowler_wickets[row["bowler"]] += row["bowler_wickets"]
        bowler_balls[row["bowler"]] += row["is_legal_ball"]
        bowler_runs[row["bowler"]] += (
            row["total_runs"] - row["byes"] - row["legbyes"] - row["penalty"]
        )

        if row["batting_result"] in {"Winner", "Loser"}:
            phase_runs[(row["batting_result"], row["phase"])] += row["total_runs"]
            phase_innings.add(
                (
                    row["match_id"],
                    row["innings"],
                    row["batting_result"],
                    row["phase"],
                )
            )

    batting = [
        {
            "player": player,
            "runs": runs,
        }
        for player, runs in batter_runs.most_common()
        if player
    ]

    bowling = []

    for bowler, wickets in bowler_wickets.most_common():
        if not bowler:
            continue

        balls = bowler_balls[bowler]
        runs = bowler_runs[bowler]

        bowling.append(
            {
                "player": bowler,
                "wickets": wickets,
                "balls": balls,
                "runs_conceded": runs,
                "economy": round(runs / (balls / 6), 2) if balls else "",
            }
        )

    phase = []

    for result in ("Winner", "Loser"):
        for phase_name in ("Powerplay", "Middle", "Death"):
            innings_count = len(
                {
                    (match_id, innings)
                    for match_id, innings, batting_result, row_phase in phase_innings
                    if batting_result == result and row_phase == phase_name
                }
            )

            runs = phase_runs[(result, phase_name)]

            phase.append(
                {
                    "batting_result": result,
                    "phase": phase_name,
                    "runs": runs,
                    "innings": innings_count,
                    "avg_runs": round(runs / innings_count, 2)
                    if innings_count
                    else "",
                }
            )

    return batting, bowling, phase


def main():
    parser = argparse.ArgumentParser(
        description="Build normalized IPL analytics tables from Cricsheet JSON."
    )

    parser.add_argument(
        "--input-dir",
        default=r"ipl_analytics_platform/data/raw/ipl_json",
    )

    parser.add_argument(
        "--output-dir",
        default="ipl_analytics_platform/data/processed",
    )

    args = parser.parse_args()

    matches = []
    innings = []
    deliveries = []
    players = []
    skipped = []

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {input_dir}")

    for path in json_files:
        try:
            match, innings_rows, delivery_rows, player_rows = parse_match(path)
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as error:
            skipped.append(
                {
                    "file": str(path),
                    "error": str(error),
                }
            )
            continue

        matches.append(match)
        innings.extend(innings_rows)
        deliveries.extend(delivery_rows)
        players.extend(player_rows)

    if not matches:
        raise SystemExit("No JSON matches were parsed. Check --input-dir.")

    batting, bowling, phase = build_summary_tables(deliveries)

    write_csv(output_dir / "matches.csv", matches, list(matches[0].keys()))
    write_csv(output_dir / "innings.csv", innings, list(innings[0].keys()))
    write_csv(output_dir / "deliveries.csv", deliveries, list(deliveries[0].keys()))
    write_csv(output_dir / "match_players.csv", players, list(players[0].keys()))

    write_csv(
        output_dir / "batting_summary.csv",
        batting,
        ["player", "runs"],
    )

    write_csv(
        output_dir / "bowling_summary.csv",
        bowling,
        ["player", "wickets", "balls", "runs_conceded", "economy"],
    )

    write_csv(
        output_dir / "phase_summary.csv",
        phase,
        ["batting_result", "phase", "runs", "innings", "avg_runs"],
    )

    if skipped:
        write_csv(
            output_dir / "skipped_files.csv",
            skipped,
            ["file", "error"],
        )

    total_files = len(matches) + len(skipped)
    seasons = sorted({row["season"] for row in matches if row["season"]})

    print(f"Matches parsed  : {len(matches)}")
    print(f"Innings parsed  : {len(innings)}")
    print(f"Deliveries      : {len(deliveries)}")
    print(f"Seasons         : {seasons[0]}-{seasons[-1]}" if seasons else "Seasons: unknown")
    print(f"Skipped files   : {len(skipped)}")
    print(f"Output folder   : {output_dir.resolve()}")

    if skipped and total_files > 0 and len(skipped) / total_files > 0.1:
        pct = 100 * len(skipped) / total_files
        print(
            f"WARNING: {pct:.0f}% of input files were skipped due to errors. "
            f"Review {output_dir / 'skipped_files.csv'} for details.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
