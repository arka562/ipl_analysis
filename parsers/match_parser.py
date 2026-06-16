import argparse
import csv
import json
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