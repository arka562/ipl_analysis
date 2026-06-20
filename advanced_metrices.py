import argparse
import csv
from collections import defaultdict
from pathlib import Path


PHASES = ("Powerplay", "Middle", "Death")
PLAYER_IMPACT_MIN_BATTING_BALLS = 600
PLAYER_IMPACT_MIN_BOWLING_BALLS = 600
DEATH_INDEX_MIN_BALLS = 120


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def pct_rank(value, values):
    if not values:
        return 0.0
    lower_or_equal = sum(1 for item in values if item <= value)
    return 100.0 * lower_or_equal / len(values)


def safe_rate(numerator, denominator, multiplier=1.0):
    return multiplier * numerator / denominator if denominator else 0.0


def build_match_lookup(matches):
    return {row["match_id"]: row for row in matches}


# ---------------------------------------------------------------------------
# Player Impact Score
# ---------------------------------------------------------------------------

def build_player_impact(deliveries):
    batting = defaultdict(lambda: defaultdict(int))
    bowling = defaultdict(lambda: defaultdict(int))

    for row in deliveries:
        batter = row["batter"]
        bowler = row["bowler"]

        batting[batter]["runs"]       += as_int(row["batter_runs"])
        batting[batter]["balls"]      += as_int(row["is_legal_ball"])
        batting[batter]["boundaries"] += int(as_int(row["batter_runs"]) in (4, 6))
        batting[batter]["dismissals"] += sum(
            1 for player in row["players_out"].split("; ") if player == batter
        )

        bowling[bowler]["wickets"] += as_int(row["bowler_wickets"])
        bowling[bowler]["balls"]   += as_int(row["is_legal_ball"])
        bowling[bowler]["runs"]    += (
            as_int(row["total_runs"])
            - as_int(row["byes"])
            - as_int(row["legbyes"])
            - as_int(row["penalty"])
        )
        bowling[bowler]["dots"] += int(as_int(row["total_runs"]) == 0)

    players = sorted((set(batting) | set(bowling)) - {""})
    raw_rows = []

    for player in players:
        bat  = batting[player]
        bowl = bowling[player]

        bat_sr       = safe_rate(bat["runs"],      bat["balls"],       100)
        bat_avg      = safe_rate(bat["runs"],      bat["dismissals"])
        boundary_pct = safe_rate(bat["boundaries"], bat["balls"],      100)
        economy      = safe_rate(bowl["runs"],     bowl["balls"],        6)
        wicket_rate  = safe_rate(bowl["wickets"],  bowl["balls"],        6)
        dot_pct      = safe_rate(bowl["dots"],     bowl["balls"],      100)

        raw_rows.append(
            {
                "player":               player,
                "batting_runs":         bat["runs"],
                "batting_balls":        bat["balls"],
                "batting_strike_rate":  round(bat_sr, 2),
                "batting_average":      round(bat_avg, 2),
                "boundary_pct":         round(boundary_pct, 2),
                "bowling_wickets":      bowl["wickets"],
                "bowling_balls":        bowl["balls"],
                "bowling_economy":      round(economy, 2) if bowl["balls"] else "",
                "wickets_per_over":     round(wicket_rate, 3),
                "dot_ball_pct":         round(dot_pct, 2),
                "_bat_sr":              bat_sr,
                "_bat_avg":             bat_avg,
                "_boundary_pct":        boundary_pct,
                "_economy":             economy,
                "_wicket_rate":         wicket_rate,
                "_dot_pct":             dot_pct,
            }
        )

    sr_values       = [r["_bat_sr"]      for r in raw_rows if r["batting_balls"]  >= PLAYER_IMPACT_MIN_BATTING_BALLS]
    avg_values      = [r["_bat_avg"]     for r in raw_rows if r["batting_balls"]  >= PLAYER_IMPACT_MIN_BATTING_BALLS]
    boundary_values = [r["_boundary_pct"] for r in raw_rows if r["batting_balls"] >= PLAYER_IMPACT_MIN_BATTING_BALLS]
    wicket_values   = [r["_wicket_rate"] for r in raw_rows if r["bowling_balls"]  >= PLAYER_IMPACT_MIN_BOWLING_BALLS]
    dot_values      = [r["_dot_pct"]     for r in raw_rows if r["bowling_balls"]  >= PLAYER_IMPACT_MIN_BOWLING_BALLS]
    economy_values  = [-r["_economy"]    for r in raw_rows if r["bowling_balls"]  >= PLAYER_IMPACT_MIN_BOWLING_BALLS]

    output = []
    for row in raw_rows:
        batting_score  = 0.0
        bowling_score  = 0.0
        bat_eligible   = row["batting_balls"]  >= PLAYER_IMPACT_MIN_BATTING_BALLS
        bowl_eligible  = row["bowling_balls"]  >= PLAYER_IMPACT_MIN_BOWLING_BALLS

        if bat_eligible:
            batting_score = (
                0.45 * pct_rank(row["_bat_sr"],       sr_values)
                + 0.35 * pct_rank(row["_bat_avg"],    avg_values)
                + 0.20 * pct_rank(row["_boundary_pct"], boundary_values)
            )
        if bowl_eligible:
            bowling_score = (
                0.45 * pct_rank(row["_wicket_rate"],  wicket_values)
                + 0.35 * pct_rank(-row["_economy"],   economy_values)
                + 0.20 * pct_rank(row["_dot_pct"],    dot_values)
            )

        if batting_score >= bowling_score * 1.4:
            role = "Batter"
        elif bowling_score >= batting_score * 1.4:
            role = "Bowler"
        else:
            role = "All-rounder"

        output.append(
            {
                "player":               row["player"],
                "role_signal":          role,
                "player_impact_score":  round(max(batting_score, bowling_score, (batting_score + bowling_score) / 2), 2),
                "batting_score":        round(batting_score, 2),
                "bowling_score":        round(bowling_score, 2),
                "impact_eligible":      int(bat_eligible or bowl_eligible),
                "batting_runs":         row["batting_runs"],
                "batting_balls":        row["batting_balls"],
                "batting_strike_rate":  row["batting_strike_rate"],
                "batting_average":      row["batting_average"],
                "boundary_pct":         row["boundary_pct"],
                "bowling_wickets":      row["bowling_wickets"],
                "bowling_balls":        row["bowling_balls"],
                "bowling_economy":      row["bowling_economy"],
                "wickets_per_over":     row["wickets_per_over"],
                "dot_ball_pct":         row["dot_ball_pct"],
            }
        )

    return sorted(
        output,
        key=lambda r: (r["impact_eligible"], r["player_impact_score"]),
        reverse=True,
    )


---------------------------------------------------------------------------
Venue Par Score
---------------------------------------------------------------------------

def build_venue_par_score(matches, innings):
    match_lookup  = build_match_lookup(matches)
    venue_scores  = defaultdict(list)
    venue_chasing = defaultdict(lambda: {"matches": 0, "chasing_wins": 0})

    for row in innings:
        if as_int(row["innings"]) == 1:
            match  = match_lookup.get(row["match_id"], {})
            venue  = match.get("venue", "")
            city   = match.get("city", "")
            if venue:
                venue_scores[(venue, city)].append(as_int(row["runs"]))

    innings_by_match = defaultdict(dict)
    for row in innings:
        innings_by_match[row["match_id"]][as_int(row["innings"])] = row["batting_team"]

    for match in matches:
        winner        = match["winner"]
        venue         = match["venue"]
        city          = match["city"]
        second_batting = innings_by_match.get(match["match_id"], {}).get(2)
        if winner and venue and second_batting:
            venue_chasing[(venue, city)]["matches"]      += 1
            venue_chasing[(venue, city)]["chasing_wins"] += int(winner == second_batting)

    rows = []
    for (venue, city), scores in venue_scores.items():
        if len(scores) < 5:
            continue

        sorted_scores = sorted(scores)
        mid = len(sorted_scores) // 2
        median = (
            sorted_scores[mid]
            if len(sorted_scores) % 2
            else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
        )
        chase = venue_chasing[(venue, city)]

        rows.append(
            {
                "venue":                     venue,
                "city":                      city,
                "matches":                   len(scores),
                "venue_par_score":           round(sum(scores) / len(scores), 2),
                "median_first_innings_score": round(median, 2),
                "highest_first_innings_score": max(scores),
                "lowest_first_innings_score":  min(scores),
                "chasing_win_pct":           round(safe_rate(chase["chasing_wins"], chase["matches"], 100), 2),
            }
        )

    return sorted(rows, key=lambda r: r["venue_par_score"], reverse=True)


# ---------------------------------------------------------------------------
# Team Phase Strength
# ---------------------------------------------------------------------------

def build_team_phase_strength(deliveries):
    team_phase        = defaultdict(lambda: defaultdict(int))
    team_phase_innings = set()

    for row in deliveries:
        key = (row["batting_team"], row["phase"])
        if row["phase"] not in PHASES or not row["batting_team"]:
            continue
        team_phase[key]["runs"]    += as_int(row["total_runs"])
        team_phase[key]["balls"]   += as_int(row["is_legal_ball"])
        team_phase[key]["wickets"] += as_int(row["wicket_count"])
        team_phase_innings.add((row["batting_team"], row["phase"], row["match_id"], row["innings"]))

    phase_run_rates = defaultdict(list)
    raw_rows = []

    for (team, phase), stats in team_phase.items():
        innings_count = len(
            {
                (match_id, innings)
                for t, p, match_id, innings in team_phase_innings
                if t == team and p == phase
            }
        )
        if innings_count < 10:
            continue

        run_rate    = safe_rate(stats["runs"],    stats["balls"],   6)
        wicket_rate = safe_rate(stats["wickets"], innings_count)

        raw_rows.append(
            {
                "team":                     team,
                "phase":                    phase,
                "innings":                  innings_count,
                "runs":                     stats["runs"],
                "run_rate":                 round(run_rate, 2),
                "wickets_lost_per_innings": round(wicket_rate, 2),
                "_run_rate":                run_rate,
            }
        )
        phase_run_rates[phase].append(run_rate)

    output = []
    for row in raw_rows:
        strength = pct_rank(row["_run_rate"], phase_run_rates[row["phase"]])
        output.append(
            {
                "team":                       row["team"],
                "phase":                      row["phase"],
                "innings":                    row["innings"],
                "runs":                       row["runs"],
                "run_rate":                   row["run_rate"],
                "wickets_lost_per_innings":   row["wickets_lost_per_innings"],
                "team_phase_strength_score":  round(strength, 2),
            }
        )

    return sorted(output, key=lambda r: (r["phase"], -r["team_phase_strength_score"]))


# ---------------------------------------------------------------------------
# Death-Over Indexes
# ---------------------------------------------------------------------------

def build_death_indexes(deliveries):
    batting = defaultdict(lambda: defaultdict(int))
    bowling = defaultdict(lambda: defaultdict(int))

    for row in deliveries:
        if row["phase"] != "Death":
            continue
        batter = row["batter"]
        bowler = row["bowler"]

        batting[batter]["runs"]       += as_int(row["batter_runs"])
        batting[batter]["balls"]      += as_int(row["is_legal_ball"])
        batting[batter]["boundaries"] += int(as_int(row["batter_runs"]) in (4, 6))

        bowling[bowler]["wickets"] += as_int(row["bowler_wickets"])
        bowling[bowler]["balls"]   += as_int(row["is_legal_ball"])
        bowling[bowler]["runs"]    += (
            as_int(row["total_runs"])
            - as_int(row["byes"])
            - as_int(row["legbyes"])
            - as_int(row["penalty"])
        )
        bowling[bowler]["dots"] += int(as_int(row["total_runs"]) == 0)

    # --- batting index ---
    bat_raw = []
    for player, stats in batting.items():
        if player and stats["balls"] >= DEATH_INDEX_MIN_BALLS:
            strike_rate  = safe_rate(stats["runs"],       stats["balls"], 100)
            boundary_pct = safe_rate(stats["boundaries"], stats["balls"], 100)
            bat_raw.append(
                {
                    "player":               player,
                    "death_runs":           stats["runs"],
                    "death_balls":          stats["balls"],
                    "death_strike_rate":    round(strike_rate, 2),
                    "death_boundary_pct":   round(boundary_pct, 2),
                    "_strike_rate":         strike_rate,
                    "_boundary_pct":        boundary_pct,
                }
            )

    sr_values       = [r["_strike_rate"]  for r in bat_raw]
    boundary_values = [r["_boundary_pct"] for r in bat_raw]

    batting_index = []
    for row in bat_raw:
        index = (
            0.70 * pct_rank(row["_strike_rate"],  sr_values)
            + 0.30 * pct_rank(row["_boundary_pct"], boundary_values)
        )
        batting_index.append(
            {
                "player":             row["player"],
                "death_batting_index": round(index, 2),
                "death_runs":         row["death_runs"],
                "death_balls":        row["death_balls"],
                "death_strike_rate":  row["death_strike_rate"],
                "death_boundary_pct": row["death_boundary_pct"],
            }
        )

    # --- bowling index ---
    bowl_raw = []
    for player, stats in bowling.items():
        if player and stats["balls"] >= DEATH_INDEX_MIN_BALLS:
            economy    = safe_rate(stats["runs"],    stats["balls"], 6)
            wicket_rate = safe_rate(stats["wickets"], stats["balls"], 6)
            dot_pct    = safe_rate(stats["dots"],    stats["balls"], 100)
            bowl_raw.append(
                {
                    "player":                player,
                    "death_wickets":         stats["wickets"],
                    "death_balls":           stats["balls"],
                    "death_runs_conceded":   stats["runs"],
                    "death_economy":         round(economy, 2),
                    "death_wickets_per_over": round(wicket_rate, 3),
                    "death_dot_ball_pct":    round(dot_pct, 2),
                    "_economy":              economy,
                    "_wicket_rate":          wicket_rate,
                    "_dot_pct":              dot_pct,
                }
            )

    economy_values = [-r["_economy"]      for r in bowl_raw]
    wicket_values  = [r["_wicket_rate"]   for r in bowl_raw]
    dot_values     = [r["_dot_pct"]       for r in bowl_raw]

    bowling_index = []
    for row in bowl_raw:
        index = (
            0.45 * pct_rank(row["_wicket_rate"], wicket_values)
            + 0.35 * pct_rank(-row["_economy"],  economy_values)
            + 0.20 * pct_rank(row["_dot_pct"],   dot_values)
        )
        bowling_index.append(
            {
                "player":                 row["player"],
                "death_bowling_index":    round(index, 2),
                "death_wickets":          row["death_wickets"],
                "death_balls":            row["death_balls"],
                "death_runs_conceded":    row["death_runs_conceded"],
                "death_economy":          row["death_economy"],
                "death_wickets_per_over": row["death_wickets_per_over"],
                "death_dot_ball_pct":     row["death_dot_ball_pct"],
            }
        )

    return (
        sorted(batting_index, key=lambda r: r["death_batting_index"],  reverse=True),
        sorted(bowling_index, key=lambda r: r["death_bowling_index"],  reverse=True),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser(description="Build advanced IPL analytics metrics.")
#     parser.add_argument("--processed-dir", default="ipl_analytics_platform/data/processed")
#     parser.add_argument("--output-dir",    default="ipl_analytics_platform/reports/advanced_metrics")
#     args = parser.parse_args()

#     processed_dir = Path(args.processed_dir)
#     output_dir    = Path(args.output_dir)

#     print("Reading processed CSVs...")
#     matches    = read_csv(processed_dir / "matches.csv")
#     innings    = read_csv(processed_dir / "innings.csv")
#     deliveries = read_csv(processed_dir / "deliveries.csv")

#     print("Building player impact scores...")
#     player_impact = build_player_impact(deliveries)

#     print("Building venue par scores...")
#     venue_par = build_venue_par_score(matches, innings)

#     print("Building team phase strength...")
#     team_phase = build_team_phase_strength(deliveries)

#     print("Building death-over indexes...")
#     death_batting, death_bowling = build_death_indexes(deliveries)

#     write_csv(
#         output_dir / "player_impact_score.csv",
#         player_impact,
#         [
#             "player", "role_signal", "player_impact_score", "batting_score", "bowling_score",
#             "impact_eligible", "batting_runs", "batting_balls", "batting_strike_rate",
#             "batting_average", "boundary_pct", "bowling_wickets", "bowling_balls",
#             "bowling_economy", "wickets_per_over", "dot_ball_pct",
#         ],
#     )
#     write_csv(
#         output_dir / "venue_par_score.csv",
#         venue_par,
#         [
#             "venue", "city", "matches", "venue_par_score", "median_first_innings_score",
#             "highest_first_innings_score", "lowest_first_innings_score", "chasing_win_pct",
#         ],
#     )
#     write_csv(
#         output_dir / "team_phase_strength.csv",
#         team_phase,
#         ["team", "phase", "innings", "runs", "run_rate", "wickets_lost_per_innings", "team_phase_strength_score"],
#     )
#     write_csv(
#         output_dir / "death_over_batting_index.csv",
#         death_batting,
#         ["player", "death_batting_index", "death_runs", "death_balls", "death_strike_rate", "death_boundary_pct"],
#     )
#     write_csv(
#         output_dir / "death_over_bowling_index.csv",
#         death_bowling,
#         [
#             "player", "death_bowling_index", "death_wickets", "death_balls",
#             "death_runs_conceded", "death_economy", "death_wickets_per_over", "death_dot_ball_pct",
#         ],
#     )

#     print(f"\nAdvanced metrics written to : {output_dir.resolve()}")
#     print(f"Player impact rows          : {len(player_impact)}")
#     print(f"Venue par rows              : {len(venue_par)}")
#     print(f"Team phase rows             : {len(team_phase)}")
#     print(f"Death batting rows          : {len(death_batting)}")
#     print(f"Death bowling rows          : {len(death_bowling)}")


# if __name__ == "__main__":
#     main()
