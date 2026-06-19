import argparse
from pathlib import Path

import duckdb


TABLES = {
    "matches": "matches.csv",
    "innings": "innings.csv",
    "deliveries": "deliveries.csv",
    "match_players": "match_players.csv",
    "batting_summary": "batting_summary.csv",
    "bowling_summary": "bowling_summary.csv",
    "phase_summary": "phase_summary.csv",
}


REPORT_QUERIES = {
    "season_summary": """
        select
            m.season,
            count(distinct m.match_id)                                              as matches,
            count(distinct d.batter)                                                as batters_used,
            count(distinct d.bowler)                                                as bowlers_used,
            sum(d.total_runs)                                                       as total_runs,
            sum(d.wicket_count)                                                     as total_wickets,
            round(sum(d.total_runs) * 6.0 / nullif(sum(d.is_legal_ball), 0), 2)    as run_rate,
            round(avg(i.runs), 2)                                                   as avg_innings_score
        from matches m
        left join deliveries d on m.match_id = d.match_id
        left join innings i    on m.match_id = i.match_id
        group by m.season
        order by m.season
    """,

    "toss_impact_by_season": """
        select
            season,
            count(*)                                                                            as completed_matches,
            sum(case when toss_winner = winner then 1 else 0 end)                              as toss_winner_wins,
            round(
                100.0 * sum(case when toss_winner = winner then 1 else 0 end) / count(*), 2
            )                                                                                   as toss_winner_win_pct
        from matches
        where winner is not null and winner <> ''
          and toss_winner is not null and toss_winner <> ''
        group by season
        order by season
    """,

    "chasing_vs_defending": """
        with innings_order as (
            select
                match_id,
                max(case when innings = 1 then batting_team end) as first_batting_team,
                max(case when innings = 2 then batting_team end) as chasing_team
            from innings
            group by match_id
        )
        select
            m.season,
            count(*)                                                                            as completed_matches,
            sum(case when m.winner = io.chasing_team       then 1 else 0 end)                  as chasing_wins,
            sum(case when m.winner = io.first_batting_team then 1 else 0 end)                  as defending_wins,
            round(
                100.0 * sum(case when m.winner = io.chasing_team then 1 else 0 end) / count(*), 2
            )                                                                                   as chasing_win_pct
        from matches m
        join innings_order io on m.match_id = io.match_id
        where m.winner is not null and m.winner <> ''
          and io.chasing_team is not null
        group by m.season
        order by m.season
    """,

    "venue_summary": """
        with first_innings as (
            select match_id, runs as first_innings_runs
            from innings
            where innings = 1
        )
        select
            m.venue,
            m.city,
            count(*)                                                                            as matches,
            round(avg(fi.first_innings_runs), 2)                                               as avg_first_innings_score,
            round(
                100.0 * sum(case when m.winner = m.toss_winner then 1 else 0 end) / count(*), 2
            )                                                                                   as toss_winner_win_pct
        from matches m
        join first_innings fi on m.match_id = fi.match_id
        where m.winner is not null and m.winner <> ''
        group by m.venue, m.city
        having count(*) >= 5
        order by avg_first_innings_score desc, matches desc
    """,

    "team_summary": """
        with team_matches as (
            select match_id, season, team_1 as team, winner from matches
            union all
            select match_id, season, team_2 as team, winner from matches
        )
        select
            team,
            count(*)                                                                            as matches,
            sum(case when team = winner then 1 else 0 end)                                     as wins,
            count(*) - sum(case when team = winner then 1 else 0 end)                          as losses_or_no_results,
            round(
                100.0 * sum(case when team = winner then 1 else 0 end) / count(*), 2
            )                                                                                   as win_pct
        from team_matches
        where team is not null and team <> ''
        group by team
        order by win_pct desc, wins desc
    """,

    "top_batters_by_phase": """
        select
            phase,
            batter,
            sum(batter_runs)                                                                    as runs,
            sum(is_legal_ball)                                                                  as balls,
            round(100.0 * sum(batter_runs) / nullif(sum(is_legal_ball), 0), 2)                 as strike_rate
        from deliveries
        where batter is not null and batter <> ''
          and phase in ('Powerplay', 'Middle', 'Death')
        group by phase, batter
        qualify row_number() over (partition by phase order by sum(batter_runs) desc) <= 10
        order by phase, runs desc
    """,

    "top_death_bowlers": """
        select
            bowler,
            sum(bowler_wickets)                                                                 as wickets,
            sum(is_legal_ball)                                                                  as balls,
            sum(total_runs - byes - legbyes - penalty)                                         as runs_conceded,
            round(
                sum(total_runs - byes - legbyes - penalty) * 6.0 / nullif(sum(is_legal_ball), 0), 2
            )                                                                                   as economy
        from deliveries
        where phase = 'Death'
          and bowler is not null and bowler <> ''
        group by bowler
        having sum(is_legal_ball) >= 120
        order by wickets desc, economy asc
        limit 25
    """,
}

def sql_path(path):
    return str(path).replace("\\", "/").replace("'", "''")


def load_tables(connection, processed_dir):
    for table_name, file_name in TABLES.items():
        csv_path = processed_dir / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required input file: {csv_path}")
        connection.execute(f"drop table if exists {table_name}")
        connection.execute(
            f"create table {table_name} as "
            f"select * from read_csv_auto('{sql_path(csv_path)}', header=true)"
        )


def export_reports(connection, reports_dir):
    reports_dir.mkdir(parents=True, exist_ok=True)
    for report_name, query in REPORT_QUERIES.items():
        output_path = reports_dir / f"{report_name}.csv"
        connection.execute(
            f"copy ({query}) to '{sql_path(output_path)}' with (header true, delimiter ',')"
        )
        count = connection.execute(f"select count(*) from ({query})").fetchone()[0]
        print(f"  {report_name}.csv — {count} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Load processed CSVs into DuckDB and generate SQL report CSVs."
    )
    parser.add_argument("--processed-dir", default="ipl_analytics_platform/data/processed")
    parser.add_argument("--warehouse-path", default="ipl_analytics_platform/data/warehouse/ipl.duckdb")
    parser.add_argument("--reports-dir", default="ipl_analytics_platform/reports/tables")
    args = parser.parse_args()

    warehouse_path = Path(args.warehouse_path)
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)

    # Always rebuild clean
    if warehouse_path.exists():
        warehouse_path.unlink()

    connection = duckdb.connect(str(warehouse_path))

    try:
        print("Loading tables into DuckDB...")
        load_tables(connection, Path(args.processed_dir))

        for table_name in TABLES:
            count = connection.execute(f"select count(*) from {table_name}").fetchone()[0]
            print(f"  {table_name}: {count} rows")

        print("\nGenerating reports...")
        export_reports(connection, Path(args.reports_dir))

        counts = {
            table_name: connection.execute(f"select count(*) from {table_name}").fetchone()[0]
            for table_name in TABLES
        }
    finally:
        connection.close()

    print(f"\nWarehouse : {warehouse_path.resolve()}")
    print(f"Reports   : {Path(args.reports_dir).resolve()}")


if __name__ == "__main__":
    main()
