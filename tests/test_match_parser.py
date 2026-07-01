"""Unit tests for parsers/match_parser.py"""
import json
import tempfile
from pathlib import Path

import pytest

from parsers.match_parser import (
    BOWLER_CREDITED_WICKETS,
    build_summary_tables,
    first_or_blank,
    parse_match,
    phase_for_over,
    season_from_info,
    write_csv,
)


class TestFirstOrBlank:
    def test_returns_first_element(self):
        assert first_or_blank(["2024-03-22", "2024-03-23"]) == "2024-03-22"

    def test_returns_blank_for_empty(self):
        assert first_or_blank([]) == ""

    def test_single_element(self):
        assert first_or_blank(["only"]) == "only"


class TestSeasonFromInfo:
    def test_integer_season(self):
        assert season_from_info({"season": 2023}) == 2023

    def test_string_season_four_digits(self):
        assert season_from_info({"season": "2022/23"}) == 2022

    def test_fallback_to_dates(self):
        assert season_from_info({"dates": ["2021-04-09"]}) == 2021

    def test_no_season_no_dates(self):
        assert season_from_info({}) is None

    def test_non_numeric_season_string(self):
        assert season_from_info({"season": "unknown", "dates": ["2020-01-01"]}) == 2020

    def test_non_numeric_season_no_dates(self):
        assert season_from_info({"season": "unknown"}) is None


class TestPhaseForOver:
    def test_powerplay(self):
        for over in range(0, 6):
            assert phase_for_over(over) == "Powerplay"

    def test_middle(self):
        for over in range(6, 16):
            assert phase_for_over(over) == "Middle"

    def test_death(self):
        for over in range(16, 20):
            assert phase_for_over(over) == "Death"

    def test_other(self):
        assert phase_for_over(20) == "Other"
        assert phase_for_over(-1) == "Other"


class TestWriteCsv:
    def test_creates_csv_file(self, tmp_path):
        output = tmp_path / "subdir" / "output.csv"
        rows = [{"name": "Virat", "runs": 100}, {"name": "Rohit", "runs": 80}]
        write_csv(output, rows, ["name", "runs"])

        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert lines[0] == "name,runs"
        assert lines[1] == "Virat,100"
        assert lines[2] == "Rohit,80"

    def test_creates_parent_directories(self, tmp_path):
        output = tmp_path / "a" / "b" / "c" / "test.csv"
        write_csv(output, [{"x": 1}], ["x"])
        assert output.exists()


class TestParseMatch:
    @pytest.fixture
    def sample_match_json(self, tmp_path):
        data = {
            "info": {
                "season": 2023,
                "dates": ["2023-04-01"],
                "event": {"name": "Indian Premier League", "match_number": "1"},
                "city": "Chennai",
                "venue": "MA Chidambaram Stadium",
                "teams": ["Chennai Super Kings", "Mumbai Indians"],
                "outcome": {"winner": "Chennai Super Kings", "by": {"runs": 10}},
                "toss": {"winner": "Chennai Super Kings", "decision": "bat"},
                "player_of_match": ["MS Dhoni"],
                "players": {
                    "Chennai Super Kings": ["MS Dhoni", "Ruturaj Gaikwad"],
                    "Mumbai Indians": ["Rohit Sharma", "Jasprit Bumrah"],
                },
                "balls_per_over": 6,
            },
            "innings": [
                {
                    "team": "Chennai Super Kings",
                    "overs": [
                        {
                            "over": 0,
                            "deliveries": [
                                {
                                    "batter": "Ruturaj Gaikwad",
                                    "non_striker": "MS Dhoni",
                                    "bowler": "Jasprit Bumrah",
                                    "runs": {"batter": 4, "extras": 0, "total": 4},
                                    "extras": {},
                                    "wickets": [],
                                },
                                {
                                    "batter": "Ruturaj Gaikwad",
                                    "non_striker": "MS Dhoni",
                                    "bowler": "Jasprit Bumrah",
                                    "runs": {"batter": 0, "extras": 1, "total": 1},
                                    "extras": {"wides": 1},
                                    "wickets": [],
                                },
                                {
                                    "batter": "Ruturaj Gaikwad",
                                    "non_striker": "MS Dhoni",
                                    "bowler": "Jasprit Bumrah",
                                    "runs": {"batter": 0, "extras": 0, "total": 0},
                                    "extras": {},
                                    "wickets": [
                                        {
                                            "player_out": "Ruturaj Gaikwad",
                                            "kind": "bowled",
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                },
                {
                    "team": "Mumbai Indians",
                    "overs": [
                        {
                            "over": 0,
                            "deliveries": [
                                {
                                    "batter": "Rohit Sharma",
                                    "non_striker": "Jasprit Bumrah",
                                    "bowler": "MS Dhoni",
                                    "runs": {"batter": 6, "extras": 0, "total": 6},
                                    "extras": {},
                                    "wickets": [],
                                },
                            ],
                        }
                    ],
                },
            ],
        }
        path = tmp_path / "123456.json"
        path.write_text(json.dumps(data))
        return path

    def test_parse_match_returns_match_metadata(self, sample_match_json):
        match, innings, deliveries, players = parse_match(sample_match_json)

        assert match["match_id"] == "123456"
        assert match["season"] == 2023
        assert match["city"] == "Chennai"
        assert match["venue"] == "MA Chidambaram Stadium"
        assert match["team_1"] == "Chennai Super Kings"
        assert match["team_2"] == "Mumbai Indians"
        assert match["winner"] == "Chennai Super Kings"
        assert match["win_by_runs"] == 10
        assert match["toss_winner"] == "Chennai Super Kings"
        assert match["toss_decision"] == "bat"
        assert match["player_of_match"] == "MS Dhoni"

    def test_parse_match_returns_innings(self, sample_match_json):
        match, innings, deliveries, players = parse_match(sample_match_json)

        assert len(innings) == 2
        assert innings[0]["innings"] == 1
        assert innings[0]["batting_team"] == "Chennai Super Kings"
        assert innings[0]["runs"] == 5  # 4 + 1(wide) + 0
        assert innings[0]["wickets"] == 1
        assert innings[1]["innings"] == 2
        assert innings[1]["batting_team"] == "Mumbai Indians"
        assert innings[1]["runs"] == 6

    def test_parse_match_returns_deliveries(self, sample_match_json):
        match, innings, deliveries, players = parse_match(sample_match_json)

        assert len(deliveries) == 4  # 3 from first innings + 1 from second
        assert deliveries[0]["batter"] == "Ruturaj Gaikwad"
        assert deliveries[0]["batter_runs"] == 4
        assert deliveries[0]["phase"] == "Powerplay"
        # Wide delivery
        assert deliveries[1]["wides"] == 1
        assert deliveries[1]["is_legal_ball"] == 0
        # Wicket delivery
        assert deliveries[2]["wicket_count"] == 1
        assert deliveries[2]["bowler_wickets"] == 1
        assert deliveries[2]["wicket_kinds"] == "bowled"
        assert deliveries[2]["players_out"] == "Ruturaj Gaikwad"

    def test_parse_match_returns_players(self, sample_match_json):
        match, innings, deliveries, players = parse_match(sample_match_json)

        assert len(players) == 4
        teams = {p["team"] for p in players}
        assert teams == {"Chennai Super Kings", "Mumbai Indians"}

    def test_batting_result_assignment(self, sample_match_json):
        match, innings, deliveries, players = parse_match(sample_match_json)

        assert innings[0]["batting_result"] == "Winner"
        assert innings[1]["batting_result"] == "Loser"


class TestBuildSummaryTables:
    def test_basic_summary(self):
        deliveries = [
            {
                "match_id": "1",
                "innings": 1,
                "batter": "PlayerA",
                "bowler": "PlayerB",
                "batter_runs": 4,
                "bowler_wickets": 0,
                "is_legal_ball": 1,
                "total_runs": 4,
                "byes": 0,
                "legbyes": 0,
                "penalty": 0,
                "batting_result": "Winner",
                "phase": "Powerplay",
            },
            {
                "match_id": "1",
                "innings": 1,
                "batter": "PlayerA",
                "bowler": "PlayerB",
                "batter_runs": 6,
                "bowler_wickets": 0,
                "is_legal_ball": 1,
                "total_runs": 6,
                "byes": 0,
                "legbyes": 0,
                "penalty": 0,
                "batting_result": "Winner",
                "phase": "Powerplay",
            },
            {
                "match_id": "1",
                "innings": 1,
                "batter": "PlayerA",
                "bowler": "PlayerB",
                "batter_runs": 0,
                "bowler_wickets": 1,
                "is_legal_ball": 1,
                "total_runs": 0,
                "byes": 0,
                "legbyes": 0,
                "penalty": 0,
                "batting_result": "Winner",
                "phase": "Powerplay",
            },
        ]
        batting, bowling, phase = build_summary_tables(deliveries)

        assert len(batting) == 1
        assert batting[0]["player"] == "PlayerA"
        assert batting[0]["runs"] == 10

        assert len(bowling) == 1
        assert bowling[0]["player"] == "PlayerB"
        assert bowling[0]["wickets"] == 1
        assert bowling[0]["balls"] == 3
        assert bowling[0]["runs_conceded"] == 10

    def test_empty_players_excluded(self):
        deliveries = [
            {
                "match_id": "1",
                "innings": 1,
                "batter": "",
                "bowler": "",
                "batter_runs": 4,
                "bowler_wickets": 0,
                "is_legal_ball": 1,
                "total_runs": 4,
                "byes": 0,
                "legbyes": 0,
                "penalty": 0,
                "batting_result": "No result",
                "phase": "Powerplay",
            },
        ]
        batting, bowling, phase = build_summary_tables(deliveries)
        assert len(batting) == 0
        assert len(bowling) == 0


class TestBowlerCreditedWickets:
    def test_contains_expected_types(self):
        assert "bowled" in BOWLER_CREDITED_WICKETS
        assert "caught" in BOWLER_CREDITED_WICKETS
        assert "lbw" in BOWLER_CREDITED_WICKETS
        assert "stumped" in BOWLER_CREDITED_WICKETS

    def test_run_out_not_credited(self):
        assert "run out" not in BOWLER_CREDITED_WICKETS
