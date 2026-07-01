"""Unit tests for warehouse_and_reports.py"""
import pytest

from warehouse_and_reports import REPORT_QUERIES, TABLES, sql_path


class TestSqlPath:
    def test_forward_slashes_preserved(self):
        assert sql_path("/home/user/data/file.csv") == "/home/user/data/file.csv"

    def test_backslashes_converted(self):
        assert sql_path("C:\\Users\\data\\file.csv") == "C:/Users/data/file.csv"

    def test_single_quotes_escaped(self):
        assert sql_path("path/with'quote.csv") == "path/with''quote.csv"

    def test_mixed_backslash_and_quote(self):
        assert sql_path("C:\\path\\it's.csv") == "C:/path/it''s.csv"

    def test_empty_string(self):
        assert sql_path("") == ""


class TestTablesConfig:
    def test_all_tables_have_csv_filenames(self):
        for table_name, file_name in TABLES.items():
            assert file_name.endswith(".csv")
            assert len(table_name) > 0

    def test_expected_tables_present(self):
        assert "matches" in TABLES
        assert "innings" in TABLES
        assert "deliveries" in TABLES
        assert "match_players" in TABLES
        assert "batting_summary" in TABLES
        assert "bowling_summary" in TABLES
        assert "phase_summary" in TABLES


class TestReportQueries:
    def test_all_queries_are_strings(self):
        for name, query in REPORT_QUERIES.items():
            assert isinstance(query, str)
            assert len(query.strip()) > 0

    def test_expected_reports_present(self):
        expected = [
            "season_summary",
            "toss_impact_by_season",
            "chasing_vs_defending",
            "venue_summary",
            "team_summary",
            "top_batters_by_phase",
            "top_death_bowlers",
        ]
        for name in expected:
            assert name in REPORT_QUERIES

    def test_queries_contain_select(self):
        for name, query in REPORT_QUERIES.items():
            assert "select" in query.lower(), f"{name} query missing SELECT"
