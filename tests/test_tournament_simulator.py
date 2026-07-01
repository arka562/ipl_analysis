"""Unit tests for tournament_simulator.py"""
import pytest

from tournament_simulator import (
    TEAM_MAPPINGS,
    TEAMS_AND_VENUES,
    generate_schedule,
)


class TestTeamsAndVenues:
    def test_ten_teams(self):
        assert len(TEAMS_AND_VENUES) == 10

    def test_all_teams_have_venues(self):
        for team, venue in TEAMS_AND_VENUES.items():
            assert isinstance(team, str)
            assert isinstance(venue, str)
            assert len(venue) > 0

    def test_known_teams_present(self):
        teams = set(TEAMS_AND_VENUES.keys())
        assert "Chennai Super Kings" in teams
        assert "Mumbai Indians" in teams
        assert "Royal Challengers Bengaluru" in teams
        assert "Kolkata Knight Riders" in teams
        assert "Rajasthan Royals" in teams


class TestTeamMappings:
    def test_mappings_target_current_teams(self):
        current_teams = set(TEAMS_AND_VENUES.keys())
        for old_name, new_name in TEAM_MAPPINGS.items():
            assert new_name in current_teams, (
                f"Mapping {old_name} -> {new_name} targets a non-current team"
            )

    def test_known_mappings(self):
        assert TEAM_MAPPINGS["Royal Challengers Bangalore"] == "Royal Challengers Bengaluru"
        assert TEAM_MAPPINGS["Kings XI Punjab"] == "Punjab Kings"
        assert TEAM_MAPPINGS["Delhi Daredevils"] == "Delhi Capitals"

    def test_no_circular_mappings(self):
        for old_name in TEAM_MAPPINGS:
            assert old_name not in TEAMS_AND_VENUES, (
                f"{old_name} is both a mapping source and a current team"
            )


class TestGenerateSchedule:
    def test_schedule_length(self):
        schedule = generate_schedule()
        # Double round-robin: 10 teams, each plays every other twice = 10 * 9 = 90
        assert len(schedule) == 90

    def test_schedule_format(self):
        schedule = generate_schedule()
        for match in schedule:
            assert len(match) == 3
            home, away, venue = match
            assert home != away
            assert home in TEAMS_AND_VENUES
            assert away in TEAMS_AND_VENUES
            assert venue == TEAMS_AND_VENUES[home]

    def test_each_team_plays_correct_number_of_matches(self):
        schedule = generate_schedule()
        teams = list(TEAMS_AND_VENUES.keys())

        for team in teams:
            home_matches = sum(1 for h, a, v in schedule if h == team)
            away_matches = sum(1 for h, a, v in schedule if a == team)
            assert home_matches == 9  # plays 9 other teams at home
            assert away_matches == 9  # plays 9 other teams away

    def test_no_self_matches(self):
        schedule = generate_schedule()
        for home, away, _ in schedule:
            assert home != away

    def test_all_pairings_covered(self):
        schedule = generate_schedule()
        teams = list(TEAMS_AND_VENUES.keys())

        for i, t1 in enumerate(teams):
            for j, t2 in enumerate(teams):
                if i == j:
                    continue
                matches = [(h, a) for h, a, v in schedule if h == t1 and a == t2]
                assert len(matches) == 1, f"Expected exactly one match {t1} vs {t2} at home"
