"""Unit tests for advanced_metrices.py"""
import pytest

from advanced_metrices import (
    DEATH_INDEX_MIN_BALLS,
    PHASES,
    PLAYER_IMPACT_MIN_BATTING_BALLS,
    PLAYER_IMPACT_MIN_BOWLING_BALLS,
    as_float,
    as_int,
    build_death_indexes,
    build_match_lookup,
    build_player_impact,
    build_team_phase_strength,
    build_venue_par_score,
    pct_rank,
    safe_rate,
)


class TestAsInt:
    def test_integer_string(self):
        assert as_int("42") == 42

    def test_float_string(self):
        assert as_int("3.7") == 3

    def test_none(self):
        assert as_int(None) == 0

    def test_empty_string(self):
        assert as_int("") == 0

    def test_non_numeric(self):
        assert as_int("abc") == 0

    def test_actual_int(self):
        assert as_int(10) == 10

    def test_actual_float(self):
        assert as_int(9.9) == 9


class TestAsFloat:
    def test_float_string(self):
        assert as_float("3.14") == pytest.approx(3.14)

    def test_integer_string(self):
        assert as_float("7") == 7.0

    def test_none(self):
        assert as_float(None) == 0.0

    def test_empty_string(self):
        assert as_float("") == 0.0

    def test_non_numeric(self):
        assert as_float("abc") == 0.0


class TestPctRank:
    def test_basic_ranking(self):
        values = [10, 20, 30, 40, 50]
        assert pct_rank(30, values) == pytest.approx(60.0)

    def test_lowest_value(self):
        values = [10, 20, 30]
        assert pct_rank(10, values) == pytest.approx(100.0 / 3)

    def test_highest_value(self):
        values = [10, 20, 30]
        assert pct_rank(30, values) == pytest.approx(100.0)

    def test_empty_values(self):
        assert pct_rank(10, []) == 0.0

    def test_all_same_values(self):
        values = [5, 5, 5, 5]
        assert pct_rank(5, values) == 100.0

    def test_value_below_all(self):
        values = [10, 20, 30]
        assert pct_rank(5, values) == pytest.approx(0.0)


class TestSafeRate:
    def test_basic_division(self):
        assert safe_rate(10, 5) == 2.0

    def test_with_multiplier(self):
        assert safe_rate(10, 5, 100) == 200.0

    def test_zero_denominator(self):
        assert safe_rate(10, 0) == 0.0

    def test_zero_numerator(self):
        assert safe_rate(0, 5) == 0.0

    def test_multiplier_six_for_economy(self):
        assert safe_rate(30, 24, 6) == pytest.approx(7.5)


class TestBuildMatchLookup:
    def test_creates_lookup_by_match_id(self):
        matches = [
            {"match_id": "100", "venue": "Eden Gardens"},
            {"match_id": "200", "venue": "Wankhede"},
        ]
        lookup = build_match_lookup(matches)
        assert lookup["100"]["venue"] == "Eden Gardens"
        assert lookup["200"]["venue"] == "Wankhede"

    def test_empty_list(self):
        assert build_match_lookup([]) == {}


class TestBuildPlayerImpact:
    def _make_delivery(self, batter, bowler, batter_runs, bowler_wickets,
                       is_legal_ball=1, total_runs=None, byes=0, legbyes=0,
                       penalty=0, players_out=""):
        if total_runs is None:
            total_runs = batter_runs
        return {
            "batter": batter,
            "bowler": bowler,
            "batter_runs": str(batter_runs),
            "bowler_wickets": str(bowler_wickets),
            "is_legal_ball": str(is_legal_ball),
            "total_runs": str(total_runs),
            "byes": str(byes),
            "legbyes": str(legbyes),
            "penalty": str(penalty),
            "players_out": players_out,
        }

    def test_basic_impact_calculation(self):
        deliveries = [
            self._make_delivery("BatA", "BowlA", 4, 0)
            for _ in range(10)
        ]
        result = build_player_impact(deliveries)
        assert len(result) == 2
        players = {r["player"] for r in result}
        assert players == {"BatA", "BowlA"}

    def test_empty_player_excluded(self):
        deliveries = [self._make_delivery("", "", 0, 0)]
        result = build_player_impact(deliveries)
        assert len(result) == 0

    def test_ineligible_players_have_zero_scores(self):
        deliveries = [
            self._make_delivery("BatA", "BowlA", 4, 0)
            for _ in range(5)
        ]
        result = build_player_impact(deliveries)
        for row in result:
            assert row["batting_score"] == 0.0
            assert row["bowling_score"] == 0.0
            assert row["impact_eligible"] == 0


class TestBuildVenueParScore:
    def test_venue_with_enough_matches(self):
        matches = [
            {"match_id": str(i), "venue": "Eden Gardens", "city": "Kolkata", "winner": "Team A"}
            for i in range(6)
        ]
        innings = [
            {"match_id": str(i), "innings": "1", "runs": str(150 + i * 10), "batting_team": "Team A"}
            for i in range(6)
        ]
        # Add second innings for chasing computation
        innings += [
            {"match_id": str(i), "innings": "2", "runs": str(140), "batting_team": "Team B"}
            for i in range(6)
        ]
        result = build_venue_par_score(matches, innings)
        assert len(result) == 1
        assert result[0]["venue"] == "Eden Gardens"
        assert result[0]["matches"] == 6

    def test_venue_with_too_few_matches_excluded(self):
        matches = [
            {"match_id": str(i), "venue": "Small Ground", "city": "Town", "winner": "Team A"}
            for i in range(4)
        ]
        innings = [
            {"match_id": str(i), "innings": "1", "runs": "150", "batting_team": "Team A"}
            for i in range(4)
        ]
        result = build_venue_par_score(matches, innings)
        assert len(result) == 0


class TestBuildTeamPhaseStrength:
    def _make_delivery(self, team, phase, total_runs, wicket_count, match_id, innings):
        return {
            "batting_team": team,
            "phase": phase,
            "total_runs": str(total_runs),
            "is_legal_ball": "1",
            "wicket_count": str(wicket_count),
            "match_id": match_id,
            "innings": str(innings),
        }

    def test_team_with_enough_innings(self):
        deliveries = []
        for i in range(12):
            for _ in range(30):
                deliveries.append(
                    self._make_delivery("CSK", "Powerplay", 1, 0, str(i), 1)
                )
        result = build_team_phase_strength(deliveries)
        assert len(result) >= 1
        assert result[0]["team"] == "CSK"
        assert result[0]["phase"] == "Powerplay"

    def test_team_with_few_innings_excluded(self):
        deliveries = []
        for i in range(5):
            deliveries.append(
                self._make_delivery("Small", "Powerplay", 6, 0, str(i), 1)
            )
        result = build_team_phase_strength(deliveries)
        assert len(result) == 0


class TestBuildDeathIndexes:
    def _make_delivery(self, batter, bowler, batter_runs, bowler_wickets,
                       total_runs=None, phase="Death"):
        if total_runs is None:
            total_runs = batter_runs
        return {
            "batter": batter,
            "bowler": bowler,
            "batter_runs": str(batter_runs),
            "bowler_wickets": str(bowler_wickets),
            "is_legal_ball": "1",
            "total_runs": str(total_runs),
            "byes": "0",
            "legbyes": "0",
            "penalty": "0",
            "phase": phase,
        }

    def test_death_deliveries_only(self):
        deliveries = [
            self._make_delivery("BatA", "BowlA", 4, 0, phase="Powerplay")
            for _ in range(200)
        ]
        batting_index, bowling_index = build_death_indexes(deliveries)
        assert len(batting_index) == 0
        assert len(bowling_index) == 0

    def test_below_minimum_balls_excluded(self):
        deliveries = [
            self._make_delivery("BatA", "BowlA", 4, 0)
            for _ in range(DEATH_INDEX_MIN_BALLS - 1)
        ]
        batting_index, bowling_index = build_death_indexes(deliveries)
        assert len(batting_index) == 0
        assert len(bowling_index) == 0

    def test_at_minimum_balls_included(self):
        deliveries = [
            self._make_delivery("BatA", "BowlA", 2, 0)
            for _ in range(DEATH_INDEX_MIN_BALLS)
        ]
        batting_index, bowling_index = build_death_indexes(deliveries)
        assert len(batting_index) == 1
        assert batting_index[0]["player"] == "BatA"
        assert len(bowling_index) == 1
        assert bowling_index[0]["player"] == "BowlA"

    def test_sorted_by_index_descending(self):
        deliveries = []
        # PlayerA: high strike rate
        for _ in range(DEATH_INDEX_MIN_BALLS):
            deliveries.append(self._make_delivery("HighSR", "Bowl1", 6, 0))
        # PlayerB: low strike rate
        for _ in range(DEATH_INDEX_MIN_BALLS):
            deliveries.append(self._make_delivery("LowSR", "Bowl2", 1, 0))

        batting_index, _ = build_death_indexes(deliveries)
        assert batting_index[0]["player"] == "HighSR"
        assert batting_index[1]["player"] == "LowSR"


class TestConstants:
    def test_phases_tuple(self):
        assert PHASES == ("Powerplay", "Middle", "Death")

    def test_min_balls_thresholds(self):
        assert PLAYER_IMPACT_MIN_BATTING_BALLS == 600
        assert PLAYER_IMPACT_MIN_BOWLING_BALLS == 600
        assert DEATH_INDEX_MIN_BALLS == 120
