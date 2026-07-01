"""Unit tests for win_probablity_curve.py"""
import pytest
import pandas as pd

from win_probablity_curve import choose_match, scale_x, scale_y


class TestScaleX:
    def test_first_over(self):
        result = scale_x(1, left=90, chart_width=920)
        assert result == pytest.approx(90.0)

    def test_last_over(self):
        result = scale_x(20, left=90, chart_width=920)
        assert result == pytest.approx(90 + 19 * 920 / 19)
        assert result == pytest.approx(1010.0)

    def test_middle_over(self):
        result = scale_x(10, left=90, chart_width=920)
        expected = 90 + 9 * 920 / 19
        assert result == pytest.approx(expected)

    def test_with_different_params(self):
        result = scale_x(1, left=0, chart_width=190)
        assert result == pytest.approx(0.0)


class TestScaleY:
    def test_zero_probability(self):
        result = scale_y(0, top=90, chart_height=440)
        assert result == pytest.approx(90 + 100 * 440 / 100)
        assert result == pytest.approx(530.0)

    def test_hundred_probability(self):
        result = scale_y(100, top=90, chart_height=440)
        assert result == pytest.approx(90.0)

    def test_fifty_probability(self):
        result = scale_y(50, top=90, chart_height=440)
        assert result == pytest.approx(90 + 50 * 440 / 100)
        assert result == pytest.approx(310.0)


class TestChooseMatch:
    def test_selects_match_with_highest_swing(self):
        data = {
            "match_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "innings": [1, 1, 2, 2, 1, 1, 2, 2],
            "over_number": [5, 10, 5, 10, 5, 10, 5, 10],
            "win_probability": [60, 65, 40, 80, 50, 55, 45, 50],
        }
        frame = pd.DataFrame(data)
        result = choose_match(frame)
        # Match 1 has swing of 80-40=40, Match 2 has swing of 50-45=5
        assert str(result) == "1"

    def test_single_innings_match_skipped(self):
        data = {
            "match_id": [1, 1, 2, 2, 2, 2],
            "innings": [1, 1, 1, 1, 2, 2],
            "over_number": [5, 10, 5, 10, 5, 10],
            "win_probability": [90, 10, 50, 55, 45, 50],
        }
        frame = pd.DataFrame(data)
        result = choose_match(frame)
        # Match 1 only has innings 1 so it's skipped; match 2 is selected
        assert str(result) == "2"

    def test_fallback_when_no_valid_matches(self):
        data = {
            "match_id": [1, 1],
            "innings": [1, 1],
            "over_number": [5, 10],
            "win_probability": [50, 60],
        }
        frame = pd.DataFrame(data)
        result = choose_match(frame)
        # Falls back to first match_id
        assert str(result) == "1"
