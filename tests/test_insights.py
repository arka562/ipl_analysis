"""Unit tests for insights.py"""
import pytest

from insights import as_float, as_int, bottom_row, fmt_num, fmt_pct, top_row


class TestAsFloat:
    def test_valid_float(self):
        assert as_float("3.14") == pytest.approx(3.14)

    def test_valid_int_string(self):
        assert as_float("42") == 42.0

    def test_none(self):
        assert as_float(None) == 0.0

    def test_empty_string(self):
        assert as_float("") == 0.0

    def test_invalid_string(self):
        assert as_float("not_a_number") == 0.0

    def test_negative(self):
        assert as_float("-5.5") == -5.5


class TestAsInt:
    def test_valid_int_string(self):
        assert as_int("10") == 10

    def test_float_string_truncates(self):
        assert as_int("7.9") == 7

    def test_none(self):
        assert as_int(None) == 0

    def test_empty_string(self):
        assert as_int("") == 0

    def test_non_numeric(self):
        assert as_int("xyz") == 0


class TestTopRow:
    def test_finds_maximum(self):
        rows = [
            {"name": "A", "score": "30"},
            {"name": "B", "score": "50"},
            {"name": "C", "score": "20"},
        ]
        result = top_row(rows, "score")
        assert result["name"] == "B"

    def test_single_row(self):
        rows = [{"name": "Only", "val": "100"}]
        result = top_row(rows, "val")
        assert result["name"] == "Only"

    def test_ties_returns_one(self):
        rows = [
            {"name": "A", "score": "50"},
            {"name": "B", "score": "50"},
        ]
        result = top_row(rows, "score")
        assert result["name"] in ("A", "B")


class TestBottomRow:
    def test_finds_minimum(self):
        rows = [
            {"name": "A", "score": "30"},
            {"name": "B", "score": "10"},
            {"name": "C", "score": "20"},
        ]
        result = bottom_row(rows, "score")
        assert result["name"] == "B"

    def test_single_row(self):
        rows = [{"name": "Only", "val": "5"}]
        result = bottom_row(rows, "val")
        assert result["name"] == "Only"


class TestFmtPct:
    def test_formats_percentage(self):
        assert fmt_pct("55.5") == "55.5%"

    def test_rounds_to_one_decimal(self):
        assert fmt_pct("33.333") == "33.3%"

    def test_integer_value(self):
        assert fmt_pct("100") == "100.0%"

    def test_none_value(self):
        assert fmt_pct(None) == "0.0%"


class TestFmtNum:
    def test_integer_result(self):
        assert fmt_num("150") == "150"

    def test_decimal_result(self):
        assert fmt_num("7.25") == "7.25"

    def test_trailing_zeros_stripped(self):
        assert fmt_num("8.50") == "8.5"

    def test_large_number_with_comma(self):
        assert fmt_num("1500") == "1,500"

    def test_none_value(self):
        assert fmt_num(None) == "0"
