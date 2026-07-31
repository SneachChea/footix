"""Tests for data_io utility functions."""

from __future__ import annotations

import pandas as pd
import pytest

from footix.data_io.utils_scrapper import (
    MAPPING_COMPETITIONS,
    add_match_id,
    to_snake_case,
)


def test_to_snake_case_camel():
    assert to_snake_case("HomeTeam") == "home_team"


def test_to_snake_case_already_snake():
    assert to_snake_case("home_team") == "home_team"


def test_to_snake_case_consecutive_caps():
    assert to_snake_case("FTAG") == "ftag"


def test_to_snake_case_single_word():
    assert to_snake_case("Date") == "date"


def test_add_match_id():
    df = pd.DataFrame(
        {
            "home_team": ["Team A"],
            "away_team": ["Team B"],
            "date": ["01/02/2023"],
        }
    )
    result = add_match_id(df)
    assert "match_id" in result.columns
    assert result["match_id"].iloc[0] == "Team A - Team B - 2023-02-01"


def test_add_match_id_parses_dayfirst():
    df = pd.DataFrame(
        {
            "home_team": ["X"],
            "away_team": ["Y"],
            "date": ["15/03/2023"],
        }
    )
    result = add_match_id(df)
    assert result["match_id"].iloc[0] == "X - Y - 2023-03-15"


def test_add_match_id_missing_columns_raises():
    df = pd.DataFrame({"home_team": ["A"]})
    with pytest.raises(ValueError, match="away_team.*date"):
        add_match_id(df)


def test_mapping_competitions_has_expected_keys():
    assert "FRA Ligue 1" in MAPPING_COMPETITIONS
    assert "ENG Premier League" in MAPPING_COMPETITIONS
    assert "DEU Bundesliga 1" in MAPPING_COMPETITIONS
    assert isinstance(MAPPING_COMPETITIONS["FRA Ligue 1"], dict)


def test_football_data_org_codes():
    expected_codes = {
        "DEU Bundesliga 1": "BL1",
        "SPA La Liga": "PD",
        "FRA Ligue 1": "FL1",
        "ENG Championship": "ELC",
        "ITA Serie A": "SA",
        "ENG Premier League": "PL",
    }
    assert {
        competition: MAPPING_COMPETITIONS[competition]["football_data_org"]["code"]
        for competition in expected_codes
    } == expected_codes
