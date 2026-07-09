"""Tests for EloDataReader and MatchupResult."""

from __future__ import annotations

import pandas as pd

from footix.data_io.data_reader import EloDataReader, MatchupResult


def test_matchup_result_from_dict():
    row = {
        "home_team": "Team_A",
        "away_team": "Team_B",
        "ftr": "H",
        "ftag": 1,
        "fthg": 2,
    }
    result = MatchupResult.from_dict(row)
    assert result.home_team == "Team_A"
    assert result.away_team == "Team_B"
    assert result.result == "H"
    assert result.home_goals == 2
    assert result.away_goals == 1


def test_elo_data_reader_init(sample_match_df):
    reader = EloDataReader(sample_match_df)
    assert len(reader) == len(sample_match_df)
    assert reader.df_data is not None


def test_elo_data_reader_sorts_by_date():
    df = pd.DataFrame(
        {
            "date": ["10/01/2023", "01/01/2023", "05/01/2023"],
            "home_team": ["A", "B", "C"],
            "away_team": ["B", "C", "A"],
            "fthg": [1, 2, 0],
            "ftag": [0, 1, 1],
            "ftr": ["H", "A", "D"],
        }
    )
    reader = EloDataReader(df)
    assert reader.df_data["date"].iloc[0] == pd.Timestamp("2023-01-01")
    assert reader.df_data["date"].iloc[2] == pd.Timestamp("2023-01-10")


def test_elo_data_reader_len(sample_match_df):
    reader = EloDataReader(sample_match_df)
    assert len(reader) == 6


def test_elo_data_reader_iter(sample_match_df):
    reader = EloDataReader(sample_match_df)
    results = list(reader)
    assert len(results) == 6
    assert all(isinstance(r, MatchupResult) for r in results)


def test_elo_data_reader_getitem(sample_match_df):
    reader = EloDataReader(sample_match_df)
    first = reader[0]
    assert isinstance(first, MatchupResult)
    assert first.home_team is not None
    assert first.away_team is not None


def test_elo_data_reader_unique_teams(sample_match_df):
    reader = EloDataReader(sample_match_df)
    teams = reader.unique_teams()
    assert isinstance(teams, list)
    assert teams == sorted(teams)
    assert "Team_A" in teams
    assert "Team_B" in teams
