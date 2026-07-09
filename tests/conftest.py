"""Shared test fixtures for the footix test suite."""

from __future__ import annotations

import pandas as pd
import pytest

from footix.strategy.bets import OddsInput


@pytest.fixture
def sample_match_df() -> pd.DataFrame:
    """A small match DataFrame with 4 teams and 6 matches.

    Each team appears both home and away; columns match the
    verify_required_column decorator requirements.
    """
    return pd.DataFrame(
        {
            "date": [
                "01/01/2023",
                "05/01/2023",
                "10/01/2023",
                "15/01/2023",
                "20/01/2023",
                "25/01/2023",
            ],
            "home_team": ["Team_A", "Team_B", "Team_C", "Team_D", "Team_A", "Team_B"],
            "away_team": ["Team_B", "Team_C", "Team_D", "Team_A", "Team_C", "Team_D"],
            "fthg": [2, 1, 0, 1, 3, 2],
            "ftag": [1, 1, 2, 1, 0, 1],
            "ftr": ["H", "D", "A", "D", "H", "H"],
        }
    )


@pytest.fixture
def sample_odds_input() -> list[OddsInput]:
    """Sample odds input for strategy tests."""
    return [
        OddsInput(home_team="Team1", away_team="Team2", odds=[2.0, 3.0, 4.0]),
        OddsInput(home_team="Team3", away_team="Team4", odds=[1.5, 4.0, 3.0]),
    ]
