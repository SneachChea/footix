"""Tests for PoissonModel (scipy-based MLE)."""

from __future__ import annotations

import pandas as pd
import pytest

from footix.models.basic_poisson import PoissonModel


def test_init_valid_params():
    model = PoissonModel(n_teams=4, n_goals=6)
    assert model.n_teams == 4
    assert model.n_goals == 6


def test_init_invalid_n_teams():
    with pytest.raises(ValueError, match="Number of teams should be positive"):
        PoissonModel(n_teams=0, n_goals=6)


def test_init_invalid_n_goals():
    with pytest.raises(ValueError, match="Number of goals must be positive"):
        PoissonModel(n_teams=4, n_goals=3)


def test_fit_sets_parameters(sample_match_df):
    model = PoissonModel(n_teams=4, n_goals=6)
    model.fit(sample_match_df)
    assert hasattr(model, "gamma")
    assert hasattr(model, "alphas")
    assert hasattr(model, "betas")
    assert hasattr(model, "dict_teams")
    assert len(model.alphas) == 4
    assert len(model.betas) == 4


def test_fit_team_count_mismatch():
    model = PoissonModel(n_teams=2, n_goals=6)
    df = pd.DataFrame(
        {
            "home_team": ["A", "B", "C"],
            "away_team": ["B", "C", "A"],
            "ftr": ["H", "A", "D"],
            "fthg": [1, 2, 0],
            "ftag": [0, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="Expecting 2 teams, only got 3"):
        model.fit(df)


def test_fit_team_not_all_played_home_away():
    model = PoissonModel(n_teams=3, n_goals=6)
    df = pd.DataFrame(
        {
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "C", "B"],
            "ftr": ["H", "A", "D"],
            "fthg": [1, 2, 0],
            "ftag": [0, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="Not every teams have played at home and away"):
        model.fit(df)


def test_predict_returns_goal_matrix(sample_match_df):
    model = PoissonModel(n_teams=4, n_goals=6)
    model.fit(sample_match_df)
    gm = model.predict("Team_A", "Team_B")
    assert len(gm.home_goals_probs) == 6
    assert len(gm.away_goals_probs) == 6


def test_predict_unknown_team(sample_match_df):
    model = PoissonModel(n_teams=4, n_goals=6)
    model.fit(sample_match_df)
    with pytest.raises(ValueError, match="not in the list"):
        model.predict("Unknown", "Team_B")
    with pytest.raises(ValueError, match="not in the list"):
        model.predict("Team_A", "Unknown")


def test_goal_expectation_positive(sample_match_df):
    model = PoissonModel(n_teams=4, n_goals=6)
    model.fit(sample_match_df)
    lamb, mu = model.goal_expectation(home_team_id=0, away_team_id=1)
    assert lamb > 0
    assert mu > 0


def test_mapping_team_index(sample_match_df):
    model = PoissonModel(n_teams=4, n_goals=6)
    mapping = model.mapping_team_index(sample_match_df["home_team"])
    assert isinstance(mapping, dict)
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in mapping.items())
    assert sorted(mapping.values()) == [0, 1, 2, 3]


def test_print_parameters_before_fit():
    model = PoissonModel(n_teams=4, n_goals=6)
    with pytest.raises(AttributeError, match="not trained"):
        model.print_parameters()
