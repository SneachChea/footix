"""Tests for PoissonModel (scipy-based MLE)."""

from __future__ import annotations

from itertools import permutations
from types import SimpleNamespace

import numpy as np
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


def test_balanced_results_are_recovered():
    """A league where every match ends 3-2 must predict 3-2, not 3-1.

    The former second sum-to-zero constraint pinned the away scoring level at
    one goal per match instead of letting the data set it.
    """
    df = pd.DataFrame(
        [
            {"home_team": home, "away_team": away, "ftr": "H", "fthg": 3, "ftag": 2}
            for home, away in permutations(["A", "B", "C"], 2)
        ]
    )
    model = PoissonModel(n_teams=3, n_goals=20)
    model.fit(df)

    lamb, mu = model.goal_expectation(0, 1)
    assert lamb == pytest.approx(3.0, abs=1e-3)
    assert mu == pytest.approx(2.0, abs=1e-3)


def test_fit_rejects_invalid_goals(sample_match_df):
    """Non-finite or negative goals must fail loudly instead of fitting NaN."""
    model = PoissonModel(n_teams=4, n_goals=6)
    missing = sample_match_df.copy()
    missing.loc[0, "fthg"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(missing)

    negative = sample_match_df.copy()
    negative.loc[0, "ftag"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        model.fit(negative)


def test_fit_raises_when_optimization_fails(sample_match_df, monkeypatch):
    """A non-converged MLE must raise instead of storing meaningless parameters."""

    def failing_minimize(*args, **kwargs):
        return SimpleNamespace(x=np.zeros(2 * 4 + 1), success=False, message="boom")

    monkeypatch.setattr("footix.models.basic_poisson.optimize.minimize", failing_minimize)
    model = PoissonModel(n_teams=4, n_goals=6)
    with pytest.raises(RuntimeError, match="not successful"):
        model.fit(sample_match_df)
    assert not hasattr(model, "gamma")


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
