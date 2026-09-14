"""Tests for optional statistics support in BayesianModel."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from footix.models.bayesian import BayesianModel, _extract_optional_stats_data, _generative_model

pytestmark = pytest.mark.bayesian


def _base_frame() -> pd.DataFrame:
    """Create a minimal dataframe with required columns."""
    return pd.DataFrame(
        {
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["B", "C", "D", "A"],
            "fthg": [1, 2, 0, 1],
            "ftag": [0, 1, 1, 2],
        }
    )


def test_extract_optional_stats_masks_partial_nans() -> None:
    """Extract stats channels and keep only valid rows per channel."""
    df = _base_frame().assign(
        hs=[10.0, np.nan, 9.0, 8.0],
        **{"as": [7.0, 6.0, np.nan, 5.0]},
        hst=[4.0, 3.0, 2.0, 2.0],
        ast=[2.0, np.nan, 1.0, 1.0],
        hc=[5.0, 4.0, 4.0, np.nan],
        ac=[3.0, 3.0, np.nan, 2.0],
    )

    stats_data = _extract_optional_stats_data(df)

    assert stats_data["has_shots"] is True
    assert stats_data["has_sot"] is True
    assert stats_data["has_corners"] is True

    assert np.array_equal(stats_data["shots_idx"], np.array([0, 3]))
    assert np.array_equal(stats_data["sot_idx"], np.array([0, 2, 3]))
    assert np.array_equal(stats_data["corners_idx"], np.array([0, 1]))


def test_extract_optional_stats_accepts_uppercase_columns() -> None:
    """Support football-data uppercase naming conventions."""
    df = _base_frame().assign(
        HS=[10, 9, 8, 11],
        AS=[6, 7, 5, 8],
        HST=[4, 3, 2, 5],
        AST=[2, 1, 2, 3],
        HC=[5, 4, 6, 7],
        AC=[3, 4, 3, 2],
    )

    stats_data = _extract_optional_stats_data(df)

    assert stats_data["has_shots"] is True
    assert stats_data["has_sot"] is True
    assert stats_data["has_corners"] is True
    assert len(stats_data["shots_idx"]) == len(df)


def test_extract_optional_stats_when_columns_missing() -> None:
    """Return disabled channels when optional columns are absent."""
    stats_data = _extract_optional_stats_data(_base_frame())

    assert stats_data["has_shots"] is False
    assert stats_data["has_sot"] is False
    assert stats_data["has_corners"] is False


def test_fit_passes_optional_stats_when_enabled(monkeypatch: Any) -> None:
    """Pass extracted optional stats into hierarchical model when enabled."""
    captured: dict[str, Any] = {}

    def fake_hierarchical_bayes(
        self: BayesianModel,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
        optional_stats: dict[str, Any] | None = None,
    ) -> str:
        captured["optional_stats"] = optional_stats
        return "trace"

    monkeypatch.setattr(BayesianModel, "hierarchical_bayes", fake_hierarchical_bayes)

    df = _base_frame().assign(hs=[10, 9, 8, 7], **{"as": [6, 5, 4, 3]})
    model = BayesianModel(n_goals=6, use_stats=True)
    model.fit(df)

    assert model.trace == "trace"
    assert captured["optional_stats"] is not None
    assert captured["optional_stats"]["has_shots"] is True


def test_fit_disables_optional_stats_when_flag_false(monkeypatch: Any) -> None:
    """Do not pass optional stats when use_stats is disabled."""
    captured: dict[str, Any] = {}

    def fake_hierarchical_bayes(
        self: BayesianModel,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
        optional_stats: dict[str, Any] | None = None,
    ) -> str:
        captured["optional_stats"] = optional_stats
        return "trace"

    monkeypatch.setattr(BayesianModel, "hierarchical_bayes", fake_hierarchical_bayes)

    df = _base_frame().assign(hs=[10, 9, 8, 7], **{"as": [6, 5, 4, 3]})
    model = BayesianModel(n_goals=6, use_stats=False)
    model.fit(df)

    assert model.trace == "trace"
    assert captured["optional_stats"] is None


@pytest.mark.parametrize(
    ("home_col", "away_col"),
    [("hs", "AS"), ("hst", "AST"), ("hc", "AC")],
)
def test_stats_channels_use_the_goals_defence_sign(home_col: str, away_col: str) -> None:
    """A stronger defence must lower the opponent's shots, as it does for goals.

    Both teams score the same in both fixtures, so the goals likelihood is
    invariant when the two defence values are swapped. Only the statistics
    channel under test can then prefer a side, and its sign decides which swap
    wins.
    """
    df = pd.DataFrame(
        {
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "fthg": [1, 1],
            "ftag": [1, 1],
            home_col: [10, 8],
            away_col: [2, 10],
        }
    )
    model = _generative_model(
        goals_home_obs=df["fthg"].to_numpy(dtype=float),
        goals_away_obs=df["ftag"].to_numpy(dtype=float),
        home_team=np.array([0, 1]),
        away_team=np.array([1, 0]),
        n_teams=2,
        use_stats=True,
        optional_stats=_extract_optional_stats_data(df),
    )
    logp = model.compile_logp()

    def log_density(defence: list[float]) -> float:
        point = model.initial_point()
        point["raw_defence"] = np.asarray(defence, dtype=float)
        return float(logp(point))

    # A shoots a lot and concedes few: its own defence must be the strong one.
    assert log_density([3.0, -3.0]) > log_density([-3.0, 3.0])
