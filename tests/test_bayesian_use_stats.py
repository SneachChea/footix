"""Tests for optional statistics support in BayesianModel."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from footix.models.bayesian import BayesianModel, _extract_optional_stats_data

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
