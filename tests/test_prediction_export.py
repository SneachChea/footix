"""Tests for prediction export utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from footix.data_io.prediction_export import (
    build_prediction_records_from_predictions,
    export_prediction_records_from_model,
)
from footix.models.score_matrix import GoalMatrix
from footix.utils.typing import SampleProbaResult


def _sample_result(seed: int = 0) -> SampleProbaResult:
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(alpha=np.array([4.0, 2.0, 3.0]), size=256)
    return SampleProbaResult(
        proba_home=raw[:, 0],
        proba_draw=raw[:, 1],
        proba_away=raw[:, 2],
    )


def _goal_matrix() -> GoalMatrix:
    home = np.array([0.24, 0.22, 0.18, 0.14, 0.10, 0.06, 0.03, 0.015, 0.01, 0.005])
    away = np.array([0.20, 0.21, 0.19, 0.15, 0.11, 0.07, 0.04, 0.02, 0.015, 0.005])
    return GoalMatrix(home, away)


def test_build_prediction_records_shapes_and_sorting() -> None:
    fixtures = [
        {
            "home": "Cotes Team A",
            "away": "Team B",
            "kickoff_paris": "2026-02-20T20:45:00",
        },
        {
            "home": "Cotes Team C",
            "away": "Team D",
            "kickoff_paris": "2026-02-19T19:00:00",
        },
    ]
    payload_metadata = {
        "league_url": "https://example.com/france/ligue-1",
        "scraped_at_paris": "2026-02-18T18:00:00+01:00",
    }

    gm = _goal_matrix()
    goal_matrices = {
        "Team A - Team B": gm,
        "Team C - Team D": gm,
    }
    samples = {
        "Team A - Team B": _sample_result(seed=7),
        "Team C - Team D": _sample_result(seed=11),
    }

    records, errors = build_prediction_records_from_predictions(
        fixtures=fixtures,
        goal_matrices=goal_matrices,
        samples=samples,
        payload_metadata=payload_metadata,
    )

    assert errors == []
    assert len(records) == 2
    assert records[0]["kickoff"] <= records[1]["kickoff"]

    first = records[0]
    assert first["league"] == "Ligue 1"
    assert len(first["home_distribution"]) == 20
    assert len(first["draw_distribution"]) == 20
    assert len(first["away_distribution"]) == 20
    assert len(first["distribution_bins"]) == 21
    assert len(first["score_matrix"]) == 5
    assert all(len(row) == 5 for row in first["score_matrix"])

    prediction = first["prediction"]
    assert (
        0.99
        <= (prediction["home_win_prob"] + prediction["draw_prob"] + prediction["away_win_prob"])
        <= 1.01
    )
    assert 0.0 <= prediction["confidence_score"] <= 1.0
    assert prediction["confidence_score"] == round(prediction["confidence_score"], 2)


def test_confidence_curve_applied_when_gamma_set() -> None:
    fixtures = [{"home": "Cotes Team A", "away": "Team B", "kickoff_paris": "2026-02-20T20:45:00"}]
    gm = _goal_matrix()
    goal_matrices = {"Team A - Team B": gm}
    samples = {"Team A - Team B": _sample_result(seed=7)}

    curved_records, _ = build_prediction_records_from_predictions(
        fixtures=fixtures,
        goal_matrices=goal_matrices,
        samples=samples,
        payload_metadata={"league_url": "https://example.com/france/ligue-1"},
        confidence_gamma=0.7,
    )
    raw_records, _ = build_prediction_records_from_predictions(
        fixtures=fixtures,
        goal_matrices=goal_matrices,
        samples=samples,
        payload_metadata={"league_url": "https://example.com/france/ligue-1"},
        confidence_gamma=None,
    )

    assert len(curved_records) == 1
    assert len(raw_records) == 1
    curved_conf = curved_records[0]["prediction"]["confidence_score"]
    raw_conf = raw_records[0]["prediction"]["confidence_score"]
    assert curved_conf != raw_conf


def test_kde_distributions_have_expected_bins() -> None:
    fixtures = [{"home": "Cotes Team A", "away": "Team B", "kickoff_paris": "2026-02-20"}]
    gm = _goal_matrix()
    goal_matrices = {"Team A - Team B": gm}
    samples = {"Team A - Team B": _sample_result(seed=15)}

    records, errors = build_prediction_records_from_predictions(
        fixtures=fixtures,
        goal_matrices=goal_matrices,
        samples=samples,
        payload_metadata={"league_url": "https://example.com/france/ligue-1"},
    )

    assert errors == []
    assert len(records) == 1
    rec = records[0]
    assert len(rec["distribution_bins"]) == 21
    assert rec["distribution_bins"][0] == "0.0"
    assert rec["distribution_bins"][-1] == "1.0"
    numeric_bins = np.asarray([float(value) for value in rec["distribution_bins"]])
    assert np.allclose(np.diff(numeric_bins), 0.05)
    assert len(rec["home_distribution"]) == 20
    assert len(rec["draw_distribution"]) == 20
    assert len(rec["away_distribution"]) == 20


def test_build_prediction_records_reports_missing_inputs() -> None:
    fixtures = [{"home": "Cotes Team A", "away": "Team B", "kickoff_paris": "2026-02-20"}]
    records, errors = build_prediction_records_from_predictions(
        fixtures=fixtures,
        goal_matrices={},
        samples={},
        payload_metadata={"league_url": "https://example.com/ligue-2"},
    )

    assert records == []
    assert len(errors) == 1
    assert errors[0]["match"] == "Team A - Team B"


class _DummyModel:
    def predict(self, home_team: str, away_team: str, **kwargs: Any) -> GoalMatrix:
        _ = kwargs
        assert home_team == "Team A"
        assert away_team == "Team B"
        return _goal_matrix()

    def get_samples(self, home_team: str, away_team: str, **kwargs: Any) -> SampleProbaResult:
        _ = kwargs
        assert home_team == "Team A"
        assert away_team == "Team B"
        return _sample_result(seed=4)


def test_export_prediction_records_from_model_with_normalizer() -> None:
    fixtures = [{"home": "Cotes TEAM A", "away": "TEAM B", "kickoff_paris": "2026-02-20"}]
    model = _DummyModel()

    records, errors = export_prediction_records_from_model(
        model=model,
        fixtures=fixtures,
        payload_metadata={"league_url": "https://example.com/france/ligue-1"},
        team_normalizer=lambda name: name.title(),
    )

    assert errors == []
    assert len(records) == 1
    assert records[0]["home_team"] == "Team A"
    assert records[0]["away_team"] == "Team B"
    assert records[0]["league"] == "Ligue 1"
    assert 0.0 <= records[0]["prediction"]["confidence_score"] <= 1.0
    assert records[0]["prediction"]["confidence_score"] == round(
        records[0]["prediction"]["confidence_score"], 2
    )
