"""Tests for EloDavidson rating system."""

from __future__ import annotations

import numpy as np
import pytest

from footix.models.elo import EloDavidson
from footix.utils.typing import ProbaResult


@pytest.fixture
def valid_agnostic() -> ProbaResult:
    return ProbaResult(proba_home=0.45, proba_draw=0.25, proba_away=0.30)


@pytest.fixture
def elo_model(valid_agnostic) -> EloDavidson:
    return EloDavidson(n_teams=4, k0=20, lambd=1.0, sigma=100, agnostic_probs=valid_agnostic)


def test_init_valid_probas(valid_agnostic):
    model = EloDavidson(n_teams=4, k0=20, lambd=1.0, sigma=100, agnostic_probs=valid_agnostic)
    assert model.n_teams == 4
    assert model.k0 == 20
    assert model.lamda == 1.0
    assert model.sigma == 100


def test_init_invalid_probas():
    with pytest.raises(ValueError, match="Probabilities do not sum to one"):
        EloDavidson(
            n_teams=4, k0=20, lambd=1.0, sigma=100, agnostic_probs=ProbaResult(0.5, 0.3, 0.1)
        )


def test_compute_kappa():
    kappa = EloDavidson.compute_kappa(P_H=0.45, P_D=0.25, P_A=0.30)
    expected = 0.25 / np.sqrt(0.45 * 0.30)
    assert np.isclose(kappa, expected)


def test_compute_eta():
    eta = EloDavidson.compute_eta(P_H=0.45, P_A=0.30)
    expected = np.log10(0.45 / 0.30)
    assert np.isclose(eta, expected)


def test_correspondance_result():
    assert EloDavidson.correspondance_result("H") == 1.0
    assert EloDavidson.correspondance_result("D") == 0.5
    assert EloDavidson.correspondance_result("A") == 0.0
    with pytest.raises(ValueError, match="result must be"):
        EloDavidson.correspondance_result("X")


def test_fit_with_dataframe(elo_model, sample_match_df):
    elo_model.fit(sample_match_df)
    assert len(elo_model.championnat) == 4
    for team in ["Team_A", "Team_B", "Team_C", "Team_D"]:
        assert team in elo_model.championnat


def test_fit_team_count_mismatch(elo_model, sample_match_df):
    elo_model.n_teams = 10
    with pytest.raises(ValueError, match="Number of teams"):
        elo_model.fit(sample_match_df)


def test_predict_returns_proba_result(elo_model, sample_match_df):
    elo_model.fit(sample_match_df)
    result = elo_model.predict("Team_A", "Team_B")
    assert isinstance(result, ProbaResult)
    assert np.isclose(sum(result), 1.0, atol=1e-10)
    assert all(0 <= p <= 1 for p in result)


def test_predict_unknown_team(elo_model, sample_match_df):
    elo_model.fit(sample_match_df)
    with pytest.raises(KeyError):
        elo_model.predict("Unknown", "Team_B")


def test_str_representation(elo_model, sample_match_df):
    elo_model.fit(sample_match_df)
    s = str(elo_model)
    assert "Team_A" in s
    assert "Team_B" in s
    assert ":" in s


def test_reset(elo_model, sample_match_df):
    elo_model.fit(sample_match_df)
    assert len(elo_model.championnat) > 0
    elo_model.reset()
    assert len(elo_model.championnat) == 0


def test_define_k_param(elo_model):
    assert elo_model.define_k_param(gamma=0) == 20.0
    assert elo_model.define_k_param(gamma=1) == 40.0
    assert elo_model.define_k_param(gamma=3) == 80.0
