from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from footix.evaluation import BacktestConfig, BacktestResult, ModelSpec, run_backtest
from footix.models.calibration import OutcomeCalibrator
from footix.models.score_matrix import GoalMatrix
from footix.utils.typing import ProbaResult


class ToyModel:
    fit_sizes: list[int] = []

    def fit(self, data: pd.DataFrame) -> None:
        self.train_end = data["kickoff"].max()
        self.fit_sizes.append(len(data))

    def predict(self, home_team: str, away_team: str) -> ProbaResult:
        _ = home_team, away_team
        return ProbaResult(0.60, 0.20, 0.20)


class ToyGoalModel:
    fit_sizes: list[int] = []

    def fit(self, data: pd.DataFrame) -> None:
        self.fit_sizes.append(len(data))

    def predict(self, home_team: str, away_team: str) -> GoalMatrix:
        _ = home_team, away_team
        return GoalMatrix([0.8, 0.15, 0.05], [0.8, 0.15, 0.05])


class StatelessModel:
    def predict(self, home_team: str, away_team: str) -> ProbaResult:
        _ = home_team, away_team
        return ProbaResult(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def _window_samples(model: object, matches: list[tuple[str, str]], market: str) -> np.ndarray:
    """Toy joint sampler: constant probabilities across 4 shared draws."""
    _ = model
    n_draws = 4
    if market == "1X2":
        probs = np.asarray([[0.6, 0.2, 0.2]] * n_draws)
        return np.broadcast_to(probs[:, None, :], (n_draws, len(matches), 3)).copy()
    probs = np.asarray([[0.5, 0.5]] * n_draws)
    return np.broadcast_to(probs[:, None, :], (n_draws, len(matches), 2)).copy()


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["02/08/2024", "03/08/2024", "09/08/2024", "10/08/2024"],
            "time": ["20:00"] * 4,
            "home_team": ["A", "C", "A", "C"],
            "away_team": ["B", "D", "B", "D"],
            "fthg": [1, 0, 0, 2],
            "ftag": [0, 1, 1, 0],
            "ftr": ["H", "A", "A", "H"],
            "b365_h": [2.0] * 4,
            "b365_d": [3.0] * 4,
            "b365_a": [4.0] * 4,
            "b365<2.5": [1.8] * 4,
            "b365>2.5": [2.2] * 4,
        }
    )


def test_run_backtest_uses_only_matches_before_cutoff() -> None:
    ToyModel.fit_sizes = []
    result = run_backtest(
        _matches(),
        [ModelSpec("toy", ToyModel, markets=("1X2",))],
        BacktestConfig(markets=("1X2",), bankroll=100.0, min_stake=0.0),
    )

    assert ToyModel.fit_sizes == [2]
    assert result.windows.iloc[0]["status"] == "ineligible"
    assert result.windows.iloc[1]["train_matches"] == 2
    assert len(result.predictions) == 2
    assert result.predictions["cutoff"].min() == pd.Timestamp("2024-08-09")


def test_run_backtest_keeps_one_bet_per_match() -> None:
    result = run_backtest(
        _matches(),
        [ModelSpec("toy", ToyModel, markets=("1X2",))],
        BacktestConfig(markets=("1X2",), bankroll=100.0, min_stake=0.0),
    )

    assert result.bets["match_id"].nunique() == len(result.bets)
    assert set(result.bets["selection"]) == {"H"}
    assert result.windows.iloc[-1]["total_stake"] <= 30.0


def test_goal_matrix_models_produce_total_goal_markets() -> None:
    result = run_backtest(
        _matches(),
        [ModelSpec("goals", ToyGoalModel, markets=("1X2", "O/U2.5"))],
        BacktestConfig(markets=("1X2", "O/U2.5"), bankroll=100.0, min_stake=0.0),
    )

    assert set(result.predictions["market"]) == {"1X2", "O/U2.5"}
    assert len(result.predictions) == 4
    assert result.bets["match_id"].nunique() == len(result.bets)


def test_stateless_model_does_not_require_fit() -> None:
    result = run_backtest(
        _matches(),
        [ModelSpec("stateless", StatelessModel, staking="none", requires_training=False)],
        BacktestConfig(markets=("1X2",), bankroll=100.0, min_stake=0.0),
    )

    assert set(result.windows["status"]) == {"ok"}
    assert len(result.predictions) == len(_matches())


def test_portfolio_torch_staking_uses_window_scenarios() -> None:
    result = run_backtest(
        _matches(),
        [
            ModelSpec(
                "scenario",
                ToyModel,
                markets=("1X2",),
                staking="portfolio_torch",
                window_samples=_window_samples,
            )
        ],
        BacktestConfig(markets=("1X2",), bankroll=100.0, min_stake=0.0, n_scenarios=200),
    )

    assert result.bets["match_id"].nunique() == len(result.bets)
    assert result.windows.iloc[-1]["total_stake"] <= 30.0


def test_portfolio_torch_requires_window_samples() -> None:
    with pytest.raises(ValueError, match="window_samples"):
        run_backtest(
            _matches(),
            [ModelSpec("scenario", ToyModel, markets=("1X2",), staking="portfolio_torch")],
            BacktestConfig(markets=("1X2",), bankroll=100.0, min_stake=0.0),
        )


def test_flat_staking_uses_robust_selection_and_fixed_stake() -> None:
    result = run_backtest(
        _matches(),
        [
            ModelSpec(
                "scenario",
                ToyModel,
                markets=("1X2",),
                staking="flat",
                window_samples=_window_samples,
            )
        ],
        BacktestConfig(
            markets=("1X2",),
            bankroll=100.0,
            min_stake=0.0,
            flat_fraction=0.01,
        ),
    )

    assert set(result.bets["stake"]) == {1.0}
    assert result.bets["match_id"].nunique() == len(result.bets)


def _matches_calibration() -> pd.DataFrame:
    """Four weekly windows of the same four matches, so a calibrator can learn."""
    weeks = []
    for week in range(4):
        frame = _matches()
        dates = pd.to_datetime(frame["date"], dayfirst=True) + pd.Timedelta(days=7 * week)
        frame["date"] = dates.dt.strftime("%d/%m/%Y")
        weeks.append(frame)
    return pd.concat(weeks, ignore_index=True)


def _probs_by_cutoff(result: BacktestResult, market: str) -> dict[pd.Timestamp, np.ndarray]:
    rows = result.predictions[result.predictions["market"] == market]
    return {
        cutoff: np.asarray(group["probabilities"].tolist())
        for cutoff, group in rows.groupby("cutoff")
    }


def test_backtest_calibrator_learns_out_of_sample() -> None:
    """The 1X2 calibrator is fitted on strictly past windows and never touches O/U."""
    frame = _matches_calibration()
    spec = ModelSpec("goals", ToyGoalModel, markets=("1X2", "O/U2.5"))
    base = BacktestConfig(markets=("1X2", "O/U2.5"), bankroll=100.0, min_stake=0.0)
    without = run_backtest(frame, [spec], base)
    with_cal = run_backtest(
        frame,
        [spec],
        replace(base, calibrator_factory=lambda: OutcomeCalibrator(warmup=4)),
    )

    raw = _probs_by_cutoff(without, "1X2")
    calibrated = _probs_by_cutoff(with_cal, "1X2")
    cutoffs = sorted(raw)

    # First predicted window has no past data to learn from: identity
    assert np.allclose(calibrated[cutoffs[0]], raw[cutoffs[0]])

    # Later windows are recalibrated from earlier windows only
    calibrator = OutcomeCalibrator(warmup=4)
    rows = without.predictions
    first = rows[(rows["market"] == "1X2") & (rows["cutoff"] == cutoffs[0])]
    for row in first.itertuples():
        calibrator.accumulate(row.probabilities, row.actual)
    calibrator.fit()
    second = rows[(rows["market"] == "1X2") & (rows["cutoff"] == cutoffs[1])]
    expected = np.asarray([calibrator.apply(row.probabilities) for row in second.itertuples()])
    assert not np.allclose(calibrated[cutoffs[1]], raw[cutoffs[1]])
    assert np.allclose(calibrated[cutoffs[1]], expected)

    # O/U2.5 is never touched by the 1X2 calibrator
    ou_raw = _probs_by_cutoff(without, "O/U2.5")
    ou_calibrated = _probs_by_cutoff(with_cal, "O/U2.5")
    for cutoff in cutoffs:
        assert np.allclose(ou_calibrated[cutoff], ou_raw[cutoff])


def test_probability_sample_positive_edge_uses_odds() -> None:
    """prob_edge_pos must use the bet's own break-even probability 1/o."""
    from footix.evaluation.backtest import _probability_sample

    def samples(model: object, home: str, away: str, market: str) -> np.ndarray:
        _ = model, home, away, market
        return np.asarray([[0.4], [0.51], [0.7]])

    spec = ModelSpec("s", object, markets=("1X2",), samples=samples)

    # odds = 2.0 => break-even p = 0.5 => P(edge > 0) = 2/3
    std, prob_edge_pos = _probability_sample(spec, object(), "A", "B", "1X2", 0, 2.0)
    assert np.isclose(prob_edge_pos, 2.0 / 3.0)
    assert std is not None and std > 0.0

    # odds = 1.5 => break-even p = 2/3; a sample at 0.55 beats the old 1/K
    # threshold (1/3) but not the true break-even: P(edge > 0) must be 0.
    def low_samples(model: object, home: str, away: str, market: str) -> np.ndarray:
        _ = model, home, away, market
        return np.asarray([[0.4], [0.51], [0.55]])

    low_spec = ModelSpec("s", object, markets=("1X2",), samples=low_samples)
    _, prob_edge_pos = _probability_sample(low_spec, object(), "A", "B", "1X2", 0, 1.5)
    assert prob_edge_pos == 0.0

    # missing odds => no positive-edge statistic
    _, prob_edge_pos = _probability_sample(spec, object(), "A", "B", "1X2", 0, None)
    assert prob_edge_pos is None


class InvalidMcmcModel(ToyModel):
    def get_diagnostics(self) -> dict[str, object]:
        return {
            "status": "invalid_mcmc",
            "max_rhat": 1.2,
            "divergences": 5,
            "min_ess_bulk": 40.0,
            "min_ess_tail": 30.0,
            "reason": "max_rhat=1.200 > 1.01",
        }


class FailedDiagnosticsModel(ToyModel):
    def get_diagnostics(self) -> dict[str, object]:
        return {"status": "failed", "reason": "sample_stats missing"}


def test_backtest_gates_invalid_mcmc_windows() -> None:
    """Windows whose fit did not converge are flagged and never scored."""
    result = run_backtest(
        _matches(),
        [ModelSpec("bad", InvalidMcmcModel, markets=("1X2",))],
        BacktestConfig(markets=("1X2",), bankroll=100.0, min_stake=0.0),
    )

    assert set(result.windows["status"]) == {"ineligible", "invalid_mcmc"}
    assert result.predictions.empty
    assert result.bets.empty
    reason = result.windows.iloc[-1]["reason"]
    assert "max_rhat=1.200" in reason and "divergences=5" in reason


def test_backtest_gates_failed_diagnostics() -> None:
    result = run_backtest(
        _matches(),
        [ModelSpec("bad", FailedDiagnosticsModel, markets=("1X2",))],
        BacktestConfig(markets=("1X2",), bankroll=100.0, min_stake=0.0),
    )

    assert set(result.windows["status"]) == {"ineligible", "failed"}
    assert result.predictions.empty


def test_backtest_reports_mcmc_status_for_valid_runs() -> None:
    result = run_backtest(
        _matches(),
        [ModelSpec("toy", ToyModel, markets=("1X2",))],
        BacktestConfig(markets=("1X2",), bankroll=100.0, min_stake=0.0),
    )

    ok = result.windows[result.windows["status"] == "ok"]
    assert not ok.empty
    assert set(ok["mcmc_status"]) == {"n/a"}


def test_backtest_calibrator_current_window_is_never_used_for_itself() -> None:
    """Outcomes of the current cutoff cannot alter its own calibrator."""
    frame = _matches_calibration()
    spec = ModelSpec("goals", ToyGoalModel, markets=("1X2",))
    base = BacktestConfig(markets=("1X2",), bankroll=100.0, min_stake=0.0)
    result = run_backtest(
        frame, [spec], replace(base, calibrator_factory=lambda: OutcomeCalibrator(warmup=4))
    )

    raw = _probs_by_cutoff(run_backtest(frame, [spec], base), "1X2")
    calibrated = _probs_by_cutoff(result, "1X2")
    cutoffs = sorted(raw)

    # Replay the calibrator as the evaluator does: fit at the start of every
    # window on strictly earlier outcomes, then predict the whole window.
    expected = {}
    calibrator = OutcomeCalibrator(warmup=4)
    for cutoff in cutoffs:
        calibrator.fit()
        expected[cutoff] = np.asarray([calibrator.apply(p) for p in raw[cutoff]])
        for probs, actual in zip(
            raw[cutoff], result.predictions[result.predictions["cutoff"] == cutoff]["actual"]
        ):
            calibrator.accumulate(probs, int(actual))

    for cutoff in cutoffs:
        assert np.allclose(calibrated[cutoff], expected[cutoff])

    # The last window's own outcomes are only in the calibrator afterwards:
    # re-running the run with one less window changes nothing for earlier
    # windows' predictions.
    assert np.allclose(calibrated[cutoffs[0]], raw[cutoffs[0]])
