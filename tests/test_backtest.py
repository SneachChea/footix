from __future__ import annotations

import pandas as pd

from footix.evaluation import BacktestConfig, ModelSpec, run_backtest
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
