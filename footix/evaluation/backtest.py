"""Chronological, expanding-window evaluation for football models.

The module deliberately evaluates one dataframe at a time. Call ``run_backtest`` once per
competition/season so that a training window can never cross a season boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from footix.metrics import accuracy, brier_score, log_loss, rps
from footix.models.score_matrix import GoalMatrix
from footix.strategy.bets import Bet
from footix.strategy.kelly_strategies import fractional_kelly
from footix.strategy.portfolio_management import optimise_portfolio_torch
from footix.utils.typing import ProbaResult

Market = Literal["1X2", "O/U2.5", "U2.5", "O2.5"]
CanonicalMarket = Literal["1X2", "O/U2.5"]
Staking = Literal["fractional_kelly", "portfolio_torch", "none"]
MARKETS: tuple[CanonicalMarket, ...] = ("1X2", "O/U2.5")


@dataclass(frozen=True)
class ModelSpec:
    """Describe how the evaluator creates, fits and queries one model."""

    name: str
    factory: Callable[[], Any]
    markets: tuple[Market, ...] = ("1X2",)
    staking: Staking = "fractional_kelly"
    fit: Callable[[Any, pd.DataFrame], None] | None = None
    samples: Callable[[Any, str, str, Market], np.ndarray] | None = None
    requires_training: bool = True


@dataclass
class BacktestConfig:
    """Configuration for the Friday-to-Friday walk-forward evaluation."""

    cutoff_weekday: int = 4
    horizon_days: int = 7
    edge_floor: float = 0.0
    bankroll: float = 1_000.0
    max_fraction: float = 0.30
    fraction_kelly: float = 0.25
    optimizer_alpha: float = 0.05
    optimizer_gamma: float | None = 0.0
    optimizer_lr: float = 0.05
    optimizer_iters: int = 5_000
    optimizer_penalty_lambda: float = 1_000.0
    markets: tuple[Market, ...] = MARKETS
    odds_columns: Mapping[str, str] = field(
        default_factory=lambda: {
            "H": "b365_h",
            "D": "b365_d",
            "A": "b365_a",
            "U2.5": "b365<2.5",
            "O2.5": "b365>2.5",
        }
    )
    min_stake: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.cutoff_weekday <= 6:
            raise ValueError("cutoff_weekday must be between 0 and 6")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.bankroll <= 0:
            raise ValueError("bankroll must be positive")
        if not 0 < self.max_fraction <= 1:
            raise ValueError("max_fraction must be in (0, 1]")
        if not 0 <= self.fraction_kelly <= 1:
            raise ValueError("fraction_kelly must be in [0, 1]")
        if not 0 < self.optimizer_alpha < 1:
            raise ValueError("optimizer_alpha must be in (0, 1)")
        if self.optimizer_iters <= 0:
            raise ValueError("optimizer_iters must be positive")
        if self.min_stake < 0:
            raise ValueError("min_stake must be non-negative")


@dataclass
class BacktestResult:
    """Long-form outputs of one walk-forward run."""

    windows: pd.DataFrame
    predictions: pd.DataFrame
    bets: pd.DataFrame


def _normalise_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    if "season" in frame and frame["season"].dropna().nunique() > 1:
        raise ValueError("run_backtest accepts one season at a time")
    required = {"home_team", "away_team", "fthg", "ftag"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required match columns: {sorted(missing)}")

    if "kickoff" in frame:
        kickoff = pd.to_datetime(frame["kickoff"], errors="raise")
    elif "date" in frame:
        dates = pd.to_datetime(frame["date"], dayfirst=True, errors="raise")
        if "time" in frame:
            kickoff = pd.to_datetime(
                dates.dt.strftime("%Y-%m-%d") + " " + frame["time"].fillna("00:00").astype(str),
                errors="raise",
            )
        else:
            kickoff = dates
    else:
        raise ValueError("Data must contain either 'kickoff' or 'date'")

    frame["kickoff"] = kickoff.dt.tz_localize(None) if kickoff.dt.tz is not None else kickoff
    if "match_id" not in frame:
        frame["match_id"] = (
            frame["home_team"].astype(str)
            + " - "
            + frame["away_team"].astype(str)
            + " - "
            + frame["kickoff"].dt.strftime("%Y-%m-%d")
        )
    return frame.sort_values("kickoff", kind="stable").reset_index(drop=True)


def _friday_cutoffs(frame: pd.DataFrame, config: BacktestConfig) -> Iterable[pd.Timestamp]:
    first = frame["kickoff"].min().normalize()
    days_since_cutoff = (first.dayofweek - config.cutoff_weekday) % 7
    cutoff = first - pd.Timedelta(days=days_since_cutoff)
    last = frame["kickoff"].max()
    while cutoff <= last:
        yield cutoff
        cutoff += pd.Timedelta(days=7)


def _canonical_market(market: str) -> CanonicalMarket:
    if market == "1X2":
        return "1X2"
    if market in {"O/U2.5", "U2.5", "O2.5"}:
        return "O/U2.5"
    raise ValueError(f"Unsupported market: {market}")


def _canonical_markets(markets: Iterable[Market]) -> tuple[CanonicalMarket, ...]:
    result: list[CanonicalMarket] = []
    for market in markets:
        canonical = _canonical_market(market)
        if canonical not in result:
            result.append(canonical)
    return tuple(result)


def _market_probabilities(prediction: Any, market: CanonicalMarket) -> np.ndarray:
    if market == "1X2":
        if isinstance(prediction, GoalMatrix):
            result = prediction.return_probas()
        elif isinstance(prediction, ProbaResult):
            result = prediction
        else:
            raise TypeError("1X2 predictions must be GoalMatrix or ProbaResult")
        return np.asarray([result.proba_home, result.proba_draw, result.proba_away], dtype=float)

    if not isinstance(prediction, GoalMatrix):
        raise TypeError(f"{market} requires a GoalMatrix prediction")
    under = prediction.less_25_goals()
    return np.asarray([under, 1.0 - under], dtype=float)


def _actual_index(row: pd.Series, market: CanonicalMarket) -> int:
    if market == "1X2":
        result = str(row.get("ftr", "")).upper()
        if result not in {"H", "D", "A"}:
            result = (
                "H" if row["fthg"] > row["ftag"] else "D" if row["fthg"] == row["ftag"] else "A"
            )
        return {"H": 0, "D": 1, "A": 2}[result]
    over = float(row["fthg"]) + float(row["ftag"]) > 2.5
    return int(over)


def _odds(row: pd.Series, selection: str, columns: Mapping[str, str]) -> float | None:
    keys = [columns.get(selection, "")]
    aliases = {
        "H": ("b365_h", "B365H"),
        "D": ("b365_d", "B365D"),
        "A": ("b365_a", "B365A"),
        "U2.5": ("b365<2.5", "B365<2.5"),
        "O2.5": ("b365>2.5", "B365>2.5"),
    }
    keys.extend(aliases.get(selection, ()))
    for key in keys:
        if key in row.index:
            value = pd.to_numeric(row[key], errors="coerce")
            if pd.notna(value) and np.isfinite(value) and float(value) > 1.0:
                return float(value)
    return None


def _probability_sample(
    spec: ModelSpec, model: Any, home: str, away: str, market: CanonicalMarket, index: int
) -> tuple[float | None, float | None]:
    if spec.samples is None:
        return None, None
    samples = np.asarray(spec.samples(model, home, away, market), dtype=float)
    category_count = samples.shape[1] if samples.ndim == 2 else 3
    if samples.ndim == 2:
        if index >= samples.shape[1]:
            return None, None
        samples = samples[:, index]
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return None, None
    std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    return std, float(np.mean(samples > 1.0 / category_count))


def _bet_candidates(
    prediction_rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    config: BacktestConfig,
) -> list[Bet]:
    by_match = {str(row["match_id"]): row for _, row in frame.iterrows()}
    candidates: list[Bet] = []
    for match_id, rows in _group_prediction_rows(prediction_rows).items():
        row = by_match[match_id]
        choices: list[Bet] = []
        for prediction in rows:
            odds = _odds(row, prediction["selection"], config.odds_columns)
            if odds is None:
                continue
            edge = prediction["probability"] * odds - 1.0
            if edge <= config.edge_floor:
                continue
            choices.append(
                Bet(
                    match_id=match_id,
                    market=prediction["selection"],
                    odds=odds,
                    prob_mean=prediction["probability"],
                    edge_std=prediction.get("edge_std"),
                    prob_edge_pos=prediction.get("prob_edge_pos"),
                )
            )
        if choices:
            candidates.append(max(choices, key=lambda bet: bet.edge_mean))
    return candidates


def _group_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["match_id"]), []).append(row)
    return grouped


def _fit(spec: ModelSpec, model: Any, train: pd.DataFrame) -> None:
    if spec.fit is None:
        model.fit(train)
    else:
        spec.fit(model, train)


def _model_bets(
    candidates: list[Bet], spec: ModelSpec, bankroll: float, config: BacktestConfig
) -> list[Bet]:
    if not candidates or spec.staking == "none":
        return candidates
    if spec.staking == "portfolio_torch":
        return optimise_portfolio_torch(
            candidates,
            bankroll=bankroll,
            max_fraction=config.max_fraction,
            alpha=config.optimizer_alpha,
            gamma=config.optimizer_gamma,
            lr=config.optimizer_lr,
            iters=config.optimizer_iters,
            penalty_lambda=config.optimizer_penalty_lambda,
            verbose=False,
        )
    return fractional_kelly(
        candidates,
        bankroll=bankroll,
        fraction_kelly=config.fraction_kelly,
        max_fraction=config.max_fraction,
        min_stake=config.min_stake,
    )


def run_backtest(
    data: pd.DataFrame, model_specs: Iterable[ModelSpec], config: BacktestConfig | None = None
) -> BacktestResult:
    """Run a strict expanding-window evaluation on one competition/season."""
    config = config or BacktestConfig()
    frame = _normalise_frame(data)
    specs = tuple(model_specs)
    if not specs:
        raise ValueError("At least one ModelSpec is required")

    windows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    bets: list[dict[str, Any]] = []
    bankrolls = {spec.name: config.bankroll for spec in specs}
    configured_markets = _canonical_markets(config.markets)

    for cutoff in _friday_cutoffs(frame, config):
        window_end = cutoff + pd.Timedelta(days=config.horizon_days)
        train = frame.loc[frame["kickoff"] < cutoff]
        target = frame.loc[(frame["kickoff"] >= cutoff) & (frame["kickoff"] < window_end)]
        if target.empty:
            continue

        for spec in specs:
            bankroll_before = bankrolls[spec.name]
            window_base = {
                "model": spec.name,
                "cutoff": cutoff,
                "window_end": window_end,
                "train_matches": len(train),
                "target_matches": len(target),
                "bankroll_before": bankroll_before,
                "final_window": window_end > frame["kickoff"].max(),
            }
            if train.empty and spec.requires_training:
                windows.append(
                    {**window_base, "status": "ineligible", "reason": "no_training_data"}
                )
                continue

            model = spec.factory()
            try:
                if spec.requires_training:
                    _fit(spec, model, train)
            except Exception as exc:
                windows.append(
                    {
                        **window_base,
                        "status": "ineligible",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            window_predictions: list[dict[str, Any]] = []
            errors = 0
            for _, match in target.iterrows():
                for market in configured_markets:
                    if market not in {_canonical_market(value) for value in spec.markets}:
                        continue
                    try:
                        prediction = model.predict(match["home_team"], match["away_team"])
                        probabilities = _market_probabilities(prediction, market)
                        actual_idx = _actual_index(match, market)
                        metric_row = {
                            "model": spec.name,
                            "cutoff": cutoff,
                            "window_end": window_end,
                            "kickoff": match["kickoff"],
                            "match_id": str(match["match_id"]),
                            "home_team": match["home_team"],
                            "away_team": match["away_team"],
                            "market": market,
                            "selection": None,
                            "probability": float(probabilities.max()),
                            "probabilities": probabilities.tolist(),
                            "actual": actual_idx,
                            "train_matches": len(train),
                            "rps": rps(probabilities, actual_idx),
                            "log_loss": log_loss(probabilities, actual_idx),
                            "brier": brier_score(probabilities, actual_idx),
                            "accuracy": accuracy(probabilities, actual_idx),
                            "edge_std": None,
                            "prob_edge_pos": None,
                        }
                        predictions.append(metric_row)
                        market_labels = {
                            "1X2": ("H", "D", "A"),
                            "O/U2.5": ("U2.5", "O2.5"),
                        }
                        for idx, label in enumerate(market_labels[market]):
                            prediction_row = {
                                **metric_row,
                                "selection": label,
                                "probability": float(probabilities[idx]),
                            }
                            sample_std, prob_edge_pos = _probability_sample(
                                spec,
                                model,
                                match["home_team"],
                                match["away_team"],
                                market,
                                idx,
                            )
                            prediction_row["edge_std"] = sample_std
                            prediction_row["prob_edge_pos"] = prob_edge_pos
                            window_predictions.append(prediction_row)
                    except Exception:
                        errors += 1

            candidates = _bet_candidates(window_predictions, target, config)
            selected = _model_bets(candidates, spec, bankroll_before, config)
            selected_by_key = {(bet.match_id, bet.market): bet for bet in selected}
            profit = 0.0
            total_stake = 0.0
            for candidate in candidates:
                bet = selected_by_key.get((candidate.match_id, candidate.market), candidate)
                stake = float(bet.stake)
                if stake <= 0:
                    continue
                match = target.loc[target["match_id"].astype(str) == bet.match_id].iloc[0]
                bet_market: CanonicalMarket = "1X2" if bet.market in {"H", "D", "A"} else "O/U2.5"
                won = _actual_index(match, bet_market) == {
                    "H": 0,
                    "D": 1,
                    "A": 2,
                    "U2.5": 0,
                    "O2.5": 1,
                }.get(bet.market, 0)
                profit += stake * (bet.odds - 1.0 if won else -1.0)
                total_stake += stake
                bets.append(
                    {
                        "model": spec.name,
                        "cutoff": cutoff,
                        "window_end": window_end,
                        "match_id": bet.match_id,
                        "market": bet_market,
                        "selection": bet.market,
                        "odds": bet.odds,
                        "probability": bet.prob_mean,
                        "edge": bet.edge_mean,
                        "edge_std": bet.edge_std,
                        "stake": stake,
                        "won": won,
                        "profit": stake * (bet.odds - 1.0 if won else -1.0),
                    }
                )
            bankroll_after = bankroll_before + profit
            bankrolls[spec.name] = bankroll_after
            windows.append(
                {
                    **window_base,
                    "status": "ok",
                    "reason": None,
                    "predicted_rows": len(window_predictions),
                    "prediction_errors": errors,
                    "bets": len([bet for bet in selected if bet.stake > 0]),
                    "total_stake": total_stake,
                    "profit": profit,
                    "bankroll_after": bankroll_after,
                }
            )

    return BacktestResult(
        windows=pd.DataFrame(windows),
        predictions=pd.DataFrame(predictions),
        bets=pd.DataFrame(bets),
    )


def poisson_spec(n_teams: int, n_goals: int = 20) -> ModelSpec:
    """Create a Poisson model specification covering 1X2 and O/U 2.5."""
    from footix.models.basic_poisson import PoissonModel

    return ModelSpec(
        name="poisson",
        factory=lambda: PoissonModel(n_teams=n_teams, n_goals=n_goals),
        markets=("1X2", "O/U2.5"),
    )


def elo_spec(**kwargs: Any) -> ModelSpec:
    """Create an Elo model specification covering 1X2."""
    from footix.models.elo import EloDavidson

    return ModelSpec(
        name="elo",
        factory=lambda: EloDavidson(**kwargs),
        markets=("1X2",),
    )


def bayesian_spec(
    n_goals: int = 20,
    n_teams: int | None = None,
    calibrate: bool = False,
    use_stats: bool = False,
    random_seed: int | None = 42,
) -> ModelSpec:
    """Create a Bayesian specification with posterior samples for all supported markets."""
    from footix.models.bayesian import BayesianModel

    def samples(model: BayesianModel, home: str, away: str, market: Market) -> np.ndarray:
        if market == "1X2":
            result = model.get_samples(home, away)
            return np.column_stack((result.proba_home, result.proba_draw, result.proba_away))
        return model.get_market_samples(home, away, market)

    return ModelSpec(
        name="bayesian",
        factory=lambda: BayesianModel(
            n_goals=n_goals,
            n_teams=n_teams,
            calibrate=calibrate,
            use_stats=use_stats,
            random_seed=random_seed,
        ),
        markets=("1X2", "O/U2.5"),
        staking="portfolio_torch",
        samples=samples,
    )


class _UniformModel:
    def fit(self, data: pd.DataFrame) -> None:
        _ = data

    def predict(self, home_team: str, away_team: str) -> ProbaResult:
        _ = home_team, away_team
        return ProbaResult(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def uniform_spec(name: str = "uniform") -> ModelSpec:
    """Create a data-free 1X2 baseline for comparison."""
    return ModelSpec(
        name=name,
        factory=_UniformModel,
        markets=("1X2",),
        staking="none",
        requires_training=False,
    )
