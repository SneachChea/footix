from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from footix.strategy.bets import Bet, OddsInput
from footix.utils.typing import SampleProbaResult


class Thresholds(NamedTuple):
    edge_floor: float
    prob_edge: Optional[float] = None


class OddsRange(NamedTuple):
    """Represents an odds range and its corresponding edge and probability thresholds.

    Attributes:
        min_odds: Minimum odds value (inclusive)
        max_odds: Maximum odds value (inclusive)
        edge: Required edge floor for this odds range
        prob_edge: Required probability of positive edge (optional)

    """

    min_odds: float
    max_odds: float
    edge: float
    prob_edge: Optional[float] = None


@dataclass(slots=True)
class EdgeFloorConfig:
    ranges: Sequence[OddsRange] = field(default_factory=list)
    default_edge_floor: float = 0.0
    default_prob_edge: Optional[float] = None

    def get_thresholds(self, odds: float) -> Thresholds:
        """Return edge-floor thresholds for the given odds."""
        # Find the first matching range, if any
        match = next(
            (r for r in self.ranges if r.min_odds <= odds <= r.max_odds),
            None,
        )

        if match is not None:
            return Thresholds(
                edge_floor=match.edge,
                prob_edge=match.prob_edge or self.default_prob_edge,
            )

        # No range matched – fall back to defaults
        return Thresholds(self.default_edge_floor, self.default_prob_edge)


def simple_select_bets(
    odds_input: list[OddsInput],
    probas: np.ndarray,
    edge_floor: float | EdgeFloorConfig = 0.0,
    single_bet_per_game: bool = True,
    outcomes: Sequence[str] = ("H", "D", "A"),
) -> list[Bet]:
    n_matches = len(odds_input)
    if probas.shape != (n_matches, 3):
        raise ValueError(f"probas must have shape ({n_matches}, 3), got {probas.shape}")

    if isinstance(edge_floor, float):
        edge_config = EdgeFloorConfig(default_edge_floor=edge_floor)
    elif isinstance(edge_floor, EdgeFloorConfig):
        edge_config = edge_floor
    else:
        raise TypeError("Edge floor should be either a float or an EdgeFloorConfig instance")

    selections: list[Bet] = []
    for idx, odd in enumerate(odds_input):
        odds_arr = np.asarray(odd.odds)

        # Compute expected edge for each outcome: edge = p*(odds-1) + (p-1)
        edges = probas[idx] * (odds_arr - 1) + (probas[idx] - 1)

        if single_bet_per_game:
            best_idx = int(np.argmax(edges))
            required_edges, _ = edge_config.get_thresholds(odds_arr[best_idx])
            if edges[best_idx] > required_edges:
                selections.append(
                    _build_bet(
                        odd,
                        outcomes=outcomes,
                        pick=best_idx,
                        prob=probas[idx, best_idx],
                    )
                )
        else:
            # Add every outcome with positive edge
            for pick in np.where(edges > edge_floor)[0]:
                selections.append(
                    _build_bet(
                        odd,
                        outcomes=outcomes,
                        pick=int(pick),
                        prob=probas[idx, pick],
                    )
                )

    return selections


def _build_bet(
    odd_input: OddsInput,
    outcomes: Sequence[str],
    pick: int,
    prob: float,
) -> Bet:
    """Construct a Bet object from the row data and computed metrics.

    Args:
        row (pd.Series): One row from the odds DataFrame.
        outcomes (list[str]): List of outcome labels, e.g. ["H","D","A"].
        odds_arr (np.ndarray): Array of odds for the three outcomes.
        pick (int): Index of the chosen outcome (0,1,2).
        edge (float): Expected edge for the chosen outcome.
        prob (float): Predicted probability for the chosen outcome.

    Returns:
        Bet: Initialized with match_id, market, odds, edge_mean, and prob_mean.

    """
    return Bet(
        match_id=odd_input.match_id,
        market=outcomes[pick],
        odds=odd_input.odds[pick],
        prob_mean=prob,
    )


def select_matches_posterior(
    odds_input: list[OddsInput],
    lambda_samples: dict[str, SampleProbaResult],
    *,
    config: EdgeFloorConfig | None = None,
    edge_floor: float = 0.1,
    prob_edge_threshold: float = 0.55,
    single_bet_per_game: bool = True,
) -> list[Bet]:
    """Select bets based on posterior probabilities computed from the Skellam distribution.

    For each match, posterior probabilities for the home-win, draw, and
    away-win outcomes are computed. The expected edge is calculated for each
    outcome. Bets are only selected if the mean edge exceeds
    the specified edge_floor and the probability of a positive edge is above
    the prob_edge_threshold. If single_bet_per_game is True, only the bet
    with the highest mean edge is kept per match.

    Args:
        odds_input (list[OddsInput]): List of odds input objects.
        lambda_samples (dict[str, tuple[np.ndarray, np.ndarray]]):
            Dictionary mapping match_id to lambda samples (home and away)
            used for posterior probability computation.
        edge_floor (float, optional): Minimum required mean edge to consider a bet.
        Defaults to 0.1.
        prob_edge_threshold (float, optional): Minimum probability of positive edge to
            consider a bet. Defaults to 0.55.
        single_bet_per_game (bool, optional): If True, only the best bet per match is
        selected. Defaults to True.

    Returns:
        list[Bet]: A sorted list of selected Bet objects, ordered by descending edge_mean.

    """
    if config is None:
        config = EdgeFloorConfig(
            default_edge_floor=edge_floor, default_prob_edge=prob_edge_threshold
        )

    selected: list[Bet] = []

    for odd in odds_input:
        p_home, p_draw, p_away = lambda_samples[odd.match_id]
        candidate_bets = []
        for market, p_samples in zip(("H", "D", "A"), (p_home, p_draw, p_away)):
            o = odd.odd_dict[market]
            edge_samples = p_samples * (o - 1.0) - (1.0 - p_samples)

            mu_edge = edge_samples.mean()
            std_edge = edge_samples.std(ddof=1)
            prob_pos = (edge_samples > 0).mean()
            p_mean = p_samples.mean()
            edge_thresholds = config.get_thresholds(odds=o)
            if mu_edge > edge_thresholds.edge_floor and prob_pos > edge_thresholds.prob_edge:
                candidate_bets.append(
                    Bet(
                        match_id=odd.match_id,
                        market=market,
                        odds=o,
                        edge_std=std_edge,
                        prob_edge_pos=prob_pos,
                        prob_mean=p_mean,
                    )
                )
        if candidate_bets:
            if single_bet_per_game:
                best_bet = max(candidate_bets, key=lambda b: b.edge_mean)
                selected.append(best_bet)
            else:
                selected.extend(candidate_bets)

    return sorted(selected, key=lambda b: b.edge_mean, reverse=True)


SELECTIONS = {"H", "D", "A", "U2.5", "O2.5"}


@dataclass(frozen=True)
class _BetStats:
    edge_samples: np.ndarray
    q_edge: float
    mean_edge: float
    prob_edge_pos: float
    rob_kelly: float


def _validate_selection(market: str) -> None:
    if market not in SELECTIONS:
        raise ValueError(
            f"Bet market must be a selection ('H', 'D', 'A', 'U2.5', 'O2.5'), got {market!r}"
        )


def _score_candidates(
    candidates: Sequence[Bet],
    probability_samples: Mapping[tuple[str, str], np.ndarray],
    alpha: float,
) -> dict[tuple[str, str], _BetStats]:
    """Compute posterior edge statistics per ``(match_id, market)`` key.

    ``Bet.market`` holds the selection itself (e.g. "H" or "U2.5"), so the
    key is unique per selection. A canonical market name like "1X2" or
    "O/U2.5" is rejected here to avoid silently collapsing distinct
    selections onto one key.

    """
    stats: dict[tuple[str, str], _BetStats] = {}
    for bet in candidates:
        _validate_selection(bet.market)
        key = (bet.match_id, bet.market)
        if key in stats:
            raise ValueError(f"Duplicate candidate key: {key}")
        samples = probability_samples.get(key)
        if samples is None:
            raise KeyError(f"No probability samples for bet {key}")
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError(f"Probability samples must be a non-empty 1D array for {key}")
        edge = bet.odds * samples - 1.0
        q_prob = float(np.quantile(samples, alpha))
        stats[key] = _BetStats(
            edge_samples=edge,
            q_edge=float(np.quantile(edge, alpha)),
            mean_edge=float(np.mean(edge)),
            prob_edge_pos=float(np.mean(edge > 0.0)),
            rob_kelly=max((bet.odds * q_prob - 1.0) / (bet.odds - 1.0), 0.0),
        )
    return stats


def _rho_by_key(
    stats: dict[tuple[str, str], _BetStats],
) -> dict[tuple[str, str], float]:
    """Share of draws in which each bet is the best edge of its match.

    The comparison is draw by draw, so every selection of a match must share
    the same posterior draw axis.

    """
    by_match: dict[str, list[tuple[str, str]]] = {}
    for key in stats:
        by_match.setdefault(key[0], []).append(key)

    rho: dict[tuple[str, str], float] = {}
    for match_id, keys in by_match.items():
        lengths = {stats[key].edge_samples.shape[0] for key in keys}
        if len(lengths) != 1:
            raise ValueError(
                f"Probability samples for match {match_id} must share the same draw axis"
            )
        edge_matrix = np.column_stack([stats[key].edge_samples for key in keys])
        argmax = np.argmax(edge_matrix, axis=1)
        for position, key in enumerate(keys):
            rho[key] = float(np.mean(argmax == position))
    return rho


def select_bets_posterior(
    candidates: Sequence[Bet],
    probability_samples: Mapping[tuple[str, str], np.ndarray],
    *,
    alpha: float = 0.10,
    delta: float = 0.0,
    rho_min: float = 0.60,
) -> list[Bet]:
    """Select bets from a conservative bound on their profitability.

    For every candidate bet the full posterior distribution of the edge
    ``e = odds * p - 1`` is built from the posterior probability samples.
    A bet is kept only when its lower bound ``Q_alpha(e)`` is above
    ``delta``, i.e. the model still considers the bet profitable at a
    pessimistic quantile. Within a match the bet with the largest robust
    Kelly fraction ``[(odds * Q_alpha(p) - 1) / (odds - 1)]_+`` is chosen,
    provided it is the best bet of the match in more than ``rho_min`` of
    the posterior draws (ambiguity filter). At most one bet per match is
    returned; stakes are left to the portfolio optimiser.

    Args:
        candidates (Sequence[Bet]): All bet opportunities, several per
            match allowed. ``market`` must hold the selection itself
            ("H", "D", "A", "U2.5", "O2.5"), not the canonical market.
        probability_samples (Mapping[tuple[str, str], np.ndarray]):
            Posterior probability samples per ``(match_id, market)`` key,
            each of shape ``(n_draws,)``. All selections of a match must
            share the same draw axis.
        alpha (float, optional): Quantile level of the pessimistic bound.
            Defaults to 0.10.
        delta (float, optional): Minimum value of the lower edge bound.
            Defaults to 0.0.
        rho_min (float, optional): Minimum share of draws in which the
            selected bet is the best edge of its match. Defaults to 0.60.

    Returns:
        list[Bet]: Selected bets, at most one per match.

    Raises:
        ValueError: If parameters are invalid, a candidate has no
            probability samples, or samples are misaligned.

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if delta < 0:
        raise ValueError("delta must be non-negative")
    if not 0 < rho_min <= 1:
        raise ValueError("rho_min must be in (0, 1]")

    stats = _score_candidates(candidates, probability_samples, alpha)
    rho = _rho_by_key(stats)

    by_match: dict[str, list[Bet]] = {}
    for bet in candidates:
        by_match.setdefault(bet.match_id, []).append(bet)

    selected: list[Bet] = []
    for match_bets in by_match.values():
        eligible = [b for b in match_bets if stats[(b.match_id, b.market)].q_edge > delta]
        if not eligible:
            continue
        best = max(eligible, key=lambda b: stats[(b.match_id, b.market)].rob_kelly)
        if rho[(best.match_id, best.market)] > rho_min:
            selected.append(best)

    return sorted(selected, key=lambda b: b.edge_mean, reverse=True)


def select_bets_diagnostics(
    candidates: Sequence[Bet],
    probability_samples: Mapping[tuple[str, str], np.ndarray],
    *,
    alpha: float = 0.10,
    delta: float = 0.0,
    rho_min: float = 0.60,
) -> pd.DataFrame:
    """Per-candidate selection diagnostics without dropping any row.

    Returns one row per candidate with the posterior edge statistics, the
    ambiguity share ``rho`` and the reason a candidate was rejected
    ("edge_bound", "not_best", "ambiguity") or ``None`` when selected.
    Intended for walk-forward tuning of ``alpha``, ``delta`` and
    ``rho_min`` before committing to a configuration.

    Args:
        Same as :func:`select_bets_posterior`.

    Returns:
        pd.DataFrame: Columns ``match_id``, ``market`` (canonical), ``selection``,
            ``q_edge``, ``mean_edge``, ``prob_edge_positive``, ``robust_kelly``,
            ``rho``, ``passed_edge_filter``, ``passed_ambiguity_filter``,
            ``rejection_reason``.

    """
    stats = _score_candidates(candidates, probability_samples, alpha)
    rho = _rho_by_key(stats)

    by_match: dict[str, list[Bet]] = {}
    for bet in candidates:
        by_match.setdefault(bet.match_id, []).append(bet)

    rows: list[dict[str, object]] = []
    for match_bets in by_match.values():
        eligible = [b for b in match_bets if stats[(b.match_id, b.market)].q_edge > delta]
        best = (
            max(eligible, key=lambda b: stats[(b.match_id, b.market)].rob_kelly)
            if eligible
            else None
        )
        for bet in match_bets:
            key = (bet.match_id, bet.market)
            s = stats[key]
            passed_edge = s.q_edge > delta
            passed_ambiguity = rho[key] > rho_min
            if not passed_edge:
                reason = "edge_bound"
            elif best is not bet:
                reason = "not_best"
            elif not passed_ambiguity:
                reason = "ambiguity"
            else:
                reason = None
            rows.append(
                {
                    "match_id": bet.match_id,
                    "market": "1X2" if bet.market in {"H", "D", "A"} else "O/U2.5",
                    "selection": bet.market,
                    "q_edge": s.q_edge,
                    "mean_edge": s.mean_edge,
                    "prob_edge_positive": s.prob_edge_pos,
                    "robust_kelly": s.rob_kelly,
                    "rho": rho[key],
                    "passed_edge_filter": passed_edge,
                    "passed_ambiguity_filter": passed_ambiguity,
                    "rejection_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def select_bets_by_probability(
    odds_input: list[OddsInput],
    probas: np.ndarray,
    prob_floor: float = 0.55,
    single_bet_per_game: bool = True,
) -> list[Bet]:
    """Select bets based on the highest predicted probabilities.

    For each match, outcomes with predicted probability greater than or equal to
    prob_floor are considered. If single_bet_per_game is True, only the outcome with the
    highest probability (if it meets the threshold) is selected per match.
    Otherwise, every outcome meeting the threshold is selected.

    Args:
        odds_input (list[OddsInput]): List of odds input objects.
        probas (np.ndarray): Array of shape (n_matches, 3) containing predicted probabilities.
        prob_floor (float, optional): Minimum acceptable probability for a bet. Defaults to 0.55.
        single_bet_per_game (bool, optional): If True, only the most probable bet per match is
                                            selected. Defaults to True.

    Returns:
        list[Bet]: A list of Bet objects selected based on the highest probability.

    """
    outcomes = ["H", "D", "A"]
    n_matches = len(odds_input)
    if probas.shape != (n_matches, 3):
        raise ValueError(f"probas must have shape ({n_matches}, 3), got {probas.shape}")

    selections: list[Bet] = []
    for idx, odd in enumerate(odds_input):
        p = probas[idx]
        if single_bet_per_game:
            best_idx = int(np.argmax(p))
            if p[best_idx] >= prob_floor:
                selections.append(
                    _build_bet(odd, outcomes=outcomes, pick=best_idx, prob=p[best_idx])
                )
        else:
            # Add every outcome with predicted probability above or equal to the threshold.
            for pick in np.where(p >= prob_floor)[0]:
                selections.append(_build_bet(odd, outcomes=outcomes, pick=int(pick), prob=p[pick]))

    return selections
