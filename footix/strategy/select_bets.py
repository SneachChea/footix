from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Sequence

import numpy as np

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
) -> list[Bet]:
    outcomes = ["H", "D", "A"]
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
    outcomes: list[str],
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
