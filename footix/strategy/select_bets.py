import numpy as np

import footix.utils.decorators as decorators
from footix.strategy._utils import _skellam_post_probs
from footix.strategy.bets import Bet, OddsInput


@decorators.verify_required_column(column_names={"home_team", "away_team", "H", "D", "A"})
def simple_select_bets(
    odds_input: list[OddsInput],
    probas: np.ndarray,
    edge_floor: float = 0.0,
    single_bet_per_game: bool = True,
) -> list[Bet]:
    """Select bets with positive expected value (p > 1/odds).

    Args:
        odds (pd.DataFrame): DataFrame with columns ['home_team', 'away_team', 'H', 'D', 'A'].
        probas (np.ndarray): Array of shape (n_matches, 3) containing predicted probabilities.
        single_bet_per_game (bool): If True, only the highest-edge bet per match is kept.

    Returns:
        List[Bet]: A list of Bet objects with positive edge.

    """
    outcomes = ["H", "D", "A"]
    n_matches = len(odds_input)
    if probas.shape != (n_matches, 3):
        raise ValueError(f"probas must have shape ({n_matches}, 3), got {probas.shape}")

    selections: list[Bet] = []
    for idx, odd in enumerate(odds_input):
        odds_arr = np.asarray(odd.odds)

        # Compute expected edge for each outcome: edge = p*(odds-1) + (p-1)
        edges = probas[idx] * (odds_arr - 1) + (probas[idx] - 1)

        if single_bet_per_game:
            best_idx = int(np.argmax(edges))
            if edges[best_idx] > edge_floor:
                selections.append(
                    _build_bet(
                        odd,
                        outcomes=outcomes,
                        pick=best_idx,
                        edge=edges[best_idx],
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
                        edge=edges[pick],
                        prob=probas[idx, pick],
                    )
                )

    return selections


def _build_bet(
    odd_input: OddsInput,
    outcomes: list[str],
    pick: int,
    edge: float,
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
        edge_mean=edge,
        prob_mean=prob,
    )


def select_matches_posterior(
    odds_input: list[OddsInput],
    lambda_samples: dict[str, tuple[np.ndarray, np.ndarray]],
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
    selected: list[Bet] = []

    for odd in odds_input:
        lam_h, lam_a = lambda_samples[odd.match_id]
        p_home, p_draw, p_away = _skellam_post_probs(lam_h, lam_a)

        candidate_bets = []
        for market, p_samples in zip(("H", "D", "A"), (p_home, p_draw, p_away)):
            o = odd.odd_dict[market]
            edge_samples = p_samples * (o - 1) - (1 - p_samples)

            mu_edge = edge_samples.mean()
            std_edge = edge_samples.std(ddof=1)
            prob_pos = (edge_samples > 0).mean()
            p_mean = p_samples.mean()

            if mu_edge > edge_floor and prob_pos > prob_edge_threshold:
                candidate_bets.append(
                    Bet(
                        match_id=odd.match_id,
                        market=market,
                        odds=o,
                        edge_mean=mu_edge,
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
