from footix.strategy.bets import Bet


def flat_staking(list_bets: list[Bet], bankroll: float, fraction_bankroll: float) -> list[Bet]:
    """Allocate a fixed portion of the bankroll to each bet.

    The stake for every bet in ``list_bets`` is set to
    ``fraction_bankroll * bankroll``.  This simple staking strategy
    assumes that all bets are independent and that the same fraction of
    the bankroll is used for each one.

    Args:
        list_bets: A list of :class:`~footix.strategy.bets.Bet` objects whose
            ``stake`` attribute will be updated.
        bankroll: The total amount of money available to stake.
        fraction_bankroll: The fraction of the bankroll to allocate to each
            bet.  Must satisfy ``fraction_bankroll * len(list_bets) <= 1``.

    Returns:
        list[Bet]: The input list with the ``stake`` attribute updated for
        each bet.

    Raises:
        ValueError: If the total required stake exceeds the available bankroll.

    """
    if fraction_bankroll * len(list_bets) > 1.0:
        raise ValueError("Too many bets for the stake")

    for bet in list_bets:
        bet.stake = fraction_bankroll * bankroll

    return list_bets
