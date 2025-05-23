from footix.strategy.bets import Bet


def flat_staking(list_bets: list[Bet], bankroll: float, fraction_bankroll: float) -> list[Bet]:
    if fraction_bankroll * len(list_bets) > 1.0:
        raise ValueError("Too many bets for the stake")

    for bet in list_bets:
        bet.stake = fraction_bankroll * bankroll

    return list_bets
