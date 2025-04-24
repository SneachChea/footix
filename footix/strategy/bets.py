from dataclasses import asdict, dataclass


@dataclass
class Bet:
    """Represents a single betting opportunity with associated edge information.

    Attributes:
        match_id (str): Identifier for the match.
        market (str): Market selection — 'H' for home, 'D' for draw, 'A' for away.
        odds (float): Decimal odds offered by the bookmaker.
        edge_mean (float): Estimated edge over the bookmaker. Computed as (p*(odds-1) - (1-p))
        prob_mean (float): Estimated probability of the event occurring based on the model.
        edge_std (Optional[float]): Standard deviation of the edge estimate.
        prob_edge_pos (Optional[float]): Probability that the edge is positive (i.e., a value bet).
        stake (Optional[float]): The stake of the bet (if selected)

    """

    match_id: str
    market: str
    odds: float
    edge_mean: float
    prob_mean: float
    edge_std: float | None = None
    prob_edge_pos: float | None = None
    stake: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[{self.match_id} | {self.market}] "
            f"odds={self.odds:.2f}, edge={self.edge_mean:.3f}, "
            f"p={self.prob_mean:.3f}, stake={self.stake or 0:.2f}"
        )

    def __repr__(self) -> str:
        return (
            f"Bet(match_id={self.match_id!r}, market={self.market!r}, odds={self.odds}, "
            f"edge_mean={self.edge_mean}, prob_mean={self.prob_mean}, edge_std={self.edge_std}, "
            f"prob_edge_pos={self.prob_edge_pos}, stake={self.stake})"
        )

    @classmethod
    def combine_many(cls, bets: list["Bet"]) -> "Bet":
        """Combines multiple independent bets into a single combined bet (accumulator).

        Args:
            bets (list[Bet]): List of Bet instances to combine.

        Returns:
            Bet: A new Bet representing the combined bet.

        """
        if not bets:
            raise ValueError("Cannot combine an empty list of bets.")

        combined_odds = 1.0
        combined_prob = 1.0
        match_ids = []
        markets = []

        for bet in bets:
            combined_odds *= bet.odds
            combined_prob *= bet.prob_mean
            match_ids.append(bet.match_id)
            markets.append(bet.market)

        combined_edge = combined_prob * (combined_odds - 1) - (1 - combined_prob)

        return cls(
            match_id=" + ".join(match_ids),
            market=" + ".join(markets),
            odds=combined_odds,
            edge_mean=combined_edge,
            prob_mean=combined_prob,
            edge_std=None,
            prob_edge_pos=None,
            stake=None,
        )

    def __add__(self, other: "Bet") -> "Bet":
        """Allows combining two bets using the + operator.

        Returns:
            Bet: A new Bet representing the combined (accumulator) bet.

        """
        if not isinstance(other, Bet):
            return NotImplemented
        return Bet.combine_many([self, other])

    def __iadd__(self, other: "Bet") -> "Bet":
        """Supports the += operator to combine this bet with another.

        Returns:
            Bet: A new Bet instance representing the combined bet.

        """
        if not isinstance(other, Bet):
            return NotImplemented
        return self + other

    def __eq__(self, bet: "Bet") -> bool:
        if (self.match_id == bet.match_id) and (self.market == bet.market):
            return True
        return False


@dataclass
class OddsInput:
    home_team: str
    away_team: str
    odds: list[float]  # in the format [H, D, A]

    @property
    def odd_dict(self) -> dict[str, float]:
        return {"H": self.odds[0], "D": self.odds[1], "A": self.odds[2]}

    @property
    def match_id(self) -> str:
        return f"{self.home_team} - {self.away_team}"
