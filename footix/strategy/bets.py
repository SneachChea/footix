from dataclasses import asdict, dataclass, field


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
        stake (float): The stake of the bet. By default stake = 0.

    """

    match_id: str
    market: str
    odds: float
    prob_mean: float
    edge_std: float | None = None
    prob_edge_pos: float | None = None
    stake: float = 0.0
    edge_mean: float = field(init=False)

    def __post_init__(self):
        self.edge_mean = self.prob_mean * (self.odds - 1) - (1.0 - self.prob_mean)

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

        return cls(
            match_id=" + ".join(match_ids),
            market=" + ".join(markets),
            odds=combined_odds,
            prob_mean=combined_prob,
            edge_std=None,
            prob_edge_pos=None,
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

    def __eq__(self, other: object) -> bool:
        """Determines if two Bet objects are equal. other (object): The object to compare with the
        current Bet instance.

        NotImplementedError: If the `other` object is not an instance of Bet.

        bool: True if the `match_id` and `market` attributes of both Bet objects are equal,
              False otherwise.

        """
        if not isinstance(other, Bet):
            raise NotImplementedError("== method works only for Bet objects.")
        if (self.match_id == other.match_id) and (self.market == other.market):
            return True
        return False


@dataclass
class OddsInput:
    """Represents the input odds for a match.

    Attributes:
        home_team (str): Name of the home team.
        away_team (str): Name of the away team.
        odds (list[float]): Decimal odds in the format [H, D, A], where:
            - H: Odds for the home team to win.
            - D: Odds for a draw.
            - A: Odds for the away team to win.

    """

    home_team: str
    away_team: str
    odds: list[float]

    @property
    def odd_dict(self) -> dict[str, float]:
        """Returns a dictionary mapping the outcomes of a match to their respective odds.

        The dictionary contains the following keys:
        - "H": Home team win odds
        - "D": Draw odds
        - "A": Away team win odds

        Returns:
            dict[str, float]: A dictionary where the keys are the outcomes ("H", "D", "A")
            and the values are the corresponding odds as floats.

        """
        return {"H": self.odds[0], "D": self.odds[1], "A": self.odds[2]}

    @property
    def match_id(self) -> str:
        return f"{self.home_team} - {self.away_team}"
