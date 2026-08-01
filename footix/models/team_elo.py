from dataclasses import dataclass


@dataclass
class EloTeam:
    """Team state used by the Elo rating system."""

    name: str
    rank: float = 0.0

    def __str__(self) -> str:
        return f"team {self.name}-rank {self.rank}"

    def __repr__(self) -> str:
        return str(self)
