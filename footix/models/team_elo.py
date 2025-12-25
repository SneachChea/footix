class EloTeam:
    """A class representing a team in the Elo rating system.

    This class stores information about a team including its name and Elo rank. The Elo rank is
    updated based on match outcomes.

    """

    def __init__(self, name: str) -> None:
        """Initialize an EloTeam with a given name.

        Args:
            name: The name of the team

        """
        self.name_ = name
        self.rank_ = 0.0

    @property
    def name(self) -> str:
        """Get the name of the team.

        Returns:
            The name of the team

        """
        return self.name_

    @property
    def rank(self) -> float:
        """Get the current Elo rank of the team.

        Returns:
            The Elo rank of the team as a float

        """
        return self.rank_

    @rank.setter
    def rank(self, new_rank: float | int) -> None:
        """Set the Elo rank of the team.

        Args:
            new_rank: The new Elo rank to set (must be a float)

        Note:
            If a non-float value is provided, a TypeError will be raised.

        """
        if isinstance(new_rank, float) or isinstance(new_rank, int):
            self.rank_ = new_rank
        else:
            raise TypeError(
                f"Rank must be a float, got {type(new_rank)} instead."
            )

    def __str__(self) -> str:
        """Return string representation of the team.

        Returns:
            A formatted string with team name and rank

        """
        return f"team {self.name}-rank {self.rank}"

    def __repr__(self) -> str:
        """Return official string representation of the team.

        Returns:
            String representation of the team

        """
        return str(self)
