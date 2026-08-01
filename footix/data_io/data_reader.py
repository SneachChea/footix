import dataclasses
from typing import Iterator

import pandas as pd

from footix.utils.decorators import verify_required_column


@dataclasses.dataclass
class MatchupResult:
    """A dataclass representing the result of a football match.

    Attributes:
        home_team (str): The name of the home team.
        away_team (str): The name of the away team.
        result (str): The final result of the match:
            ('H' for Home Win, 'A' for Away Win, 'D' for Draw).
        away_goals (float): The number of goals scored by the away team.
        home_goals (float): The number of goals scored by the home team.

    """

    home_team: str
    away_team: str
    result: str
    away_goals: float
    home_goals: float

    @staticmethod
    def from_dict(dict_row: dict) -> "MatchupResult":
        """Factory method to create a MatchupResult object from a dictionary row.

        Args:
            dict_row: Match result values using the canonical keys ``home_team``,
                ``away_team``, ``ftr``, ``ftag``, and ``fthg``.

        Returns:
            A MatchupResult populated from the dictionary row.

        """
        return MatchupResult(
            home_team=dict_row["home_team"],
            away_team=dict_row["away_team"],
            result=dict_row["ftr"],
            away_goals=dict_row["ftag"],
            home_goals=dict_row["fthg"],
        )


class EloDataReader:
    def __init__(self, df_data: pd.DataFrame):
        self.df_data = self._process_df(df_data)
        # Better performances for iteration over rows
        self.data = self.df_data.to_dict(orient="index")

    @verify_required_column(["date", "home_team", "away_team", "fthg", "ftag", "ftr"])
    def _process_df(self, df_data: pd.DataFrame) -> pd.DataFrame:
        df = df_data.copy().reset_index(drop=True)
        df = df[["date", "home_team", "away_team", "fthg", "ftag", "ftr"]]
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        df = df.sort_values(by="date", ascending=True)
        return df

    def __len__(self) -> int:
        return len(self.df_data)

    def unique_teams(self) -> list[str]:
        list_unique_team = list(
            set(self.df_data["home_team"].unique()).intersection(
                self.df_data["away_team"].unique()
            )
        )
        return sorted(list_unique_team)

    def __iter__(self) -> Iterator[MatchupResult]:
        return iter(self.__getitem__(idx) for idx in range(len(self)))

    def __getitem__(self, idx: int) -> MatchupResult:
        return MatchupResult.from_dict(self.data[idx])
