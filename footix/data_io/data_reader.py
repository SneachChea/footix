import dataclasses
from typing import Iterator, Protocol

import pandas as pd


class DataProtocol(Protocol):
    def __len__(self) -> int:
        ...

import dataclasses

@dataclasses.dataclass
class MatchupResult:
    """
    A dataclass representing the result of a football match.

    Attributes:
        home_team (str): The name of the home team.
        away_team (str): The name of the away team.
        result (str): The final result of the match ('H' for Home Win, 'A' for Away Win, 'D' for Draw).
        away_goals (float): The number of goals scored by the away team.
        home_goals (float): The number of goals scored by the home team.

    Methods:
        from_dict(dict_row: dict) -> "MatchupResult":
            Factory method to create a MatchupResult object from a dictionary row.
    """

    home_team: str
    away_team: str
    result: str
    away_goals: float
    home_goals: float

    @staticmethod
    def from_dict(dict_row: dict) -> "MatchupResult":
        """
        Factory method to create a MatchupResult object from a dictionary row.

        Parameters:
            dict_row (dict): A dictionary containing the match results with keys:
                - 'HomeTeam': The name of the home team.
                - 'AwayTeam': The name of the away team.
                - 'FTR': The final result ('H' for Home Win, 'A' for Away Win, 'D' for Draw).
                - 'FTAG': The number of goals scored by the away team.
                - 'FTHG': The number of goals scored by the home team.

        Returns:
            MatchupResult: An instance of the MatchupResult class populated with data from the dictionary row.
        """
        return MatchupResult(
            home_team=dict_row["HomeTeam"],
            away_team=dict_row["AwayTeam"],
            result=dict_row["FTR"],
            away_goals=dict_row["FTAG"],
            home_goals=dict_row["FTHG"]
        )

class EloDataReader(DataProtocol):
    def __init__(self, df_data: pd.DataFrame):
        self.data_df = df_data.copy().reset_index(drop=True)
        self._process_df()
        # Better performances for iteration over rows
        self.data = self.data_df.to_dict(orient="index")

    # TODO: Add a sanity check to verify if columns are presents in the df
    def _process_df(self) -> None:
        self.data_df = self.data_df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]
        self.data_df["Date"] = pd.to_datetime(self.data_df["Date"], dayfirst=True)
        self.data_df = self.data_df.sort_values(by="Date", ascending=True)

    def __len__(self) -> int:
        return len(self.data_df)

    def unique_teams(self) -> list[str]:
        list_unique_team = list(
            set(self.data_df["HomeTeam"].unique()).intersection(self.data_df["AwayTeam"].unique())
        )
        return sorted(list_unique_team)

    def __iter__(self) -> Iterator[MatchupResult]:
        return iter(self.__getitem__(idx) for idx in range(len(self)))

    def __getitem__(self, idx: int) -> MatchupResult:
        return MatchupResult.from_dict(self.data[idx])
