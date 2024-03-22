import numpy as np
import pandas as pd

import footix.utils.decorators as decorators


@decorators.verify_required_column(column_names=["HomeTeam", "FTHG"])
def compute_goals_home_vectors(
    data: pd.DataFrame, /, map_teams: dict, nbr_team: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(len(data))
    tau_home = np.zeros((len(data), nbr_team))
    for i, row in data.iterrows():
        j = map_teams[row["HomeTeam"]]
        x[i] = row["FTHG"]
        tau_home[i, j] = 1
    return x, tau_home


@decorators.verify_required_column(column_names=["AwayTeam", "FTAG"])
def compute_goals_away_vectors(
    data: pd.DataFrame, /, map_teams: dict, nbr_team: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(len(data))
    tau_away = np.zeros((len(data), nbr_team))
    for i, row in data.iterrows():
        j = map_teams[row["AwayTeam"]]
        x[i] = row["FTAG"]
        tau_away[i, j] = 1
    return x, tau_away
