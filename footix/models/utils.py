from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

import footix.utils.decorators as decorators


@decorators.verify_required_column(column_names=["HomeTeam", "FTHG"])
def compute_goals_home_vectors(
    data: pd.DataFrame, /, map_teams: dict, nbr_team: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute vectors representing home team goals.

    Args:
        data (pd.DataFrame): Input DataFrame with home team goals and HomeTeam column.
        map_teams (dict): Dictionary mapping team names to numerical IDs.
        nbr_team (int): Number of teams in the league.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            x representing home team goals and tau_home representing binary vectors
            for each home team.

    """
    x = np.zeros(len(data))
    tau_home = np.zeros((len(data), nbr_team))
    for i, row in data.iterrows():
        j = map_teams[row["HomeTeam"]]
        x[i] = row["FTHG"]
        tau_home[i, j] = 1
    return x, tau_home


@decorators.verify_required_column(column_names=["AwayTeam", "FTAG"])
def compute_goals_away_vectors(
    data: pd.DataFrame, /, map_teams: dict[str, int], nbr_team: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(len(data))
    tau_away = np.zeros((len(data), nbr_team))
    for i, row in data.iterrows():
        j = map_teams[row["AwayTeam"]]
        x[i] = row["FTAG"]
        tau_away[i, j] = 1
    return x, tau_away


def to_torch_tensor(
    *arrays: np.ndarray, dtype: torch.dtype = torch.float32
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Convert numpy arrays to torch tensors.

    Args:
        *arrays: Variable number of numpy arrays to convert
        dtype: Target tensor dtype (default: torch.float32)

    Returns:
        Single tensor if one array is provided, tuple of tensors if multiple arrays

    Examples:
        >>> x = np.array([1, 2, 3])
        >>> tensor_x = to_tensor(x)

        >>> x = np.array([1, 2, 3])
        >>> y = np.array([4, 5, 6])
        >>> tensor_x, tensor_y = to_tensor(x, y)

    """
    tensors = tuple(torch.from_numpy(arr).type(dtype) for arr in arrays)
    return tensors[0] if len(tensors) == 1 else tensors


def poisson_proba(lambda_param: float, k: int) -> np.ndarray:
    """Calculate the probability of achieving upto k goals given a lambda parameter.

    Parameters:     lambda_param (float): The expected number of goals.     k (int): The number of
    goals to achieve.

    Returns:     np.ndarray: An array containing the probabilities of achieving each possible
    number               of goals from 0 to n_goals, inclusive.

    """
    poisson = stats.poisson(mu=lambda_param)
    k_list = np.arange(k)
    return poisson.pmf(k=k_list)  # type:ignore
