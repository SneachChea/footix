from abc import ABC
from typing import Union, List, Tuple, Optional
import numpy as np
import pandas as pd

class CustomModel(ABC):
    def __init__(self, n_teams: int, **kwargs):
        self.n_teams = n_teams
    
    def fit(self, X_train : Union[List, np.ndarray, Tuple, pd.DataFrame], y_train: Optional[Union[List, np.ndarray, Tuple, pd.DataFrame]]):
        pass

    def predict(self, HomeTeam: str, AwayTeam: str):
        pass

