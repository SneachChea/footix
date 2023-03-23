from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

class CustomModel(ABC):
    def __init__(self, n_teams: int, **kwargs: Any):
        self.n_teams = n_teams
    
    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any)-> Any:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, HomeTeam: str, AwayTeam: str) -> Any:
        raise NotImplementedError()

