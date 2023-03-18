import numpy as np
from typing import Union, List, Tuple

def entropy(proba: Union[List, np.ndarray], outcome_idx : int) -> float:
    """
    Compute the entropy (or incertity) metric.

    Args:
        proba (Union[List, np.ndarray]): list of probabilities
        outcome_idx (int): index of the outcome, can be 0, 1, 2 for Home, Draw and Away 

    Returns:
        float: entropy metrics
    """
    pr = proba[outcome_idx]
    return -np.log(pr)/np.log(3)

def RPS(probas: Union[List, np.ndarray], outcome_idx: int)-> float:
    """
        Compute the Ranked Probability Score.

    Args:
        probas (Union[List, np.ndarray]): list of probabilities
        outcome_idx (int): index of the outcome. 0, 1, 2 for Home, Draw and Away

    Returns:
        float: RPS metrics
    """
    outcome = np.zeros_like(probas)
    outcome[outcome_idx] = 1.
    cum_probas = np.cumsum(probas)
    cum_outcome = np.cumsum(outcome)
    sum_rps = 0
    for i in range(len(outcome)):         
        sum_rps+= (cum_probas[i] - cum_outcome[i])**2
    
    return sum_rps/(len(outcome)-1)


def Zscore(probas: Union[List, np.ndarray], RPS_real: float) -> Tuple[float, float, float]:
    """
        Compute the Z-score in respect of the RPS computed

    Args:
        probas (Union[List, np.ndarray]): list of probabilities
        RPS_real (float): RPS result

    Returns:
        float: Z-score
    """

    eps = 1e-5
    N = 100
    
    def _monteCarl(probas : Union[List, np.ndarray], N : int):
        outcomes = [0, 1, 2]
        RPS_stats = np.zeros(N)
        
        if np.sum(probas)!=1.:
            probas = probas/np.sum(probas)
        for i in range(N):
            res = np.random.choice(outcomes, p=probas)
            RPS_stats[i] = RPS(probas, res)
        return np.mean(RPS_stats), np.std(RPS_stats)
    
    mu, sigma = _monteCarl(probas, N)
    
    return (RPS_real-mu)/(sigma+eps), mu, sigma