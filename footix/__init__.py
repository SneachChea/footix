"""Footix: Football analytics and prediction framework.

A comprehensive library for football match prediction, odds analysis,
and betting strategy optimization using statistical models and machine learning.

Main exports:
    - Bet: Represents a single betting opportunity
    - OddsInput: Input odds format for analysis
    - EdgeFloorConfig: Configuration for edge detection
    - OddsRange: Range of odds for filtering
    - ProbaResult: Probability prediction results
    - SampleProbaResult: Sampled probability results
"""

from .strategy.bets import Bet, OddsInput
from .strategy.select_bets import EdgeFloorConfig, OddsRange
from .utils.typing import ProbaResult, SampleProbaResult

__all__ = ["EdgeFloorConfig", "OddsRange", "Bet", "OddsInput", "ProbaResult", "SampleProbaResult"]
