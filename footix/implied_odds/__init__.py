"""Implied odds computation from predicted probabilities.

This module provides methods to convert predicted probabilities into
betting odds and market-consistent prices using various normalization techniques.

Exported functions:
    - multiplicative_method: Multiplicative odds normalization
    - power_method: Power-based odds normalization
    - shin_method: Shin's method for market-consistent odds

"""

from .implied import multiplicative_method, power_method, shin_method

__all__ = ["multiplicative_method", "power_method", "shin_method"]
