"""Utility functions and helpers for Footix.

This module provides common utilities including type definitions,
decorators, and helper functions used across the Footix package.

Submodules:
    - typing: Type definitions and aliases
    - team_name_resolver: Robust calendar-to-training team name resolver

"""

from footix.utils.team_name_resolver import (
    COMPETITION_TO_LEAGUE_KEY,
    TeamNameResolver,
    UnresolvedTeamNameError,
)

__all__ = [
    "COMPETITION_TO_LEAGUE_KEY",
    "TeamNameResolver",
    "UnresolvedTeamNameError",
]
