"""Data input/output utilities for football data sources.

This module provides interfaces and implementations for scraping and reading
football data from multiple sources (Football-Data.org, Understat, etc.).

Submodules:
    - footballdata: Football-Data.org scraper
    - understat: Understat.com data reader
    - data_reader: Generic data reading utilities
    - base_scrapper: Base classes for data scrapers
    - utils_scrapper: Scraper utility functions

"""

from .footballdata import ScrapFootballData
from .rolling_stats import add_rolling_team_features
from .understat import ScrapUnderstat

__all__ = ["ScrapFootballData", "ScrapUnderstat", "add_rolling_team_features"]
