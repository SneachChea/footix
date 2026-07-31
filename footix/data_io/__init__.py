"""Data input/output utilities for football data sources.

This module provides interfaces and implementations for scraping and reading
football data from multiple sources (football-data.co.uk, Understat,
football-data.org, APIFootball.com, etc.).

Submodules:
    - footballdata: football-data.co.uk scraper
    - understat: Understat.com data reader
    - football_data_org: football-data.org REST API provider
    - apifootball_com: APIFootball.com REST API provider
    - data_reader: Generic data reading utilities
    - base_scrapper: Base classes for data scrapers
    - utils_scrapper: Scraper utility functions

"""

from .apifootball_com import APIFootballComError, ScrapAPIFootballCom
from .football_data_org import FootballDataOrgError, ScrapFootballDataOrg
from .footballdata import ScrapFootballData
from .prediction_export import (
    build_prediction_records_from_predictions,
    export_prediction_records_from_model,
)
from .understat import ScrapUnderstat

__all__ = [
    "APIFootballComError",
    "FootballDataOrgError",
    "ScrapAPIFootballCom",
    "ScrapFootballData",
    "ScrapFootballDataOrg",
    "ScrapUnderstat",
    "build_prediction_records_from_predictions",
    "export_prediction_records_from_model",
]
