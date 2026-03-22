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
from .prediction_export import (
    build_prediction_records_from_predictions,
    export_prediction_records_from_model,
)
from .understat import ScrapUnderstat

__all__ = [
    "ScrapFootballData",
    "ScrapUnderstat",
    "build_prediction_records_from_predictions",
    "export_prediction_records_from_model",
]
