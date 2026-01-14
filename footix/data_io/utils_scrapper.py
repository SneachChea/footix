# Mapping of the different competitions to their respective slugs
import re
from typing import Any

import pandas as pd

from footix.utils.decorators import verify_required_column

MAPPING_COMPETITIONS: dict[str, dict[str, Any]] = {
    "FRA Ligue 1": {"footballdata": {"slug": "F1"}, "understat": {"slug": "Ligue_1"}},
    "FRA Ligue 2": {"footballdata": {"slug": "F2"}},
    "ENG Premier League": {"footballdata": {"slug": "E0"}, "understat": {"slug": "EPL"}},
    "ENG Championship": {"footballdata": {"slug": "E1"}},
    "DEU Bundesliga 1": {"footballdata": {"slug": "D1"}, "understat": {"slug": "Bundesliga"}},
    "DEU Bundesliga 2": {"footballdata": {"slug": "D2"}},
    "ITA Serie A": {"footballdata": {"slug": "I1"}, "understat": {"slug": "Serie_A"}},
    "ITA Serie B": {"footballdata": {"slug": "I2"}},
    "SPA La Liga": {"footballdata": {"slug": "SP1"}, "understat": {"slug": "La_Liga"}},
    "SPA La Liga 2": {"footballdata": {"slug": "SP2"}},
}


def check_competition_exists(competition: str) -> bool:
    """Check if the competition exists in the MAPPING_COMPETITIONS dictionary.

    Args:
        competition (str): The name of the competition to check.

    Returns:
        bool: True if the competition exists, False otherwise.

    """
    return competition in MAPPING_COMPETITIONS


def process_string(input_string):
    lower_string = input_string.lower()
    no_space_string = lower_string.replace(" ", "")
    return no_space_string


def to_snake_case(name: str) -> str:
    """Convert the string name into a snake case string.
    Shamelessly copied from:
    https://stackoverflow.com/questions/1175208/
    elegant-python-function-to-convert-camelcase-to-snake-case

    Args:
        name (str): the name to convert

    Returns:
        str: the name in snake case

    """
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("__([A-Z])", r"_\1", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


@verify_required_column(["home_team", "away_team", "date"])
def add_match_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable `match_id` column in the form "Home - Away - YYYY-MM-DD".

    This normalizes the date formatting so match ids are consistent across scrapers
    that use different date string formats.
    """
    tmp_df = df.copy()
    # Ensure date is datetime-like for a stable formatting
    if not pd.api.types.is_datetime64_any_dtype(tmp_df["date"]):
        tmp_df["date"] = pd.to_datetime(tmp_df["date"], dayfirst=True)
    tmp_df["match_id"] = (
        tmp_df["home_team"]
        + " - "
        + tmp_df["away_team"]
        + " - "
        + tmp_df["date"].dt.strftime("%Y-%m-%d")
    )
    return tmp_df


def canonicalize_matches_df(
    df: pd.DataFrame, *, require_columns: list[str] | None = None
) -> pd.DataFrame:
    """Canonicalize a match dataframe.

    Ensures date parsing, required columns present, sorts by date and adds a stable `match_id`.

    Args:
        df: Input dataframe with match rows.
        require_columns: List of columns that must be present (defaults to minimal match columns).

    Returns:
        The canonicalized dataframe.

    """
    cols_required = require_columns or ["date", "home_team", "away_team", "fthg", "ftag"]
    missing = [c for c in cols_required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for canonicalization: {missing}")

    tmp = df.copy()
    # Parse dates with dayfirst=True to be consistent with existing readers
    tmp["date"] = pd.to_datetime(tmp["date"], dayfirst=True)

    # Ensure minimal dtypes
    tmp["home_team"] = tmp["home_team"].astype(str)
    tmp["away_team"] = tmp["away_team"].astype(str)

    # Add stable match_id and sort
    tmp = add_match_id(tmp)
    tmp = tmp.sort_values(by="date", ascending=True).reset_index(drop=True)
    return tmp
