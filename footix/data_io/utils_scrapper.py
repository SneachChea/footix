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
    """
    Convert the string name into a snake case string.
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
def add_mathc_id(df: pd.DataFrame) -> pd.DataFrame:
    tmp_df = df.copy()
    tmp_df["match_id"] = tmp_df["home_team"] + " - " + tmp_df["away_team"] + " - " + tmp_df["date"]
    return tmp_df
