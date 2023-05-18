import pandas as pd
import pytest

from footix.utils.decorators import verify_required_column


@verify_required_column(["Home", "Away", "result"])
def process_dataframe(df: pd.DataFrame) -> None:
    pass


def test_process_dataframe_success():
    df = pd.DataFrame(
        {
            "Home": ["Chelsea", "Liverpool", "Saint Etienne"],
            "Away": ["Man U", "Betis", "PSG"],
            "result": ["D", "A", "H"],
        }
    )
    process_dataframe(df)


def test_process_dataframe_failure():
    df = pd.DataFrame(
        {"Home": ["Chelsea", "Liverpool", "Saint Etienne"], "Away": ["Man U", "Betis", "PSG"]}
    )
    with pytest.raises(ValueError):
        process_dataframe(df)
