import pandas as pd
import pytest

from footix.utils.decorators import verify_required_column


@verify_required_column({"Home", "Away", "result"})
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
        {
            "Home": ["Chelsea", "Liverpool", "Saint Etienne"],
            "Away": ["Man U", "Betis", "PSG"],
        }
    )
    with pytest.raises(ValueError):
        process_dataframe(df)


def test_no_df_passed():
    @verify_required_column({"A"})
    def func(x):
        return x

    assert func(5) == 5


def test_positional_second_arg():
    @verify_required_column({"col"})
    def func(a, df):
        pass

    df = pd.DataFrame({"col": [1]})
    func(1, df)


def test_keyword_df():
    @verify_required_column({"col"})
    def func(a, **kwargs):
        pass

    df = pd.DataFrame({"col": [1]})
    func(1, df=df)


def test_missing_columns_message():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(ValueError) as exc:
        process_dataframe(df)
    assert "Home" in str(exc.value)
