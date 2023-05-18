""" some unitest using pytest for the decorators functions presents in the folder utils/decorators.py
"""
import pytest

from footix.utils.decorators import verify_required_column
import pandas as pd

@verify_required_column(['Home', 'Away', 'result'])
def process_dataframe(df: pd.DataFrame) -> None:
    pass

def test_process_dataframe_success():
    df = pd.DataFrame({'Home': ["Chelsea", "Liverpool", "Saint Etienne"], 'Away': ["Man U", "Betis", "PSG"], 'result': ["D", "A", "H"]})
    process_dataframe(df)  # Aucune exception ne devrait être levée

def test_process_dataframe_failure():
    df = pd.DataFrame({'Home': ["Chelsea", "Liverpool", "Saint Etienne"], 'Away': ["Man U", "Betis", "PSG"]})
    with pytest.raises(ValueError):
        process_dataframe(df)  # Une exception ValueError devrait être levée
