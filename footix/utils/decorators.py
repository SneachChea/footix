import pandas as pd
from typing import Callable, Sequence

def verify_required_column(column_names: Sequence[str])-> Callable:
    """ Decorator that check if the first input argument is a pandas
    Dataframme and check if the columns in column_names are presents"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if len(args) > 0 and isinstance(args[0], pd.DataFrame):
                df = args[0]
                missing_columns = [col for col in column_names if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"The following columns are missing: {missing_columns}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

