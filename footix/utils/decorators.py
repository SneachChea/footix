from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import pandas as pd

P = ParamSpec("P")
R = TypeVar("R")

ITERABLE_IN_STR = TypeVar("ITERABLE_IN_STR", list[str], tuple[str], set[str])


def verify_required_column(column_names: ITERABLE_IN_STR) -> Callable:
    """Decorator that check if the first input argument is a pandas Dataframme and check if the
    columns in column_names are presents."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if len(args) > 0 and isinstance(args[0], pd.DataFrame):
                df = args[0]
                missing_columns = [col for col in column_names if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"The following columns are missing: {missing_columns}")
            return func(*args, **kwargs)

        return wrapper

    return decorator
