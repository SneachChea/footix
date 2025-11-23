from __future__ import annotations

from functools import wraps
from typing import (
    Callable,
    Iterable,
    ParamSpec,
    TypeVar,
)

import pandas as pd

P = ParamSpec("P")
R = TypeVar("R")


def verify_required_column(
    column_names: Iterable[str],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that validates the presence of required columns in a pandas DataFrame.

    The decorator inspects **both** positional and keyword arguments.  If a
    ``pd.DataFrame`` is supplied under the name ``df`` (positional or keyword)
    it checks that all names in *column_names* are present.  A :class:`ValueError`
    is raised with a clear message if any columns are missing.

    Parameters
    ----------
    column_names : Iterable[str]
        An iterable of column names that must exist in the DataFrame.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        The wrapped function.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Resolve the DataFrame argument
            df = None

            # 1. Positional first argument
            if args and isinstance(args[0], pd.DataFrame):
                df = args[0]

            # 2. Keyword ``df`` or any other name that points to a DataFrame
            if df is None:
                for value in kwargs.values():
                    if isinstance(value, pd.DataFrame):
                        df = value
                        break

            # If we found no DataFrame, just call the original function.
            # This mirrors the previous behaviour but makes it explicit.
            if df is None:
                return func(*args, **kwargs)

            missing_columns = [col for col in column_names if col not in df.columns]
            if missing_columns:
                # Join names for a cleaner error message
                missing_str = ", ".join(missing_columns)
                raise ValueError(
                    f"The following required columns are missing: {missing_str}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
