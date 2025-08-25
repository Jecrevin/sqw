from collections.abc import Callable
from typing import Any


def flow(value: Any, *functions: Callable[[Any], Any]) -> Any:
    """Sequentially applies a series of functions to an initial value.

    This function implements a pipeline pattern, where an initial value is
    passed through a sequence of functions. The output of each function
    serves as the input for the next function in the sequence.

    Parameters
    ----------
    value : Any
        The initial value to be processed by the function pipeline.

    *functions : Callable[[Any], Any]
        A variable number of functions to apply to the value. Each function
        must accept a single argument and return a value.

    Returns
    -------
    Any
        The final result after applying all functions in the sequence.

    Examples
    --------
    >>> def add_one(x):
    ...     return x + 1
    >>> def square(x):
    ...     return x * x
    >>> flow(5, add_one, square)
    36
    >>> flow(5, square, add_one)
    26
    """
    for func in functions:
        value = func(value)
    return value
