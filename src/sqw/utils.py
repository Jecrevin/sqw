from collections.abc import Callable
from typing import Any


def flow(value: Any, *functions: Callable[[Any], Any]) -> Any:
    for func in functions:
        value = func(value)
    return value
