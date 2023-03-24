from itertools import product
from typing import Iterable


def named_product(dict_of_iterables: dict[Iterable]) -> dict:
    """
    Iterates over the cartesian product of
    dictionaries.

    Parameters
    ----------
    dict_of_iterables
        Dictionaries to compose the product
        from.

    Yields
    ------
    Dictionaries with exact same keys taken
    from the input and values taken from
    the cartesian product of input values.
    """
    keys = dict_of_iterables.keys()
    for i in product(*dict_of_iterables.values()):
        yield dict(zip(keys, i))


def ext_format(what, fmt: str) -> str:
    """
    Turns objects into strings.

    Parameters
    ----------
    what
        An object to format.
    fmt
        The format to use.

    Returns
    -------
    The resulting string representation
    of an object.
    """
    if fmt == "suppress":
        return ""
    elif fmt is None:
        return str(what)
    else:
        return f"{what:{fmt}}"
