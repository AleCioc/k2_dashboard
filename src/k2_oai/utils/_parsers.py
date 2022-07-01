"""
Functions to parse various strings: file extensions, but also strings of coordinates
into Python lists.
"""

from __future__ import annotations

from ast import literal_eval

import numpy as np
from numpy import ndarray

__all__ = [
    "parse_str_as_coordinates",
]


def parse_str_as_coordinates(
    string: str, dtype=np.int32, sort_coordinates: bool = False
) -> ndarray:
    """Parses a string of coordinates into a list of lists of strings.
    Parameters
    ----------
    string : str
        A string of coordinates.
    dtype: numpy dtype (default = np.int32)
        The datatype of the numpy array to be returned.
    sort_coordinates: bool (default = False)
        Whether to sort the coordinates.
    Returns
    -------
    ndarray
        A list of lists of integers, denoting pixel coordinates, i.e.
        [[x1, y1], [x2, y2], ...].
    """
    if not isinstance(string, str):
        raise TypeError(f"{type(string)} is not a valid type.")

    parsed_string = (
        sorted(literal_eval(string)) if sort_coordinates else literal_eval(string)
    )

    if dtype:
        return np.array(parsed_string, dtype=dtype)
    return np.array(parsed_string)
