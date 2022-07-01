"""
Auxiliary functions to check the validity of the input arguments.
"""

__all__ = ["is_positive_odd_integer", "is_valid_method"]


def is_positive_odd_integer(arg: int) -> bool:
    """Checks the argument is positive odd integer; else raises ValueError.

    Parameters
    ----------
    arg : int
        The argument to check.

    Returns
    -------
    None
    """
    if isinstance(arg, int) and arg > 0 and arg % 2 != 0:
        return True
    else:
        raise ValueError(f"{arg} must be an odd, positive integer.")


def is_valid_method(arg: str, methods: list[str]) -> None:
    """Checks the argument is a valid method; else raises ValueError.

    Parameters
    ----------
    arg : str
        The argument to check.
    methods : list[str]

    Returns
    -------
    None
    """
    if arg not in methods:
        raise ValueError(
            f"{arg} must be either `{[', '.join(method for method in methods)]}`."
        )
    return None
