"""The do-nothing callable bound where an optional step is absent."""


def nothing(*args, **kwargs) -> None:
    """Take any arguments, do nothing, return None."""
    return None
