"""The do-nothing and pass-through callables bound where an optional step is absent."""


def nothing(*args, **kwargs) -> None:
    """Take any arguments, do nothing, return None."""
    return None


def identity(x):
    """`x` itself."""
    return x
