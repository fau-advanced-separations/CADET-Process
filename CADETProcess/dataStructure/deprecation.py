import functools
import warnings
from typing import Any, Callable, Dict

__all__ = ["deprecated_alias", "rename_kwargs"]


def deprecated_alias(**aliases: str) -> Callable:
    """
    Add alias for deprecated function arguments.

    Parameters
    ----------
    **aliases : str
        Keyword arguments where keys are old parameter names and values are new parameter names

    Returns
    -------
    Callable
        A decorator function that wraps the original function

    Examples
    --------
    @deprecated_alias(old_name='new_name')
    def example_function(new_name):
         return new_name
    """

    # Decorator function: takes the f and returns a f with the new argument names
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrap_decorated_argument(*args: Any, **kwargs: Any) -> Any:
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrap_decorated_argument

    return decorator


def rename_kwargs(
    func_name: str, kwargs: Dict[str, Any], aliases: Dict[str, str]
) -> None:
    """
    Automatically rename deprecated function arguments.

    Parameters
    ----------
    func_name : str
        Name of the function being decorated
    kwargs : Dict[str, Any]
        Dictionary of keyword arguments passed to the function
    aliases : Dict[str, str]
        Dictionary mapping old parameter names to new ones

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If both old and new parameter names are used simultaneously

    Examples
    --------
    rename_kwargs('example_function', {'old_name': 'value'}, {'old_name': 'new_name'})
    """
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError(
                    f"{func_name} received both {alias} and {new} as arguments!"
                    f" {alias} is deprecated, use {new} instead."
                )
            warnings.warn(
                message=(
                    f"`{alias}` is deprecated as an argument to `{func_name}`; use `{new}` instead."
                ),
                category=DeprecationWarning,
                stacklevel=2,
            )
            kwargs[new] = kwargs.pop(alias)
