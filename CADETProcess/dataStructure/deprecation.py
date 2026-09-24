import functools
import warnings
from typing import Any, Callable

from typing_extensions import deprecated as _te_deprecated

__all__ = ["deprecated", "deprecated_alias", "rename_kwargs"]

# (qualname, deprecated_in, removed_in), one entry per `@deprecated`-decorated callable.
# Consumed directly by tests/dataStructure/test_deprecation.py to enforce that no
# registered deprecation has an overdue removal version.
_DEPRECATIONS: list[tuple[str, str, str]] = []


def deprecated(deprecated_in: str, removed_in: str, use: str | None = None) -> Callable:
    """
    Mark a function, method, or property accessor as deprecated.

    Wraps `typing_extensions.deprecated` so call sites emit a `DeprecationWarning`
    and static type checkers see the PEP 702 `@deprecated` flag, and appends a
    `.. deprecated::` block to `__doc__` so the directive doesn't have to be
    hand-authored and kept in sync separately.

    Parameters
    ----------
    deprecated_in : str
        Version in which the callable was deprecated.
    removed_in : str
        Version in which the callable will be removed.
    use : str, optional
        Name of the replacement to mention in the warning and docs.

    Returns
    -------
    Callable
        A decorator function that wraps the original callable.

    Examples
    --------
    @deprecated(deprecated_in="0.13", removed_in="0.14", use="new_function")
    def old_function():
        return new_function()
    """
    message = f"Deprecated since v{deprecated_in}, will be removed in v{removed_in}."
    if use is not None:
        message += f" Use `{use}` instead."

    def decorator(f: Callable) -> Callable:
        _DEPRECATIONS.append((f.__qualname__, deprecated_in, removed_in))
        wrapped = _te_deprecated(message)(f)
        wrapped.__doc__ = (
            f"{wrapped.__doc__ or ''}\n\n.. deprecated:: {deprecated_in}\n    {message}\n"
        )
        return wrapped

    return decorator


def deprecated_alias(
    deprecated_in: str | None = None, removed_in: str | None = None, **aliases: str
) -> Callable:
    """
    Add alias for deprecated function arguments.

    Parameters
    ----------
    deprecated_in : str, optional
        Version in which the alias was deprecated, included in the warning message.
    removed_in : str, optional
        Version in which the alias will be removed, included in the warning message.
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
            rename_kwargs(f.__name__, kwargs, aliases, deprecated_in, removed_in)
            return f(*args, **kwargs)

        return wrap_decorated_argument

    return decorator


def rename_kwargs(
    func_name: str,
    kwargs: dict[str, Any],
    aliases: dict[str, str],
    deprecated_in: str | None = None,
    removed_in: str | None = None,
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
    deprecated_in : str, optional
        Version in which the alias was deprecated, included in the warning message.
    removed_in : str, optional
        Version in which the alias will be removed, included in the warning message.

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
            message = (
                f"`{alias}` is deprecated as an argument to `{func_name}`;"
                f" use `{new}` instead."
            )
            if deprecated_in is not None:
                message += f" Deprecated since v{deprecated_in}."
            if removed_in is not None:
                message += f" Will be removed in v{removed_in}."
            warnings.warn(message=message, category=DeprecationWarning, stacklevel=2)
            kwargs[new] = kwargs.pop(alias)
