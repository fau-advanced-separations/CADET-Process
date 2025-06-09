from collections.abc import Mapping
from functools import reduce
from operator import getitem
from typing import Any, Generator, Sequence

__all__ = [
    "check_nested",
    "generate_nested_dict",
    "update_dict_recursively",
    "get_nested_value",
    "set_nested_value",
    "get_nested_attribute",
    "set_nested_attribute",
    "get_nested_list_value",
    "set_nested_list_value",
]


def check_nested(nested_dict: dict[str, Any], path: str | list) -> bool:
    """
    Check if a key path exists in a nested dictionary.

    Parameters
    ----------
    nested_dict : dict
        dictionary in which the path is checked.
    path : str or list
        Path to the key in dot notation (string) or as a list of keys.

    Returns
    -------
    bool
        True if the item exists and is not a dictionary, False otherwise.
    """
    if isinstance(path, str):
        path = path.split(".")

    try:
        value = get_nested_value(nested_dict, path)
        return not isinstance(value, dict)
    except (KeyError, TypeError):
        return False


def generate_nested_dict(path: str | list, value: Any = None) -> dict[str, Any]:
    """
    Generate a nested dictionary from a dot-separated path.

    Parameters
    ----------
    path : str or list
        Path to create in the dictionary, given in dot notation or as a list.
    value : Any, optional
        The value to assign to the innermost key.

    Returns
    -------
    dict
        A nested dictionary created from the path.
    """
    if isinstance(path, str):
        path = path.split(".")

    nested_dict = {path[-1]: value}
    for key in reversed(path[:-1]):
        nested_dict = {key: nested_dict}
    return nested_dict


def insert_path(nested_dict: dict[str, Any], path: str | list, value: Any) -> None:
    """
    Add a key path to an existing dictionary without overwriting existing keys.

    Parameters
    ----------
    nested_dict : dict
        dictionary to update.
    path : str or list
        Path to the key in dot notation (string) or as a list of keys.
    value : Any
        Value to insert.

    Raises
    ------
    KeyError
        If an intermediate key in the path does not exist.
    """
    if isinstance(path, str):
        path = path.split(".")

    if len(path) == 1:
        nested_dict.setdefault(path[0], value)
    else:
        nested_dict.setdefault(path[0], {})
        insert_path(nested_dict[path[0]], path[1:], value)


def get_leaves(nested_dict: dict[str, Any]) -> Generator[str, None, None]:
    """
    Yield leaf keys of a nested dictionary in dot notation.

    Parameters
    ----------
    nested_dict : dict
        The nested dictionary to traverse.

    Yields
    ------
    str
        The full path to each leaf node in dot notation.
    """
    for key, value in nested_dict.items():
        if isinstance(value, dict) and value:
            for subpath in get_leaves(value):
                yield f"{key}.{subpath}"
        else:
            yield key


def set_nested_value(nested_dict: dict[str, Any], path: str | list, value: Any) -> None:
    """
    Set a value in a nested dictionary using a dot-separated path.

    Parameters
    ----------
    nested_dict : dict
        dictionary to update.
    path : str or list
        Path to the key in dot notation (string) or as a list of keys.
    value : Any
        Value to set.

    Raises
    ------
    KeyError
        If an intermediate key in the path does not exist.
    """
    if isinstance(path, str):
        path = path.split(".")

    get_nested_value(nested_dict, path[:-1])[path[-1]] = value


def get_nested_value(nested_dict: dict[str, Any], path: str | list) -> Any:
    """
    Retrieve a value from a nested dictionary using a dot-separated path.

    Parameters
    ----------
    nested_dict : dict
        dictionary to retrieve the value from.
    path : str or list
        Path to the key in dot notation (string) or as a list of keys.

    Returns
    -------
    Any
        The value stored at the specified key path.

    Raises
    ------
    KeyError
        If the key path does not exist.
    """
    if isinstance(path, str):
        path = path.split(".")

    return reduce(getitem, path, nested_dict)


def update_dict_recursively(
    target_dict: dict[str, Any],
    updates: dict[str, Any],
    only_existing_keys: bool = False,
) -> dict[str, Any]:
    """
    Recursively update `target_dict` with values from `updates`.

    Parameters
    ----------
    target_dict : dict
        The original dictionary to be updated.
    updates : dict
        The dictionary containing new values to merge into `target_dict`.
    only_existing_keys : bool, optional
        If True, only update keys that already exist in `target_dict`.
        If False, add new keys from `updates`. Default is False.

    Returns
    -------
    dict
        The updated dictionary with `updates` applied.
    """
    for key, value in updates.items():
        if only_existing_keys and key not in target_dict:
            continue  # Skip keys that don't exist in target_dict

        if isinstance(value, Mapping) and isinstance(target_dict.get(key), Mapping):
            # Recursively update nested dictionaries
            target_dict[key] = update_dict_recursively(target_dict[key], value, only_existing_keys)
        else:
            # Directly update the value
            target_dict[key] = value

    return target_dict


def get_nested_attribute(obj: Any, path: str) -> Any:
    """
    Access a nested attribute using a dot-separated path.

    Parameters
    ----------
    obj : Any
        The object to retrieve the attribute from.
    path : str
        The dot-separated path to the nested attribute.

    Returns
    -------
    Any
        The value of the nested attribute.

    Raises
    ------
    AttributeError
        If any attribute in the path does not exist.
    """
    attributes = path.split(".")
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


def set_nested_attribute(obj: Any, attr_string: str, value: Any) -> None:
    """
    Set a nested attribute using a dot-separated path.

    Parameters
    ----------
    obj : Any
        The object to modify.
    attr_string : str
        The dot-separated path to the nested attribute.
    value : Any
        The value to set.
    """
    attributes = attr_string.split(".")
    for attr in attributes[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attributes[-1], value)


def get_nested_list_value(ls: Sequence[Any], idx_tuple: tuple[int, ...]) -> Any:
    """
    Retrieve a value from a nested list structure using an index tuple.

    Parameters
    ----------
    ls : Sequence[Any]
        The nested list structure.
    idx_tuple : tuple[int, ...]
        A tuple of indices specifying the path to the desired value.

    Returns
    -------
    Any
        The value at the specified nested position.

    Raises
    ------
    IndexError
        If any index in the path is out of range.
    """
    return reduce(lambda l, i: l[i], idx_tuple, ls)  # noqa: E741


def set_nested_list_value(
    ls: Sequence[Any],
    idx_tuple: tuple[int, ...],
    value: Any,
) -> None:
    """
    Set a value in a nested list structure using an index tuple.

    Parameters
    ----------
    ls : Sequence[Any]
        The nested list structure.
    idx_tuple : tuple[int, ...]
        A tuple of indices specifying the path to the value to be set.
    value : Any
        The new value to assign.

    Raises
    ------
    IndexError
        If any index in the path is out of range.
    """
    for idx in idx_tuple[:-1]:
        ls = ls[idx]  # Navigate through the nested lists
    ls[idx_tuple[-1]] = value  # Set the final value
