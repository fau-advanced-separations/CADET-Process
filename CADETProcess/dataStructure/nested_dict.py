import collections.abc
from functools import reduce
from operator import getitem

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

def check_nested(nested_dict, path):
    """Check if item sequence exists in nested dict

    Parameters
    ----------
    nested_dict: dict
        dictionary in which path is checked
    path : str
        path of item in dot notation.

    Returns
    -------
    True if item exists and isn't a dictionary, False otherwise.

    """
    if isinstance(path, str):
        path = path.split('.')

    try:
        value = get_nested_value(nested_dict, path)
        if isinstance(value, dict):
            return False
        return True
    except (KeyError, TypeError):
        return False


def generate_nested_dict(path, value=None):
    """Generate a nested dict from path in dot notation."""
    if isinstance(path, str):
        path = path.split('.')

    nested_dict = {path[-1]: value}
    for key in reversed(path[0:-1]):
        nested_dict = {key: nested_dict}
    return nested_dict


def insert_path(nested_dict, path, value):
    """Add path to existing dict without overwriting  keys."""
    if isinstance(path, str):
        path = path.split('.')

    if len(path) == 1:
        nested_dict[path[0]] = value
    else:
        insert_path(nested_dict[path[0]], path[1:], value)


def get_leaves(nested_dict):
    """Return leaves of nested dictionary in dot notation."""
    for key, value in nested_dict.items():
        if not isinstance(value, dict):
            yield key
        else:
            for subpath in get_leaves(value):
                yield ".".join((key, subpath))


def set_nested_value(nested_dict, path, value):
    """Set a value in a nested dict using dot notation."""
    if isinstance(path, str):
        path = path.split('.')
    get_nested_value(nested_dict, path[:-1])[path[-1]] = value


def get_nested_value(nested_dict, path):
    """Access a value in a nested dict using path in dot notation."""
    if isinstance(path, str):
        path = path.split('.')
    return reduce(getitem, path, nested_dict)


def update_dict_recursively(
        target_dict: dict[str, Any],
        updates: dict[str, Any],
        only_existing_keys: bool = False
        ) -> dict[str, Any]:
    """Recursively update `target_dict` with values from `updates`.

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
            target_dict[key] = update_dict_recursively(
                target_dict[key], value, only_existing_keys
            )
        else:
            # Directly update the value
            target_dict[key] = value

    return target_dict

def get_nested_attribute(obj, path):
    """Access a nested attribute using path in dot notation."""
    attributes = path.split('.')
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


def set_nested_attribute(obj, attr_string, value):
    """Set a nested attribute using path in dot notation."""
    attributes = attr_string.split('.')
    for attr in attributes[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attributes[-1], value)


def get_nested_list_value(ls, idx_tuple):
    return reduce(lambda l, i: l[i], idx_tuple, ls)


def set_nested_list_value(ls, idx_tuple, value):
    # Navigate through the nested list to the second last index
    for idx in idx_tuple[:-1]:
        ls = ls[idx]

    # Set the value using the last index
    ls[idx_tuple[-1]] = value
