import collections.abc
from functools import reduce
from operator import getitem


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
    True, if item exists, False otherwise.

    """
    if isinstance(path, str):
        path = path.split('.')

    try:
        value = get_nested_value(nested_dict, path)
        if isinstance(value, dict):
            return False
        return True
    except:
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
    """Generator for returing leaves of nested dictionary in dot notation."""
    for key, value in nested_dict.items():
        if not isinstance(value, dict):
            yield key
        else:
            for subpath in get_leaves(value):
                yield ".".join((key, subpath))


def get_nested_value(nested_dict, path):
    if isinstance(path, str):
        path = path.split('.')
    """Access a value in a nested dict using path in dot notation."""
    return reduce(getitem, path, nested_dict)


def set_nested_value(nested_dict, path, value):
    """Set a value in a nested dict using dot notation."""
    if isinstance(path, str):
        path = path.split('.')
    get_nested_value(nested_dict, path[:-1])[path[-1]] = value


def update(d, u):
    """Recursively update dictionary d with u"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
