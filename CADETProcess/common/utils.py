import numpy as np

def approximate_jac(xk, f, epsilon, args=()):
    """Finite-difference approximation of the jacobian of a vector function

    Parameters
    ----------
    xk : array_like
        The coordinate vector at which to determine the jacobian of `f`.
    f : callable
        The function of which to determine the jacobian (partial derivatives).
        Should take `xk` as first argument, other arguments to `f` can be
        supplied in ``*args``.  Should return a vector, the values of the
        function at `xk`.
    epsilon : array_like
        Increment to `xk` to use for determining the function jacobian.
        If a scalar, uses the same finite difference delta for all partial
        derivatives.  If an array, should contain one value per element of
        `xk`.
    \\*args : args, optional
        Any other arguments that are to be passed to `f`.

    Returns
    -------
    grad : ndarray
        The partial derivatives of `f` to `xk`.

    Notes
    -----
    The function gradient is determined by the forward finite difference
    formula::

                 f(xk[i] + epsilon[i]) - f(xk[i])
        f'[:,i] = ---------------------------------
                            epsilon[i]
    """
    f0 = f(*((xk,) + args))
    jac = np.zeros((len(f0),len(xk)), float)

    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        jac[:,k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return jac

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
        if isinstance(value, dict) or value is None:
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
    """Generator for returning all leaves of a nested dictionary in dot notation
    """
    for key, value in nested_dict.items():
        if not isinstance(value, dict):
            yield key
        else:
            for subpath in get_leaves(value):
                yield ".".join((key, subpath))

from functools import reduce
def get_nested_value(nested_dict, path):
    if isinstance(path, str):
        path = path.split('.')
    """Access a value in a nested dict using path in dot notation."""
    return reduce(dict.get, path, nested_dict)

def set_nested_value(nested_dict, path, value):
    """Set a value in a nested dict using dot notation."""
    if isinstance(path, str):
        path = path.split('.')
    get_nested_value(nested_dict, path[:-1])[path[-1]] = value
    