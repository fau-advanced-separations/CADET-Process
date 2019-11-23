"""Module for definition of nonlinear constraint functions used in optimization

See also
--------
Evaluator
OptimizationProblem
objectiveFunctions
"""

from CADETProcess import CADETProcessError

def purity(performance):
    """Product purity.


    Parameters
    ----------
    performance : Performance
        Performance object from fractionation

    Returns
    -------
    c : float or np.ndarray
         Value of the nonlinear constraint function.
    """
    return performance.purity


def concentration(performance):
    """Product concentration

    Parameters
    ----------
    performance : Performance
        Performance object from fractionation

    Returns
    -------
    c : float or np.ndarray
         Value of nonlinear constraint function.
    """
    return performance.concentration


def nonlin_bounds_decorator(bounds):
    """Function for generating nonlin_bounds_decorator with performance bounds.

    Parameters
    ----------
    bounds : float or list of floats
        required values for performance

    Returns
    -------
    nonlin_bounds_decorator : function
        Decorator function to subtract Performance from bounds
    """
    def nonlin_fun_decorator(nonlin_fun):
        """Decorator function to change Performance object to RankedPerformance.

        Parameters
        ----------
        nonlin_fun : function
            nonlinear constraint function

        Returns
        -------
        obj_fun_wrapper : function
            Function for calling obj_fun with RankedPerformance
        """
        def nonlin_fun_wrapper(performance):
            """Function for calling nonlin_fun with RankedPerformance

            Parameters
            ----------
            performance : Performance
                Performance object from fractionation

            Returns
            -------
            c : value of constraint function
            """
            if isinstance(bounds, (float, int)):
                _bounds = [bounds]*performance.n_comp
            elif len(bounds) != performance.n_comp:
                raise CADETProcessError('Number of components not matching')
            else:
                _bounds = bounds
            return _bounds - nonlin_fun(performance)

        return nonlin_fun_wrapper

    return nonlin_fun_decorator
