"""
Module for different objective functions used to be evaluated in the
OptimizationProblems for ParametricStudy and optimization of Process
objects or Fractionation object.
"""
from CADETProcess.common import RankedPerformance

def parameters_product(performance):
    """Product of productivity, recovery, and eluent consumption.


    Parameters
    ----------
    performance : Performance
        Performance object from fractionation

    Returns
    -------
    f : float or np.ndarray
         Value of the objective function.
    """
    return - performance.productivity * performance.recovery * performance.eluent_consumption


def mass(performance):
    """Mass

    Parameters
    ----------
    performance : Performance
        Performance object from fractionation

    Returns
    -------
    f : float or np.ndarray
         Value of the objective function.
    """
    return - performance.mass


def ranked_objective_decorator(ranking):
    """Function for generating obj_fun_decorators with component ranking.

    Parameters
    ----------
    ranking : float or list of floats
        ranking for components

    Returns
    -------
    obj_fun_decorator : function
        Decorator function to change Performance object to RankedPerformance.
    """
    def obj_fun_decorator(obj_fun):
        """Decorator function to change Performance object to RankedPerformance.

        Parameters
        ----------
        obj_fun : function
            Objective function

        Returns
        -------
        obj_fun_wrapper : function
            Function for calling obj_fun with RankedPerformance
        """
        def obj_fun_wrapper(performance):
            """Function for calling obj_fun with RankedPerformance

            Parameters
            ----------
            performance : Performance
                Performance object from fractionation

            Returns
            -------
            f : value of objective function
            """
            performance = RankedPerformance(performance, ranking)
            return obj_fun(performance)

        return obj_fun_wrapper

    return obj_fun_decorator

def get_ranked_performance(obj_fun, ranking):
    def obj_fun_wrapper(performance):
        performance = RankedPerformance(performance, ranking)
        return obj_fun(performance)

    return obj_fun_wrapper
