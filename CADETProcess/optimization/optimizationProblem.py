import copy
from functools import wraps
import inspect
import math
import random
import warnings

from addict import Dict
import numpy as np
import hopsy
import multiprocess
import pathos

from CADETProcess import CADETProcessError
from CADETProcess import log

from CADETProcess.dataStructure import update
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    String, Switch, RangedInteger, Callable, Tuple,
    DependentlySizedList
)
from CADETProcess.dataStructure import frozen_attributes
from CADETProcess.dataStructure import (
    check_nested, generate_nested_dict, get_nested_value
)
from CADETProcess.transform import (
    NoTransform, AutoTransform, NormLinearTransform, NormLogTransform
)

from CADETProcess.metric import MetricBase


@frozen_attributes
class OptimizationProblem(metaclass=StructMeta):
    """Class for configuring optimization problems

    Defines lists, dictionaries and variables for creating an
    OptimizationProblem. If no name is set it tries to set the name by the
    evaluation_object name. An excepted AttributeError is ignored.

    Attributes
    ----------
    name : str
        Name of the optimization problem
    evaluation_object :  obj
        Object containing parameters to be optimized.
    evaluator : obj
        Object used to evaluate evaluation_object. Returns performance.
    variables : list
        List of optimization variables
    objectives: list of callables
        Functions that return value of objective function for performance.
    nonlinear_constraints: list of callables
        Functions that return value of nonlinear constraints for performance.
    linear_constraints : list
        List of all linear constraints of an OptimizationProblem.
    linear_equality_constraints : list
        List with all linear equality constrains of an OptimizationProblem.
    callbacks : list
        List of callback functions to record progress.

    """
    name = String()

    def __init__(self, name, log_level='INFO', save_log=True):
        self.name = name
        self.logger = log.get_logger(
            self.name, level=log_level, save_log=save_log
        )

        self._evaluation_objects = []
        self._evaluators = []
        self.cached_evaluators = []
        self._variables = []
        self._dependent_variables = []
        self._objectives = []
        self._nonlinear_constraints = []
        self._linear_constraints = []
        self._linear_equality_constraints = []
        self._callbacks = []

        self._x0 = None

    @property
    def evaluation_objects(self):
        """list: Object to be evaluated during optimization.

        See Also
        --------
        OptimizatonVariable
        Evaluator
        evaluate
        Performance
        objectives
        nonlinear_constraints

        """
        return self._evaluation_objects

    @property
    def evaluation_objects_dict(self):
        """dict: Evaluation objects names and objects"""
        return {obj.name: obj for obj in self.evaluation_objects}

    def add_evaluation_object(self, evaluation_object):
        """Add evaluation object to the optimization problem.

        Parameters
        ----------
        evaluation_object : obj
            evaluation object to be added to the optimization problem.

        Raises
        ------
        CADETProcessError
            If evaluation object already exists in optimization problem.

        """
        if evaluation_object in self._evaluation_objects:
            raise CADETProcessError(
                'Evaluation object already part of optimization problem.'
            )

        self._evaluation_objects.append(evaluation_object)

    @property
    def variables(self):
        """list: List of all optimization variables."""
        return self._variables

    @property
    def variable_names(self):
        """list: Optimization variable names."""
        return [var.name for var in self.variables]

    @property
    def n_variables(self):
        """int: Number of optimization variables."""
        return len(self.variables)

    @property
    def independent_variables(self):
        """list: Independent OptimizationVaribles."""
        return list(filter(lambda var: var.isIndependent, self.variables))

    @property
    def n_independent_variables(self):
        """int: Number of independent optimization variables."""
        return len(self.independent_variables)

    @property
    def dependent_variables(self):
        """list: OptimizationVaribles with dependencies."""
        return list(
            filter(lambda var: var.isIndependent is False, self.variables)
        )

    @property
    def variables_dict(self):
        """dict: All optimization variables indexed by variable name."""
        vars = {var.name: var for var in self.variables}
        dep_vars = {var.name: var for var in self.dependent_variables}
        return {**vars, **dep_vars}

    @property
    def variable_values(self):
        """list: Values of optimization variables."""
        return [var.value for var in self.variables]

    def add_variable(
            self, parameter_path=None, evaluation_objects=-1,
            lb=-math.inf, ub=math.inf, transform=None,
            component_index=None, polynomial_index=None, name=None):
        """Add optimization variable to the OptimizationProblem.

        The function encapsulates the creation of OoptimizationVariable objects
        in order to prevent invalid OptimizationVariables.

        Parameters
        ----------
        parameter_path : str, optional
            Path of the parameter including the evaluation object.
        evaluation_objects : EvaluationObject or list of EvaluationObjects
            Evaluation object to set parameters.
            If None, all evaluation objects are used.
        lb : float
            Lower bound of the variable value.
        ub : float
            Upper bound of the variable value.
        transform : {'auto', 'log', None}:
            Variable transform. The default is auto.
        component_index : int
            Index for component specific variables.
        polynomial_index : int
            Index for specific polynomial coefficient.
        name : str, optional
            Name of the variable. If None, parameter_path is used.

        Raises
        ------
        CADETProcessError
            If the Variable already exists in the dictionary.

        See Also
        --------
        evaluation_object
        OptimizationVariable
        remove_variable

        """
        if name is None:
            name = parameter_path

        if name in self.variables_dict:
            raise CADETProcessError("Variable already exists")

        if evaluation_objects is None:
            evaluation_objects = []
        elif evaluation_objects == -1:
            evaluation_objects = self.evaluation_objects
        elif not isinstance(evaluation_objects, list):
            evaluation_objects = [evaluation_objects]

        evaluation_objects = [
            self.evaluation_objects[eval_obj] if isinstance(eval_obj, str)
            else eval_obj
            for eval_obj in evaluation_objects
        ]

        var = OptimizationVariable(
            name, evaluation_objects, parameter_path,
            lb=lb, ub=ub, transform=transform,
            component_index=component_index,
            polynomial_index=polynomial_index
        )

        self._variables.append(var)

        super().__setattr__(name, var)

    def remove_variable(self, var_name):
        """Remove optimization variable from the OptimizationProblem.

        Parameters
        ----------
        var_name : str
            Name of the variable to be removed from list of variables.

        Raises
        ------
        CADETProcessError
            If required variable does not exist.

        See Also
        --------
        add_variable

        """
        try:
            var = self.variables_dict[var_name]
        except KeyError:
            raise CADETProcessError("Variable does not exist")

        self._variables.remove(var)
        self.__dict__.pop(var_name)

    def add_variable_dependency(
            self, dependent_variable, independent_variables,
            factors=None, transforms=None):
        """Add dependency between two events.

        Parameters
        ----------
        dependent_variable : str
            OptimizationVariables whose value will depend on other variables.
        independent_variables : list
            List of independent variables names.
        factors : list, optional
            List of factors used for the relation with the independent variables.
            Length must be equal the length of independent variables.
            If None, all factors are assumed to be 1.
        transforms : list, optional
            List of functions used to transform the parameter value.
            Length must be equal the length of independent variables.
            If None, no transform is applied.

        Raises
        ------
        CADETProcessError
            If dependent_variable OR independent_variables are not found.
            If length of factors not equal length of independent variables.
            If length of transforms not equal length of independent variables.

        See Also
        --------
        OptimizationVariable
        add_variable

        """
        try:
            var = self.variables_dict[dependent_variable]
        except KeyError:
            raise CADETProcessError("Cannot find dependent variable")

        if not isinstance(independent_variables, list):
            independent_variables = [independent_variables]
        if not all(
                indep in self.variables_dict
                for indep in independent_variables):
            raise CADETProcessError(
                "Cannot find one or more independent variables"
            )

        if factors is None:
            factors = [1]*len(independent_variables)

        if not isinstance(factors, list):
            factors = [factors]
        if len(factors) != len(independent_variables):
            raise CADETProcessError(
                "Length of factors must equal length of independent variables"
            )
        if transforms is None:
            transforms = [None]*len(independent_variables)

        if not isinstance(transforms, list):
            transforms = [transforms]
        if len(transforms) != len(independent_variables):
            raise CADETProcessError(
                "Length of transforms must equal length of independent variables"
            )

        for indep, fac, trans in zip(independent_variables, factors, transforms):
            indep = self.variables_dict[indep]
            var.add_dependency(indep, fac, trans)

    def untransforms(func):
        @wraps(func)
        def wrapper(self, x, *args, untransform=False, **kwargs):
            """Untransform population or individual before calling function."""
            if untransform:
                x = self.untransform(x)

            return func(self, x, *args, **kwargs)

        return wrapper

    def ensures2d(func):
        @wraps(func)
        def wrapper(self, population, *args, **kwargs):
            """Make sure population is 2d list."""
            population = np.array(population, ndmin=2)
            population = population.tolist()

            return func(self, population, *args, **kwargs)

        return wrapper

    @untransforms
    def get_dependent_values(self, x):
        """Determine values of dependent optimization variables.

        Parameters
        ----------
        x : list
            (Transformed) Optimization variables values.

        Raises
        ------
        CADETProcessError
            If length of parameters does not match.

        Returns
        -------
        x : list
            Values of all optimization variables.

        """
        if len(x) != self.n_independent_variables:
            raise CADETProcessError(
                f'Expected {self.n_independent_variables} value(s)'
            )

        optimization_problem = copy.deepcopy(self)
        variables = optimization_problem.independent_variables

        for variable, value in zip(variables, x):
            value = np.format_float_positional(
                value, precision=variable.precision, fractional=False
            )
            variable.value = float(value)

        return optimization_problem.variable_values

    @untransforms
    def set_variables(
            self, x,
            evaluation_objects=-1,
            make_copy=False):
        """Set the values from the x-vector to the EvaluationObjects.

        Parameters
        ----------
         x : array_like
            Value of the optimization variables
        evaluation_objects : list or EvaluationObject or None or -1
            Evaluations objects to set variables in.
            If None, do not set variables.
            If -1, variables are set to all evaluation objects.
            The default is -1.
        make_copy : bool
            If True, a copy of the evaluation_object attribute is made on which
            the values are set. Otherwise, the values are set on the attribute.

        Returns
        -------
        evaluation_object : object
            Returns copy of evaluation object if make_copy is True, else return
            the attribute evaluation_object with the values set.

        Raises
        ------
        CADETProcessError
            If x does not have correct length.
        ValueError
            If value of variable exceeds bounds.

        See Also
        --------
        OptimizationVariable
        evaluate

        """
        values = self.get_dependent_values(x)

        if evaluation_objects is None:
            evaluation_objects = []
        elif evaluation_objects == -1:
            evaluation_objects = self.evaluation_objects
        elif not isinstance(evaluation_objects, list):
            evaluation_objects = [evaluation_objects]

        if make_copy:
            evaluation_objects = copy.deepcopy(evaluation_objects)

        eval_obj_dict = {
            eval_obj.name: eval_obj
            for eval_obj in evaluation_objects
        }

        for variable, value in zip(self.variables, values):
            if value < variable.lb:
                raise ValueError("Exceeds lower bound")
            if value > variable.ub:
                raise ValueError("Exceeds upper bound")
            for eval_obj in variable.evaluation_objects:
                eval_obj = eval_obj_dict[eval_obj.name]
                if variable.polynomial_index is not None:
                    value_array = get_nested_value(
                        eval_obj.parameters, variable.parameter_path
                    ).copy()
                    if variable.component_index is not None:
                        value_array[
                            variable.component_index, variable.polynomial_index
                        ] = value
                    else:
                        value_array[:, variable.polynomial_index] = value

                    parameters = generate_nested_dict(
                        variable.parameter_path, value_array
                    )
                elif variable.component_index is not None:
                    value_array = get_nested_value(
                        eval_obj.parameters, variable.parameter_path
                    )

                    value_array[variable.component_index] = value

                    parameters = generate_nested_dict(
                        variable.parameter_path, value_array
                    )
                else:
                    parameters = generate_nested_dict(
                        variable.parameter_path, value
                    )

                eval_obj.parameters = parameters

        return list(eval_obj_dict.values())

    @property
    def evaluators(self):
        return self._evaluators

    @property
    def evaluators_dict(self):
        return {evaluator.name: evaluator for evaluator in self.evaluators}

    def add_evaluator(self, evaluator, cache=False, args=None, kwargs=None):
        """Add Evaluator to OptimizationProblem.

        Evaluators can be referenced by objective and constraint functions to
        perform preprocessing steps.

        Parameters
        ----------
        evaluator : callable
            Evaluation function.
        cache : bool, optional
            If True, results of the evaluator are cached. The default is False.
        args : tuple, optional
            Additional arguments for evaluation function.
        kwargs : dict, optional
            Additional keyword arguments for evaluation function.

        Raises
        ------
        TypeError
            If objective is not callable.

        """
        if not callable(evaluator):
            raise TypeError("Expected callable evaluator.")

        if str(evaluator) in self.evaluators:
            raise CADETProcessError(
                "Evaluator already exists in OptimizationProblem."
            )

        evaluator = Evaluator(
            evaluator,
            args=args,
            kwargs=kwargs,
        )
        self._evaluators.append(evaluator)

        if cache:
            self.cached_evaluators.append(evaluator)

    @property
    def objectives(self):
        return self._objectives

    @property
    def objectives_names(self):
        return [str(obj) for obj in self.objectives]

    @property
    def n_objectives(self):
        n_objectives = 0

        for objective in self.objectives:
            if len(objective.evaluation_objects) != 0:
                factor = len(objective.evaluation_objects)
            else:
                factor = 1
            n_objectives += factor*objective.n_objectives

        return n_objectives

    def add_objective(
            self,
            objective,
            n_objectives=1,
            bad_metrics=None,
            evaluation_objects=-1,
            requires=None):
        """Add objective function to optimization problem.

        Parameters
        ----------
        objective : callable or MetricBase
            Objective function.
        n_objectives : int, optional
            Number of metrics returned by objective function.
            The default is 1.
        bad_metrics : flot or list of floats, optional
            Value which is returned when evaluation fails.
        evaluation_objects : {EvaluationObject, None, -1, list}
            EvaluationObjects which are evaluated by objective.
            If None, no EvaluationObject is used.
            If -1, all EvaluationObjects are used.
        requires : {None, Evaluator, list}
            Evaluators used for preprocessing.
            If None, no preprocessing is required.

        Raises
        ------
        TypeError
            If objective is not callable.
        CADETProcessError
            If EvaluationObject is not found.
        CADETProcessError
            If Evaluator is not found.

        """
        if not callable(objective):
            raise TypeError("Expected callable objective.")

        if bad_metrics is None and isinstance(objective, MetricBase):
            bad_metrics = n_objectives * objective.bad_metrics

        if evaluation_objects is None:
            evaluation_objects = []
        elif evaluation_objects == -1:
            evaluation_objects = self.evaluation_objects
        elif not isinstance(evaluation_objects, list):
            evaluation_objects = [evaluation_objects]
        for el in evaluation_objects:
            if el not in self.evaluation_objects:
                raise CADETProcessError(
                    f"Unknown EvaluationObject: {str(el)}"
                )

        if requires is None:
            requires = []
        elif not isinstance(requires, list):
            requires = [requires]

        try:
            evaluators = [self.evaluators_dict[str(req)]for req in requires]
        except KeyError as e:
            raise CADETProcessError(f"Unknown Evaluator: {str(e)}")

        objective = Objective(
            objective,
            type='minimize',
            n_objectives=n_objectives,
            bad_metrics=bad_metrics,
            evaluation_objects=evaluation_objects,
            evaluators=evaluators
        )
        self._objectives.append(objective)

    @untransforms
    def evaluate_objectives(
            self,
            x,
            cache=None,
            cache_new=None,
            make_copy=False,
            force=False,
            return_cache_new=False,):
        """Evaluate objective functions at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.
        cache : dict, optional
            Dictionary with previously cached results.
        cache_new : dict, optional
            Dictionary for caching new results.
        make_copy : bool
            If True, a copy of the EvaluationObjects is used which is required
            for multiprocessing.
        force : bool
            If True, do not use cached results. The default if False.
        return_cache_new : bool
            If True, return intermediate results. The default is False.

        Returns
        -------
        f : list
            Values of the objective functions at point x.

        See Also
        --------
        optimization.SolverBase
        add_objective
        evaluate
        evaluate_nonlinear_constraints

        """
        self.logger.info(f'evaluate objectives at {x}')

        x = list(x)

        f = []

        if cache_new is None:
            cache_new = self.setup_cache()

        for objective in self.objectives:
            try:
                value = self._evaluate(
                    x, objective, cache, cache_new, make_copy, force,
                )
                f += value
            except CADETProcessError:
                self.logger.warn(
                    f'Evaluation of {objective.name} failed at {x}. '
                    f'Returning bad metrics.'
                )
                f += objective.bad_metrics

        return f

    @untransforms
    @ensures2d
    def evaluate_objectives_population(
            self, population, cache=None, force=False, n_cores=0):

        if cache is not None:
            manager = multiprocess.Manager()
            caches_new = manager.list()
        else:
            caches_new = []

        def eval_fun(ind):
            cache_new = self.setup_cache()
            results = self.evaluate_objectives(
                ind,
                cache=cache, cache_new=cache_new,
                make_copy=True, force=force,
            )
            caches_new.append(cache_new)

            return results

        if n_cores == 1:
            results = []
            for ind in population:
                try:
                    res = eval_fun(ind)
                    results.append(res)
                except CADETProcessError:
                    print(ind)
        else:
            if n_cores == 0:
                n_cores = None
            with pathos.multiprocessing.ProcessPool(ncpus=n_cores) as pool:
                results = pool.map(eval_fun, population)

        if cache is not None:
            for i in caches_new:
                update(cache, i)

        return results

    @untransforms
    def objective_jacobian(self, x, dx=1e-3):
        """Compute jacobian of objective functions using finite differences.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables
        dx : float
            Increment to x to use for determining the function gradient.

        Returns
        -------
        jacobian: list
            Value of the partial derivatives at point x.

        See Also
        --------
        objectives
        approximate_jac

        """
        jacobian = [
            approximate_jac(x, obj, dx)
            for obj in self.objectives
        ]
        return jacobian

    @property
    def nonlinear_constraints(self):
        return self._nonlinear_constraints

    @property
    def nonlinear_constraint_names(self):
        return [str(nonlincon) for nonlincon in self.nonlinear_constraints]

    @property
    def nonlinear_constraints_bounds(self):
        bounds = []
        for nonlincon in self.nonlinear_constraints:
            bounds += nonlincon.bounds

        return bounds

    @property
    def n_nonlinear_constraints(self):
        n_nonlinear_constraints = 0

        for nonlincon in self.nonlinear_constraints:
            if len(nonlincon.evaluation_objects) != 0:
                factor = len(nonlincon.evaluation_objects)
            else:
                factor = 1
            n_nonlinear_constraints += factor*nonlincon.n_nonlinear_constraints

        return n_nonlinear_constraints

    def add_nonlinear_constraint(
            self,
            nonlincon,
            n_nonlinear_constraints=1,
            bad_metrics=None,
            evaluation_objects=-1,
            bounds=None,
            requires=None):
        """Add nonliner constraint function to optimization problem.

        Parameters
        ----------
        nonlincon : callable
            Nonlinear constraint function.
        n_nonlinear_constraints : int, optional
            Number of metrics returned by nonlinear constraint function.
            The default is 1.
        bad_metrics : float or list of floats, optional
            Value which is returned when evaluation fails.
        evaluation_objects : {EvaluationObject, None, -1, list}
            EvaluationObjects which are evaluated by objective.
            If None, no EvaluationObject is used.
            If -1, all EvaluationObjects are used.
        bounds : scalar or list of scalars, optional
            Upper limits of constraint function.
            If only one value is given, the same value is assumed for all
            constraints. The default is 0.
        requires : {None, Evaluator, list}, optional
            Evaluators used for preprocessing.
            If None, no preprocessing is required.
            The default is None.

        Raises
        ------
        TypeError
            If constraint is not callable.
        CADETProcessError
            If EvaluationObject is not found.
        CADETProcessError
            If Evaluator is not found.

        """
        if not callable(nonlincon):
            raise TypeError("Expected callable constraint function.")

        if bad_metrics is None and isinstance(nonlincon, MetricBase):
            bad_metrics = nonlincon.bad_metrics

        if evaluation_objects is None:
            evaluation_objects = []
        elif evaluation_objects == -1:
            evaluation_objects = self.evaluation_objects
        elif not isinstance(evaluation_objects, list):
            evaluation_objects = [evaluation_objects]
        for el in evaluation_objects:
            if el not in self.evaluation_objects:
                raise CADETProcessError(
                    f"Unknown EvaluationObject: {str(el)}"
                )

        if isinstance(bounds, (float, int)):
            bounds = n_nonlinear_constraints * [bounds]
        if len(bounds) != n_nonlinear_constraints:
            raise CADETProcessError(
                f'Expected {n_nonlinear_constraints} bounds'
            )

        if requires is None:
            requires = []
        elif not isinstance(requires, list):
            requires = [requires]

        try:
            evaluators = [self.evaluators_dict[str(req)]for req in requires]
        except KeyError as e:
            raise CADETProcessError(f"Unknown Evaluator: {str(e)}")

        nonlincon = NonlinearConstraint(
            nonlincon,
            bounds=bounds,
            n_nonlinear_constraints=n_nonlinear_constraints,
            bad_metrics=bad_metrics,
            evaluation_objects=evaluation_objects,
            evaluators=evaluators
        )
        self._nonlinear_constraints.append(nonlincon)

    @untransforms
    def evaluate_nonlinear_constraints(
            self,
            x,
            cache=None,
            cache_new=None,
            make_copy=False,
            force=False):
        """Evaluate nonlinear constraint functions at point x.

        After evaluating the nonlinear constraint functions, the corresponding
        bounds are subtracted from the results.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.
        cache : dict, optional
            Dictionary with previously cached results.
        cache_new : dict, optional
            Dictionary for caching new results.
        make_copy : bool
            If True, a copy of the EvaluationObjects is used which is required
            for multiprocessing.
        force : bool
            If True, do not use cached results. The default if False.
        return_cache_new : bool
            If True, return intermediate results. The default is False.

        Returns
        -------
        c : list
            Values of the nonlinear constraint functions at point x minus the
            corresponding bounds.

        See Also
        --------
        optimization.SolverBase
        add_nonlinear_constraint
        _evaluate
        evaluate_objectives

        """
        self.logger.info(f'evaluate nonlinear constraints at {x}')

        x = list(x)

        g = []

        if cache_new is None:
            cache_new = self.setup_cache()

        for nonlincon in self.nonlinear_constraints:
            try:
                value = self._evaluate(
                    x, nonlincon, cache, cache_new, make_copy, force,
                )
                g += value
            except CADETProcessError:
                self.logger.warn(
                    f'Evaluation of {nonlincon.name} failed at {x}. '
                    f'Returning bad metrics.'
                )
                g += nonlincon.bad_metrics

        c = np.array(g) - np.array(self.nonlinear_constraints_bounds)

        return c

    @untransforms
    @ensures2d
    def evaluate_nonlinear_constraints_population(
            self, population, cache=None, force=False,
            n_cores=0):

        if cache is not None:
            manager = multiprocess.Manager()
            caches_new = manager.list()
        else:
            caches_new = []

        def eval_fun(ind):
            cache_new = self.setup_cache()
            results = self.evaluate_nonlinear_constraints(
                ind,
                cache=cache, cache_new=cache_new,
                make_copy=True, force=force,
                return_cache_new=True
            )
            caches_new.append(cache_new)

            return results

        if n_cores == 1:
            results = []
            for ind in population:
                try:
                    res = eval_fun(ind)
                    results.append(res)
                except CADETProcessError:
                    print(ind)
        else:
            if n_cores == 0:
                n_cores = None
            with pathos.multiprocessing.ProcessPool(ncpus=n_cores) as pool:
                results = pool.map(eval_fun, population)

        if cache is not None:
            for i in caches_new:
                update(cache, i)

        return results

    @untransforms
    def check_nonlinear_constraints(self, x):
        """Check if all nonlinear constraints are kept.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        flag : bool
            True if all nonlinear constraints are smaller or equal to zero,
            False otherwise.

        """
        c = self.evaluate_nonlinear_constraints(x)

        if np.any(c > 0):
            return False
        return True

    @untransforms
    def nonlinear_constraint_jacobian(self, x, dx=1e-3):
        """Compute jacobian of the nonlinear constraints at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables
        dx : float
            Increment to x to use for determining the function gradient.

        Returns
        -------
        jacobian: list
            Value of the partial derivatives at point x.

        See Also
        --------
        nonlinear_constraint_fun
        approximate_jac

        """
        jacobian = [
            approximate_jac(x, constr, dx)
            for constr in self.nonlinear_constraints
        ]
        return jacobian

    @property
    def callbacks(self):
        """list: Callback functions for recording progress."""
        return self._callbacks

    def add_callback(
            self,
            callback,
            evaluation_objects=-1,
            requires=None,
            frequency=1):

        if not callable(callback):
            raise TypeError("Expected callable callback.")

        if evaluation_objects is None:
            evaluation_objects = []
        elif evaluation_objects == -1:
            evaluation_objects = self.evaluation_objects
        elif not isinstance(evaluation_objects, list):
            evaluation_objects = [evaluation_objects]
        for el in evaluation_objects:
            if el not in self.evaluation_objects:
                raise CADETProcessError(
                    f"Unknown EvaluationObject: {str(el)}"
                )

        if requires is None:
            requires = []
        elif not isinstance(requires, list):
            requires = [requires]

        try:
            evaluators = [self.evaluators_dict[str(req)]for req in requires]
        except KeyError as e:
            raise CADETProcessError(f"Unknown Evaluator: {str(e)}")

        callback = Callback(
            callback,
            evaluation_objects=evaluation_objects,
            evaluators=evaluators,
            frequency=frequency
        )
        self._callbacks.append(callback)

    @untransforms
    def evaluate_callbacks(
            self,
            x,
            cache=None,
            cache_new=None,
            make_copy=False,
            force=False,
            return_cache_new=False,
            results_dir='./',
            current_iteration=0):
        """Evaluate callback functions at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.
        cache : dict, optional
            Dictionary with previously cached results.
        cache_new : dict, optional
            Dictionary for caching new results.
        make_copy : bool
            If True, a copy of the EvaluationObjects is used which is required
            for multiprocessing.
        force : bool
            If True, do not use cached results. The default if False.
        return_cache_new : bool
            If True, return intermediate results. The default is False.
        results_dir : path
            Path to store results (e.g. figures, tables etc).
        current_iteration : int
            Current iteration to determine if callback should be evaluated.

        Returns
        -------
        c : list
            Return values of callback functions.

        See Also
        --------
        _evaluate
        evaluate_objectives
        evaluate_nonlinear_constraints

        """
        x = list(x)

        self.logger.info(f'evaluate nonlinear constraints at {x}')

        c = []

        if cache_new is None:
            cache_new = self.setup_cache()

        for callback in self.callbacks:
            if not current_iteration % callback.frequency == 0:
                continue
            callback.results_dir = results_dir
            try:
                value = self._evaluate(
                    x, callback, cache, cache_new, make_copy, force,
                )
                c += value
            except CADETProcessError:
                self.logger.warn(
                    f'Evaluation of {callback.name} failed at {x}. '
                )

        cache_new = {key: cache_new[key] for key in cache}

        return c

    @untransforms
    @ensures2d
    def evaluate_callbacks_population(
            self, population, results_dir, cache=None, force=False, n_cores=0,
            current_iteration=0):

        def eval_fun(ind):
            results = self.evaluate_callbacks(
                ind, cache=cache, make_copy=True, force=force,
                results_dir=results_dir, current_iteration=current_iteration,
            )
            return results

        if n_cores == 1:
            results = []
            for ind in population:
                try:
                    res = eval_fun(ind)
                    results.append(res)
                except CADETProcessError:
                    print(ind)
        else:
            if n_cores == 0:
                n_cores = None
            with pathos.multiprocessing.ProcessPool(ncpus=n_cores) as pool:
                results = pool.map(eval_fun, population)

        return results

    @untransforms
    def _evaluate(
            self,
            x, func,
            cache_prev=None, cache_new=None, make_copy=False, force=False,
            *args, **kwargs):
        """Iterate over all evaluation objects and evaluate at x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.
        func : Evaluator or Objective, or Nonlinear Constraint, or Callback
            Evaluation function.
        cache_prev : dict, optional
            Dictionary with previously cached results.
        cache_new : dict, optional
            Dictionary for caching new results.
        make_copy : bool
            If True, a copy of the EvaluationObjects is used which is required
            for multiprocessing.
        force : bool
            If True, do not use cached results. The default if False.
        *args : tuple
            Additional positional arguments for evaluation function.
        **kwargs : TYPE
            Additional keyword arguments for evaluation function.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """
        self.logger.info(f'evaluate {str(func)} at {x}')

        results = []

        if func.evaluators is not None:
            requires = [*func.evaluators, func]
        else:
            requires = [func]

        if len(func.evaluation_objects) == 0:
            result = self._evaluate_inner(
                x, func, requires,
                cache_prev, cache_new, force,
                *args, **kwargs
            )
            results += result
        else:
            for el in func.evaluation_objects:
                evaluation_objects = self.set_variables(x, el, make_copy)
                eval_obj = evaluation_objects[0]

                if cache_prev is not None:
                    inner_cache_prev = cache_prev[str(eval_obj)].copy()
                else:
                    inner_cache_prev = None
                if cache_new is not None:
                    inner_cache_new = cache_new[str(eval_obj)].copy()
                else:
                    inner_cache_new = None

                result = self._evaluate_inner(
                    eval_obj, func, requires,
                    inner_cache_prev,
                    inner_cache_new,
                    force, x=x,
                    *args, **kwargs
                )
                results += result

                if cache_new is not None:
                    cache_new[str(eval_obj)] = inner_cache_new

        return results

    def _evaluate_inner(
            self, request, func, requires,
            cache_prev=None, cache_new=None, force=False, x=None,
            *args, **kwargs):
        """Iterate over all evaluation requirements and evaluate at request.

        Parameters
        ----------
        request : object
            Argument for evaluation function (e.g. x, EvaluationObject or some
            intermediate result).
        func : Evaluator or Objective, or Nonlinear Constraint, or Callback
            Evaluation function.
        requires : list
            List of steps (evaluators) required for evaluation.
        cache_prev : dict, optional
            Dictionary with previously cached results.
        cache_new : dict, optional
            Dictionary for caching new results.
        force : bool
            If True, do not use cached results. The default if False.
        x : array_like, optional.
            Value of the optimization variables.
            If None, request is used t
        *args : tuple
            Additional positional arguments for evaluation function.
        **kwargs : TYPE
            Additional keyword arguments for evaluation function.

        Raises
        ------
        CADETProcessError
            DESCRIPTION.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        self.logger.info(
            f"Evaluating {func}. "
            f"requires evaluation of {[str(req) for req in requires]}"
        )
        if x is None:
            x = request
        current_request = request

        if cache_prev is not None and not force:
            remaining = []
            for step in reversed(requires):
                try:
                    result = cache_prev[str(step)][tuple(x)]
                    self.logger.info(
                        f'Got {str(step)} results from cache.'
                    )
                    current_request = result
                    break
                except KeyError:
                    pass

                try:
                    result = cache_new[str(step)][tuple(x)]
                    self.logger.info(
                        f'Got {str(step)} results from inner cache.'
                    )
                    current_request = result
                    break
                except KeyError:
                    pass

                remaining.insert(0, step)

        else:
            remaining = requires

        self.logger.debug(
            f'Evaluating remaining functions: '
            f'{[str(step) for step in remaining]}.'
        )

        for step in remaining:
            if isinstance(step, Callback):
                step.evaluate(
                    current_request,
                    x=x, evaluation_object=request,
                    *args, **kwargs
                )
            else:
                result = step.evaluate(current_request, *args, **kwargs)
                if cache_new is not None:
                    cache_new[str(step)][tuple(x)] = result
            current_request = result

        if not isinstance(result, list):
            result = [result]

        if len(result) != func.n_metrics:
            raise CADETProcessError(
                f"Expected length {func.n_metrics} "
                f"for {str(func)}"
            )

        return result

    @property
    def lower_bounds(self):
        """list : List of the lower bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.lb for var in self.variables]

    @property
    def lower_bounds_transformed(self):
        """list : List of the lower bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.transform.lb for var in self.variables]

    @property
    def upper_bounds(self):
        """list : List of the upper bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.ub for var in self.variables]

    @property
    def upper_bounds_transformed(self):
        """list : List of the lower bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var._transform.ub for var in self.variables]

    @untransforms
    def check_bounds(self, x):
        """Checks if all bound constraints are kept.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables

        Returns
        -------
        flag : Bool
            True, if all values of x are within the bounds. False otherwise.

        """
        flag = True

        if np.any(np.less(x, self.lower_bounds)):
            flag = False
        if np.any(np.greater(x, self.upper_bounds)):
            flag = False

        return flag

    @property
    def linear_constraints(self):
        """list : linear inequality constraints of OptimizationProblem

        See Also
        --------
        add_linear_constraint
        remove_linear_constraint
        linear_equality_constraints

        """
        return self._linear_constraints

    @property
    def n_linear_constraints(self):
        """int: number of linear inequality constraints
        """
        return len(self.linear_constraints)

    def add_linear_constraint(self, opt_vars, lhs=1, b=0):
        """Add linear inequality constraints.

        Parameters
        ----------
        opt_vars : list of strings
            Names of the OptimizationVariable to be added.
        lhs : float or list, optional
            Left-hand side / coefficients of the constraints.
            If scalar, same coefficient is used for all variables.
        b : float, optional
            Constraint of inequality constraint. The default is zero.

        Raises
        ------
        CADETProcessError
            If optimization variables do not exist.
            If length of lhs coefficients does not match length of variables.

        See Also
        --------
        linear_constraints
        remove_linear_constraint
        linear_equality_constraints

        """
        if not all(var in self.variables_dict for var in opt_vars):
            raise CADETProcessError('Variable not in variables.')

        if np.isscalar(lhs):
            lhs = len(opt_vars) * [1]

        if len(lhs) != len(opt_vars):
            raise CADETProcessError(
                'Number of lhs coefficients and variables do not match.'
            )

        lincon = dict()
        lincon['opt_vars'] = opt_vars
        lincon['lhs'] = lhs
        lincon['b'] = b

        self._linear_constraints.append(lincon)

    def remove_linear_constraint(self, index):
        """Remove linear inequality constraint.

        Parameters
        ----------
        index : int
            Index of the linear inequality constraint to be removed.

        See Also
        --------
        add_linear_equality_constraint
        linear_equality_constraint

        """
        del(self._linear_constraints[index])

    @property
    def A(self):
        """np.ndarray: LHS Matrix of linear inequality constraints.

        See Also
        --------
        b
        add_linear_constraint
        remove_linear_constraint

        """
        A = np.zeros((len(self.linear_constraints), len(self.variables)))

        for lincon_index, lincon in enumerate(self.linear_constraints):
            for var_index, var in enumerate(lincon['opt_vars']):
                index = self.variables.index(self.variables_dict[var])
                A[lincon_index, index] = lincon['lhs'][var_index]

        return A

    @property
    def b(self):
        """list: Vector form of linear constraints.

        See Also
        --------
        A
        add_linear_constraint
        remove_linear_constraint

        """
        b = [lincon['b'] for lincon in self.linear_constraints]

        return np.array(b)

    @untransforms
    def evaluate_linear_constraints(self, x):
        """Calculate value of linear inequality constraints at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        constraints: np.array
            Value of the linear constraints at point x

        See Also
        --------
        A
        b
        linear_constraints

        """
        values = self.get_dependent_values(x)
        values = np.array(values)

        return self.A.dot(values) - self.b

    @untransforms
    def check_linear_constraints(self, x):
        """Check if linear inequality constraints are met at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        flag : bool
            True if linear inequality constraints are met. False otherwise.

        See Also
        --------
        linear_constraints
        evaluate_linear_constraints
        A
        b

        """
        flag = True

        if np.any(self.evaluate_linear_constraints(x) > 0):
            flag = False

        return flag

    @property
    def linear_equality_constraints(self):
        """list: linear equality constraints of OptimizationProblem

        See Also
        --------
        add_linear_equality_constraint
        remove_linear_equality_constraint
        linear_constraints

        """
        return self._linear_equality_constraints

    @property
    def n_linear_equality_constraints(self):
        """int: number of linear equality constraints"""
        return len(self.linear_constraints)

    def add_linear_equality_constraint(self, opt_vars, lhs=1, beq=0):
        """Add linear equality constraints.

        Parameters
        ----------
        opt_vars : list of strings
            Names of the OptimizationVariable to be added.
        lhs : float or list, optional
            Left-hand side / coefficients of the constraints.
            If scalar, same coefficient is used for all variables.
        b : float, optional
            Constraint of inequality constraint. The default is zero.

        Raises
        ------
        CADETProcessError
            If optimization variables do not exist.
            If length of lhs coefficients does not match length of variables.

        See Also
        --------
        linear_equality_constraints
        remove_linear_equality_constraint
        linear_constraints

        """
        if not all(var in self.variables for var in opt_vars):
            return CADETProcessError('Variables not in variables')

        if np.isscalar(lhs):
            lhs = len(opt_vars) * [1]

        if len(lhs) != len(opt_vars):
            raise CADETProcessError(
                'Number of lhs coefficients and variables do not match.'
            )

        lineqcon = dict()
        lineqcon['opt_vars'] = opt_vars
        lineqcon['lhs'] = lhs
        lineqcon['beq'] = beq

        self._linear_equality_constraints.append(lineqcon)

    def remove_linear_equality_constraint(self, index):
        """Removes at given index the added linear equality constraint.

        Parameters
        ----------
        index : int
            Index of the linear equality constraint to be removed.

        See Also
        --------
        add_linear_equality_constraint
        linear_equality_constraint

        """
        del(self._linear_equality_constraints[index])

    @property
    def Aeq(self):
        """np.ndarray: Matrix form of linear equality constraints.

        See Also
        --------
        beq
        add_linear_equality_constraint
        remove_linear_equality_constraint

        """
        Aeq = np.zeros(
            (len(self.linear_equality_constraints), len(self.variables))
        )

        for lineqcon_index, lineqcon in enumerate(
            self.linear_equality_constraints
        ):
            for var_index, var in enumerate(lineqcon.opt_vars):
                index = self.variables.index(var)
                Aeq[lineqcon_index, index] = lineqcon.lhs[var_index]

        return Aeq

    @property
    def beq(self):
        """list: Vector form of linear equality constraints.

        See Also
        --------
        Aeq
        add_linear_equality_constraint
        remove_linear_equality_constraint

        """
        beq = np.zeros((len(self.linear_equality_constraints),))
        beq = [lineqcon.beq for lineqcon in self.linear_equality_constraints]

        return beq

    @untransforms
    def evaluate_linear_equality_constraints(self, x):
        """Calculate value of linear equality constraints at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        constraints: np.array
            Value of the linear euqlity constraints at point x

        See Also
        --------
        Aeq
        beq
        linear_equality_constraints

        """
        values = self.get_dependent_values(x)
        values = np.array(values)

        return self.Aeq.dot(values) - self.beq

    def check_linear_equality_constraints(self, x):
        """Check if linear equality constraints are met at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        flag : bool
            True if linear equality constraints are met. False otherwise.

        """
        flag = True

        if np.any(self.evaluate_linear_equality_constraints(x) != 0):
            flag = False

        return flag

    @property
    def x0(self):
        """Initival values for optimization.

        Parameters
        ----------
        x0 : array_like
            Initival values for optimization.

        Raises
        ------
        CADETProcessError
            If length of x0 does not match length of optimization variables.

        """
        return self._x0

    @x0.setter
    def x0(self, x0):
        x0 = np.array(x0, ndmin=2)
        if not x0.shape[-1] == len(self.independent_variables):
            raise CADETProcessError(
                "Starting value must be given for all variables"
            )

        for x in x0:
            if not self.check_bounds(x):
                raise CADETProcessError(f'{x} exceeds bounds')
            if not self.check_linear_constraints(x):
                raise CADETProcessError(f'{x} violates linear constraints')

        if len(x0) == 1:
            x0 = x0[0]

        self._x0 = x0.tolist()

    @property
    def x0_transformed(self):
        return self.transform(self.x0)

    def transform(self, x, enforce2d=False):
        x = np.array(x, ndmin=2)
        transform = np.zeros(x.shape)

        for i, ind in enumerate(x):
            transform[i, :] = [
                var.transform_fun(value)
                for value, var in zip(ind, self.independent_variables)
            ]

        transform = transform.tolist()
        if len(transform) == 1 and not enforce2d:
            return transform[0]

        return transform

    def untransform(self, x, enforce2d=False):
        x = np.array(x, ndmin=2)
        untransform = np.zeros(x.shape)

        for i, ind in enumerate(x):
            untransform[i, :] = [
                var.untransform_fun(value)
                for value, var in zip(ind, self.independent_variables)
            ]

        untransform = untransform.tolist()
        if len(untransform) == 1 and not enforce2d:
            return untransform[0]

        return untransform

    def create_initial_values(
            self, n_samples=1, method='random', seed=None, burn_in=100000,
            set_values=True):
        """Create initial value within parameter space.

        Uses hopsy (Highly Optimized toolbox for Polytope Sampling) to retrieve
        uniformly distributed samples from the parameter space.

        Parameters
        ----------
        n_samples : int
            Number of initial values to be drawn
        method : str, optional
            chebyshev: Return center of the minimal-radius ball enclosing the
                entire set .
            random: Any random valid point in the parameter space.
        seed : int, optional
            Seed to initialize random numbers. Only used if method == 'random'
        burn_in: int, optional
            Number of samples that are created to ensure uniform sampling.
            The actual initial values are then drawn from this set.
            The default is 100000.
        set_values : bool, optional
            If True, set the created values as x0. The default is True.

        Raises
        ------
        CADETProcessError
            If method is not known.

        Returns
        -------
        values : list
            Initial values for starting the optimization.

        """
        class CustomModel():
            def __init__(self, log_space_indices: list):
                self.log_space_indices = log_space_indices

            def compute_negative_log_likelihood(self, x):
                return np.log(x[self.log_space_indices])

        log_space_indices = []
        for i, var in enumerate(self.variables):
            if (
                    isinstance(var._transform, NormLogTransform) or
                    (
                        isinstance(var._transform, AutoTransform) and
                        var._transform.use_log
                    )
                ):
                log_space_indices.append(i)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if len(log_space_indices) > 0:
                model = CustomModel(log_space_indices)
            else:
                model = hopsy.UniformModel()

            problem = hopsy.Problem(
                self.A,
                self.b,
                model
            )
            problem = hopsy.add_box_constraints(
                problem,
                self.lower_bounds,
                self.upper_bounds
            )
            try:
                problem = hopsy.round(problem)
            except np.linalg.LinAlgError:
                pass

            chebyshev = hopsy.compute_chebyshev_center(problem)

            if n_samples == 1 and method == 'chebyshev':
                values = np.array(chebyshev, ndmin=2)
            else:
                run = hopsy.Run(
                    problem,
                    starting_points=[chebyshev]
                    )
                if seed is None:
                    seed = random.randint(0, 255)
                run.random_seed = seed
                run.sample(burn_in)

                values = np.array(run.data.states[0])
                indices = np.random.randint(0, burn_in, n_samples)
                values = values[indices]

        for i, ind in enumerate(values):
            for i_var, var in enumerate(self.variables):
                values[i, i_var] = np.format_float_positional(
                    values[i, i_var],
                    precision=var.precision, fractional=False
                )

        indices = [
            i for i, variable in enumerate(self.variables)
            if variable in self.independent_variables
        ]

        values = values[:, indices]

        values = values.tolist()

        if set_values:
            self.x0 = values

        return values

    @property
    def parameters(self):
        parameters = Dict()

        parameters.variables = {
            opt.name: opt.parameters for opt in self.variables
        }
        parameters.linear_constraints = self.linear_constraints

        return parameters

    def setup_cache(self, only_cached_evalutors=False):
        """Helper function to setup cache dictionary structure.

        Structure:
        [evaluation_object][step][x]

        For example:
        [Process 1][ProcessSimulator][x] -> SimulationResults
        [Process 1][Fractionator][x] -> Performance
        [Process 1][Objective 1][x] -> f1.1
        [Process 1][Objective 2][x] -> f2.1
        [Process 1][Constraint 1][x] -> g1.1

        [Process 2][ProcessSimulator][x] -> SimulationResults
        [Process 2][Fractionator][x] -> erformance
        [Process 2][Objective 1][x] -> f1.2
        [Process 2][Objective 2][x] -> f2.2
        [Process 2][Constraint 1]x] -> g1.2

        [CustomSimulator][x] -> SimulationResults
        [Custom Objective][x] -> f3
        [Custom Constraint][x] -> g2

        """
        cache = {}

        if only_cached_evalutors:
            evaluators = self.cached_evaluators
        else:
            evaluators = self.evaluators

        for objective in self.objectives:
            if len(objective.evaluation_objects) == 0:
                cache[objective.name] = {}
                for evaluator in objective.evaluators:
                    if evaluator in evaluators:
                        cache[str(evaluator)] = {}
            else:
                for eval_obj in objective.evaluation_objects:
                    if str(eval_obj) not in cache:
                        cache[str(eval_obj)] = {}

                    cache[str(eval_obj)][objective.name] = {}

                for evaluator in objective.evaluators:
                    if evaluator in evaluators:
                        cache[str(eval_obj)][str(evaluator)] = {}

        for nonlincon in self.nonlinear_constraints:
            if len(nonlincon.evaluation_objects) == 0:
                cache[nonlincon.name] = {}
                for evaluator in nonlincon.evaluators:
                    if evaluator in evaluators:
                        cache[str(evaluator)] = {}
            else:
                for eval_obj in nonlincon.evaluation_objects:
                    if str(eval_obj) not in cache:
                        cache[str(eval_obj)] = {}

                    cache[str(eval_obj)][nonlincon.name] = {}

                for evaluator in nonlincon.evaluators:
                    if evaluator in evaluators:
                        cache[str(eval_obj)][str(evaluator)] = {}

        return cache

    def __str__(self):
        return self.name


class OptimizationVariable():
    """Class for setting the values for the optimization variables.

    Defines the attributes for optimization variables for creating an
    OptimizationVariable. Tries to get the attr of the evaluation_object.
    Raises a CADETProcessErrorif the attribute to be set is not valid.

    Attributes
    ----------
    name : str
        Name of the optimization variable.
    evaluation_objects : list
        List of evaluation objects associated with optimization variable.
    parameter_path : str
        Path of the optimization variable the evaluation_object's parameters.
    lb : float
        Lower bound of the variable.
    ub : float
        upper bound of the variable.
    transform : TransformBase
        Transformation function for parameter normalization.
    component_index : int, optional
        Index for component specific variables.
        If None, variable is assumed to be component independent.
    polynomial_index : int, optional
        Index for specific polynomial coefficient.
        If None, variable is assumed to be component independent.
    precision : int, optional
        Number of significant figures to which variable can be rounded.
        If None, variable is not rounded. The default is None.

    Raises
    ------
    CADETProcessError
        If the attribute is not valid.

    """
    _parameters = ['lb', 'ub', 'component_index', 'precision']

    def __init__(self, name,
                 evaluation_objects=None, parameter_path=None,
                 lb=-math.inf, ub=math.inf, transform=None,
                 component_index=None, polynomial_index=None,
                 precision=None):

        self.name = name
        self._value = None

        if evaluation_objects is not None:
            self.evaluation_objects = evaluation_objects
            self.parameter_path = parameter_path
            self.polynomial_index = polynomial_index
            self.component_index = component_index
        else:
            self.evaluation_objects = None
            self.parameter_path = None
            self.polynomial_index = None
            self.component_index = None

        self.lb = lb
        self.ub = ub

        if transform is None:
            transform = NoTransform(lb, ub)
        else:
            if np.isinf(lb) or np.isinf(ub):
                raise CADETProcessError(
                    "Transform requires bound constraints."
                )
            if transform == 'auto':
                transform = AutoTransform(lb, ub)
            elif transform == 'linear':
                transform = NormLinearTransform(lb, ub)
            elif transform == 'log':
                transform = NormLogTransform(lb, ub)
            else:
                raise ValueError("Unknown transform")

        self._transform = transform

        self.precision = precision

        self._dependencies = []
        self._factors = []
        self._dependency_transforms = []

    @property
    def parameter_path(self):
        return self._parameter_path

    @parameter_path.setter
    def parameter_path(self, parameter_path):
        if parameter_path is not None:
            for eval_obj in self.evaluation_objects:
                if not check_nested(eval_obj.parameters, parameter_path):
                    raise CADETProcessError(
                        'Not a valid Optimization variable'
                    )
        self._parameter_path = parameter_path

    @property
    def parameter_sequence(self):
        """tuple: Tuple of parameters path elements."""
        return tuple(self.parameter_path.split('.'))

    @property
    def transform(self):
        return self._transform

    def transform_fun(self, x):
        return self._transform.transform(x)

    def untransform_fun(self, x):
        return self._transform.untransform(x)

    @property
    def component_index(self):
        return self._component_index

    @component_index.setter
    def component_index(self, component_index):
        if component_index is not None:
            for eval_obj in self.evaluation_objects:
                parameter = get_nested_value(
                    eval_obj.parameters, self.parameter_sequence
                )
                is_polynomial = check_nested(
                    eval_obj.polynomial_parameters, self.parameter_sequence
                )
                if is_polynomial:
                    if component_index > parameter.shape[0]-1:
                        raise CADETProcessError(
                            'Index exceeds components'
                        )
                else:
                    if component_index > len(parameter)-1:
                        raise CADETProcessError('Index exceeds components')
        self._component_index = component_index

    @property
    def polynomial_index(self):
        return self._polynomial_index

    @polynomial_index.setter
    def polynomial_index(self, polynomial_index):
        for eval_obj in self.evaluation_objects:
            is_polynomial = check_nested(
                eval_obj.polynomial_parameters, self.parameter_sequence
            )
            if not is_polynomial and polynomial_index is None:
                break

            elif not is_polynomial and polynomial_index is not None:
                raise CADETProcessError(
                    'Variable is not a polynomial parameter.'
                )

            elif is_polynomial and polynomial_index is None:
                polynomial_index = 0

                break

            parameter = get_nested_value(
                eval_obj.parameters, self.parameter_sequence
            )
            if polynomial_index > parameter.shape[1] - 1:
                raise CADETProcessError(
                    'Index exceeds polynomial coefficients'
                )
        self._polynomial_index = polynomial_index

    def add_dependency(self, dependency, factor=1, transform=None):
        """Add dependency of variable on other variables.

        The value of the variable is determined with the following equation:

        $$
        v = \sum_i^{n_{dep}} \lambda_i \cdot f_i(v_{dep,i})
        $$

        Parameters
        ----------
        dependency : OptimizationVariable
            Variable object to be added as dependency.
        factor : float, optional
            Factor for the dependencies between to events. The default is 1.
        transform: callable
            Transform function for dependent variable.

        Raises
        ------
        CADETProcessError
            If the dependency already exists in list dependencies.

        """
        if dependency in self._dependencies:
            raise CADETProcessError("Dependency already exists")

        self._dependencies.append(dependency)
        self._factors.append(factor)
        if transform is None:
            def transform(t):
                return t
        self.dependency_transforms.append(transform)

    def remove_dependency(self, dependency):
        """Remove dependencies of events.

        Parameters
        ----------
        dependency : OptimizationVariable
            Variable object to remove from dependencies.

        Raises
        ------
        CADETProcessError
            If the dependency doesn't exists in list dependencies.

        """
        if dependency in self._dependencies:
            raise CADETProcessError("Dependency not found")

        index = self._dependencies(dependency)

        del(self._dependencies[index])
        del(self._factors[index])
        del(self._dependency_transforms[index])

    @property
    def dependencies(self):
        """list: Events on which the Event depends."""
        return self._dependencies

    @property
    def isIndependent(self):
        """bool: True, if event is independent, False otherwise."""
        if len(self.dependencies) == 0:
            return True
        else:
            return False

    @property
    def factors(self):
        """list: Linear coefficients for dependent variables."""
        return self._factors

    @property
    def dependency_transforms(self):
        """list: Transform functions for dependent variables."""
        return self._dependency_transforms

    @property
    def value(self):
        """float: Value of the parameter.

        If the variable is not independent, the value is calculated from its
        dependencies.

        Raises
        ------
        CADETProcessError
            If the variable is not independent.

        """
        if self.isIndependent:
            if self._value is None:
                raise CADETProcessError("Value not set.")

            value = self._value
        else:
            transformed_value = [
                f(dep.value)
                for f, dep in zip(self.transforms, self.dependencies)
            ]
            value = np.dot(transformed_value, self._factors)
        return value

    @value.setter
    def value(self, value):
        if not np.isscalar(value):
            raise TypeError("Expected scalar value")

        if value < self.lb:
            raise ValueError("Exceeds lower bound")
        if value > self.ub:
            raise ValueError("Exceeds upper bound")

        if self.isIndependent:
            self._value = value
        else:
            raise CADETProcessError("Cannot set time for dependent variables")

    @property
    def parameters(self):
        """dict: parameter dictionary."""
        return Dict({
            param: getattr(self, param) for param in self._parameters
        })

    def __repr__(self):
        if self.evaluation_objects is not None:
            string = \
                f'{self.__class__.__name__}' + \
                f'(name={self.name}, ' + \
                f'evaluation_objects=' \
                f'{[obj.name for obj in self.evaluation_objects]}, ' + \
                f'parameter_path=' \
                f'{self.parameter_path}, lb={self.lb}, ub={self.ub})'
        else:
            string = \
                f'{self.__class__.__name__}' + \
                f'(name={self.name}, lb={self.lb}, ub={self.ub})'
        return string


class Evaluator(metaclass=StructMeta):
    evaluator = Callable()
    args = Tuple()
    kwargs = Dict()

    def __init__(
            self,
            evaluator,
            name=None,
            args=None,
            kwargs=None):

        self.evaluator = evaluator

        if name is None:
            name = str(evaluator)
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, request):
        if self.args is None:
            args = ()
        else:
            args = self.args

        if self.kwargs is None:
            kwargs = {}
        else:
            kwargs = self.kwargs

        results = self.evaluator(request, *args, **kwargs)

        return results

    evaluate = __call__

    def __str__(self):
        return self.name


class Objective(metaclass=StructMeta):
    objective = Callable()
    name = String()
    type = Switch(valid=['minimize', 'maximize'])
    n_objectives = RangedInteger(lb=1)
    n_metrics = n_objectives
    bad_metrics = DependentlySizedList(dep='n_metrics', default=np.inf)

    def __init__(
            self,
            objective,
            name=None,
            type='minimize',
            n_objectives=1,
            bad_metrics=np.inf,
            evaluation_objects=None,
            evaluators=None):
        self.objective = objective

        if name is None:
            name = str(objective)
        self.name = name

        self.type = type
        self.n_objectives = n_objectives

        if np.isscalar(bad_metrics):
            bad_metrics = n_objectives * [bad_metrics]
        self.bad_metrics = bad_metrics

        self.evaluation_objects = evaluation_objects
        self.evaluators = evaluators

    def __call__(self, *args, **kwargs):
        f = self.objective(*args, **kwargs)

        if np.isscalar(f):
            f = [f]

        return f

    evaluate = __call__

    def __str__(self):
        return self.name


class NonlinearConstraint(metaclass=StructMeta):
    nonlinear_constraint = Callable()
    name = String()
    n_nonlinear_constraints = RangedInteger(lb=1)
    n_metrics = n_nonlinear_constraints
    bad_metrics = DependentlySizedList(dep='n_metrics', default=np.inf)

    def __init__(
            self,
            nonlinear_constraint,
            name=None,
            bounds=None,
            n_nonlinear_constraints=1,
            bad_metrics=np.inf,
            evaluation_objects=None,
            evaluators=None):

        self.nonlinear_constraint = nonlinear_constraint

        if name is None:
            name = str(nonlinear_constraint)
        self.name = name

        self.bounds = bounds
        self.n_nonlinear_constraints = n_nonlinear_constraints

        if np.isscalar(bad_metrics):
            bad_metrics = n_nonlinear_constraints * [bad_metrics]
        self.bad_metrics = bad_metrics

        self.evaluation_objects = evaluation_objects
        self.evaluators = evaluators

    def __call__(self, *args, **kwargs):
        g = self.nonlinear_constraint(*args, **kwargs)

        if np.isscalar(g):
            g = [g]

        return g

    evaluate = __call__

    def __str__(self):
        return self.name


class Callback(metaclass=StructMeta):
    """

    Must implement function with the following signature:
    (fractionation, x, evaluation_object, results_dir)
    """

    callback = Callable()
    name = String()
    n_metrics = 1
    frequency = RangedInteger(lb=1)

    def __init__(
            self,
            callback,
            name=None,
            evaluation_objects=None,
            evaluators=None,
            results_dir='./',
            frequency=10):
        self.callback = callback

        if name is None:
            name = str(callback)
        self.name = name

        self.evaluation_objects = evaluation_objects
        self.evaluators = evaluators
        self.results_dir = './'
        self.frequency = frequency

    def __call__(self, current_request, x, evaluation_object):
        kwargs = {}
        if 'x' in inspect.signature(self.callback).parameters:
            kwargs['x'] = x
        if 'evaluation_object' in inspect.signature(self.callback).parameters:
            kwargs['evaluation_object'] = evaluation_object
        if 'results_dir' in inspect.signature(self.callback).parameters:
            kwargs['results_dir'] = self.results_dir

        return self.callback(current_request, **kwargs)

    evaluate = __call__

    def __str__(self):
        return self.name


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

    jac = np.zeros((len(f0), len(xk)), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        jac[:, k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0

    return jac
