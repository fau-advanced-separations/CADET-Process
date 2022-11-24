from functools import wraps
import inspect
import math
from pathlib import Path
import random
import shutil
import warnings

from addict import Dict
import numpy as np
import hopsy
import pathos

from CADETProcess import CADETProcessError
from CADETProcess import log

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

from CADETProcess.optimization import ResultsCache


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
    evaluation_objects : list
        Objects containing parameters to be optimized.
    evaluators : obj
        Objects used to evaluate evaluation_object.
    cache : ResultsCache
        Cache to store (intermediate) results.
    variables : list
        List of optimization variables.
    objectives: list
        Objective functions.
    nonlinear_constraints: list of callables
        Nonlinear constraint functions.
    linear_constraints : list
        List of all linear constraints of an OptimizationProblem.
    linear_equality_constraints : list
        List with all linear equality constrains of an OptimizationProblem.
    callbacks : list
        List of callback functions to record progress.
    meta_scores: list
        Meta score functions.
    multi_criteria_decision_functions : list
        Multi criteria decision functions.

    """
    name = String()

    def __init__(
            self,
            name,
            use_diskcache=True,
            cache_directory=None,
            log_level='INFO'):
        self.name = name
        self.logger = log.get_logger(self.name, level=log_level)

        self._evaluation_objects = []
        self._evaluators = []

        self.cached_evaluators = []
        self.use_diskcache = use_diskcache
        self.cache = ResultsCache(use_diskcache, cache_directory)

        self._variables = []
        self._dependent_variables = []
        self._objectives = []
        self._nonlinear_constraints = []
        self._linear_constraints = []
        self._linear_equality_constraints = []
        self._meta_scores = []
        self._multi_criteria_decision_functions = []
        self._callbacks = []

        self._x0 = None

    def untransforms(func):
        @wraps(func)
        def wrapper(self, x, *args, untransform=False, **kwargs):
            """Untransform population or individual before calling function."""
            if untransform:
                x = self.untransform(x)

            return func(self, x, *args, **kwargs)

        return wrapper

    def gets_dependent_values(func):
        @wraps(func)
        def wrapper(self, x, *args, get_dependent_values=False, **kwargs):
            """Get dependent values of individual before calling function."""
            if get_dependent_values:
                x = self.get_dependent_values(x)

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
            If evaluation object with same name already exists in optimization
            problem.

        """
        if evaluation_object in self._evaluation_objects:
            raise CADETProcessError(
                'Evaluation object already part of optimization problem.'
            )

        if str(evaluation_object) in self.evaluation_objects_dict:
            raise CADETProcessError(
                'Evaluation object with same name already exists.'
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
    def independent_variable_names(self):
        """list: Independent optimization variable names."""
        return [var.name for var in self.independent_variables]

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
            self, name, evaluation_objects=-1, parameter_path=None,
            lb=-math.inf, ub=math.inf, transform=None,
            component_index=None, polynomial_index=None):
        """Add optimization variable to the OptimizationProblem.

        The function encapsulates the creation of OptimizationVariable objects
        in order to prevent invalid OptimizationVariables.

        Parameters
        ----------
        name : str, optional
            Name of the variable. If None, parameter_path is used.
        evaluation_objects : EvaluationObject or list of EvaluationObjects
            Evaluation object to set parameters.
            If -1, all evaluation objects are used.
            If None, no evaluation object is associated (dummy variable).
            The default is -1.
        parameter_path : str, optional
            Path of the parameter including the evaluation object.
            If None, name must be provided.
        lb : float
            Lower bound of the variable value.
        ub : float
            Upper bound of the variable value.
        transform : {'auto', 'log', 'linear', None}:
            Variable transform. The default is auto.
        component_index : int
            Index for component specific variables.
        polynomial_index : int
            Index for specific polynomial coefficient.

        Raises
        ------
        CADETProcessError
            If the Variable already exists in the dictionary.

        See Also
        --------
        evaluation_objects
        OptimizationVariable
        remove_variable

        """
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

        if parameter_path is None and len(evaluation_objects) > 0:
            parameter_path = name

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
            self, dependent_variable, independent_variables, transform):
        """Add dependency between two optimization variables.

        Parameters
        ----------
        dependent_variable : str
            OptimizationVariable whose value will depend on other variables.
        independent_variables : {str, list}
            Independent variable name or list of independent variables names.
        transform : callable
            Function to describe dependency.
            Must take all independent variables as arguments in the order as
            given by independent_variables.
            Returns transformed dependent value.

        Raises
        ------
        CADETProcessError
            If dependent_variable OR independent_variables are not found.

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

        vars = [self.variables_dict[indep] for indep in independent_variables]
        var.add_dependency(vars, transform)

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

        variables = self.independent_variables

        for variable, value in zip(variables, x):
            value = np.format_float_positional(
                value, precision=variable.precision, fractional=False
            )
            variable.value = float(value)

        return self.variable_values

    @untransforms
    def set_variables(self, x, evaluation_objects=-1):
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

        Returns
        -------
        evaluation_object : list
            Evaluation Objects with set paraemters.

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
                try:
                    eval_obj = eval_obj_dict[eval_obj.name]
                except KeyError:
                    continue

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
    def objective_names(self):
        return [str(obj) for obj in self.objectives]

    @property
    def objective_labels(self):
        labels = []
        for obj in self.objectives:
            labels += obj.labels

        return labels

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
            requires=None,
            *args, **kwargs):
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
        args : tuple, optional
            Additional arguments for objective function.
        kwargs : dict, optional
            Additional keyword arguments for objective function.

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
            force=False):
        """Evaluate objective functions at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.
        force : bool
            If True, do not use cached results. The default is False.

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
        self.logger.debug(f'evaluate objectives at {x}')

        x = list(x)
        f = []

        for objective in self.objectives:
            try:
                value = self._evaluate(x, objective, force)
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
            self, population, force=False, n_cores=-1):

        if not self.cache.use_diskcache and n_cores != 1:
            raise CADETProcessError(
                "Cannot use dict cache for multiprocessing."
            )

        def eval_fun(ind):
            results = self.evaluate_objectives(
                ind,
                force=force,
            )
            self.cache.close()

            return results

        if n_cores == 1:
            results = []
            for ind in population:
                res = eval_fun(ind)
                results.append(res)
        else:
            if n_cores == 0 or n_cores == -1:
                n_cores = None

            self.cache.close()

            with pathos.pools.ProcessPool(ncpus=n_cores) as pool:
                results = pool.map(eval_fun, population)

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
        jacobian = approximate_jac(x, self.evaluate_objectives, dx)

        return jacobian

    @property
    def nonlinear_constraints(self):
        return self._nonlinear_constraints

    @property
    def nonlinear_constraint_names(self):
        return [str(nonlincon) for nonlincon in self.nonlinear_constraints]

    @property
    def nonlinear_constraint_labels(self):
        if self.n_nonlinear_constraints > 0:
            labels = []
            for nonlincon in self.nonlinear_constraints:
                labels += nonlincon.labels

            return labels

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
            bounds=0,
            requires=None,
            *args, **kwargs):
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
        args : tuple, optional
            Additional arguments for nonlinear constraint function.
        kwargs : dict, optional
            Additional keyword arguments for nonlinear constraint function.

        Raises
        ------
        TypeError
            If nonlinear constraint function is not callable.
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
            force=False):
        """Evaluate nonlinear constraint functions at point x.

        After evaluating the nonlinear constraint functions, the corresponding
        bounds are subtracted from the results.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.
        force : bool
            If True, do not use cached results. The default is False.

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
        self.logger.debug(f'evaluate nonlinear constraints at {x}')

        x = list(x)
        g = []

        for nonlincon in self.nonlinear_constraints:
            try:
                value = self._evaluate(x, nonlincon, force)
                g += value
            except CADETProcessError:
                self.logger.warn(
                    f'Evaluation of {nonlincon.name} failed at {x}. '
                    f'Returning bad metrics.'
                )
                g += nonlincon.bad_metrics

        c = np.array(g) - np.array(self.nonlinear_constraints_bounds)

        return c.tolist()

    @untransforms
    @ensures2d
    def evaluate_nonlinear_constraints_population(
            self, population, force=False, n_cores=-1):

        if not self.cache.use_diskcache and n_cores != 1:
            raise CADETProcessError(
                "Cannot use dict cache for multiprocessing."
            )

        def eval_fun(ind):
            results = self.evaluate_nonlinear_constraints(ind, force=force)
            self.cache.close()

            return results

        if n_cores == 1:
            results = []
            for ind in population:
                res = eval_fun(ind)
                results.append(res)
        else:
            if n_cores == 0 or n_cores == -1:
                n_cores = None

            self.cache.close()

            with pathos.pools.ProcessPool(ncpus=n_cores) as pool:
                results = pool.map(eval_fun, population)

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
        c = np.array(self.evaluate_nonlinear_constraints(x))

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
        jacobian = approximate_jac(x, self.evaluate_nonlinear_constraints, dx)

        return jacobian

    @property
    def callbacks(self):
        """list: Callback functions for recording progress."""
        return self._callbacks

    @property
    def n_callbacks(self):
        return len(self.callbacks)

    def add_callback(
            self,
            callback,
            evaluation_objects=-1,
            requires=None,
            frequency=1,
            callbacks_dir=None,
            keep_progress=False,
            *args, **kwargs):
        """Add callback function for processing (intermediate) results.

        Parameters
        ----------
        callback : callable
            Callback function.
        evaluation_objects : {EvaluationObject, None, -1, list}
            EvaluationObjects which are evaluated by objective.
            If None, no EvaluationObject is used.
            If -1, all EvaluationObjects are used.
        requires : {None, Evaluator, list}, optional
            Evaluators used for preprocessing.
            If None, no preprocessing is required.
            The default is None.
        callbacks_dir : Path, optional
            Dicretory to store results. If None, folder in working directory is
            created.
        args : tuple, optional
            Additional arguments for callback function.
        kwargs : dict, optional
            Additional keyword arguments for callback function.

        Raises
        ------
        TypeError
            If callback is not callable.
        CADETProcessError
            If EvaluationObject is not found.
        CADETProcessError
            If Evaluator is not found.

        """

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
            frequency=frequency,
            callbacks_dir=callbacks_dir,
            keep_progress=keep_progress,
        )
        self._callbacks.append(callback)

    def evaluate_callbacks(
            self,
            ind,
            current_iteration=1,
            force=False):
        """Evaluate callback functions at point x.

        Parameters
        ----------
        ind : Individual
            Individual to be evalauted.
        current_iteration : int
            Current iteration to determine if callback should be evaluated.
        force : bool
            If True, do not use cached results. The default is False.

        See Also
        --------
        _evaluate
        evaluate_objectives
        evaluate_nonlinear_constraints

        """
        self.logger.debug(f'evaluate callbacks at {ind.x}')
        x = self.untransform(ind.x)

        for callback in self.callbacks:
            if not current_iteration % callback.frequency == 0:
                continue

            callback._ind = ind

            try:
                self._evaluate(x, callback, force)
            except CADETProcessError:
                self.logger.warn(
                    f'Evaluation of {callback} failed at {x}.'
                )

    def evaluate_callbacks_population(
            self, population, current_iteration, force=False, n_cores=-1):

        if not self.cache.use_diskcache and n_cores != 1:
            raise CADETProcessError(
                "Cannot use dict cache for multiprocessing."
            )

        def eval_fun(ind):
            self.evaluate_callbacks(
                ind,
                current_iteration,
                force=force
            )
            self.cache.close()

        if n_cores == 1:
            for ind in population:
                eval_fun(ind)
        else:
            if n_cores == 0 or n_cores == -1:
                n_cores = None

            self.cache.close()

            with pathos.pools.ProcessPool(ncpus=n_cores) as pool:
                pool.map(eval_fun, population)

    @property
    def meta_scores(self):
        """list: Meta scores for multi criteria selection."""
        return self._meta_scores

    @property
    def meta_score_names(self):
        return [str(meta_score) for meta_score in self.meta_scores]

    @property
    def meta_score_labels(self):
        if self.n_meta_scores > 0:
            labels = []
            for meta_score in self.meta_scores:
                labels += meta_score.labels

            return labels

    @property
    def n_meta_scores(self):
        n_meta_scores = 0

        for meta_score in self.meta_scores:
            if len(meta_score.evaluation_objects) != 0:
                factor = len(meta_score.evaluation_objects)
            else:
                factor = 1
            n_meta_scores += factor*meta_score.n_meta_scores

        return n_meta_scores

    def add_meta_score(
            self,
            meta_score,
            n_meta_scores=1,
            evaluation_objects=-1,
            requires=None):

        if not callable(meta_score):
            raise TypeError("Expected callable meta score.")

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

        meta_score = MetaScore(
            meta_score,
            n_meta_scores=n_meta_scores,
            evaluation_objects=evaluation_objects,
            evaluators=evaluators,
        )
        self._meta_scores.append(meta_score)

    @untransforms
    def evaluate_meta_scores(
            self,
            x,
            force=False):
        """Evaluate meta functions at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        m : list
            Meta scores.

        See Also
        --------
        _evaluate
        evaluate_objectives
        evaluate_nonlinear_constraints

        """
        self.logger.debug(f'evaluate meta functions at {x}')

        x = list(x)
        m = []

        for meta_score in self.meta_scores:
            try:
                value = self._evaluate(x, meta_score, force)
                m += value
            except CADETProcessError:
                self.logger.warn(
                    f'Evaluation of {meta_score.name} failed at {x}. '
                )

        return m

    @untransforms
    @ensures2d
    def evaluate_meta_scores_population(
            self, population, force=False, n_cores=-1):

        if not self.cache.use_diskcache and n_cores != 1:
            raise CADETProcessError(
                "Cannot use dict cache for multiprocessing."
            )

        def eval_fun(ind):
            results = self.evaluate_meta_scores(ind, force=force)
            self.cache.close()

            return results

        if n_cores == 1:
            results = []
            for ind in population:
                res = eval_fun(ind)
                results.append(res)
        else:
            if n_cores == 0 or n_cores == -1:
                n_cores = None

            self.cache.close()

            with pathos.pools.ProcessPool(ncpus=n_cores) as pool:
                results = pool.map(eval_fun, population)

        return results

    @property
    def multi_criteria_decision_functions(self):
        """list: Multi criteria decision functions."""
        return self._multi_criteria_decision_functions

    @property
    def n_multi_criteria_decision_functions(self):
        return len(self.multi_criteria_decision_functions)

    def add_multi_criteria_decision_function(self, decision_function):

        if not callable(decision_function):
            raise TypeError("Expected callable decision function.")

        meta_score = MultiCriteriaDecisionFunction(decision_function)
        self._multi_criteria_decision_functions.append(meta_score)

    def evaluate_multi_criteria_decision_functions(self, pareto_population):
        """Evaluate evaluate multi criteria decision functions.

        Parameters
        ----------
        pareto_population : Population
            Pareto optimal solution

        Returns
        -------
        x_pareto : list
            Value of the optimization variables.

        See Also
        --------
        _evaluate
        evaluate_objectives
        evaluate_nonlinear_constraints

        """
        self.logger.debug('Evaluate multi criteria decision functions.')

        for func in self.multi_criteria_decision_functions:
            pareto_population = func(pareto_population)

        return pareto_population

    @property
    def cached_steps(self):
        return \
            self.cached_evaluators + \
            self.objectives + \
            self.nonlinear_constraints

    def prune_cache(self):
        self.cache.prune()

    def delete_cache(self):
        self.cache.close()
        self.cache.delete_database()
        self.cache = None

    @untransforms
    def _evaluate(self, x, func, force=False):
        """Iterate over all evaluation objects and evaluate at x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.
        func : Evaluator or Objective, or Nonlinear Constraint, or Callback
            Evaluation function.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """
        self.logger.debug(f'evaluate {str(func)} at {x}')

        results = []

        if func.evaluators is not None:
            requires = [*func.evaluators, func]
        else:
            requires = [func]

        evaluation_objects = self.set_variables(x, func.evaluation_objects)
        if len(evaluation_objects) == 0:
            evaluation_objects = [None]

        for eval_obj in evaluation_objects:
            self.logger.debug(
                f"Evaluating {func}. "
                f"requires evaluation of {[str(req) for req in requires]}"
            )

            if eval_obj is None:
                current_request = x
            else:
                current_request = eval_obj

            if not force:
                remaining = []
                for step in reversed(requires):
                    try:
                        result = self.cache.get(eval_obj, step, x)
                        self.logger.debug(
                            f'Got {str(step)} results from cache.'
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
                    step.evaluate(current_request, eval_obj)
                else:
                    result = step.evaluate(current_request)
                    if step not in self.cached_steps:
                        tag = 'temp'
                    else:
                        tag = None
                    self.cache.set(eval_obj, step, x, result, tag=tag)
                current_request = result

            if not isinstance(result, list):
                result = [result]

            if len(result) != func.n_metrics:
                raise CADETProcessError(
                    f"Expected length {func.n_metrics} "
                    f"for {str(func)}"
                )

            results += result

        return results

    @property
    def lower_bounds(self):
        """list : Lower bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.lb for var in self.variables]

    @property
    def lower_bounds_transformed(self):
        """list : Transformed lower bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.transform.lb for var in self.variables]

    @property
    def lower_bounds_independent(self):
        """list : Lower bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.lb for var in self.independent_variables]

    @property
    def lower_bounds_independent_transformed(self):
        """list : Transformed lower bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.transform.lb for var in self.independent_variables]

    @property
    def upper_bounds(self):
        """list : Upper bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.ub for var in self.variables]

    @property
    def upper_bounds_transformed(self):
        """list : Transformed upper bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var._transform.ub for var in self.variables]

    @property
    def upper_bounds_independent(self):
        """list : Upper bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.ub for var in self.independent_variables]

    @property
    def upper_bounds_independent_transformed(self):
        """list : Transformed upper bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var._transform.ub for var in self.independent_variables]

    @untransforms
    @gets_dependent_values
    def check_bounds(self, x):
        """Checks if all bound constraints are kept.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables

        Returns
        -------
        flag : Bool
            True, if all values are within the bounds. False otherwise.

        """
        flag = True

        values = np.array(x)

        if np.any(np.less(values, self.lower_bounds)):
            flag = False
        if np.any(np.greater(values, self.upper_bounds)):
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
        if not isinstance(opt_vars, list):
            opt_vars = [opt_vars]

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
    @gets_dependent_values
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
        values = np.array(x)

        return self.A.dot(values) - self.b

    @untransforms
    @gets_dependent_values
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
        return len(self.linear_equality_constraints)

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
        """Initial values for optimization.

        Expected to only contain untransformed independent variables.

        Parameters
        ----------
        x0 : array_like
            Initial values for optimization.

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
            if not self.check_bounds(x, get_dependent_values=True):
                raise CADETProcessError(f'{x} exceeds bounds')
            if not self.check_linear_constraints(x, get_dependent_values=True):
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
                return np.sum(np.log(x[self.log_space_indices]))

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

            lp = hopsy.LP()
            lp.reset()
            lp.settings.thresh = 1e-15

            if len(log_space_indices) > 0:
                model = CustomModel(log_space_indices)
            else:
                model = None

            problem = hopsy.Problem(
                self.A,
                self.b,
                model,
            )

            problem = hopsy.add_box_constraints(
                problem,
                self.lower_bounds,
                self.upper_bounds,
                simplify=False,
            )

            # !!! Additional checks in place to handle PolyRound.round()
            # removing "small" dimensions.
            # Bug reported, Check for future release!
            chebyshev_orig = hopsy.compute_chebyshev_center(problem)[:, 0]

            try:
                problem_rounded = hopsy.round(problem)
            except ValueError:
                problem_rounded = problem

            if problem_rounded.A.shape[1] == problem.A.shape[1]:
                chebyshev_rounded = hopsy.compute_chebyshev_center(problem_rounded)[:, 0]

                if np.all(np.greater(chebyshev_rounded, self.lower_bounds)):
                    problem = problem_rounded
                    chebyshev = chebyshev_rounded
                else:
                    chebyshev = chebyshev_orig

            if n_samples == 1 and method == 'chebyshev':
                values = np.array(chebyshev_orig, ndmin=2)
            else:
                if seed is None:
                    seed = random.randint(0, 255)

                rng = np.random.default_rng(seed)

                mc = hopsy.MarkovChain(
                    problem,
                    proposal=hopsy.UniformCoordinateHitAndRunProposal,
                    starting_point=chebyshev
                )
                rng_hopsy = hopsy.RandomNumberGenerator(seed=seed)

                acceptance_rate, states = hopsy.sample(
                    mc, rng_hopsy, n_samples=burn_in, thinning=2
                )
                values = states[0, ...]

        independent_indices = [
            i for i, variable in enumerate(self.variables)
            if variable in self.independent_variables
        ]
        independent_values = values[:, independent_indices]

        if n_samples == 1 and method == 'chebyshev':
            values = independent_values
        else:
            values = []
            counter = 0
            while len(values) < n_samples:
                if counter > burn_in:
                    raise CADETProcessError(
                        "Cannot find invididuals that fulfill constraints."
                    )

                counter += 1
                i = rng.integers(0, burn_in)
                ind = []
                for i_var, var in enumerate(self.independent_variables):
                    ind.append(
                        float(np.format_float_positional(
                            independent_values[i, i_var],
                            precision=var.precision, fractional=False
                        ))
                    )

                if not self.check_bounds(ind, get_dependent_values=True):
                    continue
                if not self.check_linear_constraints(ind, get_dependent_values=True):
                    continue
                values.append(ind)

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

        if lb >= ub:
            raise ValueError(
                "Lower bound cannot be larger or equal than upper bound."
            )
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
        self._dependency_transform = None

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
                if self.is_polynomial:
                    if component_index > parameter.shape[0]-1:
                        raise CADETProcessError(
                            'Index exceeds components'
                        )
                else:
                    if (
                            np.isscalar(parameter) or
                            component_index > len(parameter)-1):
                        raise CADETProcessError('Index exceeds components')
        self._component_index = component_index

    @property
    def polynomial_index(self):
        return self._polynomial_index

    @polynomial_index.setter
    def polynomial_index(self, polynomial_index):
        is_polynomial = False

        for eval_obj in self.evaluation_objects:
            try:
                is_polynomial = check_nested(
                    eval_obj.polynomial_parameters, self.parameter_sequence
                )
            except AttributeError as e:
                if str(e) != "'EvaluationObject' object has no attribute 'polynomial_parameters'":
                    raise
                else:
                    is_polynomial = False
                    break

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

        self._is_polynomial = is_polynomial
        self._polynomial_index = polynomial_index

    @property
    def is_polynomial(self):
        return self._is_polynomial

    def add_dependency(self, dependencies, transform):
        """Add dependency of Variable on other Variables.

        Parameters
        ----------
        dependencies : list
            List of OptimizationVariables to be added as dependencies.
        transform: callable
            Transform function describing dependency on /independent Variables.

        Raises
        ------
        CADETProcessError
            If the variable is already dependent.
            If transform signature does not match independent Variables.

        """
        if not self.isIndependent:
            raise CADETProcessError("Variable already is dependent.")

        self._dependencies = dependencies
        self.dependency_transform = transform

    @property
    def dependencies(self):
        """list: Independent variables on which the Variable depends."""
        return self._dependencies

    @property
    def isIndependent(self):
        """bool: True, if Variable is independent, False otherwise."""
        if len(self.dependencies) == 0:
            return True
        else:
            return False

    @property
    def value(self):
        """float: Value of the parameter.

        If the Variable is not independent, the value is calculated from its
        dependencies.

        Raises
        ------
        CADETProcessError
            If the Variable is not independent.

        """
        if self.isIndependent:
            if self._value is None:
                raise CADETProcessError("Value not set.")

            value = self._value
        else:
            dependencies = [dep.value for dep in self.dependencies]
            value = self.dependency_transform(*dependencies)

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
            evaluators=None,
            labels=None,
            args=None,
            kwargs=None):

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

        self.labels = labels

        self.args = args
        self.kwargs = kwargs

    @property
    def labels(self):
        """list: List of metric labels."""
        if self._labels is not None:
            return self._labels

        try:
            labels = self.objective.labels
        except AttributeError:
            labels = [f'{self.objective}']
            if self.n_metrics > 1:
                labels = [
                    f'{self.objective}_{i}'
                    for i in range(self.n_metrics)
                ]
        return labels

    @labels.setter
    def labels(self, labels):
        if labels is not None:

            if len(labels) != self.n_metrics:
                raise CADETProcessError(
                    f"Expected {self.n_metrics} labels."
                )

        self._labels = labels

    def __call__(self, request):
        if self.args is None:
            args = ()
        else:
            args = self.args

        if self.kwargs is None:
            kwargs = {}
        else:
            kwargs = self.kwargs

        f = self.objective(request, *args, **kwargs)

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
            bounds=0,
            n_nonlinear_constraints=1,
            bad_metrics=np.inf,
            evaluation_objects=None,
            evaluators=None,
            labels=None,
            args=None,
            kwargs=None):

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

        self.labels = labels

        self.args = args
        self.kwargs = kwargs

    @property
    def labels(self):
        """list: List of metric labels."""
        if self._labels is not None:
            return self._labels

        try:
            labels = self.nonlinear_constraint.labels
        except AttributeError:
            labels = [f'{self.nonlinear_constraint}']
            if self.n_metrics > 1:
                labels = [
                    f'{self.nonlinear_constraint}_{i}'
                    for i in range(self.n_metrics)
                ]
        return labels

    @labels.setter
    def labels(self, labels):
        if labels is not None:

            if len(labels) != self.n_metrics:
                raise CADETProcessError(
                    f"Expected {self.n_metrics} labels."
                )

        self._labels = labels

    def __call__(self, request):
        if self.args is None:
            args = ()
        else:
            args = self.args

        if self.kwargs is None:
            kwargs = {}
        else:
            kwargs = self.kwargs

        g = self.nonlinear_constraint(request, *args, **kwargs)

        if np.isscalar(g):
            g = [g]

        return g

    evaluate = __call__

    def __str__(self):
        return self.name


class Callback(metaclass=StructMeta):
    """

    Must implement function with the following signature:
        results : obj
            x or final result of evaluation toolchain.
        individual : Individual, optional
            Information about current step of optimzer.
        evaluation_object : obj, optional
            Current evaluation object.
        callbacks_dir : Path, optional
            Path to store results.
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
            frequency=10,
            callbacks_dir=None,
            keep_progress=False,
            args=None,
            kwargs=None):

        self.callback = callback

        if name is None:
            name = str(callback.__name__)
        self.name = name

        self.evaluation_objects = evaluation_objects
        self.evaluators = evaluators

        self.frequency = frequency

        if callbacks_dir is not None:
            callbacks_dir = Path(callbacks_dir)
            callbacks_dir.mkdir(exist_ok=True, parents=True)
        self.callbacks_dir = callbacks_dir

        self.keep_progress = keep_progress

        self.args = args
        self.kwargs = kwargs

    def cleanup(self, callbacks_dir, current_iteration):
        if \
                not current_iteration % self.frequency == 0 \
                or current_iteration <= self.frequency:
            return

        previous_iteration = current_iteration - self.frequency

        if self.callbacks_dir is not None:
            callbacks_dir = self.callbacks_dir

        if self.keep_progress:
            new_directory = callbacks_dir / "progress" / str(previous_iteration)
            new_directory.mkdir(exist_ok=True, parents=True)

        for file in callbacks_dir.iterdir():
            if not file.is_file():
                continue
            if self.keep_progress:
                shutil.copy(file, new_directory)
            file.unlink()

    def __call__(self, request, evaluation_object):
        if self.args is None:
            args = ()
        else:
            args = self.args

        if self.kwargs is None:
            kwargs = {}
        else:
            kwargs = self.kwargs

        signature = inspect.signature(self.callback).parameters
        if 'individual' in signature:
            kwargs['individual'] = self._ind
        if 'evaluation_object' in signature:
            kwargs['evaluation_object'] = evaluation_object
        if 'callbacks_dir' in signature:
            if self.callbacks_dir is not None:
                callbacks_dir = self.callbacks_dir
            else:
                callbacks_dir = self._callbacks_dir
            kwargs['callbacks_dir'] = callbacks_dir

        return self.callback(request, *args, **kwargs)

    evaluate = __call__

    def __str__(self):
        return self.name


class MetaScore(metaclass=StructMeta):
    meta_score = Callable()
    name = String()
    n_meta_scores = RangedInteger(lb=1)
    n_metrics = n_meta_scores
    bad_metrics = DependentlySizedList(dep='n_metrics', default=np.inf)

    def __init__(
            self,
            meta_score,
            name=None,
            n_meta_scores=1,
            bad_metrics=np.inf,
            evaluation_objects=None,
            evaluators=None,
            labels=None):

        self.meta_score = meta_score

        if name is None:
            name = str(meta_score)
        self.name = name

        self.n_meta_scores = n_meta_scores

        if np.isscalar(bad_metrics):
            bad_metrics = n_meta_scores * [bad_metrics]
        self.bad_metrics = bad_metrics

        self.evaluation_objects = evaluation_objects
        self.evaluators = evaluators

        self.labels = labels

    @property
    def labels(self):
        """list: List of metric labels."""
        if self._labels is not None:
            return self._labels

        try:
            labels = self.meta_score.labels
        except AttributeError:
            labels = [f'{self.meta_score}']
            if self.n_metrics > 1:
                labels = [
                    f'{self.meta_score}_{i}'
                    for i in range(self.n_metrics)
                ]
        return labels

    @labels.setter
    def labels(self, labels):
        if labels is not None:

            if len(labels) != self.n_metrics:
                raise CADETProcessError(
                    f"Expected {self.n_metrics} labels."
                )

        self._labels = labels

    def __call__(self, *args, **kwargs):
        m = self.meta_score(*args, **kwargs)

        if np.isscalar(m):
            m = [m]

        return m

    evaluate = __call__

    def __str__(self):
        return self.name


class MultiCriteriaDecisionFunction(metaclass=StructMeta):
    decision_function = Callable()
    name = String()

    def __init__(
            self,
            decision_function,
            name=None):

        self.decision_function = decision_function

        if name is None:
            name = str(decision_function)
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.decision_function(*args, **kwargs)

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
    f0 = np.array(f(*((xk,) + args)))

    jac = np.zeros((len(f0), len(xk)), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        f_k = np.array(f(*((xk + d,) + args)))
        jac[:, k] = (f_k - f0) / d[k]
        ei[k] = 0.0

    return jac
