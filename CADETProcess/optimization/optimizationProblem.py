from collections import defaultdict
import copy
from functools import wraps
import inspect
import math
from pathlib import Path
import random
import shutil
import uuid
import warnings

from addict import Dict
import numpy as np
import hopsy

from CADETProcess import CADETProcessError
from CADETProcess import log
from CADETProcess import settings

from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import (
    ParameterBase, Sized, Typed, Bool, Integer, Float, String, Switch,
    RangedInteger, Callable, Tuple, SizedNdArray
)
from CADETProcess.dataStructure import frozen_attributes
from CADETProcess.dataStructure import (
    check_nested, generate_nested_dict, get_nested_value, get_nested_attribute,
    set_nested_list_value
)
from CADETProcess.dynamicEvents.section import (
    generate_indices, unravel, get_inhomogeneous_shape, get_full_shape
)

from CADETProcess.optimization.parallelizationBackend import SequentialBackend
from CADETProcess.transform import (
    NoTransform, AutoTransform, NormLinearTransform, NormLogTransform
)

from CADETProcess.metric import MetricBase

from CADETProcess.optimization import ResultsCache


@frozen_attributes
class OptimizationProblem(Structure):
    """Class for configuring optimization problems.

    Stores information about
    - optimization variables
    - objectives
    - linear and nonlinear constraints
    - callbacks
    - meta scores
    - multi-criteria-decision functions

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
        """Initialize OptimizationProblem.

        Parameters
        ----------
        name : str
            Name of the optimization problem.
        use_diskcache : bool, optional
            If True, use diskcache to cache intermediate results. The default is True.
        cache_directory : Path, optional
            Path for results cache database. If None, `working_directory` is used.
            Only has an effect if `use_diskcache` is True.
        log_level : {'DEBUG', INFO, WARN, ERROR, CRITICAL}, optional
            Log level. The default is 'INFO'.

        """
        self.name = name
        self.logger = log.get_logger(self.name, level=log_level)

        self._evaluation_objects_dict = {}
        self._evaluators = []

        self.cached_evaluators = []
        self.use_diskcache = use_diskcache
        self.cache_directory = cache_directory
        self.setup_cache()

        self._variables = []
        self._dependent_variables = []
        self._objectives = []
        self._nonlinear_constraints = []
        self._linear_constraints = []
        self._linear_equality_constraints = []
        self._meta_scores = []
        self._multi_criteria_decision_functions = []
        self._callbacks = []

    def untransforms(func):
        """Untransform population or individual before calling function."""
        @wraps(func)
        def wrapper(self, x, *args, untransform=False, **kwargs):
            """Untransform population or individual before calling function."""
            x = np.array(x, ndmin=1)
            if untransform:
                x = self.untransform(x)

            return func(self, x, *args, **kwargs)

        return wrapper

    def gets_dependent_values(func):
        """Get dependent values of individual before calling function."""
        @wraps(func)
        def wrapper(self, x, *args, get_dependent_values=False, **kwargs):
            if get_dependent_values:
                x = self.get_dependent_values(x)

            return func(self, x, *args, **kwargs)

        return wrapper

    def ensures2d(func):
        """Make sure population is ndarray with ndmin=2."""
        @wraps(func)
        def wrapper(self, population, *args, **kwargs):
            population = np.array(population, ndmin=2)

            return func(self, population, *args, **kwargs)

        return wrapper

    @property
    def evaluation_objects(self):
        """list: Objects to be evaluated during optimization.

        See Also
        --------
        OptimizatonVariable
        Evaluator
        evaluate
        Performance
        objectives
        nonlinear_constraints

        """
        return list(self._evaluation_objects_dict.values())

    @property
    def evaluation_objects_dict(self):
        """dict: Evaluation objects names and objects."""
        return self._evaluation_objects_dict

    def add_evaluation_object(self, evaluation_object, name=None):
        """Add evaluation object to the optimization problem.

        Parameters
        ----------
        evaluation_object : obj
            evaluation object to be added to the optimization problem.
        name : str, optional
            Name of the evaluation_object. If None, parameter_path is used.

        Raises
        ------
        CADETProcessError
            If evaluation object already exists in optimization problem.
            If evaluation object with same name already exists in optimization
            problem.

        """
        if name is None:
            name = str(evaluation_object)

        if evaluation_object in self.evaluation_objects:
            raise CADETProcessError(
                'Evaluation object already part of optimization problem.'
            )

        if name in self.evaluation_objects_dict:
            raise CADETProcessError(
                'Evaluation object with same name already exists.'
            )

        self._evaluation_objects_dict[name] = evaluation_object

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
        return list(filter(lambda var: var.is_independent, self.variables))

    @property
    def independent_variable_names(self):
        """list: Independent optimization variable names."""
        return [var.name for var in self.independent_variables]

    @property
    def n_independent_variables(self):
        """int: Number of independent optimization variables."""
        return len(self.independent_variables)

    @property
    def independent_variable_indices(self):
        """list: Indices of indpeendent variables."""
        return [
            i for i, var in enumerate(self.variable_names)
            if var in self.independent_variable_names
        ]

    @property
    def dependent_variables(self):
        """list: OptimizationVaribles with dependencies."""
        return list(
            filter(lambda var: var.is_independent is False, self.variables)
        )

    @property
    def dependent_variable_names(self):
        """list: Dependent optimization variable names."""
        return [var.name for var in self.dependent_variables]

    @property
    def n_dependent_variables(self):
        """int: Number of dependent optimization variables."""
        return len(self.dependent_variables)

    @property
    def variables_dict(self):
        """dict: All optimization variables indexed by variable name."""
        return {var.name: var for var in self.variables}

    @property
    def variable_values(self):
        """list: Values of optimization variables."""
        return [var.value for var in self.variables]

    def add_variable(
            self, name, evaluation_objects=-1, parameter_path=None,
            lb=-math.inf, ub=math.inf, transform=None, indices=None):
        """Add optimization variable to the OptimizationProblem.

        The function encapsulates the creation of OptimizationVariable objects
        in order to prevent invalid OptimizationVariables.

        Parameters
        ----------
        name : str
            Name of the variable.
        evaluation_objects : EvaluationObject or list of EvaluationObjects
            Evaluation object to set parameters.
            If -1, all evaluation objects are used.
            If None, no evaluation object is associated (dummy variable).
            The default is -1.
        parameter_path : str, optional
            Path of the parameter including the evaluation object.
            If None, name is used.
        lb : float
            Lower bound of the variable value.
        ub : float
            Upper bound of the variable value.
        transform : {'auto', 'log', 'linear', None}:
            Variable transform. The default is auto.
        indices : int  or tuple, optional
            Indices for variables that modify entries of a parameter array.
            If None, variable is assumed to be index independent.

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
        if parameter_path is not None and len(evaluation_objects) == 0:
            raise ValueError(
                "Cannot set parameter_path for variable without evaluation object "
            )

        var = OptimizationVariable(
            name, evaluation_objects, parameter_path,
            lb=lb, ub=ub, transform=transform,
            indices=indices,
        )

        self._variables.append(var)

        with warnings.catch_warnings():
            warnings.simplefilter('error')  # Treat warnings as errors

            try:
                self.check_duplicate_variables()
            except UserWarning as e:
                self._variables.remove(var)
                raise CADETProcessError(e)

        super().__setattr__(name, var)

        return var

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

    @property
    def parameter_variables(self):
        """dict: List of variables for every evaluation object parameter.

        Notes
        -----
        For evaluation objects and index dependent variables, individual keys are added.

        """
        parameter_variables = Dict()

        # Create dict with variables
        for var in self.variables:
            parameter_path = var.parameter_path
            if parameter_path is None:
                parameter_path = var.name

            if var.indices is not None:
                for i, eval_obj in enumerate(var.evaluation_objects):
                    if var.indices[i] is not None:
                        for ind in var.full_indices[0]:
                            parameter_variables[var.parameter_path][eval_obj][ind] = []
                    else:
                        parameter_variables[var.parameter_path][eval_obj] = []
            else:
                parameter_variables[parameter_path] = []

        # Populate dict with variables, add nested structure for indices, if required
        for var in self.variables:
            parameter_path = var.parameter_path
            if parameter_path is None:
                parameter_path = var.name

            if var.indices is not None:
                for i, eval_obj in enumerate(var.evaluation_objects):
                    if var.indices[i] is not None:
                        for ind in var.full_indices[0]:
                            parameter_variables[var.parameter_path][eval_obj][ind].append(var)
                    else:
                        parameter_variables[var.parameter_path][eval_obj].append(var)
            else:
                parameter_variables[parameter_path].append(var)

        return parameter_variables

    def check_duplicate_variables(self):
        """Raise warning if duplicate variables exist."""
        flag = True

        for param, variables in self.parameter_variables.items():
            if isinstance(variables, list):
                if len(variables) > 1:
                    warnings.warn(
                        f"Found multiple entries for variable {param}: {variables}"
                    )
                    flag = False
                else:
                    continue

            # Parameter variables dict also contains evaluation objects
            for eval_obj, eval_obj_variables in variables.items():
                if isinstance(eval_obj_variables, list):
                    if len(eval_obj_variables) > 1:
                        warnings.warn(
                            f"Found multiple entries for variable {param} in {eval_obj}"
                            f" : {eval_obj_variables}"
                        )
                        flag = False
                    else:
                        continue

                # Parameter variables dict also contains index variables
                for ind, index_variables in eval_obj_variables.items():
                    if len(index_variables) > 1:
                        warnings.warn(
                            f"Found multiple entries for variable `{param}` "
                            f"and index {ind} in {eval_obj}; "
                            f"caused by: {[var.name for var in index_variables]}"
                        )
                        flag = False

        return flag


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
            Value of the optimization variables in untransformed space.

        Raises
        ------
        CADETProcessError
            If length of parameters does not match.

        Returns
        -------
        x : list
            Value of all optimization variables in untransformed space.

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
    def get_independent_values(self, x):
        """Remove dependent values from x.

        Parameters
        ----------
        x : list
            Value of all optimization variables.
            Works for transformed and untransformed space.

        Raises
        ------
        CADETProcessError
            If length of parameters does not match.

        Returns
        -------
        x_independent : list
            Values of all independent optimization variables.

        """
        if len(x) != self.n_variables:
            raise CADETProcessError(
                f'Expected {self.n_variables} value(s)'
            )

        x_independent = []

        for variable, value in zip(self.variables, x):
            if variable.is_independent:
                x_independent.append(value)

        return x_independent

    @untransforms
    def set_variables(self, x, evaluation_objects=-1):
        """Set the values from the x-vector to the EvaluationObjects.

        Parameters
        ----------
        x : array_like
            Value of all optimization variables in untransformed space.
        evaluation_objects : list or EvaluationObject or None or -1
            Evaluations objects to set variables in.
            If None, do not set variables.
            If -1, variables are set to all evaluation objects.
            The default is -1.

        Returns
        -------
        evaluation_object : list
            Evaluation Objects with set parameters.

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

        for variable, value in zip(self.variables, values):
            variable.set_value(value)

    def _evaluate_individual(self, eval_funs, x, force=False):
        """Call evaluation function function at point x.

        This function iterates over all functions in eval_funs (e.g. objectives).
        To parallelize this, use _evaluate_population

        Parameters
        ----------
        eval_funs : list of callables
            Evaluation function.
        x : array_like
            Value of all optimization variables in untransformed space.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        results : list
            Values of the evaluation functions at point x.

        See Also
        --------
        evaluate_objectives
        evaluate_nonlinear_constraints
        _evaluate

        """
        x = np.asarray(x)
        results = np.empty((0,))

        for eval_fun in eval_funs:
            try:
                value = self._evaluate(x, eval_fun, force)
                results = np.hstack((results, value))
            except CADETProcessError as e:
                self.logger.warning(
                    f'Evaluation of {eval_fun.name} failed at {x} with Error "{e}". '
                    f'Returning bad metrics.'
                )
                results = np.hstack((results, eval_fun.bad_metrics))

        return results

    def _evaluate_population(self, eval_fun, population, force=False, parallelization_backend=None):
        """Evaluate eval_fun functions for each point x in population.

        Parameters
        ----------
        eval_fun : callable
            Callable to be evaluated.
        population : list
            Population.
        force : bool, optional
            If True, do not use cached values. The default is False.
        parallelization_backend : ParallelizationBackendBase, optional
            Adapter to parallelization backend library for parallel evaluation of population.

        Raises
        ------
        CADETProcessError
            DESCRIPTION.

        Returns
        -------
        results : list
            DESCRIPTION.

        """
        if parallelization_backend is None:
            parallelization_backend = SequentialBackend()

        if not self.cache.use_diskcache and parallelization_backend.n_cores != 1:
            raise CADETProcessError(
                "Cannot use dict cache for multiprocessing."
            )

        def eval_fun_wrapper(ind):
            results = eval_fun(ind, force=force)
            self.cache.close()

            return results

        if parallelization_backend.n_cores != 1:
            self.cache.close()

        results = parallelization_backend.evaluate(eval_fun_wrapper, population)

        return np.array(results, ndmin=2)

    @untransforms
    def _evaluate(self, x, func, force=False):
        """Iterate over all evaluation objects and evaluate at x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.
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

        results = np.empty((0,))

        if func.evaluators is not None:
            requires = [*func.evaluators, func]
        else:
            requires = [func]

        self.set_variables(x)
        evaluation_objects = self.evaluation_objects

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
                        key = (str(eval_obj), step.id, str(x))
                        result = self.cache.get(key)
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
                    result = np.empty((0))
                else:
                    result = step.evaluate(current_request)
                if step not in self.cached_steps:
                    tag = 'temp'
                else:
                    tag = None
                key = (str(eval_obj), step.id, str(x))
                self.cache.set(key, result, tag=tag)
                current_request = result

            if len(result) != func.n_metrics:
                raise CADETProcessError(
                    f"Got results with length {len(result)}. "
                    f"Expected length {func.n_metrics} from {str(func)}"
                )

            results = np.hstack((results, result))

        return results

    @property
    def evaluators(self):
        """list: Evaluators in OptimizationProblem."""
        return self._evaluators

    @property
    def evaluators_dict_reference(self):
        """dict: Evaluator objects indexed by original_callable."""
        return {evaluator.evaluator: evaluator for evaluator in self.evaluators}

    @property
    def evaluators_dict(self):
        """dict: Evaluator objects indexed by name."""
        return {evaluator.name: evaluator for evaluator in self.evaluators}

    def add_evaluator(self, evaluator, name=None, cache=False, args=None, kwargs=None):
        """Add Evaluator to OptimizationProblem.

        Evaluators can be referenced by objective and constraint functions to
        perform preprocessing steps.

        Parameters
        ----------
        evaluator : callable
            Evaluation function.
        name : str, optional
            Name of the evaluator.
        cache : bool, optional
            If True, results of the evaluator are cached. The default is False.
        args : tuple, optional
            Additional arguments for evaluation function.
        kwargs : dict, optional
            Additional keyword arguments for evaluation function.

        Raises
        ------
        TypeError
            If evaluator is not callable.
        CADETProcessError
            If evaluator with same name already exists.

        """
        if not callable(evaluator):
            raise TypeError("Expected callable evaluator.")

        if name is None:
            if inspect.isfunction(evaluator) or inspect.ismethod(evaluator):
                name = evaluator.__name__
            else:
                name = str(evaluator)

        if name in self.evaluators_dict:
            raise CADETProcessError("Evaluator with same name already exists.")

        evaluator = Evaluator(
            evaluator,
            name,
            args=args,
            kwargs=kwargs,
        )
        self._evaluators.append(evaluator)

        if cache:
            self.cached_evaluators.append(evaluator)

    @property
    def objectives(self):
        """list: Objective functions."""
        return self._objectives

    @property
    def objective_names(self):
        """list: Objective function names."""
        return [obj.name for obj in self.objectives]

    @property
    def objective_labels(self):
        """list: Objective function metric labels."""
        labels = []
        for obj in self.objectives:
            labels += obj.labels

        return labels

    @property
    def n_objectives(self):
        """int: Number of objectives."""
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
            name=None,
            n_objectives=1,
            bad_metrics=None,
            evaluation_objects=-1,
            labels=None,
            requires=None,
            *args, **kwargs):
        """Add objective function to optimization problem.

        Parameters
        ----------
        objective : callable or MetricBase
            Objective function.
        name : str, optional
            Name of the objective.
        n_objectives : int, optional
            Number of metrics returned by objective function.
            The default is 1.
        bad_metrics : flot or list of floats, optional
            Value which is returned when evaluation fails.
        evaluation_objects : {EvaluationObject, None, -1, list}
            EvaluationObjects which are evaluated by objective.
            If None, no EvaluationObject is used.
            If -1, all EvaluationObjects are used.
        labels : str, optional
            Names of the individual metrics.
        requires : {None, Evaluator, list}
            Evaluators used for preprocessing.
            If None, no preprocessing is required.
        args : tuple, optional
            Additional arguments for objective function.
        kwargs : dict, optional
            Additional keyword arguments for objective function.

        Warnings
        --------
        If objective with same name already exists.

        Raises
        ------
        TypeError
            If objective is not callable.
        CADETProcessError
            If EvaluationObject is not found.
            If Evaluator is not found.

        """
        if not callable(objective):
            raise TypeError("Expected callable objective.")

        if name is None:
            if inspect.isfunction(objective) or inspect.ismethod(objective):
                name = objective.__name__
            else:
                name = str(objective)

        if name in self.objective_names:
            warnings.warn("Objective with same name already exists.")

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
            evaluators = [self.evaluators_dict_reference[req] for req in requires]
        except KeyError as e:
            raise CADETProcessError(f"Unknown Evaluator: {str(e)}")

        objective = Objective(
            objective,
            name,
            type='minimize',
            n_objectives=n_objectives,
            bad_metrics=bad_metrics,
            evaluation_objects=evaluation_objects,
            evaluators=evaluators,
            labels=labels,
            args=args,
            kwargs=kwargs
        )
        self._objectives.append(objective)

    @untransforms
    def evaluate_objectives(self, x, force=False):
        """Evaluate objective functions at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        f : list
            Values of the objective functions at point x.

        See Also
        --------
        add_objective
        evaluate_objectives_population
        _call_evaluate_fun
        _evaluate

        """
        self.logger.debug(f'Evaluate objectives at {x}.')

        f = self._evaluate_individual(self.objectives, x, force=force)

        return f

    @untransforms
    @ensures2d
    def evaluate_objectives_population(self, population, force=False, parallelization_backend=None):
        """Evaluate objective functions for each point x in population.

        Parameters
        ----------
        population : list
            Population.
        force : bool, optional
            If True, do not use cached values. The default is False.
        parallelization_backend : RunnerBase, optional
            Runner to use for the evaluation of the population in
            sequential or parallel mode.

        Returns
        -------
        results : list
            Objective function values.

        See Also
        --------
        add_objective
        evaluate_objectives
        _evaluate_individual
        _evaluate
        """
        results = self._evaluate_population(
            self.evaluate_objectives, population, force, parallelization_backend
        )

        return results

    @untransforms
    def objective_jacobian(self, x, dx=1e-3):
        """Compute jacobian of objective functions using finite differences.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.
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
        """list: Nonlinear constraint functions."""
        return self._nonlinear_constraints

    @property
    def nonlinear_constraint_names(self):
        """list: Nonlinear constraint function names."""
        return [nonlincon.name for nonlincon in self.nonlinear_constraints]

    @property
    def nonlinear_constraint_labels(self):
        """list: Nonlinear constraint function metric labels."""
        labels = []
        for nonlincon in self.nonlinear_constraints:
            labels += nonlincon.labels

        return labels

    @property
    def nonlinear_constraints_bounds(self):
        """list: Bounds of nonlinear constraint functions."""
        bounds = []
        for nonlincon in self.nonlinear_constraints:
            bounds += nonlincon.bounds

        return bounds

    @property
    def n_nonlinear_constraints(self):
        """int: Number of nonlinear_constraints."""
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
            name=None,
            n_nonlinear_constraints=1,
            bad_metrics=None,
            evaluation_objects=-1,
            bounds=0,
            labels=None,
            requires=None,
            *args, **kwargs):
        """Add nonliner constraint function to optimization problem.

        Parameters
        ----------
        nonlincon : callable
            Nonlinear constraint function.
        name : str, optional
            Name of the nonlinear constraint.
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
        labels : str, optional
            Names of the individual metrics.
        requires : {None, Evaluator, list}, optional
            Evaluators used for preprocessing.
            If None, no preprocessing is required.
            The default is None.
        args : tuple, optional
            Additional arguments for nonlinear constraint function.
        kwargs : dict, optional
            Additional keyword arguments for nonlinear constraint function.

        Warnings
        --------
        If nonlinear constraint with same name already exists.

        Raises
        ------
        TypeError
            If nonlinear constraint function is not callable.
        CADETProcessError
            If EvaluationObject is not found.
            If Evaluator is not found.

        """
        if not callable(nonlincon):
            raise TypeError("Expected callable constraint function.")

        if name is None:
            if inspect.isfunction(nonlincon) or inspect.ismethod(nonlincon):
                name = nonlincon.__name__
            else:
                name = str(nonlincon)

        if name in self.nonlinear_constraint_names:
            warnings.warn("Nonlinear constraint with same name already exists.")

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
            evaluators = [self.evaluators_dict_reference[req]for req in requires]
        except KeyError as e:
            raise CADETProcessError(f"Unknown Evaluator: {str(e)}")

        nonlincon = NonlinearConstraint(
            nonlincon,
            name,
            bounds=bounds,
            n_nonlinear_constraints=n_nonlinear_constraints,
            bad_metrics=bad_metrics,
            evaluation_objects=evaluation_objects,
            evaluators=evaluators,
            labels=labels,
            args=args,
            kwargs=kwargs
        )
        self._nonlinear_constraints.append(nonlincon)

    @untransforms
    def evaluate_nonlinear_constraints(self, x, force=False):
        """Evaluate nonlinear constraint functions at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        g : list
            Nonlinear constraint function values.

        See Also
        --------
        add_nonlinear_constraint
        evaluate_nonlinear_constraints_violation
        evaluate_nonlinear_constraints_population
        _call_evaluate_fun
        _evaluate

        """
        self.logger.debug(f'Evaluate nonlinear constraints at {x}.')

        g = self._evaluate_individual(self.nonlinear_constraints, x, force=False)

        return g

    @untransforms
    @ensures2d
    def evaluate_nonlinear_constraints_population(self, population, force=False, parallelization_backend=None):
        """
        Evaluate nonlinear constraint for each point x in population.

        Parameters
        ----------
        population : list
            Population.
        force : bool, optional
            If True, do not use cached values. The default is False.
        parallelization_backend : RunnerBase, optional
            Runner to use for the evaluation of the population in
            sequential or parallel mode.

        Returns
        -------
        results : list
            Nonlinear constraints.

        See Also
        --------
        add_nonlinear_constraint
        evaluate_nonlinear_constraints
        _evaluate_individual
        _evaluate
        """
        results = self._evaluate_population(
            self.evaluate_nonlinear_constraints, population, force, parallelization_backend
        )

        return results

    @untransforms
    def evaluate_nonlinear_constraints_violation(self, x, force=False):
        """Evaluate nonlinear constraints violation at point x.

        After evaluating the nonlinear constraint functions, the corresponding
        bounds are subtracted from the results.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        cv : np.ndarray
            Nonlinear constraints violation.

        See Also
        --------
        add_nonlinear_constraint
        evaluate_nonlinear_constraints
        evaluate_nonlinear_constraints_population
        evaluate_nonlinear_constraints_violation_population
        _call_evaluate_fun
        _evaluate

        """
        self.logger.debug(f'Evaluate nonlinear constraints violation at {x}.')

        g = self._evaluate_individual(self.nonlinear_constraints, x, force=False)
        cv = np.array(g) - np.array(self.nonlinear_constraints_bounds)

        return cv

    @untransforms
    @ensures2d
    def evaluate_nonlinear_constraints_violation_population(
            self, population, force=False, parallelization_backend=None):
        """
        Evaluate nonlinear constraints violation for each point x in population.

        After evaluating the nonlinear constraint functions, the corresponding
        bounds are subtracted from the results.

        Parameters
        ----------
        population : list
            Population.
        force : bool, optional
            If True, do not use cached values. The default is False.
        parallelization_backend : RunnerBase, optional
            Runner to use for the evaluation of the population in
            sequential or parallel mode.

        Returns
        -------
        results : list
            Nonlinear constraints violation.

        See Also
        --------
        add_nonlinear_constraint
        evaluate_nonlinear_constraints_violation
        evaluate_nonlinear_constraints
        evaluate_nonlinear_constraints_population
        _evaluate_individual
        _evaluate

        """
        results = self._evaluate_population(
            self.evaluate_nonlinear_constraints_violation, population, force, parallelization_backend
        )

        return results

    @untransforms
    def check_nonlinear_constraints(self, x):
        """Check if all nonlinear constraints are met.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.

        Returns
        -------
        flag : bool
            True if all nonlinear constraints violation are smaller or equal to zero,
            False otherwise.

        """
        cv = np.array(self.evaluate_nonlinear_constraints_violation(x))

        if np.any(cv > 0):
            return False
        return True

    @untransforms
    def nonlinear_constraint_jacobian(self, x, dx=1e-3):
        """Compute jacobian of the nonlinear constraints at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.
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
    def callback_names(self):
        """list: Callback function names."""
        return [obj.name for obj in self.callbacks]

    @property
    def n_callbacks(self):
        """int: Number of callback functions."""
        return len(self.callbacks)

    def add_callback(
            self,
            callback,
            name=None,
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
        name : str, optional
            Name of the callback.
        evaluation_objects : {EvaluationObject, None, -1, list}
            EvaluationObjects which are evaluated by objective.
            If None, no EvaluationObject is used.
            If -1, all EvaluationObjects are used.
        requires : {None, Evaluator, list}, optional
            Evaluators used for preprocessing.
            If None, no preprocessing is required.
            The default is None.
        frequency : int, optional
            Number of generations after which callback is evaluated.
            The default is 1.
        callbacks_dir : Path, optional
            Dicretory to store results. If None, folder in working directory is
            created.
        args : tuple, optional
            Additional arguments for callback function.
        kwargs : dict, optional
            Additional keyword arguments for callback function.

        Warnings
        --------
        If callback with same name already exists.

        Raises
        ------
        TypeError
            If callback is not callable.
        CADETProcessError
            If EvaluationObject is not found.
            If Evaluator is not found.

        """
        if not callable(callback):
            raise TypeError("Expected callable callback.")

        if name is None:
            if inspect.isfunction(callback) or inspect.ismethod(callback):
                name = callback.__name__
            else:
                name = str(callback)

        if name in self.callback_names:
            warnings.warn("Callback with same name already exists.")

            raise CADETProcessError("Callback with same name already exists.")

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
            evaluators = [self.evaluators_dict_reference[req]for req in requires]
        except KeyError as e:
            raise CADETProcessError(f"Unknown Evaluator: {str(e)}")

        callback = Callback(
            callback,
            name,
            evaluation_objects=evaluation_objects,
            evaluators=evaluators,
            frequency=frequency,
            callbacks_dir=callbacks_dir,
            keep_progress=keep_progress,
            args=args,
            kwargs=kwargs
        )
        self._callbacks.append(callback)

    def evaluate_callbacks(self, ind, current_iteration=1, force=False):
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
        evaluate_callbacks_population
        _evaluate

        """
        self.logger.debug(f'evaluate callbacks at {ind.x}')

        for callback in self.callbacks:
            if not (
                    current_iteration == 'final'
                    or
                    current_iteration % callback.frequency == 0):
                continue

            callback._ind = ind
            callback._current_iteration = current_iteration

            try:
                self._evaluate(ind.x, callback, force)
            except CADETProcessError:
                self.logger.warning(
                    f'Evaluation of {callback} failed at {ind.x}.'
                )

    def evaluate_callbacks_population(
            self, population, current_iteration, force=False, parallelization_backend=None):
        """Evaluate callbacks for each individual ind in population.

        Parameters
        ----------
        population : list
            Population.
        current_iteration : int
            Current iteration step.
        force : bool, optional
            If True, do not use cached values. The default is False.
        parallelization_backend : RunnerBase, optional
            Runner to use for the evaluation of the population in
            sequential or parallel mode.

        Returns
        -------
        results : list
            Nonlinear constraint function values.

        See Also
        --------
        add_callback
        evaluate_callbacks
        """
        if parallelization_backend is None:
            parallelization_backend = SequentialBackend()

        if not self.cache.use_diskcache and parallelization_backend.n_cores != 1:
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

        parallelization_backend.evaluate(eval_fun, population)

    @property
    def meta_scores(self):
        """list: Meta scores for multi criteria selection."""
        return self._meta_scores

    @property
    def meta_score_names(self):
        """list: Meta score function names."""
        return [meta_score.name for meta_score in self.meta_scores]

    @property
    def meta_score_labels(self):
        """int: Meta score function metric labels."""
        labels = []
        for meta_score in self.meta_scores:
            labels += meta_score.labels

        return labels

    @property
    def n_meta_scores(self):
        """int: Number of meta score functions."""
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
            name=None,
            n_meta_scores=1,
            evaluation_objects=-1,
            requires=None):
        """Add Meta score to the OptimizationProblem.

        Parameters
        ----------
        meta_score : callable
            Objective function.
        name : str, optional
            Name of the meta score.
        n_meta_scores : int, optional
            Number of meta scores returned by callable.
            The default is 1.
        evaluation_objects : {EvaluationObject, None, -1, list}
            EvaluationObjects which are evaluated by objective.
            If None, no EvaluationObject is used.
            If -1, all EvaluationObjects are used.
        requires : {None, Evaluator, list}
            Evaluators used for preprocessing.
            If None, no preprocessing is required.

        Warnings
        --------
        If meta score with same name already exists.

        Raises
        ------
        TypeError
            If meta_score is not callable.
        CADETProcessError
            If EvaluationObject is not found.
            If Evaluator is not found.

        """
        if not callable(meta_score):
            raise TypeError("Expected callable meta score.")

        if name is None:
            if inspect.isfunction(meta_score) or inspect.ismethod(meta_score):
                name = meta_score.__name__
            else:
                name = str(meta_score)

        if name in self.meta_score_names:
            warnings.warn("Meta score with same name already exists.")

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
            evaluators = [self.evaluators_dict_reference[req]for req in requires]
        except KeyError as e:
            raise CADETProcessError(f"Unknown Evaluator: {str(e)}")

        meta_score = MetaScore(
            meta_score,
            name,
            n_meta_scores=n_meta_scores,
            evaluation_objects=evaluation_objects,
            evaluators=evaluators,
        )
        self._meta_scores.append(meta_score)

    @untransforms
    def evaluate_meta_scores(self, x, force=False):
        """Evaluate meta functions at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        m : list
            Meta scores.

        See Also
        --------
        add_meta_score
        evaluate_nonlinear_constraints_population
        _call_evaluate_fun
        _evaluate
        """
        self.logger.debug(f'Evaluate meta functions at {x}.')

        m = self._evaluate_individual(self.meta_scores, x, force=force)

        return m

    @untransforms
    @ensures2d
    def evaluate_meta_scores_population(self, population, force=False, parallelization_backend=None):
        """Evaluate meta score functions for each point x in population.

        Parameters
        ----------
        population : list
            Population.
        force : bool, optional
            If True, do not use cached values. The default is False.
        parallelization_backend : RunnerBase, optional
            Runner to use for the evaluation of the population in
            sequential or parallel mode.
        Returns
        -------
        results : list
            Meta scores.

        See Also
        --------
        add_meta_score
        evaluate_meta_scores
        _evaluate_individual
        _evaluate
        """
        results = self._evaluate_population(
            self.evaluate_meta_scores, population, force, parallelization_backend
        )

        return results

    @property
    def multi_criteria_decision_functions(self):
        """list: Multi criteria decision functions."""
        return self._multi_criteria_decision_functions

    @property
    def multi_criteria_decision_function_names(self):
        """list: Multi criteria decision function names."""
        return [mcdf.name for mcdf in self.multi_criteria_decision_functions]

    @property
    def n_multi_criteria_decision_functions(self):
        """int: number of multi criteria decision functions."""
        return len(self.multi_criteria_decision_functions)

    def add_multi_criteria_decision_function(self, decision_function, name=None):
        """Add multi criteria decision function to OptimizationProblem.

        Parameters
        ----------
        decision_function : callable
            Multi criteria decision function.
        name : str, optional
            Name of the multi criteria decision function.

        Warnings
        --------
        If multi criteria decision with same name already exists.

        Raises
        ------
        TypeError
            If decision_function is not callable.

        See Also
        --------
        add_multi_criteria_decision_function
        evaluate_multi_criteria_decision_functions
        """
        if not callable(decision_function):
            raise TypeError("Expected callable decision function.")

        if name is None:
            if inspect.isfunction(decision_function) or inspect.ismethod(decision_function):
                name = decision_function.__name__
            else:
                name = str(decision_function)

        if name in self.multi_criteria_decision_function_names:
            warnings.warn(
                "Multi criteria decision function with same name already exists."
            )

        meta_score = MultiCriteriaDecisionFunction(decision_function, name)
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
        add_multi_criteria_decision_function
        add_multi_criteria_decision_function
        """
        self.logger.debug('Evaluate multi criteria decision functions.')

        for func in self.multi_criteria_decision_functions:
            pareto_population = func(pareto_population)

        return pareto_population

    @property
    def lower_bounds(self):
        """list: Lower bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.lb for var in self.variables]

    @property
    def lower_bounds_transformed(self):
        """list: Transformed lower bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.transform.lb for var in self.variables]

    @property
    def lower_bounds_independent(self):
        """list: Lower bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.lb for var in self.independent_variables]

    @property
    def lower_bounds_independent_transformed(self):
        """list: Transformed lower bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.transform.lb for var in self.independent_variables]

    @property
    def upper_bounds(self):
        """list: Upper bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.ub for var in self.variables]

    @property
    def upper_bounds_transformed(self):
        """list: Transformed upper bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var._transform.ub for var in self.variables]

    @property
    def upper_bounds_independent(self):
        """list: Upper bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.ub for var in self.independent_variables]

    @property
    def upper_bounds_independent_transformed(self):
        """list: Transformed upper bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var._transform.ub for var in self.independent_variables]

    @untransforms
    @gets_dependent_values
    def check_bounds(self, x):
        """Check if all bound constraints are kept.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.

        Returns
        -------
        flag : Bool
            True, if all values are within the bounds. False otherwise.

        """
        flag = True

        values = np.array(x, ndmin=1)

        if np.any(np.less(values, self.lower_bounds)):
            flag = False
        if np.any(np.greater(values, self.upper_bounds)):
            flag = False

        return flag

    @property
    def linear_constraints(self):
        """list: Linear inequality constraints of OptimizationProblem.

        See Also
        --------
        add_linear_constraint
        remove_linear_constraint
        linear_equality_constraints

        """
        return self._linear_constraints

    @property
    def n_linear_constraints(self):
        """int: Number of linear inequality constraints."""
        return len(self.linear_constraints)

    def add_linear_constraint(self, opt_vars, lhs=1.0, b=0.0):
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
            lhs = np.ones(len(opt_vars))

        if len(lhs) != len(opt_vars):
            raise CADETProcessError(
                'Number of lhs coefficients and variables do not match.'
            )

        lincon = dict()
        lincon['opt_vars'] = opt_vars
        lincon['lhs'] = lhs
        lincon['b'] = float(b)

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
        del self._linear_constraints[index]

    @property
    def A(self):
        """np.ndarray: LHS Matrix of linear inequality constraints.

        See Also
        --------
        b
        add_linear_constraint
        remove_linear_constraint
        A_transformed
        A_independent
        A_independent_transformed

        """
        A = np.zeros((len(self.linear_constraints), len(self.variables)))

        for lincon_index, lincon in enumerate(self.linear_constraints):
            for var_index, var in enumerate(lincon['opt_vars']):
                index = self.variables.index(self.variables_dict[var])
                A[lincon_index, index] = lincon['lhs'][var_index]

        return A

    @property
    def A_transformed(self):
        """np.ndarray: LHS Matrix of linear inequality constraints in transformed space.

        See Also
        --------
        A
        A_independent_transformed
        A_independent

        """
        A_t = self.A.copy()
        for a in A_t:
            for j, v in enumerate(self.variables):
                t = v.transform
                if isinstance(t, NoTransform):
                    continue

                if not t.is_linear:
                    raise CADETProcessError(
                        "Non-linear transform was used in linear constraints."
                    )

                scale = (t.ub_input - t.lb_input) / (t.ub - t.lb)
                # this is fine, because it will squish more when the bounds are
                # larger prior to transformation
                # FUTURE: move to transforms module
                # *= (range old bounds) / (range new bounds)
                a[j] *= scale

        return A_t

    @property
    def A_independent(self):
        """np.ndarray: LHS Matrix of linear inequality constraints for indep variables.

        See Also
        --------
        A
        A_transformed
        A_independent_transformed

        """
        return self.A[:, self.independent_variable_indices]

    @property
    def A_independent_transformed(self):
        """np.ndarray: LHS of lin ineqs for indep variables in transformed space.

        See Also
        --------
        A
        A_transformed
        A_independent

        """
        return self.A_transformed[:, self.independent_variable_indices]

    @property
    def b(self):
        """list: Vector form of linear constraints.

        See Also
        --------
        A
        add_linear_constraint
        remove_linear_constraint
        b_transformed

        """
        b = [lincon['b'] for lincon in self.linear_constraints]

        return np.array(b)

    @property
    def b_transformed(self):
        """list: Vector form of linear constraints in transformed space.

        Transforms b to multiple variables. When the bounds of an optimization
        problem get shifted, the upper bound for a constrained variable gets
        shifted too. This shift follows the slope of the constraint.
        In addition the new upper bound needs to be scaled by the ratio between
        new bounds to old bounds.

        See Also
        --------
        b

        """
        # TODO: weird error when b_t is not float that b is not incremented
        b_t = self.b.copy()
        A_t = self.A.copy()
        for i, a in enumerate(A_t):
            for j, v in enumerate(self.variables):
                t = v.transform
                if isinstance(t, NoTransform):
                    continue

                if not t.is_linear:
                    raise CADETProcessError(
                        "Non-linear transform was used in linear constraints."
                    )
                # I realize: This sounds an awful lot like transforms that sklear offers
                # center and scale

                # FUTURE: move to transforms module
                scale_old = (t.ub_input - t.lb_input)
                scale_new = (t.ub - t.lb)
                scale = scale_old / scale_new

                mid_old = 0.5 * (t.ub_input + t.lb_input)
                mid_new = 0.5 * (t.ub + t.lb)
                v_shift = mid_new - mid_old

                b_t[i] += a[j] * v_shift * scale

        return b_t

    @untransforms
    @gets_dependent_values
    def evaluate_linear_constraints(self, x):
        """Calculate value of linear inequality constraints at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.

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
        values = np.array(x, ndmin=1)

        return self.A.dot(values) - self.b

    @untransforms
    @gets_dependent_values
    def check_linear_constraints(self, x):
        """Check if linear inequality constraints are met at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.

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
        """list: Linear equality constraints of OptimizationProblem.

        See Also
        --------
        add_linear_equality_constraint
        remove_linear_equality_constraint
        linear_constraints

        """
        return self._linear_equality_constraints

    @property
    def n_linear_equality_constraints(self):
        """int: Number of linear equality constraints."""
        return len(self.linear_equality_constraints)

    def add_linear_equality_constraint(self, opt_vars, lhs=1, beq=0, eps=0.0):
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

        lineqcon = dict()
        lineqcon['opt_vars'] = opt_vars
        lineqcon['lhs'] = lhs
        lineqcon['beq'] = beq
        lineqcon['eps_eq'] = float(eps)

        self._linear_equality_constraints.append(lineqcon)

    def remove_linear_equality_constraint(self, index):
        """Remove linear equality constraint.

        Parameters
        ----------
        index : int
            Index of the linear equality constraint to be removed.

        See Also
        --------
        add_linear_equality_constraint
        linear_equality_constraint

        """
        del self._linear_equality_constraints[index]

    @property
    def Aeq(self):
        """np.ndarray: LHS Matrix form of linear equality constraints.

        See Also
        --------
        beq
        add_linear_equality_constraint
        remove_linear_equality_constraint

        """
        Aeq = np.zeros(
            (len(self.linear_equality_constraints), len(self.variables))
        )

        for lineqcon_index, lineqcon in enumerate(self.linear_equality_constraints):
            for var_index, var in enumerate(lineqcon['opt_vars']):
                index = self.variables.index(self.variables_dict[var])
                Aeq[lineqcon_index, index] = lineqcon['lhs'][var_index]

        return Aeq

    @property
    def Aeq_transformed(self):
        """np.ndarray: LHS Matrix of linear equality constraints in transformed space.

        See Also
        --------
        Aeq
        Aeq_independent_transformed
        Aeq_independent
        """
        Aeq_t = self.Aeq.copy()
        for aeq in Aeq_t:
            for j, v in enumerate(self.variables):
                t = v.transform
                if isinstance(t, NoTransform):
                    continue

                if not t.is_linear:
                    raise CADETProcessError(
                        "Non-linear transform was used in linear constraints."
                    )

                scale = (t.ub_input - t.lb_input) / (t.ub - t.lb)
                # this is fine, because it will squish more when the bounds are
                # larger prior to transformation
                # FUTURE: move to transforms module
                # *= (range old bounds) / (range new bounds)
                aeq[j] *= scale

        return Aeq_t

    @property
    def Aeq_independent_transformed(self):
        """np.ndarray: LHS of lin ineqs for indep variables in transformed space.

        See Also
        --------
        Aeq
        Aeq_transformed
        Aeq_independent

        """
        return self.Aeq_transformed[:, self.independent_variable_indices]

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
        beq = [lineqcon['beq'] for lineqcon in self.linear_equality_constraints]

        return beq

    @property
    def beq_transformed(self):
        """list: Vector form of linear constraints in transformed space.

        Transforms b to multiple variables. When the bounds of an optimization
        problem get shifted, the upper bound for a constrained variable gets
        shifted too. This shift follows the slope of the constraint.
        In addition the new upper bound needs to be scaled by the ratio between
        new bounds to old bounds.

        See Also
        --------
        b

        """
        # TODO: weird error when b_t is not float that b is not incremented
        beq_t = self.beq.copy()
        Aeq_t = self.Aeq.copy()
        for i, aeq in enumerate(Aeq_t):
            for j, v in enumerate(self.variables):
                t = v.transform
                if isinstance(t, NoTransform):
                    continue

                if not t.is_linear:
                    raise CADETProcessError(
                        "Non-linear transform was used in linear constraints."
                    )
                # I realize: This sounds an awful lot like transforms that sklear offers
                # center and scale

                # FUTURE: move to transforms module
                scale_old = (t.ub_input - t.lb_input)
                scale_new = (t.ub - t.lb)
                scale = scale_old / scale_new

                mid_old = 0.5 * (t.ub_input + t.lb_input)
                mid_new = 0.5 * (t.ub + t.lb)
                v_shift = mid_new - mid_old

                beq_t[i] += aeq[j] * v_shift * scale

        return beq_t

    @property
    def eps_eq(self):
        """np.array: Relaxation tolerance for linear equality constraints.

        See Also
        --------
        add_linear_inequality_constraint
        """
        return np.array([
            lineqcon['eps_eq'] for lineqcon in self.linear_equality_constraints
        ])

    @untransforms
    @gets_dependent_values
    def evaluate_linear_equality_constraints(self, x):
        """Calculate value of linear equality constraints at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.

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
        values = np.array(x, ndmin=1)

        return self.Aeq.dot(values) - self.beq

    @untransforms
    @gets_dependent_values
    def check_linear_equality_constraints(self, x):
        """Check if linear equality constraints are met at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.

        Returns
        -------
        flag : bool
            True if linear equality constraints are met. False otherwise.

        """
        flag = True

        lhs = self.evaluate_linear_equality_constraints(x)
        if np.any(np.abs(lhs) >= self.eps_eq):
            flag = False

        return flag

    def transform(self, x):
        """Transform the optimization variables from untransformed parameter space.

        Parameters
        ----------
        x : list
            Value of the optimization variables in untransformed space.

        Returns
        -------
        list
            Optimization variables in transformed parameter space.
        """
        x = np.array(x)
        x_2d = np.array(x, ndmin=2)
        transform = np.zeros(x_2d.shape)

        for i, ind in enumerate(x_2d):
            transform[i, :] = [
                var.transform_fun(value)
                for value, var in zip(ind, self.independent_variables)
            ]

        return transform.reshape(x.shape).tolist()

    def untransform(self, x_transformed):
        """Untransform the optimization variables from transformed parameter space.

        Parameters
        ----------
        x_transformed : list
            Optimization variables in transformed parameter space.

        Returns
        -------
        list
            Optimization variables in untransformed parameter space.
        """
        x_transformed = np.array(x_transformed)
        x_transformed_2d = np.array(x_transformed, ndmin=2)
        untransform = np.zeros(x_transformed_2d.shape)

        for i, ind in enumerate(x_transformed_2d):
            untransform[i, :] = [
                var.untransform_fun(value)
                for value, var in zip(ind, self.independent_variables)
            ]

        return untransform.reshape(x_transformed.shape).tolist()

    @property
    def cached_steps(self):
        """list: Cached evaluation steps."""
        return \
            self.cached_evaluators + \
            self.objectives + \
            self.nonlinear_constraints

    @property
    def cache_directory(self):
        """pathlib.Path: Path for results cache database."""
        if self._cache_directory is None:
            _cache_directory = settings.working_directory / f'diskcache_{self.name}'
        else:
            _cache_directory = Path(self._cache_directory).absolute()

        if self.use_diskcache:
            _cache_directory.mkdir(exist_ok=True, parents=True)

        return _cache_directory

    @cache_directory.setter
    def cache_directory(self, cache_directory):
        self._cache_directory = cache_directory

    def setup_cache(self):
        """Setup cache to store (intermediate) results."""
        self.cache = ResultsCache(self.use_diskcache, self.cache_directory)

    def delete_cache(self, reinit=False):
        """Delete cache with (intermediate) results."""
        try:
            self.cache.delete_database()
        except AttributeError:
            pass
        if reinit:
            self.setup_cache()

    def prune_cache(self):
        """Prune cache with (intermediate) results."""
        self.cache.prune()

    def create_initial_values(
            self, n_samples=1, method='random', seed=None, burn_in=100000):
        """Create initial value within parameter space.

        Uses hopsy (Highly Optimized toolbox for Polytope Sampling) to retrieve
        uniformly distributed samples from the parameter space.

        Parameters
        ----------
        n_samples : int
            Number of initial values to be drawn
        method : str, optional
            chebyshev: Return center of the minimal-radius ball enclosing the entire set .
            random: Any random valid point in the parameter space.
        seed : int, optional
            Seed to initialize random numbers. Only used if method == 'random'
        burn_in: int, optional
            Number of samples that are created to ensure uniform sampling.
            The actual initial values are then drawn from this set.
            The default is 100000.

        Raises
        ------
        CADETProcessError
            If method is not known.

        Returns
        -------
        values : list
            Initial values for starting the optimization.

        """
        burn_in = int(burn_in)

        class CustomModel():
            def __init__(self, log_space_indices: list):
                self.log_space_indices = log_space_indices

            def compute_negative_log_likelihood(self, x):
                return np.sum(np.log(x[self.log_space_indices]))

        log_space_indices = []
        for i, var in enumerate(self.variables):
            if (
                    isinstance(var._transform, NormLogTransform)
                    or
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
                if not self.check_linear_equality_constraints(ind, get_dependent_values=True):
                    continue
                values.append(ind)

        return np.array(values)

    @property
    def parameters(self):
        parameters = Dict()

        parameters.variables = {
            opt.name: opt.parameters for opt in self.variables
        }
        parameters.linear_constraints = self.linear_constraints
        parameters.linear_equality_constraints = self.linear_equality_constraints

        return parameters

    def check_linear_constraints_transforms(self):
        """Check that variables used in linear constraints only use linear transforms.

        Returns
        -------
        bool
            True if all variables used in linear constraints have linear transforms,
            False otherwise.

        Warns
        -----
        UserWarning
            If variables used in linear constraints have non-linear transforms.
            The warning message provides details on the problematic variables.
        """
        flag = True
        for constr in self.linear_constraints + self.linear_equality_constraints:
            opt_vars = [self.variables_dict[key] for key in constr["opt_vars"]]
            for var in opt_vars:
                if not var.transform.is_linear:
                    flag = False
                    warnings.warn(
                        f"'{var.name}' uses non-linear transform and is used in "
                        f"the linear constraint: {constr}."
                        "Consider using linear transforms for these variables "
                        "or specify the constraints as non-linear constraints."
                    )

        return flag

    def check_linear_constraints_dependency(self):
        """Check that variables used in linear constraints are independent.

        Returns
        -------
        bool
            True if all variables used in linear constraints are independent,
            False otherwise.

        Warns
        -----
        UserWarning
            If variables used in linear constraints are not independent.
            The warning message provides details on the problematic variables.
        """
        flag = True

        for constr in self.linear_constraints + self.linear_equality_constraints:
            opt_vars = [self.variables_dict[key] for key in constr["opt_vars"]]
            for var in opt_vars:
                if not var.is_independent:
                    flag = False
                    warnings.warn(
                        f"'{var.name}' is not an indendent variable and is used in "
                        f"the linear constraint: {constr}."
                        "This is currently not supported."
                    )

        return flag

    def check_config(self, ignore_linear_constraints=False):
        """Check if the OptimizationProblem is configured correctly.

        Parameters
        ----------
        ignore_linear_constraints : bool, optional
            If True, linear constraint checks will be skipped. The default is False.

        Returns
        -------
        bool
            True if the OptimizationProblem is configured correctly, False otherwise.

        """
        flag = True
        if self.n_variables == 0:
            flag = False

        if not self.check_duplicate_variables():
            flag = False

        if self.n_objectives == 0:
            flag = False
        if self.n_linear_constraints + self.n_linear_equality_constraints > 0 \
                and not ignore_linear_constraints:
            if not self.check_linear_constraints_transforms():
                flag = False

            if not self.check_linear_constraints_dependency():
                flag = False

        return flag

    def __str__(self):
        return self.name


class OptimizationVariable:
    """Class for setting the values for the optimization variables.

    Defines the attributes for optimization variables for creating an
    OptimizationVariable. Tries to get the attr of the evaluation_object.
    Raises a CADETProcessError if the attribute to be set is not valid.

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
        Upper bound of the variable.
    transform : TransformBase
        Transformation function for parameter normalization.
    indices : int, or slice
        Indices for variables that modify an entry of a parameter array.
        If None, variable is assumed to be index independent.
    precision : int, optional
        Number of significant figures to which variable can be rounded.
        If None, variable is not rounded. The default is None.

    Raises
    ------
    CADETProcessError
        If the attribute is not valid.
    ValueError
        If the lower bound is larger than or equal to the upper bound.
    """

    def __init__(
        self, name, evaluation_objects=None, parameter_path=None,
        lb=-math.inf, ub=math.inf, transform=None, indices=None, precision=None,
    ):
        self.name = name
        self._value = None

        if evaluation_objects is not None:
            if not isinstance(evaluation_objects, list):
                evaluation_objects = [evaluation_objects]
            self.evaluation_objects = evaluation_objects
            self.parameter_path = parameter_path
            self.indices = indices
        else:
            self.evaluation_objects = None
            self.parameter_path = None
            self.indices = None

        if lb >= ub:
            raise ValueError("Lower bound cannot be larger or equal to upper bound.")
        self.lb = lb
        self.ub = ub

        if transform is None:
            transform = NoTransform(lb, ub)
        else:
            if np.isinf(lb) or np.isinf(ub):
                raise CADETProcessError("Transform requires bound constraints.")
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
        """str: Path of the evaluation_object parameter in dot notation."""
        return self._parameter_path

    @parameter_path.setter
    def parameter_path(self, parameter_path):
        if parameter_path is not None:
            for eval_obj in self.evaluation_objects:
                parameters = eval_obj.parameters.to_dict()  # Workaround addict issue #136
                if not check_nested(eval_obj.parameters, parameter_path):
                    raise CADETProcessError('Not a valid Optimization variable')
        self._parameter_path = parameter_path

    @property
    def parameter_sequence(self):
        """tuple: Tuple of parameters path elements."""
        return tuple(self.parameter_path.split('.'))

    @property
    def performer(self):
        """str: The name of the performer of the variable."""
        if len(self.parameter_sequence) == 1:
            return self.parameter_sequence[0]
        else:
            return ".".join(self.parameter_sequence[:-1])

    def _performer_obj(self, evaluation_object):
        if len(self.parameter_sequence) == 1:
            return evaluation_object

        return get_nested_attribute(evaluation_object, self.performer)

    def _parameter_descriptor(self, evaluation_object):
        performer_obj = self._performer_obj(evaluation_object)
        performer_class = type(performer_obj)
        try:
            descriptor = getattr(performer_class, self.parameter_sequence[-1])
        except AttributeError:
            return None

        if not isinstance(descriptor, ParameterBase):
            return None

        return descriptor

    def _parameter_type(self, evaluation_object):
        """type: Type of the parameter."""
        parameter_descriptor = self._parameter_descriptor(evaluation_object)
        if isinstance(parameter_descriptor, Typed):
            return parameter_descriptor.ty

        current_value = self._current_value(evaluation_object)
        if current_value is None:
            raise CADETProcessError(
                "Parameter is not initialized. "
                "Cannot determine parameter type."
            )

        return type(current_value)

    def _get_parameter_shape(self, evaluation_object):
        parameter_descriptor = self._parameter_descriptor(evaluation_object)

        if isinstance(parameter_descriptor, (Float, Integer, Bool)):
            return ()

        if isinstance(parameter_descriptor, Sized):
            performer_obj = self._performer_obj(evaluation_object)
            shape = parameter_descriptor.get_expected_size(performer_obj)

            if not isinstance(shape, tuple):
                shape = (shape, )

            return shape

        current_value = self._current_value(evaluation_object)
        if current_value is None:
            raise CADETProcessError(
                "Parameter is not initialized. "
                "Cannot determine parameter shape."
            )

        shape = get_inhomogeneous_shape(current_value)

        return shape

    def _current_value(self, evaluation_object):
        parameter_descriptor = self._parameter_descriptor(evaluation_object)

        if parameter_descriptor is not None:
            performer_obj = self._performer_obj(evaluation_object)
            return getattr(performer_obj, self.parameter_sequence[-1])
        else:
            return copy.copy(get_nested_value(
                evaluation_object.parameters, self.parameter_path
            ))

    def _is_sized(self, evaluation_object):
        """bool: True if descriptor is instance of Sized. False otherwise."""
        parameter_descriptor = self._parameter_descriptor(evaluation_object)

        if isinstance(parameter_descriptor, (Float, Integer, Bool)):
            return False

        if isinstance(parameter_descriptor, Sized):
            return True

        current_value = self._current_value(evaluation_object)
        if current_value is None:
            raise CADETProcessError(
                "Parameter is not initialized. "
                "Cannot determine dimensions required for setting index."
            )
        return np.array(current_value, dtype=object).size > 1

    def _is_polynomial(self, evaluation_object):
        """bool: True if descriptor is instance of NdPolynomial. False otherwise."""
        polynomial_parameters = evaluation_object.polynomial_parameters.to_dict()   # Workaround addict issue #136
        return check_nested(
            polynomial_parameters, self.parameter_path
        )

    @property
    def indices(self):
        if self._indices is None:
            return

        indices = []
        for eval_ind, eval_obj in zip(self._indices, self.evaluation_objects):
            if self._is_sized(eval_obj):
                parameter_shape = self._get_parameter_shape(eval_obj)
                if isinstance(parameter_shape, tuple):
                    eval_ind = generate_indices(parameter_shape, eval_ind)
                else:
                    if not isinstance(eval_ind, list):
                        eval_ind = [eval_ind]

                    eval_ind = [ind for ind in eval_ind]

            indices.append(eval_ind)

        return indices

    @indices.setter
    def indices(self, indices):
        if self.evaluation_objects is None and indices is not None:
            raise ValueError("Cannot specify indices without evaluation object.")

        if self.evaluation_objects is None and indices is None:
            self._indices = indices
            return

        # Make sure indices are list of len self.evaluation_objects
        if not isinstance(indices, list):
            indices = [indices]

        # Make sure indices are 2D; 1D for number of indices to set; 1D for eval objs.
        if len(indices) == 1:
            indices = len(self.evaluation_objects) * indices
        elif len(indices) != len(self.evaluation_objects):
            raise ValueError(
                f"Expected {len(self.evaluation_objects)}, got {len(indices)}"
            )

        for eval_ind, eval_obj in zip(indices, self.evaluation_objects):
            is_sized = self._is_sized(eval_obj)
            if eval_ind is not None and not is_sized:
                raise IndexError("Variables for scalar parameters cannot have indices.")

        self._indices = indices

        # Since indices are constructed on `get`, call the property here:
        try:
            _ = self.indices
        except (ValueError, TypeError) as e:
            raise e

    @property
    def is_index_specific(self):
        """bool: True if variable modifies entry of a parameter array, False otherwise."""
        if self.indices is not None:
            return True
        else:
            return False

    @property
    def full_indices(self):
        """list: Full indices."""
        full_indices = []
        for eval_ind, eval_obj in zip(self.indices, self.evaluation_objects):
            parameter_shape = self._get_parameter_shape(eval_obj)

            if isinstance(parameter_shape, tuple):
                indices = generate_indices(parameter_shape, eval_ind)
                indices = unravel(parameter_shape, indices)
                full_indices.append(indices)
            else:
                for ind in eval_ind:
                    subshape = parameter_shape[ind[0]]
                    indices = generate_indices(subshape, ind[1:])
                    indices = unravel(subshape, indices)
                    indices = [(ind[0], ) + i for i in indices]
                    full_indices.append(indices)

        return full_indices

    @property
    def n_indices(self):
        """int: Number of (full) indices."""
        if self.indices is not None:
            return len(self.full_indices)
        else:
            return 0

    @property
    def transform(self):
        return self._transform

    def transform_fun(self, x):
        return self._transform.transform(x)

    def untransform_fun(self, x):
        return self._transform.untransform(x)

    def add_dependency(self, dependencies, transform):
        """Add dependency of Variable on other Variables.

        Parameters
        ----------
        dependencies : list
            List of OptimizationVariables to be added as dependencies.
        transform: callable
            Transform function describing dependency on independent variables.

        Raises
        ------
        CADETProcessError
            If the variable is already dependent.
            If transform signature does not match independent Variables.
        """
        if not self.is_independent:
            raise CADETProcessError("Variable is already dependent.")

        self._dependencies = dependencies
        self.dependency_transform = transform

    @property
    def dependencies(self):
        """list: Independent variables on which the Variable depends."""
        return self._dependencies

    @property
    def is_independent(self):
        """bool: True if Variable is independent, False otherwise."""
        return len(self.dependencies) == 0

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
        if self.is_independent:
            if self._value is None:
                raise CADETProcessError("Value not set.")

            return self._value
        else:
            dependencies = [dep.value for dep in self.dependencies]
            return self.dependency_transform(*dependencies)

    @value.setter
    def value(self, value):
        if not self.is_independent:
            raise CADETProcessError("Cannot set value for dependent variables")

        self.set_value(value)

        self._value = value

    def set_value(self, value):
        """Set value to evaluation_objects."""
        if not np.isscalar(value):
            raise TypeError("Expected scalar value")

        if value < self.lb:
            raise ValueError("Exceeds lower bound")
        if value > self.ub:
            raise ValueError("Exceeds upper bound")

        if self.evaluation_objects is None:
            return

        for i_eval, eval_obj in enumerate(self.evaluation_objects):
            is_polynomial = self._is_polynomial(eval_obj)

            if (
                    self.indices[i_eval] is None
                    or
                    self._indices[i_eval] is None and is_polynomial):
                new_value = value
            else:
                eval_ind = self.indices[i_eval]
                parameter_shape = self._get_parameter_shape(eval_obj)
                current_value = self._current_value(eval_obj)

                if current_value is None:
                    new_value = np.full(parameter_shape, np.nan)
                else:
                    if isinstance(parameter_shape, tuple):
                        new_value = np.array(current_value, ndmin=1)
                    else:
                        new_value = np.full(get_full_shape(parameter_shape), np.nan)

                if isinstance(eval_ind, int):
                    eval_ind = [(eval_ind, )]

                for ind in eval_ind:
                    expected_shape = new_value[ind].shape
                    is_polynomial = self._is_polynomial(eval_obj)
                    if is_polynomial and len(parameter_shape) > 1 and len(ind) == 1:
                        parameter_descriptor = self._parameter_descriptor(eval_obj)
                        new_slice = parameter_descriptor.fill_values(
                            expected_shape, value
                        )
                    else:
                        new_slice = np.array(value, ndmin=1)

                    if any(isinstance(i, slice) for i in ind):
                        if new_slice.size != np.prod(expected_shape):
                            new_slice = np.broadcast_to(new_slice, expected_shape)
                        else:
                            new_slice = np.reshape(new_slice, expected_shape)

                    if isinstance(parameter_shape, tuple):
                        new_value[ind] = new_slice

                    else:
                        # Inhomogeneous arrays
                        new_value = current_value
                        for i, val in zip(self._indices, new_slice):
                            set_nested_list_value(new_value, i, val)

                parameter_type = self._parameter_type(eval_obj)
                if not isinstance(new_value, parameter_type):
                    new_value = parameter_type(new_value.tolist())

            # Set the value:
            self._set_value(eval_obj, new_value)

    def _set_value(self, evaluation_object, value):
        """Set the value to the evaluation object."""
        parameter_descriptor = self._parameter_descriptor(evaluation_object)
        if parameter_descriptor is not None:
            performer_obj = self._performer_obj(evaluation_object)
            setattr(performer_obj, self.parameter_sequence[-1], value)
        else:
            parameters = generate_nested_dict(self.parameter_sequence, value)
            evaluation_object.parameters = parameters

    @property
    def transformed_bounds(self):
        """list: Transformed bounds of the parameter."""
        return [self.transform_fun(self.lb), self.transform_fun(self.ub)]

    def __repr__(self):
        if self.evaluation_objects is not None:
            string = \
                f'{self.__class__.__name__}' + \
                f'(name={self.name}, ' + \
                f'evaluation_objects=' \
                f'{[str(obj) for obj in self.evaluation_objects]}, ' + \
                f'parameter_path=' \
                f'{self.parameter_path}, lb={self.lb}, ub={self.ub})'
        else:
            string = \
                f'{self.__class__.__name__}' + \
                f'(name={self.name}, lb={self.lb}, ub={self.ub})'
        return string


class Evaluator(Structure):
    """Wrapper class to call evaluator."""

    evaluator = Callable()
    args = Tuple()
    kwargs = Dict()

    def __init__(
            self,
            evaluator,
            name,
            args=None,
            kwargs=None):

        self.evaluator = evaluator
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.id = uuid.uuid4()

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


class Objective(Structure):
    """Wrapper class to evaluate objective functions."""

    objective = Callable()
    name = String()
    # TODO: umbennennen
    type = Switch(valid=['minimize', 'maximize'])
    n_objectives = RangedInteger(lb=1)
    n_metrics = n_objectives
    bad_metrics = SizedNdArray(size='n_metrics', default=np.inf)

    def __init__(
            self,
            objective,
            name,
            type='minimize',
            n_objectives=1,
            bad_metrics=np.inf,
            evaluation_objects=None,
            evaluators=None,
            labels=None,
            args=None,
            kwargs=None):

        self.objective = objective
        self.name = name

        self.type = type
        self.n_objectives = n_objectives

        if np.isscalar(bad_metrics):
            bad_metrics = np.tile(bad_metrics, n_objectives)
        self.bad_metrics = bad_metrics

        self.evaluation_objects = evaluation_objects
        self.evaluators = evaluators

        self.labels = labels

        self.args = args
        self.kwargs = kwargs

        self.id = uuid.uuid4()

    @property
    def labels(self):
        """list: List of metric labels."""
        if self._labels is not None:
            return self._labels

        try:
            labels = self.objective.labels
        except AttributeError:
            labels = [f'{self.name}']
            if self.n_metrics > 1:
                labels = [
                    f'{self.name}_{i}'
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

        return np.array(f, ndmin=1)

    evaluate = __call__

    def __str__(self):
        return self.name


class NonlinearConstraint(Structure):
    """Wrapper class to evaluate nonlinear constraint functions."""

    nonlinear_constraint = Callable()
    name = String()
    n_nonlinear_constraints = RangedInteger(lb=1)
    n_metrics = n_nonlinear_constraints
    bad_metrics = SizedNdArray(size='n_metrics', default=np.inf)

    def __init__(
            self,
            nonlinear_constraint,
            name,
            bounds=0,
            n_nonlinear_constraints=1,
            bad_metrics=np.inf,
            evaluation_objects=None,
            evaluators=None,
            labels=None,
            args=None,
            kwargs=None):

        self.nonlinear_constraint = nonlinear_constraint
        self.name = name

        self.bounds = bounds
        self.n_nonlinear_constraints = n_nonlinear_constraints

        if np.isscalar(bad_metrics):
            bad_metrics = np.tile(bad_metrics, n_nonlinear_constraints)
        self.bad_metrics = bad_metrics

        self.evaluation_objects = evaluation_objects
        self.evaluators = evaluators

        self.labels = labels

        self.args = args
        self.kwargs = kwargs

        self.id = uuid.uuid4()

    @property
    def labels(self):
        """list: List of metric labels."""
        if self._labels is not None:
            return self._labels

        try:
            labels = self.nonlinear_constraint.labels
        except AttributeError:
            labels = [f'{self.name}']
            if self.n_metrics > 1:
                labels = [
                    f'{self.name}_{i}'
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

        return np.array(g, ndmin=1)

    evaluate = __call__

    def __str__(self):
        return self.name


class Callback(Structure):
    """Wrapper class to evaluate callbacks.

    Callable must implement function with the following signature:
        results : obj
            x or final result of evaluation toolchain.
        _current_iteration: int
            Current iteration.
        _individual : Individual, optional
            Information about current step of optimzer.
        evaluation_object : obj, optional
            Current evaluation object.
        callbacks_dir : Path, optional
            Path to store results.
    """

    callback = Callable()
    name = String()
    n_metrics = 0
    frequency = RangedInteger(lb=1)

    def __init__(
            self,
            callback,
            name,
            evaluation_objects=None,
            evaluators=None,
            frequency=10,
            callbacks_dir=None,
            keep_progress=False,
            args=None,
            kwargs=None):

        self.callback = callback
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

        self.id = uuid.uuid4()

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
        if 'current_iteration' in signature:
            kwargs['current_iteration'] = self._current_iteration
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

        self.callback(request, *args, **kwargs)

    evaluate = __call__

    def __str__(self):
        return self.name


class MetaScore(Structure):
    """Wrapper class to evaluate meta scores."""

    meta_score = Callable()
    name = String()
    n_meta_scores = RangedInteger(lb=1)
    n_metrics = n_meta_scores
    bad_metrics = SizedNdArray(size='n_metrics', default=np.inf)

    def __init__(
            self,
            meta_score,
            name,
            n_meta_scores=1,
            bad_metrics=np.inf,
            evaluation_objects=None,
            evaluators=None,
            labels=None):

        self.meta_score = meta_score
        self.name = name

        self.n_meta_scores = n_meta_scores

        if np.isscalar(bad_metrics):
            bad_metrics = np.tile(bad_metrics, n_meta_scores)
        self.bad_metrics = bad_metrics

        self.evaluation_objects = evaluation_objects
        self.evaluators = evaluators

        self.labels = labels

        self.id = uuid.uuid4()

    @property
    def labels(self):
        """list: List of metric labels."""
        if self._labels is not None:
            return self._labels

        try:
            labels = self.meta_score.labels
        except AttributeError:
            labels = [f'{self.name}']
            if self.n_metrics > 1:
                labels = [
                    f'{self.name}_{i}'
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

        return np.array(m, ndmin=1)

    evaluate = __call__

    def __str__(self):
        return self.name


class MultiCriteriaDecisionFunction(Structure):
    """Wrapper class to evaluate multi-criteria decision functions."""

    decision_function = Callable()
    name = String()

    def __init__(self, decision_function, name):

        self.decision_function = decision_function
        self.name = name

        self.id = uuid.uuid4()

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
