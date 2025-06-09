from __future__ import annotations

import copy
import inspect
import math
import os
import random
import shutil
import typing as tp
import uuid
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, NoReturn, Optional

import hopsy
import numpy as np
import numpy.typing as npt
from addict import Dict

from CADETProcess import CADETProcessError, log, settings
from CADETProcess.dataStructure import (
    Bool,
    Callable,
    Float,
    Integer,
    NumpyProxyArray,
    ParameterBase,
    RangedInteger,
    Sized,
    SizedNdArray,
    String,
    Structure,
    Switch,
    Tuple,
    Typed,
    check_nested,
    frozen_attributes,
    generate_nested_dict,
    get_nested_attribute,
    get_nested_value,
    set_nested_list_value,
)
from CADETProcess.dynamicEvents.section import (
    generate_indices,
    get_full_shape,
    get_inhomogeneous_shape,
    unravel,
)
from CADETProcess.hashing import digest_string
from CADETProcess.metric import MetricBase
from CADETProcess.numerics import round_to_significant_digits
from CADETProcess.optimization import Individual, Population, ResultsCache
from CADETProcess.optimization.parallelizationBackend import (
    ParallelizationBackendBase,
    SequentialBackend,
)
from CADETProcess.transform import (
    AutoTransformer,
    NormLinearTransformer,
    NormLogTransformer,
    NullTransformer,
    TransformerBase,
)

__all__ = ["OptimizationProblem", "OptimizationVariable"]


@frozen_attributes
class OptimizationProblem(Structure):
    """
    Class for configuring optimization problems.

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
        name: str,
        use_diskcache: bool = True,
        cache_directory: Optional[str] = None,
        log_level: str = "INFO",
    ) -> None:
        """
        Initialize OptimizationProblem.

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

    def untransforms(func: tp.Callable) -> tp.Callable:
        """Untransform population or individual before calling function."""

        @wraps(func)
        def wrapper_untransforms(
            self: OptimizationProblem,
            x: npt.ArrayLike,
            *args: Any,
            untransform: Optional[bool] = False,
            **kwargs: Any,
        ) -> Any:
            x = np.array(x, ndmin=1)
            if untransform:
                x = self.untransform(x)

            return func(self, x, *args, **kwargs)

        return wrapper_untransforms

    def gets_dependent_values(func: tp.Callable) -> tp.Callable:
        """Get dependent values of individual before calling function."""

        @wraps(func)
        def wrapper_gets_dependent_values(
            self: OptimizationProblem,
            x: npt.ArrayLike,
            *args: Any,
            get_dependent_values: Optional[bool] = False,
            **kwargs: Any,
        ) -> Any:
            if get_dependent_values:
                x = self.get_dependent_values(x)

            return func(self, x, *args, **kwargs)

        return wrapper_gets_dependent_values

    def ensures2d(func: tp.Callable) -> tp.Callable:
        """Ensure X array is an ndarray with ndmin=2."""

        @wraps(func)
        def wrapper_ensures2d(
            self: OptimizationProblem,
            X: npt.ArrayLike,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            # Convert to 2D population
            X = np.array(X)
            X_2d = np.array(X, ndmin=2)

            # Call and ensure results are 2D
            Y = func(self, X_2d, *args, **kwargs)
            Y_2d = Y.reshape((len(X_2d), -1))

            # Reshape back to original length of X
            if X.ndim == 1:
                return Y_2d[0]
            else:
                return Y_2d

        return wrapper_ensures2d

    def ensures_minimization(scores: npt.ArrayLike) -> tp.Callable:
        """Convert maximization problems to minimization problems."""

        def wrap(func: tp.Callable) -> tp.Callable:
            @wraps(func)
            def wrapper_ensures_minimization(
                self: OptimizationProblem,
                *args: Any,
                ensure_minimization: bool = False,
                **kwargs: Any,
            ) -> Any:
                s = func(self, *args, **kwargs)

                if ensure_minimization:
                    s = self.transform_maximization(s, scores)

                return s

            return wrapper_ensures_minimization

        return wrap

    def transform_maximization(
        self: Any, s: list[float], scores: str | list
    ) -> list[float]:
        """Transform maximization problems to minimization problems."""
        factors = []
        if scores == "objectives":
            scores = self.objectives
        elif scores == "meta_scores":
            scores = self.meta_scores
        else:
            raise ValueError(f"Unknown scores: {scores}.")

        for score in scores:
            factor = 1 if score.minimize else -1
            factors += score.n_total_metrics * [factor]

        s = np.multiply(factors, s)

        return s

    @property
    def evaluation_objects(self) -> list:
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
    def evaluation_objects_dict(self) -> dict:
        """dict: Evaluation objects names and objects."""
        return self._evaluation_objects_dict

    def add_evaluation_object(
        self,
        evaluation_object: Any,
        name: Optional[str] = None,
    ) -> None:
        """
        Add evaluation object to the optimization problem.

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
                "Evaluation object already part of optimization problem."
            )

        if name in self.evaluation_objects_dict:
            raise CADETProcessError("Evaluation object with same name already exists.")

        self._evaluation_objects_dict[name] = evaluation_object

    @property
    def variables(self) -> list:
        """list: List of all optimization variables."""
        return self._variables

    @property
    def variable_names(self) -> list:
        """list: Optimization variable names."""
        return [var.name for var in self.variables]

    @property
    def n_variables(self) -> int:
        """int: Number of optimization variables."""
        return len(self.variables)

    @property
    def independent_variables(self) -> list["OptimizationVariable"]:
        """list: Independent OptimizationVaribles."""
        return list(filter(lambda var: var.is_independent, self.variables))

    @property
    def independent_variable_names(self) -> list[str]:
        """list: Independent optimization variable names."""
        return [var.name for var in self.independent_variables]

    @property
    def n_independent_variables(self) -> int:
        """int: Number of independent optimization variables."""
        return len(self.independent_variables)

    @property
    def independent_variable_indices(self) -> list:
        """list: Indices of indpeendent variables."""
        return [
            i
            for i, var in enumerate(self.variable_names)
            if var in self.independent_variable_names
        ]

    @property
    def dependent_variables(self) -> list:
        """list: OptimizationVaribles with dependencies."""
        return list(filter(lambda var: var.is_independent is False, self.variables))

    @property
    def dependent_variable_names(self) -> list:
        """list: Dependent optimization variable names."""
        return [var.name for var in self.dependent_variables]

    @property
    def n_dependent_variables(self) -> int:
        """int: Number of dependent optimization variables."""
        return len(self.dependent_variables)

    @property
    def variables_dict(self) -> dict:
        """dict: All optimization variables indexed by variable name."""
        return {var.name: var for var in self.variables}

    @property
    def variable_values(self) -> np.ndarray:
        """np.ndarray: Values of optimization variables."""
        return np.array([var.value for var in self.variables])

    def add_variable(
        self,
        name: str,
        evaluation_objects: Optional[list | object | int] = -1,
        parameter_path: Optional[str] = None,
        lb: float = -math.inf,
        ub: float = math.inf,
        transform: Optional[str] = None,
        indices: Optional[int | tuple[int]] = None,
        significant_digits: Optional[int] = None,
        pre_processing: Optional[tp.Callable] = None,
    ) -> None:
        """
        Add optimization variable to the OptimizationProblem.

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
        significant_digits : int, optional
            Number of significant figures to which variable can be rounded.
            If None, variable is not rounded. The default is None.
        pre_processing : tp.Callable, optional
            Additional step to process the value before setting it. This function must
            accept a single argument (the value) and return the processed value.

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
            self.evaluation_objects[eval_obj] if isinstance(eval_obj, str) else eval_obj
            for eval_obj in evaluation_objects
        ]

        if parameter_path is None and len(evaluation_objects) > 0:
            parameter_path = name
        if parameter_path is not None and len(evaluation_objects) == 0:
            raise ValueError(
                "Cannot set parameter_path for variable without evaluation object "
            )

        var = OptimizationVariable(
            name,
            evaluation_objects,
            parameter_path,
            lb=lb,
            ub=ub,
            transform=transform,
            indices=indices,
            significant_digits=significant_digits,
            pre_processing=pre_processing,
        )

        self._variables.append(var)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Treat warnings as errors

            try:
                self.check_duplicate_variables()
            except UserWarning as e:
                self._variables.remove(var)
                raise CADETProcessError(e)

        super().__setattr__(name, var)

        return var

    def remove_variable(self, var_name: str) -> None:
        """
        Remove optimization variable from the OptimizationProblem.

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
    def parameter_variables(self) -> dict:
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

    def check_duplicate_variables(self) -> bool:
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
        self,
        dependent_variable: str,
        independent_variables: str | list,
        transform: tp.Callable,
    ) -> None:
        """
        Add dependency between two optimization variables.

        Parameters
        ----------
        dependent_variable : str
            OptimizationVariable whose value will depend on other variables.
        independent_variables : {str, list}
            Independent variable name or list of independent variables names.
        transform : tp.Callable
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
        if not all(indep in self.variables_dict for indep in independent_variables):
            raise CADETProcessError("Cannot find one or more independent variables")

        vars = [self.variables_dict[indep] for indep in independent_variables]
        var.add_dependency(vars, transform)

    @ensures2d
    @untransforms
    def get_dependent_values(self, X_independent: npt.ArrayLike) -> np.ndarray:
        """
        Determine values of dependent optimization variables.

        Parameters
        ----------
        X_independent : array_like
            Value of the optimization variables in untransformed space.

        Raises
        ------
        CADETProcessError
            If length of parameters does not match.

        Returns
        -------
        np.ndarray
            Value of all optimization variables in untransformed space.
        """
        if X_independent.shape[1] != self.n_independent_variables:
            raise CADETProcessError(
                f"Expected {self.n_independent_variables} value(s)."
            )

        variable_values = np.zeros((len(X_independent), self.n_variables))
        independent_variables = self.independent_variables

        for i, x in enumerate(X_independent):
            for indep_variable, indep_value in zip(independent_variables, x):
                indep_variable.value = float(indep_value)

            variable_values[i, :] = self.variable_values

        return variable_values

    @ensures2d
    @untransforms
    def get_independent_values(self, X: npt.ArrayLike) -> np.ndarray:
        """
        Remove dependent values from x.

        Parameters
        ----------
        X : array_like
            Value of all optimization variables.
            Works for transformed and untransformed space.

        Raises
        ------
        CADETProcessError
            If length of parameters does not match.

        Returns
        -------
        x_independent : np.ndarray
            Values of all independent optimization variables.
        """
        if X.shape[1] != self.n_variables:
            raise CADETProcessError(f"Expected {self.n_variables} value(s).")

        independent_values = np.zeros((len(X), self.n_independent_variables))
        variables = self.variables

        for i, x in enumerate(X):
            x_independent = []
            for variable, value in zip(variables, x):
                if variable.is_independent:
                    x_independent.append(value)

            independent_values[i, :] = np.array(x_independent)

        return independent_values

    @untransforms
    @gets_dependent_values
    def set_variables(
        self,
        x: npt.ArrayLike,
        evaluation_objects: Optional[list | object | int] = -1,
    ) -> list:
        """
        Set the values from the x-vector to the EvaluationObjects.

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
        list
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
        for variable, value in zip(self.variables, x):
            variable.set_value(value)

    def _evaluate_population(
        self,
        target_functions: list[tp.Callable],
        X: npt.ArrayLike,
        parallelization_backend: ParallelizationBackendBase | None = None,
        force: bool = False,
    ) -> np.ndarray:
        """
        Evaluate target functions for each individual in population.

        Parameters
        ----------
        target_functions : list[tp.Callable]
            List of evaluation targets (e.g. objectives).
        X : npt.ArrayLike
            Population to be evaluated in untransformed space.
        parallelization_backend : ParallelizationBackendBase, optional
            Adapter to backend for parallel evaluation of population.
            By default, the individuals are evaluated sequentially.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        np.ndarray
            The results of the target functions.

        Raises
        ------
        CADETProcessError
            If dictcache is used for parallelized evaluation.
        """
        if parallelization_backend is None:
            parallelization_backend = SequentialBackend()

        if not self.cache.use_diskcache and parallelization_backend.n_cores != 1:
            raise CADETProcessError("Cannot use dict cache for multiprocessing.")

        def target_wrapper(x: npt.ArrayLike) -> np.ndarray:
            results = self._evaluate_individual(
                target_functions=target_functions,
                x=x,
                force=force,
            )
            self.cache.close()

            return results

        results = parallelization_backend.evaluate(target_wrapper, X)
        return np.array(results, ndmin=2)

    def _evaluate_individual(
        self,
        target_functions: list[tp.Callable],
        x: npt.ArrayLike,
        force: bool = False,
    ) -> np.ndarray:
        """
        Evaluate target functions for set of parameters.

        Parameters
        ----------
        x : npt.ArrayLike
            Value of all optimization variables in untransformed space.
        target_functions: list[tp.Callable],
            Evaluation functions.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        results : np.ndarray
            Values of the evaluation functions at point x.

        See Also
        --------
        evaluate_objectives
        evaluate_nonlinear_constraints
        _evaluate
        """
        x = np.asarray(x)
        results = np.empty((0,))

        for eval_fun in target_functions:
            try:
                value = self._evaluate(x, eval_fun, force)
                results = np.hstack((results, value))
            except CADETProcessError as e:
                self.logger.warning(
                    f'Evaluation of {eval_fun.name} failed at {x} with Error "{e}". '
                    f"Returning bad metrics."
                )
                results = np.hstack((results, eval_fun.bad_metrics))

        return results

    def _evaluate(
        self,
        x: npt.ArrayLike,
        func: Evaluator,
        force: bool = False,
    ) -> np.ndarray:
        """
        Iterate over all evaluation objects and evaluate at x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.
            Must include all variables, including the dependent variables.
        func : Evaluator or Objective, or Nonlinear Constraint, or Callback
            Evaluation function.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        results : np.ndarray
            Results of the evaluation functions.
        """
        self.logger.debug(f"evaluate {str(func)} at {x}")

        results = np.empty((0,))
        x_key = np.array(x).tobytes()

        if func.evaluators is not None:
            requires = [*func.evaluators, func]
        else:
            requires = [func]

        self.set_variables(x)
        evaluation_objects = func.evaluation_objects

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
                        key = (str(eval_obj), step.id, x_key)
                        result = self.cache.get(key)
                        self.logger.debug(f"Got {str(step)} results from cache.")
                        current_request = result
                        break
                    except KeyError:
                        pass

                    remaining.insert(0, step)
            else:
                remaining = requires

            self.logger.debug(
                f"Evaluating remaining functions: {[str(step) for step in remaining]}."
            )

            for step in remaining:
                if isinstance(step, Callback):
                    step.evaluate(current_request, eval_obj)
                    result = np.empty((0))
                else:
                    result = step.evaluate(current_request)

                key = (str(eval_obj), step.id, x_key)
                if not isinstance(step, Callback):
                    self.cache.set(key, result, tag=x_key)
                current_request = result

            if len(result) != func.n_metrics:
                raise CADETProcessError(
                    f"Got results with length {len(result)}. "
                    f"Expected length {func.n_metrics} from {str(func)}"
                )

            results = np.hstack((results, result))

        return results

    @property
    def evaluators(self) -> list[Evaluator]:
        """list: Evaluators in OptimizationProblem."""
        return self._evaluators

    @property
    def evaluators_dict_reference(self) -> dict:
        """dict: Evaluator objects indexed by original_callable."""
        return {evaluator.evaluator: evaluator for evaluator in self.evaluators}

    @property
    def evaluators_dict(self) -> dict:
        """dict: Evaluator objects indexed by name."""
        return {evaluator.name: evaluator for evaluator in self.evaluators}

    def add_evaluator(
        self,
        evaluator: tp.Callable,
        name: Optional[str] = None,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
    ) -> None:
        """
        Add Evaluator to OptimizationProblem.

        Evaluators can be referenced by objective and constraint functions to
        perform preprocessing steps.

        Parameters
        ----------
        evaluator : tp.Callable
            Evaluation function.
        name : str, optional
            Name of the evaluator.
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

    @property
    def objectives(self) -> list:
        """list: Objective functions."""
        return self._objectives

    @property
    def objective_names(self) -> list:
        """list: Objective function names."""
        return [obj.name for obj in self.objectives]

    @property
    def objective_labels(self) -> list:
        """list: Objective function metric labels."""
        labels = []
        for obj in self.objectives:
            labels += obj.labels

        return labels

    @property
    def n_objectives(self) -> int:
        """int: Number of objectives."""
        return sum([obj.n_total_metrics for obj in self.objectives])

    def add_objective(
        self,
        objective: tp.callable | MetricBase,
        name: Optional[str] = None,
        n_objectives: Optional[int] = 1,
        minimize: Optional[bool] = True,
        bad_metrics: Optional[float | list[float]] = None,
        evaluation_objects: None | int | list = -1,
        labels: Optional[str] = None,
        requires: Optional[Evaluator | list] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Add objective function to optimization problem.

        Parameters
        ----------
        objective : tp.Callable or MetricBase
            Objective function.
        name : str, optional
            Name of the objective.
        n_objectives : int, optional
            Number of metrics returned by objective function.
            The default is 1.
        minimize : bool, optional
            If True, objective is treated as minimization problem. The default is True.
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
            bad_metrics = objective.bad_metrics

        if evaluation_objects is None:
            evaluation_objects = []
        elif evaluation_objects == -1:
            evaluation_objects = self.evaluation_objects
        elif not isinstance(evaluation_objects, list):
            evaluation_objects = [evaluation_objects]
        for el in evaluation_objects:
            if el not in self.evaluation_objects:
                raise CADETProcessError(f"Unknown EvaluationObject: {str(el)}")

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
            n_objectives=n_objectives,
            minimize=minimize,
            bad_metrics=bad_metrics,
            evaluation_objects=evaluation_objects,
            evaluators=evaluators,
            labels=labels,
            args=args,
            kwargs=kwargs,
        )
        self._objectives.append(objective)

    @ensures2d
    @untransforms
    @gets_dependent_values
    @ensures_minimization(scores="objectives")
    def evaluate_objectives(
        self,
        X: npt.ArrayLike,
        parallelization_backend: ParallelizationBackendBase | None = None,
        force: bool = False,
    ) -> np.ndarray:
        """
        Evaluate objective functions for each individual x in population X.

        Parameters
        ----------
        X : npt.ArrayLike
            Population to be evaluated in untransformed space.
        parallelization_backend : ParallelizationBackendBase, optional
            Adapter to backend for parallel evaluation of population.
            By default, the individuals are evaluated sequentially.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        np.ndarray
            The optimization function values.

        See Also
        --------
        add_objectives
        _evaluate_population
        _evaluate_individual
        _evaluate
        """
        return self._evaluate_population(
            target_functions=self.objectives,
            X=X,
            parallelization_backend=parallelization_backend,
            force=force,
        )

    def evaluate_objectives_population(self, *args: Any, **kwargs: Any) -> None:
        """
        Evaluate objective functions for each individual in a population.

        This method is deprecated and will be removed in a future version.
        Use `evaluate_objectives` instead, which now supports the evaluation of nd arrays.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Notes
        -----
        This function issues a deprecation warning to indicate that it will be removed
        in future versions. Use `evaluate_objectives` for similar functionality.
        """
        warnings.warn(
            "This function is deprecated; use `evaluate_objectives` instead, which now "
            "directly supports the evaluation of nd arrays.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.evaluate_objectives(*args, *kwargs)

    @untransforms
    @gets_dependent_values
    def objective_jacobian(
        self,
        x: npt.ArrayLike,
        ensure_minimization: Optional[bool] = False,
        dx: Optional[float] = 1e-3,
    ) -> np.ndarray:
        """
        Compute jacobian of objective functions using finite differences.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.
        ensure_minimization : bool, default=False
            Flag that ensures minimization objective.
        dx : float
            Increment to x to use for determining the function gradient.

        Returns
        -------
        jacobian: npt.ArrayLike
            Value of the partial derivatives at point x.

        See Also
        --------
        objectives
        approximate_jac
        """
        jacobian = approximate_jac(
            x,
            self.evaluate_objectives,
            dx,
            ensure_minimization=ensure_minimization,
        )

        return jacobian

    @property
    def nonlinear_constraints(self) -> list:
        """list: Nonlinear constraint functions."""
        return self._nonlinear_constraints

    @property
    def nonlinear_constraint_names(self) -> list:
        """list: Nonlinear constraint function names."""
        return [nonlincon.name for nonlincon in self.nonlinear_constraints]

    @property
    def nonlinear_constraint_labels(self) -> list:
        """list: Nonlinear constraint function metric labels."""
        labels = []
        for nonlincon in self.nonlinear_constraints:
            labels += nonlincon.labels

        return labels

    @property
    def nonlinear_constraints_bounds(self) -> list:
        """list: Bounds of nonlinear constraint functions."""
        bounds = []
        for nonlincon in self.nonlinear_constraints:
            bounds += nonlincon.bounds

        return bounds

    @property
    def n_nonlinear_constraints(self) -> int:
        """int: Number of nonlinear_constraints."""
        return sum(
            [nonlincon.n_total_metrics for nonlincon in self.nonlinear_constraints]
        )

    def add_nonlinear_constraint(
        self,
        nonlincon: tp.Callable,
        name: Optional[str] = None,
        n_nonlinear_constraints: Optional[int] = 1,
        bad_metrics: Optional[float | list[float]] = None,
        evaluation_objects: Optional[int | list | object] = -1,
        bounds: Optional[float | list[float]] = 0,
        comparison_operator: Optional[str] = "le",
        labels: Optional[str] = None,
        requires: Optional[list | Evaluator] = None,
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ) -> None:
        """
        Add nonliner constraint function to optimization problem.

        Parameters
        ----------
        nonlincon : tp.Callable
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
        comparison_operator : {'ge', 'le'}, optional
            Comparator to define whether metric should be greater or equal to, or less
            than or equal to the specified bounds.
            The default is 'le' (lower or equal).
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
                raise CADETProcessError(f"Unknown EvaluationObject: {str(el)}")

        if isinstance(bounds, (float, int)):
            bounds = n_nonlinear_constraints * [bounds]
        if len(bounds) != n_nonlinear_constraints:
            raise CADETProcessError(f"Expected {n_nonlinear_constraints} bounds")

        if requires is None:
            requires = []
        elif not isinstance(requires, list):
            requires = [requires]

        try:
            evaluators = [self.evaluators_dict_reference[req] for req in requires]
        except KeyError as e:
            raise CADETProcessError(f"Unknown Evaluator: {str(e)}")

        nonlincon = NonlinearConstraint(
            nonlincon,
            name,
            n_nonlinear_constraints=n_nonlinear_constraints,
            bounds=bounds,
            comparison_operator=comparison_operator,
            bad_metrics=bad_metrics,
            evaluation_objects=evaluation_objects,
            evaluators=evaluators,
            labels=labels,
            args=args,
            kwargs=kwargs,
        )
        self._nonlinear_constraints.append(nonlincon)

    @ensures2d
    @untransforms
    @gets_dependent_values
    def evaluate_nonlinear_constraints(
        self,
        X: npt.ArrayLike,
        parallelization_backend: ParallelizationBackendBase | None = None,
        force: bool = False,
    ) -> np.ndarray:
        """
        Evaluate nonlinear constraint functions for each individual x in population X.

        Parameters
        ----------
        X : npt.ArrayLike
            Population to be evaluated in untransformed space.
        parallelization_backend : ParallelizationBackendBase, optional
            Adapter to backend for parallel evaluation of population.
            By default, the individuals are evaluated sequentially.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        np.ndarray
            The nonlinear constraint function values.

        See Also
        --------
        add_nonlinear_constraint
        evaluate_nonlinear_constraints_violation
        _evaluate_population
        _evaluate_individual
        _evaluate
        """
        return self._evaluate_population(
            target_functions=self.nonlinear_constraints,
            X=X,
            parallelization_backend=parallelization_backend,
            force=force,
        )

    def evaluate_nonlinear_constraints_population(
        self, *args: Any, **kwargs: Any
    ) -> None:
        """
        Evaluate nonlinear constraint functions for each individual in a population.

        This method is deprecated and will be removed in a future version.
        Use `evaluate_nonlinear_constraints` instead, which now supports the
        evaluation of nd arrays.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Notes
        -----
        This function issues a deprecation warning to indicate that it will be removed
        in future versions. Use `evaluate_nonlinear_constraints` for similar functionality.
        """
        warnings.warn(
            "This function is deprecated; use `evaluate_nonlinear_constraints` "
            "instead, which now directly supports the evaluation of nd arrays.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.evaluate_nonlinear_constraints(*args, *kwargs)

    @ensures2d
    @untransforms
    @gets_dependent_values
    def evaluate_nonlinear_constraints_violation(
        self,
        X: npt.ArrayLike,
        parallelization_backend: ParallelizationBackendBase | None = None,
        force: bool = False,
    ) -> np.ndarray:
        """
        Evaluate nonlinear constraint function violation for each x in population X.

        After evaluating the nonlinear constraint functions, the corresponding
        bounds are subtracted from the results.

        Parameters
        ----------
        X : npt.ArrayLike
            Population to be evaluated in untransformed space.
        parallelization_backend : ParallelizationBackendBase, optional
            Adapter to backend for parallel evaluation of population.
            By default, the individuals are evaluated sequentially.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        np.ndarray
            The nonlinear constraint violation function values.

        See Also
        --------
        add_nonlinear_constraint
        evaluate_nonlinear_constraints
        _evaluate_population
        _evaluate_individual
        _evaluate
        """
        factors = []
        for constr in self.nonlinear_constraints:
            factor = -1 if constr.comparison_operator == "ge" else 1
            factors += constr.n_total_metrics * [factor]

        G = self._evaluate_population(
            target_functions=self.nonlinear_constraints,
            X=X,
            parallelization_backend=parallelization_backend,
            force=force,
        )

        G_transformed = np.multiply(factors, G)
        bounds_transformed = np.multiply(factors, self.nonlinear_constraints_bounds)

        return G_transformed - bounds_transformed

    def evaluate_nonlinear_constraints_violation_population(
        self, *args: Any, **kwargs: Any
    ) -> None:
        """
        Evaluate nonlinear constraint violation for each individual in a population.

        This method is deprecated and will be removed in a future version.
        Use `evaluate_nonlinear_constraints_violation` instead, which now supports
        the evaluation of nd arrays.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Notes
        -----
        This function issues a deprecation warning to indicate that it will be removed
        in future versions. Use `evaluate_nonlinear_constraints_violation` for similar
        functionality.
        """
        warnings.warn(
            "This function is deprecated; use "
            "`evaluate_nonlinear_constraints_violation` instead, which now directly "
            "supports the evaluation of nd arrays.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.evaluate_nonlinear_constraints_violation(*args, *kwargs)

    @untransforms
    @gets_dependent_values
    def check_nonlinear_constraints(
        self,
        x: npt.ArrayLike,
        cv_nonlincon_tol: Optional[float | np.ndarray] = 0.0,
    ) -> bool:
        """
        Check if all nonlinear constraints are met.

        Parameters
        ----------
        x : npt.ArrayLike
            Value of the optimization variables in untransformed space.
        cv_nonlincon_tol : float or np.ndarray, optional
            Tolerance for checking the nonlinear constraints. If a scalar is provided,
            the same value is used for all constraints. Default is 0.0.

        Returns
        -------
        flag : bool
            True if all nonlinear constraints violation are smaller or equal to zero,
            False otherwise.

        Raises
        ------
        ValueError
            If length of `cv_nonlincon_tol` does not match the number of constraints.
        """
        cv = np.array(self.evaluate_nonlinear_constraints_violation(x))

        if np.isscalar(cv_nonlincon_tol):
            cv_nonlincon_tol = np.repeat(cv_nonlincon_tol, self.n_nonlinear_constraints)

        if len(cv_nonlincon_tol) != self.n_nonlinear_constraints:
            raise ValueError(
                f"Length of `cv_nonlincon_tol` ({len(cv_nonlincon_tol)}) does not "
                f"match number of constraints ({self.n_nonlinear_constraints})."
            )

        if np.any(cv > cv_nonlincon_tol):
            return False
        return True

    @untransforms
    @gets_dependent_values
    def nonlinear_constraint_jacobian(
        self, x: npt.ArrayLike, dx: float = 1e-3
    ) -> npt.ArrayLike:
        """
        Compute jacobian of the nonlinear constraints at point x.

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
    def callbacks(self) -> list:
        """list: Callback functions for recording progress."""
        return self._callbacks

    @property
    def callback_names(self) -> list:
        """list: Callback function names."""
        return [obj.name for obj in self.callbacks]

    @property
    def n_callbacks(self) -> int:
        """int: Number of callback functions."""
        return len(self.callbacks)

    def add_callback(
        self,
        callback: tp.Callable,
        name: Optional[str] = None,
        evaluation_objects: Optional[int | list | object] = -1,
        requires: Optional[Evaluator | list] = None,
        frequency: int = 1,
        callbacks_dir: Optional[str] = None,
        keep_progress: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Add callback function for processing (intermediate) results.

        Parameters
        ----------
        callback : tp.Callable
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
                raise CADETProcessError(f"Unknown EvaluationObject: {str(el)}")

        if requires is None:
            requires = []
        elif not isinstance(requires, list):
            requires = [requires]

        try:
            evaluators = [self.evaluators_dict_reference[req] for req in requires]
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
            kwargs=kwargs,
        )
        self._callbacks.append(callback)

    def evaluate_callbacks(
        self,
        population: Population | Individual | npt.ArrayLike,
        current_iteration: int = 0,
        parallelization_backend: ParallelizationBackendBase | None = None,
        force: bool = False,
    ) -> NoReturn:
        """
        Evaluate callback functions for each individual x in population X.

        Parameters
        ----------
        population : Population | Individual | npt.ArrayLike
            Population to be evaluated.
            If an Individual is passed, a new population will be created.
            If a numpy array is passed, a new population will be created, assuming the
            values are independent values in untransformed space.
        current_iteration : int, optional
            Current iteration step. This value is used to determine whether the
            evaluation of callbacks should be skipped according to their evaluation
            frequency. The default is 0, indicating the callbacks will be evaluated
            regardless of the specified frequency.
        parallelization_backend : ParallelizationBackendBase, optional
            Adapter to backend for parallel evaluation of population.
            By default, the individuals are evaluated sequentially.
        force : bool
            If True, do not use cached results. The default is False.

        See Also
        --------
        add_callback
        _evaluate_population
        _evaluate_individual
        _evaluate
        """
        if isinstance(population, Individual):
            ind = population
            population = Population()
            population.add_individual(ind)
        elif isinstance(population, (list, np.ndarray)):
            population = self.create_population(population)

        if parallelization_backend is None:
            parallelization_backend = SequentialBackend()

        if not self.cache.use_diskcache and parallelization_backend.n_cores != 1:
            raise CADETProcessError("Cannot use dict cache for multiprocessing.")

        def evaluate_callbacks(ind: Individual) -> None:
            for callback in self.callbacks:
                if not (
                    current_iteration == "final"
                    or current_iteration % callback.frequency == 0
                ):
                    continue

                callback._ind = ind
                callback._current_iteration = current_iteration

                try:
                    self._evaluate(ind.x, callback, force)
                except CADETProcessError as e:
                    self.logger.warning(
                        f'Evaluation of {callback} failed at {ind.x} with Error "{e}".'
                    )

        parallelization_backend.evaluate(evaluate_callbacks, population)

    def evaluate_callbacks_population(self, *args: Any, **kwargs: Any) -> None:
        """
        Evaluate callbacks functions for each individual in a population.

        This method is deprecated and will be removed in a future version.
        Use `evaluate_callbacks` instead, which now supports the evaluation of nd arrays.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Notes
        -----
        This function issues a deprecation warning to indicate that it will be removed
        in future versions. Use `evaluate_callbacks` for similar functionality.
        """
        warnings.warn(
            "This function is deprecated; use `evaluate_callbacks` instead, which now "
            "directly supports the evaluation of nd arrays.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.evaluate_callbacks(*args, *kwargs)

    @property
    def meta_scores(self) -> list:
        """list: Meta scores for multi criteria selection."""
        return self._meta_scores

    @property
    def meta_score_names(self) -> list:
        """list: Meta score function names."""
        return [meta_score.name for meta_score in self.meta_scores]

    @property
    def meta_score_labels(self) -> int:
        """int: Meta score function metric labels."""
        labels = []
        for meta_score in self.meta_scores:
            labels += meta_score.labels

        return labels

    @property
    def n_meta_scores(self) -> int:
        """int: Number of meta score functions."""
        return sum([meta_score.n_total_metrics for meta_score in self.meta_scores])

    def add_meta_score(
        self,
        meta_score: tp.Callable,
        name: Optional[str] = None,
        n_meta_scores: int = 1,
        minimize: bool = True,
        evaluation_objects: Optional[object | int | list] = -1,
        requires: Optional[Evaluator | list] = None,
    ) -> None:
        """
        Add Meta score to the OptimizationProblem.

        Parameters
        ----------
        meta_score : tp.Callable
            Objective function.
        name : str, optional
            Name of the meta score.
        n_meta_scores : int, optional
            Number of meta scores returned by callable.
            The default is 1.
        minimize : bool, optional
            If True, meta score is treated as minimization problem. The default is True.
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
                raise CADETProcessError(f"Unknown EvaluationObject: {str(el)}")

        if requires is None:
            requires = []
        elif not isinstance(requires, list):
            requires = [requires]

        try:
            evaluators = [self.evaluators_dict_reference[req] for req in requires]
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

    @ensures2d
    @untransforms
    @gets_dependent_values
    @ensures_minimization(scores="meta_scores")
    def evaluate_meta_scores(
        self,
        X: npt.ArrayLike,
        parallelization_backend: ParallelizationBackendBase | None = None,
        force: bool = False,
    ) -> np.ndarray:
        """
        Evaluate meta scores for each individual x in population X.

        Parameters
        ----------
        X : npt.ArrayLike
            Population to be evaluated in untransformed space.
        parallelization_backend : ParallelizationBackendBase, optional
            Adapter to backend for parallel evaluation of population.
            By default, the individuals are evaluated sequentially.
        force : bool
            If True, do not use cached results. The default is False.

        Returns
        -------
        np.ndarray
            The meta scores.

        See Also
        --------
        add_meta_score
        _evaluate_population
        _evaluate_individual
        _evaluate
        """
        return self._evaluate_population(
            target_functions=self.meta_scores,
            X=X,
            parallelization_backend=parallelization_backend,
            force=force,
        )

    def evaluate_meta_scores_population(self, *args: Any, **kwargs: Any) -> None:
        """
        Evaluate meta scores for each individual in a population.

        This method is deprecated and will be removed in a future version.
        Use `evaluate_meta_scores` instead, which now supports the evaluation of nd arrays.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Notes
        -----
        This function issues a deprecation warning to indicate that it will be removed
        in future versions. Use `evaluate_meta_scores` for similar functionality.
        """
        warnings.warn(
            "This function is deprecated; use `evaluate_meta_scores` instead, which now "
            "directly supports the evaluation of nd arrays.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.evaluate_meta_scores(*args, *kwargs)

    @property
    def multi_criteria_decision_functions(self) -> list:
        """list: Multi criteria decision functions."""
        return self._multi_criteria_decision_functions

    @property
    def multi_criteria_decision_function_names(self) -> list:
        """list: Multi criteria decision function names."""
        return [mcdf.name for mcdf in self.multi_criteria_decision_functions]

    @property
    def n_multi_criteria_decision_functions(self) -> int:
        """int: number of multi criteria decision functions."""
        return len(self.multi_criteria_decision_functions)

    def add_multi_criteria_decision_function(
        self,
        decision_function: tp.Callable,
        name: Optional[str] = None,
    ) -> None:
        """
        Add multi criteria decision function to OptimizationProblem.

        Parameters
        ----------
        decision_function : tp.Callable
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
            if inspect.isfunction(decision_function) or inspect.ismethod(
                decision_function
            ):
                name = decision_function.__name__
            else:
                name = str(decision_function)

        if name in self.multi_criteria_decision_function_names:
            warnings.warn(
                "Multi criteria decision function with same name already exists."
            )

        meta_score = MultiCriteriaDecisionFunction(decision_function, name)
        self._multi_criteria_decision_functions.append(meta_score)

    def evaluate_multi_criteria_decision_functions(
        self,
        pareto_population: Population,
    ) -> np.ndarray:
        """
        Evaluate evaluate multi criteria decision functions.

        Parameters
        ----------
        pareto_population : Population
            Pareto optimal solution.

        Returns
        -------
        x_pareto : np.ndarray
            Value of the optimization variables.

        See Also
        --------
        add_multi_criteria_decision_function
        """
        self.logger.debug("Evaluate multi criteria decision functions.")

        for func in self.multi_criteria_decision_functions:
            pareto_population = func(pareto_population)

        return pareto_population

    @property
    def lower_bounds(self) -> list:
        """list: Lower bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.lb for var in self.variables]

    @property
    def lower_bounds_transformed(self) -> list:
        """list: Transformed lower bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.transformer.lb for var in self.variables]

    @property
    def lower_bounds_independent(self) -> list:
        """list: Lower bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.lb for var in self.independent_variables]

    @property
    def lower_bounds_independent_transformed(self) -> list:
        """list: Transformed lower bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.transformer.lb for var in self.independent_variables]

    @property
    def upper_bounds(self) -> list:
        """list: Upper bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.ub for var in self.variables]

    @property
    def upper_bounds_transformed(self) -> list:
        """list: Transformed upper bounds of all OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.transformer.ub for var in self.variables]

    @property
    def upper_bounds_independent(self) -> list:
        """list: Upper bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.ub for var in self.independent_variables]

    @property
    def upper_bounds_independent_transformed(self) -> list:
        """list: Transformed upper bounds of independent OptimizationVariables.

        See Also
        --------
        upper_bounds

        """
        return [var.transformer.ub for var in self.independent_variables]

    @untransforms
    @gets_dependent_values
    def evaluate_bounds(self, x: npt.ArrayLike) -> np.ndarray:
        """
        Calculate bound violation.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.

        Returns
        -------
        constraints: np.ndarray
            Value of the linear constraints at point x

        See Also
        --------
        check_bounds
        lower_bounds
        upper_bounds
        """
        # Calculate the residuals for lower bounds
        lower_residual = np.maximum(0, self.lower_bounds - x)

        # Calculate the residuals for upper bounds
        upper_residual = np.maximum(0, x - self.upper_bounds)

        # Combine the residuals
        residual = lower_residual + upper_residual

        return residual

    @untransforms
    @gets_dependent_values
    def check_bounds(
        self,
        x: npt.ArrayLike,
        cv_bounds_tol: Optional[float | np.ndarray] = 0.0,
    ) -> bool:
        """
        Check if all bound constraints are kept.

        Parameters
        ----------
        x : npt.ArrayLike
            Value of the optimization variables in untransformed space.
        cv_bounds_tol : float or np.ndarray, optional
            Tolerance for checking the bound constraints. If a scalar is provided,
            the same value is used for all variables. Default is 0.0.

        Returns
        -------
        flag : bool
            True if all nonlinear constraints violation are smaller or equal to zero,
            False otherwise.

        Raises
        ------
        ValueError
            If length of `cv_bounds_tol` does not match the number of variables.
        """
        flag = True

        values = np.array(x, ndmin=1)

        if np.isscalar(cv_bounds_tol):
            cv_bounds_tol = np.repeat(cv_bounds_tol, self.n_variables)

        if len(cv_bounds_tol) != self.n_variables:
            raise ValueError(
                f"Length of `cv_bounds_tol` ({len(cv_bounds_tol)}) does not match "
                f"number of variables ({self.n_variables})."
            )

        if np.any(np.less(values, self.lower_bounds - cv_bounds_tol)):
            flag = False
        if np.any(np.greater(values, self.upper_bounds + cv_bounds_tol)):
            flag = False

        return flag

    @property
    def linear_constraints(self) -> list:
        """list: Linear inequality constraints of OptimizationProblem.

        See Also
        --------
        add_linear_constraint
        remove_linear_constraint
        linear_equality_constraints

        """
        return self._linear_constraints

    @property
    def n_linear_constraints(self) -> int:
        """int: Number of linear inequality constraints."""
        return len(self.linear_constraints)

    def add_linear_constraint(
        self,
        opt_vars: list[str],
        lhs: Optional[float | list[float]] = 1.0,
        b: Optional[float] = 0.0,
    ) -> None:
        """
        Add linear inequality constraints.

        Parameters
        ----------
        opt_vars : list of strings
            Names of the OptimizationVariable to be added.
        lhs : float or list of float, optional
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
            raise CADETProcessError("Variable not in variables.")

        if np.isscalar(lhs):
            lhs = lhs * np.ones(len(opt_vars))

        if len(lhs) != len(opt_vars):
            raise CADETProcessError(
                "Number of lhs coefficients and variables do not match."
            )

        lincon = dict()
        lincon["opt_vars"] = opt_vars
        lincon["lhs"] = lhs
        lincon["b"] = float(b)

        self._linear_constraints.append(lincon)

    def remove_linear_constraint(self, index: int) -> None:
        """
        Remove linear inequality constraint.

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
    def A(self) -> np.ndarray:
        """
        np.ndarray: LHS Matrix of linear inequality constraints.

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
            for var_index, var in enumerate(lincon["opt_vars"]):
                index = self.variables.index(self.variables_dict[var])
                A[lincon_index, index] = lincon["lhs"][var_index]

        return A

    @property
    def A_transformed(self) -> np.ndarray:
        """
        np.ndarray: LHS Matrix of linear inequality constraints in transformed space.

        See Also
        --------
        A
        A_independent_transformed
        A_independent
        """
        A_t = self.A.copy()
        for a in A_t:
            for j, v in enumerate(self.variables):
                t = v.transformer
                if isinstance(t, NullTransformer):
                    continue

                if a[j] != 0 and not t.is_linear:
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
    def A_independent(self) -> np.ndarray:
        """
        np.ndarray: LHS Matrix of linear inequality constraints for indep variables.

        See Also
        --------
        A
        A_transformed
        A_independent_transformed
        """
        return self.A[:, self.independent_variable_indices]

    @property
    def A_independent_transformed(self) -> np.ndarray:
        """
        np.ndarray: LHS of lin ineqs for indep variables in transformed space.

        See Also
        --------
        A
        A_transformed
        A_independent
        """
        return self.A_transformed[:, self.independent_variable_indices]

    @property
    def b(self) -> np.ndarray:
        """
        np.ndarray: Vector form of linear constraints.

        See Also
        --------
        A
        add_linear_constraint
        remove_linear_constraint
        b_transformed
        """
        b = [lincon["b"] for lincon in self.linear_constraints]

        return np.array(b)

    @property
    def b_transformed(self) -> np.ndarray:
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
                t = v.transformer
                if isinstance(t, NullTransformer):
                    continue

                if a[j] != 0 and not t.is_linear:
                    raise CADETProcessError(
                        "Non-linear transform was used in linear constraints."
                    )
                # I realize: This sounds an awful lot like transforms that sklear offers
                # center and scale

                # FUTURE: move to transforms module
                scale_old = t.ub_input - t.lb_input
                scale_new = t.ub - t.lb
                scale = scale_old / scale_new

                mid_old = 0.5 * (t.ub_input + t.lb_input)
                mid_new = 0.5 * (t.ub + t.lb)
                v_shift = mid_new - mid_old

                b_t[i] += a[j] * v_shift * scale

        return b_t

    @untransforms
    @gets_dependent_values
    def evaluate_linear_constraints(self, x: npt.ArrayLike) -> np.ndarray:
        """
        Calculate value of linear inequality constraints at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.

        Returns
        -------
        constraints: np.ndarray
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
    def check_linear_constraints(
        self,
        x: npt.ArrayLike,
        cv_lincon_tol: Optional[float | np.ndarray] = 0.0,
    ) -> bool:
        """
        Check if linear inequality constraints are met at point x.

        Parameters
        ----------
        x : npt.ArrayLike
            Value of the optimization variables in untransformed space.
        cv_lincon_tol : float or np.ndarray, optional
            Tolerance for checking the linear constraints. If a scalar is provided,
            the same value is used for all constraints. Default is 0.0.

        Returns
        -------
        flag : bool
            True if linear inequality constraints are met. False otherwise.

        Raises
        ------
        ValueError
            If the length of `cv_lincon_tol` does not match the number of constraints.

        See Also
        --------
        linear_constraints
        evaluate_linear_constraints
        A
        b
        """
        flag = True

        linear_constraints_values = self.evaluate_linear_constraints(x)

        if np.isscalar(cv_lincon_tol):
            cv_lincon_tol = np.repeat(cv_lincon_tol, self.n_linear_constraints)

        if len(cv_lincon_tol) != self.n_linear_constraints:
            raise ValueError(
                f"Length of `cv_lincon_tol` ({len(cv_lincon_tol)}) does not match "
                f"number of constraints ({self.n_linear_constraints})."
            )

        if np.any(linear_constraints_values > cv_lincon_tol):
            flag = False

        return flag

    @property
    def linear_equality_constraints(self) -> list:
        """list: Linear equality constraints of OptimizationProblem.

        See Also
        --------
        add_linear_equality_constraint
        remove_linear_equality_constraint
        linear_constraints

        """
        return self._linear_equality_constraints

    @property
    def n_linear_equality_constraints(self) -> int:
        """int: Number of linear equality constraints."""
        return len(self.linear_equality_constraints)

    def add_linear_equality_constraint(
        self,
        opt_vars: list[str],
        lhs: Optional[float | list[float]] = 1.0,
        beq: Optional[float] = 0.0,
        eps: Optional[float] = 0.0,
    ) -> None:
        """
        Add linear equality constraints.

        Parameters
        ----------
        opt_vars : list of strings
            Names of the OptimizationVariable to be added.
        lhs : float or list, optional
            Left-hand side / coefficients of the constraints.
            If scalar, same coefficient is used for all variables.
            The default is 1.0
        beq : float, optional
            Constraint of inequality constraint. The default is 0.0.
        eps : float, optional
            Error tolerance. The default is 0.0.

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
            raise CADETProcessError("Variable not in variables.")

        if np.isscalar(lhs):
            lhs = lhs * np.ones(len(opt_vars))

        if len(lhs) != len(opt_vars):
            raise CADETProcessError(
                "Number of lhs coefficients and variables do not match."
            )

        lineqcon = dict()
        lineqcon["opt_vars"] = opt_vars
        lineqcon["lhs"] = lhs
        lineqcon["beq"] = beq
        lineqcon["eps"] = float(eps)

        self._linear_equality_constraints.append(lineqcon)

    def remove_linear_equality_constraint(self, index: int) -> None:
        """
        Remove linear equality constraint.

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
    def Aeq(self) -> np.ndarray:
        """
        np.ndarray: LHS Matrix form of linear equality constraints.

        See Also
        --------
        beq
        add_linear_equality_constraint
        remove_linear_equality_constraint
        """
        Aeq = np.zeros((len(self.linear_equality_constraints), len(self.variables)))

        for lineqcon_index, lineqcon in enumerate(self.linear_equality_constraints):
            for var_index, var in enumerate(lineqcon["opt_vars"]):
                index = self.variables.index(self.variables_dict[var])
                Aeq[lineqcon_index, index] = lineqcon["lhs"][var_index]

        return Aeq

    @property
    def Aeq_transformed(self) -> np.ndarray:
        """
        np.ndarray: LHS Matrix of linear equality constraints in transformed space.

        See Also
        --------
        Aeq
        Aeq_independent_transformed
        Aeq_independent
        """
        Aeq_t = self.Aeq.copy()
        for aeq in Aeq_t:
            for j, v in enumerate(self.variables):
                t = v.transformer
                if isinstance(t, NullTransformer):
                    continue

                if aeq[j] != 0 and not t.is_linear:
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
    def Aeq_independent(self) -> np.ndarray:
        """
        np.ndarray: LHS Matrix of linear inequality constraints for indep variables.

        See Also
        --------
        Aeq
        Aeq_transformed
        Aeq_independent_transformed
        """
        return self.Aeq[:, self.independent_variable_indices]

    @property
    def Aeq_independent_transformed(self) -> np.ndarray:
        """
        np.ndarray: LHS of lin ineqs for indep variables in transformed space.

        See Also
        --------
        Aeq
        Aeq_transformed
        Aeq_independent
        """
        return self.Aeq_transformed[:, self.independent_variable_indices]

    @property
    def beq(self) -> np.ndarray:
        """
        np.ndarray: Vector form of linear equality constraints.

        See Also
        --------
        Aeq
        add_linear_equality_constraint
        remove_linear_equality_constraint
        """
        beq = [lincon["beq"] for lincon in self.linear_equality_constraints]

        return np.array(beq)

    @property
    def beq_transformed(self) -> np.ndarray:
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
                t = v.transformer
                if isinstance(t, NullTransformer):
                    continue

                if aeq[j] != 0 and not t.is_linear:
                    raise CADETProcessError(
                        "Non-linear transform was used in linear constraints."
                    )
                # I realize: This sounds an awful lot like transforms that sklear offers
                # center and scale

                # FUTURE: move to transforms module
                scale_old = t.ub_input - t.lb_input
                scale_new = t.ub - t.lb
                scale = scale_old / scale_new

                mid_old = 0.5 * (t.ub_input + t.lb_input)
                mid_new = 0.5 * (t.ub + t.lb)
                v_shift = mid_new - mid_old

                beq_t[i] += aeq[j] * v_shift * scale

        return beq_t

    @property
    def eps_lineq(self) -> np.ndarray:
        """
        np.array: Relaxation tolerance for linear equality constraints.

        See Also
        --------
        add_linear_inequality_constraint
        """
        return np.array(
            [lineqcon["eps"] for lineqcon in self.linear_equality_constraints]
        )

    @untransforms
    @gets_dependent_values
    def evaluate_linear_equality_constraints(self, x: npt.ArrayLike) -> np.array:
        """
        Calculate value of linear equality constraints at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables in untransformed space.

        Returns
        -------
        constraints: npt.ArrayLike
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
    def check_linear_equality_constraints(
        self,
        x: npt.ArrayLike,
        cv_lineq_tol: Optional[float | np.ndarray] = 0.0,
    ) -> bool:
        """
        Check if linear equality constraints are met at point x.

        Parameters
        ----------
        x : npt.ArrayLike
            Value of the optimization variables in untransformed space.
        cv_lineq_tol : float or np.ndarray, optional
            Tolerance for checking linear equality constraints. If a scalar is provided,
            the same value is used for all constraints. Default is 0.0.

        Returns
        -------
        flag : bool
            True if linear equality constraints are met. False otherwise.

        Raises
        ------
        ValueError
            If length of `cv_lineq_tol` does not match the number of constraints.
        """
        flag = True

        lhs = self.evaluate_linear_equality_constraints(x)

        if np.isscalar(cv_lineq_tol):
            cv_lineq_tol = np.repeat(cv_lineq_tol, len(lhs))

        if len(cv_lineq_tol) != self.n_linear_equality_constraints:
            raise ValueError(
                f"Length of `cv_lineq_tol` ({len(cv_lineq_tol)}) does not match "
                f"number of constraints ({self.n_linear_equality_constraints})."
            )

        if np.any(np.abs(lhs) > self.eps_lineq - cv_lineq_tol):
            flag = False

        return flag

    @ensures2d
    def transform(self, X_independent: npt.ArrayLike) -> np.ndarray:
        """
        Transform independent optimization variables from untransformed parameter space.

        Parameters
        ----------
        x_independent : np.ndarray
            Independent optimization variables in untransformed space.

        Returns
        -------
        np.ndarray
            Optimization variables in transformed parameter space.
        """
        transform = np.zeros(X_independent.shape)

        for i, ind in enumerate(X_independent):
            transform[i, :] = [
                var.transform(value)
                for value, var in zip(ind, self.independent_variables)
            ]

        return transform

    @ensures2d
    def untransform(self, X_transformed: npt.ArrayLike) -> np.ndarray:
        """
        Untransform the optimization variables from transformed parameter space.

        Parameters
        ----------
        X_transformed : npt.ArrayLike
            Independent optimization variables in transformed parameter space.

        Returns
        -------
        np.ndarray
            Optimization variables in untransformed parameter space.
        """
        untransform = np.zeros(X_transformed.shape)

        for i, ind in enumerate(X_transformed):
            untransform[i, :] = [
                var.untransform(value, significant_digits=var.significant_digits)
                for value, var in zip(ind, self.independent_variables)
            ]

        return untransform

    @property
    def cached_steps(self) -> list:
        """list: Cached evaluation steps."""
        return self.objectives + self.nonlinear_constraints + self.meta_scores

    @property
    def cache_directory(self) -> str:
        """pathlib.Path: Path for results cache database."""
        if self._cache_directory is None:
            if "XDG_CACHE_HOME" in os.environ:
                _cache_directory = (
                    Path(os.environ["XDG_CACHE_HOME"])
                    / "CADET-Process"
                    / digest_string(settings.working_directory)
                    / f"diskcache_{self.name}"
                )
            else:
                _cache_directory = settings.working_directory / f"diskcache_{self.name}"
        else:
            _cache_directory = Path(self._cache_directory).absolute()

        if self.use_diskcache:
            _cache_directory.mkdir(exist_ok=True, parents=True)

        return _cache_directory

    @cache_directory.setter
    def cache_directory(self, cache_directory: str) -> None:
        self._cache_directory = cache_directory

    def setup_cache(self, n_shards: Optional[int] = 1) -> None:
        """Set up cache to store (intermediate) results."""
        self.cache = ResultsCache(self.use_diskcache, self.cache_directory, n_shards)

    def delete_cache(self, reinit: Optional[bool] = False) -> None:
        """Delete cache with (intermediate) results."""
        try:
            self.cache.delete_database()
        except AttributeError:
            pass
        if reinit:
            self.setup_cache()

    def prune_cache(
        self, tag: Optional[str] = None, close: Optional[bool] = True
    ) -> None:
        """
        Prune cache with (intermediate) results.

        Parameters
        ----------
        tag : str
            Tag to be removed. The default is 'temp'.
        close : bool, optional
            If True, database will be closed after operation. The default is True.
        """
        self.cache.prune(tag, close)

    def create_hopsy_problem(
        self,
        include_dependent_variables: Optional[bool] = True,
        simplify: Optional[bool] = False,
        use_custom_model: Optional[bool] = False,
    ) -> hopsy.Problem:
        """
        Create a hopsy problem from the optimization problem.

        Parameters
        ----------
        include_dependent_variables : bool, optional
            If True, only use the hopsy problem. The default is False.
        simplify : bool, optional
            If True, simplify the hopsy problem. The default is False.
        use_custom_model : bool, optional
            If True, use custom model to improve sampling of log normalized parameters.
            The default is False.

        Returns
        -------
        problem
            hopsy.Problem
        """

        class CustomModel:
            def __init__(self, log_space_indices: list) -> None:
                self.log_space_indices = log_space_indices

            def compute_negative_log_likelihood(self, x: np.ndarray) -> float:
                return np.sum(np.log(x[self.log_space_indices]))

        if include_dependent_variables:
            variables = self.variables
        else:
            variables = self.independent_variables

        log_space_indices = []

        for i, var in enumerate(variables):
            if (
                isinstance(var.transformer, NormLogTransformer)
                or
                (isinstance(var.transformer, AutoTransformer) and var.transformer.use_log)
            ):
                log_space_indices.append(i)

        lp = hopsy.LP()
        lp.reset()
        lp.settings.thresh = 1e-15

        if len(log_space_indices) and use_custom_model > 0:
            model = CustomModel(log_space_indices)
        else:
            model = None

        if include_dependent_variables:
            A = self.A
            b = self.b
            lower_bounds = self.lower_bounds
            upper_bounds = self.upper_bounds
            Aeq = self.Aeq
            beq = self.beq
        else:
            A = self.A_independent
            b = self.b
            lower_bounds = self.lower_bounds_independent
            upper_bounds = self.upper_bounds_independent
            Aeq = self.Aeq_independent
            beq = self.beq

        problem = hopsy.Problem(
            A,
            b,
            model,
        )

        problem = hopsy.add_box_constraints(
            problem,
            lower_bounds,
            upper_bounds,
            simplify=simplify,
        )

        if self.n_linear_equality_constraints > 0:
            problem = hopsy.add_equality_constraints(
                problem,
                Aeq,
                beq,
            )

        return problem

    def get_chebyshev_center(
        self,
        include_dependent_variables: Optional[bool] = True,
    ) -> list[float]:
        """
        Compute chebychev center.

        The Chebyshev center is the center of the largest Euclidean ball that is fully
        contained within the polytope of the parameter space.

        Parameters
        ----------
        include_dependent_variables : Optional[bool], default=True
            If True, include dependent variables in population.

        Returns
        -------
        chebyshev : list[float]
            Chebyshev center.
        """
        problem = self.create_hopsy_problem(
            include_dependent_variables=False, simplify=False, use_custom_model=True
        )

        chebyshev = hopsy.compute_chebyshev_center(problem, original_space=True)[:, 0]

        if include_dependent_variables:
            chebyshev = self.get_dependent_values(chebyshev)

        return chebyshev

    def create_initial_values(
        self,
        n_samples: Optional[int] = 1,
        seed: Optional[int] = None,
        burn_in: Optional[int] = 100000,
        include_dependent_variables: Optional[bool] = True,
    ) -> np.ndarray:
        """
        Create initial value within parameter space.

        Uses hopsy (Highly Optimized toolbox for Polytope Sampling) to retrieve
        uniformly distributed samples from the parameter space.

        Parameters
        ----------
        n_samples : int, optional
            Number of initial values to be drawn. The default is 1.
        seed : int, optional
            Seed to initialize random numbers.
        burn_in : int, optional
            Number of samples that are created to ensure uniform sampling.
            The actual initial values are then drawn from this set.
            The default is 100000.
        include_dependent_variables : bool, optional
            If True, include dependent variables in population.
            The default is True.

        Raises
        ------
        CADETProcessError
            If not enough individuals fulfilling linear constraints are found.

        Returns
        -------
        values : np.ndarray
            Initial values for starting the optimization.
        """
        burn_in = int(burn_in)

        if seed is None:
            seed = random.randint(0, 255)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            problem = self.create_hopsy_problem(
                include_dependent_variables=False,
                simplify=False,
                use_custom_model=True,
            )

            problem = hopsy.round(problem, simplify=False)

            mc = hopsy.MarkovChain(
                problem,
                proposal=hopsy.UniformCoordinateHitAndRunProposal,
            )
            rng_hopsy = hopsy.RandomNumberGenerator(seed=seed)

            acceptance_rate, states = hopsy.sample(
                mc, rng_hopsy, n_samples=burn_in, thinning=2
            )
            independent_values = states[0, ...]

        rng = np.random.default_rng(seed)
        values = []
        counter = 0
        while len(values) < n_samples:
            if counter > burn_in:
                raise CADETProcessError(
                    "Cannot find individuals that fulfill constraints."
                )

            counter += 1
            i = rng.integers(0, burn_in)
            ind = []
            for i_var, var in enumerate(self.independent_variables):
                ind.append(
                    float(
                        round_to_significant_digits(
                            independent_values[i, i_var],
                            digits=var.significant_digits,
                        )
                    )
                )

            ind = self.get_dependent_values(ind)

            if not self.check_individual(
                ind, check_nonlinear_constraints=False, silent=True
            ):
                continue

            if not include_dependent_variables:
                ind = self.get_independent_values(ind)

            values.append(ind)

        return np.array(values, ndmin=2)

    @untransforms
    @gets_dependent_values
    def create_individual(
        self,
        x: np.ndarray,
        f: np.ndarray | None = None,
        f_min: np.ndarray | None = None,
        g: np.ndarray | None = None,
        cv_nonlincon: np.ndarray | None = None,
        m: np.ndarray | None = None,
        m_min: np.ndarray | None = None,
    ) -> Individual:
        """
        Create new individual from data.

        Parameters
        ----------
        x : np.ndarray
            Variable values in untransformed space.
        f : np.ndarray
            Objective values.
        f_min : np.ndarray
            Minimized objective values.
        g : np.ndarray
            Nonlinear constraint values.
        cv_nonlincon : np.ndarray
            Nonlinear constraints violation.
        m : np.ndarray
            Meta score values.
        m_min : np.ndarray
            Minimized meta score values.

        Returns
        -------
        Individual
            The newly created individual.
        """
        x_indep = self.get_independent_values(x)
        x_transformed = self.transform(x_indep)

        cv_bounds = self.evaluate_bounds(x)
        cv_lincon = self.evaluate_linear_constraints(x)
        cv_lineqcon = np.abs(self.evaluate_linear_equality_constraints(x))

        ind = Individual(
            x=x,
            x_transformed=x_transformed,
            cv_bounds=cv_bounds,
            cv_lincon=cv_lincon,
            cv_lineqcon=cv_lineqcon,
            f=f,
            f_min=f_min,
            g=g,
            cv_nonlincon=cv_nonlincon,
            m=m,
            m_min=m_min,
            independent_variable_names=self.independent_variable_names,
            objective_labels=self.objective_labels,
            nonlinear_constraint_labels=self.nonlinear_constraint_labels,
            meta_score_labels=self.meta_score_labels,
            variable_names=self.variable_names,
        )

        return ind

    @untransforms
    @gets_dependent_values
    def create_population(
        self,
        X: npt.ArrayLike,
        F: npt.ArrayLike = None,
        F_min: npt.ArrayLike | None = None,
        G: npt.ArrayLike | None = None,
        CV_nonlincon: npt.ArrayLike | None = None,
        M: npt.ArrayLike | None = None,
        M_min: npt.ArrayLike | None = None,
    ) -> Population:
        """
        Create new population from data.

        Parameters
        ----------
        X : npt.ArrayLike
            Variable values in untransformed space.
        F : npt.ArrayLike
            Objective values.
        F_min : npt.ArrayLike
            Minimized objective values.
        G : npt.ArrayLike
            Nonlinear constraint values.
        CV_nonlincon : npt.ArrayLike
            Nonlinear constraints violation.
        M : npt.ArrayLike
            Meta score values.
        M_min : npt.ArrayLike
            Minimized meta score values.

        Returns
        -------
        Population
            The newly created population.
        """
        X = np.array(X, ndmin=2)

        if F is None:
            F = len(X) * [None]
        else:
            F = np.array(F, ndmin=2)

        if F_min is None:
            F_min = F
        else:
            F_min = np.array(F_min, ndmin=2)

        if G is None:
            G = len(X) * [None]
        else:
            G = np.array(G, ndmin=2)

        if CV_nonlincon is None:
            CV_nonlincon = G
        else:
            CV_nonlincon = np.array(CV_nonlincon, ndmin=2)

        if M is None:
            M = len(X) * [None]
        else:
            M = np.array(M, ndmin=2)

        if M_min is None:
            M_min = M
        else:
            M_min = np.array(M_min, ndmin=2)

        pop = Population()
        for x, f, f_min, g, cv_nonlincon, m, m_min in zip(
            X, F, F_min, G, CV_nonlincon, M, M_min
        ):
            ind = self.create_individual(
                x,
                f=f,
                f_min=f_min,
                g=g,
                cv_nonlincon=cv_nonlincon,
                m=m,
                m_min=m_min,
            )
            pop.add_individual(ind)

        return pop

    @property
    def parameters(self) -> dict:
        """dict: Parameters of the optimization problem."""
        parameters = Dict()

        parameters.variables = {opt.name: opt.parameters for opt in self.variables}
        parameters.linear_constraints = self.linear_constraints
        parameters.linear_equality_constraints = self.linear_equality_constraints

        return parameters

    def check_linear_constraints_transforms(self) -> bool:
        """
        Check that variables used in linear constraints only use linear transforms.

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
                if not var.transformer.is_linear:
                    flag = False
                    warnings.warn(
                        f"'{var.name}' uses non-linear transform and is used in "
                        f"the linear constraint: {constr}."
                        "Consider using linear transforms for these variables "
                        "or specify the constraints as non-linear constraints."
                    )

        return flag

    def check_linear_constraints_dependency(self) -> bool:
        """
        Check that variables used in linear constraints are independent.

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

    def check_config(self, ignore_linear_constraints: Optional[bool] = False) -> bool:
        """
        Check if the OptimizationProblem is configured correctly.

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

        if (
            self.n_linear_constraints + self.n_linear_equality_constraints > 0
            and not ignore_linear_constraints
        ):
            if not self.check_linear_constraints_transforms():
                flag = False

            if not self.check_linear_constraints_dependency():
                flag = False

        return flag

    @untransforms
    @gets_dependent_values
    def check_individual(
        self,
        x: npt.ArrayLike,
        cv_bounds_tol: Optional[float | np.ndarray] = 0.0,
        cv_lincon_tol: Optional[float | np.ndarray] = 0.0,
        cv_lineqcon_tol: Optional[float | np.ndarray] = 0.0,
        check_nonlinear_constraints: bool = False,
        cv_nonlincon_tol: Optional[float | np.ndarray] = 0.0,
        silent: bool = False,
    ) -> bool:
        """
        Check if individual is valid.

        Parameters
        ----------
        x : npt.ArrayLike
            Value of the optimization variables in untransformed space.
        cv_bounds_tol : float or np.ndarray, optional
            Tolerance for checking the bound constraints. Default is 0.0.
        cv_lincon_tol : float or np.ndarray, optional
            Tolerance for checking the linear inequality constraints. Default is 0.0.
        cv_lineqcon_tol : float or np.ndarray, optional
            Tolerance for checking the linear equality constraints. Default is 0.0.
        check_nonlinear_constraints : bool, optional
            If True, also check nonlinear constraints. Note that depending on the
            nonlinear constraints, this can be an expensive operation.
            The default is False.
        cv_nonlincon_tol : float or np.ndarray, optional
            Tolerance for checking the nonlinear constraints. Default is 0.0.
        silent : bool, optional
            If True, suppress warnings. The default is False.

        Returns
        -------
        bool
            True if the individual is valid, False otherwise.
        """
        flag = True

        if not self.check_bounds(x, cv_bounds_tol):
            if not silent:
                warnings.warn("Individual does not satisfy bounds.")
            flag = False

        if not self.check_linear_constraints(x, cv_lincon_tol):
            if not silent:
                warnings.warn("Individual does not satisfy linear constraints.")
            flag = False

        if not self.check_linear_equality_constraints(x, cv_lineqcon_tol):
            if not silent:
                warnings.warn(
                    "Individual does not satisfy linear equality constraints."
                )
            flag = False

        if check_nonlinear_constraints:
            if not self.check_nonlinear_constraints(x, cv_nonlincon_tol):
                flag = False
                if not silent:
                    warnings.warn("Individual does not satisfy nonlinear constraints.")

        return flag

    def __str__(self) -> str:
        """str: Name of the optimization problem."""
        return self.name


class OptimizationVariable:
    """
    Class for setting the values for the optimization variables.

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
    transformer : TransformerBase
        Transformation function for parameter normalization.
    indices : int, or slice
        Indices for variables that modify an entry of a parameter array.
        If None, variable is assumed to be index independent.
    significant_digits : int, optional
        Number of significant figures to which variable can be rounded.
        If None, variable is not rounded. The default is None.
    pre_processing : tp.Callable, optional
        Additional step to process the value before setting it. This function must
        accept a single argument (the value) and return the processed value.

    Raises
    ------
    CADETProcessError
        If the attribute is not valid.
    ValueError
        If the lower bound is larger than or equal to the upper bound.
    """

    def __init__(
        self,
        name: str,
        evaluation_objects: Optional[list] = None,
        parameter_path: Optional[str] = None,
        lb: float = -math.inf,
        ub: float = math.inf,
        transform: Optional[TransformerBase] = None,
        indices: Optional[int] = None,
        significant_digits: Optional[int] = None,
        pre_processing: Optional[tp.Callable] = None,
    ) -> None:
        """Initialize Optimization Variable."""
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
            transformer = NullTransformer(lb, ub)
        else:
            if np.isinf(lb) or np.isinf(ub):
                raise CADETProcessError("Transform requires bound constraints.")
            if transform == "auto":
                transformer = AutoTransformer(lb, ub)
            elif transform == "linear":
                transformer = NormLinearTransformer(lb, ub)
            elif transform == "log":
                transformer = NormLogTransformer(lb, ub)
            else:
                raise ValueError("Unknown transform")

        self._transformer = transformer
        self.significant_digits = significant_digits

        self._dependencies = []
        self._dependency_transform = None

        self.pre_processing = pre_processing

    @property
    def parameter_path(self) -> str:
        """str: Path of the evaluation_object parameter in dot notation."""
        return self._parameter_path

    @parameter_path.setter
    def parameter_path(self, parameter_path: str) -> None:
        if parameter_path is not None:
            for eval_obj in self.evaluation_objects:
                parameters = eval_obj.parameters.to_dict()  # Workaround addict issue #136
                if not check_nested(parameters, parameter_path):
                    raise CADETProcessError("Not a valid Optimization variable")
        self._parameter_path = parameter_path

    @property
    def parameter_sequence(self) -> tuple:
        """tuple: Tuple of parameters path elements."""
        return tuple(self.parameter_path.split("."))

    @property
    def performer(self) -> str:
        """str: The name of the performer of the variable."""
        if len(self.parameter_sequence) == 1:
            return self.parameter_sequence[0]
        else:
            return ".".join(self.parameter_sequence[:-1])

    def _performer_obj(self, evaluation_object: Any) -> Any:
        if len(self.parameter_sequence) == 1:
            return evaluation_object

        return get_nested_attribute(evaluation_object, self.performer)

    def _parameter_descriptor(self, evaluation_object: Any) -> Any:
        performer_obj = self._performer_obj(evaluation_object)
        performer_class = type(performer_obj)
        try:
            descriptor = getattr(performer_class, self.parameter_sequence[-1])
        except AttributeError:
            return None

        if not isinstance(descriptor, ParameterBase):
            return None

        return descriptor

    def _parameter_type(self, evaluation_object: Any) -> type:
        """type: Type of the parameter."""
        parameter_descriptor = self._parameter_descriptor(evaluation_object)
        if isinstance(parameter_descriptor, Typed):
            return parameter_descriptor.ty

        current_value = self._current_value(evaluation_object)
        if current_value is None:
            raise CADETProcessError(
                "Parameter is not initialized. Cannot determine parameter type."
            )

        return type(current_value)

    def _get_parameter_shape(
        self, evaluation_object: object
    ) -> list[tuple[int, ...]] | tuple[int, ...]:
        parameter_descriptor = self._parameter_descriptor(evaluation_object)

        if isinstance(parameter_descriptor, (Float, Integer, Bool)):
            return ()

        if isinstance(parameter_descriptor, Sized):
            performer_obj = self._performer_obj(evaluation_object)
            shape = parameter_descriptor.get_expected_size(performer_obj)

            if not isinstance(shape, tuple):
                shape = (shape,)

            return shape

        current_value = self._current_value(evaluation_object)
        if current_value is None:
            raise CADETProcessError(
                "Parameter is not initialized. Cannot determine parameter shape."
            )

        shape = get_inhomogeneous_shape(current_value)

        return shape

    def _current_value(self, evaluation_object: Any) -> Any:
        parameter_descriptor = self._parameter_descriptor(evaluation_object)

        if parameter_descriptor is not None:
            performer_obj = self._performer_obj(evaluation_object)
            return getattr(performer_obj, self.parameter_sequence[-1])
        else:
            return copy.copy(
                get_nested_value(evaluation_object.parameters, self.parameter_path)
            )

    def _is_sized(self, evaluation_object: Any) -> bool:
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

    def _is_polynomial(self, evaluation_object: Any) -> bool:
        """bool: True if descriptor is instance of NdPolynomial. False otherwise."""
        polynomial_parameters = evaluation_object.polynomial_parameters.to_dict()  # Workaround addict issue #136 # noqa: E501
        return check_nested(polynomial_parameters, self.parameter_path)

    @property
    def indices(self) -> list[int]:
        """list[int]: List of parameter indices that are modified by optimization variable."""
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
    def indices(self, indices: list | int | None) -> None:
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
    def is_index_specific(self) -> bool:
        """bool: True if variable modifies entry of a parameter array, False otherwise."""
        if self.indices is not None:
            return True
        else:
            return False

    @property
    def full_indices(self) -> list[int]:
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
                    indices = [(ind[0],) + i for i in indices]
                    full_indices.append(indices)

        return full_indices

    @property
    def n_indices(self) -> int:
        """int: Number of (full) indices."""
        if self.indices is not None:
            return len(self.full_indices)
        else:
            return 0

    @property
    def transformer(self) -> TransformerBase:
        """TransformerBase: The variable transformer instance."""
        return self._transformer

    def transform(self, x: float, *args: Any, **kwargs: Any) -> float:
        """
        Apply the transformation to the input.

        Parameters
        ----------
        x : float
            The input data to be transformed.
        *args : Any
            Additional positional arguments passed to the transformer's `transform` method.
        **kwargs : Any
            Additional keyword arguments passed to the transformer's `transform` method.

        Returns
        -------
        float
            The transformed data.
        """
        return self.transformer.transform(x, *args, **kwargs)

    def untransform(self, x: float, *args: Any, **kwargs: Any) -> float:
        """
        Apply the inverse transformation to the input.

        Parameters
        ----------
        x : float
            The input data to be untransformed.
        *args : Any
            Additional positional arguments passed to the transformer's `untransform` method.
        **kwargs : Any
            Additional keyword arguments passed to the transformer's `untransform` method.

        Returns
        -------
        float
            The untransformed data.
        """
        return self.transformer.untransform(x, *args, **kwargs)

    def add_dependency(self, dependencies: list, transform: tp.Callable) -> None:
        """
        Add dependency of Variable on other Variables.

        Parameters
        ----------
        dependencies : list
            List of OptimizationVariables to be added as dependencies.
        transform: tp.Callable
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
    def dependencies(self) -> list:
        """list: Independent variables on which the Variable depends."""
        return self._dependencies

    @property
    def is_independent(self) -> bool:
        """bool: True if Variable is independent, False otherwise."""
        return len(self.dependencies) == 0

    @property
    def value(self) -> float:
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
            value = self.dependency_transform(*dependencies)
            value = round_to_significant_digits(value, self.significant_digits)
            return value

    @value.setter
    def value(self, value: Any) -> None:
        if not self.is_independent:
            raise CADETProcessError("Cannot set value for dependent variables")

        self.set_value(value)

    def set_value(self, value: Any) -> None:
        """Set value to evaluation_objects."""
        if not np.isscalar(value):
            raise TypeError("Expected scalar value")

        value = round_to_significant_digits(
            value,
            digits=self.significant_digits,
        )

        if value < self.lb:
            raise ValueError("Exceeds lower bound")
        if value > self.ub:
            raise ValueError("Exceeds upper bound")

        self._value = value

        if self.evaluation_objects is None:
            return

        indicies = self.indices

        for i_eval, eval_obj in enumerate(self.evaluation_objects):
            is_polynomial = self._is_polynomial(eval_obj)

            if (
                indicies[i_eval] is None
                or self._indices[i_eval] is None
                and is_polynomial
            ):
                new_value = value
            else:
                eval_ind = indicies[i_eval]
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
                    eval_ind = [(eval_ind,)]

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
                if (
                    parameter_type is not NumpyProxyArray
                    and not isinstance(new_value, parameter_type)
                ):
                    new_value = parameter_type(new_value.tolist())

            # Set the value:
            self._set_value_in_evaluation_object(eval_obj, new_value)

    def _set_value_in_evaluation_object(self, evaluation_object: object, value: Any) -> None:
        """Set the value to the evaluation object."""
        if self.pre_processing is not None:
            value = self.pre_processing(value)
        parameter_descriptor = self._parameter_descriptor(evaluation_object)
        if parameter_descriptor is not None:
            performer_obj = self._performer_obj(evaluation_object)
            setattr(performer_obj, self.parameter_sequence[-1], value)
        else:
            parameters = generate_nested_dict(self.parameter_sequence, value)
            evaluation_object.parameters = parameters

    @property
    def transformed_bounds(self) -> list:
        """list: Transformed bounds of the parameter."""
        return [self.transform(self.lb), self.transform(self.ub)]

    def __repr__(self) -> str:
        """str: String representation."""
        if self.evaluation_objects is not None:
            string = (
                f"{self.__class__.__name__}"
                + f"(name={self.name}, "
                + f"evaluation_objects="
                f"{[str(obj) for obj in self.evaluation_objects]}, "
                + f"parameter_path="
                f"{self.parameter_path}, lb={self.lb}, ub={self.ub})"
            )
        else:
            string = (
                f"{self.__class__.__name__}"
                + f"(name={self.name}, lb={self.lb}, ub={self.ub})"
            )
        return string


class Evaluator(Structure):
    """Wrapper class to call evaluator."""

    evaluator = Callable()
    args = Tuple()
    kwargs = Dict()

    def __init__(
        self,
        evaluator: tp.Callable,
        name: str,
        args: Any = None,
        kwargs: Any = None,
    ) -> None:
        self.evaluator = evaluator
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.id = uuid.uuid4()

    def __call__(self, request: Any) -> Any:
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

    def __str__(self) -> str:
        """str: The name of the evaluator."""
        return self.name


class Metric(Structure):
    """Wrapper class to evaluate metrics (e.g. objective/nonlincon) functions."""

    func = Callable()
    name = String()
    n_metrics = RangedInteger(lb=1)
    bad_metrics = SizedNdArray(size="n_metrics", default=np.inf)

    def __init__(
        self,
        func: tp.Callable,
        name: str,
        n_metrics: int = 1,
        bad_metrics: float | np.ndarray = np.inf,
        evaluation_objects: Optional[object] = None,
        evaluators: Optional[list[Evaluator]] = None,
        labels: list[str] = None,
        args: Any = None,
        kwargs: Any = None,
    ) -> None:
        self.func = func
        self.name = name

        self.n_metrics = n_metrics

        if np.isscalar(bad_metrics):
            bad_metrics = np.tile(bad_metrics, n_metrics)
        self.bad_metrics = bad_metrics

        self.evaluation_objects = evaluation_objects
        self.evaluators = evaluators

        self.labels = labels

        self.args = args
        self.kwargs = kwargs

        self.id = uuid.uuid4()

    @property
    def n_total_metrics(self) -> int:
        """int: Total number of objectives."""
        n_eval_objects = len(self.evaluation_objects) if self.evaluation_objects else 1
        return n_eval_objects * self.n_metrics

    @property
    def labels(self) -> list[str]:
        """list: List of metric labels."""
        if self._labels is not None:
            return self._labels

        try:
            labels = self.func.labels
        except AttributeError:
            labels = [f"{self.name}"]
            if self.n_metrics > 1:
                labels = [f"{self.name}_{i}" for i in range(self.n_metrics)]

        if len(self.evaluation_objects) > 1:
            labels = [
                f"{eval_obj}_{label}"
                for label in labels
                for eval_obj in self.evaluation_objects
            ]
        return labels

    @labels.setter
    def labels(self, labels: list[str]) -> None:
        if labels is not None:
            if len(labels) != self.n_metrics:
                raise CADETProcessError(f"Expected {self.n_metrics} labels.")

        self._labels = labels

    def __call__(self, request: Any) -> np.ndarray:
        """
        Evaluate the metric function with the given request and predefined arguments.

        Parameters
        ----------
        request
            The input to the metric function, typically representing the current state
            or results of intermediate steps in the optimization process.

        Returns
        -------
        ndarray
            The evaluated metric values, returned as a NumPy array.
        """
        if self.args is None:
            args = ()
        else:
            args = self.args

        if self.kwargs is None:
            kwargs = {}
        else:
            kwargs = self.kwargs

        f = self.func(request, *args, **kwargs)

        return np.array(f, ndmin=1)

    evaluate = __call__

    def __str__(self) -> str:
        """Name of the metric."""
        return self.name


class Objective(Metric):
    """Wrapper class to evaluate objective functions."""

    objective = Metric.func
    n_objectives = Metric.n_metrics
    minimize = Bool(default=True)

    def __init__(
        self, *args: Any, n_objectives: int = 1, minimize: bool = True, **kwargs: Any
    ) -> None:
        self.minimize = minimize

        super().__init__(*args, n_metrics=n_objectives, **kwargs)


class NonlinearConstraint(Metric):
    """Wrapper class to evaluate nonlinear constraint functions."""

    nonlinear_constraint = Metric.func
    n_nonlinear_constraints = Metric.n_metrics
    comparison_operator = Switch(valid=["le", "ge"], default="le")

    def __init__(
        self,
        *args: Any,
        n_nonlinear_constraints: int = 1,
        bounds: int = 0,
        comparison_operator: str = "le",
        **kwargs: Any,
    ) -> None:
        self.bounds = bounds
        self.comparison_operator = comparison_operator

        super().__init__(*args, n_metrics=n_nonlinear_constraints, **kwargs)


class Callback(Structure):
    """
    Wrapper class to evaluate callbacks.

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
        callback: tp.Callable,
        name: str,
        evaluation_objects: Optional[list[object]] = None,
        evaluators: Optional[list[Evaluator]] = None,
        frequency: int = 10,
        callbacks_dir: Optional[str] = None,
        keep_progress: bool = False,
        args: Any = None,
        kwargs: Any = None,
    ) -> None:
        self.callback = callback
        self.name = name

        self.evaluation_objects = evaluation_objects
        self.evaluators = evaluators

        self.frequency = frequency

        if callbacks_dir is not None:
            callbacks_dir = Path(callbacks_dir)
            callbacks_dir.mkdir(exist_ok=True, parents=True)
        self.callbacks_dir = callbacks_dir
        self._callbacks_dir = callbacks_dir

        self.keep_progress = keep_progress

        self.args = args
        self.kwargs = kwargs

        self.id = uuid.uuid4()

    def cleanup(self, callbacks_dir: str, current_iteration: int) -> None:
        if (
            not current_iteration % self.frequency == 0
            or current_iteration <= self.frequency
        ):
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

    def __call__(self, request: Any, evaluation_object: Any) -> Any:
        if self.args is None:
            args = ()
        else:
            args = self.args

        if self.kwargs is None:
            kwargs = {}
        else:
            kwargs = self.kwargs

        signature = inspect.signature(self.callback).parameters
        if "current_iteration" in signature:
            kwargs["current_iteration"] = self._current_iteration
        if "individual" in signature:
            kwargs["individual"] = self._ind
        if "evaluation_object" in signature:
            kwargs["evaluation_object"] = evaluation_object
        if "callbacks_dir" in signature:
            if self.callbacks_dir is not None:
                callbacks_dir = self.callbacks_dir
            else:
                callbacks_dir = self._callbacks_dir
            if callbacks_dir is not None:
                kwargs["callbacks_dir"] = callbacks_dir

        self.callback(request, *args, **kwargs)

    evaluate = __call__

    def __str__(self) -> str:
        """str: Name of the callback."""
        return self.name


class MetaScore(Metric):
    """Wrapper class to evaluate meta scores."""

    meta_score = Metric.func
    n_meta_scores = Metric.n_metrics
    minimize = Bool(default=True)

    def __init__(
        self, *args: Any, n_meta_scores: int = 1, minimize: bool = True, **kwargs: Any
    ) -> None:
        self.minimize = minimize

        super().__init__(*args, n_metrics=n_meta_scores, **kwargs)


class MultiCriteriaDecisionFunction(Structure):
    """Wrapper class to evaluate multi-criteria decision functions."""

    decision_function = Callable()
    name = String()

    def __init__(self, decision_function: tp.Callable, name: str) -> None:
        self.decision_function = decision_function
        self.name = name

        self.id = uuid.uuid4()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.decision_function(*args, **kwargs)

    evaluate = __call__

    def __str__(self) -> str:
        """str: Name of the multi-criteria decision function."""
        return self.name


def approximate_jac(
    xk: npt.ArrayLike,
    f: tp.Callable,
    epsilon: npt.ArrayLike,
    args: Any = (),
    **kwargs: Any,
) -> np.ndarray:
    r"""
    Finite-difference approximation of the jacobian of a vector function.

    Parameters
    ----------
    xk : array_like
        The coordinate vector at which to determine the jacobian of `f`.
    f : tp.Callable
        The function of which to determine the jacobian (partial derivatives).
        Should take `xk` as first argument, other arguments to `f` can be
        supplied in ``*args``.  Should return a vector, the values of the
        function at `xk`.
    epsilon : array_like
        Increment to `xk` to use for determining the function jacobian.
        If a scalar, uses the same finite difference delta for all partial
        derivatives.  If an array, should contain one value per element of
        `xk`.
    *args : args, optional
        Any other arguments that are to be passed to `f`.
    **kwargs
        Additional arguments that are to be passed to `f`.

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
        f_k = np.array(f(*((xk + d,) + args), **kwargs))
        jac[:, k] = (f_k - f0) / d[k]
        ei[k] = 0.0

    return jac
