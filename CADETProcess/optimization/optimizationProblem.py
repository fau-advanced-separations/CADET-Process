import copy
import math
import random
import warnings
import time

from addict import Dict
import numpy as np
from scipy import optimize
import hopsy 
import multiprocess
import pathos

from CADETProcess import CADETProcessError
from CADETProcess import log

from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import String
from CADETProcess.dataStructure import frozen_attributes

from CADETProcess.common import approximate_jac
from CADETProcess.common import get_bad_performance

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
    eval_dict : dict
        Database for storing evaluated individuals
    """
    name = String()

    def __init__(
            self, evaluation_object, evaluator=None, name=None, save_log=False
        ):
        if evaluator is not None:
            self.evaluator = evaluator
        else:
            self._evaluator = None

        self.evaluation_object = evaluation_object

        if name is None:
            try:
                name = evaluation_object.name + '_optimization'
            except AttributeError:
                msg = '__init__() missing 1 required positional argument: \'name\''
                raise TypeError(msg)
        self.name = name

        if save_log:
            self.logger = log.get_logger(self.name, log_directory=self.name)
        else:
            self.logger = log.get_logger(self.name)

        self._variables = []
        self._objectives = []
        self._nonlinear_constraints = []
        self._linear_constraints = []
        self._linear_equality_constraints = []
        self._x0 = None
        
    @property
    def evaluation_object(self):
        """obj : Object that contains all parameters that are optimized.

        Returns
        -------
        evaluation_object : obj
            Object to be evaluated during optimization.

        See also
        --------
        OptimizatonVariable
        Evaluator
        evaluate
        Performance
        objectives
        nonlinear_constraints
        """
        return self._evaluation_object

    @evaluation_object.setter
    def evaluation_object(self, evaluation_object):
        self._evaluation_object = evaluation_object

    @property
    def evaluator(self):
        """Object that performs evaluation object during optimization.

        The evaluator has to implement an evaluate method that returns a
        Performance object when called with the evaluation_object. The
        performance Object is required for calculating the objective function
        and nonlinear constraint function.

        Parameters
        ----------
        evaluator : obj
            Object to be evaluated during optimization.

        Raises
        ------
        CADETProcessError
            If evaluation object does not implement evaluation method.

        See also
        --------
        evaluation_object
        evaluate
        Performance
        objectives
        nonlinear_constraints
        """
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator):
        try:
            getattr(evaluator, 'evaluate')
        except TypeError:
            raise TypeError('Evaluator has to implement evaluate method')

        self._evaluator = evaluator

    @property
    def variables(self):
        """list: List of all optimization variables.
        """
        return self._variables

    @property
    def variables_dict(self):
        """Returns a dictionary with all events in a process.

        Returns
        -------
        events_dict : dict
            Dictionary with all events and durations, indexed by Event.name.
        """
        return {var.name: var for var in self._variables}

    @property
    def n_variables(self):
        """int: Number of optimization variables.
        """
        return len(self.variables)

    def add_variable(self, parameter_path, name=None, lb=-math.inf, ub=math.inf,
                     component_index=None):
        """Add optimization variable to the OptimizationProblem.

        The function encapsulates the creation of OoptimizationVariable objects
        in order to prevent invalid OptimizationVariables.

        Parameters
        ----------
        attr : str
            Attribute of the variable to be added.
        name : str
            Name of the variable to be added, default value is None.
        lb : float
            Lower bound of the variable value.
        ub : float
            Upper bound of the variable value.
        component_index : int
            Index for compnent specific variables

        Raises
        ------
        CADETProcessError
            If the Variable already exists in the dictionary.

        See also
        --------
        evaluation_object
        OptimizationVariable
        remove_variable
        """
        if name is None:
            name = parameter_path

        if name in self.variables_dict:
            raise CADETProcessError("Variable already exists")

        var = OptimizationVariable(name, self.evaluation_object, parameter_path,
                 lb=lb, ub=ub, component_index=component_index)

        self._variables.append(var)
        super().__setattr__(name, var)


    def remove_variable(self, var_name):
        """Removes variables from the OptimizationProblem.

        Parameters
        ----------
        var_name : str
            Name of the variable to be removed from list of variables.

        Raises
        ------
        CADETProcessError
            If required variable does not exist.

        See also
        --------
        add_variable
        """
        try:
            var = self.variables_dict[var_name]
        except KeyError:
            raise CADETProcessError("Variable does not exist")

        self._variables.remove(var)
        self.__dict__.pop(var_name)


    def set_variables(self, x, make_copy=False):
        """Sets the values from the x-vector to the OptimizationVariables.

        Parameters
        ----------
         x : array_like
            Value of the optimization variables

        make_copy : Bool
            If True, a copy of the evaluation_object attribute is made on which
            the values are set. Otherwise, the values are set on the attribute.

        Returns
        -------
        evaluation_object : object
            Returns copy of evaluation object if make_copy is True, else returns
            the attribute evaluation_object with the values set.

        Raises
        ------
        ValueError
            If value of variable exceeds bounds

        See also
        --------
        OptimizationVariable
        evaluate
        """
        if len(x) != self.n_variables:
            raise CADETProcessError('Expected {} variables'.format(self.n_variables))
        if make_copy:
            evaluation_object = copy.deepcopy(self.evaluation_object)
        else:
            evaluation_object = self.evaluation_object

        for variable, value in zip(self.variables, x):
            if value < variable.lb:
                raise ValueError("Exceeds lower bound")
            if value > variable.ub:
                raise ValueError("Exceeds upper bound")

            if variable.component_index is not None:
                value_list = get_nested_value(
                    evaluation_object.parameters, variable.parameter_path
                )
                value_list[variable.component_index] = value
                parameters = generate_nested_dict(variable.parameter_path, value_list)
            else:
                parameters = generate_nested_dict(variable.parameter_path, value)
            evaluation_object.parameters = parameters

        return evaluation_object

    def evaluate(self, x, make_copy=False, cache=None, force=False):
        """Evaluates the evaluation object at x.

        For performance reasons, the function first checks if the function has
        already been evaluated at x and returns the results. If force is True,
        the lookup is ignored and the function is evaluated regularly.

        Parameters
        ----------
         x : array_like
            Value of the optimization variables
        force : bool
            If True, reevaluation of previously evaluated values is enforced.

        Returns
        -------
        performance : Performance
            Performance object from fractionation.
        """
        x = np.array(x)
        
        # Try to get cached results
        if cache is not None and not force:
            try:
                performance = cache[tuple(x.tolist())]
                return performance
            except KeyError:
                pass

        # Get evaluation_object
        evaluation_object = self.set_variables(x, make_copy)

        # Set bad results if constraints are not met
        if not self.check_linear_constraints(x):
            self.logger.warn(
                f'Linear constraints not met at {x}. Returning bad performance.\
                cycle time: {evaluation_object.process_meta.cycle_time}'
            )
            performance = get_bad_performance(evaluation_object.n_comp)
        # Pass evaluation_object to evaluator and evaluate
        elif self.evaluator is not None:
            try:
                performance = self.evaluator.evaluate(evaluation_object)
            except CADETProcessError:
                self.logger.warn('Evaluation failed. Returning bad performance')
                performance = get_bad_performance(evaluation_object.n_comp)
        else:
            performance = evaluation_object.performance

        if cache is not None:
            cache[tuple(x.tolist())] = performance

        self.logger.info('{} evaluated at x={} yielded {}'.format(
                self.evaluation_object.name, 
                tuple(x.tolist()), 
                performance.to_dict())
        )

        return performance
    
    def evaluate_population(self, population, n_cores=0):
        manager = multiprocess.Manager()
        cache = manager.dict()

        eval_fun = lambda ind: self.evaluate(ind, make_copy=True, cache=cache)

        if n_cores == 1:
           for ind in population:
               try:
                   eval_fun(ind)
               except CADETProcessError:
                   print(ind)
        else:
            if n_cores == 0:
                n_cores = None
            with pathos.multiprocessing.ProcessPool(ncpus=n_cores) as pool:
                pool.map(eval_fun, population)
                
        return cache
        

    @property
    def objectives(self):
        return self._objectives
    
    @property
    def n_objectives(self):
        return len(self.objectives)
    
    def add_objective(self, objective_fun):
        """Add objective function to optimization problem.

        Parameters
        ----------
        objective_fun : function
            Objective function. Funtion should take a Performance object as
            argument and return a scalar value.

        Raises
        ------
        TypeError
            If objective_fun is not callable.
        """
        if not callable(objective_fun):
            raise TypeError("Expected callable object")

        self._objectives.append(objective_fun)
        
    def evaluate_objectives(self, x, *args, **kwargs):
        """Function that evaluates at x and computes objective function.

        This function is usually presented to the optimization solver. The
        actual evaluation of the evaluation object is peformed by the evaluate
        function which also logs the results.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

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
        performance = self.evaluate(x, *args, **kwargs)
        f = np.array([obj(performance) for obj in self.objectives])
        return f

    def objective_gradient(self, x, dx=0.1):
        """Calculate the gradient of the objective functions at point x.

        Gradient is approximated using finite differences.

        Parameters
        ----------
        x : ndarray
            Value of the optimization variables.
        dx : float
            Increment to x to use for determining the function gradient.

        Returns
        -------
        grad: list
            The partial derivatives of the objective functions at point x.

        See Also
        --------
        OptimizationProblem
        objectives
        """
        dx = [dx]*len(x)
        grad = [optimize.approx_fprime(x, obj, dx) for obj in self.objectives]

        return grad

    @property
    def nonlinear_constraints(self):
        return self._nonlinear_constraints
    
    @property
    def n_nonlinear_constraints(self):
        return len(self.nonlinear_constraints)
    
    def add_nonlinear_constraint(self, nonlinear_constraint_fun):
        """Add nonlinear constraint function to optimization problem.

        Parameters
        ----------
        nonlinear_constraint_fun: function
            Nonlinear constraint function. Funtion should take a Performance 
            object as argument and return a scalar value or an array.

        Raises
        ------
        TypeError
            If nonlinear_constraint_fun is not callable.
        """
        if not callable(nonlinear_constraint_fun):
            raise TypeError("Expected callable object")

        self._nonlinear_constraints.append(nonlinear_constraint_fun)

    def evaluate_nonlinear_constraints(self, x, *args, **kwargs):
        """Function that evaluates at x and computes nonlinear constraitns.

        This function is usually presented to the optimization solver. The
        actual evaluation of the evaluation object is peformed by the evaluate
        function which also logs the results.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        c : list
            Value(s) of the constraint functions at point x.

        See also
        --------
        nonlinear_constraints
        evaluate
        evaluate_objectives
        """
        performance = self.evaluate(x, *args, **kwargs)
        c = np.array(
            [constr(performance) for constr in self.nonlinear_constraints]
        )
        return c

    def check_nonlinear_constraints(self, x):
        """Checks if all nonlinear constraints are kept.

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
        for constr in c:
            if np.any(constr > 0):
                return False
        return True

    def nonlinear_constraint_jacobian(self, x, dx=1e-3):
        """Return the jacobian matrix of the nonlinear constraints at point x.

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

        See also
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
    def lower_bounds(self):
        """list : List of the lower bounds of all OptimizationVariables.

        See also
        --------
        upper_bounds
        """
        return [var.lb for var in self.variables]

    @property
    def upper_bounds(self):
        """list : List of the upper bounds of all OptimizationVariables.

        See also
        --------
        upper_bounds
        """
        return [var.ub for var in self.variables]

    def check_bounds(self, x):
        """Checks if all bound constraints are kept.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables

        First sets a local variable named flag to True. Then checks if the
        values of the list x are exceeding the lower and upper bounds and sets
        the value of the flag to False. For this the list x is converted into
        an np.array.

        Returns
        -------
        flag : Bool
            Returns True, if the values of the list x are in the defined bounds.
            Returns False if the values of the list x are violating the bound
            constraints.
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

    def add_linear_constraint(self, opt_vars, factors, b=0):
        """Add linear inequality constraints.

        Parameters
        -----------
        opt_vars : list of strings
            Names of the OptimizationVariable to be added.
        factors : list of  integers
            Factors for OptimizationVariables.
        b : float, optional
            Constraint of inequality constraint; default set to zero.

        Raises
        -------
        CADETProcessError
            If optimization variables do not exist.
            If length of factors does not match length of optimization variables.

        See also
        ---------
        linear_constraints
        remove_linear_constraint
        linear_equality_constraints
        """
        if not all(var in self.variables_dict for var in opt_vars):
            raise CADETProcessError('Variable not in variables')

        if len(factors) != len(opt_vars):
            raise CADETProcessError('Factors length does not match variables')

        lincon = dict()
        lincon['opt_vars'] = opt_vars
        lincon['factors'] = factors
        lincon['b'] = b

        self._linear_constraints.append(lincon)

    def remove_linear_constraint(self, index):
        """Removes linear inequality constraint.

        Parameters
        ----------
        index : int
            Index of the linear inequality constraint to be removed.
        
        See also
        --------
        add_linear_equality_constraint
        linear_equality_constraint
        """
        del(self._linear_constraints[index])
        

    @property
    def A(self):
        """np.ndarray: Matrix form of linear inequality constraints.

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
                A[lincon_index, index] = lincon['factors'][var_index]

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
        x = np.array(x)
        return self.A.dot(x) - self.b

    def check_linear_constraints(self, x):
        """Check if linear inequality constraints are met at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        flag : bool
            Returns True if linear inequality constraints are met. False otherwise.

        See also:
        ---------
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
        """int: number of linear equality constraints
        """
        return len(self.linear_constraints)

    def add_linear_equality_constraint(self, opt_vars, factors, beq=0):
        """Add linear equality constraints.

        Parameters
        -----------
        opt_vars : list of strings
            Names of the OptimizationVariable to be added.
        factors : list of  integers
            Factors for OptimizationVariables.
        b_eq : float, optional
            Constraint of equality constraint; default set to zero.

        Raises
        -------
        CADETProcessError
            If optimization variables do not exist.
            If length of factors does not match length of optimization variables.

        See also
        --------
        linear_equality_constraints
        remove_linear_equality_constraint
        linear_constraints
        """
        if not all(var in self.variables for var in opt_vars):
            return CADETProcessError('Variables not in variables')

        if len(factors) != len(opt_vars):
            return CADETProcessError('Factors length does not match variables')

        lineqcon = dict()
        lineqcon['opt_vars'] = opt_vars
        lineqcon['factors'] = factors
        lineqcon['beq'] = beq

        self._linear_equality_constraints.append(lineqcon)

    def remove_linear_equality_constraint(self, index):
        """Removes at given index the added linear equality conctraint.

        Parameters
        ----------
        index : int
            Index of the linear equality constraint to be removed.

        See also
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
                Aeq[lineqcon_index, index] = lineqcon.factors[var_index]

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
        x = np.array(x)
        return self.Aeq.dot(x) - self.beq

    def check_linear_equality_constraints(self, x):
        """Check if linear equality constraints are met at point x.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        flag : bool
            Returns True if linear equality constraints are met. False otherwise.
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
            If the initial value does not match length of optimization variables
        """
        return self._x0

    @x0.setter
    def x0(self, x0):
        if not len(x0) == len(self.variables):
            raise CADETProcessError(
                "Starting value must be given for all variables"
            )
        self._x0 = x0


    def create_initial_values(self, n_samples=1, method='random', seed=None):
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

        Returns
        -------
        init : ndarray
            Initial values for starting the optimization.
        """
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
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
                
            starting_points = [hopsy.compute_chebyshev_center(problem)]
            run = hopsy.Run(
                problem,
                starting_points=starting_points
                )
            if seed is None:
                seed = random.randint(0,255)
            run.random_seed = seed
            run.sample(n_samples)
                    
        states = np.array(run.data.states[0])
        
        if n_samples == 1:
            if method == 'chebyshev':
                states = hopsy.compute_chebyshev_center(problem)
            elif method == 'random':
                states = states[0]
            else:
                raise CADETProcessError("Unexpected method.")
        
        return states

    @property
    def parameters(self):
        parameters = Dict()

        parameters.variables = {opt.name: opt.parameters
                                for opt in self.variables}
        parameters.linear_constraints = self.linear_constraints

        return parameters


    def __str__(self):
        return self.name

from CADETProcess.dataStructure import (
    check_nested, generate_nested_dict, get_nested_value
)
class OptimizationVariable():
    """Class for setting the values for the optimization variables.

    Defines the attributes for optimization variables for creating an
    OptimizationVariable. Tries to get the attr of the evaluation_object.
    Raises a CADETProcessErrorif the attribute to be set is not valid.

    Attributes
    -----------
    evaluation_object :  obj
        Object to be evaluated, can be a wrapped process or a fractionation
        object.
    name : str
        Name of the optimization variable.
    attribute : str
        Path of the optimization variable in the evaluation_object's parameters.
    value : float
        Value of the optimization variable.
    lb : float
        Lower bound of the variable.
    ub : float
        upper bound of the variable.
    component_index : int
        Index for compnent specific variables

    Raises
    ------
    CADETProcessError
        If the attribute is not valid.
    """
    _parameters = ['lb', 'ub', 'component_index']

    def __init__(self, name, evaluation_object, parameter_path,
                 lb=-math.inf, ub=math.inf, value=0.0, component_index=None):

        self.name = name

        self.evaluation_object = evaluation_object
        self.parameter_path = parameter_path
        self.component_index = component_index

        self.lb = lb
        self.ub = ub

    @property
    def parameter_path(self):
        return self._parameter_path

    @parameter_path.setter
    def parameter_path(self, parameter_path):
        if not check_nested(self.evaluation_object.parameters, parameter_path):
            raise CADETProcessError('Not a valid Optimization variable')
        self._parameter_path = parameter_path

    @property
    def parameter_sequence(self):
        """tuple: Tuple of parameters path elements.
        """
        return tuple(self.parameter_path.split('.'))

    @property
    def component_index(self):
        return self._component_index

    @component_index.setter
    def component_index(self, component_index):
        if component_index is not None:
            parameter = get_nested_value(
                    self.evaluation_object.parameters, self.parameter_sequence)

            if component_index > len(parameter)-1:
                raise CADETProcessError('Index exceeds components')
        self._component_index = component_index

    @property
    def parameters(self):
        """Returns the parameters in a list.

        Returns
        -------
        parameters : dict
            list with all the parameters.
        """
        return Dict({param: getattr(self, param) for param in self._parameters})

    def __repr__(self):
        return '{}(name={}, evaluation_object={}, parameter_path={},\
                    lb={}, ub={}'.format(
                self.__class__.__name__, self.name, self.evaluation_object.name,
                self.parameter_path, self.lb, self.ub)
    