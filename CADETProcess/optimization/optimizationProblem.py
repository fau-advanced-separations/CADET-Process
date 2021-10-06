import copy
import math
import warnings

from addict import Dict
import numpy as np
from scipy import optimize
import hopsy

from CADETProcess import CADETProcessError
from CADETProcess.common import log
from CADETProcess.common import frozen_attributes

from CADETProcess.common import StructMeta
from CADETProcess.common import String

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

    def __init__(self, evaluation_object, evaluator=None, name=None,
                 save_log=False):
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

        If force is True, set to None and the str of x is a key of the
        eval_dict dictionary the value of the eval_dict with x as the key is
        returned. The starting point is set to x to start evaluating at given
        point. The variables are set for point x and the results are defined by
        calling the method evaluate of the evaluation_object. The value of the
        key at x in the eval_dict is set to results.

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
        evaluate_nonlinear_constraint_fun
        """
        performance = self.evaluate(x, *args, **kwargs)
        f = [obj(performance) for obj in self.objectives]
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
        c = [constr(performance) for constr in self.nonlinear_constraints]
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


    def add_linear_constraint(self, opt_vars, factors, b=0):
        """Adds linear conctraints.

        First checks if the required variable of the opt_vars is in the
        variables_dict and raises an CADETProcessError if not. Also the length of
        the list factors and opt_vars is checked for correct setting of the
        factors. It raises a CADETProcessError if not. Then a Addict dict named
        lincon is created and the factors, b and the opt_vars are saved into
        the lincon dictionary.

        Parameters
        -----------
        opt_vars : list
            Name as list of strings of the OptimizationVariable to be added.
        factors : list
            Factors as list of integers of the opt_vars
        b : float
            Vector form of linear constraints, default set to zero.

        Raises
        -------
        CADETProcessError
            If required variables of the opt_vars is not listed in the
            variables_dict.
            If the length of the factors does not match the length of the
            opt_vars.

        Returns
        -------
        linear_equality_constraints : list
            List of dictionaries, containing the linear constraints of an
            OptimizationProblem.

        See also
        ---------
        variables_dict
        remove_linear_constraint
        add_linear_equality_constraint
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
        """Removes linear constraints.

        Parameters
        ----------
        index : int
            Index of the list, where the linear constraints has to be removed.
        """
        del(self._linear_constraints[index])

    @property
    def linear_constraints(self):
        """list : linear constraints of OptimizationProblem
        """
        return self._linear_constraints

    @property
    def A(self):
        """Matrix form of linear constraints.

        First creates a matrix with the length of the linear_constraints for
        the rows and the length of the variables as columns and fills the value
        with zero. For each lincon_index and lincon in the list
        linear_constraints and for each variable and its index in the list
        opt_vars of the lincon dictionary the index is set with the index of
        the list variables by the variableof the variables_dict. The index and
        the lincon_index from matrix A is set to the var_index and factors,
        respectively.

        Returns
        -------
        A : ndarray
            Matrix of the linear constraints.

        See also
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
        """Vector form of linear constraints.

        Creates a zero-valued vector with the length of the linear_constraints.
        The values are filled with the the dictionary lincon for each entry in
        the linear_constraintslist and saves them into a list.

        Returns
        -------
        b : list
            Vector with the linear constraints.

        see also
        --------
        A
        add_linear_constraint
        remove_linear_constraint
        """
        b = [lincon['b'] for lincon in self.linear_constraints]

        return np.array(b)

    def linear_constraint_fun(self, x):
        """Return the value of linear constraints at point x.

        Returns the value of linear constraints at point x by pointwise
        substratcion of the matrix A by b.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        constraints: array_like
            Value of the linear constraints at point x

        See also
        --------
        A
        b
        linear_constraints
        """
        return self.A.dot(x) - self.b

    def check_linear_constraints(self, x):
        """Checks the the value of the linear_concstraints_fun at point x.

        Sets a local variable flag to True. Sets this variable to False if any
        value of the linear_constraint_fun at point x is greater than zero.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        flag : bool
            Returns True for linear_constraint_fun(x) smaller zero.
            False for linear_constraint_fun(x) greater zero.

        See also:
        ---------
        linear_constraints
        linear_equality_constraint_fun
        A
        b
        """
        flag = True

        if np.any(self.linear_constraint_fun(x) > 0):
            flag = False

        return flag

    def add_linear_equality_constraint(self, opt_vars, factors, beq=0):
        """Adds linear_equality_constraints.

        First checks if the required variable of the opt_vars is in the
        variables_dict and raises an CADETProcessError if not. Also the length of
        the list factors and opt_vars is checked for correct setting of the
        factors. It raises a CADETProcessError if not. Then a Addict dict named
        lineqcon is created and the factors, b and the opt_vars are saved into
        the lineqcon dictionary.

        Parameters
        -----------
        opt_vars : list
            Name as list of strings of the OptimizationVariable to be added.
        factors : list
            Factors as list of integers of the opt_vars
        beq : andarray
            Vector form of linear equality constraints, default set to zero.

        Raises
        -------
        CADETProcessError
            If required variables of the opt_vars is not listed in the
            variables_dict.
            If the length of the factors does not match the length of the
            opt_vars.

        Returns
        -------
        linear_equality_constraints : list
            List of dictionaries, containing the linear equality constraints of
            an OptimizationProblem.

        See also
        --------
        add_linear_constraint
        remove_linear_equality_constraint
        linear_equality_constraints
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
            Index of the linear_equality_constraints list where the
                linear equality contraints variable has to be removed.

        See also
        --------
        add_linear_equality_constraint
        linear_equality_constraint
        """
        del(self._linear_equality_constraints[index])

    @property
    def linear_equality_constraints(self):
        """Returns the list linear_equality constraints.

        Returns
        -------
        linear_equality_constraints : list
            List of dictionaries, containing the linear equality constraints of
            an OptimizationProblem.

        See also
        --------
        add_linear_equality_constraint
        remove_linear_equality_constraint
        """
        return self._linear_equality_constraints

    @property
    def Aeq(self):
        """Matrix form of linear equality constraints.

        First creates a matrix with the length of the linear_constraints for
        the rows and the length of the variables as columns and fills the value
        with zero named Aeq. For each lineqcon_index and lineqcon in the list
        linear_equality_constraints and for each variable and its index in the
        list opt_vars of the lineqcon dictionary the index is set with the
        index of the list variables by the variableof the variables_dict. The
        index and the lineqcon_index from matrix A is set to the var_index and
        factors, respectively.

        Returns
        -------
        Aeq : ndarray
            Matrix of the linear equality constraints.

        see also
        --------
        beq
        add_linear_equality_constraint
        remove_linear_equality_constraint
        """
        Aeq = np.zeros(
                (len(self.linear_equality_constraints), len(self.variables)))

        for lineqcon_index, lineqcon in enumerate(
                self.linear_equality_constraints):
            for var_index, var in enumerate(lineqcon.opt_vars):
                index = self.variables.index(var)
                Aeq[lineqcon_index, index] = lineqcon.factors[var_index]

        return Aeq

    @property
    def beq(self):
        """Vector form of linear equality constraints.

        Creates a zero-valued vector with the length of the
        linear_equality_constraints. The values are filled with the the
        dictionary lineqcon for each entry in
        the linear_constraintslist and saves them into a list.

        Returns
        -------
        beq : list
            Vector with the linear equality constraints.

        see also
        --------
        Aeq
        add_linear_equality_constraint
        remove_linear_equality_constraint
        """
        beq = np.zeros((len(self.linear_equality_constraints),))
        beq = [lineqcon.beq for lineqcon in self.linear_equality_constraints]

        return beq

    def linear_equality_constraint_fun(self, x):
        """Return the value of linear equality constraints at point x.

        Returns the value of linear equality constraints at point x by
        pointwise substratcion of the matrix Aeq by beq.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        constraints: array_like
            Value of the linear euqlity constraints at point x

        See also
        --------
        Aeq
        beq
        linear_equality_constraints
        """
        return self.Aeq.dot(x) - self.beq

    def check_linear_equality_constraints(self, x):
        """Checks the if the value of the linear_equality_concstraints_fun at
        point x.

        Sets a local variable flag to True. Sets this variable to False if any
        value of the linear_equality_constraint_fun at point x is greater than
        zero.

        Parameters
        ----------
        x : array_like
            Value of the optimization variables.

        Returns
        -------
        flag : bool
            Returns True for linear_constraintequality_fun(x) smaller zero.
            False for linear_equality_constraint_fun(x) greater zero.
        """
        flag = True

        if np.any(self.linear_equality_constraint_fun(x) != 0):
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
        if self._x0 is None:
            self._x0 = self.create_initial_values()
        return self._x0

    @x0.setter
    def x0(self, x0):
        if not len(x0) == len(self.variables):
            raise CADETProcessError(
                "Starting value must be given for all variables"
            )
        self._x0 = x0


    def create_initial_values(self, n_samples=1):
        """Function for creating set of initial values.

        The function tries to find a random number between the lower and upper
        bounds. If no bounds are set for one variable, the lowest value of all
        other lower bounds is used as lower bounds and the highest value of all
        other upper bounds is used instead.

        The values are rounded to one decimal place.

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
            run = hopsy.Run(problem)
        
            run.starting_points = [hopsy.compute_chebyshev_center(problem)]
        run.sample(n_samples)
                    
        states = np.array(run.data.states[0])
        
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

from CADETProcess.common.utils import check_nested, generate_nested_dict, \
                                    get_nested_value
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
