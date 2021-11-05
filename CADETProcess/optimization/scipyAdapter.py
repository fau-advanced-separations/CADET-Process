import copy
import time
import warnings

from scipy import optimize
from scipy.optimize import OptimizeWarning
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Bool, Switch, UnsignedInteger, UnsignedFloat
from CADETProcess.optimization import SolverBase, OptimizationResults

class SciPyInterface(SolverBase):
    """Wrapper around scipy's optimization suite.

    Defines the bounds and all constraints, saved in a constraint_object. Also
    the jacobian matrix is defined for several solvers.
    """
    finite_diff_rel_step = UnsignedFloat(default=1e-2)
    tol = UnsignedFloat()
    jac = '2-point'

    def run(self, optimization_problem):
        """Solve the optimization problem using any of the scipy methodss

        Returns
        -------
        results : OptimizationResults
            Optimization results including optimization_problem and solver
            configuration.

        See also
        --------
        COBYLA
        TrustConstr
        NelderMead
        SLSQP
        CADETProcess.optimization.OptimizationProblem.evaluate_objectives
        options
        scipy.optimize.minimize
        """
        if optimization_problem.n_objectives > 1:
            raise CADETProcessError("Can only handle single objective.")
        
        cache = dict()
        objective_function = \
            lambda x: optimization_problem.evaluate_objectives(x, cache=cache)[0]
            
        start = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=OptimizeWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            scipy_results = optimize.minimize(
                objective_function,
                x0=optimization_problem.x0,
                method=str(self),
                tol = self.tol,
                jac=self.jac,
                constraints=self.get_constraint_objects(optimization_problem),
                options=self.options
                )
        elapsed = time.time() - start

        if not scipy_results.success:
            raise CADETProcessError('Optimization Failed')

        x = scipy_results.x

        eval_object = copy.deepcopy(optimization_problem.evaluation_object)
        if optimization_problem.evaluator is not None:
            frac = optimization_problem.evaluator.evaluate(eval_object)
            performance = frac.performance
        else:
            frac = None
            performance = optimization_problem.evaluate(x, force=True)

        f = optimization_problem.evaluate_objectives(x)
        c = optimization_problem.evaluate_nonlinear_constraints(x)

        results = OptimizationResults(
            optimization_problem = optimization_problem,
            evaluation_object = eval_object,
            solver_name = str(self),
            solver_parameters = self.options,
            exit_flag = scipy_results.status,
            exit_message = scipy_results.message,
            time_elapsed = elapsed,
            x = list(x),
            f = f,
            c = c,
            frac = frac,
            performance = performance.to_dict()
        )

        return results


    def get_bounds(self, optimization_problem):
        """Returns the optimized bounds of a given optimization_problem as a
        Bound object.

        Optimizes the bounds of the optimization_problem by calling the method
        optimize.Bounds. Keep_feasible is set to True.

        Returns
        -------
        bounds : Bounds
            Returns the optimized bounds as an object called bounds.
        """
        return optimize.Bounds(
            optimization_problem.lower_bounds,
            optimization_problem.upper_bounds,
            keep_feasible=True
        )

    def get_constraint_objects(self, optimization_problem):
        """Defines the constraints of the optimization_problem and resturns
        them into a list.

        First defines the lincon, the linequon and the nonlincon constraints.
        Returns the constrainst in a list.

        Returns
        -------
        constraint_objects : list
            List containing  a sorted list of all constraints of an
            optimization_problem, if they're not None.

        See also
        --------
        lincon_obj
        lincon_obj
        nonlincon_obj
        """
        lincon = self.get_lincon_obj(optimization_problem)
        lineqcon = self.get_lineqcon_obj(optimization_problem)
        nonlincon = self.get_nonlincon_obj(optimization_problem)

        constraints = [lincon, lineqcon, *nonlincon]

        return [con for con in constraints if con is not None]

    def get_lincon_obj(self, optimization_problem):
        """Returns the optimized linear constraint as an object.

        Sets the lower and upper bounds of the optimization_problem and returns
        optimized linear constraints. Keep_feasible is set to True.

        Returns
        -------
        lincon_obj : LinearConstraint
            Linear Constraint object with optimized upper and lower bounds of b
            of the optimization_problem.

        See also
        --------
        constraint_objects
        A
        b
        """
        lb = [-np.inf]*len(optimization_problem.b)
        ub = optimization_problem.b

        return optimize.LinearConstraint(
            optimization_problem.A, lb, ub, keep_feasible=True
        )

    def get_lineqcon_obj(self, optimization_problem):
        """Returns the optimized linear equality constraints as an object.

        Checks the length of the beq first, before setting the bounds of the
        constraint. Sets the lower and upper bounds of the
        optimization_problem and returns optimized linear equality constraints.
        Keep_feasible is set to True.

        Returns
        -------
        None: bool
            If the length of the beq of the optimization_problem is equal zero.
        lineqcon_obj : LinearConstraint
            Linear equality Constraint object with optimized upper and lower
            bounds of beq of the optimization_problem.

        See also
        --------
        constraint_objects
        Aeq
        beq
        """
        if len(optimization_problem.beq) == 0:
            return None

        lb = optimization_problem.beq
        ub = optimization_problem.beq

        return optimize.LinearConstraint(
            optimization_problem.Aeq, lb, ub, keep_feasible=True
        )

    def get_nonlincon_obj(self, optimization_problem):
        """Returns the optimized nonlinear constraints as an object.

        Checks the length of the nonlinear_constraints first, before setting
        the bounds of the constraint. Tries to set the bounds from the list
        nonlinear_constraints from the optimization_problem for the lower
        bounds and sets the upper bounds for the length of the
        nonlinear_constraints list. If a TypeError is excepted it sets the
        lower bound by the first entry of the nonlinear_constraints list and
        the upper bound to infinity. Then a local variable named
        finite_diff_rel_step is defined. After setting the bounds it returns
        the optimized nonlinear constraints as an object with the
        finite_diff_rel_step and the jacobian matrix. The jacobian matrix is
        got by calling the method nonlinear_constraint_jacobian from the
        optimization_problem. Keep_feasible is set to True.

        Returns
        -------
        None: bool
            If the length of the nonlinear_constraints of the
            optimization_problem is equal zero.
        nonlincon_obj : NonlinearConstraint
            Linear equality Constraint object with optimized upper and lower
            bounds of beq of the optimization_problem.

        See also
        --------
        constraint_objects
        nonlinear_constraints
        """
        if optimization_problem.nonlinear_constraints is None:
            return None
        
        def makeConstraint(i):
            constr = optimize.NonlinearConstraint(
                lambda x: optimization_problem.evaluate_nonlinear_constraints(x)[i],
                lb=-np.inf, ub=0,
                finite_diff_rel_step=self.finite_diff_rel_step,
                keep_feasible=True
                )
            return constr

        constraints = []
        for i, constr in enumerate(optimization_problem.nonlinear_constraints):
            constraints.append(makeConstraint(i))

        return constraints

    def __str__(self):
        return self.__class__.__name__


class TrustConstr(SciPyInterface):
    """Class from scipy for optimization with trust-constr as method for the
    solver.

    This class is a wrapper for the method trust-constr from the optimization
    suite of the scipy interface. It defines the solver options in the local
    variable options as a dictionary and implements the abstract method run for
    running the optimization.
    """
    gtol = UnsignedFloat(default=1e-6)
    xtol = UnsignedFloat(default=1e-8)
    barrier_tol = UnsignedFloat(default=1e-8)
    initial_constr_penalty = UnsignedFloat(default=1.0)
    initial_tr_radius = UnsignedFloat(default=1.0)
    initial_barrier_parameter = UnsignedFloat(default=0.01)
    initial_barrier_tolerance = UnsignedFloat(default=0.01)
    factorization_method = None
    maxiter = UnsignedInteger(default=1000)
    verbose = UnsignedInteger(default=0)
    disp = Bool(default=False)
    _options = [
        'gtol', 'xtol', 'barrier_tol', 'finite_diff_rel_step',
        'initial_constr_penalty',
        'initial_tr_radius', 'initial_barrier_parameter',
        'initial_barrier_tolerance', 'factorization_method',
        'maxiter','verbose', 'disp'
    ]

    jac = Switch(default='3-point', valid=['2-point', '3-point', 'cs'])

    def __str__(self):
        return 'trust-constr'

class COBYLA(SciPyInterface):
    """Class from scipy for optimization with COBYLA as method for the
    solver.

    This class is a wrapper for the method COBYLA from the optimization
    suite of the scipy interface. It defines the solver options in the local
    variable options as a dictionary and implements the abstract method run for
    running the optimization.
    """
    rhobeg = UnsignedFloat(default=1)
    maxiter = UnsignedInteger(default=1000)
    disp = Bool(default=False)
    catol = UnsignedFloat(default=0.0002)
    _options = ['rhobeg', 'maxiter', 'disp', 'catol']


class NelderMead(SciPyInterface):
    """
    """
    maxiter = UnsignedInteger()
    maxfev = UnsignedInteger()
    initial_simplex = None
    xatol = UnsignedFloat(default=0.01)
    fatol = UnsignedFloat(default=0.01)
    adaptive = Bool(default=True)
    _options = [
        'maxiter', 'maxfev', 'initial_simplex', 'xatol', 'fatol', 'adaptive'
    ]


class SLSQP(SciPyInterface):
    """Class from scipy for optimization with SLSQP as method for the
    solver.

    This class is a wrapper for the method SLSQP from the optimization
    suite of the scipy interface. It defines the solver options in the local
    variable options as a dictionary and implements the abstract method run for
    running the optimization.
    """
    ftol = UnsignedFloat(default=1e-2)
    eps = UnsignedFloat(default=1e-6)
    disp = Bool(default=False)
    _options = ['ftol', 'eps', 'disp']
