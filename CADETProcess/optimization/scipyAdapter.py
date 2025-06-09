import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import optimize
from scipy.optimize import OptimizeWarning

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Bool, Switch, UnsignedFloat, UnsignedInteger
from CADETProcess.optimization import OptimizationProblem, OptimizerBase


class SciPyInterface(OptimizerBase):
    """
    Wrapper around scipy's optimization suite.

    Defines the bounds and all constraints, saved in a constraint_object. Also
    the jacobian matrix is defined for several solvers.

    Parameters
    ----------
    finite_diff_rel_step : None or array_like, optional
        Relative step size to use for the numerical approximation of the jacobian.
        The absolute step size `h` is computed as `h = rel_step * sign(x) * max(1, abs(x))`,
        possibly adjusted to fit into the bounds. For `method='3-point'`,
        the sign of `h` is ignored.
        If `None` (default), the step size is selected automatically.
    tol : float, optional
        Tolerance for termination. When tol is specified, the selected minimization
        algorithm sets some relevant solver-specific tolerance(s) equal to tol.
        For detailed control, use solver-specific options.
    jac : {'2-point', '3-point', 'cs'}
        Method for computing the gradient vector. Only applicable to specific
        solvers (CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg,
        trust-krylov, trust-exact, and trust-constr).
        The default is 2-point.

    See Also
    --------
    COBYLA
    TrustConstr
    NelderMead
    SLSQP
    CADETProcess.optimization.OptimizationProblem.evaluate_objectives
    options
    scipy.optimize.minimize

    """

    finite_diff_rel_step = UnsignedFloat()
    tol = UnsignedFloat()
    jac = Switch(valid=["2-point", "3-point", "cs"], default="2-point")

    def _run(
        self,
        optimization_problem: OptimizationProblem,
        x0: Optional[list] = None,
    ) -> None:
        """
        Solve the optimization problem using any of the scipy methods.

        Parameters
        ----------
        results : OptimizationResults
            Optimization results including optimization_problem and solver
            configuration.
        x0 : list, optional
            Initial values of independent variables in untransformed space.

        See Also
        --------
        COBYLA
        TrustConstr
        NelderMead
        SLSQP
        CADETProcess.optimization.OptimizationProblem.evaluate_objectives
        options
        scipy.optimize.minimize
        """
        self.n_evals = 0

        if optimization_problem.n_objectives > 1:
            raise CADETProcessError("Can only handle single objective.")

        def objective_function(x: npt.ArrayLike) -> np.ndarray:
            return optimization_problem.evaluate_objectives(
                x,
                untransform=True,
                get_dependent_values=True,
                ensure_minimization=True,
            )[0]

        def callback_function(x: npt.ArrayLike, state: dict = None) -> bool:
            """
            Report progress after evaluation.

            Notes
            -----
            Currently, this evaluates all functions again. This should not be a problem
            since objectives and constraints are automatically cached.

            Unfortunately, only `trust-constr` returns a `state` which contains the
            current best point. Hence, the internal pareto front is used.
            """
            self.n_evals += 1

            x = x.tolist()
            f = optimization_problem.evaluate_objectives(
                x,
                untransform=True,
                get_dependent_values=True,
                ensure_minimization=True,
            )
            g = optimization_problem.evaluate_nonlinear_constraints(
                x,
                untransform=True,
                get_dependent_values=True,
            )
            cv = optimization_problem.evaluate_nonlinear_constraints_violation(
                x,
                untransform=True,
                get_dependent_values=True,
            )

            self.run_post_processing(x, f, g, cv, self.n_evals)

            return False

        if x0 is None:
            x0 = optimization_problem.create_initial_values(
                1, include_dependent_variables=False
            )[0]

        x0_transformed = optimization_problem.transform(x0)

        options = self.specific_options
        if self.results.n_gen > 0:
            x0 = self.results.population_last.x[0, :]
            self.n_evals = self.results.n_evals
            options["maxiter"] = self.maxiter - self.n_evals
            if str(self) == "COBYLA":
                options["maxiter"] -= 1

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            scipy_results = optimize.minimize(
                objective_function,
                x0=x0_transformed,
                method=str(self),
                tol=self.tol,
                jac=self.jac,
                constraints=self.get_constraint_objects(optimization_problem),
                bounds=self.get_bounds(optimization_problem),
                options=options,
                callback=callback_function,
            )

        self.results.success = bool(scipy_results.success)
        self.results.exit_flag = scipy_results.status
        self.results.exit_message = scipy_results.message

    def get_bounds(self, optimization_problem: OptimizationProblem) -> optimize.Bounds:
        """
        Return the optimized bounds of a given optimization_problem as a Bound object.

        Optimizes the bounds of the optimization_problem by calling the method
        optimize.Bounds. Keep_feasible is set to True.

        Returns
        -------
        bounds : Bounds
            Returns the optimized bounds as an object called bounds.
        """
        return optimize.Bounds(
            optimization_problem.lower_bounds_independent_transformed,
            optimization_problem.upper_bounds_independent_transformed,
            keep_feasible=True,
        )

    def get_constraint_objects(self, optimization_problem: OptimizationProblem) -> list:
        """
        Return constraints as objets.

        Returns
        -------
        constraint_objects : list
            List containing lists of all constraint types of the optimization_problem.
            If type of constraints is not defined, it is replaced with None.

        See Also
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

    def get_lincon_obj(
        self, optimization_problem: OptimizationProblem
    ) -> optimize.LinearConstraint:
        """
        Return the linear constraints as an object.

        Returns
        -------
        lincon_obj : LinearConstraint
            Linear Constraint object with lower and upper bounds of b of the
            optimization_problem.

        See Also
        --------
        constraint_objects
        A
        b
        """
        if optimization_problem.n_linear_constraints == 0:
            return None

        lb = [-np.inf] * len(optimization_problem.b)
        ub = optimization_problem.b_transformed

        return optimize.LinearConstraint(
            optimization_problem.A_independent_transformed, lb, ub, keep_feasible=True
        )

    def get_lineqcon_obj(
        self, optimization_problem: OptimizationProblem
    ) -> optimize.LinearConstraint:
        """
        Return the linear equality constraints as an object.

        Returns
        -------
        lineqcon_obj : LinearConstraint
            Linear equality Constraint object with lower and upper bounds of beq of the
            optimization_problem.

        See Also
        --------
        constraint_objects
        Aeq
        beq
        """
        if optimization_problem.n_linear_equality_constraints == 0:
            return None

        lb = optimization_problem.beq_transformed - optimization_problem.eps_lineq
        ub = optimization_problem.beq_transformed + optimization_problem.eps_lineq

        return optimize.LinearConstraint(
            optimization_problem.Aeq_independent_transformed, lb, ub, keep_feasible=True
        )

    def get_nonlincon_obj(self, optimization_problem: OptimizationProblem) -> list:
        """
        Return the optimized nonlinear constraints as an object.

        Returns
        -------
        nonlincon_obj : list
            Nonlinear constraint violation objects with bounds the optimization_problem.

        See Also
        --------
        constraint_objects
        nonlinear_constraints
        """
        if optimization_problem.n_nonlinear_constraints == 0:
            return [None]

        opt = optimization_problem

        def makeConstraint(i: int) -> optimize.NonlinearConstraint:
            """
            Create optimize.NonlinearConstraint object.

            Parameters
            ----------
            i : int
                Variable index

            Returns
            -------
            constr : optimize.NonlinearConstraint
                Constraint object.

            Notes
            -----
            Note, this is necessary to avoid side effects when creating the function
            in the main loop.
            """
            constr = optimize.NonlinearConstraint(
                lambda x: opt.evaluate_nonlinear_constraints_violation(
                    x,
                    untransform=True,
                    get_dependent_values=True,
                )[i],
                lb=-np.inf,
                ub=0,
                finite_diff_rel_step=self.finite_diff_rel_step,
                keep_feasible=True,
            )
            return constr

        constraints = []
        for i in range(opt.n_nonlinear_constraints):
            constraints.append(makeConstraint(i))

        return constraints

    def __str__(self) -> str:
        """str: String representation."""
        return self.__class__.__name__


class TrustConstr(SciPyInterface):
    """
    Wrapper for the trust-constr optimization method from the scipy optimization suite.

    It defines the solver options in the 'options' variable as a dictionary.

    Supports:
        - Linear constraints.
        - Linear equality constraints.
        - Nonlinear constraints.
        - Bounds.

    Parameters
    ----------
    gtol : UnsignedFloat, optional
        Tolerance for termination by the norm of the Lagrangian gradient.
        The algorithm will terminate when both the infinity norm (i.e., max abs value)
        of the Lagrangian gradient and the constraint violation are smaller than gtol.
        Default is 1e-8.
    xtol : UnsignedFloat, optional
        Tolerance for termination by the change of the independent variable.
        The algorithm will terminate when tr_radius < xtol,
        where tr_radius is the radius of the trust region used in the algorithm.
        Default is 1e-8.
    barrier_tol : UnsignedFloat, optional
        Threshold on the barrier parameter for the algorithm termination.
        When inequality constraints are present, the algorithm will terminate only
        when the barrier parameter is less than barrier_tol.
        Default is 1e-8.
    initial_tr_radius : float, optional
        Initial trust radius. The trust radius gives the maximum distance between solution points
        in consecutive iterations. It reflects the trust the algorithm puts in the local
        approximation of the optimization problem. For an accurate local approximation, the
        trust-region should be large, and for an approximation valid only close to the current
        point, it should be a small one. The trust radius is automatically updated throughout the
        optimization process, with initial_tr_radius being its initial value. Default is 1.
    initial_constr_penalty : float, optional
        Initial constraints penalty parameter. The penalty parameter is used for balancing
        the requirements of decreasing the objective function and satisfying the constraints.
        It is used for defining the merit function: merit_function(x) = fun(x)
        + constr_penalty * constr_norm_l2(x), where constr_norm_l2(x) is the l2 norm of a vector
        containing all the constraints. The merit function is used for accepting or rejecting trial
        points, and constr_penalty weights the two conflicting goals of reducing the objective
        function and constraints. The penalty is automatically updated throughout the optimization
        process, with initial_constr_penalty being its initial value. Default is 1.
    initial_barrier_parameter : float, optional
        Initial barrier parameter. Used only when inequality constraints are present.
        For dealing with optimization problems min_x f(x) subject to inequality constraints
        c(x) <= 0,
        the algorithm introduces slack variables, solving the problem
        min_(x, s) f(x) + barrier_parameter * sum(ln(s))
        subject to the equality constraints
        c(x) + s = 0
        instead of the original problem.
        This subproblem is solved for decreasing values of barrier_parameter and with decreasing
        tolerances for the termination, starting with initial_barrier_parameter for the barrier
        parameter. Default is 0.1.
    initial_barrier_tolerance : float, optional
        Initial tolerance for the barrier subproblem. Used only when inequality constraints are
        present. For dealing with optimization problems min_x f(x) subject to inequality constraints
        c(x) <= 0,
        the algorithm introduces slack variables, solving the problem
        min_(x, s) f(x) + barrier_parameter * sum(ln(s))
        subject to the equality constraints
        c(x) + s = 0
        instead of the original problem.
        This subproblem is solved for decreasing values of barrier_parameter
        and with decreasing tolerances for the termination, starting with initial_barrier_tolerance
        for the barrier tolerance. Default is 0.1.
    factorization_method : str or None, optional
        Method to factorize the Jacobian of the constraints.
        Use None (default) for auto selection or one of:
        - 'NormalEquation'
        - 'AugmentedSystem'
        - 'QRFactorization'
        - 'SVDFactorization'.
        The methods 'NormalEquation' and 'AugmentedSystem' can be used only with sparse
        constraints. The methods 'QRFactorization' and 'SVDFactorization' can be used
        only with dense constraints.
        Default is None.
    maxiter : UnsignedInteger, optional
        Maximum number of algorithm iterations. Default is 1000.
    verbose : UnsignedInteger, optional
        Level of algorithm's verbosity:
        - 0 (default) for silent
        - 1 for a termination report
        - 2 for progress during iterations
        - 3 for more complete progress report.
    disp : Bool, optional
        If True, then verbose will be set to 1 if it was 0. Default is False.
    """

    supports_linear_constraints = True
    supports_linear_equality_constraints = True
    supports_nonlinear_constraints = True
    supports_bounds = True

    gtol = UnsignedFloat(default=1e-8)
    xtol = UnsignedFloat(default=1e-8)
    barrier_tol = UnsignedFloat(default=1e-8)
    initial_tr_radius = UnsignedFloat(default=1.0)
    initial_constr_penalty = UnsignedFloat(default=1.0)
    initial_barrier_parameter = UnsignedFloat(default=0.1)
    initial_barrier_tolerance = UnsignedFloat(default=0.1)
    factorization_method = Switch(
        valid=[
            "NormalEquation",
            "AugmentedSystem",
            "QRFactorization",
            "SVDFactorization",
        ]
    )
    maxiter = UnsignedInteger(default=1000)
    verbose = UnsignedInteger(default=0)
    disp = Bool(default=False)

    x_tol = xtol            # Alias for uniform interface
    cv_nonlincon_tol = gtol  # Alias for uniform interface
    n_max_evals = maxiter   # Alias for uniform interface
    n_max_iter = maxiter    # Alias for uniform interface

    _specific_options = [
        "gtol",
        "xtol",
        "barrier_tol",
        "finite_diff_rel_step",
        "initial_constr_penalty",
        "initial_tr_radius",
        "initial_barrier_parameter",
        "initial_barrier_tolerance",
        "factorization_method",
        "maxiter",
        "verbose",
        "disp",
    ]

    def __str__(self) -> str:
        """str: String representation."""
        return "trust-constr"


class COBYLA(SciPyInterface):
    """
    Wrapper for the COBYLA optimization method from the scipy optimization suite.

    It defines the solver options in the 'options' variable as a dictionary.

    Supports:
        - Linear constraints
        - Linear equality constraints
        - Nonlinear constraints

    Parameters
    ----------
    rhobeg : float, default 1
        Reasonable initial changes to the variables.
    tol : float, default 0.0002
        Final accuracy in the optimization (not precisely guaranteed).
        This is a lower bound on the size of the trust region.
    disp : bool, default False
        Set to True to print convergence messages.
        If False, verbosity is ignored and set to 0.
    maxiter : int, default 10000
        Maximum number of function evaluations.
    catol : float, default 2e-4
        Absolute tolerance for constraint violations.
    """

    supports_linear_constraints = True
    supports_linear_equality_constraints = True
    supports_nonlinear_constraints = True
    supports_bounds = True

    rhobeg = UnsignedFloat(default=1)
    tol = UnsignedFloat(default=0.0002)
    maxiter = UnsignedInteger(default=10000)
    disp = Bool(default=False)
    catol = UnsignedFloat(default=0.0002)

    x_tol = tol                 # Alias for uniform interface
    cv_nonlincon_tol = catol    # Alias for uniform interface
    n_max_evals = maxiter       # Alias for uniform interface
    n_max_iter = maxiter        # Alias for uniform interface

    _specific_options = ["rhobeg", "tol", "maxiter", "disp", "catol"]


class NelderMead(SciPyInterface):
    """
    Wrapper for the Nelder-Mead optimization method from the scipy optimization suite.

    Supports:
        - Bounds.

    It defines the solver options in the 'options' variable as a dictionary.

    Parameters
    ----------
    maxiter : UnsignedInteger
        Maximum allowed number of iterations. The default = 1000.
    initial_simplex : None or array_like, optional
        Initial simplex. If given, it overrides x0.
        initial_simplex[j, :] should contain the coordinates of the jth vertex of the
        N+1 vertices in the simplex, where N is the dimension.
    xatol : UnsignedFloat, optional
        Absolute error in xopt between iterations that is acceptable for convergence.
    fatol : UnsignedFloat, optional
        Absolute error in f(xopt) between iterations that is acceptable for convergence.
    adaptive : Bool, optional
        Adapt algorithm parameters to dimensionality of the problem.
        Useful for high-dimensional minimization.
    disp : Bool, optional
        Set to True to print convergence messages.
    """

    supports_bounds = True

    maxiter = UnsignedInteger(default=1000)
    initial_simplex = None
    xatol = UnsignedFloat(default=1e-3)
    fatol = UnsignedFloat(default=1e-3)
    adaptive = Bool(default=True)
    disp = Bool(default=False)

    x_tol = xatol           # Alias for uniform interface
    f_tol = fatol           # Alias for uniform interface
    n_max_evals = maxiter   # Alias for uniform interface
    n_max_iter = maxiter    # Alias for uniform interface

    _specific_options = [
        "maxiter",
        "initial_simplex",
        "xatol",
        "fatol",
        "adaptive",
        "disp",
    ]

    def __str__(self) -> str:
        """str: String representation."""
        return "Nelder-Mead"


class SLSQP(SciPyInterface):
    """
    Wrapper for the SLSQP optimization method from the scipy optimization suite.

    It defines the solver options in the 'options' variable as a dictionary.

    Supports:
        - Linear constraints
        - Linear equality constraints
        - Nonlinear constraints
        - Bounds

    Parameters
    ----------
    ftol : float, default 1e-2
        Precision goal for the value of f in the stopping criterion.
    eps : float, default 1e-6
        Step size used for numerical approximation of the Jacobian.
    disp : bool, default False
        Set to True to print convergence messages.
        If False, verbosity is ignored and set to 0.
    maxiter : int, default 1000
        Maximum number of iterations.
    iprint: int, optional
        The verbosity of fmin_slsqp :
            iprint <= 0 : Silent operation
            iprint == 1 : Print summary upon completion (default)
            iprint >= 2 : Print status of each iterate and summary
    """

    supports_linear_constraints = True
    supports_linear_equality_constraints = True
    supports_nonlinear_constraints = True
    supports_bounds = True

    ftol = UnsignedFloat(default=1e-2)
    eps = UnsignedFloat(default=1e-6)
    disp = Bool(default=False)
    maxiter = UnsignedInteger(default=1000)
    iprint = UnsignedInteger(ub=2, default=1)

    f_tol = ftol            # Alias for uniform interface
    n_max_evals = maxiter   # Alias for uniform interface
    n_max_iter = maxiter    # Alias for uniform interface

    _specific_options = [
        "ftol",
        "eps",
        "disp",
        "maxiter",
        "finite_diff_rel_step",
        "iprint",
    ]
