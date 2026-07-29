import warnings
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from scipy import optimize
from scipy.optimize import OptimizeResult, OptimizeWarning

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import (
    Bool,
    Float,
    Switch,
    UnsignedFloat,
    UnsignedInteger,
)
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
    COBYQA
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
        COBYQA
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
                ensure_minimization=True,
            )[0]

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
            if str(self) in ["COBYLA", "COBYQA"]:
                options['maxiter'] -= 1

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
                callback=self.get_callback(optimization_problem),
            )

        # Manually run callback for COBYLA
        # see also: https://github.com/scipy/scipy/issues/24598
        if str(self) == "COBYLA":
            callback = self.get_callback(optimization_problem)
            callback(scipy_results.x)

        self.results.success = bool(scipy_results.success)
        self.results.exit_flag = scipy_results.status
        self.results.exit_message = scipy_results.message

    def get_bounds(self, optimization_problem: OptimizationProblem) -> optimize.Bounds:
        """
        Configure the bound constraints of a given optimization problem.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            The given optimizaiton problem.

        Returns
        -------
        bounds : Bounds
            Bound constraints of the optimization problem.
        """
        ts = optimization_problem.transformed_space
        return optimize.Bounds(
            ts.lower_bounds,
            ts.upper_bounds,
            keep_feasible=True,
        )

    def get_constraint_objects(self, optimization_problem: OptimizationProblem) -> list:
        """
        Return the constraints as an object.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            The given optimizaiton problem.

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

        ts = optimization_problem.transformed_space
        lb = [-np.inf] * optimization_problem.n_linear_constraints
        ub = ts.b

        return optimize.LinearConstraint(
            ts.A, lb, ub, keep_feasible=True
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

        ts = optimization_problem.transformed_space

        return optimize.LinearConstraint(
            ts.A_eq, ts.b_eq, ts.b_eq, keep_feasible=True
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

    def get_callback(self, optimization_problem: OptimizationProblem) -> Callable:
        """
        Configure callback function.

        Parameters
        ----------
        optimization : OptimizationProblem

        Returns
        -------
        Callable
            The callback funcction

        Note, different optimizers in Scipy support different callback signatures.
        We try to use the more modern `OptimizeResult` signature which is more likely
        to also contain the current best value. For older optimizers, `xk` is used.
        However, there is ambiguity in what that point actually is (see also:
        https://github.com/scipy/scipy/issues/21061).

        """
        if isinstance(self, (COBYLA, SLSQP)):
            def callback(x: npt.ArrayLike, state: dict = None) -> bool:
                """
                Report progress after evaluation.

                Notes
                -----
                Currently, this evaluates all functions again. This should not be a problem
                since objectives and constraints are automatically cached.
                """
                self.n_evals += 1

                x = x.tolist()
                f = optimization_problem.evaluate_objectives(
                    x,
                    untransform=True,
                        ensure_minimization=True,
                )
                g = optimization_problem.evaluate_nonlinear_constraints(
                    x,
                    untransform=True,
                    )
                self.run_post_processing(x, f, g, self.n_evals)

                return False

            return callback

        def callback(intermediate_result: OptimizeResult) -> None:
            """
            Report progress after evaluation.

            Parameters
            ----------
            intermediate_result: OptimizeResult
                The current state of the optimization.
            """
            self.n_evals += 1

            x_transformed = intermediate_result.x
            f = intermediate_result.fun

            g = optimization_problem.evaluate_nonlinear_constraints(
                x_transformed,
                untransform=True,
            )
            self.run_post_processing(x_transformed, f, g, self.n_evals)

        return callback

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


class COBYQA(SciPyInterface):
    """Wrapper for the COBYQA optimization method from the scipy optimization suite.

    It defines the solver options in the 'options' variable as a dictionary.

    Supports:
        - Linear constraints
        - Linear equality constraints
        - Nonlinear constraints
        - Bounds

    Parameters
    ----------
    disp : bool, default False
        Set to True to print information about the optimization procedure.
    maxfev : int
        Maximum number of function evaluations.
        The default is None.
    maxiter : int
        Maximum number of iterations.
        The default is None.
    f_target : float
        Target value for the objective function.The optimization procedure is
        terminated when the objective function value of a feasible point (see
        `feasibility_tol` below) is less than or equal to this target.
        The default is `-np.inf`
    feasibility_tol : float
        Absolute tolerance for the constraint violation.
        The default is 1e-8
    initial_tr_radius : float
        Initial trust-region radius. Typically, this value should be in the order of one
        tenth of the greatest expected change to the variables.
        The default is 1.0
    final_tr_radius : float
        Final trust-region radius. It should indicate the accuracy required in the final
        values of the variables. If provided, this option overrides the value of `tol`
        in the `minimize` function.
        The default is 1e-6
    """

    supports_linear_constraints = True
    supports_linear_equality_constraints = True
    supports_nonlinear_constraints = True
    supports_bounds = True

    disp = Bool(default=False)

    maxfev = UnsignedInteger()
    maxiter = UnsignedInteger()
    f_target = Float(default=-np.inf)
    feasibility_tol = UnsignedFloat(default=1e-8)
    initial_tr_radius = UnsignedFloat(default=1.0)
    final_tr_radius = UnsignedFloat(default=1e-6)

    x_tol = final_tr_radius             # Alias for uniform interface
    cv_nonlincon_tol = feasibility_tol  # Alias for uniform interface
    n_max_evals = maxfev                # Alias for uniform interface
    n_max_iter = maxiter                # Alias for uniform interface

    _specific_options = [
        "disp",
        "maxfev",
        "maxiter",
        "f_target",
        "feasibility_tol",
        "initial_tr_radius",
        "final_tr_radius",
    ]


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


class LBFGSB(SciPyInterface):
    """
    Wrapper for the L-BFGS-B optimization method from the scipy optimization suite.

    It defines the solver options in the 'options' variable as a dictionary.

    Supports:
        - Bounds.

    Parameters
    ----------
    maxcor : UnsignedInteger, optional
        Maximum number of variable metric corrections used to define the limited
        memory matrix. Default is 10.
    ftol : UnsignedFloat, optional
        Precision goal for the value of f in the stopping criterion:
        (f^k - f^{k+1}) / max{|f^k|, |f^{k+1}|, 1} <= ftol.
        Default is 2.220446049250313e-09.
    gtol : UnsignedFloat, optional
        The iteration stops when max{|proj g_i|, i = 1, ..., n} <= gtol, where
        proj g_i is the i-th component of the projected gradient.
        Default is 1e-5.
    eps : UnsignedFloat, optional
        Absolute step size used for numerical approximation of the Jacobian, if
        no analytical Jacobian is provided. Default is 1e-8.
    maxfun : UnsignedInteger, optional
        Maximum number of function evaluations. Note that this limit can be
        exceeded since the gradient is also evaluated numerically.
        Default is 15000.
    maxiter : UnsignedInteger, optional
        Maximum number of iterations. Default is 15000.
    maxls : UnsignedInteger, optional
        Maximum number of line search steps per iteration. Default is 20.
    """

    supports_bounds = True

    maxcor = UnsignedInteger(default=10)
    ftol = UnsignedFloat(default=2.220446049250313e-09)
    gtol = UnsignedFloat(default=1e-5)
    eps = UnsignedFloat(default=1e-8)
    maxfun = UnsignedInteger(default=15000)
    maxiter = UnsignedInteger(default=15000)
    maxls = UnsignedInteger(default=20)

    f_tol = ftol            # Alias for uniform interface
    n_max_evals = maxfun    # Alias for uniform interface
    n_max_iter = maxiter    # Alias for uniform interface

    _specific_options = [
        "maxcor",
        "ftol",
        "gtol",
        "eps",
        "maxfun",
        "maxiter",
        "maxls",
        "finite_diff_rel_step",
    ]

    def __str__(self) -> str:
        """str: String representation."""
        return "L-BFGS-B"


class LeastSquares(OptimizerBase):
    """
    Wrapper for the least_squares optimization method from the scipy optimization
    suite (Trust Region Reflective, dogbox, or Levenberg-Marquardt).

    Unlike the other adapters in this module, this does not scalarize the
    OptimizationProblem's objectives before optimizing. Instead, every objective
    component is treated as a residual, and the solver minimizes the sum of their
    squares directly using its own Jacobian-based (Gauss-Newton-style) steps. This
    typically converges far more robustly than a generic quasi-Newton method on a
    pre-summed scalar, especially when the scalarized objective is ill-conditioned.

    This is only appropriate when every registered objective is itself a residual/
    error-like quantity that should be driven toward zero (e.g. NRMSE) - not a
    general-purpose replacement for arbitrary objectives, since least_squares
    actively steers each component toward zero rather than merely decreasing it.

    Supports:
        - Bounds.

    Does not support linear, linear equality, or nonlinear constraints; scipy's
    least_squares has no mechanism for them.

    Note, unlike the other adapters in this module, `scipy.optimize.least_squares`
    provides no per-iteration callback hook. Intermediate iterates are therefore
    not reported to `OptimizationResults`; only the final result is post-processed.

    Parameters
    ----------
    method : {'trf', 'dogbox', 'lm'}, optional
        Algorithm to execute. 'trf' (default) and 'dogbox' support bounds; 'lm'
        (Levenberg-Marquardt) does not and raises if the problem has finite bounds.
    loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}, optional
        Loss function applied to the residuals. 'linear' (default) is standard
        least squares; the others down-weight outlying residuals for robustness.
        Only used by 'trf' and 'dogbox'.
    f_scale : UnsignedFloat, optional
        Soft margin between inlier and outlier residuals for the robust loss
        functions above. Default is 1.0. Has no effect for loss='linear'.
    ftol : UnsignedFloat, optional
        Tolerance for termination by the change of the cost function. Default 1e-8.
    xtol : UnsignedFloat, optional
        Tolerance for termination by the change of the independent variables.
        Default 1e-8.
    gtol : UnsignedFloat, optional
        Tolerance for termination by the norm of the gradient. Default 1e-8.
    jac : {'2-point', '3-point', 'cs'}, optional
        Method for numerically approximating the Jacobian. Default is '2-point'.
    diff_step : UnsignedFloat, optional
        Relative step size for the numerical approximation of the Jacobian (scipy's
        equivalent of the other adapters' `finite_diff_rel_step`). If None (default),
        the step size is selected automatically.

    See Also
    --------
    CADETProcess.optimization.OptimizationProblem.evaluate_objectives
    scipy.optimize.least_squares
    """

    supports_single_objective = True
    supports_multi_objective = True
    supports_bounds = True

    method = Switch(valid=["trf", "dogbox", "lm"], default="trf")
    loss = Switch(
        valid=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="linear"
    )
    f_scale = UnsignedFloat(default=1.0)
    ftol = UnsignedFloat(default=1e-8)
    xtol = UnsignedFloat(default=1e-8)
    gtol = UnsignedFloat(default=1e-8)
    jac = Switch(valid=["2-point", "3-point", "cs"], default="2-point")
    diff_step = UnsignedFloat()

    x_tol = xtol  # Alias for uniform interface
    f_tol = ftol  # Alias for uniform interface

    _specific_options = [
        "method",
        "loss",
        "f_scale",
        "ftol",
        "xtol",
        "gtol",
        "jac",
        "diff_step",
    ]

    def _run(
        self,
        optimization_problem: OptimizationProblem,
        x0: Optional[list] = None,
    ) -> None:
        """
        Solve the optimization problem using scipy.optimize.least_squares.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            Optimization problem to be solved. Every registered objective
            component is treated as a residual whose sum of squares is minimized.
        x0 : list, optional
            Initial values of independent variables in untransformed space.

        See Also
        --------
        CADETProcess.optimization.OptimizationProblem.evaluate_objectives
        scipy.optimize.least_squares
        """
        self.n_evals = 0

        def residuals(x: npt.ArrayLike) -> np.ndarray:
            self.n_evals += 1
            return optimization_problem.evaluate_objectives(
                x,
                untransform=True,
                ensure_minimization=True,
            )

        if x0 is None:
            x0 = optimization_problem.create_initial_values(
                1, include_dependent_variables=False
            )[0]

        x0_transformed = optimization_problem.transform(x0)

        lb, ub = self.get_bounds(optimization_problem)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            scipy_results = optimize.least_squares(
                residuals,
                x0=x0_transformed,
                jac=self.jac,
                bounds=(lb, ub),
                method=self.method,
                loss=self.loss,
                f_scale=self.f_scale,
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
                diff_step=self.diff_step,
                max_nfev=self.n_max_evals,
            )

        self.results.success = bool(scipy_results.success)
        self.results.exit_flag = scipy_results.status
        self.results.exit_message = scipy_results.message

        self.run_post_processing(
            [scipy_results.x],
            [scipy_results.fun],
            None,
            self.n_evals,
        )

    def get_bounds(
        self, optimization_problem: OptimizationProblem
    ) -> tuple:
        """
        Configure the bound constraints of a given optimization problem.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            The given optimization problem.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Lower and upper bounds, as expected by scipy.optimize.least_squares.
        """
        ts = optimization_problem.transformed_space
        return ts.lower_bounds, ts.upper_bounds

    def __str__(self) -> str:
        """str: String representation."""
        return self.__class__.__name__
