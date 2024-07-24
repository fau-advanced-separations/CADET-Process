import math

import cyipopt
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Switch, UnsignedFloat
from CADETProcess.optimization import OptimizerBase


class IPOPT(OptimizerBase):
    """Class from scipy for optimization with trust-constr as method for the
    solver.

    This class is a wrapper for the method trust-constr from the optimization
    suite of the scipy interface. It defines the solver options in the local
    variable options as a dictionary and implements the abstract method run for
    running the optimization.

    """
    supports_linear_constraints = True
    supports_linear_equality_constraints = True
    supports_nonlinear_constraints = True

    tol = UnsignedFloat(default=1e-7)
    mu_target = Switch(valid=['adaptive'], default='adaptive')
    _options = ['tol', 'mu_target']

    def run(self, optimization_problem):
        """Solve the optimization problem using IPOPT.

        Returns
        -------
        results : OptimizationResults
            Optimization results including optimization_problem and solver
            configuration.

        See Also
        --------
        CADETProcess.optimization.OptimizerBase.optimize
        CADETProcess.optimization.OptimizationProblem.evaluate_objectives

        """
        if optimization_problem.n_objectives > 1:
            raise CADETProcessError("Can only handle single objective.")

        n_lin = optimization_problem.n_linear_constraints
        n_nonlin = optimization_problem.n_nonlinear_constraints
        cl_lin = n_lin * [-math.inf]
        cl_nonlin = n_nonlin * [-math.inf]
        cu_lin = n_lin * [0]
        cu_nonlin = n_nonlin * [0]

        problem = cyipopt.Problem(
           n=optimization_problem.n_variables,
           m=n_lin + n_nonlin,
           problem_obj=IPOPTProblem(optimization_problem),
           lb=optimization_problem.lower_bounds,
           ub=optimization_problem.upper_bounds,
           cl=cl_lin + cl_nonlin,
           cu=cu_lin + cu_nonlin,
        )

        problem.add_option('mu_strategy', self.mu_target)
        problem.add_option('tol', self.tol)

        if optimization_problem.x0 is None:
            optimization_problem.create_initial_values(1)
        x0 = optimization_problem.x0

        if len(np.array(optimization_problem.x0).shape) > 1:
            x0 = x0[0]

        x, info = problem.solve(x0)


class IPOPTProblem():
    def __init__(self, optimization_problem):
        self.optimization_problem = optimization_problem

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return self.optimization_problem.evaluate_objectives(x)[0]

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return self.optimization_problem.objective_jacobian(x)[0]

    def constraints(self, x):
        """Returns the constraints."""
        return self.optimization_problem.evaluate_nonlinear_constraints(x)

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return self.optimization_problem.nonlinear_constraint_jacobian(x)

    # def hessian(self, x, lagrange, obj_factor):
    #     """Returns the non-zero values of the Hessian."""

    #     H = obj_factor*np.array((
    #         (2*x[3], 0, 0, 0),
    #         (x[3],   0, 0, 0),
    #         (x[3],   0, 0, 0),
    #         (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

    #     H += lagrange[0]*np.array((
    #         (0, 0, 0, 0),
    #         (x[2]*x[3], 0, 0, 0),
    #         (x[1]*x[3], x[0]*x[3], 0, 0),
    #         (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

    #     H += lagrange[1]*2*np.eye(4)

    #     row, col = self.hessianstructure()

    #     return H[row, col]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        msg = "Objective value at iteration #{:d} is - {:g}"

        print(msg.format(iter_count, obj_value))

        self.n_evals = iter_count

        # self.run_post_evaluation_processing(x, f, g, self.n_evals)

    def __str__(self):
        return 'IPOPT'
