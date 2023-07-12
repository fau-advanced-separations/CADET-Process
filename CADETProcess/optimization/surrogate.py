from pathlib import Path
from copy import deepcopy
import numpy as np

from sklearn.gaussian_process import (
    GaussianProcessRegressor, GaussianProcessClassifier)
from sklearn.base import BaseEstimator

from matplotlib import pyplot as plt

class Surrogate:
    def __init__(
        self,
        optimization_problem,
        population=None,
        n_samples=10000,
        # TODO: consider which attributes of optimization problem are necessary
    ):
        from CADETProcess.optimization import OptimizationProblem
        self.optimization_problem: OptimizationProblem = deepcopy(optimization_problem)
        self.surrogate_model_F: BaseEstimator = None
        self.surrogate_model_G: BaseEstimator = None
        self.surrogate_model_M: BaseEstimator = None
        self.surrogate_model_CV: BaseEstimator = None
        if population is not None:
            self.fit_gaussian_process(population)

        self.n_samples = n_samples

        # if opt
        # save a backup of bounds
        self.lower_bounds_copy = optimization_problem.lower_bounds.copy()
        self.upper_bounds_copy = optimization_problem.upper_bounds.copy()

    def uncondition(self):
        # see condition_objectives

        for var, lb, ub in zip(
            self.optimization_problem.variables,
            self.lower_bounds_copy,
            self.upper_bounds_copy
        ):
            var.lb = lb
            var.ub = ub

    def fit_gaussian_process(self, population):
        """TODO: rename to `update`?"""
        self._population = population

        X = population.x_untransformed
        F = population.f
        G = population.g
        M = population.m
        CV = population.cv

        gp_f = GaussianProcessRegressor()
        gp_f.fit(X, F)
        self.surrogate_model_F = gp_f

        if G is not None:
            gp_g = GaussianProcessRegressor()
            gp_g.fit(X, G)
            self.surrogate_model_G = gp_g

        if M is not None:
            gp_m = GaussianProcessRegressor()
            gp_m.fit(X, M)
            self.surrogate_model_M = gp_m

        if CV is not None:
            gp_cv = GaussianProcessClassifier()
            gp_cv.fit(X, CV)
            self.surrogate_model_CV = gp_cv



    def estimate_objectives(self, X, return_std=False):
        return self.surrogate_model_F.predict(X, return_std=return_std)


    def estimate_non_linear_constraints(self, X):
        G_est = self.surrogate_model_G.predict(X)
        CV_est = self.surrogate_model_CV.predict(X)
        return G_est, CV_est

    def estimate_meta_scores(self, X):
        M_est = self.surrogate_model_M.predict(X)
        return M_est

    def condition_on(
            self,
            conditional_vars: dict = {},
            eps=1e-5
        ):

        # TODO: should check if the condition is inside the constriants
        #       otherwise HopsyF_est throws an error
        # This is somehwat dangerous maybe because it plays with the bounds
        # of the optimization problem. Make sure this is safe before
        # implementing
        # TODO: Pragmatically: Deppcopy
        # raise NotImplementedError("This method is potentially unsafe.")

        free_vars = {}
        for var in self.optimization_problem.variables:
            var_index = self.optimization_problem.get_variable_index(var.name)
            if var.name in conditional_vars:
                conditioning_value = conditional_vars[var.name]
                var.lb = conditioning_value - eps
                var.ub = conditioning_value + eps

            else:
                free_vars.update({var.name: var_index})

        # X = self.optimization_problem.create_initial_values(
        #     n_samples=self.n_samples,
        # )
        # F = self.estimate_objectives(X)


        return free_vars


    def get_conditional_and_free_indices(self, conditional_vars={}):
        free_var_idx = []
        cond_var_idx = []
        for v in self.optimization_problem.variable_names:
            idx = self.optimization_problem.get_variable_index(v)
            if v in conditional_vars:
                cond_var_idx.append(idx)
            else:
                free_var_idx.append(idx)

        return cond_var_idx, free_var_idx

    def condition_constraints(self, conditional_vars={}):
        cond_var_idx, free_var_idx = self.get_conditional_and_free_indices(
            conditional_vars
        )

        cvar_values = list(conditional_vars.values())

        A = self.optimization_problem.A.copy()
        b = self.optimization_problem.b.copy().astype(float)

        b_cond = b - A[:, cond_var_idx].dot(cvar_values)
        A_cond = A[:, free_var_idx]

        return A_cond, b_cond


    def condition_optimization_problem(self, conditional_vars={}):
        op = deepcopy(self.optimization_problem)

        # calculate conditional constraints matrices
        A_cond, b_cond = self.condition_constraints(conditional_vars)

        n_lincons = op.n_linear_constraints
        n_lineqcons = op.n_linear_equality_constraints
        n_variables = op.n_independent_variables

        cond_var_idx, free_var_idx = self.get_conditional_and_free_indices(
            conditional_vars
        )


        # remove variables
        [op.remove_variable(vn) for vn in conditional_vars.keys()]

        # set up new inequality constraints
        for i in range(n_lincons):
            lincon = op.linear_constraints[i]
            lincon_vars = lincon["opt_vars"]

            op.remove_linear_constraint(i)
            op.add_linear_constraint(
                opt_vars=[v for v in lincon_vars if v not in conditional_vars],
                lhs=A_cond[i],
                b=b_cond[i]
            )

        # set up new equality constraints
        for i in range(n_lineqcons):
            lineqcon = op.linear_equality_constraints[i]
            lineqcon_vars = lineqcon["opt_vars"]

            op.remove_linear_equality_constraint(i)
            # TODO: must use Aeq and Beq
            op.add_linear_equality_constraint(
                opt_vars=[v for v in lineqcon_vars if v not in conditional_vars],
                lhs=A_cond[i],
                beq=b_cond[i]
            )

        obj = op.objectives[0]
        obj_fun = obj.objective

        def complete_x(x):
            x_complete = np.zeros(n_variables)
            x_complete[cond_var_idx] = list(conditional_vars.values())
            x_complete[free_var_idx] = x

            return x_complete

        def conditioned_objective(x):
            x_complete = complete_x(x)

            return obj_fun(x_complete)

        def surrogate_obj_fun(x):
            x_complete = complete_x(x)

            return self.surrogate_model_F.predict(x_complete.reshape((1,-1)))

        obj.objective = surrogate_obj_fun

        return op, cond_var_idx, free_var_idx

    def find_minimum(self, var_index, plot_directory):
        """
        DONE: 1. determine true minimum of optimization problem or use other
        TODO: 2. implement finding of starting point
        TODO: 3. find out why the optimizer does not converge on true solution
                 despite having a clear and simple problem.
                 - draw conditioned space and then compare surrogate with true
                   problem.


        FIXME: using a linear space for x-fix may violate constraints
        FIXME: decoupling the optimizer from the valid parameter space has
               conflicts with constraints. How to I skip parameter proposals
               that are not feasible, because of the variable which should not
               be optimized
               I most likely need update linear constraints
        """
        # from CADETProcess.optimization import Objective
        from CADETProcess.optimization import TrustConstr, SLSQP
        from scipy.optimize import minimize

        var = self.optimization_problem.variables[var_index]

        n = 11
        x_space = np.linspace(var.lb, var.ub, n)
        f_minimum = np.full((n, ), np.nan)
        x_optimal = np.full((n, self.optimization_problem.n_variables), fill_value=np.nan)

        for i, x_cond in enumerate(x_space):
            op, cond_var_idx, free_var_idx = self.condition_optimization_problem(
                conditional_vars={var.name: x_cond}
            )

            try:
                op.create_initial_values(method="random", n_samples=1)
            except ValueError:
                continue

            # fig, ax = plt.subplots(1,1)
            # ax.scatter(X, F)
            # ax.set_xlim(-2, 2)
            # fig.savefig(f'{plot_directory / f"feasible_space_cond_{var_index}.png"}')

            optimizer = SLSQP()
            optimizer.optimize(
                op,
                reinit_cache=True,
                # x0=[0.0,0.0,0.0],
                save_results=False,
            )

            x_free = optimizer.results.x_untransformed

            # TODO: catches a bug where optimizer constructs a population
            if len(x_free) > 1:
                assert np.allclose(np.diff(x_free, axis=0), 0)
                x_free = x_free[0]

            x_optimal[i, cond_var_idx] = x_cond
            x_optimal[i, free_var_idx] = x_free


            f = op.evaluate_objectives(x_free)
            f_minimum[i] = f

            self.uncondition()


        return f_minimum, x_optimal






    def plot_parameter_objective_space(self, show=True, plot_directory=None):
        """
        TODO: for different optimiztation tasks (multi-objective, non-linear
        constraints. Create wrappers around the problem)
        TODO: als mean value + standard deviation mit tatsächlich ausgewrteten
        punkten (maybe integrieren in bestehende plots)
        TODO: contours? Lösen über fill between mit verschiedenen quantilen
        TODO: wie ist die marginalisierung in partial dependence plots gelöst
        """

        X = self.optimization_problem.create_initial_values(
            n_samples=self.n_samples, seed=1238
        )
        F_mean, F_std = self.estimate_objectives(X, return_std=True)
        # f_min, x_opt = self.find_minimum(0, plot_directory)

        # XF = np.column_stack((X, F_mean))
        # XF_sorted = XF[XF[:, 0].argsort()]
        # from sklearn.inspection import partial_dependence


        # len(XF)
        # XF_slc = XF_sorted[0:10, :]
        # X_min = XF_slc[XF_slc[:, -1].argmin()]
        # self.surrogate_model_F

        variable_names = self.optimization_problem.variable_names
        fig, axes = plt.subplots(2,2, sharex="col")
        for row, var_y in zip(axes, variable_names):
            for ax, var_x in zip(row, variable_names):
                x_idx = self.optimization_problem.get_variable_index(var_x)
                y_idx = self.optimization_problem.get_variable_index(var_y)

                if var_y == var_x:
                    ax.scatter(X[:, x_idx], F_mean, s=5, label="obj. fun",
                               alpha=.01)
                    f_min, x_opt = self.find_minimum(x_idx, plot_directory)

                    ax.plot(x_opt[:, x_idx], f_min, color="red", lw=.5)

                    print("debug")
                    # part_dep = partial_dependence(
                    #     self.surrogate_model_F, X=X, features=[i],
                    #     percentiles=(0,1), method = "brute")

                    # ax.plot(part_dep["values"][0], part_dep["average"][0],
                    #         color="red", lw=.5)
                    # ax.legend()
                else:
                    ax.scatter(X[:, x_idx], X[:, y_idx], s=5, c=F_mean)



        [ax.set_xlabel(var_x) for ax, var_x in zip(axes[-1, :], variable_names)]
        [ax.set_ylabel(var_y) for ax, var_y in zip(axes[:,  0], variable_names)]

        fig.tight_layout()

        if plot_directory is not None:
            plot_directory = Path(plot_directory)
            fig.savefig(f'{plot_directory / "surrogate_spaces.png"}')

        if not show:
            plt.close(fig)
