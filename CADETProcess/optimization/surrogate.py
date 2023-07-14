from pathlib import Path
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import (
    GaussianProcessRegressor, GaussianProcessClassifier)
from sklearn.base import BaseEstimator
import hopsy

from CADETProcess.optimization import (
    OptimizationProblem, TrustConstr, SLSQP, OptimizationResults
)

class Surrogate:
    """
    Surrogate class for optimization problems.
    """

    def __init__(
        self,
        optimization_problem: OptimizationProblem,
        optimization_results: OptimizationResults,
        n_samples=10000,
    ):
        """
        Initialize the Surrogate class.

        Parameters:
        - optimization_problem (OptimizationProblem): The optimization problem.
        - population (np.ndarray, optional): Initial population for fitting the
        surrogate models. Defaults to None.
        - n_samples (int, optional): Number of samples for surrogate model
        evaluation. Defaults to 10000.
        """

        self.n_samples = n_samples
        self.population = optimization_results.population_all
        self.optimization_problem = deepcopy(optimization_problem)
        self.plot_directory = optimization_results.plot_directory

        self.lower_bounds_copy = optimization_problem.lower_bounds.copy()
        self.upper_bounds_copy = optimization_problem.upper_bounds.copy()

        self.surrogate_model_F: BaseEstimator = None
        self.surrogate_model_G: BaseEstimator = None
        self.surrogate_model_M: BaseEstimator = None
        self.surrogate_model_CV: BaseEstimator = None

        self.fit_gaussian_process()


    def uncondition(self):
        """
        Reset the optimization problem bounds to their original values.
        """
        for var, lb, ub in zip(
            self.optimization_problem.variables,
            self.lower_bounds_copy,
            self.upper_bounds_copy
        ):
            var.lb = lb
            var.ub = ub

    def fit_gaussian_process(self):
        """
        Fit Gaussian process surrogate models to the population.

        Parameters:
        - population (np.ndarray): The population for fitting the surrogate models.
        """
        X = self.population.x_untransformed
        F = self.population.f
        G = self.population.g
        M = self.population.m
        CV = self.population.cv

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
        """
        Estimate the objectives using the surrogate model.

        Parameters:
        - X (np.ndarray): The input samples.
        - return_std (bool, optional): Whether to return the standard deviation
        of the predictions. Defaults to False.

        Returns:
        - np.ndarray: The estimated objectives.
        """
        return self.surrogate_model_F.predict(X, return_std=return_std)


    def estimate_non_linear_constraints(self, X):
        """
        Estimate the non-linear constraints using the surrogate model.

        Parameters:
        - X (np.ndarray): The input samples.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: The estimated non-linear
        constraints (G, CV).
        """
        G_est = self.surrogate_model_G.predict(X)
        CV_est = self.surrogate_model_CV.predict(X)
        return G_est, CV_est

    def estimate_meta_scores(self, X):
        """
        Estimate the meta scores using the surrogate model.

        Parameters:
        - X (np.ndarray): The input samples.

        Returns:
        - np.ndarray: The estimated meta scores.
        """
        M_est = self.surrogate_model_M.predict(X)
        return M_est

    def get_conditional_and_free_indices(self, conditional_vars={}):
        """
        Get the indices of the conditional and free variables.

        Parameters:
        - conditional_vars (dict, optional): Dictionary of variable names and
        their corresponding values to condition on. Defaults to an empty
        dictionary.

        Returns:
        - Tuple[List[int], List[int]]: The indices of the conditional variables
        and the free variables.
        """
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
        """
        Condition the constraints based on the given variables.

        Parameters:
        - conditional_vars (dict, optional): Dictionary of variable names and
        their corresponding values to condition on. Defaults to an empty
        dictionary.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: The conditioned inequality constraints
        (A_cond, b_cond).
        """
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
        """
        Condition the optimization problem based on the given variables.

        Parameters:
        - conditional_vars (dict, optional): Dictionary of variable names and
        their corresponding values to condition on. Defaults to an empty
        dictionary.

        Returns:
        - Tuple[OptimizationProblem, List[int], List[int]]: The conditioned
        optimization problem, indices of the conditional variables, and indices
        of the free variables.
        """
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
        free_variables = op.independent_variables


        # remove old lincons and set up new inequality constraints
        [op.remove_linear_constraint(0) for i in range(n_lincons)]
        for i in range(n_lincons):

            op.add_linear_constraint(
                opt_vars=[v.name for v in free_variables],
                lhs=A_cond[i],
                b=b_cond[i]
            )

        # set up new equality constraints
        [op.remove_linear_equality_constraint(0) for i in range(n_lineqcons)]
        for i in range(n_lineqcons):
            # TODO: must use Aeq and Beq
            op.add_linear_equality_constraint(
                opt_vars=[v for v in free_variables],
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

    def find_minimum(self, var_index):
        """
        Find the minimum of the optimization problem with respect to the given
        variable.

        Parameters:
        - var_index (int): The index of the variable to optimize.
        - plot_directory (str): The directory to save the plot to.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: The minimum objective values and the
        corresponding optimal points.
        """

        var = self.optimization_problem.variables[var_index]

        n = 21
        x_space = np.linspace(var.lb, var.ub, n)
        f_minimum = np.full((n, ), np.nan)
        x_optimal = np.full((n, self.optimization_problem.n_variables), fill_value=np.nan)

        for i, x_cond in enumerate(x_space):
            op, cond_var_idx, free_var_idx = self.condition_optimization_problem(
                conditional_vars={var.name: x_cond}
            )

            try:
                problem = op.create_hopsy_problem(simplify=True)
                chebyshev_orig = hopsy.compute_chebyshev_center(problem)[:, 0]

            except ValueError as e:
                if str(e) == "All inequality constraints are redundant, implying that the polytope is a single point.":
                    chebyshev_orig = None
                else:
                    continue

            optimizer = SLSQP()
            optimizer.optimize(
                op,
                x0=chebyshev_orig,
                reinit_cache=True,
                save_results=False,
            )

            x_free = optimizer.results.x_untransformed

            if len(x_free) > 1:
                assert np.allclose(np.diff(x_free, axis=0), 0)

            x_free = x_free[0]

            x_optimal[i, cond_var_idx] = x_cond
            x_optimal[i, free_var_idx] = x_free


            f = op.evaluate_objectives(x_free)
            f_minimum[i] = f

            self.uncondition()


        return f_minimum, x_optimal

    def plot_parameter_objective_space(self):
        """
        Plot the parameter-objective space.

        Parameters:
        - show (bool, optional): Whether to show the plot. Defaults to True.
        - plot_directory (str, optional): The directory to save the plot to.
        Defaults to None.
        """

        X = self.optimization_problem.create_initial_values(
            n_samples=self.n_samples, seed=1238
        )
        F_mean, F_std = self.estimate_objectives(X, return_std=True)

        variable_names = self.optimization_problem.variable_names
        n_vars = self.optimization_problem.n_variables
        fig, axes = plt.subplots(n_vars, n_vars, sharex="col")
        for row, var_y in zip(axes, variable_names):
            for ax, var_x in zip(row, variable_names):
                x_idx = self.optimization_problem.get_variable_index(var_x)
                y_idx = self.optimization_problem.get_variable_index(var_y)

                if var_y == var_x:
                    ax.scatter(X[:, x_idx], F_mean, s=5, label="obj. fun",
                               alpha=.01)
                    f_min, x_opt = self.find_minimum(x_idx)

                    ax.plot(x_opt[:, x_idx], f_min, color="red", lw=.5)

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

        if self.plot_directory is not None:
            plot_directory = Path(self.plot_directory)
            fig.savefig(f'{plot_directory / "surrogate_spaces.png"}')
