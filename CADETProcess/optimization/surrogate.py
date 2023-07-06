from pathlib import Path
import numpy as np

from sklearn.gaussian_process import (
    GaussianProcessRegressor, GaussianProcessClassifier)
from sklearn.base import BaseEstimator

from matplotlib import pyplot as plt

class Surrogate:
    def __init__(
        self,
        optimization_problem,
        population
    ):
        self.optimization_problem = optimization_problem
        self.surrogate_model_F: BaseEstimator = None
        self.surrogate_model_G: BaseEstimator = None
        self.surrogate_model_M: BaseEstimator = None
        self.surrogate_model_CV: BaseEstimator = None
        # self.fit_gaussian_process(population)

        # save a backup of bounds
        self.lower_bounds_copy = optimization_problem.lower_bounds.copy()
        self.upper_bounds_copy = optimization_problem.upper_bounds.copy()

    def _reset_bounds_on_variables(self):
        # see condition_objectives
        raise NotImplementedError("This method is potentially unsafe.")

        for var, lb, ub in zip(
            self.optimization_problem.variables,
            self.lower_bounds_copy,
            self.upper_bounds_copy
        ):
            var.lb = lb
            var.ub = ub

    def fit_gaussian_process(self, population):
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



    def estimate_objectives(self, X):
        F_est = self.surrogate_model_F.predict(X)
        return F_est

    def estimate_non_linear_constraints(self, X):
        G_est = self.surrogate_model_G.predict(X)
        CV_est = self.surrogate_model_CV.predict(X)
        return G_est, CV_est

    def estimate_meta_scores(self, X):
        M_est = self.surrogate_model_M.predict(X)
        return M_est

    def estimate_feasible_objectives_space(self, n_samples=1000):
        X = self.optimization_problem.create_initial_values(
            n_samples=n_samples,
            method="random",
        )
        F = self.estimate_objectives(X)

        return X, F


    def condition_objectives(
            self,
            conditional_vars: dict = {},
            n_samples=1000,
            eps=1e-5
        ):

        # TODO: should check if the condition is inside the constriants
        #       otherwise Hopsy throws an error
        # This is somehwat dangerous maybe because it plays with the bounds
        # of the optimization problem. Make sure this is safe before
        # implementing
        raise NotImplementedError("This method is potentially unsafe.")

        free_vars = {}
        for var in self.optimization_problem.variables:
            var_index = self.optimization_problem.get_variable_index(var.name)
            if var.name in conditional_vars:
                conditioning_value = conditional_vars[var.name]
                var.lb = conditioning_value - eps
                var.ub = conditioning_value + eps

            else:
                free_vars.update({var.name: var_index})

        X, F = self.estimate_feasible_objectives_space(n_samples=n_samples)

        self._reset_bounds_on_variables()

        return X, F, free_vars

    def plot_parameter_objective_space(self, show=True, plot_directory=None):
        X, F = self.estimate_feasible_objectives_space()

        variable_names = self.optimization_problem.variable_names
        fig, axes = plt.subplots(3,3)
        for i, (row, var_x) in enumerate(zip(axes, variable_names)):
            for j, (ax, var_y) in enumerate(zip(row, variable_names)):
                if i == j:
                    ax.scatter(X[:, i], F, s=5, )
                    ax.set_ylabel("f")
                    ax.set_xlabel(var_x)
                else:
                    ax.scatter(X[:, i], X[:, j], s=5, c=F)
                    ax.set_ylabel(var_y)
                    ax.set_xlabel(var_x)

        if plot_directory is not None:
            plot_directory = Path(plot_directory)
            fig.savefig(f'{plot_directory / "surrogate_spaces.png"}')

        if not show:
            plt.close(fig)
