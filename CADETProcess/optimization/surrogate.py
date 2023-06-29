import numpy as np

from sklearn.gaussian_process import (
    GaussianProcessRegressor, GaussianProcessClassifier)
from sklearn.base import BaseEstimator

from CADETProcess.optimization import (
    Population, OptimizationProblem, OptimizationResults)

class Surrogate:
    def __init__(
        self,
        optimization_problem: OptimizationProblem,
        population: Population
    ):
        self.optimization_problem = optimization_problem
        self.surrogate_model_F: BaseEstimator = None
        self.surrogate_model_G: BaseEstimator = None
        self.surrogate_model_M: BaseEstimator = None
        self.surrogate_model_CV: BaseEstimator = None
        self.fit_gaussian_process(population)

        # save a backup of bounds
        self.lower_bounds_copy = optimization_problem.lower_bounds.copy()
        self.upper_bounds_copy = optimization_problem.upper_bounds.copy()

    def _reset_bounds_on_variables(self):
        for var, lb, ub in zip(
            self.optimization_problem.variables,
            self.lower_bounds_copy,
            self.upper_bounds_copy
        ):
            var.lb = lb
            var.ub = ub

    def fit_gaussian_process(self, population: Population):
        X = population.x
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
        objectives = []
        F_est = self.surrogate_model_F.predict(X)
        objectives.append(F_est)

        if self.surrogate_model_G is not None:
            G_est = self.surrogate_model_G.predict(X)
            objectives.append(G_est)

        if self.surrogate_model_M is not None:
            M_est = self.surrogate_model_M.predict(X)
            objectives.append(M_est)

        if self.surrogate_model_CV is not None:
            CV_est = self.surrogate_model_CV.predict(X)
            objectives.append(CV_est)

        return np.array(objectives).T


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

        free_vars = {}
        for var in self.optimization_problem.variables:
            var_index = self.optimization_problem.get_variable_index(var.name)
            if var.name in conditional_vars:
                conditioning_value = conditional_vars[var.name]
                var.lb = conditioning_value - eps
                var.ub = conditioning_value + eps

            else:
                free_vars.update({var.name: var_index})

        X, F = self.approximate_objectives(n_samples=n_samples)

        self._reset_bounds_on_variables()

        return X, F, free_vars
