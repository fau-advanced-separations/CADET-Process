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
        optimization_results: OptimizationResults,
        n_samples=10000,
    ):
        """
        Initialize the Surrogate class.

        Parameters
        ----------
        optimization_results : OptimizationResults
            The optimization results from a successful optimization process.
        n_samples : int, optional
            Number of samples for surrogate model evaluation. Defaults to 10000.
        """

        self.n_samples = n_samples
        self.population = optimization_results.population_all
        self.optimization_problem = deepcopy(
            optimization_results.optimization_problem
        )
        self.plot_directory = optimization_results.plot_directory

        self.lower_bounds_copy = self.optimization_problem.lower_bounds.copy()
        self.upper_bounds_copy = self.optimization_problem.upper_bounds.copy()

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

        Parameters
        ----------
        population : np.ndarray
            The population for fitting the surrogate models.
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

        Parameters
        ----------
        X : np.ndarray
            The input samples.
        return_std : bool, optional
            Whether to return the standard deviation of the predictions.
            Defaults to False.

        Returns
        -------
        out : np.ndarray
            The estimated objectives.
        """
        return self.surrogate_model_F.predict(X, return_std=return_std)


    def estimate_non_linear_constraints(self, X):
        """
        Estimate the non-linear constraints using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        out : Tuple[np.ndarray, np.ndarray]
            The estimated non-linear constraints (G, CV).
        """
        G_est = self.surrogate_model_G.predict(X)
        CV_est = self.surrogate_model_CV.predict(X)
        return G_est, CV_est

    def estimate_meta_scores(self, X):
        """
        Estimate the meta scores using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        out : np.ndarray
            The estimated meta scores.
        """
        M_est = self.surrogate_model_M.predict(X)
        return M_est

    def get_conditional_and_free_indices(self, conditional_vars: dict = None):
        """
        Get the indices of the conditional and free variables.

        Parameters
        ----------
        conditional_vars : dict, optional
        Dictionary of variable names and their corresponding values to condition
        on. Defaults to None.

        Returns
        -------
        out : Tuple[List[int], List[int]]
            The indices of the conditional variables and the free variables.
        """
        if conditional_vars is None:
            conditional_vars = {}
        free_var_idx = []
        cond_var_idx = []
        for v in self.optimization_problem.variable_names:
            idx = self.optimization_problem.get_variable_index(v)
            if v in conditional_vars:
                cond_var_idx.append(idx)
            else:
                free_var_idx.append(idx)

        return cond_var_idx, free_var_idx

    def condition_constraints(self, conditional_vars: dict = None):
        """
        Condition the constraints based on the given variables.

        Parameters
        ----------
        conditional_vars : dict, optional
            Dictionary of variable names and their corresponding values to
            condition on. Defaults to None.

        Returns
        -------
        Out : np.ndarray, np.ndarray
            The conditioned inequality constraints (A_cond, b_cond).
        """
        if conditional_vars is None:
            conditional_vars = {}

        cond_var_idx, free_var_idx = self.get_conditional_and_free_indices(
            conditional_vars
        )

        cvar_values = list(conditional_vars.values())

        A = self.optimization_problem.A.copy()
        b = self.optimization_problem.b.copy().astype(float)

        b_cond = b - A[:, cond_var_idx].dot(cvar_values)
        A_cond = A[:, free_var_idx]

        return A_cond, b_cond


    def condition_optimization_problem(
        self, conditional_vars: dict = None, use_surrogate=True
    ):
        """
        Condition the optimization problem based on the given variables.

        Parameters
        ----------
        conditional_vars : dict, optional
            Dictionary of variable names and their corresponding values to
            condition on. Defaults to None.
        use_surrogate : bool, optional
            whether the minimizer should use the surrogate or the true
            objective function. Defaults to using the surrogate (True)

        Returns
        -------
        out : Tuple[OptimizationProblem, List[int], List[int]]
            The conditioned optimization problem, indices of the conditional
            variables, and indices of the free variables.
        """
        if conditional_vars is None:
            conditional_vars = {}

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

        if use_surrogate:
            obj.objective = surrogate_obj_fun
        else:
            obj.objective = conditioned_objective

        return op, cond_var_idx, free_var_idx


    def find_x0(
        self, cond_var_index: int, cond_var_value: float,
        x_tol=0.0, plot=False,
    ):
        """
        find an x for completing a minimum boundary w.r.t. a conditioning
        variable.

        Parameters
        ----------
        cond_var_index : int
            the index of the conditioning variable in the population
        cond_var_value : float
            the value at which the conditioning variable is fixed
        x_tol : float
            the minimum distance to neighboring x a new point needs to have
        plot : bool
            flag if the new candidate point is plotted

        Returns
        -------
        out:
        """
        f = self.population.f.reshape((-1, ))
        X = self.population.x_untransformed

        x = X[:, cond_var_index]
        x_search = cond_var_value

        # compute distance between x_search and x and then look for lowest
        # function values
        delta_x = x - x_search
        distance = np.sqrt(delta_x ** 2 )
        closest = [
            i for _, i in sorted(zip(distance, range(len(distance))))
        ]

        pois_left = []
        pois_right = []
        for i in closest[slice(6)]:
            delta_xi = delta_x[i]
            f_xi = f[i]
            p = (f_xi, delta_xi, i)
            if delta_xi < -x_tol:
                pois_left.append(p)
            elif delta_xi > x_tol:
                pois_right.append(p)
            else:
                continue


        if len(pois_left) == 0 or len(pois_right) == 0:
            return

        fl, dxl, il = sorted(pois_left)[0]
        fr, dxr, ir = sorted(pois_right)[0]

        # compute a new point according to the weighted distances of the
        # acquired points above
        x_new = X[[il, ir]]
        weights = np.abs([dxl, dxr])
        weights /= weights.sum()

        x_new_weighted = np.einsum("ij,i -> j", x_new, np.flip(weights))

        if plot:
            f_search = f.min() * 1.2
            ax = plt.gca()
            ax.scatter(x, f, color="tab:blue", alpha=.7)
            ax.scatter([x_search], [f_search], color="tab:red", marker="o", ls="")
            ax.plot([x_search, x[il]], [f_search, f[il]], lw=.5, color="tab:red")
            ax.plot([x_search, x[ir]], [f_search, f[ir]], lw=.5, color="tab:red")

        return x_new_weighted

    def find_minimum(self, var_index, use_surrogate=True):
        """
        Find the minimum of the optimization problem with respect to the given
        variable.

        Parameters
        ----------
        var_index : int
            The index of the variable to optimize.
        use_surrogate : bool, optional
            whether the minimizer should use the surrogate or the true
            objective function. Defaults to using the surrogate (True)

        Returns
        -------
        out : Tuple[np.ndarray, np.ndarray]
            The minimum objective values and the corresponding optimal points.
        """

        var = self.optimization_problem.variables[var_index]

        n = 21
        x_space = np.linspace(var.lb, var.ub, n)
        f_minimum = np.full((n, ), np.nan)
        x_optimal = np.full((n, self.optimization_problem.n_variables), fill_value=np.nan)

        for i, x_cond in enumerate(x_space):
            op, cond_var_idx, free_var_idx = self.condition_optimization_problem(
                conditional_vars={var.name: x_cond},
                use_surrogate=use_surrogate
            )

            try:
                problem = op.create_hopsy_problem(simplify=True)
                chebyshev_orig = hopsy.compute_chebyshev_center(problem)[:, 0]

            except ValueError as e:
                if str(e) == "All inequality constraints are redundant, implying that the polytope is a single point.":
                    chebyshev_orig = None
                else:
                    continue

            x0_weighted = self.find_x0(cond_var_index=var_index, cond_var_value=x_cond)

            if x0_weighted is None:
                x0 = chebyshev_orig
            else:
                x0 = x0_weighted[free_var_idx]

            optimizer = SLSQP()
            optimizer.optimize(
                op,
                x0=x0,
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

    def plot_parameter_objective_space(
        self, X, f, var_x, ax=None, use_surrogate=True,
    ):
        """
        plots the objective value as a function of var_x of the optimization
        problem.

        Parameters
        ----------
        X : ndarray
            the population of parameters
        f : ndarray
            the objective value w.r.t the parameters of X
        var_x : str, optional
            the variable to be conditioned upon
        use_surrogate : bool, optional
            whether the minimizer should use the surrogate or the true
            objective function. Defaults to using the surrogate (True)

        Returns
        -------
        out : tuple(ndarray, ndarray)
            a tuple of x and f_min(x)

        Examples
        --------
        Apply a surrogate model on 1 generation of U_NSGA3 on a constrained
        optimization problem.
        >>> from matplotlib import pyplot as plt
        >>> from CADETProcess.optimization import U_NSGA3
        >>> from tests.optimization_problem_fixtures import LinearConstraintsSooTestProblem2

        calculate 1 generation
        >>> optimizer = U_NSGA3(n_max_gen=1)
        >>> problem = LinearConstraintsSooTestProblem2("linear")
        >>> optimizer.optimize(
        >>>     optimization_problem=problem,
        >>>     results_directory="work/pymoo/",
        >>>     use_checkpoint=False,
        >>>     save_results=True,
        >>> )

        fit the surrogate model
        >>> surrogate = Surrogate(
        >>>     optimization_results=optimizer.results,
        >>> )

        plot the population and the mimimum F w.r.t. x
        >>> fig, ax = plt.subplots(1,1)
        >>> x_opt, f_min = surrogate.plot_parameter_objective_space(
        >>>    X=surrogate.population.x_untransformed,
        >>>    f=surrogate.population.f.reshape((-1,)),
        >>>    var_x="var_0",
        >>>    ax=ax,
        >>>    use_surrogate=False
        >>> )
        >>> ax.set_xlabel("$var_0$")
        >>> ax.set_ylabel("$F(var_0)$")
        >>> fig.tight_layout()
        >>> fig.savefig(f"{surrogate.plot_directory}/f_x.png")

        """
        x_idx = self.optimization_problem.get_variable_index(var_x)
        x = X[:, x_idx]
        f_min, x_opt = self.find_minimum(
            x_idx, use_surrogate=use_surrogate
        )

        if ax is None:
            ax = plt.subplot()
        ax.scatter(x, f, s=5, label="obj. fun", alpha=.75)
        ax.plot(x_opt[:, x_idx], f_min, color="red", lw=.5)

        return x_opt, f_min

    def pairplot_parameter_objective(self, use_surrogate=True):
        """
        Create a pairplot that iterates over variables and arranges them in
        a n x n grid, where n is the number of variables. The diagonal contains
        the objective value as a function of the variable x.

        Parameters
        ----------
        use_surrogate : bool, optional
            whether the minimizer should use the surrogate or the true
            objective function. Defaults to using the surrogate (True)

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
                    self.plot_parameter_objective_space(
                        x=X, f=F_mean, var_x=var_x, ax=ax,
                        use_surrogate=use_surrogate
                    )

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
