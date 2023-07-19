from pathlib import Path
from copy import deepcopy
import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import (
    GaussianProcessRegressor, GaussianProcessClassifier)
from sklearn.base import BaseEstimator
import hopsy

from CADETProcess import CADETProcessError
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
        self.optimization_problem: OptimizationProblem = deepcopy(
            optimization_results.optimization_problem
        )
        self.plot_directory = optimization_results.plot_directory

        self.lower_bounds_copy = self.optimization_problem.lower_bounds.copy()
        self.upper_bounds_copy = self.optimization_problem.upper_bounds.copy()

        self.surrogate_model_F: BaseEstimator = None
        self.surrogate_model_G: BaseEstimator = None
        self.surrogate_model_M: BaseEstimator = None
        self.surrogate_model_feasible: BaseEstimator = None

        self.read_population(population=optimization_results.population_all)
        self.fit_gaussian_process()

    def read_population(self, population):
        """
        read attributes from the population

        Parameters
        ----------
        population : Population
            The population for fitting the surrogate models.
        """
        self.X = population.x
        self.F = population.f
        self.G = population.g
        self.M = population.m
        self.CV = population.cv

    @property
    def feasible(self):
        if self.CV is not None:
            return np.all(self.CV < 0, axis=1)
        else:
            return self.all_feasible(self.X)

    @staticmethod
    def all_feasible(X):
        return np.full(len(X), fill_value=True)

    def update(self):
        """
        updates the surrogate model with new individuals
        """
        self.X = np.row_stack(
            [self.X, self.X_new]
        )

        self.F = np.row_stack(
            [self.F, self.F_new]
        )

        if len(self.G_new) > 0 :
            self.G = np.row_stack(
                [self.G, self.G_new]
            )

        if len(self.CV_new) > 0 :
            self.CV = np.row_stack(
                [self.CV, self.CV_new]
            )



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
        """

        gp_f = GaussianProcessRegressor()
        gp_f.fit(self.X, self.F)
        self.surrogate_model_F = gp_f

        if self.G is not None:
            gp_g = GaussianProcessRegressor()
            gp_g.fit(self.X, self.G)
            self.surrogate_model_G = gp_g

        if self.M is not None:
            gp_m = GaussianProcessRegressor()
            gp_m.fit(self.X, self.M)
            self.surrogate_model_M = gp_m

        if self.feasible is not None:
            # TODO: catch error where points are only feasible or infeasible
            if np.all(self.feasible):
                gp_feasible = self.all_feasible
            else:
                gp_feasible = GaussianProcessClassifier()
                gp_feasible.fit(self.X, self.feasible)
            self.surrogate_model_feasible = gp_feasible

    def estimate_objectives(self, X):
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
        out : np.ndarray with ndim=2
            The estimated objectives.
        """

        X_ = np.array(X, ndmin=2)
        F_mean_est = self.surrogate_model_F.predict(X_)
        # always cast as multi objective problem
        F_mean_est = F_mean_est.reshape((len(X_), -1))
        if X.ndim == 1:
            return F_mean_est[0]
        else:
            return F_mean_est

    def estimate_objectives_standard_deviation(self, X):
        """
        Estimate the standard deviation of the objective function evaluated
        at X using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        out : np.ndarray
            The estimated objectives.
        """

        X_ = np.array(X, ndmin=2)
        _, F_std_est = self.surrogate_model_F.predict(X_, return_std=True)
        F_std_est = F_std_est.reshape((len(X_), -1))

        if X.ndim == 1:
            return F_std_est[0]
        else:
            return F_std_est

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
            The estimated non-linear constraints (G, feasible points).
        """
        X_ = np.array(X, ndmin=2)
        G_est = self.surrogate_model_G.predict(X_)
        G_est = G_est.reshape((len(X_), -1))

        if X.ndim == 1:
            return G_est[0]
        else:
            return G_est

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
        self,
        conditional_vars: dict = None,
        objective_index: list = None,
        use_surrogate=True
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
        op = deepcopy(self.optimization_problem)

        if conditional_vars is None:
            conditional_vars = {}

        if objective_index is None:
            objective_index = np.arange(op.n_objectives)

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

        def complete_x(x):
            x_complete = np.zeros(n_variables)
            x_complete[cond_var_idx] = list(conditional_vars.values())
            x_complete[free_var_idx] = x

            return x_complete

        # generate conditioned nonlinear constraints
        for nlc in op.nonlinear_constraints:
            nlc_func = nlc.nonlinear_constraint

            nlc.nonlinear_constraint = self.condition_model_or_surrogate(
                model_func=nlc_func,
                surrogate_func=self.estimate_non_linear_constraints,
                use_surrogate=use_surrogate,
                complete_x=complete_x,
            )

        obj_index = {}
        oi = 0
        for i, obj in enumerate(op.objectives):
            j = 0
            while j < obj.n_objectives:
                if oi in objective_index:
                    obj_index.update({oi: (i, j)})
                oi += 1
                j += 1

        # conditioned original function of otimization problem
        objectives = []
        for obj_id, (obj_func_idx, obj_return_idx) in obj_index.items():
            obj = op.objectives[obj_func_idx]
            obj_fun = obj.objective
            obj.n_objectives = 1

            if len(obj.evaluators) > 0:
                first_eval = obj.evaluators[0]

            obj.objective = self.condition_model_or_surrogate(
                model_func=obj_fun,
                surrogate_func=self.estimate_objectives,
                use_surrogate=use_surrogate,
                complete_x=complete_x,
                return_idx=obj_return_idx
            )

            objectives.append(obj)

        op._objectives = objectives

        return op, cond_var_idx, free_var_idx

    def condition_model_or_surrogate(
        self,
        model_func,
        surrogate_func,
        use_surrogate,
        complete_x,
        return_idx=None,
    ):
        """
        convenience wrapper around condition_function
        """
        if use_surrogate:
            func = surrogate_func
        else:
            func = model_func

        conditioned_func = self.condition_function(
            func=func, complete_x=complete_x, return_idx=return_idx,
        )

        return conditioned_func

    @staticmethod
    def condition_function(func, complete_x, return_idx=None) -> callable:
        """
        completes input x with the x-value of the conditioning
        variable.
        If func requires an evaluator, x are the intermediate results and
        stay as

        Then casts the output to the proper dimensionality according to the
        problem.

        """
        def conditioned_func(x):
            x_complete = complete_x(x)
            assert x_complete.ndim == 1, "currently only supports evaluation of individuals"

            f = func(x_complete)


            f = np.array(f, ndmin=1)

            # catch case where only a single individual (1-D X) is given and
            # the number of objectives is > 1
            # TODO: currently I think this is always the case
            # if x_complete.ndim == 1:
            #     f_ = f[0]
            # else:
            #     f_ = f

            # index the objective of interest and cast to an array of 1-D
            if return_idx is not None:
                f_return = np.array(f[return_idx], ndmin=1)
            else:
                f_return = f

            return f_return

        return conditioned_func

    def find_x0(
        self, cond_var_index: int, cond_var_value: float,
        objective_index: list,
        x_tol=0.0, n_neighbors=10, plot=False
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
        objective_index : int
            the indices of the objectives for which an x_0 should be found
        x_tol : float
            the minimum distance to neighboring x a new point needs to have
        plot : bool
            flag if the new candidate point is plotted

        Returns
        -------
        out:

        Todos
        -----
        TODO: find a method to find_x0 also if no points are available to the left
              as an idea. Use points to the right and extrapolate
              Also, consider that the point is optimized in any case, so the
              point only needs to be feasible
        TODO:
        """
        if objective_index is None:
            objective_index = np.arange(self.optimization_problem.n_objectives)

        f = self.F[self.feasible, objective_index]
        X = self.X[self.feasible]

        x = X[:, cond_var_index].reshape((-1, ))
        x_search = cond_var_value

        var = self.optimization_problem.variables[cond_var_index]

        # transform vars
        x_trans = (x - var.lb) / (var.ub - var.lb)
        f_trans = (f-np.min(f))/(np.max(f)-np.min(f))
        x_search_trans = (x_search - var.lb) / (var.ub - var.lb)

        delta_x_trans = x_trans - x_search_trans
        # no need to calculate delta_f because: f_trans = f - f_opt = f - 0 = f
        distance = np.sqrt(delta_x_trans ** 2 + f_trans ** 2 )
        closest = [
            i for _, i in sorted(zip(distance, range(len(distance))))
        ]

        pois_left = []
        pois_right = []
        # for i in closest[slice(n_neighbors)]:
        for i in closest:
            # TODO: does it check for f distance?
            delta_xi = delta_x_trans[i]
            f_xi = f[i]
            p = (f_xi, delta_xi, i)
            if delta_xi < -x_tol and len(pois_left) < n_neighbors:
                pois_left.append(p)
            elif delta_xi > x_tol and len(pois_right) < n_neighbors:
                pois_right.append(p)
            else:
                continue

            if len(pois_left) + len(pois_right) == n_neighbors * 2:
                break

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

    def optimize_conditioned_problem(
        self, optimization_problem, x0,
    ):

        optimizer = TrustConstr()
        optimizer.optimize(
            optimization_problem,
            x0=x0,
            reinit_cache=True,
            save_results=False,
        )

        x_free = optimizer.results.x

        if len(x_free) > 1:
            assert np.allclose(np.diff(x_free, axis=0), 0)

        x_free = x_free[0]
        return x_free



    def find_minimum(self, var_index, use_surrogate=True, n=21):
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

        n_vars = self.optimization_problem.n_variables
        n_objs = self.optimization_problem.n_objectives

        var = self.optimization_problem.variables[var_index]

        f_minimum = np.full((n, n_objs), fill_value=np.nan)
        x_optimal = np.full((n, n_objs, n_vars), fill_value=np.nan)

        x_space = np.linspace(var.lb, var.ub, n)

        for objective_index in range(n_objs):

            for i, x_cond in enumerate(x_space):
                op, cond_var_idx, free_var_idx = self.condition_optimization_problem(
                    conditional_vars={var.name: x_cond},
                    objective_index=[objective_index],
                    use_surrogate=use_surrogate
                )

                try:
                    problem = op.create_hopsy_problem(simplify=True)
                    chebyshev_orig = hopsy.compute_chebyshev_center(problem)[:, 0]

                except ValueError as e:
                    if str(e) == "All inequality constraints are redundant, implying that the polytope is a single point.":
                        problem = op.create_hopsy_problem(simplify=False)
                        chebyshev_orig = hopsy.compute_chebyshev_center(problem)[:, 0]
                        # TODO: stop here and record single optimal point
                        # chebyshev_orig = None
                    else:
                        continue

                x0_weighted = self.find_x0(
                    cond_var_index=cond_var_idx[0],
                    objective_index=objective_index,
                    cond_var_value=x_cond,
                    x_tol=0.0,
                    n_neighbors=1
                )

                if x0_weighted is not None:
                    x0 = x0_weighted[free_var_idx]
                else:
                    x0 = chebyshev_orig

                try:
                    x_free = self.optimize_conditioned_problem(
                        optimization_problem=op,
                        x0=x0,
                    )
                except CADETProcessError:
                    x_free = self.optimize_conditioned_problem(
                        optimization_problem=op,
                        x0=chebyshev_orig,
                    )
                except ValueError as e:
                    if "`x0` is infeasible" in str(e):
                        continue
                    else:
                        raise e

                x_opt = np.full(n_vars, fill_value=np.nan)
                x_opt[cond_var_idx] = x_cond
                x_opt[free_var_idx] = x_free
                x_optimal[i, objective_index] = x_opt

                f = op.evaluate_objectives(x_opt[free_var_idx])
                f_minimum[i, objective_index] = f

                # TODO: check if needed
                self.uncondition()


        return f_minimum, x_optimal

    def fill_gaps(self, cond_var, step, n_neighbors, optimize=False):
        idx = self.optimization_problem.get_variable_index(cond_var)
        var = self.optimization_problem.variables[idx]

        var_search = np.arange(var.lb, var.ub, step)

        X = []
        F = []
        G = []
        CV = []
        for x_cond in var_search:
            x = self.find_x0(
                cond_var_index=idx,
                cond_var_value=x_cond,
                x_tol=0.0,
                n_neighbors=n_neighbors,
                plot=False
            )

            if x is not None:
                if optimize:
                    op, cond_var_idx, free_var_idx = self.condition_optimization_problem(
                        conditional_vars={cond_var: x_cond},
                        use_surrogate=True
                    )

                    try:
                        x_free = self.optimize_conditioned_problem(
                            optimization_problem=op,
                            x0=x[free_var_idx]
                        )
                    except CADETProcessError:
                        warnings.warn(f"skipped {x}, because it violated constraints.")
                        continue

                    x_opt = np.full(self.optimization_problem.n_variables, np.nan)
                    x_opt[cond_var_idx] = x_cond
                    x_opt[free_var_idx] = x_free

                    x = x_opt

                # evaluate true function
                f = self.optimization_problem.evaluate_objectives(x)
                X.append(x)
                F.append(f)

                if self.optimization_problem.n_nonlinear_constraints > 0:
                    g = self.optimization_problem.evaluate_nonlinear_constraints(x)
                    cv = self.optimization_problem.evaluate_nonlinear_constraints_violations(x)
                    G.append(g)
                    CV.append(cv)

        self.X_new = np.array(X)
        self.F_new = np.array(F)
        self.G_new = np.array(G)
        self.CV_new = np.array(CV)

    def plot_parameter_objective_space(
        self, X, f, var_x, axes=None, use_surrogate=True,
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
        >>>    X=surrogate.X,
        >>>    f=surrogate.F.reshape((-1,)),
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

        if self.feasible is not None:
            alpha = (self.feasible.astype(float)+(1/3)) / (4/3)

        n_objectives = self.optimization_problem.n_objectives

        if axes is None:
            _, axes = plt.subplots(n_objectives)

        for oi, ax in enumerate(axes):
            ax.scatter(x, f[:, oi], s=10, label="obj. fun", alpha=alpha)
            ax.plot(x_opt[:, oi, x_idx], f_min[:, oi], color="red", lw=.5)

            # standard deviation currently seems not really a meaningful
            # measure
            # if use_surrogate:
            #     x_opt_nonan = x_opt[~np.all(np.isnan(x_opt), axis=(1,2)),:]
            #     F_mean = self.estimate_objectives(x_opt_nonan[:, oi, :])
            #     F_std = self.estimate_objectives_standard_deviation(
            #         x_opt_nonan[:, oi, :]
            #     )
            #     # F_std = F_std.reshape((len(x_opt_nonan), -1))

            #     ax.fill_between(
            #         x_opt_nonan[:, oi, x_idx],
            #         F_mean[:, oi]-F_std[:, oi], F_mean[:, oi]+F_std[:, oi],
            #         color="red", alpha=.5
            #     )

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
