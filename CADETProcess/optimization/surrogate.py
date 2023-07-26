from pathlib import Path
from copy import deepcopy
import warnings
from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import hopsy

from CADETProcess import CADETProcessError
from CADETProcess.optimization import (
    OptimizationProblem,
    TrustConstr,
    SLSQP,
    OptimizationResults,
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
        self.surrogate_model_CV: BaseEstimator = None
        self.surrogate_model_M: BaseEstimator = None

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
        self.X = np.row_stack([self.X, self.X_new])

        self.F = np.row_stack([self.F, self.F_new])

        if len(self.G_new) > 0:
            self.G = np.row_stack([self.G, self.G_new])

        if len(self.CV_new) > 0:
            self.CV = np.row_stack([self.CV, self.CV_new])

        self.fit_gaussian_process()

    def train_gp(self, X, Y):
        """fits a GP on scaled input and output"""

        X_scaler = StandardScaler().fit(X)
        Y_scaler = StandardScaler().fit(Y)

        gpr = GaussianProcessRegressor()
        gpr.fit(X=X_scaler.transform(X), y=Y_scaler.transform(Y))

        return gpr, X_scaler, Y_scaler

    def fit_gaussian_process(self):
        """
        Fit Gaussian process surrogate models to the population.
        """

        gp_f, self.X_scaler, self.F_scaler = self.train_gp(self.X, self.F)
        self.surrogate_model_F = gp_f

        if self.G is not None:
            gp_g, _, self.G_scaler = self.train_gp(self.X, self.G)
            self.surrogate_model_G = gp_g

        if self.CV is not None:
            gp_cv, _, self.CV_scaler = self.train_gp(self.X, self.CV)
            self.surrogate_model_CV = gp_cv

        if self.M is not None:
            raise NotImplementedError("Meta scores are currently not implemented.")

    # TODO: write a wrapper for casting of X and result. This is the same
    #       for all `estimate_...`` functions
    @staticmethod
    def fix_dimensions(func):
        """
        this wrapper makes sure that X dimensions are 2D for usage with sklearn
        models and processes the output of the resulting Y depending on the
        dimensionality of X-input so that it matches the output of
        `OptimizationProblem`
        """

        def estimator(self, X):
            # cast X to 2D for transforming it
            X_ = np.array(X, ndmin=2)

            Y_ = func(self, X_)

            # cast Y_ to N-D depending on the input Y dimensionality
            # (i.e. the number of Y features)
            Y = Y_.reshape((len(X_), -1))

            # return an individual or a population depending on the length of X
            if X.ndim == 1:
                return Y[0]
            else:
                return Y

        return estimator

    @fix_dimensions
    def estimate_objectives(self, X):
        """
        Estimate the objectives using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        out : np.ndarray
            The estimated objectives.
        """
        X_scaled = self.X_scaler.transform(X)
        F_scaled = self.surrogate_model_F.predict(X_scaled)
        F = self.F_scaler.inverse_transform(np.array(F_scaled, ndmin=2))
        return F

    @fix_dimensions
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

        raise NotImplementedError("scaled standard deviation not implemented yet")

    @fix_dimensions
    def estimate_non_linear_constraints(self, X):
        """
        Estimate the non-linear constraints using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        out : np.ndarray
            The estimated non-linear constraints G.
        """
        X_scaled = self.X_scaler.transform(X)
        G_scaled = self.surrogate_model_G.predict(X_scaled)
        G = self.G_scaler.inverse_transform(np.array(G_scaled, ndmin=2))
        return G

    @fix_dimensions
    def estimate_nonlinear_constraints_violation(self, X):
        """
        Estimate the non-linear constraints violations using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        out : np.ndarray
            The estimated non-linear constraints violations CV.
        """
        X_scaled = self.X_scaler.transform(X)
        CV_scaled = self.surrogate_model_CV.predict(X_scaled)
        CV = self.CV_scaler.inverse_transform(np.array(CV_scaled, ndmin=2))
        return CV

    def estimate_check_nonlinear_constraints(self, X):
        """
        checks if estimated nonlinear constraints were violated.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        out : np.array(bool)
            Boolean array indicating if X were valid (basen on nonlinear
            constraints)
        """
        cv_est = self.estimate_nonlinear_constraints_violation(X)
        cv_est_ = np.array(cv_est, ndmin=2)
        ok_est = np.all(cv_est_ < 0, axis=1)
        ok_est = np.array(ok_est, ndmin=1)

        if X.ndim == 1:
            return ok_est[0]
        else:
            return ok_est

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
        raise NotImplementedError("Meta scores are not implemented yet.")

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

    def condition_linear_constraints(self, conditional_vars: dict = None):
        """
        Condition the linear constraints based on the given variables.

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
        use_surrogate=True,
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
        A_cond, b_cond = self.condition_linear_constraints(conditional_vars)

        n_lincons = op.n_linear_constraints
        n_lineqcons = op.n_linear_equality_constraints
        n_variables = op.n_independent_variables

        # nonlincons = op.nonlinear_constraints

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
                opt_vars=[v.name for v in free_variables], lhs=A_cond[i], b=b_cond[i]
            )

        # set up            start_index_nlc_surrogate new equality constraints
        [op.remove_linear_equality_constraint(0) for i in range(n_lineqcons)]
        for i in range(n_lineqcons):
            raise NotImplementedError("Linear Constraints are not yet implemented")
            # TODO: must use Aeq and Beq
            op.add_linear_equality_constraint(
                opt_vars=[v for v in free_variables], lhs=A_cond[i], beq=b_cond[i]
            )

        def complete_x(x):
            """completes X as a 1D array"""
            if len(x) != len(free_var_idx):
                raise ValueError(
                    f"x ({x}) must be of the same length as the number of free "
                    f"variables ({len(free_variables)})."
                )
            x_complete = np.zeros(n_variables)
            x_complete[cond_var_idx] = list(conditional_vars.values())
            x_complete[free_var_idx] = x

            return x_complete

        for i, evaluator in enumerate(op.evaluators):
            if i == 0:
                evaluator_func = evaluator.evaluator
                # TODO: insert a new evaluator in der chain to condition x

                # currently this will only condition the first of all evaluators

                # I think with Jos problem there is an issue that
                # the evaluations of the evaluator are cached. WIth the surrogate
                # model they should be different for objective and nonlinear_constraints
                # After modifying the order of the evaluator things make
                # more sense. Since there is only one evaluator. Modifying this,
                # already changes the evaluator for the 2nd evaluator

                # condition and return everything
                # Does not work for nonlinear constraints on the
                # true simulation function. Could it have to do with
                # an extra untransform?
                evaluator.evaluator = self.condition_model_or_surrogate(
                    model_func=evaluator_func,
                    surrogate_func=lambda res: res,
                    use_surrogate=use_surrogate,
                    complete_x=complete_x,
                    is_evaluator=True,
                )

                # set flag to show that the evaluator is conditioned
                evaluator._is_conditioned = True
            else:
                evaluator._is_conditioned = False

        conditioned_nonlincons_kwargs = []

        # generate conditioned nonlinear constraints
        start_index_nlc_surrogate = 0
        for nlc in op.nonlinear_constraints:
            nlc_func = nlc.nonlinear_constraint
            n_nlc = nlc.n_nonlinear_constraints

            if use_surrogate:
                return_idx = np.arange(
                    start=start_index_nlc_surrogate,
                    stop=start_index_nlc_surrogate + n_nlc,
                )
            else:
                return_idx = None
                # return_idx = np.arange(
                #     start=0,
                #     stop=n_nlc
                # )

            start_index_nlc_surrogate += n_nlc

            if len(nlc.evaluators) > 0:
                first_evaluator = nlc.evaluators[0]
                if not first_evaluator._is_conditioned:
                    raise AssertionError(f"{nlc} has unconditioned first evaluator")

                eval_func = first_evaluator.evaluator

                # does not need use return index when
                # first_evaluator.evaluator = self.condition_model_or_surrogate(
                #     model_func=eval_func,
                #     surrogate_func=lambda res: res,
                #     use_surrogate=use_surrogate,
                #     complete_x=complete_x,
                #     is_evaluator=True,
                # )

                # in case of the surrogate results are simply passed through
                # and correspondingly indexed during post processing
                # complete x is not necessary since x are intermediate
                # results here
                conditioned_nlc_func = self.condition_model_or_surrogate(
                    model_func=nlc_func,
                    surrogate_func=self.estimate_non_linear_constraints,
                    use_surrogate=use_surrogate,
                    complete_x=lambda x: x,
                    is_evaluator=False,
                    return_idx=return_idx,
                )

            else:
                conditioned_nlc_func = self.condition_model_or_surrogate(
                    model_func=nlc_func,
                    surrogate_func=self.estimate_non_linear_constraints,
                    use_surrogate=use_surrogate,
                    complete_x=complete_x,
                    return_idx=return_idx,
                )

            max_cv = self.CV.max(axis=0)[return_idx]
            not_so_bad_metrics = max_cv + 2 * np.abs(max_cv)
            # TODO: do we have to condition evaluation objects?
            # TODO: consider implementing the improved
            # TODO: do the nlc funcs of the conditioned NLC also have to take
            #       args and kwargs?
            # nlc.nonlinear_constraint = conditioned_nlc_func
            conditioned_nlc_kwargs = dict(
                nonlincon=conditioned_nlc_func,
                name=nlc.name,
                n_nonlinear_constraints=n_nlc,
                # bad_metrics=not_so_bad_metrics,
                bad_metrics=nlc.bad_metrics,
                evaluation_objects=nlc.evaluation_objects,
                bounds=[0.0] * n_nlc,
                # this is set later to avoid a lookup error in evaluators_dict_reference
                # TODO: refactor later
                requires=None,
                labels=nlc.labels,
                # TODO: at some points possibly args and kwargs will be needed
                # args=nlc.args,
                # kwargs=nlc.kwargs,
            )

            conditioned_nonlincons_kwargs.append(
                (conditioned_nlc_kwargs, nlc.evaluators)
            )

        op._nonlinear_constraints = []
        for i, (cnlc_kwargs, requires) in enumerate(conditioned_nonlincons_kwargs):
            op.add_nonlinear_constraint(**cnlc_kwargs)
            # Duct tape programming TODO: refactor later
            op.nonlinear_constraints[i].evaluators = requires

        assert (
            op.n_nonlinear_constraints
            == self.optimization_problem.n_nonlinear_constraints
        )

        obj_index = {}
        oi = 0
        for i, obj in enumerate(op.objectives):
            j = 0
            while j < obj.n_objectives:
                if oi in objective_index:
                    if use_surrogate:
                        obj_index.update({oi: (i, oi)})
                    else:
                        obj_index.update({oi: (i, j)})
                oi += 1
                j += 1

        # conditioned original function of otimization problem
        objectives = []
        for obj_id, (obj_func_idx, obj_return_idx) in obj_index.items():
            obj = op.objectives[obj_func_idx]
            obj_func = obj.objective
            obj.n_objectives = 1

            # if the problem has an evaluator, the evaluator needs to be
            # conditioned to take only free x and complete it with the
            # condiitoning value and return all information.
            # The objective function than needs to be conditioned to take
            # the complete input of the evaluator and return only the objective
            # of interest
            # note: This only needs
            if len(obj.evaluators) > 0:
                first_evaluator = obj.evaluators[0]
                if not first_evaluator._is_conditioned:
                    raise AssertionError(f"{obj} has unconditioned first evaluator")

                eval_func = first_evaluator.evaluator

                # condition and return everything
                # first_evaluator.evaluator = self.condition_model_or_surrogate(
                #     model_func=eval_func,
                #     surrogate_func=lambda res: res,
                #     use_surrogate=use_surrogate,
                #     complete_x=complete_x,
                #     is_evaluator=True,
                # )

                # in case of the surrogate results are simply passed through
                # and correspondingly indexed during post processing
                # complete x is not necessary since x are intermediate
                # results here

                obj.objective = self.condition_model_or_surrogate(
                    model_func=obj_func,
                    surrogate_func=self.estimate_objectives,
                    use_surrogate=use_surrogate,
                    complete_x=lambda x: x,
                    is_evaluator=False,
                    return_idx=obj_return_idx,
                )

            else:
                obj.objective = self.condition_model_or_surrogate(
                    model_func=obj_func,
                    surrogate_func=self.estimate_objectives,
                    use_surrogate=use_surrogate,
                    complete_x=complete_x,
                    return_idx=obj_return_idx,
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
        is_evaluator=False,
        return_idx=None,
    ):
        """
        convenience wrapper around condition_function
        """
        if use_surrogate:
            func = surrogate_func
            # if the surrogate has an evaluator, simply forward x but make sure
            # it's an array
            # if is_evaluator:
            #     complete_x = lambda x: np.array(x, ndmin=1)
        else:
            func = model_func
            # if not is_evaluator:
            #     complete_x = lambda res: res

        conditioned_func = self.condition_function(
            func=func,
            complete_x=complete_x,
            return_idx=return_idx,
            is_evaluator=is_evaluator,
        )

        return conditioned_func

    @staticmethod
    def condition_function(
        func,
        complete_x,
        return_idx=None,
        is_evaluator=False,
    ) -> callable:
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

            f = func(x_complete)

            if is_evaluator:
                return f

            f = np.array(f, ndmin=1)

            # index the objective of interest and cast to an array of 1-D
            if return_idx is not None:
                f_return = np.array(f[return_idx], ndmin=1)
            else:
                f_return = f

            return f_return

        return conditioned_func

    def find_x0(
        self,
        cond_var_index: int,
        cond_var_value: float,
        objective_index: list,
        x_tol=0.0,
        n_neighbors=10,
        plot=False,
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
        """
        if objective_index is None:
            objective_index = np.arange(self.optimization_problem.n_objectives)

        f = self.F[self.feasible, objective_index]
        X = self.X[self.feasible]

        x = X[:, cond_var_index].reshape((-1,))
        x_search = cond_var_value

        var = self.optimization_problem.variables[cond_var_index]

        # transform vars
        x_trans = (x - var.lb) / (var.ub - var.lb)
        f_trans = (f - np.min(f)) / (np.max(f) - np.min(f))
        x_search_trans = (x_search - var.lb) / (var.ub - var.lb)

        delta_x_trans = x_trans - x_search_trans
        # no need to calculate delta_f because: f_trans = f - f_opt = f - 0 = f
        distance = np.sqrt(delta_x_trans**2 + f_trans**2)
        closest = [i for _, i in sorted(zip(distance, range(len(distance))))]

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
            ax.scatter(x, f, color="tab:blue", alpha=0.7)
            ax.scatter([x_search], [f_search], color="tab:red", marker="o", ls="")
            ax.plot([x_search, x[il]], [f_search, f[il]], lw=0.5, color="tab:red")
            ax.plot([x_search, x[ir]], [f_search, f[ir]], lw=0.5, color="tab:red")

        return x_new_weighted

    def optimize_conditioned_problem(
        self,
        optimization_problem,
        x0,
    ):
        optimizer = TrustConstr(gtol=1e-3, xtol=1e-5, n_max_evals=100)
        # optimizer = TrustConstr()
        optimizer.optimize(
            optimization_problem,
            x0=x0,
            reinit_cache=True,
            save_results=False,
        )

        x_free = optimizer.results.x

        if len(x_free) > 1:
            assert np.allclose(np.diff(x_free, axis=0), 0, atol=0.001)

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
        lp = hopsy.LP()
        lp.reset()
        lp.settings.simplify_only = True

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
                    use_surrogate=use_surrogate,
                )

                op._callbacks = []

                try:
                    # TODO: previously `simplify=True` however, sometimes
                    # this does not work, especially when the problem is
                    # complicated!!` For Jos Problem
                    # Better solution. improve `find_x0`` to also give values
                    # near the edges of the problem
                    problem = op.create_hopsy_problem(simplify=True)
                    chebyshev_orig = hopsy.compute_chebyshev_center(problem)[:, 0]

                except ValueError as e:
                    if (
                        str(e)
                        == "All inequality constraints are redundant, implying that the polytope is a single point."
                    ):
                        problem = op.create_hopsy_problem(simplify=False)
                        chebyshev_orig = hopsy.computee_chebyshev_center(problem)[:, 0]
                        # TODO: stop here and record single optimal point
                        # chebyshev_orig = None
                        # this case is currently never encountered
                    else:
                        continue

                x0_weighted = self.find_x0(
                    cond_var_index=cond_var_idx[0],
                    objective_index=objective_index,
                    cond_var_value=x_cond,
                    x_tol=0.0,
                    n_neighbors=1,
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

                except ValueError as e:
                    if "`x0` is infeasible" in str(e):
                        warnings.warn(
                            f"Surrogate.find_minimum.optimize_conditioned_problem"
                            f"(op=op(condition={x_cond}), x0={x0}): {e} "
                            "This is likely due to a condition outside of a "
                            "feasible region."
                        )
                        continue
                    else:
                        raise e

                x_opt = np.full(n_vars, fill_value=np.nan)
                x_opt[cond_var_idx] = x_cond
                x_opt[free_var_idx] = x_free
                x_optimal[i, objective_index] = x_opt

                f = op.evaluate_objectives(x_opt[free_var_idx])
                f_minimum[i, objective_index] = f

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
                plot=False,
            )

            if x is not None:
                if optimize:
                    (
                        op,
                        cond_var_idx,
                        free_var_idx,
                    ) = self.condition_optimization_problem(
                        conditional_vars={cond_var: x_cond}, use_surrogate=True
                    )

                    try:
                        x_free = self.optimize_conditioned_problem(
                            optimization_problem=op, x0=x[free_var_idx]
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
                    cv = self.optimization_problem.evaluate_nonlinear_constraints_violations(
                        x
                    )
                    G.append(g)
                    CV.append(cv)

        self.X_new = np.array(X)
        self.F_new = np.array(F)
        self.G_new = np.array(G)
        self.CV_new = np.array(CV)

    def plot_parameter_objective_space(
        self,
        X,
        f,
        var_x,
        axes=None,
        use_surrogate=True,
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
        f_min, x_opt = self.find_minimum(x_idx, use_surrogate=use_surrogate)

        if self.feasible is not None:
            alpha = (self.feasible.astype(float) + (1 / 3)) / (4 / 3)
            color = ["tab:green" if f else "tab:blue" for f in self.feasible]

        n_objectives = self.optimization_problem.n_objectives

        if axes is None:
            fig, axes = plt.subplots(n_objectives)
            if isinstance(axes, plt.Axes):
                axes = [axes]
            self_contained_figure = True
        else:
            self_contained_figure = False

        for oi, ax in enumerate(axes):
            ax.scatter(x, f[:, oi], s=10, label="obj. fun", alpha=alpha, color=color)
            ax.plot(x_opt[:, oi, x_idx], f_min[:, oi], color="tab:red", lw=0.5)

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

        if self_contained_figure == True:
            fig.show()

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
                        x=X, f=F_mean, var_x=var_x, ax=ax, use_surrogate=use_surrogate
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
        [ax.set_ylabel(var_y) for ax, var_y in zip(axes[:, 0], variable_names)]

        fig.tight_layout()

        if self.plot_directory is not None:
            plot_directory = Path(self.plot_directory)
            fig.savefig(f'{plot_directory / "surrogate_spaces.png"}')

    def plot_population_vs_surrogate(self):
        """
        diagnostic plot to test if the surrogate model is capable of reproducing
        the target space
        """
        variables = self.optimization_problem.variables
        objectives = self.optimization_problem.objectives
        fig, axes = plt.subplots(
            nrows=len(objectives), ncols=len(variables), sharey="row", sharex="col"
        )

        new_X = self.optimization_problem.create_initial_values(n_samples=1000)

        F_est = self.estimate_objectives(new_X)
        CV_est = self.estimate_nonlinear_constraints_violation(new_X)
        feasible_est = np.all(CV_est <= 0, axis=1)

        for xi, var in enumerate(variables):
            x = self.X[:, xi]
            x_est = new_X[:, xi]
            x_lab = var.name

            for fi, obj in enumerate(objectives):
                f = self.F[:, fi]
                f_est = F_est[:, fi]
                f_lab = obj.name
                ax: plt.Axes = axes[fi, xi]
                ax.scatter(
                    x,
                    f,
                    alpha=np.where(self.feasible, 0.5, 0.05),
                    c=np.where(self.feasible, "tab:green", "black"),
                )
                ax.scatter(
                    x_est,
                    f_est,
                    alpha=np.where(feasible_est, 0.5, 0.05),
                    c=np.where(feasible_est, "tab:blue", "tab:red"),
                    marker="x",
                )
                ax.set_xlabel(x_lab)
                ax.set_ylabel(f_lab)
