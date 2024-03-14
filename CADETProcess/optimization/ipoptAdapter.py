from typing import Optional

import cyipopt
import numpy as np
import numpy.typing as npt

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Switch, UnsignedFloat, UnsignedInteger
from CADETProcess.optimization import OptimizationProblem
from CADETProcess.optimization.optimizer import OptimizerBase


class _IPOPTProblem(cyipopt.Problem):
    """cyipopt.Problem subclass that bridges CADET-Process and IPOPT."""

    def __init__(
        self,
        optimizer: "IPOPT",
        optimization_problem: OptimizationProblem,
        n: int,
        m: int,
        lb: npt.ArrayLike,
        ub: npt.ArrayLike,
        cl: npt.ArrayLike,
        cu: npt.ArrayLike,
    ) -> None:
        super().__init__(n=n, m=m, lb=lb, ub=ub, cl=cl, cu=cu)
        self._optimizer = optimizer
        self._opt = optimization_problem

    def objective(self, x: npt.ArrayLike) -> float:
        try:
            return self._opt.evaluate_objectives(
                x,
                untransform=True,
                get_dependent_values=True,
                ensure_minimization=True,
            )[0]
        except ValueError:
            return 1e20

    def gradient(self, x: npt.ArrayLike) -> np.ndarray:
        try:
            return self._opt.objective_jacobian(
                x,
                untransform=True,
                get_dependent_values=True,
                ensure_minimization=True,
                dx=np.sqrt(np.finfo(float).eps),
            )[0]
        except ValueError:
            return np.zeros(len(x))

    def constraints(self, x: npt.ArrayLike) -> np.ndarray:
        opt = self._opt
        ts = opt.transformed_space
        parts = []
        if opt.n_linear_constraints > 0:
            parts.append(ts.A @ x)
        if opt.n_linear_equality_constraints > 0:
            parts.append(ts.A_eq @ x)
        if opt.n_nonlinear_constraints > 0:
            try:
                parts.append(
                    opt.evaluate_nonlinear_constraints_violation(
                        x, untransform=True, get_dependent_values=True
                    )
                )
            except ValueError:
                parts.append(np.ones(opt.n_nonlinear_constraints))
        return np.concatenate(parts) if parts else np.empty(0)

    def jacobian(self, x: npt.ArrayLike) -> np.ndarray:
        opt = self._opt
        ts = opt.transformed_space
        parts = []
        if opt.n_linear_constraints > 0:
            parts.append(ts.A)
        if opt.n_linear_equality_constraints > 0:
            parts.append(ts.A_eq)
        if opt.n_nonlinear_constraints > 0:
            parts.append(
                opt.nonlinear_constraint_jacobian(
                    x,
                    untransform=True,
                    get_dependent_values=True,
                    dx=np.sqrt(np.finfo(float).eps),
                )
            )
        return np.vstack(parts).flatten() if parts else np.empty(0)

    def intermediate(
        self,
        alg_mod: int,  # noqa: ARG002
        iter_count: int,  # noqa: ARG002
        obj_value: float,  # noqa: ARG002
        inf_pr: float,  # noqa: ARG002
        inf_du: float,  # noqa: ARG002
        mu: float,  # noqa: ARG002
        d_norm: float,  # noqa: ARG002
        regularization_size: float,  # noqa: ARG002
        alpha_du: float,  # noqa: ARG002
        alpha_pr: float,  # noqa: ARG002
        ls_trials: int,  # noqa: ARG002
    ) -> bool:
        self._optimizer.n_evals += 1
        iterate = self.get_current_iterate()
        if iterate is None:
            return True
        x = iterate["x"].tolist()
        opt = self._opt
        try:
            f = opt.evaluate_objectives(
                x, untransform=True, get_dependent_values=True, ensure_minimization=True
            )
            g = opt.evaluate_nonlinear_constraints(
                x, untransform=True, get_dependent_values=True
            )
            cv = opt.evaluate_nonlinear_constraints_violation(
                x, untransform=True, get_dependent_values=True
            )
            self._optimizer.run_post_processing(x, f, g, cv, self._optimizer.n_evals)
        except Exception:
            pass
        return True


class IPOPT(OptimizerBase):
    """Wrapper for the IPOPT optimizer via cyipopt's native Problem interface.

    Supports:
        - Linear constraints
        - Linear equality constraints
        - Nonlinear constraints
        - Bounds

    Parameters
    ----------
    tol : float, optional
        Convergence tolerance for dual infeasibility, primal infeasibility,
        and complementarity. Default is 1e-8.
    acceptable_tol : float, optional
        Looser tolerance; IPOPT terminates with an acceptable solution if
        this criterion holds for several consecutive iterations.
        Default is 1e-6.
    mu_strategy : {'adaptive', 'monotone'}, optional
        Barrier parameter update strategy. Default is 'adaptive'.
    maxiter : int, optional
        Maximum number of iterations. Default is 3000.
    """

    supports_linear_constraints = True
    supports_linear_equality_constraints = True
    supports_nonlinear_constraints = True
    supports_bounds = True

    tol = UnsignedFloat(default=1e-8)
    acceptable_tol = UnsignedFloat(default=1e-6)
    mu_strategy = Switch(valid=["adaptive", "monotone"], default="adaptive")
    maxiter = UnsignedInteger(default=3000)

    x_tol = tol
    n_max_iter = maxiter
    n_max_evals = maxiter

    def _run(
        self,
        optimization_problem: OptimizationProblem,
        x0: Optional[list] = None,
    ) -> None:
        """Solve the optimization problem using IPOPT.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            Problem to solve.
        x0 : list, optional
            Initial values in untransformed space.

        See Also
        --------
        CADETProcess.optimization.OptimizationProblem.evaluate_objectives
        cyipopt.Problem
        """
        self.n_evals = 0

        if optimization_problem.n_objectives > 1:
            raise CADETProcessError("Can only handle single objective.")

        if x0 is None:
            x0 = optimization_problem.create_initial_values(
                1, include_dependent_variables=False
            )[0]

        x0_transformed = optimization_problem.transform(x0)
        opt = optimization_problem
        ts = opt.transformed_space

        # Assemble constraint bounds.
        cl_parts = []
        cu_parts = []
        if opt.n_linear_constraints > 0:
            cl_parts.append(np.full(opt.n_linear_constraints, -np.inf))
            cu_parts.append(np.asarray(ts.b, dtype=float))
        if opt.n_linear_equality_constraints > 0:
            beq = np.asarray(ts.b_eq, dtype=float)
            cl_parts.append(beq)
            cu_parts.append(beq)
        if opt.n_nonlinear_constraints > 0:
            cl_parts.append(np.full(opt.n_nonlinear_constraints, -np.inf))
            cu_parts.append(np.zeros(opt.n_nonlinear_constraints))

        m = (
            opt.n_linear_constraints
            + opt.n_linear_equality_constraints
            + opt.n_nonlinear_constraints
        )
        cl = np.concatenate(cl_parts) if cl_parts else np.empty(0)
        cu = np.concatenate(cu_parts) if cu_parts else np.empty(0)

        nlp = _IPOPTProblem(
            optimizer=self,
            optimization_problem=opt,
            n=opt.n_independent_variables,
            m=m,
            lb=ts.lower_bounds,
            ub=ts.upper_bounds,
            cl=cl,
            cu=cu,
        )
        nlp.add_option("max_iter", self.maxiter)
        nlp.add_option("tol", float(self.tol))
        nlp.add_option("acceptable_tol", float(self.acceptable_tol))
        nlp.add_option("mu_strategy", str(self.mu_strategy))
        nlp.add_option("print_level", 0)

        x_opt, info = nlp.solve(x0_transformed)

        try:
            x_final = x_opt.tolist()
            f_final = opt.evaluate_objectives(
                x_final, untransform=True, get_dependent_values=True, ensure_minimization=True
            )
            g_final = opt.evaluate_nonlinear_constraints(
                x_final, untransform=True, get_dependent_values=True
            )
            cv_final = opt.evaluate_nonlinear_constraints_violation(
                x_final, untransform=True, get_dependent_values=True
            )
            self.run_post_processing(x_final, f_final, g_final, cv_final, self.n_evals)
        except Exception:
            pass

        self.results.success = info["status"] in (0, 1)
        self.results.exit_flag = abs(info["status"])
        self.results.exit_message = info["status_msg"].decode()

    def __str__(self) -> str:
        """str: String representation."""
        return "IPOPT"
