"""BoFire Bayesian optimization adapter.

Wraps BoFire's ask/tell strategy interface for single-objective
(``SoboStrategy``) and multi-objective (``MoboStrategy``) Bayesian
optimization with optional nonlinear constraint handling.
"""

from typing import Any, Optional

import bofire.strategies.api as bofire_strategies
import numpy as np
import pandas as pd
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.aggregation import ScaleKernel
from bofire.data_models.kernels.continuous import (
    MaternKernel,
    RBFKernel,
)
from bofire.data_models.objectives.api import (
    MinimizeObjective,
    MinimizeSigmoidObjective,
)
from bofire.data_models.strategies.api import (
    MoboStrategy,
    QparegoStrategy,
    SoboStrategy,
)
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates

from CADETProcess.dataStructure import Switch, UnsignedFloat, UnsignedInteger
from CADETProcess.optimization import OptimizationProblem
from CADETProcess.optimization.optimizer import OptimizerBase


def _build_domain(
    optimization_problem: OptimizationProblem,
    steepness: float = 10.0,
) -> tuple[Domain, list[str], list[str], list[str]]:
    """Construct a BoFire ``Domain`` from an ``OptimizationProblem``.

    Returns the domain and three parallel lists of column names used in the
    ask/tell DataFrames: input keys, objective output keys, and constraint
    output keys.

    Parameters
    ----------
    optimization_problem
        The CADET-Process optimization problem.

    Returns
    -------
    domain : Domain
    input_keys : list[str]
    objective_keys : list[str]
    constraint_keys : list[str]
    """
    opt = optimization_problem
    ts = opt.transformed_space

    # --- Inputs (one per independent variable, in transformed space) ---
    input_features: list[ContinuousInput] = []
    input_keys: list[str] = []
    for i, var in enumerate(opt.independent_variables):
        key = f"x_{var.name}"
        input_features.append(
            ContinuousInput(
                key=key,
                bounds=(float(ts.lower_bounds[i]), float(ts.upper_bounds[i])),
            )
        )
        input_keys.append(key)

    # --- Objective outputs ---
    objective_keys: list[str] = []
    output_features: list[ContinuousOutput] = []
    for label in opt.objective_labels:
        key = f"obj_{label}"
        objective_keys.append(key)
        output_features.append(
            ContinuousOutput(
                key=key,
                objective=MinimizeObjective(w=1.0),
            )
        )

    # --- Nonlinear constraint outputs ---
    # CADET-Process convention: constraint violation <= 0 is feasible.
    # Map to MinimizeSigmoidObjective with turning point at 0: the strategy
    # learns a GP over the constraint value and prefers the region <= 0.
    constraint_keys: list[str] = []
    for label in opt.nonlinear_constraint_labels:
        key = f"con_{label}"
        constraint_keys.append(key)
        output_features.append(
            ContinuousOutput(
                key=key,
                objective=MinimizeSigmoidObjective(w=1.0, tp=0.0, steepness=steepness),
            )
        )

    # --- Linear constraints (on the transformed/input space) ---
    bofire_constraints: list[Any] = []
    if opt.n_linear_constraints > 0:
        A = ts.A
        b = ts.b
        for row_idx in range(A.shape[0]):
            coeffs = A[row_idx].tolist()
            # BoFire LinearInequalityConstraint: coefficients @ x <= rhs
            bofire_constraints.append(
                LinearInequalityConstraint(
                    features=input_keys,
                    coefficients=coeffs,
                    rhs=float(b[row_idx]),
                )
            )
    if opt.n_linear_equality_constraints > 0:
        A_eq = ts.A_eq
        b_eq = ts.b_eq
        for row_idx in range(A_eq.shape[0]):
            coeffs = A_eq[row_idx].tolist()
            bofire_constraints.append(
                LinearEqualityConstraint(
                    features=input_keys,
                    coefficients=coeffs,
                    rhs=float(b_eq[row_idx]),
                )
            )

    domain = Domain(
        inputs=Inputs(features=input_features),
        outputs=Outputs(features=output_features),
        constraints=Constraints(constraints=bofire_constraints)
        if bofire_constraints
        else Constraints(constraints=[]),
    )
    return domain, input_keys, objective_keys, constraint_keys


def _x_to_dataframe(
    X: np.ndarray,
    input_keys: list[str],
) -> pd.DataFrame:
    """Convert a 2-D array of transformed parameter vectors to a DataFrame."""
    X = np.atleast_2d(X)
    return pd.DataFrame(X, columns=input_keys)


def _dataframe_to_x(
    df: pd.DataFrame,
    input_keys: list[str],
) -> np.ndarray:
    """Extract the input columns from a candidates DataFrame as a 2-D array."""
    return df[input_keys].to_numpy()


def _evaluate_batch(
    opt: OptimizationProblem,
    X_transformed: np.ndarray,
    parallelization_backend: Any,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Evaluate objectives and constraints for a batch of transformed points."""
    F = np.atleast_2d(
        opt.evaluate_objectives(
            X_transformed,
            untransform=True,
            get_dependent_values=True,
            ensure_minimization=True,
            parallelization_backend=parallelization_backend,
        )
    )
    G = None
    CV = None
    if opt.n_nonlinear_constraints > 0:
        G = np.atleast_2d(
            opt.evaluate_nonlinear_constraints(
                X_transformed,
                untransform=True,
                get_dependent_values=True,
                parallelization_backend=parallelization_backend,
            )
        )
        CV = np.atleast_2d(
            opt.evaluate_nonlinear_constraints_violation(
                X_transformed,
                untransform=True,
                get_dependent_values=True,
                parallelization_backend=parallelization_backend,
            )
        )
    return F, G, CV


def _build_experiments(
    X_transformed: np.ndarray,
    input_keys: list[str],
    F: np.ndarray,
    obj_keys: list[str],
    CV: np.ndarray | None,
    con_keys: list[str],
) -> pd.DataFrame:
    """Assemble an experiments DataFrame for ``strategy.tell()``."""
    df = _x_to_dataframe(X_transformed, input_keys)
    for j, key in enumerate(obj_keys):
        df[key] = F[:, j]
    if con_keys and CV is not None:
        for j, key in enumerate(con_keys):
            df[key] = CV[:, j]
    return df


def _population_to_experiments(
    pop: Any,
    input_keys: list[str],
    obj_keys: list[str],
    con_keys: list[str],
) -> pd.DataFrame:
    """Rebuild an experiments DataFrame from a saved Population."""
    df = _x_to_dataframe(pop.x_transformed, input_keys)
    F = np.atleast_2d(pop.f_minimized)
    for j, key in enumerate(obj_keys):
        df[key] = F[:, j]
    if con_keys and pop.cv_nonlincon is not None:
        CV = np.atleast_2d(pop.cv_nonlincon)
        for j, key in enumerate(con_keys):
            df[key] = CV[:, j]
    return df


_KERNEL_MAP = {
    "matern_2.5": lambda: ScaleKernel(base_kernel=MaternKernel(ard=True, nu=2.5)),
    "matern_1.5": lambda: ScaleKernel(base_kernel=MaternKernel(ard=True, nu=1.5)),
    "rbf": lambda: ScaleKernel(base_kernel=RBFKernel(ard=True)),
}


def _build_surrogate_specs(
    domain: Domain,
    kernel: str,
) -> BotorchSurrogates | None:
    """Build surrogate specs with the requested kernel, or None for default."""
    if kernel == "auto":
        return None

    kernel_obj = _KERNEL_MAP[kernel]()
    surrogates = []
    for output_feature in domain.outputs.features:
        surrogates.append(
            SingleTaskGPSurrogate(
                inputs=domain.inputs,
                outputs=Outputs(features=[output_feature]),
                kernel=kernel_obj,
            )
        )
    return BotorchSurrogates(surrogates=surrogates)


class BoFire(OptimizerBase):
    """Bayesian optimization via BoFire.

    Uses ``SoboStrategy`` for single-objective and ``MoboStrategy`` for
    multi-objective problems.  Nonlinear constraints are modeled as
    additional GP outputs with sigmoid objectives so that the acquisition
    function preferentially explores feasible regions.

    Parameters
    ----------
    n_init : int, optional
        Number of initial samples drawn via ``create_initial_values``
        before fitting the first surrogate.  Default is 10.
    batch_size : int, optional
        Number of candidates requested per ask/tell iteration.
        Default is 1.
    seed : int, optional
        Random seed for the BoFire strategy.  Default is 12345.
    steepness : float, optional
        Steepness of the sigmoid objective used for nonlinear constraints.
        Higher values create a sharper feasibility boundary.  Default is 10.
    strategy : {"auto", "sobo", "mobo", "qparego"}, optional
        BO strategy.  ``"auto"`` selects ``SoboStrategy`` for single-objective
        and ``MoboStrategy`` for multi-objective problems.  ``"qparego"``
        scalarizes multi-objective problems with random Chebyshev weights,
        using a single GP instead of one per objective.  Default is ``"auto"``.
    kernel : {"auto", "matern_2.5", "matern_1.5", "rbf"}, optional
        GP kernel.  ``"auto"`` uses BoFire's default (RBF with dimensionality-
        scaled priors).  Default is ``"auto"``.
    """

    is_population_based = True
    supports_single_objective = True
    supports_multi_objective = True
    supports_bounds = True
    supports_linear_constraints = True
    supports_linear_equality_constraints = True
    supports_nonlinear_constraints = True
    ignore_linear_constraints_config = True

    n_init = UnsignedInteger(default=10)
    batch_size = UnsignedInteger(default=1)
    seed = UnsignedInteger(default=12345)
    steepness = UnsignedFloat(default=10.0)
    strategy = Switch(
        default="auto", valid=["auto", "sobo", "mobo", "qparego"]
    )
    kernel = Switch(default="auto", valid=["auto", "matern_2.5", "matern_1.5", "rbf"])

    def _run(
        self,
        optimization_problem: OptimizationProblem,
        x0: Optional[list] = None,
    ) -> None:
        opt = optimization_problem
        n_init = self.n_init

        domain, input_keys, obj_keys, con_keys = _build_domain(
            opt, steepness=self.steepness
        )

        # --- Select strategy ---
        surrogate_specs = _build_surrogate_specs(domain, self.kernel)
        strategy_kwargs: dict[str, Any] = {"domain": domain, "seed": self.seed}
        if surrogate_specs is not None:
            strategy_kwargs["surrogate_specs"] = surrogate_specs

        strategy_choice = self.strategy
        if strategy_choice == "auto":
            strategy_choice = "sobo" if opt.n_objectives == 1 else "mobo"

        strategy_cls = {
            "sobo": SoboStrategy,
            "mobo": MoboStrategy,
            "qparego": QparegoStrategy,
        }[strategy_choice]
        strategy = bofire_strategies.map(strategy_cls(**strategy_kwargs))

        # --- Restore from checkpoint ---
        if self.results.populations:
            for pop in self.results.populations:
                df = _population_to_experiments(
                    pop, input_keys, obj_keys, con_keys
                )
                strategy.tell(df)
            generation = len(self.results.populations)
            n_evals = self.results.n_evals
        else:
            # --- Initial samples ---
            if x0 is not None:
                X_init = np.atleast_2d(x0)
            else:
                X_init = opt.create_initial_values(
                    n_init, seed=self.seed, include_dependent_variables=False
                )

            X_init_transformed = np.array(
                [opt.transform(x) for x in X_init]
            )

            F_init, G_init, CV_init = _evaluate_batch(
                opt, X_init_transformed, self.parallelization_backend
            )

            self.run_post_processing(
                X_init_transformed.tolist(),
                F_init.tolist(),
                G_init.tolist() if G_init is not None else None,
                0,
            )

            experiments = _build_experiments(
                X_init_transformed, input_keys,
                F_init, obj_keys,
                CV_init, con_keys,
            )
            strategy.tell(experiments)
            generation = 1
            n_evals = len(X_init)
        while n_evals < self.n_max_evals:
            candidates = strategy.ask(self.batch_size)
            X_batch = _dataframe_to_x(candidates, input_keys)

            F_batch, G_batch, CV_batch = _evaluate_batch(
                opt, X_batch, self.parallelization_backend
            )

            new_experiments = _build_experiments(
                X_batch, input_keys,
                F_batch, obj_keys,
                CV_batch, con_keys,
            )
            strategy.tell(new_experiments)

            self.run_post_processing(
                X_batch.tolist(),
                F_batch.tolist(),
                G_batch.tolist() if G_batch is not None else None,
                generation,
            )

            n_evals += len(X_batch)
            generation += 1

        self.results.success = True
        self.results.exit_flag = 0
        self.results.exit_message = "Maximum number of evaluations reached."

    def __str__(self) -> str:  # noqa: D105
        return "BoFire"
