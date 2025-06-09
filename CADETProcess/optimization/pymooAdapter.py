import warnings
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.ref_dirs import get_reference_directions

from CADETProcess.dataStructure import UnsignedFloat, UnsignedInteger
from CADETProcess.optimization import (
    OptimizationProblem,
    OptimizerBase,
    ParallelizationBackendBase,
)


class PymooInterface(OptimizerBase):
    """Wrapper around pymoo."""

    is_population_based = True

    supports_multi_objective = True
    supports_linear_constraints = True
    supports_linear_equality_constraints = True
    supports_nonlinear_constraints = True
    supports_bounds = True

    ignore_linear_constraints_config = True

    seed = UnsignedInteger(default=12345)
    pop_size = UnsignedInteger()

    xtol = UnsignedFloat(default=1e-8)
    ftol = UnsignedFloat(default=0.0025)
    cvtol = UnsignedFloat(default=1e-6)

    n_max_gen = UnsignedInteger()
    n_skip = UnsignedInteger(default=0)

    x_tol = xtol                # Alias for uniform interface
    f_tol = ftol                # Alias for uniform interface
    cv_nonlincon_tol = cvtol    # Alias for uniform interface
    n_max_iter = n_max_gen      # Alias for uniform interface

    _specific_options = [
        "seed",
        "pop_size",
        "xtol",
        "ftol",
        "cvtol",
        "n_max_gen",
        "n_skip",
    ]

    def _run(self, optimization_problem: OptimizationProblem, x0: Optional[list] = None) -> None:
        """
        Solve optimization problem using functional pymoo implementation.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            DESCRIPTION.
        x0 : list, optional
            Initial population of independent variables in untransformed space.

        Returns
        -------
        results : OptimizationResults
            Optimization results including optimization_problem and solver
            configuration.

        See Also
        --------
        evaluate_objectives
        options
        """
        pop_size = self.get_population_size(optimization_problem)

        if x0 is not None:
            pop = x0
        else:
            pop = optimization_problem.create_initial_values(
                pop_size, seed=self.seed, include_dependent_variables=False
            )

        pop = np.array(pop, ndmin=2)

        if len(pop) < pop_size:
            warnings.warn(
                "Initial population smaller than popsize. Creating missing entries."
            )
            n_remaining = pop_size - len(pop)
            remaining = optimization_problem.create_initial_values(
                n_remaining, seed=self.seed, include_dependent_variables=False
            )
            pop = np.vstack((pop, remaining))
        elif len(pop) > pop_size:
            warnings.warn("Initial population larger than popsize. Omitting overhead.")
            pop = pop[0:pop_size]

        pop = np.array(optimization_problem.transform(pop))

        problem = PymooProblem(optimization_problem, self.parallelization_backend)

        ref_dirs = get_reference_directions(
            "energy",
            optimization_problem.n_objectives,
            pop_size,
            seed=1,
        )

        algorithm = self._cls(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=pop,
            repair=RepairIndividuals(self, optimization_problem),
        )

        n_max_gen = self.get_max_number_of_generations(optimization_problem)

        termination = DefaultMultiObjectiveTermination(
            xtol=self.xtol,
            cvtol=self.cvtol,
            ftol=self.ftol,
            n_max_gen=n_max_gen,
            n_max_evals=self.n_max_evals,
            n_skip=self.n_skip,
        )

        algorithm.setup(
            problem,
            termination=termination,
            seed=self.seed,
            verbose=True,
            save_history=False,
            output=MultiObjectiveOutput(),
        )

        # Restore previous results from checkpoint
        for pop in self.results.populations:
            _ = algorithm.ask()
            if optimization_problem.n_nonlinear_constraints > 0:
                pop = Population.new("X", pop.x, "F", pop.f, "G", pop.cv)
                pop.apply(lambda ind: ind.evaluated.update({"F", "G"}))
                algorithm.evaluator.eval(problem, pop, evaluate_values_of=["F", "G"])
            else:
                pop = Population.new("X", pop.x, "F", pop.f)
                pop.apply(lambda ind: ind.evaluated.update({"F"}))
                algorithm.evaluator.eval(problem, pop, evaluate_values_of=["F"])
            algorithm.evaluator.n_eval += len(pop)
            algorithm.tell(infills=pop)

        while algorithm.has_next():
            # Get current generation
            pop = algorithm.ask()
            X = pop.get("X").tolist()

            # Evaluate objectives and report results
            algorithm.evaluator.eval(problem, pop)

            F = pop.get("F").tolist()
            if optimization_problem.n_nonlinear_constraints > 0:
                G = pop.get("CADET_G").tolist()
                CV = pop.get("CADET_CV").tolist()
            else:
                G = None
                CV = None

            # Handle issue of pymoo not handling np.inf
            pop.set("F", np.nan_to_num(F, posinf=1e300))

            algorithm.tell(infills=pop)

            # Post generation processing
            X_opt = algorithm.opt.get("X").tolist()
            self.run_post_processing(X, F, G, CV, algorithm.n_gen - 1, X_opt)

        if algorithm.n_gen >= n_max_gen:
            success = True
            exit_flag = 1
            exit_message = "Max number of generations exceeded."
        else:
            success = True
            exit_flag = 0
            exit_message = "Success"

        self.results.success = success
        self.results.exit_flag = exit_flag
        self.results.exit_message = exit_message

        return self.results

    def get_population_size(
            self,
            optimization_problem: OptimizationProblem,
    ) -> int:
        """
        Determine the population size for an optimization problem.

        This method calculates the population size based on the number of independent
        variables in the optimization problem. If `pop_size` is not set, it defaults to
        the minimum of 400 or the maximum of 50 times the number of independent variables
        and 50.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            The optimization problem for which to determine the population size.

        Returns
        -------
        int
            The population size.
        """
        if self.pop_size is None:
            return min(400, max(50 * optimization_problem.n_independent_variables, 50))
        else:
            return self.pop_size

    def get_max_number_of_generations(
            self,
            optimization_problem: OptimizationProblem,
    ) -> int:
        """
        Determine the maximum number of generations for an optimization problem.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            The optimization problem for which to determine the maximum number of generations.

        Returns
        -------
        int
            The maximum number of generations.
        """
        if self.n_max_gen is None:
            return min(100, max(10 * optimization_problem.n_independent_variables, 40))
        else:
            return self.n_max_gen


class NSGA2(PymooInterface):
    """NSGA2 Algorithm."""

    _cls = NSGA2

    def __str__(self) -> str:
        """str: String representation."""
        return "NSGA2"


class U_NSGA3(PymooInterface):
    """U-NSGA3 Algorithm."""

    _cls = UNSGA3

    def __str__(self) -> str:
        """str: String representation."""
        return "UNSGA3"


class PymooProblem(Problem):
    """Class to implement Pymoo Problem interface."""

    def __init__(
        self,
        optimization_problem: OptimizationProblem,
        parallelization_backend: ParallelizationBackendBase,
        **kwargs: Any,
    ) -> None:
        self.optimization_problem = optimization_problem
        self.parallelization_backend = parallelization_backend

        super().__init__(
            n_var=optimization_problem.n_independent_variables,
            n_obj=optimization_problem.n_objectives,
            n_ieq_constr=optimization_problem.n_nonlinear_constraints,
            xl=optimization_problem.lower_bounds_independent_transformed,
            xu=optimization_problem.upper_bounds_independent_transformed,
            **kwargs,
        )

    def _evaluate(self, X: npt.ArrayLike, out: dict, *args: Any, **kwargs: Any) -> None:
        opt = self.optimization_problem
        if opt.n_objectives > 0:
            F = opt.evaluate_objectives(
                X,
                untransform=True,
                get_dependent_values=True,
                ensure_minimization=True,
                parallelization_backend=self.parallelization_backend,
            )
            out["F"] = np.array(F)

        if opt.n_nonlinear_constraints > 0:
            G = opt.evaluate_nonlinear_constraints(
                X,
                untransform=True,
                get_dependent_values=True,
                parallelization_backend=self.parallelization_backend,
            )
            CV = opt.evaluate_nonlinear_constraints_violation(
                X,
                untransform=True,
                get_dependent_values=True,
                parallelization_backend=self.parallelization_backend,
            )
            out["G"] = np.array(CV)

            out["CADET_G"] = G
            out["CADET_CV"] = CV


class RepairIndividuals(Repair):
    """Class to repair individuals."""

    def __init__(
        self,
        optimizer: OptimizerBase,
        optimization_problem: OptimizationProblem,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize repair individual object."""
        self.optimizer = optimizer
        self.optimization_problem = optimization_problem
        super().__init__(*args, **kwargs)

    def _do(self, problem: OptimizationProblem, X: npt.ArrayLike, **kwargs: Any) -> npt.ArrayLike:
        # Check if linear (equality) constraints are met
        X_new = None
        for i, ind in enumerate(X):
            if not self.optimization_problem.check_individual(
                ind,
                untransform=True,
                get_dependent_values=True,
                cv_bounds_tol=self.optimizer.cv_bounds_tol,
                cv_lincon_tol=self.optimizer.cv_lincon_tol,
                cv_lineqcon_tol=self.optimizer.cv_lineqcon_tol,
                check_nonlinear_constraints=False,
            ):
                if X_new is None:
                    X_new = self.optimization_problem.create_initial_values(
                        len(X), include_dependent_variables=False
                    )
                x_new = X_new[i, :]
                X[i, :] = self.optimization_problem.transform(x_new)

        return X
