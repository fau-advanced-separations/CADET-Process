import numpy as np

from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3

from CADETProcess.dataStructure import UnsignedInteger, UnsignedFloat
from CADETProcess.optimization import OptimizerBase


class PymooInterface(OptimizerBase):
    """Wrapper around pymoo."""
    supports_multi_objective = True
    supports_linear_constraints = True
    supports_linear_equality_constraints = True
    supports_nonlinear_constraints = True

    seed = UnsignedInteger(default=12345)
    pop_size = UnsignedInteger()
    xtol = UnsignedFloat(default=1e-8)
    cvtol = UnsignedFloat(default=1e-6)
    cv_tol = cvtol
    ftol = UnsignedFloat(default=0.0025)
    n_max_gen = UnsignedInteger()
    n_max_evals = UnsignedInteger(default=100000)
    _options = [
        'seed', 'pop_size', 'xtol', 'cvtol', 'ftol', 'n_max_gen', 'n_max_evals',
    ]

    def run(self, optimization_problem, x0=None):
        """Solve optimization problem using functional pymoo implementation.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            DESCRIPTION.
        x0 : list, optional
            Initial population

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
        self.remove_similar = False

        if x0 is not None:
            pop = x0
        else:
            pop = optimization_problem.create_initial_values(
                self._population_size, method='chebyshev', seed=self.seed
            )
            pop = optimization_problem.transform(pop)

        pop = np.array(pop, ndmin=2)

        if len(pop) < self._population_size:
            n_remaining = self._population_size - len(pop)
            remaining = optimization_problem.create_initial_values(
                n_remaining, method='chebyshev', seed=self.seed
            )
            pop = np.vstack((pop, remaining))
        elif len(pop) > self._population_size:
            pop = pop[0:self._population_size]

        problem = PymooProblem(optimization_problem, self.n_cores)

        ref_dirs = get_reference_directions(
            "energy",
            optimization_problem.n_objectives,
            self._population_size,
            seed=1
        )

        algorithm = self._cls(
            ref_dirs=ref_dirs,
            pop_size=self._population_size,
            sampling=pop,
            repair=RepairIndividuals(optimization_problem),
        )

        termination = DefaultMultiObjectiveTermination(
            xtol=self.xtol,
            cvtol=self.cvtol,
            ftol=self.ftol,
            n_max_gen=self._max_number_of_generations,
            n_max_evals=self.n_max_evals
        )

        algorithm.setup(
            problem, termination=termination,
            seed=self.seed, verbose=True, save_history=False,
            output=MultiObjectiveOutput(),
        )

        n_gen = 1
        while algorithm.has_next():
            # Get current generation
            pop = algorithm.ask()
            X = pop.get("X").tolist()

            # Evaluate objectives and report results
            algorithm.evaluator.eval(problem, pop)

            F = pop.get("F").tolist()
            if self.optimization_problem.n_nonlinear_constraints > 0:
                G = pop.get("G").tolist()
            else:
                G = len(X)*[None]

            algorithm.tell(infills=pop)

            # Post generation processing
            X_opt = algorithm.opt.get("X").tolist()
            self.run_post_generation_processing(X, F, G, n_gen, X_opt)


            n_gen += 1

        if n_gen >= self._max_number_of_generations:
            exit_message = 'Max number of generations exceeded.'
            exit_flag = 1
        else:
            exit_flag = 0
            exit_message = 'success'

        self.results.exit_flag = exit_flag
        self.results.exit_message = exit_message

        return self.results

    @property
    def _population_size(self):
        if self.pop_size is None:
            return min(
                400, max(
                    50*self.optimization_problem.n_independent_variables, 50
                )
            )
        else:
            return self.pop_size

    @property
    def _max_number_of_generations(self):
        if self.n_max_gen is None:
            return min(
                100, max(
                    10*self.optimization_problem.n_independent_variables, 40
                    )
            )
        else:
            return self.n_max_gen


class NSGA2(PymooInterface):
    _cls = NSGA2

    def __str__(self):
        return 'NSGA2'


class U_NSGA3(PymooInterface):
    _cls = UNSGA3

    def __str__(self):
        return 'UNSGA3'


class PymooProblem(Problem):
    def __init__(self, optimization_problem, n_cores, **kwargs):
        self.optimization_problem = optimization_problem
        self.n_cores = n_cores

        super().__init__(
            n_var=optimization_problem.n_independent_variables,
            n_obj=optimization_problem.n_objectives,
            n_constr=optimization_problem.n_nonlinear_constraints,
            xl=optimization_problem.lower_bounds_independent_transformed,
            xu=optimization_problem.upper_bounds_independent_transformed,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        opt = self.optimization_problem
        if opt.n_objectives > 0:
            F = opt.evaluate_objectives_population(
                x,
                untransform=True,
                n_cores=self.n_cores,
            )
            out["F"] = np.array(F)

        if opt.n_nonlinear_constraints > 0:
            G = opt.evaluate_nonlinear_constraints_population(
                x,
                untransform=True,
                n_cores=self.n_cores,
            )
            out["G"] = np.array(G)


class RepairIndividuals(Repair):
    def __init__(self, optimization_problem, *args, **kwargs):
        self.optimization_problem = optimization_problem
        super().__init__(*args, **kwargs)

    def _do(self, problem, Z, **kwargs):
        # Check if linear constraints are met
        for i, ind in enumerate(Z):
            if not self.optimization_problem.check_linear_constraints(
                    ind, untransform=True, get_dependent_values=True):
                x_new = self.optimization_problem.create_initial_values(method='random')
                Z[i, :] = self.optimization_problem.transform(x_new)

        return Z
