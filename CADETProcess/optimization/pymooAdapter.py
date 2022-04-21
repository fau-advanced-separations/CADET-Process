import os
import random
import time

import dill
import numpy as np

import pymoo
from pymoo.core.problem import Problem
from pymoo.factory import get_reference_directions
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.core.repair import Repair

from CADETProcess.dataStructure import UnsignedInteger, UnsignedFloat
from CADETProcess.optimization import OptimizerBase, OptimizationResults
from CADETProcess.optimization import Individual


class PymooInterface(OptimizerBase):
    """Wrapper around pymoo."""
    seed = UnsignedInteger(default=12345)
    x_tol = UnsignedFloat(default=1e-8)
    cv_tol = UnsignedFloat(default=1e-6)
    f_tol = UnsignedFloat(default=0.0025)
    pop_size = UnsignedInteger()
    nth_gen = UnsignedInteger(default=1)
    n_last = UnsignedInteger(default=30)
    n_max_gen = UnsignedInteger()
    n_max_evals = UnsignedInteger(default=100000)
    n_cores = UnsignedInteger(default=0)
    _options = [
        'x_tol', 'cv_tol', 'f_tol', 'nth_gen',
        'n_last', 'n_max_gen', 'n_max_evals',
    ]

    def run(
            self, optimization_problem,
            use_checkpoint=True, update_parameters=True):
        """Solve optimization problem using functional pymoo implementation.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            DESCRIPTION.
        use_checkpoint : bool, optional
            If True, try continuing fom checkpoint. The default is True.
        update_parameters : bool, optional
            If True, update parameters when loading from checkpoint.
            The default is True.

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
        self.optimization_problem = optimization_problem

        self.problem = PymooProblem(
            optimization_problem, self.progress.cache, self.n_cores
        )

        checkpoint_path = os.path.join(
            self.working_directory, f'{optimization_problem.name}.checkpoint'
        )

        if use_checkpoint and os.path.isfile(checkpoint_path):
            random.seed(self.seed)
            with open(checkpoint_path, "rb") as dill_file:
                algorithm = dill.load(dill_file)

            if update_parameters:
                self.update_algorithm(algorithm)
            self.progress = algorithm.progress

            self.logger.info("Continue optimization from checkpoint.")
        else:
            algorithm = self.setup_algorithm()
            algorithm.progress = self.progress

        start = time.time()

        while algorithm.has_next():
            algorithm.next()

            with open(checkpoint_path, "wb") as dill_file:
                dill.dump(algorithm, dill_file)

            for ind in algorithm.pop:
                if self.optimization_problem.n_nonlinear_constraints > 0:
                    ind = Individual(
                        ind.X.tolist(),
                        ind.F.tolist(),
                        ind.G.tolist()
                    )
                else:
                    ind = Individual(
                        ind.X.tolist(),
                        ind.F.tolist(),
                    )
                self.progress.add_individual(ind)

            self.progress.hall_of_fame = []
            for opt in algorithm.opt:
                if self.optimization_problem.n_nonlinear_constraints > 0:
                    ind = Individual(
                        opt.X.tolist(),
                        opt.F.tolist(),
                        opt.G.tolist()
                    )
                else:
                    ind = Individual(
                        opt.X.tolist(),
                        opt.F.tolist(),
                    )
                self.progress.hall_of_fame.append(ind)

            self.progress.update_history()
            self.progress.prune_cache()

            if self.progress.results_directory is not None:
                self.progress.save_callback(
                    n_cores=self.n_cores, n_gen=algorithm.n_gen
                )
                self.progress.save_progress()

            self.progress.prune_cache()

            self.logger.info(f'Finished Generation {algorithm.n_gen}')
            if self.optimization_problem.n_nonlinear_constraints > 0:
                for ind in self.progress.hall_of_fame:
                    self.logger.info(f'x: {ind.x}, f: {ind.f}, g: {ind.g}')
            else:
                for ind in self.progress.hall_of_fame:
                    self.logger.info(f'x: {ind.x}, f: {ind.f}')

        elapsed = time.time() - start

        results = OptimizationResults(
            optimization_problem=optimization_problem,
            optimizer=str(self),
            optimizer_options=self.options,
            exit_flag=0,
            exit_message='success',
            time_elapsed=elapsed,
            x=self.progress.x_hof.tolist(),
            f=self.progress.f_hof,
            g=self.progress.g_hof,
            progress=self.progress,
        )
        return results

    @property
    def _population_size(self):
        if self.pop_size is None:
            return min(400, max(50*self.optimization_problem.n_variables, 50))
        else:
            return self.pop_size

    @property
    def _max_number_of_generations(self):
        if self.n_max_gen is None:
            return min(100, max(10*self.optimization_problem.n_variables, 40))
        else:
            return self.n_max_gen

    def setup_algorithm(self):
        if self.optimization_problem.x0 is not None:
            pop = self.optimization_problem.x0_transformed
        else:
            pop = self.optimization_problem.create_initial_values(
                self._population_size, method='chebyshev', seed=self.seed
            )

        pop = np.array(pop, ndmin=2)

        if len(pop) < self._population_size:
            n_remaining = self._population_size - len(pop)
            remaining = self.optimization_problem.create_initial_values(
                n_remaining, method='chebyshev', seed=self.seed
            )
            pop.append(remaining)
        elif len(pop) > self._population_size:
            pop = pop[0:self._population_size]

        algorithm = pymoo.factory.get_algorithm(
            str(self),
            ref_dirs=self.ref_dirs,
            pop_size=self._population_size,
            sampling=pop,
            repair=RepairIndividuals(self.optimization_problem),
        )
        algorithm.setup(
            self.problem, termination=self.termination,
            seed=self.seed, verbose=True, save_history=False,
        )
        return algorithm

    def update_algorithm(self, algorithm):
        algorithm.pop_size = self._population_size
        algorithm.problem.n_cores = self.n_cores
        algorithm.termination.terminations[0].n_max_gen = \
            self._max_number_of_generations
        algorithm.has_terminated = \
            not algorithm.termination.do_continue(algorithm)

    @property
    def termination(self):
        termination = MultiObjectiveDefaultTermination(
            x_tol=self.x_tol,
            cv_tol=self.cv_tol,
            f_tol=self.f_tol,
            nth_gen=self.nth_gen,
            n_last=self.n_last,
            n_max_gen=self._max_number_of_generations,
            n_max_evals=self.n_max_evals
        )
        return termination

    @property
    def ref_dirs(self):
        ref_dirs = get_reference_directions(
            "energy",
            self.optimization_problem.n_objectives,
            self._population_size,
            seed=1
        )
        return ref_dirs


class NSGA2(PymooInterface):
    def __str__(self):
        return 'nsga2'


class U_NSGA3(PymooInterface):
    def __str__(self):
        return 'unsga3'


class PymooProblem(Problem):
    def __init__(self, optimization_problem, cache, n_cores, **kwargs):
        self.optimization_problem = optimization_problem
        self.cache = cache
        self.n_cores = n_cores

        super().__init__(
            n_var=optimization_problem.n_variables,
            n_obj=optimization_problem.n_objectives,
            n_constr=optimization_problem.n_nonlinear_constraints,
            xl=optimization_problem.lower_bounds,
            xu=optimization_problem.upper_bounds,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        opt = self.optimization_problem
        if opt.n_objectives > 0:
            f = opt.evaluate_objectives_population(
                x,
                cache=self.cache,
                n_cores=self.n_cores
                )
            out["F"] = np.array(f)

        if opt.n_nonlinear_constraints > 0:
            g = opt.evaluate_nonlinear_constraints_population(
                x,
                cache=self.cache,
                n_cores=self.n_cores
            )
            out["G"] = np.array(g)


class RepairIndividuals(Repair):
    def __init__(self, optimization_problem):
        self.optimization_problem = optimization_problem

    def _do(self, problem, pop, **kwargs):
        Z = pop.get("X")

        # Check if linear constraints are met
        for i, ind in enumerate(Z):
            if not self.optimization_problem.check_linear_constraints(ind):
                x_new = self.optimization_problem.create_initial_values(
                    method='random', set_values=False
                )
                Z[i, :] = self.optimization_problem.transform(x_new)

        # set the design variables for the population
        pop.set("X", Z)
        return pop
