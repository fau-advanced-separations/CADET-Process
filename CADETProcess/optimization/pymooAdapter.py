import os
import random
import time

import numpy as np

import pymoo
from pymoo.core.problem import Problem
from pymoo.factory import get_reference_directions
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.core.repair import Repair

from CADETProcess.dataStructure import UnsignedInteger, UnsignedFloat
from CADETProcess.optimization import OptimizerBase, OptimizationResults


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
        '_population_size', '_max_number_of_generations',
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
        self.cache = {}

        self.problem = PymooProblem(
            optimization_problem, self.cache, self.n_cores
        )

        checkpoint_path = os.path.join(
            self.working_directory, 'pymoo_checkpoint.npy'
        )

        if use_checkpoint and os.path.isfile(checkpoint_path):
            random.seed(self.seed)
            algorithm, = np.load(checkpoint_path, allow_pickle=True).flatten()
            if update_parameters:
                self.update_algorithm(algorithm)
        else:
            algorithm = self.setup_algorithm()

        start = time.time()

        while algorithm.has_next():
            algorithm.next()

            np.save(checkpoint_path, algorithm)

            res = algorithm.result()
            self.logger.info(f'Generation {gen}: x: {res.X}, f: {res.F}')

        elapsed = time.time() - start
        res = algorithm.result()

        results = OptimizationResults(
            optimization_problem=optimization_problem,
            optimizer=str(self),
            optimizer_options=self.options,
            exit_flag=0,
            exit_message='success',
            time_elapsed=elapsed,
            x=algorithm.result().X.tolist(),
            f=algorithm.result().F.tolist(),
            g=algorithm.result().G.tolist(),
            history=res.history,
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
        algorithm = pymoo.factory.get_algorithm(
            str(self),
            ref_dirs=self.ref_dirs,
            pop_size=self._population_size,
            sampling=self.optimization_problem.create_initial_values(
                self._population_size, method='chebyshev', seed=self.seed
            ),
            repair=RoundIndividuals(self.optimization_problem),
        )
        algorithm.setup(
            self.problem, termination=self.termination,
            seed=self.seed, verbose=True, save_history=True,
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


class RoundIndividuals(Repair):
    def __init__(self, optimization_problem):
        self.optimization_problem = optimization_problem

    def _do(self, problem, pop, **kwargs):
        Z = pop.get("X")

        # Round all individuals
        Z = np.round(Z, 2)

        # Check if linear constraints are met
        for i, ind in enumerate(Z):
            if not self.optimization_problem.check_linear_constraints(ind):
                Z[i, :] = self.optimization_problem.create_initial_values(
                    method='random'
                )

        # set the design variables for the population
        pop.set("X", Z)
        return pop
