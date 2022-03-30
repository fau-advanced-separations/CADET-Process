from abc import abstractmethod
import copy
import time
import warnings

import numpy as np
import hopsy

from CADETProcess import CADETProcessError
from CADETProcess.common import UnsignedInteger
from CADETProcess.optimization import OptimizerBase, OptimizationResults


class HopsyInterface(OptimizerBase):
    """Wrapper around hopsy's parameter sampling suite."""
    n_samples = UnsignedInteger(default=1000)

    @abstractmethod
    def evaluation_routine():
        return

    @abstractmethod
    def model():
        return

    def run(self, optimization_problem):
        """Solve the optimization problem using any of the scipy methodss

        Returns
        -------
        results : OptimizationResults
            Optimization results including optimization_problem and solver
            configuration.

        See Also
        --------
        evaluate_objective_fun
        options
        
        """
        cache = dict()
        start = time.time()
        success = self.evaluation_routine(optimization_problem, cache)
        elapsed = time.time() - start

        return cache

        # min_results = results.min()
        # min_index = np.where(results == min_results)[0][0]
        # x = results[min_index]

        # eval_object = copy.deepcopy(optimization_problem.evaluation_object)
        # if optimization_problem.evaluator is not None:
        #     frac = self.evaluator.evaluate(eval_object)
        #     performance = frac.performance
        # else:
        #     frac = None
        #     performance = optimization_problem.evaluate(x, force=True)

        # f = optimization_problem.objective_fun(performance)
        # c = optimization_problem.evaluate_nonlinear_constraint_fun(x)

        # results = OptimizationResults(
        #         optimization_problem = optimization_problem.name,
        #         evaluation_object = eval_object,
        #         optimizer = str(self),
        #         optimizer_options = self.options,
        #         exit_flag = 0,
        #         exit_message = 'success',
        #         time_elapsed = elapsed,
        #         x = list(x),
        #         f = f,
        #         c = c,
        #         frac = frac,
        #         performance = performance.to_dict()
        #         )


    def __str__(self):
        return self.__class__.__name__


class ModelWrapper:
    def __init__(self, optimization_problem, cache):
        self._optimization_problem = optimization_problem
        self._cache = cache

    def compute_negative_log_likelihood(self, x):
        f = self._optimization_problem.evaluate_objective_fun(
            x, cache=self._cache
        )
        return f


class HitAndRun(HopsyInterface):
    def evaluation_routine(self, optimization_problem, cache):
        """Solve the optimization problem using hit and run algorithm.

        Returns
        -------
        states : np.array

        See Also
        --------
        evaluate_objective_fun
        options
        """
        model = ModelWrapper(optimization_problem, cache)

        problem = hopsy.Problem(
            optimization_problem.A,
            optimization_problem.b,
            model
        )
        problem = hopsy.add_box_constraints(
            problem,
            optimization_problem.lower_bounds,
            optimization_problem.upper_bounds
        )

        run = hopsy.Run(problem)

        run.starting_points = [hopsy.compute_chebyshev_center(problem)]

        run.sample_until_convergence = True
        run.number_of_samples = self.n_samples

        success = True
        try:
            run.sample()
        except:
            success = False
            raise CADETProcessError('Optimization Failed')

        return success


from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial

class UniformScreening(HopsyInterface):
    """Uniform Model for HOPSY from toolbox.
    """
    def evaluation_routine(self, optimization_problem, cache):
        """Solve the problem by uniformly sampling the parameter space.

        Returns
        -------
        success: Bool
            True if successful, False otherwise.

        See Also
        --------
        evaluate_objective_fun
        options
        """
        population = self.get_population(optimization_problem)

        success = True

        # fun = lambda x: optimization_problem.evaluate_objective_fun(x, cache=cache)
        # with Pool(processes=4) as pool:
        #     pool.map(fun, population)

        try:
            for i, ind in enumerate(population):
                print(i)
                optimization_problem.evaluate_objective_fun(ind, cache=cache)
            #     # Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
            #     Parallel(n_jobs=2)(
            #         delayed(optimization_problem.evaluate_objective_fun)(i, make_copy=True, cache=cache)
            #         for i in population
            #     )
        except:
            raise
            success = False

        return success
