import os
import random
import time

import numpy as np

import pymoo
from pymoo.core.problem import Problem
from pymoo.factory import get_reference_directions
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.core.repair import Repair

from CADETProcess.common import settings
from CADETProcess.dataStructure import UnsignedInteger, UnsignedFloat
from CADETProcess.optimization import SolverBase, OptimizationResults


class PymooInterface(SolverBase):
    """Wrapper around pymoo.
    """
    seed = UnsignedInteger(default=12345)
    x_tol = UnsignedFloat(default=1e-8)
    cv_tol = UnsignedFloat(default=1e-6)
    f_tol = UnsignedFloat(default=0.0025)
    pop_size = UnsignedInteger(default=100)
    nth_gen = UnsignedInteger(default=1)
    n_last = UnsignedInteger(default=30)
    n_max_gen = UnsignedInteger(default=100)
    n_max_evals = UnsignedInteger(default=100000)
    n_cores = UnsignedInteger(default=0)
    _options = [
        'x_tol', 'cv_tol', 'f_tol', 'nth_gen',
        'n_last', 'n_max_gen', 'n_max_evals',
    ] 
    
    def run(self, optimization_problem, use_checkpoint=True):
        """Solve the optimization problem using the functional pymoo implementation.
        
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
        
        ieqs = [
            lambda x: optimization_problem.evaluate_linear_constraints(x)[0]
        ]
        
        self.problem = PymooProblem(optimization_problem, self.n_cores)

        if use_checkpoint and os.path.isfile(self.pymoo_checkpoint_path):
            random.seed(self.seed)
            algorithm, = np.load(
                self.pymoo_checkpoint_path, allow_pickle=True
            ).flatten()
        else:
            algorithm = self.setup_algorithm()
            
        start = time.time()
        while algorithm.has_next():
            algorithm.next()
            np.save(self.pymoo_checkpoint_path, algorithm)
            print(algorithm.result().X, algorithm.result().F)

        elapsed = time.time() - start
        res = algorithm.result()

        x = res.X
        eval_object = optimization_problem.set_variables(x, make_copy=True)
        if self.optimization_problem.evaluator is not None:
            frac = optimization_problem.evaluator.simulate_and_fractionate(
                eval_object,
            )
            performance = frac.performance
        else:
            frac = None
            performance = optimization_problem.evaluate(x, force=True)
        
        results = OptimizationResults(
            optimization_problem=optimization_problem,
            evaluation_object=eval_object,
            solver_name=str(self),
            solver_parameters=self.options,
            exit_flag=0,
            exit_message='success',
            time_elapsed=elapsed,
            x=res.X.tolist(),
            f=res.F,
            c=res.CV,
            frac=frac,
            performance=performance.to_dict(),
            history=res.history,
        )
        return results
    
    @property
    def pymoo_checkpoint_path(self):
        pymoo_checkpoint_path = os.path.join(
            settings.project_directory,
            self.optimization_problem.name + '/pymoo_checkpoint.npy'
        )
        return pymoo_checkpoint_path
    
    @property
    def population_size(self):
        if self.pop_size is None:
            return min(200, max(25*self.optimization_problem.n_variables,50))
        else:
            return self.pop_size

    @property
    def max_number_of_generations(self):
        if self.n_max_gen is None:
            return min(100, max(10*self.optimization_problem.n_variables,40))
        else:
            return self.n_max_gen
        
    def setup_algorithm(self):
        algorithm = pymoo.factory.get_algorithm(
            str(self),
            ref_dirs=self.ref_dirs,
            pop_size=self.population_size,
            sampling=self.optimization_problem.create_initial_values(
                self.population_size, method='chebyshev', seed=self.seed
            ),
            repair=RoundIndividuals(self.optimization_problem),
        )
        algorithm.setup(
            self.problem, termination=self.termination, seed=self.seed, verbose=True
        )
        return algorithm 
    
    @property
    def termination(self):
        termination = MultiObjectiveDefaultTermination(
            x_tol=self.x_tol,
            cv_tol=self.cv_tol,
            f_tol=self.f_tol,
            nth_gen=self.nth_gen,
            n_last=self.n_last,
            n_max_gen=self.n_max_gen,
            n_max_evals=self.n_max_evals
        )
        return termination

    @property
    def ref_dirs(self):
        ref_dirs = get_reference_directions(
            "energy", 
            self.optimization_problem.n_objectives,
            self.population_size,
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
    def __init__(self, optimization_problem, n_cores, **kwargs):
        self.optimization_problem = optimization_problem
        self.n_cores = n_cores
        super().__init__(
            n_var=optimization_problem.n_variables,
            n_obj=optimization_problem.n_objectives,
            n_constr=optimization_problem.n_linear_constraints,
            xl=optimization_problem.lower_bounds,
            xu=optimization_problem.upper_bounds,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        cache = self.optimization_problem.evaluate_population(x, self.n_cores)
        
        f = []
        g = []
        for ind in x:
            f.append(
                self.optimization_problem.evaluate_objectives(ind, cache=cache)
            )
            g.append(
                self.optimization_problem.evaluate_nonlinear_constraints(ind, cache=cache)
            )
                
        out["F"] = np.array(f)
        out["G"] = np.array(g)


class RoundIndividuals(Repair):
    def __init__(self, optimization_problem):
        self.optimization_problem = optimization_problem
        
    def _do(self, problem, pop, **kwargs):
        Z = pop.get("X")

        # Round all individuals
        Z = np.round(Z,2)
        
        # Check if linear constraints are met
        for i, ind in enumerate(Z):
            if not self.optimization_problem.check_linear_constraints(ind):
                Z[i,:] = self.optimization_problem.create_initial_values(method='random')
            
        # set the design variables for the population
        pop.set("X", Z)
        return pop