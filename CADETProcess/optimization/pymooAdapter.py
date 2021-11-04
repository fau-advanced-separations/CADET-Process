from abc import abstractmethod
import copy
import time
import warnings

import numpy as np

import pymoo
from pymoo.core.problem import Problem
from pymoo.factory import get_reference_directions
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3


from CADETProcess import CADETProcessError
from CADETProcess.common import Bool, Switch, UnsignedInteger, UnsignedFloat
from CADETProcess.optimization import SolverBase, OptimizationResults


class PymooInterface(SolverBase):
    """Wrapper around pymoo.
    """
    x_tol = UnsignedFloat(default=1e-8)
    cv_tol = UnsignedFloat(default=1e-6)
    f_tol = UnsignedFloat(default=0.0025)
    pop_size = UnsignedInteger(default=100)
    nth_gen = UnsignedInteger(default=1)
    n_last = UnsignedInteger(default=30)
    n_max_gen = UnsignedInteger(default=100)
    n_max_evals = UnsignedInteger(default=100000)
    _options = [
        'x_tol', 'cv_tol', 'f_tol', 'nth_gen',
        'n_last', 'n_max_gen', 'n_max_evals',
    ] 
    
    def run(self, optimization_problem):
        """Solve the optimization problem using the functional pymoo implementation.
        
        Returns
        -------
        results : OptimizationResults
            Optimization results including optimization_problem and solver
            configuration.

        See also
        --------
        evaluate_objectives
        options

        Todo
        ----
        - [ ] checkpoints
        - [ ] enforce feasibility
        - [x] Parallelization
        - [x] multi objective in optimization problem
        - [x] linear constraints
        - [x] Reference Directions
        - [x] Initial population (using hopsy?)
        - [x] Termination
        """
        self.optimization_problem = optimization_problem
        
        ieqs = [
            lambda x: optimization_problem.evaluate_linear_constraints(x)[0]
        ]
        
        class PymooProblem(Problem):
            def __init__(self, **kwargs):
                super().__init__(
                    n_var=optimization_problem.n_variables,
                    n_obj=optimization_problem.n_objectives,
                    n_constr=optimization_problem.n_linear_constraints,
                    xl=optimization_problem.lower_bounds,
                    xu=optimization_problem.upper_bounds,
                    **kwargs
                )
        
            def _evaluate(self, x, out, *args, **kwargs):
                cache = optimization_problem.evaluate_population(x)
                
                f = []
                g = []
                for ind in x:
                    f.append(
                        optimization_problem.evaluate_objectives(ind, cache=cache)
                    )
                    g.append(
                        optimization_problem.evaluate_nonlinear_constraints(ind, cache=cache)
                    )
                        
                out["F"] = np.array(f)
                out["G"] = np.array(g)

        problem = PymooProblem()

        termination = MultiObjectiveDefaultTermination(
            x_tol=self.x_tol,
            cv_tol=self.cv_tol,
            f_tol=self.f_tol,
            nth_gen=self.nth_gen,
            n_last=self.n_last,
            n_max_gen=self.n_max_gen,
            n_max_evals=self.n_max_evals
        )

        start = time.time()
        res = minimize(
            problem,
            self.algorithm,
            termination,
            pf=problem.pareto_front(),
            seed=1,
            verbose=True,
            save_history=True,
        )
        elapsed = time.time() - start

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
    @property 
    def algorithm(self):
        algorithm = pymoo.algorithms.moo.nsga2.NSGA2(
            pop_size=self.population_size,
            sampling=self.optimization_problem(self.population_size)
        )
        return algorithm

class U_NSGA3(PymooInterface):
    @property
    def algorithm(self):
        algorithm = pymoo.algorithms.moo.unsga3.UNSGA3(
            ref_dirs=self.ref_dirs,
            pop_size=self.population_size,
            sampling=self.optimization_problem.create_initial_values(
                self.population_size, method='chebyshev'
            )
        )
        return algorithm
