import copy
import time

import numpy as np

import GPyOpt as GPyOptApi

from CADETProcess.common import UnsignedInteger, UnsignedFloat
from CADETProcess.optimization import SolverBase, OptimizationResults

class GPyOpt(SolverBase):
    """Wrapper GPyOpt optimization suite for bayesian optimization.
    """
    max_iter = UnsignedInteger(default=50)
    max_time = UnsignedFloat(default=36000)
    _options = ['max_iter', 'max_time']


    def run(self, optimization_problem):
        """Running the optimization with the method .

        Runs the optimization by calling the minimze method from the optimize
        method of the scipy interface. The method is set to trust-constr.
        Therefore the function, the constraints, the solver options and the
        starting point x0 are also set. The results from the optimization are
        returned for the results of the optimization_problem.

        Parameters
        ----------
        optimization_problem : OptimizationProblem

        Returns
        -------
        results : dict
            The results of the optimization for the given optimization_problem.

        See also
        --------
        TrustConstr
        evaluate_objective_fun
        x0
        constraint_objects
        options
        scipy.optimize.minimize
        """
        start = time.time()

        # wrap objective function to convert from np.ndarray to list
        model_fun = lambda x: optimization_problem.evaluate_objective_fun(
                x.tolist()[0])

        self.gpyopt = GPyOptApi.methods.BayesianOptimization(
                f = model_fun,
                domain = self.get_domain(optimization_problem),
                constraints = self.get_constraints(optimization_problem))

        self.max_iter = min(1000, max(100*len(optimization_problem.variables),50))

        self.gpyopt.run_optimization(self.max_iter)

        elapsed = time.time() - start

        x = self.gpyopt.x_opt

        eval_object = optimization_problem.set_variables(x, make_copy=True)
        if optimization_problem.evaluator is not None:
            frac = optimization_problem.evaluator.evaluate(
                    eval_object, return_frac=True)
            performance = frac.performance
        else:
            frac = None
            performance = optimization_problem.evaluate(x, force=True)
        f = optimization_problem.objective_fun(performance)

        results = OptimizationResults(
                optimization_problem = optimization_problem,
                evaluation_object = eval_object,
                solver_name = str(self),
                solver_parameters = self.options,
                exit_flag = 1,
                exit_message = 'GPyOpt terminated successfully',
                time_elapsed = elapsed,
                x = list(x),
                f = f,
                c = None,
                frac = frac,
                performance = performance.to_dict()
                )

        return results

    def get_domain(self, optimization_problem):
        domain =  [{'name': var.name,
                 'type': 'continuous',
                 'domain': (var.lb, var.ub)}
                    for var in optimization_problem.variables]
        return domain

    def get_constraints(self, optimization_problem):
        constraints = []
        for A_row, b_row in zip(optimization_problem.A, optimization_problem.b):
            row = np.array2string(A_row, separator=',', sign='+')
            row = ''.join(c for c in row if c not in '[]')
            row = row.split(',')

            string = ''
            for index, el in enumerate(row):
                string += '{}*x[:,{}] '.format(el, index)
            string += ' - {}'.format(b_row)
            constraints.append({
                    'name': 'constraint_index',
                    'constraint': string
                        })
        return constraints
