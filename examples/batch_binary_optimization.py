#!/usr/bin/env python3

"""
===============================================
Optimize Batch Chromatography of Binary Mixture
===============================================

"""

from CADETProcess.simulation import Cadet, ProcessEvaluator
from CADETProcess.optimization import OptimizationProblem
from CADETProcess.common import RankedPerformance

from examples.batch_binary import batch_binary

process_simulator = Cadet()
process_simulator.evaluate_stationarity = True

ranking = [1, 1]
purity_required = [0.95, 0.95]
evaluator = ProcessEvaluator(process_simulator, purity_required, ranking)

optimization_problem = OptimizationProblem(batch_binary, evaluator, save_log=True)
optimization_problem.cache_results = True

def objective_function(performance):
    performance = RankedPerformance(performance, ranking)
    return - performance.mass * performance.recovery * performance.eluent_consumption

optimization_problem.objective_fun = objective_function

optimization_problem.add_variable('cycle_time', lb=10, ub=600)
optimization_problem.add_variable('feed_duration.time', lb=10, ub=300)

optimization_problem.add_linear_constraint(['feed_duration.time', 'cycle_time'], [1,-1])

if __name__ == "__main__":
    from CADETProcess.optimization import DEAP as Solver

    opt_solver = Solver()
    results = opt_solver.optimize(optimization_problem, save_results=True)