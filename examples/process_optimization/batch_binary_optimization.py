#!/usr/bin/env python3

"""
===============================================
Optimize Batch Chromatography of Binary Mixture
===============================================

"""

import sys
sys.path.append('../../')

from CADETProcess.simulation import Cadet, ProcessEvaluator
from CADETProcess.fractionation import FractionationOptimizer
from CADETProcess.optimization import OptimizationProblem
from CADETProcess.common import RankedPerformance

from examples.forward_simulation.batch_binary import batch_binary

process_simulator = Cadet()
process_simulator.evaluate_stationarity = True

purity_required = [0.95, 0.95]
ranking = [1, 1]
def fractionation_objective(performance):
    performance = RankedPerformance(performance, ranking)
    return - performance.mass

fractionation_optimizer = FractionationOptimizer(purity_required, fractionation_objective)

evaluator = ProcessEvaluator(process_simulator, fractionation_optimizer)

optimization_problem = OptimizationProblem(batch_binary, evaluator, save_log=True)

def objective_function(performance):
    performance = RankedPerformance(performance, ranking)
    return - performance.mass * performance.recovery * performance.eluent_consumption

optimization_problem.add_objective(objective_function)

optimization_problem.add_variable('cycle_time', lb=10, ub=600)
optimization_problem.add_variable('feed_duration.time', lb=10, ub=300)

optimization_problem.add_linear_constraint(['feed_duration.time', 'cycle_time'], [1,-1])

if __name__ == "__main__":
    from CADETProcess.optimization import DEAP as Solver

    opt_solver = Solver()
    results = opt_solver.optimize(optimization_problem, save_results=True, use_multicore=False)