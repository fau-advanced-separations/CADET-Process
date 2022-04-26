#!/usr/bin/env python3

"""
===============================================
Optimize Batch Chromatography of Binary Mixture
===============================================

"""
import sys
sys.path.append('../../')

import numpy as np

from CADETProcess.reference import ReferenceIO
from CADETProcess.simulator import Cadet
from CADETProcess.comparison import Comparator
from CADETProcess.optimization import OptimizationProblem

# Reference Data
data = np.loadtxt('./reference_data/dextran.csv', delimiter=',')
time_experiment = data[:, 0]
dextran_experiment = data[:, 1]

reference = ReferenceIO(
    'dextran experiment', time_experiment, dextran_experiment
)

# Setup Comparator
comparator = Comparator()

comparator.add_reference(reference)

# comparator.add_difference_metric(
#     'SSE', reference, 'column.outlet'
# )
comparator.add_difference_metric(
    'Shape', reference, 'column.outlet', use_derivative=False
)

# comparator.add_difference_metric(
#     'Shape', reference, 'column.outlet', start=5*60, end=7*60,
#     components=['Dextran'],
#     reference_component_index=[0]
# )

# Setup Optimization Problem
optimization_problem = OptimizationProblem('dextran')

from reference_simulation.dextran_pulse import process
optimization_problem.add_evaluation_object(process)

optimization_problem.add_variable('flow_sheet.column.bed_porosity', lb=0.3, ub=0.6)
optimization_problem.add_variable(
    'flow_sheet.column.axial_dispersion', lb=1e-8, ub=1e-5, transform='auto'
)

simulator = Cadet(temp_dir='/dev/shm/')
optimization_problem.add_evaluator(simulator, cache=True)

optimization_problem.add_objective(
    comparator,
    n_objectives=comparator.n_metrics,
    requires=[simulator]
)


def callback(simulation_results, x, evaluation_object, results_dir='./'):
    comparator.plot_comparison(
        simulation_results,
        file_name=f'{results_dir}/{evaluation_object}_{x}_comparison.png',
    )
    plt.close()


optimization_problem.add_callback(
    callback, requires=[simulator], frequency=1
)

from CADETProcess.optimization import U_NSGA3
optimizer = U_NSGA3()
optimizer.progress_frequency = 1
optimizer.n_cores = 4
optimizer.pop_size = 4
optimizer.n_max_gen = 5

from CADETProcess import settings
settings.set_working_directory('./dextran')


optimization_results = optimizer.optimize(
    optimization_problem,
    use_checkpoint=True
)
