# ---
# jupyter:
#   jupytext:
#     formats: md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
from pathlib import Path
import sys

root_dir = Path('../../../../').resolve()
sys.path.append(root_dir.as_posix())

# %% [markdown]
# ## Fit Binding Model Parameters
#
# This example demonstrates how to estimate SMA binding parameters based on multiple gradient elution chromatograms.
# It assumes familiarity with the `CADETProcess` process models, as introduced in the
# **Fit Column Transport Parameters** example.

# %%
import os

import numpy as np

from CADETProcess.processModel import ComponentSystem, FlowSheet, Process, Inlet, Outlet, LumpedRateModelWithPores
from CADETProcess.processModel import StericMassAction
from CADETProcess.reference import ReferenceIO
from CADETProcess.comparison import Comparator
from CADETProcess.simulator import Cadet
from CADETProcess.optimization import OptimizationProblem


# %% [markdown]
# ## Process creation
# To simplify the creation of multiple `Process` instances with the correct configurations,
# the process creation was wrapped into a function below:

# %%
def create_process(cv_length=30):
    # Component System
    component_system = ComponentSystem()
    component_system.add_component('Salt')
    component_system.add_component('A')

    Q_ml_min = 0.5  # ml/min
    Q_m3_s = Q_ml_min / (60 * 1e6)

    initial_salt_concentration = 50
    final_salt_concentration = 500

    # Unit Operations
    inlet = Inlet(component_system, name='inlet')
    inlet.flow_rate = Q_m3_s

    column = create_column_model(component_system, final_salt_concentration, initial_salt_concentration)

    column_volume = column.length * (column.diameter / 2) ** 2 * np.pi

    outlet = Outlet(component_system, name='outlet')

    flow_sheet = FlowSheet(component_system)

    flow_sheet.add_unit(inlet)
    flow_sheet.add_unit(column)
    flow_sheet.add_unit(outlet)

    flow_sheet.add_connection(inlet, column)
    flow_sheet.add_connection(column, outlet)

    # Process
    process = Process(flow_sheet, f'{cv_length}')

    load_duration = 9
    t_gradient_start = 90.0
    gradient_volume = cv_length * column_volume
    gradient_duration = gradient_volume / inlet.flow_rate[0]
    duration_post_gradient_wash = gradient_duration / 10 + 180
    process.cycle_time = t_gradient_start + gradient_duration + duration_post_gradient_wash
    t_gradient_end = t_gradient_start + gradient_duration

    c_load = np.array([initial_salt_concentration, 1.0])
    c_wash = np.array([initial_salt_concentration, 0.0])
    c_elute = np.array([final_salt_concentration, 0.0])
    gradient_slope = (c_elute - c_wash) / gradient_duration
    c_gradient_poly = np.array(list(zip(c_wash, gradient_slope)))
    c_post_gradient_wash = np.array([final_salt_concentration, 0.0])

    process.add_event('load', 'flow_sheet.inlet.c', c_load)
    process.add_event('wash', 'flow_sheet.inlet.c', c_wash, load_duration)
    process.add_event('grad_start', 'flow_sheet.inlet.c', c_gradient_poly, t_gradient_start)
    process.add_event('grad_end', 'flow_sheet.inlet.c', c_post_gradient_wash, t_gradient_end)

    return process


def create_column_model(component_system, final_salt_concentration, initial_salt_concentration):
    column = LumpedRateModelWithPores(component_system, name='column')
    column.length = 0.02  # m
    column.diameter = 0.008
    column.particle_radius = 1.7e-5
    column.axial_dispersion = 3.16e-7
    column.bed_porosity = 0.3
    column.particle_porosity = 0.5
    column.film_diffusion = [1e-4, 1e-4]
    binding_model = create_binding_model(component_system, final_salt_concentration)
    column.binding_model = binding_model
    column.c = [initial_salt_concentration, 0, ]
    column.cp = [initial_salt_concentration, 0]
    column.q = [binding_model.capacity, 0]
    return column


def create_binding_model(component_system, final_salt_concentration):
    binding_model = StericMassAction(component_system, name='SMA')
    binding_model.is_kinetic = True
    binding_model.characteristic_charge = [0.0, 10]
    binding_model.capacity = 400.0
    binding_model.reference_solid_phase_conc = binding_model.capacity
    binding_model.reference_liquid_phase_conc = final_salt_concentration
    binding_model.steric_factor = [0.0, 10]
    binding_model.adsorption_rate = [0.0, 10]
    binding_model.desorption_rate = [0.0, 1e3]
    return binding_model


# %% [markdown]
# ## Creating "experimental" data
#
# To run the parameter estimation algorithm, we need experimental data. This function generates _in-silico_ based
# "experimental" data for us to use.

# %%
def create_in_silico_experimental_data():
    def save_csv(results, directory=None, filename=None, units=None, noise_percentage=5):
        if not os.path.exists(directory):
            os.makedirs(directory)

        if units is None:
            units = results.solution.keys()

        for unit in units:
            solution = results.solution[unit]["outlet"].solution
            solution_without_salt = solution[:, 1:]
            solution_without_salt = (solution_without_salt + (np.random.random(solution_without_salt.shape) - 0.5)
                                     * noise_percentage / 100 * solution_without_salt.max(axis=0)
                                     )
            solution[:, 1:] = solution_without_salt
            solution_times = results.solution[unit]["outlet"].time

            full_solution = np.concatenate([np.atleast_2d(solution_times), solution.T]).T

            if filename is None:
                filename = unit + '_output.csv'

            file_path = os.path.join(directory, filename)
            with open(file_path, "w") as file_handle:
                np.savetxt(file_handle, full_solution, delimiter=",")

    simulator = Cadet()
    case_dir = "experimental_data"
    os.makedirs(case_dir, exist_ok=True)

    for cv in [5, 30, 120]:
        process = create_process(cv_length=cv)
        simulation_results = simulator.simulate(process)
        save_csv(simulation_results, directory=case_dir, filename=f"{cv}.csv", units=["outlet"])


# %% [markdown]
# Below are two convenience functions, written to simplifly loading `references` and creating `comparators`.

# %%
def load_reference(file_name, component_index=2):
    data = np.loadtxt(file_name, delimiter=',')

    time_experiment = data[:, 0]
    c_experiment = data[:, component_index]

    reference = ReferenceIO(
        f'Peak {file_name}',
        time_experiment, c_experiment,
        component_system=ComponentSystem(["A"])
    )
    return reference


def create_comparator(reference):
    comparator = Comparator(name=f"Comp {reference.name}")
    comparator.add_reference(reference)
    comparator.add_difference_metric(
        'PeakPosition', reference,
        'outlet.outlet', components=["A"]
    )
    comparator.add_difference_metric(
        'PeakHeight', reference,
        'outlet.outlet', components=["A"]
    )
    return comparator


# %%
if __name__ == '__main__':
    create_in_silico_experimental_data()

    optimization_problem = OptimizationProblem('SMA_binding_parameters')

    simulator = Cadet()
    optimization_problem.add_evaluator(simulator)

    comparators = dict()

    for cv in [5, 30, 120]:
        process = create_process(cv)

        reference = load_reference(f"experimental_data/{cv}.csv")

        optimization_problem.add_evaluation_object(process)
        comparator = create_comparator(reference)

        # Here I stored the comparators for easy access from within the callback function.
        # Note, that the dictionary keys need to be identical to the process names
        # as defined in create_process() for this to work.
        comparators[str(cv)] = comparator

        optimization_problem.add_objective(
            comparator,
            name=f"Objective {comparator.name}",
            evaluation_objects=[process],  # limit this comparator to be applied to only this one process
            n_objectives=comparator.n_metrics,
            requires=[simulator]
        )

    optimization_problem.add_variable(
        name='adsorption_rate',
        parameter_path='flow_sheet.column.binding_model.adsorption_rate',
        lb=1e-3, ub=1e3,
        transform='auto',
        indices=[1]  # modify only the protein (component index 1) parameter
    )

    optimization_problem.add_variable(
        name='desorption_rate',
        parameter_path='flow_sheet.column.binding_model.desorption_rate',
        lb=1e-3, ub=1e3,
        transform='auto',
        indices=[1]  # modify only the protein (component index 1) parameter
    )

    optimization_problem.add_variable(
        name='equilibrium_constant',
        evaluation_objects=None,
        lb=1e-4, ub=1e3,
        transform='auto',
        indices=[1]  # modify only the protein (component index 1) parameter
    )

    optimization_problem.add_variable(
        name='kinetic_constant',
        evaluation_objects=None,
        lb=1e-4, ub=1e3,
        transform='auto',
        indices=[1]  # modify only the protein (component index 1) parameter
    )

    optimization_problem.add_variable(
        name='characteristic_charge',
        parameter_path='flow_sheet.column.binding_model.characteristic_charge',
        lb=1, ub=50,
        transform='auto',
        indices=[1]  # modify only the protein (component index 1) parameter
    )

    optimization_problem.add_variable_dependency(
        dependent_variable="desorption_rate",
        independent_variables=["kinetic_constant", ],
        transform=lambda k_kin: 1 / k_kin
    )

    optimization_problem.add_variable_dependency(
        dependent_variable="adsorption_rate",
        independent_variables=["kinetic_constant", "equilibrium_constant"],
        transform=lambda k_kin, k_eq: k_eq / k_kin
    )

    def callback(simulation_results, individual, evaluation_object, callbacks_dir='./'):
        comparator = comparators[evaluation_object.name]
        comparator.plot_comparison(
            simulation_results,
            file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
            show=False
        )

    optimization_problem.add_callback(callback, requires=[simulator])

    print(optimization_problem.variable_names)
    x0 = [1, 1, 1e-2, 1e-3, 10]
    ind = optimization_problem.create_individual(x0)
    optimization_problem.evaluate_callbacks(ind)

# %% [markdown]
# ```{note}
# For performance reasons, the optimization is currently not run when building the documentation.
# In future, we will try to sideload pre-computed results to also discuss them here.
# ```

# %%
# if __name__ == '__main__':
#     from CADETProcess.optimization import U_NSGA3
#
#     optimizer = U_NSGA3()
#     optimizer.n_cores = 4
#     optimizer.pop_size = 50
#     optimizer.n_max_gen = 5
#
#     optimization_results = optimizer.optimize(
#         optimization_problem,
#         use_checkpoint=False
#     )

# %%
