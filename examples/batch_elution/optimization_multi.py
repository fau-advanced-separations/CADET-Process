# ---
# jupyter:
#   jupytext:
#     formats: md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# (batch_elution_optimization_multi)=
# # Optimize Batch Elution Process (Multi-Objective)
#
# ## Setup Optimization Problem

# %%
from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('batch_elution_multi')

from examples.batch_elution.process import process
optimization_problem.add_evaluation_object(process)

optimization_problem.add_variable('cycle_time', lb=10, ub=600)
optimization_problem.add_variable('feed_duration.time', lb=10, ub=300)

optimization_problem.add_linear_constraint(
    ['feed_duration.time', 'cycle_time'], [1, -1]
)

# %% [markdown]
# ## Setup Simulator

# %%
from CADETProcess.simulator import Cadet
process_simulator = Cadet()
process_simulator.evaluate_stationarity = True

optimization_problem.add_evaluator(process_simulator)

# %% [markdown]
# ## Setup Fractionator

# %%
from CADETProcess.fractionation import FractionationOptimizer
frac_opt = FractionationOptimizer()

optimization_problem.add_evaluator(
    frac_opt,
    kwargs={
        'purity_required': [0.95, 0.95],
        'ignore_failed': False,
        'allow_empty_fractions': False,
    }
)

# %% [markdown]
# ## Setup Objectives

# %%
from CADETProcess.performance import Productivity, Recovery, EluentConsumption

productivity = Productivity()
optimization_problem.add_objective(
    productivity,
    n_objectives=2,
    requires=[process_simulator, frac_opt],
    minimize=False,
)

recovery = Recovery()
optimization_problem.add_objective(
    recovery,
    n_objectives=2,
    requires=[process_simulator, frac_opt],
    minimize=False,
)

eluent_consumption = EluentConsumption()
optimization_problem.add_objective(
    eluent_consumption,
    n_objectives=2,
    requires=[process_simulator, frac_opt],
    minimize=False,
)


# %% [markdown]
# ## Add callback for post-processing

# %%
def callback(fractionation, individual, evaluation_object, callbacks_dir):
    fractionation.plot_fraction_signal(
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_fractionation.png',
        show=False
    )

optimization_problem.add_callback(
    callback, requires=[process_simulator, frac_opt]
)

# %% [markdown]
# ## Configure Optimizer

# %%
from CADETProcess.optimization import U_NSGA3
optimizer = U_NSGA3()

# %% [markdown]
# ## Run Optimization
#
# ```{note}
# For performance reasons, the optimization is currently not run when building the documentation.
# In future, we will try to sideload pre-computed results to also discuss them here.
# ```
#
# ```
# if __name__ == '__main__':
#     results = optimizer.optimize(
#         optimization_problem,
#         use_checkpoint=True,
#     )
# ```
