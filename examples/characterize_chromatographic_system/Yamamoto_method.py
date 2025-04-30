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
# # The Yamamoto method
#
# This example demonstrates how to estimate SMA binding parameters based on multiple gradient elution chromatograms
# using the Yamamoto method.

# %%
import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.tools.yamamoto import GradientExperiment, fit_parameters

from binding_model_parameters import create_column_model, create_in_silico_experimental_data

if __name__ == "__main__":
    component_system = ComponentSystem(['Salt', 'Protein'])
    column = create_column_model(component_system, final_salt_concentration=600, initial_salt_concentration=50)

    column_volume = column.length * ((column.diameter / 2) ** 2) * np.pi

    create_in_silico_experimental_data()

    exp_5cv = np.loadtxt("experimental_data/5.csv", delimiter=",")
    exp_30cv = np.loadtxt("experimental_data/30.csv", delimiter=",")
    exp_120cv = np.loadtxt("experimental_data/120.csv", delimiter=",")

    experiment_1 = GradientExperiment(exp_5cv[:, 0], exp_5cv[:, 1], exp_5cv[:, 2], 5 * column_volume)
    experiment_2 = GradientExperiment(exp_30cv[:, 0], exp_30cv[:, 1], exp_30cv[:, 2], 30 * column_volume)
    experiment_3 = GradientExperiment(exp_120cv[:, 0], exp_120cv[:, 1], exp_120cv[:, 2], 120 * column_volume)

    experiments = [experiment_1, experiment_2, experiment_3]

    for experiment in experiments:
        experiment.plot()

    yamamoto_results = fit_parameters(experiments, column)

    print('yamamoto_results.characteristic_charge =', yamamoto_results.characteristic_charge)
    print('yamamoto_results.k_eq =', yamamoto_results.k_eq)

    yamamoto_results.plot()
