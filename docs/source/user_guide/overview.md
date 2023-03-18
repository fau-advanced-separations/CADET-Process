(overview)=
# Framework Overview

**CADET-Process** is written in Python, a free and open-source programming language that gained a lot of popularity in the scientific community in recent years.
One of the main advantages of Python is the easy integration of other scientific and numerical packages.
This makes it especially useful for a modular approach such as the one presented.
The following figure gives a general overview of the program structure and workflow.

```{figure} ./figures/framework_overview.svg
:name: framework_overview

Overview of the framework modules and their relations.
White boxes represent input configurations and solution objects, blue boxes internal tools and procedures, and green boxes external tools.
For a detailed explanation, see text.
```

The {class}`~CADETProcess.processModel.Process` is an abstract representation of the chromatographic process configuration including the operational and design parameters.
Processes can be simulated using a {class}`Simulator <CADETProcess.simulator.SimulatorBase>` which solves the underlying equations.
The {class}`Simulator <CADETProcess.simulator.SimulatorBase>` adapter acts as an abstract interface to external solvers (e.g. **CADET**) and translates the internal configuration to the corresponding format of the solver.
After the computation is finished, the {class}`~CADETProcess.simulationResults.SimulationResults` are returned and can be further evaluated (see {ref}`simulation_guide`).
If a {class}`~CADETProcess.stationarity.StationarityEvaluator` is configured to test for cyclic stationarity, more chromatographic cycles are be simulated until stationarity is reached (see {ref}`stationarity_guide`).

For processing the {class}`~CADETProcess.simulationResults.SimulationResults`, different modules are provided.
For example, they can be compared to experimental data (or other simulations) using a {class}`~CADETProcess.comparison.Comparator` which computes residuals such as the sum of squared errors (see also {ref}`comparison_guide`).
Moreover, the {class}`~CADETProcess.fractionation.Fractionator` module automatically determines fractionation times of the simulated chromatograms and determines process performance indicators such as purity, yield, and productivity (see {ref}`fractionation_guide`).

These metrics can be used as objectives in an {class}`~CADETProcess.optimization.OptimizationProblem` class which serves to configure optimization studies.
Here, any process parameter can be added as optimization variable and the evaluation methods can be used to construct objectives and constraint functions.
This enables many different scenarios such as {ref}`process optimization <batch_elution_optimization_single>` and {ref}`parameter estimation <fit_column_transport>`.
Again, an abstract {class}`Optimizer <CADETProcess.optimization.OptimizerBase>` provides an interface to external optimization algorithms such as {class}`U-NSGA-3 <CADETProcess.optimization.U_NSGA3>` (see {ref}`optimization_guide`).
