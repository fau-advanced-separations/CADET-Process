(overview)=
# Overview

**CADET-Process** is written in Python, a free and open-source programming language that gained a lot of popularity in the scientific community in recent years.
One of the main advantages of Python is the easy integration of other scientific and numerical packages.
This makes it especially useful for a modular approach such as the one presented.
The following {ref}`figure <framework_overview>` gives a general overview of the program structure and workflow.

```{figure} ./figures/framework_overview.svg
:name: framework_overview

Overview of the framework modules and their relations.
White boxes represent input configurations and solution objects, blue boxes internal tools and procedures, and green boxes external tools.
For a detailed explanation, see text.
```

The `ProcessModel` is an abstract representation of the chromatographic process configuration including the operational and design parameters.
Processes can be simulated using a `ProcessSimulator` which solves the underlying equations.
The `ProcesSimulator` adapter acts as an abstract interface to external solvers (e.g. **CADET**) and translates the internal configuration to the corresponding format of the solver. 
After the computation is finished, the `Solution` is returned and can be further evaluated.
If the `StationarityEvaluator` is configured to test for cyclic stationarity, more chromatographic cycles are be simulated until stationarity is reached (see {ref}`stationarity_tutorial`).

For processing the simulation results, different modules are provided.
For example, the `Solution` can be compared to experimental data (or other simulations) using a `Comparator` which computes residuals such as the sum of squared errors (see also {ref}`parameter_estimation_tutorial`).
Moreover, the `Fractionator` module automatically determines fractionation times of the simulated chromatograms and determines process performance indicators such as purity, yield, and productivity (see {ref}`fractionation_tutorial`).

These metrics can be used as objectives in an `OptimizationProblem` class which serves to configure optimization studies.
Here, any process parameter can be added as optimization variable and the evaluation methods can be used to construct objectives and constraint functions.
This enables many different scenarios such as {ref}`process optimization <process_optimization_tutorial>` and {ref}`parameter estimation <parameter_estimation_tutorial>`.
Again, an abstract `OptimizationSolver` provides an interface to external optimization algorithms such as **NSGA-3**.

For this demonstration, the process model from {ref}`this example <batch_elution_example>` is used.
