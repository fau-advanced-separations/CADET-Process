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

The `ProcessModel` is an abstract representation of the chromatographic process configuration and the required operational and design parameters.
The process is simulated using a `ProcessSimulator`.
This adapter acts  as an interface to external solvers (e.g. **CADET**) and translates the internal configuration to the format of the solver. 
After the computation is finished, the `Solution` is returned and evaluated.
Optionally, more cycles can be simulated if the `StationarityEvaluator` is configured to  test for cyclic stationarity.

The `Fractionator` module automatically determines fractionation times of the simulated chromatograms and determines process `Performance` indicators such as purity, yield, and productivity.
Moreover, the `Solution` can be compared to experimental data (or other simulations) using the `ScoringSystem` which computes `Scores` such as the sum of squared errors.

The `OptimizationProblem` is an abstraction to configure optimization studies.
Here,  any process parameter can be added as optimization variable and the evaluation methods can be used to construct objectives and constraint functions.
This enables many different scenarios such as process optimization or parameter estimation.
Again, the `OptimizationSolver` provides an interface to external optimization algorithms such as **NSGA-3**.

