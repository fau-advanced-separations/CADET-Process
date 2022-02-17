(overview)=
# Overview

**CADET-Process** is written in Python, a free and open-source programming language that gained a lot of popularity in the scientific community in recent years.
One of the main advantages of Python is the easy integration of other scientific and numerical packages.
This makes it especially useful for a modular approach such as the one presented.
The following {ref}`figure <framework_overview>` gives a general overview of the program structure and workflow.

```{figure} ../_static/framework_overview.png
:name: framework_overview

Overview of the framework modules and their relations. For a detailed explanation, see text.
```

The `ProcessModel` is an abstract representation of the chromatographic process configuration and the required operational and design parameters.
The process is simulated using the `SimulationAdapter` as an interface to external solver and the resulting Chromatogram can be tested for cyclic `Stationarity`.

For `ProcessOptimization`, a bi-level problem is formulated to reduce the overall complexity of the optimization problem.
On the top level, the variables of the `ProcessModel` are optimized, and on the lower level, the process Performance is determined by calling a `FractionationOptimization` subroutine that optimizes the fractionation times of the simulated Chromatograms.
In both cases, the `OptimizationAdapter` provides an interface to external Optimizers.

