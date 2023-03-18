(process_evaluation_guide)=
# Process Evaluation
**CADET-Process** offers multiple tools for evaluating simulation results.

To quantify the performance of a chromatographic process, performance indicators such as purity or recovery yield must be calculated from the chromatograms.
The {mod}`~CADETProcess.fractionation` module provides a {class}`~CADETProcess.fractionation.Fractionator` class for this purpose.
It enables the automatic determination of optimal fractionation times and the calculation of various performance indicators.

Furthermore, the {mod}`~CADETProcess.comparison` module allows for quantitative comparison of simulation results with experimental data or reference simulations.
This module provides various metrics for quantifying differences between datasets and calculating residuals, which is crucial for parameter estimation.

```{toctree}
:maxdepth: 2

fractionation
comparison
```
