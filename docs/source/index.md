# CADET-Process

The [**CADET**](https://cadet.github.io) core simulator is a very powerful numerical engine that can simulate a large variety of physico-chemical models.
However, the configuration files are error prone and can be difficult to work with.
This is especially true when multiple unit operations are involved.
Moreover, the structure of the configuration file can change during process optimization when dynamic switching is involved, making the direct use of **CADET** impossible without another layer of abstraction.

In this context **CADET-Process** was developed.
The package facilitates modeling processes by providing an object oriented model builder.
This interface layer provides convenient access to all model parameters in the system.
It automatically checks validity of the parameter values and provides good default values where possible.
To simplify the setup of dynamic changes of flow sheet connectivity or other parameters, they are defined on completely independent time lines.
To simulate the process, the entire configuration is converted to a regular **CADET** configuration file on demand.

This simplifies the setup of **CADET** simulations and reduces the risk of ill-defined configuration files.
Importantly, **CADET-Process** also facilitates the setup of elaborate switching schemes such as complex gradients, recycling systems, or carousel systems by enabling the definition of event dependencies within the system.

The package also includes tools to evaluate cyclic stationarity of processes, to determine optimal fractionation times, and to calculate common performance indicators such as yield, purity, and productivity.
Moreover, utility functions for calculate reaction equilibria and buffer capacities, as well as convenient functions for plotting simulation results are provided.

Finally, these processes can be optimized by defining an objective function (with constraints) and using one of the integrated optimization algorithms such as NSGA-3.
This can be used to improve process performance or to determine any of the physico-chemical model parameters.

The source code can be found on [Github](https://github.com/fau-advanced-separations/CADET-Process) and a scientific paper was published in the [Processes journal](https://doi.org/10.3390/pr8010065). If you use **CADET-Process**, please cite the following publication:
```
@Article{Schmoelder2020,
  author  = {Schmölder, Johannes and Kaspereit, Malte},
  title   = {A {{Modular Framework}} for the {{Modelling}} and {{Optimization}} of {{Advanced Chromatographic Processes}}},
  doi     = {10.3390/pr8010065},
  number  = {1},
  pages   = {65},
  volume  = {8},
  journal = {Processes},
  year    = {2020},
}
```

## Installation
**CADET-Process** can be installed with the following command:

```
pip install CADET-Process
```

To use **CADET-Process**, make sure, that **CADET** is also installed. 
This can for example be done using [conda](https://docs.conda.io/en/latest/):
```
conda install -c conda-forge cadet
```
For more information, see the [CADET Documentation](https://cadet.github.io/master/getting_started/installation.html).

## Free software
CADET-Process is free software: you can redistribute it and/or modify it under the terms of the {ref}`GNU General Public License version 3 <license>`

## Note
This software is work in progress and being actively developed.
Breaking changes and extensive restructuring may occur in any commit and release.
If you encounter problems of have questions, feel free to ask for support in the **CADET-Forum**:https://forum.cadet-web.de
Please report any bugs that you find [here](https://github.com/fau-advanced-separations/CADET-Process/issues).
Pull requests on [GitHub](https://github.com/fau-advanced-separations/CADET-Process) are also welcome.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

getting_started/overview
getting_started/simulation
getting_started/fractionation
getting_started/cyclic_stationarity
getting_started/optimization
```

```{toctree}
:maxdepth: 2
:caption: Advanced features
:hidden:

<!-- advanced_use/event_dependencies -->
<!-- advanced_use/carousel_builder -->
advanced_use/buffer_equilibria
<!-- advanced_use/create_inlet_profile -->
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

reference/process_model/index
reference/simulation
reference/fractionation
reference/optimization
```

```{toctree}
:maxdepth: 2
:caption: References
:hidden:

license
bibliography
```
