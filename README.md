# CADET-Process

The [**CADET**](https://cadet.github.io) core simulator is a very powerful numerical engine that can simulate a large variety of physico-chemical models used in chromatography and other biochemical processes.
However, the configuration files of **CADET** can be complex and difficult to work with.
This is especially relevant when multiple unit operations are involved which is often the case for complex integrated processes.
Moreover, the structure of the configuration file may change during process optimization, for example when the order of dynamic events changes, making the direct use of **CADET** impossible without another layer of abstraction.

In this context [**CADET-Process**](https://cadet-process.readthedocs.io/en/latest/) was developed.
The package facilitates modeling processes using an object oriented model builder.
This interface layer provides convenient access to all model parameters in the system.
It automatically checks validity of the parameter values and sets reasonable default values where possible.
This simplifies the setup of **CADET** simulations and reduces the risk of ill-defined configurations files.

Importantly, **CADET-Process** enables the modelling of elaborate switching schemes and advanced chromatographic operating modes such as complex gradients, recycling systems, or multi-column systems by facilitating the definition of dynamic changes of flow sheet connectivity or any other time dependent parameters.

The package also includes tools to evaluate cyclic stationarity of processes, and routines to determine optimal fractionation times required determine common performance indicators such as yield, purity, and productivity.
Moreover, utility functions for calculating reaction equilibria and buffer capacities, as well as convenient functions for plotting simulation results are provided.

Finally, these processes can be optimized by defining an objective function (with constraints) and using one of the integrated optimization algorithms such as NSGA-3.
This can be used to determine any of the physico-chemical model parameters and to improve process performance.

For more information and tutorials, please refer to the [documentation](https://cadet-process.readthedocs.io/en/latest/).
The source code is freely available on [*GitHub*](https://github.com/fau-advanced-separations/CADET-Process), and a scientific paper was published in [*MDPI Processes*](https://doi.org/10.3390/pr8010065).
If **CADET-Process** is useful to you, please cite the following publication:

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
This can for example be done using [conda](https://github.com/conda-forge/miniforge):

```
conda install -c conda-forge cadet
```

For more information, see the [CADET Documentation](https://cadet.github.io/master/getting_started/installation.html).

## Free software

CADET-Process is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License version 3](https://github.com/fau-advanced-separations/CADET-Process/blob/master/LICENSE.md).

## Note

This software is work in progress and being actively developed.
Breaking changes and extensive restructuring may occur in any commit and release.
If you encounter problems or if you have questions, feel free to ask for support in the [**CADET-Forum**](https://forum.cadet-web.de).
Please report any bugs that you find [here](https://github.com/fau-advanced-separations/CADET-Process/issues).
Pull requests on [GitHub](https://github.com/fau-advanced-separations/CADET-Process) are also welcome.

## Acknowledgments

Please refer to the [list of contributors](CONTRIBUTORS.md) who helped building and funding this project.

## Contributing

Please read [CONTRIBUTING](CONTRIBUTING.md) for more details.
