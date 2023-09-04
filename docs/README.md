# CADET-Process Documentation

To build the documentation locally, install sphinx and other dependencies by running

```
pip install -e .[docs]
```
from the CADET-Process root directory.
Make sure to also install **CADET-Core**, e.g. using `conda`.

Then, in the `docs` folder run:

```
sphinx-build -b html source build
```

The output is in the `build` directory and can be opened with any browser.

To build the documentation for all releases and the master branch, run:

This documentation is published under https://readthedocs.org/projects/cadet-process/
