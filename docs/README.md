Documentation
=============
## CADET-Process Documentation


To build the documentation locally, install sphinx and other dependencies to an environment with a CADET-Process installation by running (from the CADET-Process root directory)

```
pip install -e .[docs]
```

Then, in the `docs` folder run:

```
sphinx-build -b html source build
```

The output is in the `build` directory and can be opened with any browser.

To build the documentation for all releases and the master branch, run:

This documentation is published under https://readthedocs.org/projects/cadet-process/
