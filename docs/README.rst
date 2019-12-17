Documentation
=============

We currently use Sphinx for generating the API and reference documentation for
CADET-Process.

If you only want to get the documentation, note that pre-built
versions can be found at

    https://readthedocs.org/projects/cadet-process/

## Instructions

In addition to installing CADET-PROCESS and its dependencies, install the Python
packages need to build the documentation by entering::

    pip install -r requirements.txt

in the ``doc/`` directory.

To build the HTML documentation, enter::

    make html