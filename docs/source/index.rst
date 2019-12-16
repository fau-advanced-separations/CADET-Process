.. CADET-Process documentation master file, created by
   sphinx-quickstart on Mon Dec 16 19:29:35 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview of CADET-Process
=========================

CADET-Process is a Python package for the systematic development of preparative
chromatographic processes. It is intended for optimal design of conventional and
advanced concepts. The Python-based platform simplifies the implementation of
new processes and design problems by decoupling design tasks into individual
modules for


-  Setting up models for the desired process structure and
   the specific chromatographic column(s),
-  Solving the model equations for simulating the process,
-  Determining process performance by evaluating the resulting chromatograms,
-  Optimization of continuous variables, timed events, and process structures.

Interfaces to external libraries provide flexibility regarding the choice of
column model, solver, and optimizer.

Free software
-------------

CADET-Process is free software: you can redistribute it and/or modify it under
the terms of the :doc:`GNU General Public License version 3 </license>`.

We welcome contributions. Join us on
`GitHub <https://github.com/fau-advanced-separations/CADET-Process>`_.


Documentation
-------------

.. only:: html

    :Release: |version|
    :Date: |today|

.. toctree::
   :maxdepth: 1

   reference/index
   license


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
