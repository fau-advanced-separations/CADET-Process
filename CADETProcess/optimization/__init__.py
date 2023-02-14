"""
===============================================
Optimization (:mod:`CADETProcess.optimization`)
===============================================

.. currentmodule:: CADETProcess.optimization

The ``optimization`` module provides functionality for minimizing (or maximizing)
objective functions, possibly subject to constraints. It includes interfaces to several
optimization suites, notably, ``scipy.optimize`` and ``pymoo``.

OptimizationProblem
===================
.. autosummary::
   :toctree: generated/

   OptimizationProblem


Optimizer
=========

Base
----

.. autosummary::
   :toctree: generated/

   OptimizerBase

Scipy
-----

.. autosummary::
   :toctree: generated/

   TrustConstr
   COBYLA
   NelderMead
   SLSQP

Pymoo
-----

.. autosummary::
   :toctree: generated/

   NSGA2
   U_NSGA3


Population
==========
.. autosummary::
   :toctree: generated/

   Individual
   Population


Results
=======
.. autosummary::
   :toctree: generated/

   OptimizationResults


Cache
=====
.. autosummary::
   :toctree: generated/

   ResultsCache

"""

from .individual import *
from .population import *
from .cache import *
from .results import *
from .optimizationProblem import *
from .optimizer import *
from .scipyAdapter import COBYLA, TrustConstr, NelderMead, SLSQP
from .pymooAdapter import NSGA2, U_NSGA3
