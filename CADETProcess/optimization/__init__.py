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

Ax
--

.. autosummary::
   :toctree: generated/

   BotorchModular
   GPEI
   NEHVI

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

ParallelizationBackend
======================
.. autosummary::
   :toctree: generated/

   ParallelizationBackendBase
   SequentialBackend
   Joblib
   Pathos

"""

from .individual import *
from .population import *
from .cache import *
from .results import *
from .optimizationProblem import *
from .parallelizationBackend import *
from .optimizer import *
from .scipyAdapter import COBYLA, TrustConstr, NelderMead, SLSQP
from .pymooAdapter import NSGA2, U_NSGA3

import importlib

try:
    from .axAdapater import BotorchModular, GPEI, NEHVI, qNParEGO

    ax_imported = True
except ImportError:
    ax_imported = False


def __getattr__(name):
    if name in ("BotorchModular", "GPEI", "NEHVI", "qNParEGO"):
        if ax_imported:
            module = importlib.import_module("axAdapter", package=__name__)
            return getattr(module, name)
        else:
            raise ImportError(
                "The AxInterface class could not be imported. "
                "This may be because the 'ax' package, which is an optional dependency, is not installed. "
                "To install it, run 'pip install CADET-Process[ax]'"
            )
    raise AttributeError(f"module {__name__} has no attribute {name}")
