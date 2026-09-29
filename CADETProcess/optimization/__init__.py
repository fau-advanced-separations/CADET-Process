"""
===============================================
Optimization (:mod:`CADETProcess.optimization`)
===============================================

.. currentmodule:: CADETProcess.optimization

The ``optimization`` module provides functionality for minimizing (or maximizing)
objective functions, possibly subject to constraints. It includes interfaces to several
optimization suites, notably, ``scipy.optimize``, ``cyipopt``, and ``pymoo``.

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
   COBYQA
   NelderMead
   SLSQP
   LBFGSB
   LeastSquares

IPOPT
-----

.. autosummary::
   :toctree: generated/

   IPOPT

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

BoFire
------

.. autosummary::
   :toctree: generated/

   BoFire

Population
==========
.. autosummary::
   :toctree: generated/

   IndividualView
   Population
   ParetoFront


Results
=======
.. autosummary::
   :toctree: generated/

   OptimizationResults


ParallelizationBackend
======================
.. autosummary::
   :toctree: generated/

   ParallelizationBackendBase
   SequentialBackend
   Joblib
   Pathos

"""

from .population import *
from .results import *
from .optimization_problem import OptimizationProblem
from .parallelizationBackend import *
from .optimizer import *
from .scipyAdapter import COBYLA, COBYQA, TrustConstr, NelderMead, SLSQP, LBFGSB, LeastSquares
from .pymooAdapter import NSGA2, U_NSGA3

import importlib

try:
    from .axAdapter import BotorchModular, GPEI, NEHVI, qNParEGO

    ax_imported = True
except ImportError:
    ax_imported = False

try:
    from .bofireAdapter import BoFire

    bofire_imported = True
except ImportError:
    bofire_imported = False


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
    if name == "BoFire":
        if bofire_imported:
            module = importlib.import_module("bofireAdapter", package=__name__)
            return getattr(module, name)
        else:
            raise ImportError(
                "The BoFire class could not be imported. "
                "This may be because the 'bofire' package, which is an optional dependency, is not installed. "
                "To install it, run 'pip install CADET-Process[bofire]'"
            )
    if name == "IPOPT":
        if ipopt_imported:
            module = importlib.import_module("ipoptAdapter", package=__name__)
            return getattr(module, name)
        else:
            raise ImportError(
                "IPOPT could not be imported. "
                "The 'cyipopt' package is an optional dependency. "
                "To install it, run 'pip install CADET-Process[ipopt]' "
                "or 'conda install -c conda-forge cyipopt'."
            )
    raise AttributeError(f"module {__name__} has no attribute {name}")


try:
    from .ipoptAdapter import IPOPT

    ipopt_imported = True
except ImportError:
    ipopt_imported = False
