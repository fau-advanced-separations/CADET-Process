"""
=========================================
Simulator (:mod:`CADETProcess.simulator`)
=========================================

.. currentmodule:: CADETProcess.simulator

The ``simulator`` module provides functionality for simulating a ``Process``.

Simulator
=========

.. autosummary::
   :toctree: generated/

   SimulatorBase

CADET
-----

.. autosummary::
   :toctree: generated/

   Cadet

Futher settings:

.. autosummary::
   :toctree: generated/

   ModelSolverParametersGroup
   UnitParametersGroup
   AdsorptionParametersGroup
   ReactionParametersGroup
   SolverParametersGroup
   SolverTimeIntegratorParametersGroup
   ReturnParametersGroup
   SensitivityParametersGroup


SimulationResults
=================

After simulation, the ``Simulator`` returns a ``SimulationResults`` object.

.. currentmodule:: CADETProcess.simulationResults

.. autosummary::
   :toctree: generated/

   SimulationResults

"""

from .simulator import *
from .cadetAdapter import *
