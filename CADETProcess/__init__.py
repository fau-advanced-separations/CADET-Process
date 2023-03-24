"""
CADET-Process
========

CADET-Process is a Python package for modelling, simulating and optimizing
advanced chromatographic systems. It serves as an inteface for CADET, but also
for other solvers.

See https://cadet-process.readthedocs.io for complete documentation.
"""
# Version information
name = 'CADET-Process'
__version__ = '0.7.0'

# Imports
from .CADETProcessError import *

from . import log

from .settings import Settings
settings = Settings()

from . import dataStructure
from . import transform
from . import plotting
from . import dynamicEvents
from . import processModel
from . import smoothing
from . import solution
from . import reference
from .simulationResults import SimulationResults
from . import metric
from . import performance
from . import optimization
from . import comparison
from . import stationarity
from . import simulator
from . import fractionation
from . import equilibria
from . import modelBuilder
from . import tools
