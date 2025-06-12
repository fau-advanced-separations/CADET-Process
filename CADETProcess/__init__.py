"""
CADET-Process
========

CADET-Process is a Python package for modelling, simulating and optimizing
advanced chromatographic systems. It serves as an inteface for CADET, but also
for other solvers.

See https://cadet-process.readthedocs.io for complete documentation.
"""

# Version information
name = "CADET-Process"
__version__ = "0.11.0"

# Imports
from .CADETProcessError import *

from .settings import Settings

settings = Settings()
