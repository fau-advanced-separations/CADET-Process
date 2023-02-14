"""
===============================================
ModelBuilder (:mod:`CADETProcess.modelBuilder`)
===============================================

.. currentmodule:: CADETProcess.modelBuilder

The ``modelBuilder`` module provides functionality for setting up complex ``Process``
models

CarouselBuilder
===============

A module for building carousel systems like SMB, MCSGP, etc.

.. autosummary::
   :toctree: generated/

   SerialZone
   ParallelZone
   CarouselBuilder


CompartmentBuilder
==================

A module for building compartment model systems.

.. autosummary::
   :toctree: generated/

   CompartmentBuilder

"""

from . import carouselBuilder
from .carouselBuilder import *
from .compartmentBuilder import *
