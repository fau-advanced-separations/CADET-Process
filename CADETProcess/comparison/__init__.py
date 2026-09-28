"""
===========================================
Comparison (:mod:`CADETProcess.comparison`)
===========================================

.. currentmodule:: CADETProcess.comparison

Classes for comparing simulation results with a reference.

Difference Metrics
==================

.. autosummary::
    :toctree: generated/

    DifferenceBase
    SSE
    RMSE
    NRMSE
    FractionationSSE
    FractionationNRMSE
    Norm
    L1
    L2
    AbsoluteArea
    RelativeArea
    Shape
    PeakHeight
    PeakPosition
    BreakthroughHeight
    BreakthroughPosition

Comparator
==========

.. autosummary::
    :toctree: generated/

    Comparator

"""

from .difference import *
from .comparator import *
