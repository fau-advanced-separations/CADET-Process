"""
=================================================
Instruments (:mod:`CADETProcess.instruments`)
=================================================

.. currentmodule:: CADETProcess.instruments

Templates for common LC instrument configurations and experimental data loaders.

LC System Base
==============

.. autosummary::
    :toctree: generated/

    LCFlowSheet
    LCProcess
    Phase
    ValveEvent
    PhasedProcess
    ValvePosition
    Breakthrough
    StepElution
    PulseInjection
    Step
    LWE

Knauer
======

.. autosummary::
    :toctree: generated/

    KnauerExperimentalData

"""  # noqa

from .base import (
    LCFlowSheet,
    LCProcess,
    Phase,
    ValveEvent,
    PhasedProcess,
    ValvePosition,
    Breakthrough,
    StepElution,
    PulseInjection,
    Step,
    LWE,
)
from .knauer import KnauerExperimentalData

__all__ = [
    "LCFlowSheet",
    "LCProcess",
    "Phase",
    "ValveEvent",
    "PhasedProcess",
    "ValvePosition",
    "Breakthrough",
    "StepElution",
    "PulseInjection",
    "Step",
    "LWE",
    "KnauerExperimentalData",
]
