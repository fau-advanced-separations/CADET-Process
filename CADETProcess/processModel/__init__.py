"""
================================================
Process Model (:mod:`CADETProcess.processModel`)
================================================

.. currentmodule:: CADETProcess.processModel

Classes for modelling processes.

ComponentSystem
===============

.. autosummary::
    :toctree: generated/

    Species
    Component
    ComponentSystem

Reaction Models
===============

.. autosummary::
    :toctree: generated/

    Reaction
    CrossPhaseReaction
    ReactionBaseClass
    NoReaction
    MassActionLaw
    MassActionLawParticle

Binding Models
==============

.. autosummary::
    :toctree: generated/

    BindingBaseClass
    NoBinding
    Linear
    Langmuir
    LangmuirLDF
    BiLangmuir
    BiLangmuirLDF
    FreundlichLDF
    StericMassAction
    AntiLangmuir
    Spreading
    SelfAssociation
    BiStericMassAction
    MultistateStericMassAction
    SimplifiedMultistateStericMassAction
    Saska
    GeneralizedIonExchange

Unit Operation Models
=====================

.. autosummary::
    :toctree: generated/

    UnitBaseClass
    Inlet
    Outlet
    Cstr
    TubularReactor
    LumpedRateModelWithoutPores
    LumpedRateModelWithPores
    GeneralRateModel

Discretization
--------------

.. autosummary::
    :toctree: generated/

    discretization

Solution Recorder
-----------------

.. autosummary::
    :toctree: generated/

    solutionRecorder

Flow Sheet
==========

.. autosummary::
    :toctree: generated/

    FlowSheet

Process
=======

.. autosummary::
    :toctree: generated/

    Process

"""
from . import componentSystem
from .componentSystem import  *

from . import reaction
from .reaction import *

from . import binding
from .binding import *

from . import discretization
from .discretization import *

from . import unitOperation
from .unitOperation import *

from . import flowSheet
from .flowSheet import *

from . import process
from .process import *

__all__ = [s for s in dir()]
