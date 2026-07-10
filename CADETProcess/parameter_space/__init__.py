"""
===========================================================
Parameter Space (:mod:`CADETProcess.parameter_space`)
===========================================================

.. currentmodule:: CADETProcess.parameter_space

``ParameterSpace`` defines a feasible input domain and writes parameter
values into evaluation objects.  It is the foundation
``OptimizationProblem`` builds on, but is also directly usable on its own
for design-of-experiments studies, surrogate training data, and
sensitivity screenings.

Space
=====
.. autosummary::
    :toctree: generated/

    ParameterSpace
    TransformedSpace

Parameters
==========
.. autosummary::
    :toctree: generated/

    ParameterBase
    RangedParameter
    ChoiceParameter

Sampling
========
.. autosummary::
    :toctree: generated/

    SamplerBase
    HopsySampler
    LatinHypercubeSampler
    SobolSampler
    chebyshev_center

Constraints and Dependencies
=============================
.. autosummary::
    :toctree: generated/

    LinearConstraint
    LinearEqualityConstraint
    ParameterDependency

Mappers
=======
.. autosummary::
    :toctree: generated/

    ParameterMapperBase
    DotPathMapper
    IndexedMapper
    CallableMapper
    parse_path

"""  # noqa

from CADETProcess.parameter_space.constraints import (
    LinearConstraint,
    LinearEqualityConstraint,
)
from CADETProcess.parameter_space.dependencies import ParameterDependency
from CADETProcess.parameter_space.mappers import (
    CallableMapper,
    DotPathMapper,
    IndexedMapper,
    ParameterMapperBase,
    parse_path,
)
from CADETProcess.parameter_space.parameters import (
    ChoiceParameter,
    ParameterBase,
    RangedParameter,
)
from CADETProcess.parameter_space.sampling import (
    HopsySampler,
    LatinHypercubeSampler,
    SamplerBase,
    SobolSampler,
    chebyshev_center,
)
from CADETProcess.parameter_space.space import ParameterSpace
from CADETProcess.parameter_space.transformed_space import TransformedSpace

__all__ = [
    "ParameterSpace",
    "TransformedSpace",
    "SamplerBase",
    "HopsySampler",
    "LatinHypercubeSampler",
    "SobolSampler",
    "chebyshev_center",
    "ParameterBase",
    "RangedParameter",
    "ChoiceParameter",
    "LinearConstraint",
    "LinearEqualityConstraint",
    "ParameterDependency",
    "ParameterMapperBase",
    "DotPathMapper",
    "IndexedMapper",
    "CallableMapper",
    "parse_path",
]
