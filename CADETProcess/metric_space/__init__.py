"""
=====================================================
Metric Space (:mod:`CADETProcess.metric_space`)
=====================================================

.. currentmodule:: CADETProcess.metric_space

``MetricSpace`` collects named output declarations (``Metric``) and the
problem-level annotations that give them meaning: optimization direction
(``Objective``) and operator/bound constraints (``Constraint``).  Direction
and normalization are properties of the registration, not of the metric
itself: the same metric can be an objective in one problem and a plain
output in another.

.. autosummary::
    :toctree: generated/

    Metric
    MetricSpace
    Objective
    Constraint

"""  # noqa

from CADETProcess.metric_space.metric import Metric
from CADETProcess.metric_space.space import Constraint, MetricSpace, Objective

__all__ = [
    "Metric",
    "MetricSpace",
    "Objective",
    "Constraint",
]
