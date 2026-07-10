"""
=============================================================
Evaluation Pipeline (:mod:`CADETProcess.evaluation_pipeline`)
=============================================================

.. currentmodule:: CADETProcess.evaluation_pipeline

``EvaluationPipeline`` is a DAG-based evaluation engine: named evaluator
nodes are registered against a ``ParameterSpace`` and run in dependency
order, with per-node caching and failure propagation.  It is directly
usable without an optimizer, and is the backend ``OptimizationProblem``
builds on internally.

.. autosummary::
    :toctree: generated/

    EvaluationPipeline
    EvaluationFailure

"""  # noqa

from CADETProcess.evaluation_pipeline.errors import EvaluationFailure
from CADETProcess.evaluation_pipeline.pipeline import EvaluationPipeline

__all__ = ["EvaluationPipeline", "EvaluationFailure"]
