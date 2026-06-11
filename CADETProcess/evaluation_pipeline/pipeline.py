from __future__ import annotations

import inspect
import re
import uuid as _uuid_mod
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
from pipefunc import PipeFunc, Pipeline

from CADETProcess.parameter_space.space import ParameterSpace

from .errors import EvaluationFailure

__all__ = ["EvaluationPipeline"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTEXT_ARG = "__eval_context__"


class _EvaluationContext:
    """Stable, serializable cache key combining a parameter vector and an evaluation object.

    Pipefunc caches by argument value.  Evaluation objects are mutable, so passing
    the raw object would produce stale cache hits whenever ``set_values`` mutates it.
    Wrapping ``(x_key, obj_uuid)`` gives a key that is:
    - correct: distinct for different x or different objects
    - stable: UUID survives pickling, so disk/shared-memory caches work across processes
    """

    __slots__ = ("x_key", "obj", "_uuid")

    def __init__(self, x_key: tuple, obj: Any, obj_uuid: str) -> None:
        self.x_key = x_key
        self.obj = obj
        self._uuid = obj_uuid

    def __hash__(self) -> int:
        return hash((self.x_key, self._uuid))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _EvaluationContext)
            and self.x_key == other.x_key
            and self._uuid == other._uuid
        )


def _validate_identifier(name: str) -> None:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"{name!r} is not a valid Python identifier")


def _wrap_with_failure_propagation(func: Callable, stage: str) -> Callable:
    """Return a wrapper that propagates EvaluationFailure and catches exceptions.

    If any positional or keyword argument is an `EvaluationFailure`, that failure
    is returned immediately without calling `func`.  Otherwise `func` is called
    normally; any exception is caught and returned as a new `EvaluationFailure`.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for v in (*args, *kwargs.values()):
            if isinstance(v, EvaluationFailure):
                return v
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return EvaluationFailure(stage=stage, reason=str(e), exc=e)

    return wrapper


def _make_injection_wrapper(func: Callable, requires: list[str]) -> Callable:
    """Wrap `func` so its parameter names match `requires`.

    Pipefunc wires nodes by matching argument names to upstream `output_name` values.
    When a user function's argument names do not match the upstream output names,
    this wrapper creates an intermediate function whose signature does match,
    forwarding the arguments positionally to `func`.

    The `requires` entries must be valid Python identifiers.
    """
    for name in requires:
        _validate_identifier(name)

    @wraps(func)
    def wrapper(**kwargs: Any) -> Any:
        return func(*(kwargs[name] for name in requires))

    sig_params = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
        for name in requires
    ]
    wrapper.__signature__ = inspect.Signature(sig_params)
    return wrapper


def _make_context_wrapper(func: Callable) -> Callable:
    """Wrap a root node so it receives ``_EvaluationContext`` and extracts the obj.

    The wrapped function's signature is ``(__eval_context__,)`` so pipefunc wires
    it to the single pipeline root arg.  The user's function is called with the
    raw evaluation object extracted from the context.
    """

    def wrapper(**kwargs: Any) -> Any:
        ctx: _EvaluationContext = kwargs[_CONTEXT_ARG]
        return func(ctx.obj)

    wrapper.__signature__ = inspect.Signature(
        [inspect.Parameter(_CONTEXT_ARG, inspect.Parameter.KEYWORD_ONLY)]
    )
    return wrapper


def _make_node(
    func: Callable,
    output_name: str,
    requires: list[str] | None,
) -> PipeFunc:
    """Build a `PipeFunc` node with failure propagation and optional arg injection."""
    safe = _wrap_with_failure_propagation(func, stage=output_name)
    if requires is not None:
        node_func = _make_injection_wrapper(safe, requires)
    else:
        node_func = _make_context_wrapper(safe)
    return PipeFunc(node_func, output_name=output_name, cache=True)


class EvaluationPipeline:
    """DAG-based evaluation engine backed by `pipefunc.Pipeline`.

    Parameterized by a `ParameterSpace`; cannot be constructed without one.
    `evaluate(x)` writes `x` into the evaluation objects via `space.set_values`,
    then runs the registered node graph and returns all (or selected) named outputs.

    `pipefunc` is an internal implementation detail.  No `PipeFunc` or `Pipeline`
    objects appear in the public API.

    Parameters
    ----------
    space : ParameterSpace
        Owns the evaluation objects and knows how to write a parameter vector
        into them.  `evaluate` delegates to `space.set_values(x)`.

    Examples
    --------
    >>> pipeline = EvaluationPipeline(space)
    >>> pipeline.add_evaluator(simulate, output_name="simulation_results")
    >>> pipeline.add_evaluator(
    ...     fractionate, output_name="fractionation_results",
    ...     requires=["simulation_results"],
    ... )
    >>> results = pipeline.evaluate(x)
    >>> results["fractionation_results"]
    """

    def __init__(
        self,
        space: ParameterSpace,
        cache_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(space, ParameterSpace):
            raise TypeError(f"Expected ParameterSpace, got {type(space).__name__}")
        self._space = space
        self._cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        self._nodes: list[PipeFunc] = []
        self._output_names: list[str] = []
        self._pipeline: Pipeline | None = None
        self._obj_uuids: dict[int, str] = {}  # id(obj) → stable UUID

    # ------------------------------------------------------------------
    # Registration

    def add_evaluator(
        self,
        func: Callable,
        output_name: str,
        requires: list[str] | None = None,
    ) -> None:
        """Register a callable as a named node in the evaluation DAG.

        Parameters
        ----------
        func : callable
            The function to register.  Its return value becomes the named output.
            When `requires` is None, `func`'s own argument names must match
            upstream `output_name` values so pipefunc can wire them.
            When `requires` is given, a wrapper is generated whose argument names
            match the `requires` list, so `func`'s argument names are irrelevant.
        output_name : str
            Name of this node's output in the DAG.  Must be a valid Python
            identifier and unique within this pipeline.
        requires : list[str], optional
            Ordered list of upstream output names to inject as positional arguments
            to `func`.  When given, `func` must accept ``len(requires)`` positional
            arguments.  When None, pipefunc wires `func` by its own argument names.

        Raises
        ------
        TypeError
            If `func` is not callable.
        ValueError
            If `output_name` is already registered or is not a valid identifier,
            or if any entry in `requires` is not a valid identifier.
        """
        if not callable(func):
            raise TypeError(f"Expected callable, got {type(func).__name__}")
        _validate_identifier(output_name)
        if output_name in self._output_names:
            raise ValueError(f"output_name {output_name!r} is already registered")

        node = _make_node(func, output_name, requires)
        self._nodes.append(node)
        self._output_names.append(output_name)
        self._pipeline = None  # invalidate cached pipeline

    # ------------------------------------------------------------------
    # Evaluation

    @property
    def output_names(self) -> list[str]:
        """list[str]: All registered output names, in registration order."""
        return list(self._output_names)

    def _get_pipeline(self) -> Pipeline:
        """Return the cached pipeline, building it if necessary.

        The pipeline is rebuilt whenever ``add_evaluator`` is called after a
        previous build.  Within a stable registration phase the same instance
        is reused across ``evaluate`` calls.
        """
        if self._pipeline is None:
            if not self._nodes:
                raise RuntimeError(
                    "No evaluators registered.  Call add_evaluator before evaluate."
                )
            if self._cache_dir is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._pipeline = Pipeline(
                    list(self._nodes),
                    cache_type="disk",
                    cache_kwargs={"cache_dir": str(self._cache_dir)},
                )
            else:
                self._pipeline = Pipeline(list(self._nodes), cache_type="lru")
        return self._pipeline

    def _obj_uuid(self, obj: Any) -> str:
        """Return the stable UUID for *obj*, assigning one on first encounter."""
        key = id(obj)
        if key not in self._obj_uuids:
            self._obj_uuids[key] = str(_uuid_mod.uuid4())
        return self._obj_uuids[key]

    def evaluate(
        self,
        x: Any,
        targets: list[str] | None = None,
        bypass_cache: bool = False,
    ) -> dict[str, Any]:
        """Set parameter values and run the evaluation graph.

        The `Pipeline` instance is reused across calls.  Cache invalidation is
        handled by ``_EvaluationContext``: each call constructs a context keyed on
        ``(x_key, obj_uuid)``, so results for different x values are naturally
        distinct cache entries without manual cache clearing.  Intermediate nodes
        shared by multiple targets within a single call are computed only once.

        Parameters
        ----------
        x : array-like
            Parameter vector passed to `space.set_values`.
        targets : list[str], optional
            Output names to compute.  `None` computes all registered outputs.
            Requesting a subset exploits pipefunc's lazy evaluation: only the
            subgraph needed for the requested outputs is executed.
        bypass_cache : bool
            When True, clear the pipeline cache before evaluating so all nodes
            are recomputed from scratch.  Useful for debugging to confirm that
            results are not stale.

        Returns
        -------
        dict[str, Any]
            Mapping from output name to result.  For a single evaluation object
            the values are plain results (or `EvaluationFailure`).  For multiple
            evaluation objects the values are lists indexed by evaluation object.
            Results may be `EvaluationFailure` instances when a node failed.
        """
        x_key: tuple = tuple(np.asarray(x, dtype=float).ravel())
        if bypass_cache:
            # Append a nonce so every node sees a guaranteed cache miss.
            # Existing entries for other x values are unaffected.
            x_key = x_key + (_uuid_mod.uuid4().hex,)
        self._space.set_values(x)

        if targets is None:
            targets = self._output_names
        else:
            unknown = [t for t in targets if t not in self._output_names]
            if unknown:
                raise ValueError(f"Unknown target(s): {unknown}")

        eval_objs = self._space.evaluation_objects
        if not eval_objs:
            raise RuntimeError("ParameterSpace has no evaluation objects.")

        pipeline = self._get_pipeline()

        def _run_for(obj: Any) -> dict[str, Any]:
            """Run all targets for one evaluation object in a single pipeline call."""
            ctx = _EvaluationContext(x_key, obj, self._obj_uuid(obj))
            if len(targets) == 1:
                value = pipeline(targets[0], **{_CONTEXT_ARG: ctx})
                return {targets[0]: value}
            values = pipeline.run(targets, kwargs={_CONTEXT_ARG: ctx})
            assert len(values) == len(targets), (
                f"Expected {len(targets)} results from pipeline.run, got {len(values)}"
            )
            return dict(zip(targets, values))

        if len(eval_objs) == 1:
            return _run_for(eval_objs[0])

        results: dict[str, list] = {t: [] for t in targets}
        for obj in eval_objs:
            for t, v in _run_for(obj).items():
                results[t].append(v)
        return results
