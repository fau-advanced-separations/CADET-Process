from __future__ import annotations

import inspect
import re
import uuid as _uuid_mod
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any

from pipefunc import PipeFunc, Pipeline

from CADETProcess.parameter_space.space import ParameterSpace

from .errors import EvaluationFailure

__all__ = ["EvaluationPipeline"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTEXT_ARG = "__eval_context__"
# Output name of the ``set_values`` root node and the axis input name a mapped
# node consumes: ``evaluation_contexts[object] -> output[object]``.
_EVALUATION_CONTEXTS = "evaluation_contexts"
# Stable UUID slot for the zero-evaluation-object mode: cache entries are
# keyed on (x_key, uuid), and without an object the x_key alone identifies
# the evaluation.  A fixed constant keeps keys stable across processes.
_NO_OBJECT_UUID = "__no_evaluation_object__"


class _EvaluationContext:
    """Stable, serializable cache key combining a parameter assignment and an evaluation object.

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

    def __reduce__(self) -> tuple:
        # Only x_key and _uuid define identity (see __hash__/__eq__).
        # obj is runtime-only and may not be picklable (e.g. local lambdas in
        # event transforms).  Cross-process cache lookups only need the key.
        return (_EvaluationContext, (self.x_key, None, self._uuid))


def _validate_identifier(name: str) -> None:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"{name!r} is not a valid Python identifier")


def _wrap_with_failure_propagation(func: Callable, stage: str) -> Callable:
    """Return a wrapper that propagates EvaluationFailure and catches exceptions.

    If any positional or keyword argument is an `EvaluationFailure`, that failure
    is returned immediately without calling `func`.  Otherwise `func` is called
    normally; any exception is caught and returned as a new `EvaluationFailure`.

    Unclassified exceptions default to `recoverable=True` (transient: do not
    cache), since a misclassified transient failure permanently poisons a
    legitimately good x, while the opposite mistake only costs an occasional
    re-crash at a duplicate x.  A node author who knows a failure is
    deterministic (CADET solver failure, validation rejection) classifies it
    explicitly by setting `recoverable = False` on the raised exception.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for v in (*args, *kwargs.values()):
            if isinstance(v, EvaluationFailure):
                return v
        try:
            return func(*args, **kwargs)
        except Exception as e:
            recoverable = getattr(e, "recoverable", True)
            return EvaluationFailure(stage=stage, reason=str(e), exc=e, recoverable=recoverable)

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


def _make_root_wrapper(func: Callable, arg_name: str) -> Callable:
    """Wrap a root node so it receives a ``_EvaluationContext`` and extracts the obj.

    The wrapper's single argument is named *arg_name* so pipefunc wires it to the
    graph root: ``_CONTEXT_ARG`` for the legacy per-object loop (one context
    injected per call), or ``_EVALUATION_CONTEXTS`` for a mapped node (one context
    element per object axis iteration).  Either way the user's function is called
    with the raw evaluation object extracted from the context.
    """

    def wrapper(**kwargs: Any) -> Any:
        ctx: _EvaluationContext = kwargs[arg_name]
        return func(ctx.obj)

    wrapper.__signature__ = inspect.Signature(
        [inspect.Parameter(arg_name, inspect.Parameter.KEYWORD_ONLY)]
    )
    return wrapper


def _guard_recoverable_writes(cache: Any) -> None:
    """Patch *cache* so recoverable ``EvaluationFailure`` values are never written.

    pipefunc's per-node cache has no native hook to skip a write conditionally
    on the result value, so the cache's own ``put`` is wrapped in place.
    Recoverable (transient) failures must not be cached: caching one would
    permanently poison a legitimately good x across restarts.  Deterministic
    failures (``recoverable=False``) still cache normally, since re-running a
    deterministic crash just crashes again.
    """
    original_put = cache.put

    def put(key: Any, value: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(value, EvaluationFailure) and value.recoverable:
            return
        original_put(key, value, *args, **kwargs)

    cache.put = put


def _make_set_values_node(space: ParameterSpace) -> PipeFunc:
    """Build the unmapped root node that writes ``x`` and emits the context sequence.

    The node takes the assignment mapping ``x`` (the genuine graph root), writes
    it into every evaluation object via ``space.set_values``, and returns an
    ordered ``list[_EvaluationContext]`` carrying the live configured objects.
    Downstream mapped nodes introduce the ``object`` axis by indexing this
    sequence; the node itself is unmapped, so ``set_values`` runs once per call,
    not once per object (the single-write-path invariant).

    The closure captures only *space*, never the pipeline or ``OptimizationProblem``
    (the anti-recursion invariant), which is what the space-owned per-object UUID
    makes possible.

    ``cache=False``: the node mutates shared objects, so a cache hit would return
    stale-state references; re-running every call preserves current behavior and
    costs nothing (the simulation caches downstream).
    """

    def set_values(x: Mapping[str, Any]) -> list[_EvaluationContext]:
        space.set_values(x)
        x_key: tuple = tuple(
            (p.name, x[p.name]) for p in space.independent_parameters
        )
        objs = space.evaluation_objects
        if not objs:
            # Objectless mode is one sentinel context, not an empty axis: an
            # empty fan would leave a downstream scalar reducer with nothing.
            return [_EvaluationContext(x_key, dict(x), _NO_OBJECT_UUID)]
        return [
            _EvaluationContext(x_key, obj, space.evaluation_object_uuid(obj))
            for obj in objs
        ]

    set_values.__signature__ = inspect.Signature(
        [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    return PipeFunc(set_values, output_name=_EVALUATION_CONTEXTS, cache=False)


def _make_node(
    func: Callable,
    output_name: str,
    requires: list[str] | None,
    cache: bool = True,
    mapspec: str | None = None,
) -> PipeFunc:
    """Build a `PipeFunc` node with failure propagation and optional arg injection.

    A node with ``requires`` receives its named upstream outputs (single elements
    under a mapspec, whole values otherwise).  A root node (``requires is None``)
    consumes the object context: the ``evaluation_contexts`` axis when it is
    mapped, or the legacy per-object ``_CONTEXT_ARG`` root when it is not.
    """
    safe = _wrap_with_failure_propagation(func, stage=output_name)
    if requires is not None:
        node_func = _make_injection_wrapper(safe, requires)
    elif mapspec is not None:
        node_func = _make_root_wrapper(safe, _EVALUATION_CONTEXTS)
    else:
        node_func = _make_root_wrapper(safe, _CONTEXT_ARG)
    return PipeFunc(node_func, output_name=output_name, cache=cache, mapspec=mapspec)


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
        Owns the evaluation objects and knows how to write parameter values
        into them.  `evaluate` decodes the vector and delegates to
        `space.set_values`.

    Examples
    --------
    >>> pipeline = EvaluationPipeline(space)
    >>> pipeline.add_evaluator(simulate, output_name="simulation_results")
    >>> pipeline.add_evaluator(
    ...     fractionate, output_name="fractionation_results",
    ...     requires=["simulation_results"],
    ... )
    >>> results = pipeline.evaluate({"length": 0.5})
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
        # Parallel per-node metadata, kept alongside ``_nodes`` so graph-shape
        # decisions (mapped vs. legacy loop, mixed-root guard) read plain data
        # instead of re-introspecting PipeFunc objects.
        self._mapspecs: list[str | None] = []
        self._requires: list[list[str] | None] = []
        self._pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Registration

    def add_evaluator(
        self,
        func: Callable,
        output_name: str,
        requires: list[str] | None = None,
        cache: bool = True,
        mapspec: str | None = None,
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
        cache : bool
            Whether to cache the output of this node.  Defaults to True.
            Pass False for nodes with side effects (e.g. callbacks) where
            repeated execution is intentional and results need not be stored.
        mapspec : str, optional
            pipefunc axis-mapping string declaring this node as mapped over the
            object axis, e.g. ``"evaluation_contexts[object] -> out[object]"`` for
            a root evaluator or ``"simulation_result[object] -> out[object]"`` for
            a downstream one.  When any node is mapped the graph is executed once
            via ``pipeline.map`` over the whole object axis instead of the legacy
            per-object loop.  A node without a mapspec is a whole-value consumer.

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

        node = _make_node(func, output_name, requires, cache=cache, mapspec=mapspec)
        self._nodes.append(node)
        self._output_names.append(output_name)
        self._mapspecs.append(mapspec)
        self._requires.append(requires)
        self._pipeline = None  # invalidate cached pipeline

    # ------------------------------------------------------------------
    # Evaluation

    @property
    def output_names(self) -> list[str]:
        """list[str]: All registered output names, in registration order."""
        return list(self._output_names)

    @property
    def _is_mapped(self) -> bool:
        """Whether any registered node declares a mapspec (mapped execution)."""
        return any(m is not None for m in self._mapspecs)

    def _graph_nodes(self) -> list[PipeFunc]:
        """Nodes for the pipefunc graph, prepending ``set_values`` when mapped.

        A mapped graph roots on ``x`` and fans the object axis off the
        ``evaluation_contexts`` sequence, so the ``set_values`` node must exist.
        An unmapped graph is byte-for-byte the legacy per-object graph.
        """
        if not self._is_mapped:
            return list(self._nodes)
        unmapped_roots = [
            name
            for name, mapspec, requires in zip(
                self._output_names, self._mapspecs, self._requires
            )
            if mapspec is None and requires is None
        ]
        if unmapped_roots:
            raise ValueError(
                "A mapped graph cannot contain an unmapped root evaluator "
                f"(requires=None, mapspec=None): {unmapped_roots}. Give it a "
                "mapspec ('evaluation_contexts[object] -> out[object]') so it fans "
                "over the object axis. Whole-value fan-in over a mapped output is "
                "not yet supported."
            )
        return [_make_set_values_node(self._space), *self._nodes]

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
            nodes = self._graph_nodes()
            if self._cache_dir is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._pipeline = Pipeline(
                    nodes,
                    cache_type="disk",
                    # lru_shared=False keeps the in-memory LRU a plain dict; the
                    # shared variant backs it with a multiprocessing.Manager
                    # process that is never torn down and deadlocks interpreter
                    # exit on Python 3.12.  A process-shared cache buys nothing
                    # here: parallelism is per-individual, so each worker holds
                    # its own pipeline copy.
                    cache_kwargs={
                        "cache_dir": str(self._cache_dir),
                        "lru_shared": False,
                    },
                    validate_type_annotations=False,
                )
            else:
                self._pipeline = Pipeline(
                    nodes,
                    cache_type="hybrid",
                    # shared=False: plain in-process dict instead of a
                    # Manager-backed one.  See the disk branch above.
                    cache_kwargs={"shared": False},
                    validate_type_annotations=False,
                )
            _guard_recoverable_writes(self._pipeline.cache)
        return self._pipeline

    def evaluate(
        self,
        assignment: Mapping[str, Any],
        targets: list[str] | None = None,
        bypass_cache: bool = False,
        evaluation_objects: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Set parameter values and run the evaluation graph.

        The `Pipeline` instance is reused across calls.  Cache invalidation is
        handled by ``_EvaluationContext``: each call constructs a context keyed on
        ``(x_key, obj_uuid)``, where ``x_key`` is the registration-ordered
        ``(name, value)`` tuple of the independent assignment.  Results for
        different assignments are naturally distinct cache entries without manual
        cache clearing; assignments that only differ in a categorical value are
        distinct entries too.  Intermediate nodes shared by multiple targets
        within a single call are computed only once.

        When the parameter space has no evaluation objects, the assignment
        itself becomes the root: root nodes (those without ``requires``)
        receive the assignment mapping instead of an evaluation object, and
        the run follows the single-result return convention.

        Parameters
        ----------
        assignment : Mapping
            Values for the independent parameters by name, in physical units.
            Order-insensitive.  Numeric vectors are an encoding owned by
            ``TransformedSpace``; decode first:
            ``pipeline.evaluate(space.transformed_space.decode(x))``.
        targets : list[str], optional
            Output names to compute.  `None` computes all registered outputs.
            Requesting a subset exploits pipefunc's lazy evaluation: only the
            subgraph needed for the requested outputs is executed.
        bypass_cache : bool
            When True, clear the pipeline cache before evaluating so all nodes
            are recomputed from scratch.  Useful for debugging to confirm that
            results are not stale.
        evaluation_objects : list, optional
            Restrict the run to these registered evaluation objects.  `None`
            runs all registered objects.  The return convention follows the
            selected subset: one object gives plain values, several give lists.

        Returns
        -------
        dict[str, Any]
            Mapping from output name to result.  For a single evaluation object
            (or none registered) the values are plain results (or
            `EvaluationFailure`).  For multiple evaluation objects the values
            are lists indexed by evaluation object.  Results may be
            `EvaluationFailure` instances when a node failed.
        """
        if not isinstance(assignment, Mapping):
            raise TypeError(
                "evaluate takes a named assignment (Mapping of parameter name "
                "to value); numeric vectors are an encoding owned by "
                "TransformedSpace — decode first: "
                "pipeline.evaluate(space.transformed_space.decode(x))."
            )

        if targets is None:
            targets = self._output_names
        else:
            unknown = [t for t in targets if t not in self._output_names]
            if unknown:
                raise ValueError(f"Unknown target(s): {unknown}")

        if self._is_mapped:
            return self._evaluate_mapped(
                assignment, targets, bypass_cache, evaluation_objects
            )

        self._space.set_values(assignment)
        x_key: tuple = tuple(
            (p.name, assignment[p.name])
            for p in self._space.independent_parameters
        )
        if bypass_cache:
            # Append a nonce so every node sees a guaranteed cache miss.
            # Existing entries for other assignments are unaffected.
            x_key = x_key + (_uuid_mod.uuid4().hex,)

        eval_objs = self._space.evaluation_objects
        if evaluation_objects is not None:
            if not evaluation_objects:
                raise ValueError("evaluation_objects must not be empty; pass None for all.")
            unknown_objs = [o for o in evaluation_objects if o not in eval_objs]
            if unknown_objs:
                raise ValueError(f"Unknown evaluation object(s): {unknown_objs}")
            eval_objs = list(evaluation_objects)

        pipeline = self._get_pipeline()

        def _run_for_ctx(ctx: _EvaluationContext) -> dict[str, Any]:
            """Run all targets for one root context in a single pipeline call."""
            if len(targets) == 1:
                value = pipeline(targets[0], **{_CONTEXT_ARG: ctx})
                return {targets[0]: value}
            values = pipeline.run(targets, kwargs={_CONTEXT_ARG: ctx})
            assert len(values) == len(targets), (
                f"Expected {len(targets)} results from pipeline.run, got {len(values)}"
            )
            return dict(zip(targets, values))

        if not eval_objs:
            # Zero-evaluation-object mode: the assignment itself is the root.
            ctx = _EvaluationContext(x_key, dict(assignment), _NO_OBJECT_UUID)
            return _run_for_ctx(ctx)

        def _run_for(obj: Any) -> dict[str, Any]:
            return _run_for_ctx(
                _EvaluationContext(x_key, obj, self._space.evaluation_object_uuid(obj))
            )

        if len(eval_objs) == 1:
            return _run_for(eval_objs[0])

        results: dict[str, list] = {t: [] for t in targets}
        for obj in eval_objs:
            for t, v in _run_for(obj).items():
                results[t].append(v)
        return results

    def _evaluate_mapped(
        self,
        assignment: Mapping[str, Any],
        targets: list[str],
        bypass_cache: bool,
        evaluation_objects: list[Any] | None,
    ) -> dict[str, Any]:
        """Run a mapped graph once via ``pipeline.map`` over the object axis.

        The ``set_values`` node writes ``x`` and emits the ordered
        ``evaluation_contexts`` sequence; ``pipeline.map`` fans every mapped node
        over that axis.  Two kinds of target are reshaped differently:

        - A mapped target yields one value per object; it follows the loop's
          convention (one object, or objectless, unwraps to a plain value;
          several give a list).
        - An unmapped target is a whole-value consumer (fan-in): it received the
          full per-object array and reduced it to a single value, which is
          returned as-is.  Only downstream fan-in is possible here; an unmapped
          root is rejected earlier at graph build.

        Failure handling for fan-in is deliberately minimal: pipefunc hands the
        reducer a ``MaskedArray``, and the ``bad_metrics``-preserving policy for
        failed objects is a separate step; this path assumes the happy case.

        Sequential (``parallel=False``) is deliberate for the first increment;
        parallel mapped execution is deferred alongside ``map_async``.
        """
        if evaluation_objects is not None:
            raise NotImplementedError(
                "Restricting evaluation_objects is not yet supported on the mapped "
                "path; per-node subset routing is a separate step. Omit it to run "
                "all objects, or use a graph without mapspecs."
            )
        if bypass_cache:
            raise NotImplementedError(
                "bypass_cache is not yet supported on the mapped path."
            )
        # Build first so graph-validity errors (e.g. an unmapped root) surface
        # before reshaping.
        pipeline = self._get_pipeline()
        result = pipeline.map(
            {"x": dict(assignment)},
            output_names=targets,
            parallel=False,
        )

        mapspec_by_name = dict(zip(self._output_names, self._mapspecs))
        # Objectless (0) and single-object (1) both unwrap to a plain value.
        single = len(self._space.evaluation_objects) <= 1
        out: dict[str, Any] = {}
        for t in targets:
            if mapspec_by_name[t] is None:
                # Whole-value fan-in: the node already reduced to one value.
                out[t] = result[t].output
            else:
                values = list(result[t].output)
                out[t] = values[0] if single else values
        return out
