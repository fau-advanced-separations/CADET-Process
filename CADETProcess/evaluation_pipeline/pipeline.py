from __future__ import annotations

import inspect
import re
import uuid as _uuid_mod
from collections.abc import Callable, Mapping
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
# Output name of the ``set_values`` root node and the axis input name a mapped
# node consumes: ``evaluation_contexts[object] -> output[object]``.
_EVALUATION_CONTEXTS = "evaluation_contexts"
# The one mapped axis name.  All mapspec strings are generated through
# ``_generate_mapspec`` so the string form cannot drift; user-supplied
# ``mapspec=`` strings are the escape hatch and pass through verbatim.
_OBJECT_AXIS = "object"
# Stable UUID slot for the zero-evaluation-object mode: cache entries are
# keyed on (x_key, uuid), and without an object the x_key alone identifies
# the evaluation.  A fixed constant keeps keys stable across processes.
_NO_OBJECT_UUID = "__no_evaluation_object__"
# Whole-value root inputs to the ``set_values`` node on the mapped path: the
# per-call resolved object subset and the cache-busting nonce.  Both are
# consumed whole (never indexed by a mapspec), so ``pipeline.map`` treats them
# as scalar roots, not mapped axes.
_RUN_OBJECTS_ARG = "evaluation_objects"
_CACHE_NONCE_ARG = "cache_nonce"


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


def _resolve_evaluation_objects(
    all_objects: list[Any], requested: list[Any] | None
) -> list[Any]:
    """Resolve a per-call evaluation-object restriction against the registered set.

    `None` runs all registered objects in registration order.  A list restricts
    the run to those objects in request order (matching the legacy loop, which
    does ``list(evaluation_objects)``); an unknown object or an empty list
    raises.  This is a whole-run restriction, distinct from mid-graph per-node
    subset routing: excluded objects never enter the graph.
    """
    if requested is None:
        return list(all_objects)
    if not requested:
        raise ValueError("evaluation_objects must not be empty; pass None for all.")
    unknown = [o for o in requested if o not in all_objects]
    if unknown:
        raise ValueError(f"Unknown evaluation object(s): {unknown}")
    return list(requested)


def _find_failure(value: Any, collects: bool) -> EvaluationFailure | None:
    """Return a propagating failure found in *value*, else None.

    A scalar argument is a failure when it is itself an `EvaluationFailure`.
    A fan-in node (`collects`) also fails when any element of its mapped input
    array is one: a failed object must fail the aggregate (the `bad_metrics`
    default) rather than reach the user reducer as a sentinel among floats.
    The array scan is gated on `collects` so per-object nodes never iterate a
    large numeric data array looking for sentinels that cannot be there.
    """
    if isinstance(value, EvaluationFailure):
        return value
    if collects and isinstance(value, np.ndarray):
        for element in value.ravel():
            if isinstance(element, EvaluationFailure):
                return element
    return None


def _demask(value: Any) -> Any:
    """Strip the mask from a `MaskedArray` collector input, else pass through.

    A `collects` fan-in receives its mapped axis as a ``MaskedArray``; by the
    time the reducer runs, ``_find_failure`` has guaranteed no element is masked
    (a failed object was propagated already).  Hand the reducer a plain
    ``ndarray``: numpy's masked-array reductions (``np.max``/``np.min``) crash on
    an all-unmasked array via a scalar ``.view``, silently turning a legitimate
    worst-case reducer into a ``bad_metrics`` result.
    """
    if isinstance(value, np.ma.MaskedArray):
        return np.asarray(value)
    return value


def _wrap_with_failure_propagation(
    func: Callable, stage: str, collects: bool = False
) -> Callable:
    """Return a wrapper that propagates EvaluationFailure and catches exceptions.

    If any argument is an `EvaluationFailure` (or, for a `collects` fan-in node,
    contains one in its mapped array), that failure is returned immediately
    without calling `func`.  Otherwise `func` is called normally; any exception
    is caught and returned as a new `EvaluationFailure`.

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
            failure = _find_failure(v, collects)
            if failure is not None:
                return failure
        if collects:
            args = tuple(_demask(a) for a in args)
            kwargs = {k: _demask(v) for k, v in kwargs.items()}
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

    Two further whole-value root inputs carry per-call control that a baked
    closure cannot see: *evaluation_objects* is the already-resolved run subset
    (excluded objects never enter the axis), and *cache_nonce* is appended to
    ``x_key`` when set (``bypass_cache``) so every downstream node misses this
    run while other assignments' cache entries stay intact.
    """

    def set_values(
        x: Mapping[str, Any],
        evaluation_objects: list[Any],
        cache_nonce: str | None,
    ) -> list[_EvaluationContext]:
        space.set_values(x)
        x_key: tuple = tuple(
            (p.name, x[p.name]) for p in space.independent_parameters
        )
        if cache_nonce is not None:
            x_key = x_key + (cache_nonce,)
        if not evaluation_objects:
            # Objectless mode is one sentinel context, not an empty axis: an
            # empty fan would leave a downstream scalar reducer with nothing.
            return [_EvaluationContext(x_key, dict(x), _NO_OBJECT_UUID)]
        return [
            _EvaluationContext(x_key, obj, space.evaluation_object_uuid(obj))
            for obj in evaluation_objects
        ]

    set_values.__signature__ = inspect.Signature(
        [
            inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                _RUN_OBJECTS_ARG, inspect.Parameter.POSITIONAL_OR_KEYWORD
            ),
            inspect.Parameter(
                _CACHE_NONCE_ARG, inspect.Parameter.POSITIONAL_OR_KEYWORD
            ),
        ]
    )
    return PipeFunc(set_values, output_name=_EVALUATION_CONTEXTS, cache=False)


class _NodeSpec:
    """Registration record for one evaluator; the PipeFunc is built later.

    Node construction is deferred to ``_get_pipeline`` because per-object
    semantics resolve against the whole graph: a collector
    (``per_object=False``) registered last makes every earlier default node
    mapped, so effective mapspecs cannot be finalized at registration time.
    """

    __slots__ = ("func", "output_name", "requires", "cache", "per_object", "mapspec")

    def __init__(
        self,
        func: Callable,
        output_name: str,
        requires: list[str] | None,
        cache: bool,
        per_object: bool | None,
        mapspec: str | None,
    ) -> None:
        self.func = func
        self.output_name = output_name
        self.requires = requires
        self.cache = cache
        self.per_object = per_object
        self.mapspec = mapspec


def _generate_mapspec(
    output_name: str,
    requires: list[str] | None,
    axis_inputs: set[str],
) -> str:
    """Generate the mapspec string for a per-object node.

    The single point where mapspec strings are written, so the string form
    cannot drift as signatures evolve.  Inputs in *axis_inputs* carry the
    object axis and are indexed; other inputs (outputs of collectors) are
    consumed whole.  A root node (``requires is None``) fans over the
    ``evaluation_contexts`` sequence.
    """
    if requires is None:
        input_names = [_EVALUATION_CONTEXTS]
        axis_inputs = {_EVALUATION_CONTEXTS}
    else:
        input_names = requires
    inputs = ", ".join(
        f"{name}[{_OBJECT_AXIS}]" if name in axis_inputs else name
        for name in input_names
    )
    return f"{inputs} -> {output_name}[{_OBJECT_AXIS}]"


def _mapspec_output_has_axis(mapspec: str) -> bool:
    """Whether an explicit mapspec string produces an axis-bearing output."""
    _, _, rhs = mapspec.partition("->")
    return f"[{_OBJECT_AXIS}]" in rhs


def _make_node(
    func: Callable,
    output_name: str,
    requires: list[str] | None,
    cache: bool = True,
    mapspec: str | None = None,
    collects: bool = False,
) -> PipeFunc:
    """Build a `PipeFunc` node with failure propagation and optional arg injection.

    A node with ``requires`` receives its named upstream outputs (single elements
    under a mapspec, whole values otherwise).  A root node (``requires is None``)
    consumes the object context: the ``evaluation_contexts`` axis when it is
    mapped, or the legacy per-object ``_CONTEXT_ARG`` root when it is not.

    ``collects`` marks a mapped fan-in node (whole-value consumer of a mapped
    axis): its wrapper scans its input array and propagates a failure if any
    object failed, so a failed object fails the aggregate rather than reaching
    the user reducer as a sentinel among floats.
    """
    safe = _wrap_with_failure_propagation(func, stage=output_name, collects=collects)
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
        self._specs: list[_NodeSpec] = []
        self._output_names: list[str] = []
        self._pipeline: Pipeline | None = None
        # Effective per-node mapspecs (explicit or generated), finalized at
        # build time; None entries are whole-value consumers.
        self._effective_mapspecs: dict[str, str | None] = {}

    # ------------------------------------------------------------------
    # Registration

    def add_evaluator(
        self,
        func: Callable,
        output_name: str,
        requires: list[str] | None = None,
        cache: bool = True,
        per_object: bool | None = None,
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
        per_object : bool, optional
            Whether this node runs once per evaluation object (the default
            semantic) or once, collecting the complete per-object array of its
            inputs (``False``, a whole-value consumer that may reduce it).
            Declaring ``False`` on any node engages mapped execution for the
            graph; the effective mapspec strings are then generated at build
            time through one central helper.  Leaving it unset keeps the
            per-object default, which the legacy loop and the mapped engine
            implement identically.  Mutually exclusive with `mapspec`.
        mapspec : str, optional
            Escape hatch: a raw pipefunc axis-mapping string, e.g.
            ``"evaluation_contexts[object] -> out[object]"``, passed through
            verbatim.  When any node is mapped the graph is executed once via
            ``pipeline.map`` over the whole object axis instead of the legacy
            per-object loop.

        Raises
        ------
        TypeError
            If `func` is not callable.
        ValueError
            If `output_name` is already registered or is not a valid identifier,
            if any entry in `requires` is not a valid identifier, if both
            `per_object` and `mapspec` are supplied, or if ``per_object=False``
            is declared on a root node (nothing to collect).
        """
        if not callable(func):
            raise TypeError(f"Expected callable, got {type(func).__name__}")
        _validate_identifier(output_name)
        for required in requires or []:
            _validate_identifier(required)
        if output_name in self._output_names:
            raise ValueError(f"output_name {output_name!r} is already registered")
        if per_object is not None and mapspec is not None:
            raise ValueError(
                "per_object and mapspec are mutually exclusive: per_object is "
                "the normal API, mapspec the raw escape hatch; supply one."
            )
        if per_object is False and requires is None:
            raise ValueError(
                f"per_object=False on root node {output_name!r}: a root has no "
                "mapped upstream to collect. Declare requires= or drop "
                "per_object."
            )

        self._specs.append(
            _NodeSpec(func, output_name, requires, cache, per_object, mapspec)
        )
        self._output_names.append(output_name)
        self._pipeline = None  # invalidate cached pipeline

    # ------------------------------------------------------------------
    # Evaluation

    @property
    def output_names(self) -> list[str]:
        """list[str]: All registered output names, in registration order."""
        return list(self._output_names)

    @property
    def _is_mapped(self) -> bool:
        """Whether the graph runs via mapped execution.

        The mapped engine engages only when required: an explicit mapspec, or
        a collector (``per_object=False``), whose fan-in semantics the legacy
        loop cannot express.  An explicit ``per_object=True`` alone does not
        engage it: the legacy loop already implements per-object semantics,
        and staying legacy keeps subset and ``bypass_cache`` support until
        mapped-path parity lands.
        """
        return any(
            s.mapspec is not None or s.per_object is False for s in self._specs
        )

    def _resolve_mapspecs(self) -> dict[str, str | None]:
        """Finalize the effective mapspec for every node.

        In an unmapped graph every entry is None (byte-for-byte the legacy
        graph).  In a mapped graph, default (``per_object`` unset or True)
        nodes get generated mapspecs: roots fan over ``evaluation_contexts``;
        downstream nodes fan over whichever of their inputs carry the axis.  A
        node whose inputs all come from collectors has no axis to fan over and
        stays a whole-value node.  Specs are processed producers-first so
        axis-bearing propagates through chains regardless of registration
        order.
        """
        if not self._is_mapped:
            return {s.output_name: None for s in self._specs}

        effective: dict[str, str | None] = {}
        axis_bearing: dict[str, bool] = {}
        pending = list(self._specs)
        registered = {s.output_name for s in self._specs}
        while pending:
            progressed = False
            for spec in list(pending):
                deps = [
                    r for r in (spec.requires or []) if r in registered
                ]
                if any(d not in axis_bearing for d in deps):
                    continue  # a producer is not resolved yet
                if spec.mapspec is not None:
                    effective[spec.output_name] = spec.mapspec
                    axis_bearing[spec.output_name] = _mapspec_output_has_axis(
                        spec.mapspec
                    )
                elif spec.per_object is False:
                    effective[spec.output_name] = None
                    axis_bearing[spec.output_name] = False
                else:  # per-object default (unset or explicit True)
                    axis_inputs = {d for d in deps if axis_bearing[d]}
                    if spec.requires is None or axis_inputs:
                        effective[spec.output_name] = _generate_mapspec(
                            spec.output_name, spec.requires, axis_inputs
                        )
                        axis_bearing[spec.output_name] = True
                    else:
                        # Downstream of collectors only: no axis to fan over.
                        effective[spec.output_name] = None
                        axis_bearing[spec.output_name] = False
                pending.remove(spec)
                progressed = True
            if not progressed:
                cycle = [s.output_name for s in pending]
                raise ValueError(f"Cyclic requires among nodes: {cycle}")
        return effective

    def _graph_nodes(self) -> list[PipeFunc]:
        """Build the pipefunc nodes, prepending ``set_values`` when mapped.

        A mapped graph roots on ``x`` and fans the object axis off the
        ``evaluation_contexts`` sequence, so the ``set_values`` node must
        exist.  An unmapped graph is byte-for-byte the legacy per-object
        graph.
        """
        self._effective_mapspecs = self._resolve_mapspecs()
        mapped = self._is_mapped
        nodes = [
            _make_node(
                s.func,
                s.output_name,
                s.requires,
                cache=s.cache,
                mapspec=self._effective_mapspecs[s.output_name],
                # A mapped whole-value consumer (mapspec None, has upstreams) is
                # a fan-in: scan its input array for failed objects.
                collects=(
                    mapped
                    and self._effective_mapspecs[s.output_name] is None
                    and s.requires is not None
                ),
            )
            for s in self._specs
        ]
        if not mapped:
            return nodes
        return [_make_set_values_node(self._space), *nodes]

    def _get_pipeline(self) -> Pipeline:
        """Return the cached pipeline, building it if necessary.

        The pipeline is rebuilt whenever ``add_evaluator`` is called after a
        previous build.  Within a stable registration phase the same instance
        is reused across ``evaluate`` calls.
        """
        if self._pipeline is None:
            if not self._specs:
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

        eval_objs = _resolve_evaluation_objects(
            self._space.evaluation_objects, evaluation_objects
        )

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
          returned as-is.  Roots without an explicit declaration fan per object
          (the per-object default); ``per_object=False`` roots are rejected at
          registration.

        A failed object propagates its ``EvaluationFailure`` sentinel in the
        mapped array, and a fan-in node fails the aggregate rather than reducing
        over it (see ``_find_failure``); no ``MaskedArray`` is involved.

        Per-call control reaches the baked ``set_values`` root through two
        whole-value ``pipeline.map`` inputs: the resolved run subset restricts
        the object axis, and a cache nonce (set only when ``bypass_cache``)
        forces every node to miss this run.

        Sequential (``parallel=False``) is deliberate for the first increment;
        parallel mapped execution is deferred alongside ``map_async``.
        """
        effective_objects = _resolve_evaluation_objects(
            self._space.evaluation_objects, evaluation_objects
        )
        cache_nonce = _uuid_mod.uuid4().hex if bypass_cache else None
        # ``output_names`` trims the run to the requested targets (unrequested
        # callbacks never fire).  pipefunc builds the trimmed subpipeline over
        # the main pipeline's (failure-guarded) cache, so cache entries persist
        # across calls and shared upstream work is reused across target sets
        # (pipefunc/pipefunc#975, released in 0.93.1; the pinned floor).
        pipeline = self._get_pipeline()
        result = pipeline.map(
            {
                "x": dict(assignment),
                _RUN_OBJECTS_ARG: effective_objects,
                _CACHE_NONCE_ARG: cache_nonce,
            },
            output_names=set(targets),
            parallel=False,
        )

        mapspec_by_name = self._effective_mapspecs
        # Objectless (0) and single-object (1) both unwrap to a plain value;
        # base this on the effective (restricted) objects, not the full space.
        single = len(effective_objects) <= 1
        out: dict[str, Any] = {}
        for t in targets:
            if mapspec_by_name[t] is None:
                # Whole-value fan-in: the node already reduced to one value.
                out[t] = result[t].output
            else:
                values = list(result[t].output)
                out[t] = values[0] if single else values
        return out
