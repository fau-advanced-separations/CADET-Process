from __future__ import annotations

import copy
import inspect
from typing import Any, Callable, Optional, Sequence

from CADETProcess.fractionation import FractionationOptimizer
from CADETProcess.parameter_space.parameters import (
    LinearConstraint,
    ParameterDotPathSetter,
    ParameterSpace,
    RangedParameter,
)
from CADETProcess.simulator import Cadet
from examples.batch_elution.process import process
from pipefunc import PipeFunc, Pipeline


class EvaluationPipeline:
    """
    Lightweight evaluation pipeline wrapper around `pipefunc.Pipeline`.

    Parameters
    ----------
    parameter_space : ParameterSpace
        Parameter space that knows how to set values on the evaluation objects.
    root_input_name : str, optional
        Explicit name of the first evaluator's root input argument.
        If not given, inferred from the first evaluator's signature.

    Attributes
    ----------
    evaluators : dict[str, PipeFunc]
        Mapping from evaluator name to `PipeFunc`.
    pipeline : Pipeline
        Lazily constructed `Pipeline` over the added evaluators.
    """

    def __init__(self, parameter_space: ParameterSpace,
                 root_input_name: Optional[str] = None) -> None:
        self.parameter_space = parameter_space
        self.evaluators: dict[str, PipeFunc] = {}
        self._pipeline: Optional[Pipeline] = None
        self._root_input_name = root_input_name

    def add_evaluator(
        self,
        evaluator: Callable[..., Any],
        output_name: str,
        name: Optional[str] = None,
        cache: bool = True
    ) -> None:
        """
        Register an evaluator node.

        Parameters
        ----------
        evaluator : callable
            Function to add to the pipeline. Its input should match the previous
            node's output.
        output_name : str
            Name for this node's output within the pipeline DAG.
        name : str, optional
            Stable identifier for the node; defaults to `evaluator.__name__`.
        cache : bool, default=True
            Whether to enable LRU caching for this node.

        Raises
        ------
        TypeError
            If `evaluator` is not callable.
        ValueError
            If `name` already exists.
        """
        if not callable(evaluator):
            raise TypeError("Expected callable evaluator.")
        if name is None:
            name = evaluator.__name__ if hasattr(evaluator, "__name__") else str(evaluator)
        if name in self.evaluators:
            raise ValueError(f"Evaluator with name '{name}' already exists.")
        self.evaluators[name] = PipeFunc(evaluator, output_name=output_name, cache=cache)
        self._pipeline = None  # invalidate so it rebuilds with the new node

    @property
    def pipeline(self) -> Pipeline:
        """
        Build (once) and return the underlying `Pipeline`.

        Returns
        -------
        Pipeline
            Configured pipeline with all registered evaluators.

        Raises
        ------
        RuntimeError
            If no evaluators have been added.
        """
        if self._pipeline is None:
            if not self.evaluators:
                raise RuntimeError("No evaluators added.")
            self._pipeline = Pipeline(
                self.evaluators.values(),
                cache_type="lru",
                cache_kwargs={"shared": False},
                profile=True,
            )
        return self._pipeline

    def _infer_root_input_name(self) -> str:
        """
        Determine the name of the root input argument for the first evaluator.

        Returns
        -------
        str
            The inferred parameter name.

        Raises
        ------
        RuntimeError
            If it cannot locate a suitable callable/parameter.
        """
        if self._root_input_name:
            return self._root_input_name
        first_node = next(iter(self.evaluators.values()))
        fn = getattr(first_node, "func", None) or getattr(first_node, "__call__", None)
        if fn is None:
            raise RuntimeError("Could not locate the underlying callable of the first evaluator.")
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                return p.name
        raise RuntimeError("No suitable input parameter found on the first evaluator.")

    def __call__(self, target: str, x: Sequence[float], **kwargs: Any) -> list[Any]:
        """
        Evaluate the pipeline for all evaluation objects after setting parameter values.

        Parameters
        ----------
        target : str
            Name of the terminal output to compute (must match an `output_name`).
        x : Sequence[float]
            Vector of parameter values in *untransformed* space for
            `ParameterSpace.set_values`.
        **kwargs
            Forwarded keyword arguments to the pipeline call.

        Returns
        -------
        list[Any]
            List of results, one per evaluation object in
            parameter_space.evaluation_objects`.
        """
        self.parameter_space.set_values(x)
        root_kw = self._infer_root_input_name()

        results: list[Any] = []
        for evaluation_object in self.parameter_space.evaluation_objects:
            result = self.pipeline(target, **{root_kw: evaluation_object}, **kwargs)
            results.append(result)
        return results


new_process = copy.deepcopy(process)


def build_parameter_space(new_process: Any) -> ParameterSpace:
    """
    Construct a `ParameterSpace` for batch elution with basic constraints.

    Parameters
    ----------
    new_process : Any
        Process-like object whose attributes will be set by parameter mappers.

    Returns
    -------
    ParameterSpace
        Space with `cycle_time` and `feed_duration` and constraint `feed < cycle`.
    """
    ps = ParameterSpace()
    cycle = RangedParameter(
        name="cycle_time",
        parameter_type=float,
        lb=60.0,
        ub=1800.0,
        normalization="auto",
        mappers=[ParameterDotPathSetter([new_process], "cycle_time")],
    )
    feed = RangedParameter(
        name="feed_duration",
        parameter_type=float,
        lb=5.0,
        ub=300.0,
        normalization="auto",
        mappers=[ParameterDotPathSetter([new_process], "feed_duration.time")],
    )
    ps.add_parameter(cycle)
    ps.add_parameter(feed)
    ps.add_linear_constraint(
        LinearConstraint(parameters=[feed, cycle], lhs=[1.0, -1.0], b=-1e-9)  # feed < cycle
    )
    return ps


def run_simulation(new_process: Any) -> Any:
    """
    Simulate the process with CADET.

    Parameters
    ----------
    new_process : Any
        Configured process instance.

    Returns
    -------
    Any
        Simulation results returned by `Cadet.simulate`.
    """
    process_simulator = Cadet()
    process_simulator.evaluate_stationarity = True
    simulation_results = process_simulator.simulate(new_process)
    return simulation_results


def fractionate(simulation_results: Any) -> Any:
    """
    Optimize fractionation given simulation results.

    Parameters
    ----------
    simulation_results : Any
        Output from the simulator.

    Returns
    -------
    Any
        Fractionation result object with productivity, recovery, etc.
    """
    frac_opt = FractionationOptimizer()
    frac = frac_opt.optimize_fractionation(
        simulation_results,
        purity_required=[0.95, 0.95],
        ignore_failed=False,
        allow_empty_fractions=True,
    )
    return frac


def eval_prod(frac: Any) -> float:
    """
    Extract productivity from a fractionation result.

    Parameters
    ----------
    frac : Any
        Fractionation result.

    Returns
    -------
    float
        Productivity metric.
    """
    return frac.productivity


def eval_recovery(frac: Any) -> float:
    """
    Extract recovery from a fractionation result.

    Parameters
    ----------
    frac : Any
        Fractionation result.

    Returns
    -------
    float
        Recovery metric.
    """
    return frac.recovery


def eval_eluentcons(frac: Any) -> float:
    """
    Extract eluent consumption from a fractionation result.

    Parameters
    ----------
    frac : Any
        Fractionation result.

    Returns
    -------
    float
        Eluent consumption.
    """
    return frac.eluent_consumption


parameter_space = build_parameter_space(new_process)
batch_elution_evaluation_pipeline = EvaluationPipeline(parameter_space)

batch_elution_evaluation_pipeline.add_evaluator(
    run_simulation,
    output_name="simulation_results",
    cache=True,
)

batch_elution_evaluation_pipeline.add_evaluator(
    fractionate,
    output_name="frac",
    cache=True,
)

batch_elution_evaluation_pipeline.add_evaluator(
    eval_prod,
    output_name="prod",
    cache=True,
)

batch_elution_evaluation_pipeline.add_evaluator(
    eval_recovery,
    output_name="recovery",
    cache=True,
)

batch_elution_evaluation_pipeline.add_evaluator(
    eval_eluentcons,
    output_name="eluent",
    cache=True,
)

print("Productivity:", batch_elution_evaluation_pipeline("prod", x=[600.0, 60.0]))
print("Recovery:", batch_elution_evaluation_pipeline("recovery", x=[600.0, 60.0]))
print("Eluent consumption:", batch_elution_evaluation_pipeline("eluent", x=[600.0, 60.0]))
