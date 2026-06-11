from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvaluationFailure:
    """Propagating failure token returned when a pipeline node cannot produce a result.

    Instead of raising, failed nodes return an `EvaluationFailure`.
    Downstream nodes detect it and pass it through without executing.
    The `bad_metrics` fallback at the optimizer boundary is a separate concern
    handled by `OptimizationProblem` (step 6).

    Parameters
    ----------
    stage : str
        `output_name` of the node that failed or first propagated the failure.
    reason : str
        Human-readable description of the failure cause.
    recoverable : bool
        Whether the caller may attempt to recover (e.g. retry with a different x).
        The default is False.
    exc : BaseException or None
        The original exception, if any.  Preserved for debugging and re-raising.
    """

    stage: str
    reason: str
    recoverable: bool = False
    exc: BaseException | None = field(default=None, repr=False, compare=False)

    def __str__(self) -> str:  # noqa: D105
        tag = " (recoverable)" if self.recoverable else ""
        return f"EvaluationFailure at '{self.stage}': {self.reason}{tag}"
