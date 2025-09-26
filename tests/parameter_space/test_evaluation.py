
from __future__ import annotations

from dataclasses import dataclass

import pytest
from CADETProcess.projects.pipelines.batch_elution import EvaluationPipeline


@dataclass
class DummyProcess:
    value: float = 0.0


class DummyParameterSpace:
    """
    Minimal double for your real ParameterSpace.
    - Has .evaluation_objects
    - Has .set_values(x) to write values into each object
    - Optional: .size / .describe for better error messages if you add guards later
    """
    def __init__(self, evaluation_objects):
        self.evaluation_objects = list(evaluation_objects)

    @property
    def size(self) -> int:
        return 1

    def describe(self):
        return ["value"]

    def set_values(self, x):
        if len(x) != 1:
            raise ValueError(f"Expected x of length 1, got {len(x)}")
        v = float(x[0])
        for obj in self.evaluation_objects:
            obj.value = v


@pytest.fixture
def evaluation_objects():
    return [DummyProcess(), DummyProcess()]


@pytest.fixture
def parameter_space(evaluation_objects):
    return DummyParameterSpace(evaluation_objects)


@pytest.fixture
def pipeline(parameter_space):
    return EvaluationPipeline(parameter_space)
