import types
import warnings

import matplotlib
import numpy as np
import pytest
from scipy import stats

matplotlib.use("Agg")

from CADETProcess import CADETProcessError
from CADETProcess.comparison import SSE, Comparator
from CADETProcess.processModel import ComponentSystem
from CADETProcess.reference import ReferenceIO

_component_system = ComponentSystem(1)
_time = np.linspace(0, 100, 1001)
_signal = stats.norm.pdf(_time, 50, 5).reshape(-1, 1)

SOLUTION_PATH = "outlet.outlet"


@pytest.fixture
def reference():
    return ReferenceIO("signal", _time, _signal, component_system=_component_system)


@pytest.fixture
def metric(reference):
    return SSE(reference)


@pytest.fixture
def simulation_results(reference):
    stub = types.SimpleNamespace()
    stub.solution_cycles = {"outlet": {"outlet": [reference]}}
    return stub


@pytest.fixture
def comparator():
    return Comparator()


class TestAddDifferenceMetric:
    def test_instance_registered(self, comparator, metric):
        returned = comparator.add_difference_metric(metric, SOLUTION_PATH)
        assert returned is metric
        assert metric in comparator.metrics
        assert comparator.solution_paths[metric] == SOLUTION_PATH

    def test_instance_no_warning(self, comparator, metric):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            comparator.add_difference_metric(metric, SOLUTION_PATH)
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_string_emits_deprecation_warning(self, comparator, reference):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            comparator.add_reference(reference)
            comparator.add_difference_metric("SSE", reference.name, SOLUTION_PATH)
        deprecation_categories = [
            x.category for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_categories) >= 1
        assert len(comparator.metrics) == 1

    def test_string_unknown_metric_raises(self, comparator, reference):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            comparator.add_reference(reference)
            with pytest.raises(CADETProcessError):
                comparator.add_difference_metric("NoSuchMetric", reference.name, SOLUTION_PATH)

    def test_wrong_type_raises(self, comparator):
        with pytest.raises(TypeError):
            comparator.add_difference_metric(42, SOLUTION_PATH)


class TestEvaluate:
    def test_sse_identical_signals_is_zero(self, comparator, metric, simulation_results):
        comparator.add_difference_metric(metric, SOLUTION_PATH)
        result = comparator.evaluate(simulation_results)
        np.testing.assert_almost_equal(result, [0.0])

    def test_evaluate_multiple_metrics(self, comparator, reference, simulation_results):
        from CADETProcess.comparison import NRMSE
        metric_sse = SSE(reference)
        metric_nrmse = NRMSE(reference)
        comparator.add_difference_metric(metric_sse, SOLUTION_PATH)
        comparator.add_difference_metric(metric_nrmse, SOLUTION_PATH)
        result = comparator.evaluate(simulation_results)
        assert len(result) == 2
        np.testing.assert_almost_equal(result, [0.0, 0.0])


class TestPlotComparison:
    def test_smoke(self, comparator, metric, simulation_results):
        comparator.add_difference_metric(metric, SOLUTION_PATH)
        fig, axs = comparator.plot_comparison(simulation_results)
        assert fig is not None
        assert len(axs) == 1

    def test_raises_without_metrics(self, comparator, simulation_results):
        with pytest.raises(CADETProcessError):
            comparator.plot_comparison(simulation_results)
