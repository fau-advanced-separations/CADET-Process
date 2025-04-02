from unittest.mock import Mock

import numpy as np
import pytest
from addict import Dict
from CADETProcess import SimulationResults
from CADETProcess.fractionation import FractionationOptimizer
from CADETProcess.processModel import ComponentSystem
from CADETProcess.solution import SolutionIO


def gauss_func(x, mu, sig):
    """There has to be a numpy gauss function."""
    exp = (x - mu) * (x - mu) / (2.0 * sig * sig)
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(- exp) / 2
    )


class GaussianPulse(SolutionIO):
    """Class to create a faked SolutionIO object with a Gauss function."""

    def __init__(self, solution_struct, comp_system):
        """solution struct has to be a dictionary with
        {"component":[[y0, sigma],...]}
        """
        n_time = 1001
        n_comp = len(solution_struct)

        time = np.linspace(0, 10, n_time)
        solution = np.zeros((n_time, n_comp))
        q_const = np.ones(time.shape)

        for comp_index, component in enumerate(solution_struct.values()):
            for gaussian in component:
                t0 = np.searchsorted(time, gaussian[0])
                sigma = np.searchsorted(time, gaussian[1])
                solution[:, comp_index] += gauss_func(np.arange(n_time), t0, sigma)

        super().__init__("GaussianPulse", comp_system, time, solution, q_const)


# %%

class RectanglePulse(SolutionIO):
    """Class to create a faked SolutionIO object with a Rectangle function."""

    def __init__(self, solution_struct, comp_system):
        """solution struct has to be a dictionary with
        {"component":[[start, end],...]}
        """

        n_time = 1001
        n_comp = len(solution_struct)

        time = np.linspace(0, 10, n_time)
        solution = np.zeros((n_time, n_comp))

        # Find the indices for the start and end times
        for comp_index, component in enumerate(solution_struct.values()):
            for rect in component:
                start_index = np.searchsorted(time, rect[0])
                end_index = np.searchsorted(time, rect[1])
                solution[start_index:end_index, comp_index] = 1

        q_const = np.ones(time.shape)

        super().__init__('rect_pulse', comp_system, time, solution, q_const)


def get_simulation_results(process, chromatograms, component_system):
    simulation_results = Mock()
    simulation_results.__class__ = SimulationResults

    simulation_results.chromatograms = chromatograms
    simulation_results.component_system = component_system
    simulation_results.process = process
    return simulation_results


@pytest.fixture
def comp_1_rect_1_chrom_1():
    component_system = ComponentSystem(1)
    sol = [RectanglePulse({"a": [[3, 7]]}, component_system)]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1])
    })

    return (process, sol, component_system)


@pytest.fixture
def comp_1_rect_2_chrom_1():
    component_system = ComponentSystem(1)
    sol = [RectanglePulse({"a": [[2, 4], [7, 9]]}, component_system)]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1])
    })

    return (process, sol, component_system)


@pytest.fixture
def comp_1_rect_2_close_chrom_1():
    component_system = ComponentSystem(1)
    sol = [RectanglePulse({"a": [[2, 4.9], [5.1, 9]]}, component_system)]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1])
    })

    return (process, sol, component_system)


@pytest.fixture
def comp_1_rect_2_chrom_2():
    component_system = ComponentSystem(1)
    sol = [
        RectanglePulse({"a": [[2, 4], [7, 9]]}, component_system),
        RectanglePulse({"a": [[2, 4], [7, 9]]}, component_system)
    ]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1])
    })

    return (process, sol, component_system)


@pytest.fixture
def comp_2_rect_1_chrom_1():
    component_system = ComponentSystem(2)
    sol = [
        RectanglePulse({"a": [[2, 4]], "b": [[7, 9]]}, component_system)
    ]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1, 1])
    })

    return (process, sol, component_system)


@pytest.fixture
def comp_2_rect_1_touching_chrom_1():
    component_system = ComponentSystem(2)
    sol = [
        RectanglePulse({"a": [[2, 5]], "b": [[5, 7]]}, component_system)
    ]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1, 1])
    })

    return (process, sol, component_system)


@pytest.fixture
def comp_2_rect_1_overlapping_chrom_1():
    component_system = ComponentSystem(2)
    sol = [
        RectanglePulse({"a": [[2, 5.1]], "b": [[5, 7]]}, component_system)
    ]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1, 1])
    })

    return (process, sol, component_system)


@pytest.fixture
def comp_1_gauss_1_chrom_1():
    component_system = ComponentSystem(1)
    sol = [
        GaussianPulse({"a": [[5, 1]]}, component_system)
    ]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1, 1])
    })

    return (process, sol, component_system)


@pytest.fixture
def comp_2_gauss_1_chrom_1():
    component_system = ComponentSystem(2)
    sol = [
        GaussianPulse({"a": [[3, 1]], "b": [[7, 1]]}, component_system)
    ]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1, 1])
    })

    return (process, sol, component_system)


@pytest.fixture
def comp_2_gauss_1_close_chrom_1():
    component_system = ComponentSystem(2)
    sol = [
        GaussianPulse({"a": [[4, 1.5]], "b": [[6, 1.5]]}, component_system)
    ]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1, 1])
    })

    return (process, sol, component_system)


@pytest.fixture
def comp_3_gauss_1_chrom_1():
    component_system = ComponentSystem(3)
    sol = [
        GaussianPulse({"a": [[2.5, 0.5]], "b": [[5, 0.5]], "c": [[7.5, 0.5]]}, component_system)
    ]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1, 1, 1])
    })

    return (process, sol, component_system)


@pytest.fixture
def comp_2_gauss_2_chrom_1():
    component_system = ComponentSystem(2)
    sol = [
        GaussianPulse({"a": [[2.5, 0.5], [7.5, 0.5]], "b": [[5, 0.5]]}, component_system)
    ]
    process = Dict({
        "V_solid": 1,
        "cycle_time": sol[0].cycle_time,
        "V_eluent": 1,
        "m_feed": np.array([1, 1])
    })

    return (process, sol, component_system)


@pytest.fixture
def COBYLA():
    fractionation_optimizer = FractionationOptimizer()
    return fractionation_optimizer


@pytest.mark.parametrize("optimizer, parameter, expected_performance", [
    ("COBYLA", "comp_1_rect_1_chrom_1", [0.95]),
    ("COBYLA", "comp_1_rect_2_chrom_1", [0.95]),
    ("COBYLA", "comp_1_rect_2_close_chrom_1", [0.95]),
    ("COBYLA", "comp_1_rect_2_chrom_2", [0.95]),
    ("COBYLA", "comp_2_rect_1_chrom_1", [0.95, 0.95]),
    ("COBYLA", "comp_2_rect_1_touching_chrom_1", [0.95, 0.95]),
    ("COBYLA", "comp_2_rect_1_overlapping_chrom_1", [0.95, 0.95]),
    ("COBYLA", "comp_2_gauss_1_chrom_1", [0.95, 0.95]),
    ("COBYLA", "comp_1_gauss_1_chrom_1", [0.95]),
    ("COBYLA", "comp_2_gauss_1_close_chrom_1", [0.95, 0.95]),
    ("COBYLA", "comp_3_gauss_1_chrom_1", [0.95, 0.95, 0.95]),
    ("COBYLA", "comp_2_gauss_2_chrom_1", [0.95, 0.95])
    ])
def test_cases(optimizer, parameter, expected_performance, request):
    parameter = request.getfixturevalue(parameter)
    simulation_results = get_simulation_results(*parameter)

    fractionation_optimizer = request.getfixturevalue(optimizer)
    frac = fractionation_optimizer.optimize_fractionation(
         simulation_results, expected_performance
    )
    np.testing.assert_array_less(np.array(expected_performance) - 1e-3, frac.performance.purity)
