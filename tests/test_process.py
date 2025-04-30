import copy

import numpy as np
import pytest
from CADETProcess.processModel import ComponentSystem, FlowSheet, Inlet, Process
from examples.batch_elution.process import process as batch_elution_process

TEST_PROFILE_CYCLE_TIME = 155 * 60


def new_inlet_only_process() -> Process:
    """Create a new inlet-only process for testing."""
    component_system = ComponentSystem(1)
    flow_sheet = FlowSheet(component_system)
    inlet = Inlet(component_system, "inlet")
    flow_sheet.add_unit(inlet)
    process = Process(flow_sheet, "inlet_only")
    process.cycle_time = TEST_PROFILE_CYCLE_TIME
    return process


def new_batch_elution_process() -> Process:
    """Create a deep copy of the batch elution process for testing."""
    return copy.deepcopy(batch_elution_process)


@pytest.fixture(params=["inlet_only"])
def process_fixture(request) -> Process:
    """
    Pytest fixture that initializes a fresh process for every test.

    Supports `inlet_only` (default) and `batch_elution` (for future tests).
    """
    if request.param == "inlet_only":
        return new_inlet_only_process()
    elif request.param == "batch_elution":
        return new_batch_elution_process()
    else:
        raise ValueError(f"Unknown process type: {request.param}")


def derivative_of_negative_gaussian(
    t: np.ndarray,
    center: float,
    sigma: float,
    amplitude: float,
    constant_value: float,
) -> np.ndarray:
    """
    Compute the derivative of a negative Gaussian function.

    Used for generating flow rate profiles.
    """
    neg_gauss = -amplitude * np.exp(-((t - center) ** 2) / (2 * sigma**2))
    neg_gauss_derivative = -((t - center) / (sigma**2)) * neg_gauss
    return neg_gauss_derivative + constant_value


def generate_input_profile(
    profile_type: str = "gaussian",
    duration: float = TEST_PROFILE_CYCLE_TIME,
    n_points: int = 1,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a time and flow rate profile for different test cases."""
    if profile_type == "gaussian":
        time_points = np.linspace(0, duration, n_points)
        pulse_center = kwargs.get("pulse_center", duration * 0.6)
        sigma = kwargs.get("sigma", duration * 0.1)
        amplitude = kwargs.get("amplitude", 1e-10)
        constant_value = kwargs.get("constant_value", 1e-5)
        input_profile = derivative_of_negative_gaussian(
            time_points, pulse_center, sigma, amplitude, constant_value
        )

    elif profile_type == "constant":
        time_points = np.linspace(0, duration, n_points)
        input_profile = np.full_like(time_points, kwargs.get("value", 1e-5))

    elif profile_type == "linear":
        time_points = np.linspace(0, duration, n_points)
        input_profile = np.linspace(
            kwargs.get("start", -1e-5), kwargs.get("end", 1e-5), n_points
        )

    elif profile_type == "two_point":
        time_points = np.array(
            [0, 1]
        )  # At least two points to prevent interpolation errors
        input_profile = np.array([kwargs.get("value", 1e-5), kwargs.get("value", 1e-5)])

    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

    return time_points, input_profile


def test_add_flow_rate_profile_negative_gaussian(process_fixture: Process) -> None:
    """Test adding a negative Gaussian flow rate profile."""
    process = process_fixture
    time_points, input_profile = generate_input_profile(
        "gaussian", n_points=TEST_PROFILE_CYCLE_TIME
    )

    process.add_flow_rate_profile("inlet", time_points, input_profile)

    assert len(process.events) > 0, "No events were added to the process"
    assert np.allclose(
        input_profile,
        process.parameter_timelines["flow_sheet.inlet.flow_rate"].value(time_points),
        rtol=1e-2,
        atol=1e-8,
    ), "Event does not match the expected flow rate profile."


def test_add_flow_rate_profile_two_point(process_fixture: Process) -> None:
    """Test adding a flow rate profile with only two points."""
    process = process_fixture
    time_points, input_profile = generate_input_profile("two_point")

    process.add_flow_rate_profile(
        "inlet", time_points, input_profile, interpolation_method="pchip"
    )

    assert len(process.events) > 0, "No events were added for a two-point profile"
    assert np.allclose(
        input_profile,
        process.parameter_timelines["flow_sheet.inlet.flow_rate"].value(time_points),
        rtol=1e-2,
        atol=1e-8,
    ), "Event does not match the expected flow rate profile."


def test_add_flow_rate_profile_constant_function(process_fixture: Process) -> None:
    """Test adding a constant flow rate profile."""
    process = process_fixture
    time_points, input_profile = generate_input_profile("constant", n_points=50)

    process.add_flow_rate_profile(
        "inlet", time_points, input_profile, interpolation_method="pchip"
    )

    assert len(process.events) > 0, "No events were added for a constant function"
    assert np.allclose(
        input_profile,
        process.parameter_timelines["flow_sheet.inlet.flow_rate"].value(time_points),
        rtol=1e-2,
        atol=1e-8,
    ), "Event does not match the expected flow rate profile."


def test_add_flow_rate_profile_unordered_time(process_fixture: Process) -> None:
    """Test exception when time points are unordered."""
    process = process_fixture
    time_points = np.array([10, 0, 20, 5])
    input_profile = np.array([1e-5, 1e-5, 1e-5, 1e-5])

    with pytest.raises(ValueError, match="`x` must be strictly increasing sequence."):
        process.add_flow_rate_profile(
            "inlet", time_points, input_profile, interpolation_method="pchip"
        )


@pytest.mark.parametrize("method", ["cubic", "pchip", None])
def test_add_flow_rate_profile_different_interpolation_methods(
    process_fixture: Process, method: str
) -> None:
    """Test different interpolation methods for adding a flow rate profile."""
    process = process_fixture
    time_points, input_profile = generate_input_profile(
        "gaussian", n_points=TEST_PROFILE_CYCLE_TIME
    )

    process.add_flow_rate_profile(
        "inlet", time_points, input_profile, interpolation_method=method
    )

    assert len(process.events) > 0, (
        f"No events were added for interpolation method {method}"
    )
    assert np.allclose(
        input_profile,
        process.parameter_timelines["flow_sheet.inlet.flow_rate"].value(time_points),
        rtol=1e-2,
        atol=1e-8,
    ), f"Event does not match the expected flow rate profile for method {method}."
