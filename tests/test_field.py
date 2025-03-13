import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from CADETProcess.field import Field, FieldInterpolator


# %% Testing utilities

def assert_equal(value, expected, message=""):
    """Assert equality."""
    message = f"Test failed: {message}. Expected {expected}, got {value}."
    assert value == expected, message


def assert_shape(shape, expected_shape, message=""):
    """Assert equality of shapes."""
    message = f"Test failed: {message}. Expected shape {expected_shape}, got {shape}."
    assert shape == expected_shape, message


# %% Initialization

def test_field_initialization():
    """Test initialization of Field class."""
    # Scalar field
    dimensions = {"axial": np.linspace(0, 10, 11), "radial": np.linspace(0, 5, 6)}
    viscosity = Field(name="viscosity", dimensions=dimensions)
    assert_shape(viscosity.shape, (11, 6), "Scalar field shape")
    assert_equal(viscosity.n_dof, 11 * 6, "Scalar field degrees of freedom")

    # Vector field
    concentration = Field(name="concentration", dimensions=dimensions, n_components=3)
    assert_shape(concentration.shape, (11, 6, 3), "Vector field shape")
    assert_equal(concentration.n_dof, 11 * 6 * 3, "Vector field degrees of freedom")

    # Custom data
    data = np.ones((11, 6, 3))
    concentration_with_data = Field(
        name="concentration", dimensions=dimensions, n_components=3, data=data
    )
    assert_shape(concentration_with_data.shape, (11, 6, 3), "Custom data field shape")
    assert_array_equal(concentration_with_data.data_flat, np.ones(11 * 6 * 3))

    assert_equal(viscosity.n_dimensions, 2)
    assert_equal(viscosity.n_cells, 11 * 6)

    viscosity.data_flat = np.ones(11 * 6)
    assert_array_equal(viscosity.data, np.ones((11, 6)))

    with pytest.raises(ValueError):
        viscosity.data_flat = np.ones(42)

    with pytest.raises(ValueError):
        viscosity.data = np.ones((1, 2, 3))


# %% Plotting

def test_plotting():
    """Test Field plotting results."""
    # 1D Plot
    dimensions = {"x": np.linspace(0, 10, 11)}
    field_1D = Field(
        name="1D Field",
        dimensions=dimensions,
        n_components=2,
        data=np.random.random((11, 2)),
    )
    fig, ax = field_1D.plot()
    assert isinstance(ax, plt.Axes), "1D plot returns one axis"

    # 2D Plot
    dimensions = {"x": np.linspace(0, 10, 11), "y": np.linspace(0, 5, 6)}
    field_2D = Field(
        name="2D Field",
        dimensions=dimensions,
        n_components=3,
        data=np.random.random((11, 6, 3)),
    )
    fig, axes = field_2D.plot()
    assert len(axes) == 3, "2D plot returns one axis per component"

    # plot 3D field
    dimensions = {
        "x": np.linspace(0, 10, 11),
        "y": np.linspace(0, 5, 6),
        "z": np.linspace(0, 2, 3),
    }
    field_3D = Field(
        name="3D Field",
        dimensions=dimensions,
        n_components=4,
        data=np.random.random((11, 6, 3, 4)),
    )
    with pytest.raises(ValueError):
        field_3D.plot()

    fig, axis = field_3D.plot(fixed_dims={"x": 1, "y": 1})
    assert isinstance(axis, plt.Axes), "3D plot with two fixed dimensions has one axis"


# %% Slicing

def test_field_slicing():
    """Test Field slicing."""
    dimensions = {"axial": np.linspace(0, 10, 11), "radial": np.linspace(0, 5, 6)}
    field = Field(name="concentration", dimensions=dimensions, n_components=3)

    # Slice along one dimension
    field_sliced = field[{"axial": 0}]
    assert_equal(
        len(field_sliced.dimensions), 1, "Field slicing reduces dimensionality"
    )
    assert_shape(field_sliced.shape, (6, 3), "Field slicing shape")

    # Slice along all dimensions
    field_sliced_all = field[{"axial": 0, "radial": 0}]
    assert_equal(
        len(field_sliced_all.dimensions), 0, "Full slicing removes all dimensions"
    )
    assert_shape(field_sliced_all.shape, (3,), "Full slicing results in vector")


# %% Normalization

def test_field_normalization():
    """Test Field normalization."""
    # Define dimensions and data
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 5, 6)
    z = np.outer(np.sin(x), np.cos(y))  # Example data

    # Create a field
    field = Field(name="temperature", dimensions={"x": x, "y": y}, data=z)

    # Normalize the field
    normalized_field = field.normalize()

    # Test 1: Check dimensions and structure
    assert field.shape == normalized_field.shape, "Normalized field shape mismatch."
    np.testing.assert_equal(field.dimensions, normalized_field.dimensions)

    # Test 2: Verify data normalization
    normalized_data = normalized_field.data
    assert np.isclose(np.min(normalized_data), 0.0), "Normalized data minimum is not 0."
    assert np.isclose(np.max(normalized_data), 1.0), "Normalized data maximum is not 1."

    # Test 3: Ensure original field is unchanged
    assert np.array_equal(field.data, z), "Original field data was modified."


# %% Interpolation and Resampling

def test_temperature_use_case():
    """Test examplary use case of interpolating temperature from a Field."""
    temperature_profile_dimensions = {
        "time": np.linspace(0, 3600, 3601),
        "axial": np.linspace(0, 0.5, 6),
    }

    # Idea, only change over time, start with T = 20C, end with 30C
    temperature_data = 20 * np.ones((3601, 6))

    temperature_field = Field(
        name="temperature",
        dimensions=temperature_profile_dimensions,
        data=temperature_data,
    )

    temperature_interpolator = FieldInterpolator(temperature_field)

    def calculate_temperature_at_t_x(t, x):
        return temperature_interpolator(time=t, axial=x)

    def calculate_adsorption_from_temperature(k_0, k_1, T):
        return k_0 * np.exp(k_1 / T)


def test_interpolated_field():
    """Test interpolation of a field."""
    dimensions = {"axial": np.linspace(0, 10, 11), "radial": np.linspace(0, 5, 6)}
    data = np.random.random((11, 6, 3))
    concentration = Field(
        name="concentration", dimensions=dimensions, n_components=3, data=data
    )

    # Interpolated field
    interp_field = FieldInterpolator(concentration)
    result = interp_field(axial=1.5, radial=2.1)
    assert_shape(result.shape, (3,), "FieldInterpolator output components")


def test_resampling():
    """Test field resampling."""
    dimensions = {"x": np.linspace(0, 10, 11), "y": np.linspace(0, 5, 6)}
    field = Field(
        name="concentration",
        dimensions=dimensions,
        n_components=2,
        data=np.random.random((11, 6, 2)),
    )

    # Resample one dimension
    resampled_field = field.resample({"x": 50})
    assert_shape(resampled_field.shape, (50, 6, 2), "Resampling one dimension")

    # Resample all dimensions
    resampled_field_all = field.resample({"x": 50, "y": 25})
    assert_shape(resampled_field_all.shape, (50, 25, 2), "Resampling all dimensions")


def test_field_interpolation_and_derivatives():
    """Test field interpolation and derivatives."""
    # Define test dimensions and data
    x = np.linspace(0, 10, 11)  # 11 points along x
    y = np.linspace(0, 5, 6)  # 6 points along y
    z = np.outer(np.sin(x), np.cos(y))  # Generate 2D scalar field data

    # Create a Field
    field = Field(name="test_field", dimensions={"x": x, "y": y}, data=z)

    # Wrap with FieldInterpolator
    interp_field = FieldInterpolator(field)

    # Test 1: Evaluate at an arbitrary point
    eval_result = interp_field(x=2.5)
    assert_shape(
        eval_result.shape,
        (6,),
        "Evaluation result should be a scalar for scalar fields.",
    )
    eval_result = interp_field(x=2.5, y=1.1)
    assert_shape(
        eval_result.shape, (), "Evaluation result should be a scalar for scalar fields."
    )

    # Test 2: Compute derivative along x
    dx_field = interp_field.derivative("x")
    assert dx_field.data.shape == field.data.shape, "Derivative shape mismatch!"

    # Test 3: Compute anti-derivative along y
    int_y_field = interp_field.anti_derivative("y", initial_value=0)
    assert int_y_field.data.shape == field.data.shape, "Anti-derivative shape mismatch!"

    # Test 4: Verify dimensions
    assert dx_field.data.shape == field.data.shape, "Derivative shape mismatch!"
    assert int_y_field.data.shape == field.data.shape, "Anti-derivative shape mismatch!"

    # Test 5: Edge Case - Evaluate at boundary
    boundary_value = interp_field(x=0, y=5)
    assert_shape(
        boundary_value.shape,
        (),
        "Boundary evaluation should return a scalar for scalar fields.",
    )


# %% Run tests

if __name__ == "__main__":
    pytest.main([__file__])
