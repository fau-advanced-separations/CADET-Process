import matplotlib.pyplot as plt
import numpy as np
import pytest
from CADETProcess.fields import Field


# %% Fixtures
@pytest.fixture
def coords():
    """Provide standard 3D coordinate set."""
    return {
        "time": np.linspace(0, 100, 101),
        "axial": np.linspace(0, 10, 11),
        "radial": np.linspace(0, 1, 5),
    }


@pytest.fixture
def components():
    """Provide standard component names."""
    return ["A", "B"]


# %% Initialization
@pytest.mark.parametrize(
    "expected_dims,components",
    [
        (("time", "axial", "radial"), None),
        (("time", "axial", "radial", "component"), ["A", "B"]),
    ],
)
def test_field_shapes(coords, components, expected_dims):
    """Test that field dimensions are correct for scalar and vector fields."""
    f = Field(coords, components=components)
    assert f.data.dims == expected_dims


# %% Interpolation
def test_interp_removes_selected_coords(coords):
    """Test that interpolation removes selected coordinates."""
    f = Field(coords)
    f2 = f.interp(time=42.3, axial=5.1)
    assert "time" not in f2.data.dims
    assert "axial" not in f2.data.dims
    assert "radial" in f2.data.dims


def test_vector_field_interp(coords, components):
    """Test that interpolation preserves component dimension for vector fields."""
    f = Field(coords, components=components)
    f2 = f.interp(time=42.3, axial=5.1)
    assert "component" in f2.data.dims
    assert f2.data.shape[-1] == len(components)


# %% Selection
def test_sel_component(coords):
    """Test selection and indexing of components."""
    f = Field({"time": np.linspace(0, 10, 11)}, components=["A", "B", "C"])

    fa = f.sel(component="A")
    assert "component" not in fa.data.dims

    fab = f.sel(component=["A", "B"])
    assert "component" in fab.data.dims

    fabt = f.sel(component="A", time=0.1234, method="nearest")
    assert "component" not in fabt.data.dims
    assert "time" not in fabt.data.dims

    fb = f.isel(component=1)
    assert "component" not in fb.data.dims


# %% Plotting
def test_plot_1d_scalar(coords):
    """Test plotting a 1D scalar field."""
    f = Field({"time": coords["time"]}, name="Scalar 1D", unit="mM")
    obj = f.plot()
    assert isinstance(obj, plt.Axes)


def test_plot_1d_vector(coords, components):
    """Test plotting a 1D vector field."""
    f = Field({"time": coords["time"]}, components=components, name="Vector 1D")
    obj = f.plot()
    assert isinstance(obj, plt.Axes)


def test_plot_2d_scalar(coords):
    """Test plotting a 2D scalar field."""
    f = Field({"time": coords["time"], "axial": coords["axial"]}, name="Scalar 2D")
    obj = f.plot()
    assert isinstance(obj, plt.Axes)


def test_plot_2d_vector(coords, components):
    """Test plotting a 2D vector field."""
    f = Field({"time": coords["time"], "axial": coords["axial"]}, components=components)
    obj = f.plot()
    assert isinstance(obj, np.ndarray)


def test_plot_raises_for_too_many_dims(coords, components):
    """Test that plotting raises an error for fields with too many dimensions."""
    f = Field(coords, components=components)
    with pytest.raises(ValueError):
        f.plot()


# %% Run tests

if __name__ == "__main__":
    pytest.main([__file__])
