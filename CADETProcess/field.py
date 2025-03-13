from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


class Field:
    """
    Represents a single field in a state block.

    Parameters
    ----------
    name : str
        Name of the field (e.g., "concentration").
    dimensions : dict[str, npt.ArrayLike]
        Mapping of dimension names to their coordinate arrays.
    n_components : Optional[int], optional
        Number of components in the field (e.g., species in a concentration field).
        If None, the field is treated as a scalar field.
    data : Optional[npt.ArrayLike], optional
        Initial data for the field. If None, data is initialized as zeros.

    """

    def __init__(
            self,
            name: str,
            dimensions: dict[str, npt.ArrayLike],
            n_components: Optional[int] = None,
            data: Optional[npt.ArrayLike] = None,
            ):
        """Construct the object."""
        self.name = name
        self.dimensions = {k: np.array(v) for k, v in dimensions.items()}
        self.n_components = n_components

        # Initialize the data array
        if data is None:
            data = np.zeros(self.shape)
        self.data = data

    @property
    def is_scalar_field(self) -> bool:
        """bool: True if field does not have components, False otherwise."""
        return self.n_components is None

    @property
    def data(self) -> np.ndarray:
        """np.ndarray: The data array for the field."""
        return self._data

    @data.setter
    def data(self, value: npt.ArrayLike):
        """Set the data array after validating its shape."""
        value = np.array(value)
        if value.shape != self.shape:
            raise ValueError(
                f"Assigned data shape {value.shape} does not match the expected shape "
                f"{self.shape}."
            )
        self._data = value

    @property
    def n_dimensions(self) -> int:
        """int: Return the number of dimensions."""
        return len(self.dimensions)

    @property
    def n_cells(self) -> int:
        """int: Return the total number of cells from the product of dimensions."""
        return int(np.prod(self.dimension_shape))

    @property
    def dimension_shape(self) -> tuple[int, ...]:
        """tuple[int]: Return the shape derived from dimensions."""
        return tuple(len(v) for v in self.dimensions.values())

    @property
    def shape(self) -> tuple[int, ...]:
        """tuple[int]: Return the complete shape of the data array."""
        shape = self.dimension_shape
        if not self.is_scalar_field:
            shape += (self.n_components,)
        return shape

    @property
    def n_dof(self) -> int:
        """int: Return the total number of degrees of freedom."""
        return np.prod(self.shape)

    @property
    def data_flat(self) -> np.ndarray:
        """np.ndarray: Return the data array flattened into one dimension."""
        return self.data.reshape(-1)

    @data_flat.setter
    def data_flat(self, data_flat: np.ndarray) -> None:
        """Set the data array from a flattened array."""
        if data_flat.size != self.n_dof:
            raise ValueError(
                f"Flattened data size {data_flat.size} does not match the "
                f"field's degrees of freedom {self.n_dof}."
            )
        self.data = data_flat.reshape(self.shape)

    def plot(
            self,
            title: str = None,
            fixed_dims: Optional[dict[str, float | int]] = None,
            method: str = "surface",
            fig: Optional[plt.Figure] = None,
            axes: Optional[plt.Axes | list[plt.Axes]] = None,
            ) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
        """
        Plot the field data, supporting 1D, 2D, and slicing for higher dimensions.

        Parameters
        ----------
        title : str, optional
            Title for the plot.
        fixed_dims : Optional[dict[str, float | int]]
            Fixed dimensions to slice for higher-dimensional fields.
            Only required for fields with more than 2 dimensions.
        method : str, optional
            Plotting method: "surface" or "heatmap". Default is "surface".
        fig : matplotlib.figure.Figure, optional
            Predefined figure to draw on. If not provided, a new figure will be created.
        axes : Optional[plt.Axes | list[plt.Axes]], optional
            Predefined axes to draw on. If not provided, new axes will be created.

        Returns
        -------
        tuple[plt.Figure, plt.Axes | list[plt.Axes]]
            The figure and list of axes used for the plot.

        Raises
        ------
        ValueError
            If the field has more than 2 dimensions and `fixed_dims` is not provided.
            If an unknown plotting method is specified.

        """
        n_dims = len(self.dimensions)

        # Handle higher-dimensional fields
        if n_dims > 2:
            if fixed_dims is None:
                raise ValueError(
                    f"Field has {n_dims} dimensions. "
                    "Please provide fixed_dims to reduce to 1D or 2D for plotting."
                )
            # Slice along specified dimensions
            sliced_field = self[fixed_dims]
            # Recursively call plot on the reduced field
            return sliced_field.plot(title=title, method=method, fig=fig, axes=axes)

        # 1D Plot
        if n_dims == 1:
            x = list(self.dimensions.values())[0]
            y = self.data
            component_names = (
                [f"Component {i}" for i in range(self.n_components)]
                if self.n_components else ["Value"]
            )
            return plot_1D(x, y, component_names, title or self.name, fig, axes)

        # 2D Plot
        if n_dims == 2:
            x, y = self.dimensions.values()
            z = self.data
            component_names = (
                [f"Component {i}" for i in range(self.n_components)]
                if self.n_components else ["Value"]
            )
            if method == "surface":
                return plot_2D_surface(
                    x, y, z, component_names, title or self.name, fig, axes
                )
            elif method == "heatmap":
                return plot_2D_heatmap(
                    x, y, z, component_names, title or self.name, fig, axes
                )
            else:
                raise ValueError(f"Unknown plotting method: {method}")

    def resample(self, resample_dims: dict[str, int | npt.ArrayLike]) -> "Field":
        """
        Resample the field data to new dimensions using interpolation.

        Parameters
        ----------
        resample_dims : dict[str, int | npt.ArrayLike]
            Dictionary specifying the resampling for each dimension.
            - If a dimension key is missing, it remains unchanged.
            - If a value is an integer, it generates a linspace with that many points.
            - If a value is an array, it is used directly.

        Returns
        -------
        Field
            A new Field object with the interpolated data and updated dimensions.

        """
        # Validate resample_dims keys
        invalid_keys = set(resample_dims.keys()) - set(self.dimensions.keys())
        if invalid_keys:
            raise ValueError(
                f"Invalid dimension keys in resample_dims: {invalid_keys}. "
                f"Valid dimensions are {list(self.dimensions.keys())}."
            )

        # Generate new dimensions
        new_dimensions = {}
        for dim, original_coords in self.dimensions.items():
            if dim in resample_dims:
                value = resample_dims[dim]
                if isinstance(value, int):
                    new_dimensions[dim] = np.linspace(
                        original_coords.min(), original_coords.max(), value
                    )
                elif isinstance(value, np.ndarray):
                    new_dimensions[dim] = value
                else:
                    raise TypeError(
                        f"Dimension '{dim}' must be an integer or an array."
                    )
            else:
                # Keep the original coordinates if not specified in resample_dims
                new_dimensions[dim] = original_coords

        # Use FieldInterpolator for interpolation
        interpolated_field = FieldInterpolator(self)
        interpolated_data = interpolated_field(**new_dimensions)

        # Return a new Field
        return Field(
            name=f"{self.name}_resampled",
            dimensions=new_dimensions,
            n_components=self.n_components,
            data=interpolated_data,
        )

    def normalize(self) -> "NormalizedField":
        """
        Normalize the field data.

        Returns
        -------
        NormalizedField
            A new NormalizedField instance with normalized data.

        Notes
        -----
        The normalization is performed using the formula:
            normalized_data = (data - min(data)) / (max(data) - min(data))

        """
        return NormalizedField(self)

    def derivative(self, dimension: str) -> "Field":
        """
        Compute the derivative of the field along a specified dimension.

        Parameters
        ----------
        dimension : str
            The dimension along which to compute the derivative.

        Returns
        -------
        Field
            A new Field representing the derivative.

        """
        if dimension not in self.dimensions:
            raise ValueError(f"Dimension '{dimension}' not found in field.")

        # Use FieldInterpolator for derivative computation
        interpolated_field = FieldInterpolator(self)
        return interpolated_field.derivative(dimension)

    def anti_derivative(self, dimension: str, initial_value: float = 0) -> "Field":
        """
        Compute the anti-derivative of the field along a specified dimension.

        Parameters
        ----------
        dimension : str
            The dimension along which to compute the anti-derivative.
        initial_value : float, optional
            Initial value for the anti-derivative. Default is 0.

        Returns
        -------
        Field
            A new Field representing the anti-derivative.

        """
        if dimension not in self.dimensions:
            raise ValueError(f"Dimension '{dimension}' not found in field.")

        # Use FieldInterpolator for anti-derivative computation
        interpolated_field = FieldInterpolator(self)
        return interpolated_field.anti_derivative(dimension, initial_value)

    def __getitem__(self, slices: dict[str, float | int]) -> Union["Field", np.ndarray]:
        """
        Slice the field along specified dimensions.

        Parameters
        ----------
        slices : dict[str, float | int]
            A dictionary specifying the dimension(s) to slice.

        Returns
        -------
        "Field" | np.ndarray
            A new Field object with reduced dimensions if some dimensions are
            unspecified.
            A scalar or vector (np.ndarray) if all dimensions are specified.

        """
        # Ensure valid dimensions are being sliced
        for dim in slices.keys():
            if dim not in self.dimensions:
                raise KeyError(f"Invalid dimension '{dim}'.")

        # Generate slicing indices
        slice_indices = []
        reduced_dimensions = {}
        for dim, coords in self.dimensions.items():
            if dim in slices:
                # Find the closest index for the specified coordinate
                coord = slices[dim]
                index = np.searchsorted(coords, coord)
                if not (0 <= index < len(coords)):
                    raise ValueError(
                        f"Coordinate {coord} out of bounds for dimension '{dim}'."
                    )
                slice_indices.append(index)
            else:
                # Keep the dimension if not being sliced
                slice_indices.append(slice(None))
                reduced_dimensions[dim] = coords

        # Slice the data
        sliced_data = self.data[tuple(slice_indices)]

        # Always return a Field, even when all dimensions are sliced
        return Field(
            name=self.name,
            dimensions=reduced_dimensions,
            n_components=self.n_components,
            data=sliced_data,
        )

    def __repr__(self) -> str:
        """Represantator."""
        components = (
            f", components={self.n_components}"
            if self.n_components
            else ", scalar=True"
        )
        return (
            f"Field(name='{self.name}'"
            + f", dimensions={list(self.dimensions.keys())}{components})"
        )


class NormalizedField(Field):
    """
    Represents a normalized version of a Field.

    Parameters
    ----------
    field : Field
        The original field instance to normalize.

    """

    def __init__(self, field: Field):
        """Initialize the base Field class with the same parameters."""
        super().__init__(
            name=f"Normalized_{field.name}",
            dimensions=field.dimensions,
            n_components=field.n_components,
            data=field.data,
        )
        self.field = field  # Keep reference to the original field
        self._compute_normalization()

    def _compute_normalization(self):
        """Compute the min and max values for normalization."""
        data_min = np.min(self.field.data)
        data_max = np.max(self.field.data)

        if data_max == data_min:
            raise ValueError("Cannot normalize a field with zero data range.")

        self.data = (self.field.data - data_min) / (data_max - data_min)


class FieldInterpolator:
    """
    Wrapper around a Field that provides interpolation functionality.

    Parameters
    ----------
    field : Field
        The original field instance to wrap.

    """

    def __init__(self, field):
        """Construct the object."""
        self.field = field
        self.data = np.array(field.data)  # Immutable snapshot of field data
        self._initialize_interpolators()

    def _initialize_interpolators(self):
        """Initialize interpolators based on the wrapped field's data."""
        grid_points = list(self.field.dimensions.values())
        if self.field.n_components:
            self._interpolators = [
                RegularGridInterpolator(
                    grid_points, self.field.data[..., i], method="pchip"
                )
                for i in range(self.field.n_components)
            ]
        else:
            self._interpolator = RegularGridInterpolator(grid_points, self.field.data)

    def __call__(self, **kwargs: npt.ArrayLike) -> np.ndarray:
        """
        Evaluate the interpolated field at given coordinates.

        Parameters
        ----------
        **kwargs : npt.ArrayLike
            Coordinate arrays for each dimension. If a dimension is not specified,
            its full original structure will be retained.

        Returns
        -------
        np.ndarray
            Interpolated values at the specified coordinates.
            If some dimensions are unspecified, their original structure is preserved.

        """
        # Separate provided and missing dimensions
        provided_dims = set(kwargs.keys())
        all_dims = set(self.field.dimensions.keys())
        # missing_dims = all_dims - provided_dims

        # Validate dimensions
        if not provided_dims.issubset(all_dims):
            invalid_dims = provided_dims - all_dims
            raise ValueError(f"Invalid dimensions specified: {invalid_dims}")

        # Create query points, filling in original coordinates for missing dimensions
        query_coords = [
            kwargs.get(dim, self.field.dimensions[dim])
            for dim in self.field.dimensions.keys()
        ]
        mesh = np.meshgrid(*query_coords, indexing="ij")
        query_points = np.stack([grid.ravel() for grid in mesh], axis=-1)

        # Perform interpolation
        if self.field.n_components:
            interpolated_components = [
                interp(query_points).reshape(mesh[0].shape)
                for interp in self._interpolators
            ]
            result = np.stack(interpolated_components, axis=-1)
        else:
            result = self._interpolator(query_points).reshape(mesh[0].shape)

        # Reshape the result based on the query to retain missing dimensions
        output_shape = self._determine_output_shape(kwargs)
        return result.reshape(output_shape)

    def _determine_output_shape(
            self,
            query: dict[str, Union[float, npt.ArrayLike]]
            ) -> tuple[int, ...]:
        """
        Determine the shape of the interpolated result based on the query.

        Parameters
        ----------
        query : dict[str, Union[float, npt.ArrayLike]]
            The coordinates provided for each dimension in the query.

        Returns
        -------
        tuple[int, ...]
            The shape of the interpolated result.

        """
        output_shape = []
        for dim, original_coords in self.field.dimensions.items():
            if dim in query:
                queried_value = query[dim]
                if np.isscalar(queried_value):
                    continue  # Scalars do not contribute to the shape
                else:
                    output_shape.append(len(queried_value))  # Retain array length
            else:
                output_shape.append(len(original_coords))
                # Retain full original dimension

        # Add n_components as the last dimension for vector fields
        if self.field.n_components:
            output_shape.append(self.field.n_components)

        return tuple(output_shape)

    def derivative(self, dimension: str):
        """
        Compute the derivative of the field along the specified dimension.

        Parameters
        ----------
        dimension : str
            Dimension along which to compute the derivative.

        Returns
        -------
        Field
            A new field representing the derivative.

        """
        if dimension not in self.field.dimensions:
            raise ValueError(f"Dimension '{dimension}' not found in field.")

        axis = list(self.field.dimensions.keys()).index(dimension)
        coords = self.field.dimensions[dimension]

        # Compute derivative for multi-component fields
        if self.field.n_components:
            derivative_data = np.stack(
                [
                    np.gradient(self.field.data[..., i], coords, axis=axis)
                    for i in range(self.field.n_components)
                ],
                axis=-1,
            )
        else:
            derivative_data = np.gradient(self.field.data, coords, axis=axis)

        # Return new Field for the derivative
        return Field(
            name=f"{self.field.name}_derivative_{dimension}",
            dimensions=self.field.dimensions,
            n_components=self.field.n_components,
            data=derivative_data,
        )

    def anti_derivative(self, dimension: str, initial_value: float = 0):
        """
        Compute the anti-derivative of the field along the specified dimension.

        Parameters
        ----------
        dimension : str
            Dimension along which to compute the anti-derivative.
        initial_value : float, optional
            Initial value for the anti-derivative. Default is 0.

        Returns
        -------
        Field
            A new field representing the anti-derivative.

        """
        if dimension not in self.field.dimensions:
            raise ValueError(f"Dimension '{dimension}' not found in field.")

        axis = list(self.field.dimensions.keys()).index(dimension)
        coords = self.field.dimensions[dimension]

        # Compute anti-derivative for multi-component fields
        if self.field.n_components:
            anti_derivative_data = np.stack(
                [
                    np.cumsum(self.field.data[..., i] * np.gradient(coords), axis=axis)
                    + initial_value
                    for i in range(self.field.n_components)
                ],
                axis=-1,
            )
        else:
            anti_derivative_data = (
                np.cumsum(self.field.data * np.gradient(coords), axis=axis)
                + initial_value
            )

        # Return new Field for the anti-derivative
        return Field(
            name=f"{self.field.name}_anti_derivative_{dimension}",
            dimensions=self.field.dimensions,
            n_components=self.field.n_components,
            data=anti_derivative_data,
        )

    def integral(self, integration_limits: dict[str, tuple[float, float]] = None) -> Field:
        """

        Compute the definite integral of the field over specified dimensions.

        Parameters
        ----------
        integration_limits : dict[str, tuple[float, float]], optional
            A dictionary specifying the integration bounds for each dimension.
            If a dimension is not specified, the entire range is used.
            Example: {"x": (0, 5), "y": (-1, 1)}

        Returns
        -------
        Field
            A new Field object representing the integral over the specified dimensions.

        Raises
        ------
        ValueError
            If any specified integration dimension is invalid.

        Notes
        -----
        - The integration is performed using the trapezoidal rule (`numpy.trapz`).
        - If all dimensions are integrated out, the result is a scalar field.
        """

        if integration_limits is None:
            integration_limits = {}

        # Validate the specified dimensions
        invalid_dims = set(integration_limits.keys()) - set(self.field.dimensions.keys())
        if invalid_dims:
            raise ValueError(f"Invalid dimensions specified: {invalid_dims}")

        data = self.field.data
        remaining_dimensions = self.field.dimensions.copy()

        # Perform definite integration over each specified dimension
        for dim, coords in self.field.dimensions.items():
            if dim in integration_limits:
                start, end = integration_limits[dim]
                mask = (coords >= start) & (coords <= end)
                coords = coords[mask]  # Restrict coordinate range
                data = data[mask, ...]  # Restrict data along this dimension

            axis = list(self.field.dimensions.keys()).index(dim)
            dx = np.gradient(coords)  # Compute spacing
            data = np.trapz(data, x=coords, axis=axis)
            del remaining_dimensions[dim]  # Remove integrated dimension

        # If all dimensions are integrated, return a scalar field
        if not remaining_dimensions:
            return Field(name=f"{self.field.name}_integral", dimensions={}, data=np.array(data))

        return Field(
            name=f"{self.field.name}_integral",
            dimensions=remaining_dimensions,
            n_components=self.field.n_components,
            data=data,
        )

    def __repr__(self) -> str:
        """Return represantation."""
        return f"FieldInterpolator({self.field})"


def plot_1D(
        x: np.ndarray,
        y: np.ndarray,
        component_names: list[str],
        title: str,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 1D field with a line for each component.

    Parameters
    ----------
    x : np.ndarray
        The x-axis data (1D coordinate array).
    y : np.ndarray
        The y-axis data, with shape (len(x), n_components).
    component_names : list[str]
        Names of the components to label in the legend.
    title : str
        Title of the plot.
    fig : matplotlib.figure.Figure, optional
        Predefined figure to draw on. If not provided, a new figure will be created.
    ax : matplotlib.axes.Axes, optional
        Predefined axis to draw on. If not provided, a new axis will be created.

    Returns
    -------
    tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]
        The figure and list containing the single axis used for the plot.

    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    for i, component in enumerate(component_names):
        ax.plot(x, y[:, i], label=component)

    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Values")
    ax.legend()

    return fig, ax


def plot_2D_surface(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        component_names: list[str],
        title: str,
        fig: Optional[plt.Figure] = None,
        axes: Optional[list[plt.Axes]] = None,
        ) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot a 2D field with a surface for each component.

    Parameters
    ----------
    x : np.ndarray
        The x-axis data (1D coordinate array).
    y : np.ndarray
        The y-axis data (1D coordinate array).
    z : np.ndarray
        The z-axis data, with shape (len(x), len(y), n_components).
    component_names : list[str]
        Names of the components to label the subplots.
    title : str
        Title of the plot.
    fig : matplotlib.figure.Figure, optional
        Predefined figure to draw on. If not provided, a new figure will be created.
    axes : list[matplotlib.axes.Axes], optional
        Predefined axes to draw on. If not provided, new axes will be created.

    Returns
    -------
    tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]
        The figure and list of axes used for the plot.

    """
    n_components = z.shape[-1]

    if axes is None:
        fig, axes = plt.subplots(
            1,
            n_components,
            figsize=(5 * n_components, 5),
            subplot_kw={"projection": "3d"},
        )
        axes = [axes] if n_components == 1 else axes

    X, Y = np.meshgrid(x, y, indexing="ij")
    for i, ax in enumerate(axes):
        ax.plot_surface(X, Y, z[..., i], cmap="viridis")
        ax.set_title(f"{title} - {component_names[i]}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Values")

    return fig, axes


def plot_2D_heatmap(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        component_names: list[str],
        title: str,
        fig: Optional[plt.Figure] = None,
        axes: Optional[list[plt.Axes]] = None,
        ) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot a 2D field with a heatmap for each component.

    Parameters
    ----------
    x : np.ndarray
        The x-axis data (1D coordinate array).
    y : np.ndarray
        The y-axis data (1D coordinate array).
    z : np.ndarray
        The z-axis data, with shape (len(x), len(y), n_components).
    component_names : list[str]
        Names of the components to label the subplots.
    title : str
        Title of the plot.
    fig : matplotlib.figure.Figure, optional
        Predefined figure to draw on. If not provided, a new figure will be created.
    axes : list[matplotlib.axes.Axes], optional
        Predefined axes to draw on. If not provided, new axes will be created.

    Returns
    -------
    tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]
        The figure and list of axes used for the plot.

    """
    n_components = z.shape[-1]

    if axes is None:
        fig, axes = plt.subplots(
            1, n_components, figsize=(5 * n_components, 5), constrained_layout=True
        )
        axes = [axes] if n_components == 1 else axes

    X, Y = np.meshgrid(x, y, indexing="ij")
    for i, ax in enumerate(axes):
        c = ax.pcolormesh(X, Y, z[..., i], cmap="viridis", shading="auto")
        fig.colorbar(c, ax=ax, orientation="vertical")
        ax.set_title(f"{title} - {component_names[i]}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

    return fig, axes

# %%

dims = {
    "x": np.linspace(0, 10, 100),
    "y": np.linspace(-5, 5, 50),
    "z": np.linspace(-1, 1, 20),
}
data = np.random.rand(100, 50, 20)  # 3D field

field = Field(name="example", dimensions=dims, data=data)
interp = FieldInterpolator(field)

# Integrate over "x" from 2 to 8 and "y" from -2 to 3
integrated_field = interp.integral({"x": (2, 8), "y": (-2, 3)})
print(integrated_field)
