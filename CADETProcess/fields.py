from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr


class Field:
    """Represent internal state or solution of a unit operation."""

    def __init__(
        self,
        coordinates: Optional[dict[str, npt.ArrayLike | tuple[npt.ArrayLike, str]]] = None,
        data: Optional[npt.ArrayLike] = None,
        components: Optional[list[str]] = None,
        name: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> None:
        """
        Represent a scalar or vector field as a DataArray.

        Parameters
        ----------
        coordinates : Optional[dict[str, npt.ArrayLike | tuple[npt.ArrayLike, str]]]
            Coordinate labels and values, optionally with units as (values, unit).
        data : Optional[npt.ArrayLike]
            Values with shape `grid_shape` (and `len(components)` if vector field).
            If None, filled with random numbers (for dev).
        components : Optional[list[str]]
            Names of components (for vector fields). If None, treated as scalar field.
        name : Optional[str]
            Name of the field (e.g., "temperature").
        unit : Optional[str]
            Unit of the field (data).
        """
        # Unpack coordinates and coord_units from tuples
        coords = {}
        coord_units = {}
        if coordinates is None:
            coordinates = {}
        for dim, coord in coordinates.items():
            if isinstance(coord, tuple):
                coords[dim] = np.asarray(coord[0])
                coord_units[dim] = coord[1]
            else:
                coords[dim] = np.asarray(coord)
        self.coordinates = coords

        # Build shape and dims
        dims = list(coords.keys())
        if components is not None:
            dims.append("component")

        # Handle 0D case
        if not dims:
            # Scalar field: data is 0D, dims is empty
            grid_shape = ()
        else:
            # nD field: data is nD, dims is non-empty
            grid_shape = tuple(len(v) for v in coords.values())

        if components is not None:
            grid_shape += (len(components),)
        if data is None:
            data = np.random.rand(*grid_shape)  # only for development
        data = np.asarray(data)

        if data.shape != grid_shape:
            raise ValueError(
                f"Data shape {data.shape} does not match expected shape {grid_shape}"
            )

        # Build DataArray coordinates
        da_coords = coords.copy()
        dims = list(coords.keys())
        if components is not None:
            da_coords["component"] = components
            dims.append("component")

        # Store DataArray
        self._data = xr.DataArray(data, coords=da_coords, dims=dims, name=name)

        # Set unit for data
        if unit is not None:
            self._data.attrs["units"] = unit

        # Set units for coordinates
        for dim, dim_unit in coord_units.items():
            self._data[dim].attrs["units"] = dim_unit

    @property
    def name(self) -> str:
        """Return the name of the Field."""
        return self._data.name

    @property
    def data(self) -> xr.DataArray:
        """Return the underlying xarray DataArray."""
        return self._data

    @property
    def components(self) -> Optional[list[str]]:
        """Return component names if present, else None."""
        return (
            self._data.coords["component"].values.tolist()
            if "component" in self._data.dims
            else None
        )

    def wraps_xr(method_name: str) -> Callable:
        """Wrap xr.DataArray method."""
        def xr_decorator(func: Callable) -> Callable:
            @wraps(func)
            def xr_wrapper(
                self: "Field",
                *args: Any,
                **kwargs: Any,
            ) -> "Field":
                """Wrap xr.DataArray method."""
                da = getattr(self._data, method_name)(*args, **kwargs)
                coords = {k: da.coords[k].values for k in da.dims if k != "component"}
                comps = (
                    da.coords["component"].values.tolist()
                    if "component" in da.dims
                    else None
                )
                return Field(coords, data=da.values, components=comps)
            return xr_wrapper
        return xr_decorator

    @wraps_xr("sel")
    def sel(self, **kwargs: Any) -> "Field":
        """Slice the field using xarray's sel."""
        pass

    @wraps_xr("isel")
    def isel(self, drop: bool = False, **kwargs: Any) -> "Field":
        """Slice the field using xarray's isel (by integer index)."""
        pass

    @wraps_xr("interp")
    def interp(self, **kwargs: Any) -> "Field":
        """Interpolate the field along given coordinates."""
        pass

    @wraps_xr("differentiate")
    def differentiate(self, **kwargs: Any) -> "Field":
        """Differentiate the array with second order accurate central differences."""
        pass

    @wraps_xr("integrate")
    def integrate(self, **kwargs: Any) -> "Field":
        """Integrate the array along the given coordinate using the trapezoidal rule."""
        pass

    def plot(
        self,
        sel: Optional[dict[str, float | list[str]]] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> plt.Axes | np.ndarray[plt.Axes]:
        """
        Plot the field using xarray's plotting backend.

        Parameters
        ----------
        sel : dict[str, float or list[str]], optional
            Coordinates (including "component") at which to slice before plotting.
        ax : matplotlib.axes.Axes, optional
            Axis to plot into (1D or 2D scalar only). If None, create new.
            Note, this does currently not work for faceted plots.
        **kwargs: Any
            Passed to xarray plot method.

        Returns
        -------
        plt.Axes | np.ndarray[plt.Axes]
            The axes object(s) that were used for plotting
        """
        da = self._data
        if sel is not None:
            da = da.sel(sel, method="nearest")

        plot_dims = [d for d in da.dims if d != "component"]
        ndim = len(plot_dims)

        if ndim > 2:
            raise ValueError(
                f"Field has {ndim} dims after slicing; plot only supports 1D or 2D."
            )

        # Create new figure for 1D or scalar 2D
        if ax is None and (ndim == 1 or (ndim == 2 and "component" not in da.dims)):
            fig, ax = plt.subplots()

        # --- 1D case ---
        if ndim == 1:
            if "component" in da.dims:
                out = da.plot.line(
                    ax=ax,
                    hue="component",
                    **kwargs
                )
            else:
                out = da.plot.line(
                    ax=ax,
                    **kwargs
                )
            return ax

        # --- 2D case ---
        if ndim == 2:
            if "component" in da.dims:
                out = da.plot(
                    col="component",
                    **kwargs
                )
                return out.axs
            else:
                out = da.plot(
                    ax=ax,
                    **kwargs
                )
                return ax

    def __getitem__(self, key: str) -> "Field":
        """Extract a single component as a new Field."""
        if "component" not in self._data.dims:
            raise ValueError("No components defined.")

        da = self._data.sel(component=key, drop=True)
        coords = {k: da.coords[k].values for k in da.dims if k != "component"}

        return Field(coords, data=da.values)

    def __repr__(self) -> str:
        """str: String representation of the Field."""
        return f"Field({repr(self._data)})"
