"""
=======================================
Solution (:mod:`CADETProcess.solution`)
=======================================

.. currentmodule:: CADETProcess.solution

Module to store and plot solution of simulation.

.. autosummary::
   :toctree: generated/

   SolutionBase
   SolutionIO
   SolutionBulk
   SolutionParticle
   SolutionSolid
   SolutionVolume


Method to slice a Solution:

.. autosummary::
   :toctree: generated/

   slice_solution
   slice_solution_front


"""  # noqa

from __future__ import annotations

import copy
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from scipy import integrate
from scipy.interpolate import PchipInterpolator

from CADETProcess import CADETProcessError, plotting, smoothing, transform
from CADETProcess.dataStructure import (
    SizedNdArray,
    String,
    Structure,
    Typed,
    UnsignedFloat,
    Vector,
)
from CADETProcess.dynamicEvents import TimeLine
from CADETProcess.processModel import ComponentSystem

__all__ = [
    "SolutionIO",
    "SolutionBulk",
    "SolutionParticle",
    "SolutionSolid",
    "SolutionVolume",
    "slice_solution",
    "slice_solution_front",
]


class SolutionBase(Structure):
    """
    Base class for solutions of component systems.

    This class represents a solution of a component system at different time points.
    It provides several properties that allow access to information about the solution,
    such as the number of components, the total concentration, and the local purity.

    Attributes
    ----------
    name : str
        Name of the solution.
    time : np.ndarray
        Array of time points.
    solution : np.ndarray
        Array of solution values.
    c_min : float
        Minimum concentration threshold, below which concentrations are considered zero.
    dimensions : list of str
        Names of the dimensions in the solution.

    Notes
    -----
    This class is not meant to be used directly, but to be subclassed by specific
    solution types.
    """

    name = String()
    time = Vector()
    component_system = Typed(ty=ComponentSystem)
    solution = SizedNdArray(size="solution_shape")
    c_min = UnsignedFloat(default=1e-6)

    dimensions = ["time"]

    def __init__(
        self,
        name: str,
        component_system: ComponentSystem,
        time: npt.ArrayLike,
        solution: npt.ArrayLike,
    ) -> None:
        """
        Initialize the SolutionBase.

        Parameters
        ----------
        name : str
            The name of the solution.
        component_system : ComponentSystem
            The component system associated with the solution.
        time : npt.ArrayLike
            An array-like structure representing time points.
        solution : npt.ArrayLike
            An array-like structure representing the solution data.
        """
        self.name = name
        self.component_system = component_system
        self.time = time
        self.solution = solution

        self.update_solution()

    def update_solution(self) -> None:
        """Update the solution."""
        pass

    def update_transform(self) -> None:
        """Update the transforms."""
        pass

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self.component_system.n_comp

    @property
    def cycle_time(self) -> float:
        """float: Cycle time."""
        return self.time[-1]

    @property
    def nt(self) -> int:
        """int: Number of time steps."""
        return len(self.time)

    @property
    def component_coordinates(self) -> np.ndarray:
        """np.ndarray: Indices of the components."""
        return np.arange(self.n_comp)

    @property
    def coordinates(self) -> dict[str, np.ndarray]:
        """dict[str, np.ndarray]: Coordinates of the Solution."""
        coordinates = {}

        for c in self.dimensions:
            v = getattr(self, c)
            if v is None:
                continue
            coordinates[c] = v
        return coordinates

    @property
    def solution_shape(self) -> tuple[int]:
        """tuple[int]: (Expected) shape of the solution."""
        return tuple(len(c) for c in self.coordinates.values())

    @property
    def total_concentration_components(self) -> np.ndarray:
        """np.ndarray: Total concentration of components (sum of species)."""
        component_concentration = np.zeros(
            self.solution.shape[0:-1] + (self.component_system.n_components,)
        )

        counter = 0
        for index, comp in enumerate(self.component_system):
            comp_indices = slice(counter, counter + comp.n_species)
            c_comp = np.sum(self.solution[..., comp_indices], axis=-1)
            component_concentration[..., index] = c_comp
            counter += comp.n_species

        return component_concentration

    @property
    def total_concentration(self) -> np.ndarray:
        """np.ndarray: Total concentration (sum) of all components."""
        return np.sum(self.solution, axis=-1, keepdims=True)

    @property
    def local_purity_components(self) -> np.ndarray:
        """np.ndarray: Local purity of each component."""
        solution = self.total_concentration_components
        c_total = self.total_concentration
        c_total[c_total < self.c_min] = np.nan

        purity = np.zeros(solution.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            purity = solution / c_total

        purity = np.nan_to_num(purity)

        return purity

    @property
    def local_purity_species(self) -> np.ndarray:
        """np.ndarray: Local purity of components."""
        solution = self.solution
        c_total = self.total_concentration
        c_total[c_total < self.c_min] = np.nan

        purity = np.zeros(solution.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            purity = solution / c_total

        purity = np.nan_to_num(purity)

        return purity

    def __str__(self) -> str:
        return self.name


class SolutionIO(SolutionBase):
    """
    Solution representing streams at the inlet or outlet of a ``UnitOperation``.

    Notes
    -----
    The `flow_rate` attribute is implemented as TimeLine to improve interpolation of
    signals with discontinuous flow.
    """

    dimensions = SolutionBase.dimensions + ["component_coordinates"]

    def __init__(
        self,
        name: str,
        component_system: ComponentSystem,
        time: npt.ArrayLike,
        solution: npt.ArrayLike,
        flow_rate: TimeLine | npt.ArrayLike,
    ) -> None:
        """
        Initialize the SolutionIO.

        Parameters
        ----------
        name : str
            The name of the solution object.
        component_system : ComponentSystem
            The component system associated with the solution.
        time : npt.ArrayLike
            An array-like structure representing time points.
        solution : npt.ArrayLike
            An array-like structure representing the solution data.
        flow_rate : TimeLine | npt.ArrayLike
            The flow rate data, which can be a TimeLine object or an array-like structure.
        """
        if not isinstance(flow_rate, TimeLine):
            flow_rate = TimeLine.from_profile(time, flow_rate)
        self.flow_rate = flow_rate

        super().__init__(name, component_system, time, solution)

    def update_solution(self) -> None:
        """Update solution method."""
        self._solution_interpolated = None
        self._dm_dt_interpolated = None
        self.update_transform()

    @property
    def derivative(self) -> "SolutionIO":
        """SolutionIO: Derivative of this solution."""
        derivative = copy.deepcopy(self)
        derivative_fun = derivative.solution_interpolated.derivative
        derivative.solution = derivative_fun(derivative.time)
        derivative.update_solution()

        return derivative

    @property
    def antiderivative(self) -> "SolutionIO":
        """SolutionIO: Antiderivative of this solution."""
        antiderivative = copy.deepcopy(self)
        antiderivative_fun = antiderivative.solution_interpolated.antiderivative
        antiderivative.solution = antiderivative_fun(antiderivative.time)
        antiderivative.update_solution()

        return antiderivative

    def update_transform(self) -> None:
        """Update the Transformer."""
        self.transform = transform.NormLinearTransformer(
            np.min(self.solution, axis=0),
            np.max(self.solution, axis=0),
            allow_extended_input=True,
            allow_extended_output=True,
        )

    @property
    def solution_interpolated(self) -> InterpolatedSignal:
        """InterpolatedSignal: The interpolated signal."""
        if self._solution_interpolated is None:
            self._solution_interpolated = InterpolatedSignal(self.time, self.solution)

        return self._solution_interpolated

    @property
    def dm_dt_interpolated(self) -> InterpolatedSignal:
        """Return the dm/dt interpolated signal."""
        if self._dm_dt_interpolated is None:
            dm_dt = self.solution * self.flow_rate.value(self.time)
            self._dm_dt_interpolated = InterpolatedSignal(self.time, dm_dt)

        return self._dm_dt_interpolated

    def normalize(self) -> SolutionIO:
        """SolutionIO: Normalize the solution using the transformation function."""
        solution = copy.deepcopy(self)

        solution.solution = self.transform.transform(self.solution)
        solution.update_solution()

        return solution

    def resample(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        nt: Optional[int] = 5001,
    ) -> "SolutionIO":
        """
        Resample solution to nt time points.

        Parameters
        ----------
        start : Optional[float]
            Start time of the fraction. If None, the first time point is used. The
            default is None.
        end: Optional[float]
            End time of the fraction. If None, the last time point is used. The
            default is None.
        nt : int, optional
            Number of points to resample. The default is 5001.

        Returns
        -------
        SolutionIO
            The resampled solution object.
        """
        solution = copy.deepcopy(self)

        if start is None:
            start = self.time[0]
        if end is None:
            end = self.time[-1]

        solution.time = np.linspace(start, end, nt)
        solution.solution = self.solution_interpolated(solution.time)
        solution.update_solution()

        return solution

    def smooth_data(
        self,
        s: Optional[float | list[float]] = None,
        crit_fs: Optional[float | list[float]] = None,
        crit_fs_der: Optional[float | list[float]] = None,
    ) -> "SolutionIO":
        """
        Smooth data.

        Parameters
        ----------
        s : Optional[float | list[float]]
            DESCRIPTION.
        crit_fs : Optional[float | list[float]]
            DESCRIPTION.
        crit_fs_der : Optional[float | list[float]]
            DESCRIPTION.

        Returns
        -------
        SolutionIO
            The smoothed solution object.
        """
        solution = copy.deepcopy(self)

        if None in (s, crit_fs, crit_fs_der):
            s_ = []
            crit_fs_ = []
            crit_fs_der_ = []

            for i in range(self.n_comp):
                s_i, crit_fs_i, crit_fs_der_i = smoothing.find_smoothing_factors(
                    self.time, self.solution[..., i], rmse_target=1e-3
                )
                s_.append(s_i)
                crit_fs_.append(crit_fs_i)
                crit_fs_der_.append(crit_fs_der_i)

            if s is None:
                s = s_
            if crit_fs is None:
                crit_fs = crit_fs_
            if crit_fs_der is None:
                crit_fs_der = crit_fs_der_

        if np.isscalar(s):
            s = self.n_comp * [s]
        elif len(s) == 1:
            s = self.n_comp * s

        if np.isscalar(crit_fs):
            crit_fs = self.n_comp * [crit_fs]
        elif len(crit_fs) == 1:
            crit_fs = self.n_comp * crit_fs

        if np.isscalar(crit_fs_der):
            crit_fs_der = self.n_comp * [crit_fs_der]
        elif len(crit_fs_der) == 1:
            crit_fs_der = self.n_comp * crit_fs_der

        solution = np.zeros((self.solution.shape))
        for i, (s, crit_fs, crit_fs_der) in enumerate(zip(s, crit_fs, crit_fs_der)):
            smooth, smooth_der = smoothing.full_smooth(
                self.time, self.solution[..., i], crit_fs, s, crit_fs_der
            )
            solution[..., i] = smooth

        solution.solution = solution
        solution.update_solution()

        return solution

    def integral(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> np.ndarray:
        """
        Peak area in a fraction interval.

        Parameters
        ----------
        start : Optional[float]
            Start time of the fraction. If None, the first time point is used. The
            default is None.
        end: Optional[float]
            End time of the fraction. If None, the last time point is used. The
            default is None.

        Returns
        -------
        np.ndarray
            Area of each component in the fraction.
        """
        if start is None:
            start = self.time[0]

        if end is None:
            end = self.cycle_time

        return self.solution_interpolated.integral(start, end)

    def create_fraction(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> Fraction:  # noqa:F821
        """
        Create fraction in interval [start, end].

        Parameters
        ----------
        start : Optional[float]
            Start time of the fraction. If None, the first time point is used. The
            default is None.
        end: Optional[float]
            End time of the fraction. If None, the last time point is used. The
            default is None.

        Returns
        -------
        Fraction
            a Fraction object in the interval [start, end]
        """
        if start is None:
            start = self.time[0]

        if end is None:
            end = self.cycle_time

        from CADETProcess.fractionation import Fraction

        mass = self.fraction_mass(start, end)
        volume = self.fraction_volume(start, end)
        return Fraction(mass, volume, start, end)

    def fraction_mass(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> np.ndarray:
        """
        Component mass in a fraction interval.

        Parameters
        ----------
        start : Optional[float]
            Start time of the fraction. If None, the first time point is used. The
            default is None.
        end: Optional[float]
            End time of the fraction. If None, the last time point is used. The
            default is None.

        Returns
        -------
        np.ndarray
            Mass of all components in the fraction.
        """
        if start is None:
            start = self.time[0]

        if end is None:
            end = self.cycle_time

        # Note, we do not use self.dm_dt_interpolated to better account for
        # discontinuities in the flow rate profile, by passing the section times to
        # `quad_vec`. Maybe this can be improved in the future.
        def dm_dt(t: float, flow_rate: np.ndarray, solution: np.ndarray) -> np.ndarray:
            dm_dt = flow_rate.value(t) * solution(t)
            return dm_dt

        points = None
        if len(self.flow_rate.section_times) > 2:
            points = self.flow_rate.section_times[1:-1]

        mass = integrate.quad_vec(
            dm_dt,
            start,
            end,
            epsabs=1e-6,
            epsrel=1e-8,
            args=(self.flow_rate, self.solution_interpolated),
            points=points,
        )[0]

        return mass

    def fraction_volume(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> float:
        """
        Volume of a fraction interval.

        Parameters
        ----------
        start : Optional[float]
            Start time of the fraction. If None, the first time point is used. The
            default is None.
        end: Optional[float]
            End time of the fraction. If None, the last time point is used. The
            default is None.

        Returns
        -------
        float
            Volume of the fraction
        """
        if start is None:
            start = self.time[0]

        if end is None:
            end = self.cycle_time

        return float(self.flow_rate.integral(start, end).squeeze())

    @plotting.create_and_save_figure
    def plot(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        components: Optional[list[str]] = None,
        layout: Optional[plotting.Layout] = None,
        y_max: Optional[float] = None,
        x_axis_in_minutes: Optional[bool] = True,
        ax: Optional[Axes] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot the entire time_signal for each component.

        Parameters
        ----------
        start : Optional[float]
            Start time for plotting in seconds. If None is provided, the first data
            point will be used as the start time. The default is None.
        end : Optional[float]
            End time for plotting in seconds. If None is provided, the last data point
            will be used as the end time. The default is None.
        components : Optional[list[str]]
            List of components to be plotted. If None, all components are plotted.
            The default is None.
        layout : Optional[plotting.Layout]
            Plot layout options. The default is None.
        y_max : Optional[float]
            Maximum value of the y-axis. If None, the value is automatically
            determined from the data. The default is None.
        x_axis_in_minutes : Optional[bool], default=True
            If True, the x-axis will be plotted using minutes.
        ax : Optional[Axes]
            Axes to plot on. If None, a new figure is created.
        *args : Any
            Optional arguments passed down to _plot_solution_1D.
        **kwargs : Any
            Optional arguments passed down to _plot_solution_1D.

        Returns
        -------
        Axes
            Axes object with concentration profile.

        See Also
        --------
        _plot_solution_1D
        slice_solution
        plotlib
        plot_purity
        """
        solution = slice_solution(
            self,
            components=components,
            use_total_concentration=False,
            use_total_concentration_components=False,
            coordinates={"time": [start, end]},
        )

        x = solution.time
        if x_axis_in_minutes:
            x = x / 60
            if start is not None:
                start = start / 60
            if end is not None:
                end = end / 60

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = "$time~/~s$"
            if x_axis_in_minutes:
                layout.x_label = "$time~/~min$"
            layout.y_label = "$c~/~mM$"
            layout.x_lim = (start, end)
        if y_max is not None:
            layout.y_lim = (None, y_max)

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        return ax

    @plotting.create_and_save_figure
    def plot_purity(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        components: Optional[list[str]] = None,
        layout: Optional[plotting.Layout] = None,
        y_max: Optional[float] = None,
        x_axis_in_minutes: Optional[bool] = True,
        plot_components_purity: Optional[bool] = True,
        plot_species_purity: Optional[bool] = False,
        alpha: Optional[float] = 1,
        hide_labels: Optional[bool] = False,
        show_legend: Optional[bool] = True,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """
        Plot local purity for each component of the concentration profile.

        Parameters
        ----------
        start : Optional[float]
            Start time for plotting in seconds. If None is provided, the first data
            point will be used as the start time. The default is None.
        end : Optional[float]
            End time for plotting in seconds. If None is provided, the last data point
            will be used as the end time. The default is None.
        components : Optional[list[str]]
            List of components to be plotted. If None, all components are plotted.
            Note that if components are excluded, they will also not be considered in
            the calculation of the purity.
        layout : Optional[plotting.Layout]
            Plot layout options.
        y_max : Optional[float]
            Maximum value of y axis.
            If None, value is automatically deferred from solution.
        x_axis_in_minutes : Optional[bool], default=True
            If True, the x-axis will be plotted using minutes.
        plot_components_purity : Optional[bool], default=True
            If True, plot purity of total component concentration.
        plot_species_purity : Optional[bool], default=False
            If True, plot purity of individual species.
        alpha : Optional[float], default=1
            Opacity of line.
        hide_labels : Optional[bool], default=False
            If True, hide labels.
        show_legend : Optional[bool], default=True
            If True, show legend.
        ax : Optional[Axes]
            Axes to plot on.

        Returns
        -------
        Axes
            Axes with plot of purity over time.

        Raises
        ------
        CADETProcessError
            If solution has less than 2 components.

        See Also
        --------
        slice_solution
        plotlib
        plot
        """
        solution = slice_solution(
            self,
            components=components,
            use_total_concentration=False,
            use_total_concentration_components=False,
            coordinates={"time": [start, end]},
        )

        if solution.n_comp < 2:
            raise CADETProcessError(
                "Purity undefined for systems with less than 2 components."
            )

        x = solution.time
        if x_axis_in_minutes:
            x = x / 60
            if start is not None:
                start = start / 60
            if end is not None:
                end = end / 60

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = r"$time~/~s$"
            if x_axis_in_minutes:
                layout.x_label = r"$time~/~min$"
            layout.y_label = r"$Purity ~/~\%$"
            if start is not None:
                start /= 60
            if end is not None:
                end /= 60
            layout.x_lim = (start, end)
        if y_max is not None:
            layout.y_lim = (None, y_max)

        local_purity_components = solution.local_purity_components * 100
        local_purity_species = solution.local_purity_species * 100

        colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        species_index = 0
        for i, comp in enumerate(solution.component_system.components):
            color = next(colors)
            if hide_labels:
                label = None
            else:
                label = comp.name

            if plot_components_purity:
                y = local_purity_components[..., i]

                ax.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    alpha=alpha,
                )

            if plot_species_purity:
                if comp.n_species == 1:
                    species_index += 1
                    continue

                for s, species in enumerate(comp.species):
                    label = s

                    y = local_purity_species[..., species_index]

                    ax.plot(
                        x,
                        y,
                        "--",
                        label=label,
                        color=color,
                        alpha=alpha,
                    )
                    species_index += 1

        plotting.set_layout(
            ax,
            layout,
            show_legend,
        )

        return ax


class SolutionBulk(SolutionBase):
    """
    Solution in the bulk phase of the ``UnitOperation``.

    Dimensions: NCOL * NRAD * NCOMP
    """

    dimensions = SolutionBase.dimensions + [
        "axial_coordinates",
        "radial_coordinates",
        "component_coordinates",
    ]

    def __init__(
        self,
        name: str,
        component_system: ComponentSystem,
        time: npt.ArrayLike,
        solution: npt.ArrayLike,
        axial_coordinates: Optional[npt.ArrayLike] = None,
        radial_coordinates: Optional[npt.ArrayLike] = None,
    ) -> None:
        """
        Initialize the SolutionBulk.

        Parameters
        ----------
        name : str
            The name of the solution object.
        component_system : ComponentSystem
            The component system associated with the solution.
        time : npt.ArrayLike
            An array-like structure representing time points.
        solution : npt.ArrayLike
            An array-like structure representing the solution data.
        axial_coordinates : Optional[npt.ArrayLike],
            An array-like structure representing axial coordinates.
            If None, it is assumed that the model has a singleton axial dimension.
        radial_coordinates : Optional[npt.ArrayLike],
            An array-like structure representing radial coordinates.
            If None, it is assumed that the model has a singleton bulk dimension.
        """
        self.axial_coordinates = axial_coordinates
        self.radial_coordinates = radial_coordinates

        super().__init__(name, component_system, time, solution)

    @property
    def ncol(self) -> int:
        """int: Number of axial discretization points."""
        if self.axial_coordinates is None:
            return
        return len(self.axial_coordinates)

    @property
    def nrad(self) -> int:
        """int: Number of radial discretization points."""
        if self.radial_coordinates is None:
            return
        return len(self.radial_coordinates)

    @plotting.create_and_save_figure
    def plot(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        components: Optional[list[str]] = None,
        layout: Optional[plotting.Layout] = None,
        y_max: Optional[float] = None,
        x_axis_in_minutes: Optional[bool] = True,
        ax: Optional[Axes] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot the entire time_signal for each component.

        Parameters
        ----------
        start : Optional[float]
            Start time for plotting in seconds. If None is provided, the first data
            point will be used as the start time. The default is None.
        end : Optional[float]
            End time for plotting in seconds. If None is provided, the last data point
            will be used as the end time. The default is None.
        components : Optional[list[str]]
            List of components to be plotted. If None, all components are plotted.
        layout : Optional[plotting.Layout]
            Plot layout options.
        y_max : Optional[float]
            Maximum value of y axis.
            If None, value is automatically deferred from solution.
        x_axis_in_minutes : Optional[bool], default=True
            If True, the x-axis will be plotted using minutes.
        ax : Optional[Axes]
            Axes to plot on.
        *args : Any
            Optional arguments passed down to _plot_solution_1D.
        **kwargs : Any
            Optional arguments passed down to _plot_solution_1D.

        Returns
        -------
        Axes
            Axes object with concentration profile.

        Raises
        ------
        CADETProcessError
            If solution is not 1D.

        See Also
        --------
        _plot_solution_1D
        slice_solution
        plotlib
        plot_purity
        """
        if not (self.ncol is None and self.nrad is None):
            raise CADETProcessError(
                "Solution has more than single dimension. "
                "Please use `plot_at_time` or `plot_at_position`."
            )

        solution = slice_solution(
            self,
            components=components,
            use_total_concentration=False,
            use_total_concentration_components=False,
            coordinates={"time": [start, end]},
        )

        x = solution.time
        if x_axis_in_minutes:
            x = x / 60
            if start is not None:
                start = start / 60
            if end is not None:
                end = end / 60

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = "$time~/~s$"
            if x_axis_in_minutes:
                layout.x_label = "$time~/~min$"
            layout.y_label = "$c~/~mM$"
            layout.x_lim = (start, end)
            if y_max is not None:
                layout.y_lim = (None, y_max)

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        return ax

    @plotting.create_and_save_figure
    def plot_at_time(
        self,
        t: float,
        components: Optional[list[str]] = None,
        layout: Optional[plotting.Layout] = None,
        ax: Optional[Axes] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot bulk solution over space at given time.

        Parameters
        ----------
        t : float
            Time for plotting in seconds.
            If t == -1, the final solution is plotted.
        components : Optional[list[str]]
            List of components to be plotted. If None, all components are plotted.
        layout : Optional[plotting.Layout]
            Plot layout options.
        ax : Optional[Axes]
            Axes to plot on.
        *args : Any
            Optional arguments passed down to _plot_solution_1D.
        **kwargs : Any
            Optional arguments passed down to _plot_solution_1D.

        Returns
        -------
        Axes
            Axes object with concentration profile.

        See Also
        --------
        _plot_solution_1D
        slice_solution
        plot_at_position
        plotlib
        """
        solution = slice_solution(
            self,
            components=components,
            use_total_concentration=False,
            use_total_concentration_components=False,
            coordinates={"time": [t, t]},
        )

        x = self.axial_coordinates

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = "$z~/~m$"
            layout.y_label = "$c~/~mM$"

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        plotting.add_text(ax, f"time = {t:.2f} s")

        return ax

    @plotting.create_and_save_figure
    def plot_at_position(
        self,
        z: float,
        start: Optional[float] = None,
        end: Optional[float] = None,
        components: Optional[list[str]] = None,
        layout: Optional[plotting.Layout] = None,
        x_axis_in_minutes: bool = True,
        ax: Optional[Axes] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot bulk solution over time at given position.

        Parameters
        ----------
        z : float
            Position for plotting.
        start : Optional[float]
            Start time for plotting in seconds. If None is provided, the first data
            point will be used as the start time. The default is None.
        end : Optional[float]
            End time for plotting in seconds. If None is provided, the last data point
            will be used as the end time. The default is None.
        components : Optional[list[str]]
            List of components to be plotted. If None, all components are plotted.
        layout : Optional[plotting.Layout]
            Plot layout options.
            If None, value is automatically deferred from solution.
        x_axis_in_minutes : bool
            If True, the x-axis will be plotted using minutes. The default is True.
        ax : Optional[Axes]
            Axes to plot on.
        *args : Any
            Optional arguments passed down to _plot_solution_1D.
        **kwargs : Any
            Optional arguments passed down to _plot_solution_1D.

        Returns
        -------
        Axes
            Axes object with concentration profile.

        See Also
        --------
        _plot_solution_1D
        slice_solution
        plot_at_position
        plotlib
        """
        solution = slice_solution(
            self,
            components=components,
            use_total_concentration=False,
            use_total_concentration_components=False,
            coordinates={"axial_coordinates": [z, z]},
        )

        x = self.time
        if x_axis_in_minutes:
            x = x / 60
            if start is not None:
                start = start / 60
            if end is not None:
                end = end / 60

        x = self.time / 60

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = "$time~/~s$"
            if x_axis_in_minutes:
                layout.x_label = "$time~/~min$"
            layout.y_label = "$c~/~mM$"

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        plotting.add_text(ax, f"z = {z:.2f} m")

        return ax


class SolutionParticle(SolutionBase):
    """
    Solution in the particle liquid phase of the ``UnitOperation``.

    Dimensions: NCOL * NRAD * sum_{j}^{NPARTYPE}{NCOMP * NPAR,j}
    """

    dimensions = SolutionBase.dimensions + [
        "axial_coordinates",
        "radial_coordinates",
        "particle_coordinates",
        "component_coordinates",
    ]

    def __init__(
        self,
        name: str,
        component_system: ComponentSystem,
        time: npt.ArrayLike,
        solution: npt.ArrayLike,
        axial_coordinates: Optional[npt.ArrayLike] = None,
        radial_coordinates: Optional[npt.ArrayLike] = None,
        particle_coordinates: Optional[npt.ArrayLike] = None,
    ) -> None:
        """
        Initialize the SolutionParticle.

        Parameters
        ----------
        name : str
            The name of the solution object.
        component_system : ComponentSystem
            The component system associated with the solution.
        time : npt.ArrayLike
            An array-like structure representing time points.
        solution : npt.ArrayLike
            An array-like structure representing the solution data.
        axial_coordinates : Optional[npt.ArrayLike],
            An array-like structure representing axial coordinates.
            If None, it is assumed that the model has a singleton axial dimension.
        radial_coordinates : Optional[npt.ArrayLike],
            An array-like structure representing radial coordinates.
            If None, it is assumed that the model has a singleton bulk dimension.
        particle_coordinates : Optional[npt.ArrayLike],
            An array-like structure representing particle coordinates.
            If None, it is assumed that the model has a singleton particle dimension.
        """
        self.axial_coordinates = axial_coordinates
        self.radial_coordinates = radial_coordinates
        self.particle_coordinates = particle_coordinates

        super().__init__(name, component_system, time, solution)

    @property
    def ncol(self) -> int:
        """int: Number of axial discretization points."""
        if self.axial_coordinates is None:
            return
        return len(self.axial_coordinates)

    @property
    def nrad(self) -> int:
        """int: Number of radial discretization points."""
        if self.radial_coordinates is None:
            return
        return len(self.radial_coordinates)

    @property
    def npar(self) -> int:
        """int: Number of particle discretization points."""
        if self.particle_coordinates is None:
            return
        return len(self.particle_coordinates)

    @plotting.create_and_save_figure
    def plot(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        components: Optional[list[str]] = None,
        layout: Optional[plotting.Layout] = None,
        y_max: Optional[float] = None,
        x_axis_in_minutes: Optional[bool] = True,
        ax: Optional[Axes] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot the entire particle liquid phase solution for each component.

        Parameters
        ----------
        start : Optional[float]
            Start time for plotting in seconds. If None is provided, the first data
            point will be used as the start time. The default is None.
        end : Optional[float]
            End time for plotting in seconds. If None is provided, the last data point
            will be used as the end time. The default is None.
        components : Optional[list[str]]
            List of components to be plotted. If None, all components are plotted.
        layout : Optional[plotting.Layout]
            Plot layout options.
        y_max : Optional[float]
            Maximum value of y axis.
            If None, value is automatically deferred from solution.
        x_axis_in_minutes : Optional[bool], default=True
            If True, the x-axis will be plotted using minutes. The default is True.
        ax : Optional[Axes]
            Axes to plot on.
        *args : Any
            Optional arguments passed down to _plot_solution_1D.
        **kwargs : Any
            Optional arguments passed down to _plot_solution_1D.

        Returns
        -------
        ax : Axes
            Axes object with concentration profile.

        Raises CADETProcessError
            If solution is not 1D.

        See Also
        --------
        _plot_solution_1D
        slice_solution
        plotlib
        plot_purity
        """
        if not (self.ncol is None and self.nrad is None and self.npar is None):
            raise CADETProcessError(
                "Solution has more than single dimension. Please use `plot_at_time`."
            )

        solution = slice_solution(
            self,
            components=components,
            use_total_concentration=False,
            use_total_concentration_components=False,
            coordinates={"time": [start, end]},
        )

        x = solution.time
        if x_axis_in_minutes:
            x = x / 60
            if start is not None:
                start = start / 60
            if end is not None:
                end = end / 60

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = "$time~/~s$"
            if x_axis_in_minutes:
                layout.x_label = "$time~/~min$"
            layout.y_label = "$c~/~mM$"
            layout.x_lim = (start, end)
        if y_max is not None:
            layout.y_lim = (None, y_max)

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        return ax

    def _plot_1D(
        self,
        t: float,
        components: Optional[list[str]] = None,
        layout: Optional[plotting.Layout] = None,
        ax: Optional[Axes] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot particle solution over space at given time.

        Parameters
        ----------
        t : float
            Time for plotting.
            If t == -1, the final solution is plotted.
        components : Optional[list[str]]
            List of components to be plotted. If None, all components are plotted.
        layout : Optional[plotting.Layout]
            Plot layout options.
        ax : Optional[Axes]
            Axes to plot on.
        *args : Any
            Optional arguments passed down to _plot_solution_1D.
        **kwargs : Any
            Optional arguments passed down to _plot_solution_1D.

        Returns
        -------
        Axes
            Axes object with concentration profile.

        See Also
        --------
        _plot_solution_1D
        _plot_2D
        slice_solution
        plotlib
        """
        solution = slice_solution(
            self,
            components=components,
            use_total_concentration=False,
            use_total_concentration_components=False,
            coordinates={"time": [t, t]},
        )

        x = self.axial_coordinates

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = "$z~/~m$"
            layout.y_label = "$c~/~mM$"

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        plotting.add_text(ax, f"time = {t:.2f} s")

        return ax

    def _plot_2D(
        self,
        ax: Axes,
        t: float,
        comp: int = 0,
        vmax: Optional[float] = None,
    ) -> Axes:
        x = self.axial_coordinates
        y = self.particle_coordinates

        if not self.time[0] <= t <= self.time[-1]:
            raise ValueError("Time exceeds bounds.")
        i = np.where(t <= self.time)[0][0]
        v = self.solution[i, :, :, comp].transpose()

        if vmax is None:
            vmax = v.max()
        try:
            mesh = ax.get_children()[0]
            mesh.set_array(v.flatten())
        except AttributeError:
            mesh = ax.pcolormesh(x, y, v, shading="gouraud", vmin=0, vmax=vmax)

        plotting.add_text(ax, f"time = {t:.2f} s")

        layout = plotting.Layout()
        layout.x_label = "$z~/~m$"
        layout.y_label = "$r~/~m$"
        layout.title = f"Solid phase concentration, comp={comp}"

        plotting.set_layout(ax, layout)
        plt.colorbar(mesh)

        return ax

    @plotting.create_and_save_figure
    def plot_at_time(
        self,
        t: float,
        comp: Optional[int] = None,
        vmax: Optional[float] = None,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """
        Plot particle liquid solution for a given component over space at given time.

        Parameters
        ----------
        t : float
            Solution time at with to plot.
        comp : Optional[int]
            Component index.
        vmax: Optional[float]
            Maximum data value to scale color map.
        ax : Optional[Axes]
            Axes to plot on.

        Returns
        -------
        Axes
            Axes object with concentration profile.

        See Also
        --------
        CADETProcess.plotting
        """
        if self.npar is None:
            ax = self._plot_1D(ax, t, vmax)
        else:
            if comp is None and self.n_comp > 1:
                raise ValueError("Must specify component index.")

            ax = self._plot_2D(ax, t, comp, vmax)

        return ax


class SolutionSolid(SolutionBase):
    """
    Solution in the particle solid phase of the ``UnitOperation``.

    Dimensions: NCOL * NRAD * sum_{j}^{NPARTYPE}{NBOUND,j * NPAR,j}
    """

    dimensions = SolutionBase.dimensions + [
        "axial_coordinates",
        "radial_coordinates",
        "particle_coordinates",
        "component_coordinates",
    ]

    def __init__(
        self,
        name: str,
        component_system: ComponentSystem,
        bound_states: list[int],
        time: npt.ArrayLike,
        solution: npt.ArrayLike,
        axial_coordinates: Optional[npt.ArrayLike] = None,
        radial_coordinates: Optional[npt.ArrayLike] = None,
        particle_coordinates: Optional[npt.ArrayLike] = None,
    ) -> None:
        """
        Initialize the SolutionSolid.

        Parameters
        ----------
        name : str
            The name of the solution object.
        component_system : ComponentSystem
            The component system associated with the solution.
        bound_states: list[int]
            The number of bound states per component.
        time : npt.ArrayLike
            An array-like structure representing time points.
        solution : npt.ArrayLike
            An array-like structure representing the solution data.
        axial_coordinates : Optional[npt.ArrayLike],
            An array-like structure representing axial coordinates.
            If None, it is assumed that the model has a singleton axial dimension.
        radial_coordinates : Optional[npt.ArrayLike],
            An array-like structure representing radial coordinates.
            If None, it is assumed that the model has a singleton bulk dimension.
        particle_coordinates : Optional[npt.ArrayLike],
            An array-like structure representing particle coordinates.
            If None, it is assumed that the model has a singleton particle dimension.
        """
        self.bound_states = bound_states

        self.axial_coordinates = axial_coordinates
        self.radial_coordinates = radial_coordinates
        self.particle_coordinates = particle_coordinates

        super().__init__(name, component_system, time, solution)

    @property
    def n_bound(self) -> int:
        """int: Number of bound states."""
        return sum(self.bound_states)

    @property
    def ncol(self) -> int:
        """int: Number of axial discretization points."""
        if self.axial_coordinates is None:
            return None
        else:
            return len(self.axial_coordinates)

    @property
    def nrad(self) -> int:
        """int: Number of radial discretization points."""
        if self.radial_coordinates is None:
            return
        return len(self.radial_coordinates)

    @property
    def npar(self) -> int:
        """int: Number of particle discretization points."""
        if self.particle_coordinates is None:
            return
        return len(self.particle_coordinates)

    @plotting.create_and_save_figure
    def plot(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        components: Optional[list[str]] = None,
        layout: Optional[plotting.Layout] = None,
        y_max: Optional[float] = None,
        x_axis_in_minutes: Optional[bool] = True,
        ax: Optional[Axes] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot the entire solid phase solution for each component.

        Parameters
        ----------
        start : Optional[float]
            Start time for plotting in seconds. If None is provided, the first data
            point will be used as the start time. The default is None.
        end : Optional[float]
            End time for plotting in seconds. If None is provided, the last data point
            will be used as the end time. The default is None.
        components : Optional[list[str]]
            List of components to be plotted. If None, all components are plotted.
        layout : Optional[plotting.Layout]
            Plot layout options.
        y_max : Optional[float]
            Maximum value of y axis.
            If None, value is automatically deferred from solution.
        x_axis_in_minutes : Optional[bool], default=True
            If True, the x-axis will be plotted using minutes. The default is True.
        ax : Optional[Axes]
            Axes to plot on.
        *args : Any
            Optional arguments passed down to _plot_solution_1D.
        **kwargs : Any
            Optional arguments passed down to _plot_solution_1D.

        Returns
        -------
        ax : Axes
            Axes object with concentration profile.

        Raises CADETProcessError
            If solution is not 1D.

        See Also
        --------
        _plot_solution_1D
        slice_solution
        plotlib
        plot_purity
        """
        if not (self.ncol is None and self.nrad is None and self.npar is None):
            raise CADETProcessError(
                "Solution has more than single dimension. Please use `plot_at_time`."
            )

        solution = slice_solution(
            self,
            components=components,
            use_total_concentration=False,
            use_total_concentration_components=False,
            coordinates={"time": [start, end]},
        )

        x = solution.time
        if x_axis_in_minutes:
            x = x / 60
            if start is not None:
                start = start / 60
            if end is not None:
                end = end / 60

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = "$time~/~s$"
            if x_axis_in_minutes:
                layout.x_label = "$time~/~min$"
            layout.y_label = "$c~/~mM$"
            layout.x_lim = (start, end)
        if y_max is not None:
            layout.y_lim = (None, y_max)

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        return ax

    def _plot_1D(
        self,
        t: float,
        components: list[str] | None = None,
        layout: plotting.Layout | None = None,
        ax: Axes | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot solid solution over space at given time.

        Parameters
        ----------
        t : float
            Time for plotting, in seconds.
            If t == -1, the final solution is plotted.
        components : list, optional
            List of components to be plotted. If None, all components are plotted.
            The default is None.
        layout : plotting.Layout, optional
            Plot layout options. If None, a new instance is created.
            The default is None.
        ax : Axes
            Axes to plot on.
        *args : Any
            Optional arguments passed down to _plot_solution_1D.
        **kwargs : Any
            Optional arguments passed down to _plot_solution_1D.

        Returns
        -------
        ax : Axes
            Axes object with concentration profile.

        See Also
        --------
        _plot_solution_1D
        slice_solution
        plot_at_position
        plotlib
        """
        solution = slice_solution(
            self,
            components=components,
            use_total_concentration=False,
            use_total_concentration_components=False,
            coordinates={"time": [t, t]},
        )

        x = self.axial_coordinates

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = "$z~/~m$"
            layout.y_label = "$c~/~mM$"

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        plotting.add_text(ax, f"time = {t:.2f} s")

        return ax

    def _plot_2D(
        self,
        ax: Axes,
        t: float,
        comp: int,
        bound_state: int = 0,
        vmax: Optional[float] = None,
    ) -> tuple[Axes, QuadMesh]:
        x = self.axial_coordinates
        y = self.particle_coordinates

        if not self.time[0] <= t <= self.time[-1]:
            raise ValueError("Time exceeds bounds.")
        t_i = np.where(t <= self.time)[0][0]
        c_i = np.sum(self.bound_states[0:comp]) + bound_state
        v = self.solution[t_i, :, :, c_i].transpose()

        if vmax is None:
            vmax = v.max()

        mesh = ax.pcolormesh(x, y, v, shading="gouraud", vmin=0, vmax=vmax)

        plotting.add_text(ax, f"time = {t:.2f} s")

        layout = plotting.Layout()
        layout.title = (
            f"Solid phase concentration, comp={comp}, bound_state={bound_state}"
        )
        layout.x_label = "$z~/~m$"
        layout.y_label = "$r~/~m$"
        layout.labels = self.component_system.species[c_i]
        plotting.set_layout(ax, layout)

        return ax, mesh

    @plotting.create_and_save_figure
    def plot_at_time(
        self,
        t: float,
        comp: Optional[int] = None,
        bound_state: Optional[int] = 0,
        vmax: Optional[float] = None,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """
        Plot particle solid solution for a given bound state over space at given time.

        Parameters
        ----------
        t : float
            Solution time at with to plot.
        comp : Optional[int], default=None
            Number of components.
        bound_state : int, default=0
            Bound state index.
        vmax: Optional[float]
            Maximum data value to scale color map.
        ax : Optional[Axes]
            Axes to plot on.

        Returns
        -------
        Axes
            Axes object with concentration profile.

        See Also
        --------
        CADETProcess.plotting
        """
        if self.npar is None:
            ax = self._plot_1D(ax, t, vmax)
        else:
            if comp is None and self.n_comp > 1:
                raise ValueError("Must specify component index.")
            if comp is None:
                comp = 0
            if bound_state is None and self.bound_states[comp] > 1:
                raise ValueError("Must specify bound state index.")

            ax, mesh = self._plot_2D(ax, t, comp, bound_state, vmax=vmax)
            plt.colorbar(mesh, ax)
        return ax


class SolutionVolume(SolutionBase):
    """Volume solution (of e.g. CSTR)."""

    @property
    def solution_shape(self) -> tuple[int, ...]:
        """tuple: (Expected) shape of the solution."""
        return (self.nt,)

    @plotting.create_and_save_figure
    def plot(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        x_axis_in_minutes: Optional[bool] = True,
        ax: Optional[Axes] = None,
        update_layout: Optional[bool] = True,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot the unit operation"s volume over time.

        Parameters
        ----------
        start : Optional[float]
            Start time for plotting in seconds. If None is provided, the first data
            point will be used as the start time. The default is None.
        end : Optional[float]
            End time for plotting in seconds. If None is provided, the last data point
            will be used as the end time. The default is None.
        x_axis_in_minutes : Optional[bool], default=True
            If True, the x-axis will be plotted using minutes.
        ax : Optional[Axes]
            Axes to plot on.
        update_layout : Optional[bool], default=True
            If True, update the figure"s layout.
        **kwargs : Any
            Additional arguments passed down to ax.plot()

        Returns
        -------
        Axes
            Axes object with the plot.

        See Also
        --------
        CADETProcess.plot
        """
        x = self.time
        if x_axis_in_minutes:
            x = x / 60
            if start is not None:
                start = start / 60
            if end is not None:
                end = end / 60

        y = self.solution * 1000

        y_min = np.min(y)
        y_max = 1.1 * np.max(y)

        layout = plotting.Layout()
        layout.x_label = "$time~/~s$"
        if x_axis_in_minutes:
            layout.x_label = "$time~/~min$"
        layout.y_label = "$V~/~L$"
        layout.x_lim = (start, end)
        layout.y_lim = (y_min, y_max)
        ax.plot(x, y, **kwargs)

        if update_layout:
            plotting.set_layout(ax, layout)

        return ax


def _plot_solution_1D(
    ax: Axes,
    x: np.ndarray,
    solution: SolutionBase,
    layout: Optional[Any] = None,
    plot_species: bool = False,
    plot_components: bool = True,
    plot_total_concentration: bool = False,
    alpha: int = 1,
    hide_labels: bool = False,
    hide_species_labels: bool = False,
    secondary_axis: Optional[Axes] = None,
    secondary_layout: Optional[Any] = None,
    show_legend: bool = True,
    update_layout: bool = True,
) -> Axes:
    sol = solution.solution
    c_total_comp = solution.total_concentration_components

    if secondary_axis is not None:
        ax_secondary = ax.twinx()
    else:
        ax_secondary = None

    species_index = 0
    y_min = 0
    y_max = 0
    y_min_sec = 0
    y_max_sec = 0
    colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for i, comp in enumerate(solution.component_system.components):
        color = next(colors)
        if hide_labels or not update_layout:
            label = None
        else:
            label = comp.name

        if secondary_axis is not None and comp.name in secondary_axis.components:
            a = ax_secondary
        else:
            a = ax

        if plot_components:
            y_comp = c_total_comp[..., i]
            if secondary_axis is not None and comp.name in secondary_axis.components:
                if secondary_axis.transform is not None:
                    y_comp = secondary_axis.transform(y_comp)
                y_min_sec = min(min(y_comp), y_min_sec)
                y_max_sec = max(max(y_comp), y_max_sec)
            else:
                y_min = np.min((np.min(y_comp), y_min))
                y_max = np.max((np.max(y_comp), y_max))

            y_comp = np.squeeze(y_comp)

            a.plot(
                x,
                y_comp,
                linestyle="-",
                color=color,
                alpha=alpha,
                label=label,
            )

        if plot_species:
            if comp.n_species == 1:
                if plot_components:
                    species_index += 1
                    continue

            linestyle_iter = iter(plotting.linestyle_cycler)

            for s, species in enumerate(comp.label):
                if hide_species_labels or not update_layout:
                    label = None
                else:
                    label = species

                if (
                    comp.n_species == 1
                    and not plot_total_concentration
                    and not plot_components
                ):
                    linestyle = "-"
                else:
                    linestyle = next(linestyle_iter)["linestyle"]

                if (
                    secondary_axis is not None
                    and secondary_axis.transform is not None
                    and comp.name in secondary_axis.components
                ):
                    y_spec = secondary_axis.transform(sol[..., species_index])
                    y_min_sec = min(min(y_spec), y_min_sec)
                    y_max_sec = max(max(y_spec), y_max_sec)
                else:
                    y_spec = sol[..., species_index]
                    y_min = min(min(y_spec), y_min)
                    y_max = max(max(y_spec), y_max)

                y_spec = np.squeeze(y_spec)

                a.plot(
                    x,
                    y_spec,
                    linestyle=linestyle,
                    label=label,
                    color=color,
                    alpha=alpha,
                )
                species_index += 1

    if plot_total_concentration:
        if hide_labels or not update_layout:
            label = None
        else:
            label = "Total concentration"

        y_total = solution.total_concentration
        y_total = np.squeeze(y_total)
        y_min = min(min(y_total), y_min)
        y_max = max(max(y_total), y_max)
        a.plot(
            x,
            y_total,
            "-",
            label=label,
            color="k",
            alpha=alpha,
        )

    if layout.y_lim is None:
        layout.y_lim = (y_min, 1.1 * y_max)

    if secondary_axis is not None and secondary_layout is None:
        secondary_layout = plotting.Layout()
        secondary_layout.y_label = secondary_axis.y_label
        secondary_layout.y_lim = (y_min_sec, 1.1 * y_max_sec)

    if update_layout:
        plotting.set_layout(
            ax,
            layout,
            show_legend,
            ax_secondary,
            secondary_layout,
        )

    return ax


class InterpolatedSignal:
    def __init__(self, time: np.ndarray, signal: npt.ArrayLike) -> None:
        if len(signal.shape) == 1:
            signal = np.array(signal, ndmin=2).transpose()
        self._solutions = [
            PchipInterpolator(time, signal[:, comp]) for comp in range(signal.shape[1])
        ]
        self._derivatives = [signal.derivative() for signal in self._solutions]
        self._antiderivatives = [signal.antiderivative() for signal in self._solutions]

    @property
    def solutions(self) -> list[PchipInterpolator]:
        """list[PchipInterpolator]: Interpolators for each component."""
        return self._solutions

    @property
    def derivatives(self) -> list[PchipInterpolator]:
        """list[PchipInterpolator]: Derivative interpolators for each component."""
        return self._derivatives

    def derivative(self, time: np.ndarray) -> np.ndarray:
        """
        Calculate derivatives of the solution spline(s) at given time points.

        Parameters
        ----------
        time : np.ndarray
            The time points to evaluate the derivatives at.

        Returns
        -------
        np.ndarray
            Derivatives of the solution spline(s) at given time.
        """
        der = np.empty((len(time), len(self.solutions)))
        for comp, der_i in enumerate(self.derivatives):
            der[:, comp] = der_i(time)

        return der

    @property
    def antiderivatives(self) -> list[PchipInterpolator]:
        """list[PchipInterpolator]: Antiderivative interpolators for each component."""
        return self._antiderivatives

    def antiderivative(self, time: np.ndarray) -> np.ndarray:
        """
        Antiderivative of the solution of the spline(s) at given time.

        Parameters
        ----------
        time : np.ndarray
            The time points to evaluate the antiderivatives at.

        Returns
        -------
        np.ndarray
            Antiderivatives of the solution spline(s) at given time.
        """
        anti = np.empty((len(time), len(self.solutions)))
        for comp, anti_i in enumerate(self.antiderivatives):
            anti[:, comp] = anti_i(time)

        return anti

    def integral(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> np.ndarray:
        """
        Definite integral between start and end.

        Parameters
        ----------
        start : Optional[float]
            Lower integration bound.
        end: Optional[float]
            Upper integration bound.

        Returns
        -------
        np.ndarray
            Definite integral of the solution spline(s) between limits.
        """
        return np.array([comp.integrate(start, end) for comp in self._solutions])

    def __call__(self, t: float) -> np.ndarray:
        return np.array([comp(t) for comp in self._solutions]).transpose()


def slice_solution(
    solution_original: SolutionBase,
    components: Optional[str | list[str]] = None,
    use_total_concentration: Optional[bool] = False,
    use_total_concentration_components: Optional[bool] = False,
    coordinates: Optional[dict[str, Optional[tuple[Optional[int], Optional[int]]]]] = None,
) -> SolutionBase:
    """
    Slice a `Solution` object along specified dimensions, components or both.

    Parameters
    ----------
    solution_original : SolutionBase
        The `Solution` object to slice.
    components : Optional[str | list[str]]
        The names of the components to keep in the sliced `Solution`.
        If `None`, all components are kept. The default is None.
    use_total_concentration : Optional[bool], default=False
        If `True`, only the total concentration data is kept in the sliced `Solution`.
    use_total_concentration_components : Optional[bool], default=False
        If `True`, the total concentration data is kept for each individual species
        of each component in the sliced `Solution`.
    coordinates : Optional[dict[str, Optional[tuple[Optional[int], Optional[int]]]]]
        A dictionary mapping dimensions to slice coordinates. Each dimension in the
        solution object is represented by a key in the dictionary, and the corresponding
        value is a tuple of two elements specifying the start and end coordinates of the
        slice along that dimension.
        If a value is `None`, the corresponding coordinate is not sliced.

    Returns
    -------
    SolutionBase
        A new solution object representing the sliced data.

    Raises
    ------
    ValueError
        If any of the slice coordinates exceeds the bounds of its corresponding
        dimension.
    CADETProcessError
        If any of the specified components or dimensions does not exist in the original
        solution.
    """
    solution = copy.deepcopy(solution_original)

    slices = ()
    if coordinates is not None:
        coordinates = copy.deepcopy(coordinates)
        for i, (dim, coord) in enumerate(solution.coordinates.items()):
            if dim == "component_coordinates":
                continue
            if dim in coordinates:
                sl = list(coordinates.pop(dim))

                # Update slice bounds if they are None
                if sl[0] is None:
                    sl[0] = coord[0]
                if sl[1] is None:
                    sl[1] = coord[-1]

                # Check bounds
                if not (coord[0] <= sl[0] <= sl[1] <= coord[-1]):
                    raise ValueError(f"{dim} coordinates exceed bounds.")

                # Calculate start and end indices using searchsorted
                start_index = np.searchsorted(coord, sl[0], side="right") - 1
                end_index = np.searchsorted(coord, sl[1], side="left") + 1

                # Ensure only a single entry is returned if start and end elements are the same
                if sl[0] == sl[1]:
                    end_index = start_index + 1

                # Create slice and update solution
                sl = slice(start_index, end_index)
                slices += (sl,)
                setattr(solution, dim, coord[sl])
            else:
                slices += (slice(None),)

        if len(coordinates) > 0:
            raise CADETProcessError(f"Unknown dimensions: {coordinates.keys()}")

        solution.solution = solution_original.solution[slices]

    # First calculate total_concentration of components, if required.
    if use_total_concentration_components:
        sol_total_concentration_comp = solution.total_concentration_components
        for comp in solution.component_system:
            if comp.n_species > 1:
                comp._species = []
                comp.add_species(comp.name)
        solution.solution = sol_total_concentration_comp

    # Then, slice components. Note that component index can only be used if total
    # concentration of components has already been calculated and set as solution array.
    if components is not None:
        if not isinstance(components, list):
            components = [components]
        components = copy.deepcopy(components)

        component_system = copy.deepcopy(solution.component_system)
        component_indices = []
        for i, (name, component) in enumerate(
            solution.component_system.components_dict.items()
        ):
            if name not in components:
                component_system.remove_component(component.name)
            else:
                name = str(name)
                components.remove(name)
                if use_total_concentration_components:
                    component_indices.append(i)
                else:
                    component_indices += solution.component_system.indices[component.name]

        if len(components) != 0:
            raise CADETProcessError(f"Unknown components: {components}")

        solution_components = solution.solution[..., component_indices]
        solution.component_system = component_system
        solution.solution = solution_components

    # Only calculate total concentration after removing unwanted components.
    if use_total_concentration:
        solution_total_concentration = solution.total_concentration
        solution.component_system = ComponentSystem(["total_concentration"])
        solution.solution = solution_total_concentration

    solution.update_solution()

    return solution


def slice_solution_front(
    solution_original: SolutionIO,
    min_percent: Optional[float] = 0.02,
    max_percent: Optional[float] = 0.98,
    use_max_slope: Optional[bool] = False,
    return_indices: Optional[bool] = False,
) -> SolutionIO | tuple[SolutionIO, int, int]:
    """
    Slice the front of a given solution.

    Parameters
    ----------
    solution_original : SolutionIO
        The `Solution` object to slice.
    min_percent : Optional[float]
        Minimum percentage of the peak height to consider. Default is 0.02.
    max_percent : Optional[float]
        Maximum percentage of the peak height to consider. Default is 0.98.
    use_max_slope : Optional[bool]
        If True, cut use the maximum slope as end. Default if False.
    return_indices : Optional[bool]
        If True, also return the start end end indices of the front. Default if False.

    Returns
    -------
    SolutionIO or tuple[SolutionIO, int, int]
        The sliced `SolutionIO` object. If `return_indices` is True, also returns the
        start and end indices of the slice.
    """
    solution = copy.deepcopy(solution_original)

    # Initialize new sequence with zeros
    solution.solution[:] = 0

    # Determine whether to use derivative array
    solution_array = solution_original.solution
    if use_max_slope:
        solution_array = solution_original.derivative.solution

    # Determine the max value and its index
    max_value = np.max(solution_array)
    max_index = np.argmax(solution_array)

    # Determine the min and max percent values
    select_min = min_percent * max_value
    select_max = max_percent * max_value

    # Find the indices for slicing using logical indexing
    idx_min = np.where(solution_array[:max_index] <= select_min)[0][-1]
    idx_max = np.where(solution_array[:max_index] >= select_max)[0][0]

    # Slice the sequence
    solution.solution[idx_min : idx_max + 1] = \
        solution_original.solution[idx_min : idx_max + 1]
    solution.update_solution()

    if not return_indices:
        return solution

    return solution, idx_min, idx_max
