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

"""

import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import integrate

from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    String, UnsignedInteger, UnsignedFloat, Vector, DependentlySizedNdArray
)

from CADETProcess.processModel import ComponentSystem
from CADETProcess.dynamicEvents import TimeLine

from CADETProcess import plotting
from CADETProcess import CADETProcessError

from CADETProcess import smoothing
from CADETProcess import transform


__all__ = [
    'SolutionIO',
    'SolutionBulk',
    'SolutionParticle',
    'SolutionSolid',
    'SolutionVolume',
    'slice_solution',
]


class SolutionBase(metaclass=StructMeta):
    """Base class for solutions of component systems.

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
    solution = DependentlySizedNdArray(dep='solution_shape')
    c_min = UnsignedFloat(default=1e-6)

    dimensions = ['time']

    def __init__(self, name, component_system, time, solution):
        self.name = name
        self.component_system_original = component_system
        self.time_original = time
        self.solution_original = solution

        self.reset()

    def reset(self):
        """Reset component system, time, and solution arrays to their original values."""
        self.component_system = self.component_system_original
        self.time = self.time_original
        self.solution = self.solution_original

    def update(self):
        """Update the solution."""
        pass

    def update_transform(self):
        """Update the transforms."""
        pass

    @property
    def component_system(self):
        """ComponentSystem: ComponentSystem of the Solution object."""
        return self._component_system

    @component_system.setter
    def component_system(self, component_system):
        if not isinstance(component_system, ComponentSystem):
            raise TypeError('Expected ComponentSystem')
        self._component_system = component_system

    @property
    def n_comp(self):
        """int: Number of components."""
        return self.component_system.n_comp

    @property
    def cycle_time(self):
        """float: Cycle time."""
        return self.time[-1]

    @property
    def nt(self):
        """int: Number of time steps."""
        return len(self.time)

    @property
    def component_coordinates(self):
        """np.ndarray: Indices of the components."""
        return np.arange(self.n_comp)

    @property
    def coordinates(self):
        """np.ndarray: Coordinates of the Solution."""
        coordinates = {}

        for c in self.dimensions:
            v = getattr(self, c)
            if v is None:
                continue
            coordinates[c] = v
        return coordinates

    @property
    def solution_shape(self):
        """tuple: (Expected) shape of the solution."""
        return tuple(len(c) for c in self.coordinates.values())

    @property
    def total_concentration_components(self):
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
    def total_concentration(self):
        """np.ndarray: Total concentration (sum) of all components."""
        return np.sum(self.solution, axis=-1, keepdims=True)

    @property
    def local_purity_components(self):
        """np.ndarray: Local purity of each component."""
        solution = self.total_concentration_components
        c_total = self.total_concentration
        c_total[c_total < self.c_min] = np.nan

        purity = np.zeros(solution.shape)
        with np.errstate(divide='ignore', invalid='ignore'):
            purity = solution/c_total

        purity = np.nan_to_num(purity)

        return purity

    @property
    def local_purity_species(self):
        """np.ndarray: Local purity of components."""
        solution = self.solution
        c_total = self.total_concentration
        c_total[c_total < self.c_min] = np.nan

        purity = np.zeros(solution.shape)
        with np.errstate(divide='ignore', invalid='ignore'):
            purity = solution/c_total

        purity = np.nan_to_num(purity)

        return purity


class SolutionIO(SolutionBase):
    """Solution representing streams at the inlet or outlet of a ``UnitOperation``.

    Notes
    -----
    The `flow_rate` attribute is implemented as TimeLine to improve interpolation of
    signals with discontinuous flow.

    """

    dimensions = SolutionBase.dimensions + ['component_coordinates']

    def __init__(self, name, component_system, time, solution, flow_rate):

        super().__init__(name, component_system, time, solution)

        if not isinstance(flow_rate, TimeLine):
            flow_rate = TimeLine.from_profile(time, flow_rate)
        self.flow_rate = flow_rate

        self.reset()

    @property
    def derivative(self):
        """SolutionIO: Derivative of this solution."""
        derivative = copy.deepcopy(self)
        derivative.reset()
        derivative_fun = derivative.solution_interpolated.derivative
        derivative.solution_original = derivative_fun(derivative.time)
        derivative.reset()

        return derivative

    @property
    def antiderivative(self):
        """SolutionIO: Antiderivative of this solution."""
        antiderivative = copy.deepcopy(self)
        antiderivative.reset()
        antiderivative_fun = antiderivative.solution_interpolated.antiderivative
        antiderivative.solution_original = antiderivative_fun(antiderivative.time)
        antiderivative.reset()

        return antiderivative

    def reset(self):
        super().reset()
        self.is_resampled = False
        self.is_normalized = False
        self.is_smoothed = False

        self.s = None
        self.crit_fs = None
        self.crit_fs_der = None
        self.is_smoothed = False

        self.update()
        self.update_transform()

    def update(self):
        self._solution_interpolated = None
        self._dm_dt_interpolated = None

    def update_transform(self):
        self.transform = transform.NormLinearTransform(
            np.min(self.solution, axis=0),
            np.max(self.solution, axis=0),
            allow_extended_input=True,
            allow_extended_output=True
        )

    @property
    def solution_interpolated(self):
        if self._solution_interpolated is None:
            self._solution_interpolated = \
                InterpolatedSignal(self.time, self.solution)

        return self._solution_interpolated

    @property
    def dm_dt_interpolated(self):
        if self._dm_dt_interpolated is None:
            self.resample()
            dm_dt = self.solution * self.flow_rate.value(self.time)
            self._dm_dt_interpolated = InterpolatedSignal(self.time, dm_dt)

        return self._dm_dt_interpolated

    def normalize(self):
        """Normalize the solution using the transformation function."""
        if self.is_normalized:
            return

        self.solution = self.transform.transform(self.solution)
        self.update()
        self.is_normalized = True

    def denormalize(self):
        """Denormalize the solution using the transformation function."""
        if not self.is_normalized:
            return

        self.solution = self.transform.untransform(self.solution)
        self.update()
        self.is_normalized = False

    def resample(self, start=None, end=None, nt=5001):
        """Resample solution to nt time points.

        Parameters
        ----------
        nt : int, optional
            Number of points to resample. The default is 5001.

        """
        if self.is_resampled:
            return

        if start is None:
            start = self.time[0]
        if end is None:
            end = self.time[-1]

        solution_interpolated = self.solution_interpolated
        self.time = np.linspace(start, end, nt)
        self.solution = solution_interpolated(self.time)

        self.update()

        self.is_resampled = True

    def smooth_data(self, s=None, crit_fs=None, crit_fs_der=None):
        """Smooth data.

        Parameters
        ----------
        s : float or list, optional
            DESCRIPTION. The default is 0.
        crit_fs : float or list
            DESCRIPTION.
        crit_fs_der : float or list
            DESCRIPTION.

        """
        if self.is_smoothed:
            return

        if not self.is_resampled:
            self.resample()

        if not self.is_normalized:
            normalized = True
            self.normalize()
        else:
            normalized = False

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
        self.s = s

        if np.isscalar(crit_fs):
            crit_fs = self.n_comp * [crit_fs]
        elif len(crit_fs) == 1:
            crit_fs = self.n_comp * crit_fs
        self.crit_fs = crit_fs

        if np.isscalar(crit_fs_der):
            crit_fs_der = self.n_comp * [crit_fs_der]
        elif len(crit_fs_der) == 1:
            crit_fs_der = self.n_comp * crit_fs_der
        self.crit_fs_der = crit_fs_der

        solution = np.zeros((self.solution.shape))
        for i, (s, crit_fs, crit_fs_der) in enumerate(zip(s, crit_fs, crit_fs_der)):
            smooth, smooth_der = smoothing.full_smooth(
                self.time, self.solution[..., i],
                crit_fs, s, crit_fs_der
            )
            solution[..., i] = smooth

        self.solution = solution

        if normalized:
            self.denormalize()

        self.update()

        self.is_smoothed = True

    def integral(self, start=None, end=None):
        """Peak area in a fraction interval.

        Parameters
        ----------
        start : float
            Start time of the fraction

        end: float
            End time of the fraction

        Returns
        -------
        Area : np.ndarray
            Mass of all components in the fraction

        """
        if end is None:
            end = self.cycle_time

        return self.solution_interpolated.integral(start, end)

    def fraction_mass(self, start=None, end=None):
        """Component mass in a fraction interval

        Parameters
        ----------
        start : float
            Start time of the fraction

        end: float
            End time of the fraction

        Returns
        -------
        fraction_mass : np.ndarray
            Mass of all components in the fraction

        """
        if end is None:
            end = self.cycle_time

        def dm_dt(t, flow_rate, solution):
            dm_dt = flow_rate.value(t)*solution(t)
            return dm_dt

        points = None
        if len(self.flow_rate.section_times) > 2:
            points = self.flow_rate.section_times[1:-1]

        mass = integrate.quad_vec(
            dm_dt,
            start, end,
            epsabs=1e-6,
            epsrel=1e-8,
            args=(self.flow_rate, self.solution_interpolated),
            points=points,
        )[0]

        return mass

    def fraction_volume(self, start=None, end=None):
        """Volume of a fraction interval

        Parameters
        ----------
        start : float
            Start time of the fraction

        end: float
            End time of the fraction

        Returns
        -------
        fraction_volume : np.ndarray
            Volume of the fraction

        """
        if end is None:
            end = self.cycle_time

        return float(self.flow_rate.integral(start, end))

    @plotting.create_and_save_figure
    def plot(
            self,
            start=None,
            end=None,
            components=None,
            layout=None,
            y_max=None,
            ax=None,
            *args,
            **kwargs,
            ):
        """Plot the entire time_signal for each component.

        Parameters
        ----------
        start : float, optional
            Start time for plotting. The default is 0.
        end : float, optional
            End time for plotting.
        components : list, optional.
            List of components to be plotted. If None, all components are plotted.
        layout : plotting.Layout
            Plot layout options.
        y_max : float, optional
            Maximum value of y axis.
            If None, value is automatically deferred from solution.
        ax : Axes
            Axes to plot on.

        Returns
        -------
        ax : Axes
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
            coordinates={'time': [start, end]}
        )

        x = solution.time / 60

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = '$time~/~min$'
            layout.y_label = '$c~/~mM$'
            if start is not None:
                start /= 60
            if end is not None:
                end /= 60
            layout.x_lim = (start, end)
        if y_max is not None:
            layout.y_lim = (None, y_max)

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        return ax

    @plotting.create_and_save_figure
    def plot_purity(
            self,
            start=None, end=None, y_max=None,
            layout=None,
            only_plot_components=False,
            alpha=1, hide_labels=False,
            show_legend=True,
            ax=None):
        """Plot local purity for each component of the concentration profile.

        Parameters
        ----------
        start : float
            start time for plotting
        end : float
            end time for plotting
        ax : Axes
            Axes to plot on.

        Returns
        -------
        ax : Axes
            Axes with plot of purity over time.

        See Also
        --------
        plotlib
        plot

        """
        time = self.time / 60

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = r'$time~/~min$'
            layout.y_label = r'$Purity ~/~\%$'
            layout.x_lim = (start, end)
            layout.y_lim = (0, 100)

        local_purity_components = self.local_purity_components * 100
        local_purity_species = self.local_purity_species * 100

        species_index = 0
        for i, comp in enumerate(self.component_system.components):
            color = next(ax._get_lines.prop_cycler)['color']
            if hide_labels:
                label = None
            else:
                label = comp.name

            y = local_purity_components[..., i]

            ax.plot(
                time, y,
                label=label,
                color=color,
                alpha=alpha
            )

            if not only_plot_components:
                if comp.n_species == 1:
                    species_index += 1
                    continue

                for s, species in enumerate(comp.species):
                    label = s

                    y = local_purity_species[..., species_index]

                    ax.plot(
                        time, y, '--',
                        label=label,
                        color=color,
                        alpha=alpha
                    )
                    species_index += 1

        plotting.set_layout(
            ax,
            layout,
            show_legend,
        )

        return ax


class SolutionBulk(SolutionBase):
    """Solution in the bulk phase of the ``UnitOperation``.

    Dimensions: NCOL * NRAD * NCOMP

    """

    dimensions = SolutionBase.dimensions + [
        'axial_coordinates',
        'radial_coordinates',
        'component_coordinates',
    ]

    def __init__(
            self,
            name,
            component_system,
            time, solution,
            axial_coordinates=None, radial_coordinates=None
            ):
        self.name = name
        self.component_system_original = component_system
        self.time_original = time

        self.axial_coordinates = axial_coordinates
        # Account for dimension reduction in case of only one cell (e.g. LRMP)
        if radial_coordinates is not None and len(radial_coordinates) == 1:
            radial_coordinates = None
        self.radial_coordinates = radial_coordinates

        self.solution_original = solution

        self.reset()

    @property
    def ncol(self):
        """int: Number of axial discretization points."""
        if self.axial_coordinates is None:
            return
        return len(self.axial_coordinates)

    @property
    def nrad(self):
        """int: Number of radial discretization points."""
        if self.radial_coordinates is None:
            return
        return len(self.radial_coordinates)

    @plotting.create_and_save_figure
    def plot(
            self,
            start=None,
            end=None,
            components=None,
            layout=None,
            y_max=None,
            ax=None,
            *args,
            **kwargs,
            ):
        """Plot the entire time_signal for each component.

        Parameters
        ----------
        start : float, optional
            Start time for plotting. The default is 0.
        end : float, optional
            End time for plotting.
        components : list, optional.
            List of components to be plotted. If None, all components are plotted.
        layout : plotting.Layout
            Plot layout options.
        y_max : float, optional
            Maximum value of y axis.
            If None, value is automatically deferred from solution.
        ax : Axes
            Axes to plot on.

        Returns
        -------
        ax : Axes
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
            start=start,
            end=end,
        )

        x = solution.time / 60

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = '$time~/~min$'
            layout.y_label = '$c~/~mM$'
            if start is not None:
                start /= 60
            if end is not None:
                end /= 60
            layout.x_lim = (start, end)
            if y_max is not None:
                layout.y_lim = (None, y_max)

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        return ax

    @plotting.create_and_save_figure
    def plot_at_time(
            self,
            t,
            components=None,
            layout=None,
            ax=None,
            *args,
            **kwargs,
            ):
        """Plot bulk solution over space at given time.

        Parameters
        ----------
        t : float
            Time for plotting
            If t == -1, the final solution is plotted.
        components : list, optional.
            List of components to be plotted. If None, all components are plotted.
        layout : plotting.Layout
            Plot layout options.
        ax : Axes
            Axes to plot on.

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
            coordinates={'time': [t, t]},
        )

        x = self.axial_coordinates

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = '$z~/~m$'
            layout.y_label = '$c~/~mM$'

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        plotting.add_text(ax, f'time = {t:.2f} s')

        return ax

    @plotting.create_and_save_figure
    def plot_at_position(
            self,
            z,
            components=None,
            layout=None,
            ax=None,
            *args,
            **kwargs,
            ):
        """Plot bulk solution over time at given position.

        Parameters
        ----------
        z : float
            Position for plotting.
        components : list, optional.
            List of components to be plotted. If None, all components are plotted.
        layout : plotting.Layout
            Plot layout options.
        ax : Axes
            Axes to plot on.

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
            coordinates={'axial_coordinates': [z, z]},
        )

        x = self.time / 60

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = '$time~/~min$'
            layout.y_label = '$c~/~mM$'

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        plotting.add_text(ax, f'z = {z:.2f} m')

        return ax


class SolutionParticle(SolutionBase):
    """Solution in the particle liquid phase of the ``UnitOperation``.

    Dimensions: NCOL * NRAD * sum_{j}^{NPARTYPE}{NCOMP * NPAR,j}

    """

    dimensions = SolutionBase.dimensions + [
        'axial_coordinates',
        'radial_coordinates',
        'particle_coordinates',
        'component_coordinates',
    ]

    def __init__(
            self,
            name,
            component_system,
            time, solution,
            axial_coordinates=None,
            radial_coordinates=None,
            particle_coordinates=None
            ):

        self.axial_coordinates = axial_coordinates
        # Account for dimension reduction in case of only one cell (e.g. LRMP)
        if radial_coordinates is not None and len(radial_coordinates) == 1:
            radial_coordinates = None
        self.radial_coordinates = radial_coordinates
        # Account for dimension reduction in case of only one cell (e.g. CSTR)
        if particle_coordinates is not None and len(particle_coordinates) == 1:
            particle_coordinates = None
        self.particle_coordinates = particle_coordinates

        super().__init__(name, component_system, time, solution)

    @property
    def ncol(self):
        """int: Number of axial discretization points."""
        if self.axial_coordinates is None:
            return
        return len(self.axial_coordinates)

    @property
    def nrad(self):
        """int: Number of radial discretization points."""
        if self.radial_coordinates is None:
            return
        return len(self.radial_coordinates)

    @property
    def npar(self):
        if self.particle_coordinates is None:
            return
        return len(self.particle_coordinates)

    def _plot_1D(
            self,
            t,
            components=None,
            layout=None,
            ax=None,
            *args,
            **kwargs,
            ):
        """Plot bulk solution over space at given time.

        Parameters
        ----------
        t : float
            Time for plotting
            If t == -1, the final solution is plotted.
        components : list, optional.
            List of components to be plotted. If None, all components are plotted.
        layout : plotting.Layout
            Plot layout options.
        ax : Axes
            Axes to plot on.

        Returns
        -------
        ax : Axes
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
            coordinates={'time': [t, t]},
        )

        x = self.axial_coordinates

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = '$z~/~m$'
            layout.y_label = '$c~/~mM$'

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        plotting.add_text(ax, f'time = {t:.2f} s')

        return ax


    def _plot_2D(self, t, comp, vmax, ax=None):
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
        except:
            mesh = ax.pcolormesh(x, y, v, shading='gouraud', vmin=0, vmax=vmax)

        plotting.add_text(ax, f'time = {t:.2f} s')

        layout = plotting.Layout()
        layout.x_label = '$z~/~m$'
        layout.y_label = '$r~/~m$'
        layout.title = f'Solid phase concentration, comp={comp}'

        plotting.set_layout(ax, layout)
        plt.colorbar(mesh)

        return ax

    @plotting.create_and_save_figure
    def plot_at_time(self, t, comp=0, vmax=None, ax=None):
        """Plot particle liquid solution over space at given time.

        Parameters
        ----------
        t : float
            time for plotting
        comp : int
            component index
        ax : Axes
            Axes to plot on.

        See Also
        --------
        CADETProcess.plotting
        """
        if self.npar is None:
            ax = self._plot_1D(ax, t, vmax)
        else:
            ax = self._plot_2D(ax, t, comp, vmax)

        return ax


class SolutionSolid(SolutionBase):
    """Solution in the particle solid phase of the ``UnitOperation``.

    Dimensions: NCOL * NRAD * sum_{j}^{NPARTYPE}{NBOUND,j * NPAR,j}

    """

    n_bound = UnsignedInteger()

    dimensions = SolutionBase.dimensions + [
        'axial_coordinates',
        'radial_coordinates',
        'particle_coordinates',
        'component_coordinates',
    ]

    def __init__(
            self,
            name,
            component_system, bound_states,
            time, solution,
            axial_coordinates=None,
            radial_coordinates=None,
            particle_coordinates=None):

        self.bound_states = bound_states

        self.axial_coordinates = axial_coordinates
        # Account for dimension reduction in case of only one cell (e.g. LRMP)
        if radial_coordinates is not None and len(radial_coordinates) == 1:
            radial_coordinates = None
        self.radial_coordinates = radial_coordinates
        # Account for dimension reduction in case of only one cell (e.g. CSTR)
        if particle_coordinates is not None and len(particle_coordinates) == 1:
            particle_coordinates = None
        self.particle_coordinates = particle_coordinates

        super().__init__(name, component_system, time, solution)

    @property
    def n_comp(self):
        """int: Number of components."""
        return sum(self.bound_states)

    @property
    def ncol(self):
        """int: Number of axial discretization points."""
        if self.axial_coordinates is None:
            return None
        else:
            return len(self.axial_coordinates)

    @property
    def nrad(self):
        """int: Number of radial discretization points."""
        if self.radial_coordinates is None:
            return
        return len(self.radial_coordinates)

    @property
    def npar(self):
        if self.particle_coordinates is None:
            return
        return len(self.particle_coordinates)

    @plotting.create_and_save_figure
    def plot(
            self,
            start=None,
            end=None,
            components=None,
            layout=None,
            y_max=None,
            ax=None,
            *args,
            **kwargs,
            ):
        """Plot the entire time_signal for each component.

        Parameters
        ----------
        start : float, optional
            Start time for plotting. The default is 0.
        end : float, optional
            End time for plotting.
        components : list, optional.
            List of components to be plotted. If None, all components are plotted.
        layout : plotting.Layout
            Plot layout options.
        y_max : float, optional
            Maximum value of y axis.
            If None, value is automatically deferred from solution.
        ax : Axes
            Axes to plot on.

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
        if not (self.ncol is None and self.nrad is None):
            raise CADETProcessError(
                "Solution has more than single dimension. "
                "Please use `plot_at_time`."
            )

        solution = slice_solution(
            self,
            components=components,
            use_total_concentration=False,
            use_total_concentration_components=False,
            start=start,
            end=end,
        )

        x = solution.time / 60

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = '$time~/~min$'
            layout.y_label = '$c~/~mM$'
            if start is not None:
                start /= 60
            if end is not None:
                end /= 60
            layout.x_lim = (start, end)
        if y_max is not None:
            layout.y_lim = (None, y_max)

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        return ax

    def _plot_1D(
            self,
            t,
            components=None,
            layout=None,
            ax=None,
            *args,
            **kwargs,
            ):
        """Plot bulk solution over space at given time.

        Parameters
        ----------
        t : float
            Time for plotting
            If t == -1, the final solution is plotted.
        components : list, optional.
            List of components to be plotted. If None, all components are plotted.
        layout : plotting.Layout
            Plot layout options.
        ax : Axes
            Axes to plot on.

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
            coordinates={'time': [t, t]},
        )

        x = self.axial_coordinates

        if layout is None:
            layout = plotting.Layout()
            layout.x_label = '$z~/~m$'
            layout.y_label = '$c~/~mM$'

        ax = _plot_solution_1D(ax, x, solution, layout, *args, **kwargs)

        plotting.add_text(ax, f'time = {t:.2f} s')

        return ax

    def _plot_2D(self, t, comp, state, vmax, ax=None):
        x = self.axial_coordinates
        y = self.particle_coordinates

        if not self.time[0] <= t <= self.time[-1]:
            raise ValueError("Time exceeds bounds.")
        t_i = np.where(t <= self.time)[0][0]
        c_i = comp*self.n_bound + state
        v = self.solution[t_i, :, :, c_i].transpose()

        if vmax is None:
            vmax = v.max()

        mesh = ax.pcolormesh(x, y, v, shading='gouraud', vmin=0, vmax=vmax)

        plotting.add_text(ax, f'time = {t:.2f} s')

        layout = plotting.Layout()
        layout.title = f'Solid phase concentration, comp={comp}, state={state}'
        layout.x_label = '$z~/~m$'
        layout.y_label = '$r~/~m$'
        layout.labels = self.component_system.labels[c_i]
        plotting.set_layout(ax, layout)

        return ax, mesh

    @plotting.create_and_save_figure
    def plot_at_time(self, t, comp=0, state=0, vmax=None, ax=None):
        """Plot particle solid solution over space at given time.

        Parameters
        ----------
        t : float
            time for plotting
        comp : int, optional
            component index
        state : int, optional
            bound state
        vmax : float, optional
            Maximum values for plotting.
        ax : Axes
            Axes to plot on.

        See Also
        --------
        CADETProcess.plotting
        """
        if self.npar is None:
            ax = self._plot_1D(ax, t, vmax)
        else:
            ax, mesh = self._plot_2D(ax, t, comp, vmax)
            plt.colorbar(mesh, ax)
        return ax


class SolutionVolume(SolutionBase):
    """Volume solution (of e.g. CSTR)."""

    @property
    def solution_shape(self):
        """tuple: (Expected) shape of the solution"""
        return (self.nt, 1)

    @plotting.create_and_save_figure
    def plot(self, start=None, end=None, ax=None, update_layout=True, **kwargs):
        """Plot the whole time_signal for each component.

        Parameters
        ----------
        start : float
            start time for plotting
        end : float
            end time for plotting
        ax : Axes
            Axes to plot on.

        See Also
        --------
        CADETProcess.plot
        """
        x = self.time / 60
        y = self.solution * 1000

        y_min = np.min(y)
        y_max = 1.1 * np.max(y)

        layout = plotting.Layout()
        layout.x_label = '$time~/~min$'
        layout.y_label = '$V~/~L$'
        if start is not None:
            start /= 60
        if end is not None:
            end /= 60
        layout.x_lim = (start, end)
        layout.y_lim = (y_min, y_max)
        ax.plot(x, y, **kwargs)

        if update_layout:
            plotting.set_layout(ax, layout)

        return ax


def _plot_solution_1D(
        ax,
        x,
        solution,
        layout=None,
        plot_species=False,
        plot_components=True,
        plot_total_concentration=False,
        alpha=1, hide_labels=False, hide_species_labels=False,
        secondary_axis=None, secondary_layout=None,
        show_legend=True,
        update_layout=True,
        ):

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

    for i, comp in enumerate(solution.component_system.components):
        color = next(ax._get_lines.prop_cycler)['color']
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
                x, y_comp,
                linestyle='-',
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

                if comp.n_species == 1 \
                        and not plot_total_concentration\
                        and not plot_components:
                    linestyle = '-'
                else:
                    linestyle = next(linestyle_iter)['linestyle']

                if secondary_axis is not None \
                        and secondary_axis.transform is not None \
                        and comp.name in secondary_axis.components:
                    y_spec = secondary_axis.transform(sol[..., species_index])
                    y_min_sec = min(min(y_spec), y_min_sec)
                    y_max_sec = max(max(y_spec), y_max_sec)
                else:
                    y_spec = sol[..., species_index]
                    y_min = min(min(y_spec), y_min)
                    y_max = max(max(y_spec), y_max)

                y_spec = np.squeeze(y_spec)

                a.plot(
                    x, y_spec,
                    linestyle=linestyle,
                    label=label,
                    color=color,
                    alpha=alpha
                )
                species_index += 1

    if plot_total_concentration:
        if hide_labels or not update_layout:
            label = None
        else:
            label = 'Total concentration'

        y_total = solution.total_concentration
        y_total = np.squeeze(y_total)
        y_min = min(min(y_total), y_min)
        y_max = max(max(y_total), y_max)
        a.plot(
            x, y_total, '-',
            label=label,
            color='k',
            alpha=alpha
        )

    if layout.y_lim is None:
        layout.y_lim = (y_min, 1.1*y_max)

    if secondary_axis is not None and secondary_layout is None:
        secondary_layout = plotting.Layout()
        secondary_layout.y_label = secondary_axis.y_label
        secondary_layout.y_lim = (y_min_sec, 1.1*y_max_sec)

    if update_layout:
        plotting.set_layout(
            ax,
            layout,
            show_legend,
            ax_secondary,
            secondary_layout,
        )

    return ax


class InterpolatedSignal():
    def __init__(self, time, signal):
        if len(signal.shape) == 1:
            signal = np.array(signal, ndmin=2).transpose()
        self._solutions = [
                PchipInterpolator(time, signal[:, comp])
                for comp in range(signal.shape[1])
                ]
        self._derivatives = [signal.derivative() for signal in self._solutions]
        self._antiderivatives = [signal.antiderivative() for signal in self._solutions]

    @property
    def solutions(self):
        return self._solutions

    @property
    def derivatives(self):
        return self._derivatives

    def derivative(self, time):
        """Derivatives of the solution spline(s) at given time points.

        Parameters
        ----------
        time : np.ndarray
            The time points to evaluate the derivatives at.

        Returns
        -------
        der : ndarray
            Derivatives of the solution spline(s) at given time.

        """
        der = np.empty((len(time), len(self.solutions)))
        for comp, der_i in enumerate(self.derivatives):
            der[:, comp] = der_i(time)

        return der

    @property
    def antiderivatives(self):
        return self._antiderivatives

    def antiderivative(self, time):
        """ Return all antiderivative of the spline(s) at given time.

        x : np.ndarray
            The time points to evaluate the antiderivatives at.

        Returns
        -------
        anti : ndarray
            Antiderivatives of the solution spline(s) at given time.

        """
        anti = np.empty((len(time), len(self.solutions)))
        for comp, anti_i in enumerate(self.antiderivatives):
            anti[:, comp] = anti_i(time)

        return anti

    def integral(self, start=None, end=None):
        """Definite integral between start and end.

        Parameters
        ----------
        start : float
            Lower integration bound.
        end : end
            Upper integration bound.

        Returns
        -------
        integral : np.ndarray
            Definite integral of the solution spline(s) between limits.

        """
        return np.array([
            self._solutions[comp].integral(start, end)
            for comp in range(len(self._solutions))
        ])

    def __call__(self, t):
        return np.array([
            self._solutions[comp](t) for comp in range(len(self._solutions))
        ]).transpose()


def slice_solution(
        solution_original,
        components=None,
        use_total_concentration=False,
        use_total_concentration_components=False,
        coordinates=None):
    """Slice a `Solution` object along specified dimensions, components or both.

    Parameters
    ----------
    solution_original : Solution
        The `Solution` object to slice.
    components : str or list of str, optional
        The names of the components to keep in the sliced `Solution`.
        If `None`, all components are kept. Defaults to `None`.
    use_total_concentration : bool, optional
        If `True`, only the total concentration data is kept in the sliced `Solution`.
        Defaults to `False`.
    use_total_concentration_components : bool, optional
        If `True`, the total concentration data is kept for each individual species
        of each component in the sliced `Solution`. Defaults to `False`.
    coordinates : dict, optional
        A dictionary mapping dimensions to slice coordinates. Each dimension in the
        `Solution` object is represented by a key in the dictionary, and the corresponding
        value is a tuple of two or three elements specifying the start, stop and step
        coordinates of the slice along that dimension. If a value is `None`, the corresponding
        coordinate is not sliced. Defaults to `None`.

    Returns
    -------
    Solution
        A new `Solution` object representing the sliced data.

    Raises
    ------
    ValueError
        If any of the slice coordinates exceeds the bounds of its corresponding dimension.
    CADETProcessError
        If any of the specified components or dimensions does not exist in the original `Solution`.
    """

    solution = copy.deepcopy(solution_original)

    slices = ()
    if coordinates is not None:
        coordinates = copy.deepcopy(coordinates)
        for i, (dim, coord) in enumerate(solution.coordinates.items()):
            if dim == 'component_coordinates':
                continue
            if dim in coordinates:
                sl = list(coordinates.pop(dim))
                if sl[0] is None:
                    sl[0] = coord[0]
                if sl[1] is None:
                    sl[1] = coord[-1]
                if not coord[0] <= sl[0] <= sl[1] <= coord[-1]:
                    raise ValueError(f"{dim} coordinates exceed bounds.")
                start_index = np.where(coord >= sl[0])[0][0]
                if sl[1] is not None:
                    end_index = np.where(coord >= sl[1])[0][0] + 1
                else:
                    end_index = None
                sl = slice(start_index, end_index)
                slices += (sl,)
                setattr(solution, dim, coord[sl])
            else:
                slices += (slice(None),)

        if len(coordinates) > 0:
            raise CADETProcessError(f"Unknown dimensions: {coordinates.keys()}")

        solution.solution = solution.solution[slices]

    if components is not None:
        if not isinstance(components, list):
            components = [components]
        components = copy.deepcopy(components)

        component_system = copy.deepcopy(solution.component_system)
        component_indices = []
        for i, (name, component) in enumerate(
                solution.component_system.components_dict.items()):
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

        solution.component_system = component_system
        solution.solution = solution.solution[..., component_indices]

    if use_total_concentration_components:
        sol = solution.total_concentration_components
        for comp in solution.component_system:
            if comp.n_species > 1:
                comp._species = []
                comp.add_species(comp.name)
        solution.solution = sol

    if use_total_concentration:
        sol = solution.total_concentration
        solution.component_system = ComponentSystem(['total_concentration'])
        solution.solution = sol

    solution.update()
    solution.update_transform()

    return solution
