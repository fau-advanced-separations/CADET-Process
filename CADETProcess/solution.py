"""Store and plot solution of simulation.

NCOL/N_Z: Number of axial cells
z: Axial position

NRAD/N_rho: Number of
rho: Annulus cell (cylinder shell)

NCOMP/N_C: Number of components
i: component index

NPARTYPE/N_P: Number of particle types
j: Particle type index

NPAR/N_R: Number of particle cells
r: Radial position (particle)

IO: NCOL * NRAD
Bulk/Interstitial: NCOL * NRAD * NCOMP
Particle_liquid: NCOL * NRAD * sum_{j}^{NPARTYPE}{NCOMP * NPAR,j}
Particle_solid: NCOL * NRAD * sum_{j}^{NPARTYPE}{NBOUND,j * NPAR,j}
Flux: NCOL * NRAD * NPARTYPE * NCOMP
"""

import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import integrate

from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    String, UnsignedInteger, Vector, DependentlySizedNdArray
)

from CADETProcess.processModel import ComponentSystem
from CADETProcess.dynamicEvents import TimeLine

from CADETProcess import plotting
from CADETProcess import CADETProcessError

from CADETProcess import smoothing
from CADETProcess import transform


class SolutionBase(metaclass=StructMeta):
    name = String()
    time = Vector()
    solution = DependentlySizedNdArray(dep='solution_shape')

    _coordinates = []

    def __init__(self, name, component_system, time, solution):
        self.name = name
        self.component_system_original = component_system
        self.time_original = time
        self.solution_original = solution

        self.reset()

    def reset(self):
        self.component_system = self.component_system_original
        self.time = self.time_original
        self.solution = self.solution_original

    @property
    def component_system(self):
        return self._component_system

    @component_system.setter
    def component_system(self, component_system):
        if not isinstance(component_system, ComponentSystem):
            raise TypeError('Expected ComponentSystem')
        self._component_system = component_system

    @property
    def n_comp(self):
        return self.component_system.n_comp

    @property
    def cycle_time(self):
        return self.time[-1]

    @property
    def nt(self):
        return len(self.time)

    @property
    def coordinates(self):
        coordinates = {}

        for c in self._coordinates:
            v = getattr(self, c)
            if v is None:
                continue
            coordinates[c] = v
        return coordinates

    @property
    def solution_shape(self):
        """tuple: (Expected) shape of the solution
        """
        coordinates = tuple(len(c) for c in self.coordinates.values())
        return (self.nt,) + coordinates + (self.n_comp,)

    @property
    def total_concentration_components(self):
        """np.array: Total concentration of components (sum of species)."""
        return component_concentration(self.component_system, self.solution)

    @property
    def total_concentration(self):
        """np.array: Total concentration all of components."""
        return total_concentration(self.component_system, self.solution)

    @property
    def local_purity_components(self):
        return purity(self.component_system, self.solution, sum_species=True)

    @property
    def local_purity_species(self):
        """np.array: Local purity of components."""
        return purity(self.component_system, self.solution, sum_species=False)


class SolutionIO(SolutionBase):
    """Solution at unit inlet or outlet.

    IO: NCOL * NRAD

    Notes
    -----
    The flow_rate is implemented as TimeLine to improve interpolation of
    signals with discontinuous flow.

    """

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
        if self.is_normalized:
            return

        self.solution = self.transform.transform(self.solution)
        self.update()
        self.is_normalized = True

    def denormalize(self):
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

    def integral(self, start=0, end=None):
        """Peak area in a fraction interval.

        Parameters
        ----------
        start : float
            Start time of the fraction

        end: float
            End time of the fraction

        Returns
        -------
        Area : np.array
            Mass of all components in the fraction

        """
        if end is None:
            end = self.cycle_time

        return self.solution_interpolated.integral(start, end)

    def fraction_mass(self, start=0, end=None):
        """Component mass in a fraction interval

        Parameters
        ----------
        start : float
            Start time of the fraction

        end: float
            End time of the fraction

        Returns
        -------
        fraction_mass : np.array
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

    def fraction_volume(self, start=0, end=None):
        """Volume of a fraction interval

        Parameters
        ----------
        start : float
            Start time of the fraction

        end: float
            End time of the fraction

        Returns
        -------
        fraction_volume : np.array
            Volume of the fraction

        """
        if end is None:
            end = self.cycle_time

        return float(self.flow_rate.integral(start, end))

    @plotting.create_and_save_figure
    def plot(
            self,
            start=0, end=None, y_max=None,
            layout=None,
            only_plot_components=False,
            alpha=1, hide_labels=False,
            secondary_axis=None, secondary_layout=None,
            show_legend=True,
            ax=None):
        """Plots the whole time_signal for each component.

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
            Axes object with concentration profile.

        See Also
        --------
        plotlib
        plot_purity
        """
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

        ax = _plot_solution_1D(
            self,
            layout=layout,
            only_plot_components=only_plot_components,
            alpha=alpha,
            hide_labels=hide_labels,
            secondary_axis=secondary_axis,
            secondary_layout=secondary_layout,
            show_legend=show_legend,
            ax=ax,
        )

        return ax

    @plotting.create_and_save_figure
    def plot_derivative(
            self,
            start=0, end=None, y_max=None,
            layout=None,
            only_plot_components=False,
            alpha=1, hide_labels=False,
            secondary_axis=None, secondary_layout=None,
            show_legend=True,
            ax=None):
        """Plots the whole time_signal for each component.

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
            Axes object with concentration profile.

        See Also
        --------
        plotlib
        plot_purity
        """
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

        import copy
        solution_derivative = copy.deepcopy(self)
        der_fun = self.solution_interpolated.derivative
        solution_derivative.solution = der_fun(self.time)
        solution_derivative.update()

        ax = _plot_solution_1D(
            solution_derivative,
            layout=layout,
            only_plot_components=only_plot_components,
            alpha=alpha,
            hide_labels=hide_labels,
            secondary_axis=secondary_axis,
            secondary_layout=secondary_layout,
            show_legend=show_legend,
            ax=ax,
        )

        return ax

    @plotting.create_and_save_figure
    def plot_purity(
            self,
            start=0, end=None, y_max=None,
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
            layout.x_label = '$time~/~min$'
            layout.y_label = '$Purity ~/~\%$'
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
    """Interstitial solution.

    Bulk/Interstitial: NCOL * NRAD * NCOMP
    """
    _coordinates = ['axial_coordinates', 'radial_coordinates']

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
        if self.axial_coordinates is None:
            return
        return len(self.axial_coordinates)

    @property
    def nrad(self):
        if self.radial_coordinates is None:
            return
        return len(self.radial_coordinates)

    @plotting.create_and_save_figure
    def plot(
            self,
            start=0, end=None, y_max=None,
            layout=None,
            only_plot_components=False,
            alpha=1, hide_labels=False,
            secondary_axis=None, secondary_layout=None,
            show_legend=True,
            ax=None):
        """Plots the whole time_signal for each component.

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
            Axes object with concentration profile.

        See Also
        --------
        plotlib
        plot_purity
        """
        if not (self.ncol is None and self.nrad is None):
            raise CADETProcessError(
                "Solution has more single dimension. Please use `plot_at_time`"
                "or `plot_at_location`."
            )

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

        ax = _plot_solution_1D(
            self,
            layout=layout,
            only_plot_components=only_plot_components,
            alpha=alpha,
            hide_labels=hide_labels,
            secondary_axis=secondary_axis,
            secondary_layout=secondary_layout,
            show_legend=show_legend,
            ax=ax,
        )

        return ax

    @plotting.create_and_save_figure
    def plot_at_time(self, t, overlay=None, y_min=None, y_max=None, ax=None):
        """Plot bulk solution over space at given time.

        Parameters
        ----------
        t : float
            Time for plotting
            If t == -1, the final solution is plotted.
        ax : Axes
            Axes to plot on.

        See Also
        --------
        plot_at_location
        CADETProcess.plotting
        """
        x = self.axial_coordinates

        if t == -1:
            t = self.time[-1]

        if not self.time[0] <= t <= self.time[-1]:
            raise ValueError("Time exceeds bounds.")
        t_i = np.where(t <= self.time)[0][0]

        y = self.solution[t_i, :]
        if y_max is None:
            y_max = 1.1*np.max(y)
        if y_min is None:
            y_min = min(0, np.min(y))

        ax.plot(x, y)

        plotting.add_text(ax, f'time = {t:.2f} s')

        if overlay is not None:
            y_max = np.max(overlay)
            plotting.add_overlay(ax, overlay)

        layout = plotting.Layout()
        layout.x_label = '$z~/~m$'
        layout.y_label = '$c~/~mM$'
        layout.y_lim = (y_min, y_max)
        plotting.set_layout(ax, layout)

        return ax

    @plotting.create_and_save_figure
    def plot_at_location(
            self, z, overlay=None, y_min=None, y_max=None, ax=None):
        """Plot bulk solution over time at given location.

        Parameters
        ----------
        z : float
            space for plotting
        ax : Axes
            Axes to plot on.

        See Also
        --------
        plot_at_time
        CADETProcess.plotting
        """
        x = self.time

        if not self.axial_coordinates[0] <= z <= self.axial_coordinates[-1]:
            raise ValueError("Axial coordinate exceets boundaries.")
        z_i = np.where(z <= self.axial_coordinates)[0][0]

        y = self.solution[:, z_i]
        if y_max is None:
            y_max = 1.1*np.max(y)
        if y_min is None:
            y_min = min(0, np.min(y))

        ax.plot(x, y)

        plotting.add_text(ax, f'z = {z:.2f} m')

        if overlay is not None:
            y_max = np.max(overlay)
            plotting.add_overlay(ax, overlay)

        layout = plotting.Layout()
        layout.x_label = '$time~/~min$'
        layout.y_label = '$c~/~mM$'
        layout.y_lim = (y_min, y_max)
        plotting.set_layout(ax, layout)

        return ax


class SolutionParticle(SolutionBase):
    """Mobile phase solution inside the particles.

    Particle_liquid: NCOL * NRAD * sum_{j}^{NPARTYPE}{NCOMP * NPAR,j}
    """

    _coordinates = [
        'axial_coordinates', 'radial_coordinates', 'particle_coordinates'
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
        if self.axial_coordinates is None:
            return
        return len(self.axial_coordinates)

    @property
    def nrad(self):
        if self.radial_coordinates is None:
            return
        return len(self.radial_coordinates)

    @property
    def npar(self):
        if self.particle_coordinates is None:
            return
        return len(self.particle_coordinates)

    def _plot_1D(self, t, ymax, ax=None):
        x = self.axial_coordinates

        if not self.time[0] <= t <= self.time[-1]:
            raise ValueError("Time exceeds bounds.")
        t_i = np.where(t <= self.time)[0][0]
        y = self.solution[t_i, :]

        if ymax is None:
            ymax = 1.1*np.max(y)

        ax.plot(x, y)

        plotting.add_text(ax, f'time = {t:.2f} s')

        layout = plotting.Layout()
        layout.x_label = '$z~/~m$'
        layout.y_label = '$c~/~mM$'
        layout.y_lim = (0, ymax)
        plotting.set_layout(ax, layout)

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
    """Solid phase solution inside the particles.

    Particle_solid: NCOL * NRAD * sum_{j}^{NPARTYPE}{NBOUND,j * NPAR,j}
    """

    n_bound = UnsignedInteger()

    _coordinates = [
        'axial_coordinates', 'radial_coordinates', 'particle_coordinates'
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
        return sum(self.bound_states)

    @property
    def ncol(self):
        if self.axial_coordinates is None:
            return None
        else:
            return len(self.axial_coordinates)

    @property
    def nrad(self):
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
            start=0, end=None, y_max=None,
            layout=None,
            only_plot_components=False,
            alpha=1, hide_labels=False,
            secondary_axis=None, secondary_layout=None,
            show_legend=True,
            ax=None):
        """Plots the whole time_signal for each component.

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
            Axes object with concentration profile.

        See Also
        --------
        plotlib
        plot_purity
        """
        if not (self.ncol is None and self.nrad is None):
            raise CADETProcessError(
                "Solution has more single dimension. "
                "Please use `plot_at_time`."
            )

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

        ax = _plot_solution_1D(
            self,
            layout=layout,
            only_plot_components=only_plot_components,
            alpha=alpha,
            hide_labels=hide_labels,
            secondary_axis=secondary_axis,
            secondary_layout=secondary_layout,
            show_legend=show_legend,
            ax=ax,
        )

        return ax

    def _plot_1D(self, t, y_min=None, y_max=None, ax=None):
        x = self.axial_coordinates

        if not self.time[0] <= t <= self.time[-1]:
            raise ValueError("Time exceeds bounds.")
        t_i = np.where(t <= self.time)[0][0]
        y = self.solution[t_i, :]

        if y_max is None:
            y_max = 1.1*np.max(y)
        if y_min is None:
            y_min = min(0, np.min(y))

        ax.plot(x, y)

        plotting.add_text(ax, f'time = {t:.2f} s')

        layout = plotting.Layout()
        layout.x_label = '$z~/~m$'
        layout.y_label = '$c~/~mM$'
        layout.labels = self.component_system.labels
        layout.y_lim = (y_min, y_max)
        plotting.set_layout(ax, layout)

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
    def plot(self, start=0, end=None, overlay=None, ax=None):
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
        y_max = np.max(y)

        ax.plot(x, y)

        if overlay is not None:
            y_max = 1.1*np.max(overlay)
            plotting.add_overlay(ax, overlay)

        layout = plotting.Layout()
        layout.x_label = '$time~/~min$'
        layout.y_label = '$V~/~L$'
        if start is not None:
            start /= 60
        if end is not None:
            end /= 60
        layout.x_lim = (start, end)
        layout.y_lim = (y_min, y_max)
        plotting.set_layout(ax, layout)

        return ax


def _plot_solution_1D(
        solution,
        layout=None,
        only_plot_components=False,
        alpha=1, hide_labels=False, hide_species_labels=True,
        secondary_axis=None, secondary_layout=None,
        show_legend=True,
        ax=None):

    time = solution.time / 60
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
        if hide_labels:
            label = None
        else:
            label = comp.name

        if secondary_axis is not None \
                and i in secondary_axis.component_indices:
            a = ax_secondary
        else:
            a = ax

        if secondary_axis is not None \
                and secondary_axis.transform is not None \
                and i in secondary_axis.component_indices:
            y = secondary_axis.transform(c_total_comp[..., i])
        else:
            y = c_total_comp[..., i]

        if secondary_axis is not None \
                and i in secondary_axis.component_indices:
            y_min_sec = min(min(y), y_min_sec)
            y_max_sec = max(max(y), y_max_sec)
        else:
            y_min = min(min(y), y_min)
            y_max = max(max(y), y_max)

        a.plot(
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
                if hide_species_labels:
                    label = None
                else:
                    label = s
                if secondary_axis is not None \
                        and i in secondary_axis.component_indices:
                    a = ax_secondary
                else:
                    a = ax

                if secondary_axis is not None \
                        and secondary_axis.transform is not None \
                        and i in secondary_axis.component_indices:
                    y = secondary_axis.transform(sol[..., species_index])
                else:
                    y = sol[..., species_index]

                a.plot(
                    time, y, '--',
                    label=label,
                    color=color,
                    alpha=alpha
                )
                species_index += 1
    if layout.y_lim is None:
        layout.y_lim = (y_min, 1.1*y_max)

    if secondary_axis is not None and secondary_layout is None:
        secondary_layout = plotting.Layout()
        secondary_layout.y_label = secondary_axis.y_label
        secondary_layout.y_lim = (y_min_sec, 1.1*y_max_sec)

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
        time : np.array
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

        x : np.array
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

    def integral(self, start=0, end=-1):
        """Definite integral between start and end.

        Parameters
        ----------
        start : float
            Lower integration bound.
        end : end
            Upper integration bound.

        Returns
        -------
        integral : np.array
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


def purity(
        component_system, solution,
        sum_species=True, min_value=1e-6):
    """Local purity profile of solution.

    To exclude components from purity calculation, refer to ComponentSystem.

    Parameters
    ----------
    component_system : ComponentSystem
        DESCRIPTION.
    solution : np.array
        Array containing concentrations.
    sum_species : bool, optional
        DESCRIPTION. The default is True.
    min_value : float, optional
        Minimum concentration to be considered. Everything below min_value
        will be considered as 0. The default is 1e-6.

    Returns
    -------
    purity
        Local purity of components/species.
        If sum_species, shape is nt * n_components, nt * n_comp otherwise.
        Excluding components does not affect the shape.

    See Also
    --------
    ComponentSystem
    """
    c_total = total_concentration(
        component_system, solution,
        exclude=component_system.exclude_from_purity
    )

    if sum_species:
        solution = component_concentration(component_system, solution)

    exclude_indices = []
    for i, comp in enumerate(component_system):
        if comp.exclude_from_purity:
            if sum_species:
                exclude_indices.append(i)
            else:
                exclude_indices += component_system.indices[comp.name]

    purity = np.zeros(solution.shape)

    c_total[c_total < min_value] = np.nan

    for i, s in enumerate(solution.transpose()):
        if i in exclude_indices:
            continue

        with np.errstate(divide='ignore', invalid='ignore'):
            purity[:, i] = np.divide(s, c_total)

    purity = np.nan_to_num(purity)

    return purity


def component_concentration(component_system, solution):
    """Compute total concentration of components by summing up species.

    Parameters
    ----------
    component_system : ComponentSystem
        ComponentSystem containing information about subspecies.
    solution : np.array
        Solution array.

    Returns
    -------
    component_concentration : np.array
        Total concentration of components.

    """
    component_concentration = np.zeros(
        solution.shape[0:-1] + (component_system.n_components,)
    )

    counter = 0
    for index, comp in enumerate(component_system):
        comp_indices = slice(counter, counter+comp.n_species)
        c_total = np.sum(solution[..., comp_indices], axis=1)
        component_concentration[:, index] = c_total
        counter += comp.n_species

    return component_concentration


def total_concentration(component_system, solution, exclude=None):
    """Compute total concentration of all components.

    Parameters
    ----------
    component_system : ComponentSystem
        ComponentSystem containing information about subspecies.
    solution : np.array
        Solution array.
    Exclude : list
        Component names to be excluded from total concentration.

    Returns
    -------
    total_concentration : np.array
        Total concentration of all components.

    """
    if exclude is None:
        exclude = []

    total_concentration = np.zeros(solution.shape[0:-1])

    counter = 0
    for index, comp in enumerate(component_system):
        if comp.name not in exclude:

            comp_indices = slice(counter, counter+comp.n_species)
            c_total = np.sum(solution[..., comp_indices], axis=1)

            total_concentration += c_total

        counter += comp.n_species

    return total_concentration
