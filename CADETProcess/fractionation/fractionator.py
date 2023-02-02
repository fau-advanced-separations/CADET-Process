from functools import wraps
import os

from addict import Dict
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess import settings
from CADETProcess.dataStructure import String
from CADETProcess.dynamicEvents import EventHandler
from CADETProcess import plotting
from CADETProcess.performance import Performance
from CADETProcess.solution import slice_solution
from CADETProcess import SimulationResults

from CADETProcess.fractionation.fractions import Fraction, FractionPool


class Fractionator(EventHandler):
    """Class for Chromatogram Fractionation.

    To set Events for starting and ending a fractionation it inherits from the
    EventHandler class. It defines a ranking list for components as a
    DependentlySizedUnsignedList with the number of components as dependent
    size. The time_signal to fractionate is defined. If no ranking is set,
    every component is equivalently.

    """

    name = String(default='Fractionator')
    performance_keys = [
        'mass', 'concentration', 'purity', 'recovery',
        'productivity', 'eluent_consumption'
    ]

    def __init__(
            self,
            simulation_results,
            components=None,
            use_total_concentration_components=True,
            *args, **kwargs):

        self.components = components
        self.use_total_concentration_components = use_total_concentration_components
        self.simulation_results = simulation_results
        self._cycle_time = None

        super().__init__(*args, **kwargs)

    @property
    def simulation_results(self):
        return self._simulation_results

    @simulation_results.setter
    def simulation_results(self, simulation_results):
        if not isinstance(simulation_results, SimulationResults):
            raise TypeError('Expected SimulationResults')

        if len(simulation_results.chromatograms) == 0:
            raise CADETProcessError(
                'Simulation results do not contain chromatogram'
            )

        self._simulation_results = simulation_results

        self._chromatograms = [
            slice_solution(
                chrom,
                components=self.components,
                use_total_concentration_components=self.use_total_concentration_components
            )
            for chrom in simulation_results.chromatograms
        ]

        m_feed = np.zeros((self.component_system.n_comp, ))
        counter = 0
        for comp, indices in simulation_results.component_system.indices.items():
            if comp in self.component_system.names:
                m_feed_comp = simulation_results.process.m_feed[indices]
                if self.use_total_concentration_components:
                    m_feed[counter] = np.sum(m_feed_comp)
                    counter += 1
                else:
                    n_species = len(indices)
                    m_feed[counter:counter+n_species] = m_feed_comp
                    counter += n_species
        self.m_feed = m_feed

        self._fractionation_states = Dict({
            chrom: []
            for chrom in self.chromatograms
        })
        self._chromatogram_events = Dict({
            chrom: []
            for chrom in self.chromatograms
        })

        self.reset()

    @property
    def component_system(self):
        return self.chromatograms[0].component_system

    def call_by_chrom_name(func):
        @wraps(func)
        def wrapper(self, chrom, *args, **kwargs):
            """Enable calling functions with chromatogram object or name."""
            if isinstance(chrom, str):
                try:
                    chrom = self.chromatograms_dict[chrom]
                except KeyError:
                    raise CADETProcessError('Not a valid unit')
            return func(self, chrom, *args, **kwargs)

        return wrapper

    @property
    def chromatograms(self):
        """list: Chromatograms to be fractionized.

        See Also
        --------
        add_chromatogram
        SoltionIO
        reset
        cycle_time
        """
        return self._chromatograms

    @property
    def chromatograms_dict(self):
        """dict: Chromatogram names and objects."""
        return {chrom.name: chrom for chrom in self.chromatograms}

    @property
    def chromatogram_names(self):
        """list: Chromatogram names"""
        return [chrom.name for chrom in self.chromatograms]

    @property
    def n_chromatograms(self):
        """int: Number of Chromatograms Fractionator."""
        return len(self.chromatograms)

    @property
    def chromatogram_events(self):
        chrom_events = {
            chrom: sorted(events, key=lambda evt: evt.time)
            for chrom, events in self._chromatogram_events.items()
        }

        return chrom_events

    @property
    def process(self):
        return self.simulation_results.process

    @property
    def n_comp(self):
        """int: Number of components to be fractionized"""
        return self.chromatograms[0].n_comp

    @property
    def cycle_time(self):
        """float: cycle time"""
        if self._cycle_time is None:
            return self.process.cycle_time
        return self._cycle_time

    @cycle_time.setter
    def cycle_time(self, cycle_time):
        self._cycle_time = cycle_time

    @property
    def time(self):
        """np.ndarray: solution times of Chromatogram."""
        return self.chromatograms[0].time

    @plotting.create_and_save_figure
    def plot_fraction_signal(
            self, chromatogram=None, ax=None, *args, **kwargs):
        """Plot the signal without the waste fractions.

        Parameters
        ----------
        chromatogram : SolutionIO, optional
            Chromatogram to be plotted. If None, the first one is plotted.
        ax : Axes
            Axes to plot on.

        Returns
        -------
        ax : Axes
            Axes with plot of parameter state.

        See Also
        --------
        CADETProcess.plot
        plot_purity

        """
        if chromatogram is None:
            chromatogram = \
                list(self.performer_timelines['fractionation_states'].keys())[0]
        if isinstance(chromatogram, str):
            chromatogram = self.chromatograms_dict[chromatogram]

        time_line = \
            self.performer_timelines['fractionation_states'][chromatogram.name]

        try:
            start = kwargs['start']
        except KeyError:
            start = 0
        try:
            end = kwargs['end']
        except KeyError:
            end = np.max(chromatogram.time)

        _,  ax = chromatogram.plot(show=False, ax=ax, *args, **kwargs)

        y_max = 1.1*np.max(chromatogram.solution)

        fill_regions = []
        for sec in time_line.sections:
            comp_index = int(np.where(sec.coeffs)[0])
            if comp_index == self.n_comp:
                color_index = -1
                text = 'W'
            else:
                color_index = comp_index
                text = str(comp_index + 1)

            if sec.start != sec.end:
                fill_regions.append(plotting.FillRegion(
                    start=sec.start/60,
                    end=sec.end/60,
                    y_max=y_max,
                    color_index=color_index,
                    text=text
                    )
                )

        if len(time_line.sections) == 0:
            fill_regions.append(plotting.FillRegion(
                start=sec.start/60,
                end=sec.end/60,
                y_max=y_max,
                color_index=-1,
                text='W'
                )
            )

        plotting.add_fill_regions(ax, fill_regions, (start, end))

        return ax

    @property
    def fractionation_states(self):
        """dict: Fractionation state of Chromatograms.

        Notes
        -----
            This is just a dummy variable to support interfacing with Events.

        """
        return self._fractionation_states

    @call_by_chrom_name
    def set_fractionation_state(self, chrom, state):
        """Set fractionation states of Chromatogram.

        Parameters
        ----------
        chrom : SoltionIO
            Chromatogram object which is to be fractionated.
        state : int or list of floats
            New fractionation state of the Chromatogram.

        Raises
        ------
        CADETProcessError
            If Chromatogram not in Fractionator
            If state is integer and the state >= the n_comp.
            If the length of the states is unequal the state_length.
            If the sum of the states is not equal to 1.

        """
        if chrom not in self.chromatograms:
            raise CADETProcessError('Chromatogram not in Fractionator')

        state_length = self.n_comp + 1

        if state_length == 0:
            fractionation_state = []

        if type(state) is int:
            if state >= state_length:
                raise CADETProcessError('Index exceeds fractionation states')

            fractionation_state = [0] * state_length
            fractionation_state[state] = 1
        else:
            if len(state) != state_length:
                raise CADETProcessError(f'Expected length {state_length}.')

            elif sum(state) != 1:
                raise CADETProcessError('Sum of fractions must be 1')

            fractionation_state = state

        self._fractionation_states[chrom] = fractionation_state

    @property
    def n_fractions_per_pool(self):
        """list: number of fractions per pool."""
        return [pool.n_fractions for pool in self.fraction_pools]

    @property
    def fraction_pools(self):
        """List of the component and waste fraction pools.

        For every event, the end time is determined and a Fraction object is
        created which holds information about start and end time, as well as
        the mass and the volume of the fraction. The fractions are pooled
        depending on the event state.

        Returns
        -------
        fraction_pools : list
            List with fraction pools.

        """
        if self._fraction_pools is None:
            self._fraction_pools = [
                FractionPool(self.n_comp) for _ in range(self.n_comp + 1)
            ]

            for chrom_index, chrom in enumerate(self.chromatograms):
                chrom_events = self.chromatogram_events[chrom]
                for evt_index, evt in enumerate(chrom_events):
                    target = int(np.nonzero(evt.state)[0])

                    frac_start = evt.time

                    if evt_index < len(chrom_events) - 1:
                        frac_end = chrom_events[evt_index + 1].time
                        fraction = self._create_fraction(
                            chrom_index, frac_start, frac_end
                        )
                        self.add_fraction(fraction, target)
                    else:
                        frac_end = self.cycle_time
                        fraction = self._create_fraction(
                            chrom_index, frac_start, frac_end
                        )
                        self.add_fraction(fraction, target)

                        frac_start = 0
                        frac_end = chrom_events[0].time
                        fraction = self._create_fraction(
                            chrom_index, frac_start, frac_end
                        )
                        self.add_fraction(fraction, target)

        return self._fraction_pools

    def _create_fraction(self, chrom_index, start, end):
        """Helper function to create Fraction object calculate mass.

        Parameters
        ----------
        start : float
            start time of the fraction
        start : float
            start time of the fraction

        Returns
        -------
        fraction : Fraction
            Chromatogram fraction

        """
        mass = self.chromatograms[chrom_index].fraction_mass(start, end)
        volume = self.chromatograms[chrom_index].fraction_volume(start, end)
        return Fraction(mass, volume)

    def add_fraction(self, fraction, target):
        """Add Fraction to the FractionPool of target component.

        Notes
        -----
            Waste is always the last fraction

        """
        if not isinstance(fraction, Fraction):
            raise TypeError('Expected Fraction')

        if target not in range(self.n_comp + 1):
            raise CADETProcessError('Not a valid target')
        self._fraction_pools[target].add_fraction(fraction)

    @property
    def mass(self):
        """ndarray: Component mass in corresponding fraction pools."""
        if self._mass is None:
            self._mass = np.array([
                pool.mass[comp]
                for comp, pool in enumerate(self.fraction_pools[:-1])
            ])
        return self._mass

    @property
    def concentration(self):
        """ndarray: Component concentration in corresponding fraction pool."""
        return np.array([
            pool.concentration[comp]
            for comp, pool in enumerate(self.fraction_pools[:-1])
        ])

    @property
    def purity(self):
        """ndarray: Component purity in corresponding fraction pool."""
        return np.array([
            pool.purity[comp]
            for comp, pool in enumerate(self.fraction_pools[:-1])
        ])

    @property
    def recovery(self):
        """ndarray: Component recovery yield in corresponding fraction pool."""
        with np.errstate(divide='ignore', invalid='ignore'):
            recovery = self.mass / self.m_feed

        return np.nan_to_num(recovery)

    @property
    def productivity(self):
        """ndarray: Specific productivity in corresponding fraction pool."""
        return self.mass / (
            self.cycle_time * self.process.V_solid
        )

    @property
    def eluent_consumption(self):
        """ndarray: Specific eluent consumption in corresponding fraction pool.

        Notes
        -----
            This is the inverse of the regularly used specific eluent
            consumption. It is preferred here in order to avoid numeric issues
            if the collected mass is 0.
        """
        return self.mass / self.process.V_eluent

    @property
    def performance(self):
        self.reset()
        return Performance(
            self.mass, self.concentration, self.purity,
            self.recovery, self.productivity, self.eluent_consumption,
            self.component_system
        )

    def reset(self):
        """Reset the results when fractionation times are changed."""
        self._fractionation_state = None
        self._fraction_pools = None
        self._mass = None

    def add_fractionation_event(
            self, event_name, target, time, chromatogram=None):
        if chromatogram is None and self.n_chromatograms == 1:
            chromatogram = self.chromatograms[0]
        elif isinstance(chromatogram, str):
            try:
                chromatogram = self.chromatograms_dict[f"{chromatogram}"]
            except KeyError:
                raise CADETProcessError("Could not find chromatogram.")
        else:
            raise CADETProcessError("Expected chromatogram.")

        param_path = f'fractionation_states.{chromatogram.name}'
        evt = self.add_event(
            event_name, param_path, target, time
        )
        self._chromatogram_events[chromatogram].append(evt)

        self.reset()

    def initial_values(self, purity_required=0.95):
        """Create events from chromatogram with minimum purity.

        Function creates fractions for areas in the chromatogram, where the
        local purity profile is higher than the purity required.

        Parameters
        ----------
        purity_required : float or list of floats
            Minimum purity required for the components in the fractionation

        Raises
        ------
        ValueError
            If size of purity parameter does not math number of components
        """
        if isinstance(purity_required, float):
            purity_required = [purity_required]*self.n_comp
        elif len(purity_required) != self.n_comp:
            raise ValueError(
                f'Expected purity array with size {self.n_comp}'
            )

        self._events = []
        self._chromatogram_events = Dict({
            chrom: [] for chrom in self.chromatograms
        })
        self.reset()

        for chrom_index, chrom in enumerate(self.chromatograms):
            purity_min = np.zeros(chrom.solution.shape)
            purity_min[chrom.local_purity_components > purity_required] = 1
            diff = np.vstack((
                purity_min[0, :] - purity_min[-1, :],
                np.diff(purity_min, axis=0))
            )

            for comp in range(self.n_comp):
                if purity_required[comp] > 0:
                    on_indices = np.where(diff[:, comp] == 1)
                    on_indices = on_indices[0]
                    for index, on_evt in enumerate(on_indices):
                        time = chrom.time[int(on_evt)]
                        event_name = \
                            'chrom_' + str(chrom_index) + \
                            '_comp_' + str(comp) + \
                            '_start_' + str(index)
                        param_path = f'fractionation_states.{chrom.name}'
                        evt = self.add_event(
                            event_name, param_path, comp, time
                        )
                        self._chromatogram_events[chrom].append(evt)

                    off_indices = np.where(diff[:, comp] == -1)
                    off_indices = off_indices[0]
                    for index, off_evt in enumerate(off_indices):
                        time = chrom.time[int(off_evt)]
                        event_name = \
                            'chrom_' + str(chrom_index) + \
                            '_comp_' + str(comp) + \
                            '_end_' + str(index)
                        param_path = f'fractionation_states.{chrom.name}'
                        evt = self.add_event(
                            event_name, param_path, self.n_comp, time
                        )
                        self._chromatogram_events[chrom].append(evt)

    @property
    def parameters(self):
        parameters = super().parameters
        parameters['fractionation_states'] = {
            chrom.name: self.fractionation_states[chrom]
            for chrom in self.chromatograms
        }

        return Dict(parameters)

    @parameters.setter
    def parameters(self, parameters):
        try:
            fractionation_states = parameters.pop('fractionation_states')
            for chrom, state in fractionation_states.items():
                self.set_fractionation_state(chrom, state)
        except KeyError:
            pass

        super(Fractionator, self.__class__).parameters.fset(self, parameters)

    @property
    def section_dependent_parameters(self):
        return self.parameters

    def save(self, case_dir, start=0, end=None):
        path = os.path.join(settings.working_directory, case_dir)

        for index, chrom in enumerate(self.chromatograms):
            chrom.plot(save_path=path + f'/chrom_{index}.png')
            chrom.plot_purity(
                start=start, end=end, save_path=path + '/chrom_purity.png'
            )

        for chrom in enumerate(self.chromatograms):
            self.plot_fraction_signal(
                chromatogram=chrom,
                start=start, end=end,
                save_path=path + f'/fractionation_signal_{index}.png',
                index=index
            )

    def __str__(self):
        return self.__class__.__name__
