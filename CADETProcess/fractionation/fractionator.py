import os

from addict import Dict
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.common import settings
from CADETProcess.dataStructure import String
from CADETProcess.dynamicEvents import EventHandler
from CADETProcess.common import plotting
from CADETProcess.common import Performance
from CADETProcess.common import Chromatogram
from CADETProcess.common import ProcessMeta

from CADETProcess.fractionation.fractions import Fraction, FractionPool

class Fractionator(EventHandler):
    """Class for Chromatogram Fractionation.

    To set Events for starting and ending a fractionation it inherits from the
    EventHandler class. It defines a ranking list for components as a
    DependentlySizedUnsignedList with the number of components as dependent
    size. The time_signal to fractionate is defined. If no ranking is set,
    every component is equivalently.

    Attributes
    ----------
    chromatogram : Chromatogram
        Object of the class TimeSignal, array with the concentration over time
        for a simulated process.
    """
    
    name = String(default='Fractionator')
    performance_keys = [
        'mass', 'concentration', 'purity', 'recovery',
        'productivity', 'eluent_consumption'
    ]

    def __init__(self, process_meta, *args, **kwargs):
        self.process_meta = process_meta
        self._chromatograms = []
        self._fractionation_states = Dict()
        self._chromatogram_events = Dict()
        self.reset()

        super().__init__(*args, **kwargs)

    def _chrom_name_decorator(func):
        def wrapper(self, chrom, *args, **kwargs):
            """Enable calling functions with chromatogram object or name.
            """
            if isinstance(chrom, str):
                try:
                    chrom = self.chromatograms_dict[chrom]
                except KeyError:
                    raise CADETProcessError('Not a valid unit')
            return func(self, chrom, *args, **kwargs)

        return wrapper

    @property
    def chromatograms(self):
        """Chromatogram to be fractionized.

        After setting, the cycle time is set and the results are reset.

        Parameters
        ----------
        chrmatogram : Chromatogram
            Chromatogram to be fractionized.

        Raises
        ------
        CADETProcessError
            If the chromatogram is not a Chromatogram instance

        Returns
        -------
        chromatogram : Chromatogram
            Chromatogram to be fractionized

        See also
        ---------
        Chromatogram
        reset
        cycle_time
        """
        return self._chromatograms

    @property
    def chromatograms_dict(self):
        """dict: Chromatogram names and objects.
        """
        return {chrom.name: chrom for chrom in self.chromatograms}

    @property
    def chromatogram_names(self):
        """list: Chromatogram names
        """
        return [chrom.name for chrom in self.chromatograms]

    @property
    def number_of_chromatograms(self):
        """int: Number of Chromatograms Fractionator.
        """
        return len(self.chromatograms)

    @property
    def chromatogram_events(self):
        chrom_events = {
            chrom: sorted(events, key=lambda evt: evt.time)
            for chrom, events in self._chromatogram_events.items()
            }

        return chrom_events

    def add_chromatogram(self, chromatogram):
        """Add Chromatogram to list of chromatograms to be fractionized.

        Parameters
        ----------
        chromatogram : Chromatogram
            Chromatogram object to be added to the Fractionator.

        Raises
        ------
        TypeError
            If unit is no instance of Chromatogram.
        CADETProcessError
            If unit already exists in flow sheet.
            If n_comp does not match with other Chromatograms.
            If cycle times does not match with other Chromatograms.
        """
        if not isinstance(chromatogram, Chromatogram):
            raise TypeError('Expected Chromatogram')
        if len(self._chromatograms) > 0:
            if chromatogram.n_comp != self._chromatograms[0].n_comp:
                raise CADETProcessError('Number of components does not match.')
            if chromatogram.cycle_time != self._chromatograms[0].cycle_time:
                raise CADETProcessError('Cycle_time does not match')
        self._chromatograms.append(chromatogram)
        self._fractionation_states[chromatogram] = []
        self._chromatogram_events[chromatogram] = []

        self.reset()

    @property
    def process_meta(self):
        return self._process_meta

    @process_meta.setter
    def process_meta(self, process_meta):
        if not isinstance(process_meta, ProcessMeta):
            raise TypeError('Expected ProcessMeta')
        self._process_meta = process_meta

    @property
    def n_comp(self):
        """Shortcut property to Chromatogram.n_comp.

        Returns
        --------
        n_comp : int
            Number of components to be fractionized
        """
        return self.chromatograms[0].n_comp

    @property
    def cycle_time(self):
        """Shortcut property to Chromatogram.cycle_time.

        Returns
        --------
        cycle_time : ndarray
            Cycle time of process.
        """
        return self.chromatograms[0].cycle_time

    @property
    def time(self):
        """Shortcut property to Chromatogram time vector.

        Returns
        --------
        time : NdArray
            Sets the time from TimeSignal object.
        """
        return self.chromatograms[0].time

    @plotting.save_fig
    def plot_fraction_signal(
            self, start=0, end=None, index=0, secondary_axis=None
            ):
        """Plot the signal without the waste fractions.

        Parameters
        ----------
        start : float, optional
            Start time of the plot. The default is 0.
        end : TYPE, optional
            End time of the plot. The default is None.
        index : int, optional
            Chromatogram index. The default is 0.

        Returns
        -------
        fig, axs.

        See also
        --------
        CADETProcess.plot
        plot_purity
        """
        chrom = self.chromatograms[index]
        time_line = self.performer_timelines['fractionation_states'][chrom.name]
        x = chrom.time/60
        y = chrom.signal
        ymax = 1.1*np.max(chrom.signal)

        fig, ax = plotting.setup_figure()

        ax.plot(x,y)

        fill_regions = []
        for sec in time_line.sections:
            comp_index = int(np.where(sec.coeffs)[0])
            if comp_index == self.n_comp:
                color_index=-1
                text = 'W'
            else:
                color_index = comp_index
                text = str(comp_index + 1)

            if sec.start != sec.end:
                fill_regions.append(plotting.FillRegion(
                    start=sec.start/60,
                    end=sec.end/60,
                    y_max=ymax,
                    color_index=color_index,
                    text=text
                    )
                )
        if end is None:
            end = np.max(x)

        if len(time_line.sections) == 0:
            fill_regions.append(plotting.FillRegion(
                start=sec.start/60,
                end=sec.end/60,
                y_max=ymax,
                color_index=-1,
                text='W'
                )
            )

        plotting.add_fill_regions(ax, fill_regions, (start, end))

        layout = plotting.Layout()
        layout.x_label = '$time~/~min$'
        layout.y_label = '$c~/~mM$'
        layout.xlim = (start, end)
        layout.ylim = (0, ymax)

        plotting.set_layout(fig, ax, layout)

        return fig, ax

    @property
    def fractionation_states(self):
        """dict: Fractionation state of Chromatograms.
        Notes
        -----
        This is just a dummy variable to support interfacing with Events.
        """
        return self._fractionation_states

    @_chrom_name_decorator
    def set_fractionation_state(self, chrom, state):
        """Set fractionation states of Chromatogram.

        Parameters
        ----------
        chrom : Chromatogram
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
        if chrom not in self._chromatograms:
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
                raise CADETProcessError('Expected length {}.'.format(state_length))

            elif sum(state) != 1:
                raise CADETProcessError('Sum of fractions must be 1')

            fractionation_state = state

        self._fractionation_states[chrom] = fractionation_state

    @property
    def fraction_pools(self):
        """Returns a list of the component and waste fraction pools.

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
        """Helper function to create Fraction object calculate mass

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
        """
        Waste is the last fraction
        """
        if not isinstance(fraction, Fraction):
            raise TypeError('Expected Fraction')

        if target not in range(self.n_comp + 1):
            raise CADETProcessError('Not a valid target')
        self._fraction_pools[target].add_fraction(fraction)


    @property
    def mass(self):
        """ndarray: Collected component mass in corresponding fraction pools.
        """
        if self._mass is None:
            self._mass = np.array(
                [pool.mass[comp]
                 for comp, pool in enumerate(self.fraction_pools[:-1])]
            )
        return self._mass

    @property
    def concentration(self):
        """ndarray: Component concentration in corresponding fraction pool.
        """
        return np.array(
            [pool.concentration[comp]
             for comp, pool in enumerate(self.fraction_pools[:-1])]
        )

    @property
    def purity(self):
        """ndarray: Component purity in corresponding fraction pool.
        """
        return np.array(
            [pool.purity[comp]
             for comp, pool in enumerate(self.fraction_pools[:-1])]
        )

    @property
    def recovery(self):
        """ndarray: Component recovery yield in corresponding fraction pool.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            recovery = self.mass / self.process_meta.m_feed

        return np.nan_to_num(recovery)

    @property
    def productivity(self):
        """ndarray: Specific productivity for components in corresponding
        fraction pool.
        """
        return self.mass / (
            self.process_meta.cycle_time * self.process_meta.V_solid
        )

    @property
    def eluent_consumption(self):
        """ndarray: Component mass per unit volume of eluent in corresponding
        fraction pool.

        Note
        ----
        This is the inverse of the regularly used specific eluent consumption
        It is preferred in order to avoid numeric issues if collected mass is 0.
        """
        return self.mass / self.process_meta.V_eluent

    @property
    def performance(self):
        self.reset()
        return Performance(
            self.mass, self.concentration, self.purity,
            self.recovery, self.productivity, self.eluent_consumption
        )

    def reset(self):
        """Resets the results when fractionation times are changed.
        """
        self._fractionation_state = None
        self._fraction_pools = None
        self._mass = None


    def initial_values(self, purity_required=0.95):
        """Create events from chromatogram with minimum purity.

        Function creates fractions for areas in the chromatogram, where the
        local purity profile is higher than the purity required.

        Parameters
        -----------
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
            raise ValueError('Expected array with size {}'.format(
                    self.chromatogram.n_comp))

        self._events = []
        self._chromatogram_events = Dict({
            chrom: [] for chrom in self.chromatograms
        })
        self.reset()

        for chrom_index, chrom in enumerate(self.chromatograms):
            purity_min = np.zeros(chrom.signal.shape)
            purity_min[chrom.local_purity > purity_required] = 1
            diff = np.vstack(
                (purity_min[0,:] - purity_min[-1,:], np.diff(purity_min, axis=0))
            )

            for comp in range(self.n_comp):
                if purity_required[comp] > 0:
                    on_indices = np.where(diff[:,comp] == 1)
                    on_indices = on_indices[0]
                    for index, on_evt in enumerate(on_indices):
                        time = chrom.time[int(on_evt)]
                        event_name = \
                            'chrom_' + str(chrom_index) + \
                            '_comp_' + str(comp) + \
                            '_start_' + str(index)
                        param_path = 'fractionation_states.{}'.format(chrom.name)
                        evt = self.add_event(
                            event_name, param_path, comp, time
                        )
                        self._chromatogram_events[chrom].append(evt)

                    off_indices = np.where(diff[:,comp] == -1)
                    off_indices = off_indices[0]
                    for index, off_evt in enumerate(off_indices):
                        time = chrom.time[int(off_evt)]
                        event_name = \
                            'chrom_' + str(chrom_index) + \
                            '_comp_' + str(comp) + \
                            '_end_' + str(index)
                        param_path = 'fractionation_states.{}'.format(chrom.name)
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
        path = os.path.join(settings.project_directory, case_dir)

        for index, chrom in enumerate(self.chromatograms):
            chrom.plot(save_path=path + '/chrom_{}.png'.format(index))
            chrom.plot_purity(
                start=start, end=end, save_path=path + '/chrom_purity.png'
            )

        for index, chrom in enumerate(self.chromatograms):
            self.plot_fraction_signal(
                start=start, end=end,
                save_path=path + '/fractionation_signal_{}.png'.format(index),
                index=index
            )
