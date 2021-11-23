from collections import defaultdict
import math

from addict import Dict
import numpy as np
from scipy import integrate
from scipy import interpolate

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import UnsignedInteger
from CADETProcess.dataStructure import CachedPropertiesMixin, cached_property_if_locked

from CADETProcess.dynamicEvents import EventHandler
from CADETProcess.dynamicEvents import Section, TimeLine

from CADETProcess.common import ProcessMeta

from .flowSheet import FlowSheet
from .unitOperation import Source

class Process(EventHandler):
    """Class for defining the dynamic changes of a flow sheet.

    Attributes
    ----------
    flow_sheet : FlowSheet
        Superstructure of the chromatographic process.
    name : str
        Name of the process object to be simulated.
    system_state : NoneType
        State of the process object, default set to None.
    system_state_derivate : NoneType
        Derivative of the state, default set to None.

    See also
    --------
    EventHandler
    FlowSheet
    ProcessMeta
    Simulator
    """
    _initial_states = ['system_state', 'system_state_derivative']
    _n_cycles = UnsignedInteger(default=1)

    def __init__(self, flow_sheet, name, *args, **kwargs):
        self.flow_sheet = flow_sheet
        self.name = name

        self.system_state = None
        self.system_state_derivative = None

        super().__init__(*args, **kwargs)

    @property
    def n_comp(self):
        return self.flow_sheet.n_comp

    @property
    def flow_sheet(self):
        """FlowSheet: flow sheet of the process model.

        Raises
        -------
        TypeError:
            If flow_sheet is not an instance of FlowSheet.

        Returns
        -------
        flow_sheet : FlowSheet
            Superstructure of the chromatographic process.
        """
        return self._flow_sheet

    @flow_sheet.setter
    def flow_sheet(self, flow_sheet):
        if not isinstance(flow_sheet, FlowSheet):
            raise TypeError('Expected FlowSheet')
        self._flow_sheet = flow_sheet

    @property
    def m_feed(self):
        """ndarray: Mass of the feed components entering the system in one cycle.
        !!! Account for dynamic flow rates and concentrations!
        """
        flow_rate_timelines = self.flow_rate_timelines

        feed_all = np.zeros((self.n_comp,))
        for feed in self.flow_sheet.feed_sources:
            feed_flow_rate_time_line = flow_rate_timelines[feed.name].total
            feed_signal_param = 'flow_sheet.{}.c'.format(feed.name)
            if feed_signal_param in self.parameter_timelines:
                feed_signal_time_line = self.parameter_timelines[feed_signal_param]
            else:
                feed_signal_time_line = TimeLine()
                feed_section = Section(
                    0, self.cycle_time, feed.c, n_entries=self.n_comp, degree=3
                )
                feed_signal_time_line.add_section(feed_section)

            m_i  = [
                integrate.quad(
                    lambda t: \
                        feed_flow_rate_time_line.value(t) \
                        * feed_signal_time_line.value(t)[comp], 
                        0, self.cycle_time, points=self.event_times
                    )[0] for comp in range(self.n_comp)
            ]
            
            feed_all += np.array(m_i)
            
        return feed_all
            
    @property
    def V_eluent(self):
        """float: Volume of the eluent entering the system in one cycle.
        """
        flow_rate_timelines = self.flow_rate_timelines

        V_all = 0
        for eluent in self.flow_sheet.eluent_sources:
            eluent_time_line = flow_rate_timelines[eluent.name]['total']
            V_eluent = eluent_time_line.integral()
            V_all += V_eluent
            
        return float(V_all)

    @property
    def V_solid(self):
        """float: Volume of all solid phase material used in flow sheet.
        """
        return sum(
            [unit.volume_solid for unit in self.flow_sheet.units_with_binding]
        )

    @cached_property_if_locked
    def flow_rate_timelines(self):
        """dict: TimeLine of flow_rate for all unit_operations.
        """
        flow_rate_timelines = {
            unit.name: {
                'total': TimeLine(),
                'destinations': defaultdict(TimeLine)
                }
            for unit in self.flow_sheet.units
        }
        
        # Create dummy section state for Processes without events
        if len(self.section_states) == 0:
            it = [(None, {})]
        else:
            it = self.section_states.items()
        
        for i, (time, state) in enumerate(it):
            start = self.section_times[i]
            end = self.section_times[i+1]

            flow_rates = self.flow_sheet.get_flow_rates(state)

            for unit, flow_rate in flow_rates.items():
                section = Section(
                    start, end, flow_rate.total, n_entries=1, degree=3
                )
                flow_rate_timelines[unit]['total'].add_section(section)
                for dest, flow_rate_dest in flow_rate.destinations.items():
                    section = Section(
                        start, end, flow_rate_dest, n_entries=1, degree=3
                    )
                    flow_rate_timelines[unit]['destinations'][dest].add_section(section)   
                
        return Dict(flow_rate_timelines)
    
    @cached_property_if_locked
    def flow_rate_section_states(self):
        """dict: Flow rates for all units for every section time.
        """
        section_states = {
            time: {
                unit.name: {
                    'total': [],
                    'destinations': defaultdict(dict)
                } for unit in self.flow_sheet.units
            } for time in self.section_times[0:-1]
        }
        
        for sec_time in self.section_times[0:-1]:
            for unit, unit_flow_rates in self.flow_rate_timelines.items():
                section_states[sec_time][unit]['total'] = \
                    unit_flow_rates['total'].coefficients(sec_time)[0]
                
                for dest, tl in unit_flow_rates.destinations.items():
                    section_states[sec_time][unit]['destinations'][dest] = \
                        tl.coefficients(sec_time)[0]

        return Dict(section_states)


    @property
    def time(self):
        """np.array: Returns time vector for one cycle

        Todo
        ----
        Remove from Process; Check also EventHandler.plot_events()
        
        See Also
        --------
        cycle_time
        _time_complete
        """
        cycle_time = self.cycle_time
        return np.linspace(0, cycle_time, math.ceil(cycle_time))

    @property
    def _time_complete(self):
        """np.array: time vector for mulitple cycles of a process.
        
        See Also
        --------
        time
        _n_cycles
        """
        complete_time = self._n_cycles * self.cycle_time
        indices = self._n_cycles*math.ceil(self.cycle_time) - (self._n_cycles-1)
        return np.round(np.linspace(0, complete_time, indices), 1)


    @property
    def system_state(self):
        return self._system_state

    @system_state.setter
    def system_state(self, system_state):
        self._system_state = system_state

    @property
    def system_state_derivative(self):
        return self._system_state_derivative

    @system_state_derivative.setter
    def system_state_derivative(self, system_state_derivative):
        self._system_state_derivative = system_state_derivative

    @property
    def parameters(self):
        parameters = super().parameters

        parameters['flow_sheet'] = self.flow_sheet.parameters

        return Dict(parameters)

    @parameters.setter
    def parameters(self, parameters):
        try:
            self.flow_sheet.parameters = parameters.pop('flow_sheet')
        except KeyError:
            pass

        super(Process, self.__class__).parameters.fset(self, parameters)
    
    @property
    def section_dependent_parameters(self):
        parameters = Dict()
        parameters.flow_sheet = self.flow_sheet.section_dependent_parameters

        return parameters

    @property
    def polynomial_parameters(self):
        parameters = Dict()
        parameters.flow_sheet = self.flow_sheet.polynomial_parameters

        return parameters

    @property
    def initial_state(self):
        initial_state = {state: getattr(self, state)
            for state in self._initial_states}
        initial_state['flow_sheet'] = self.flow_sheet.initial_state

        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        try:
            self.flow_sheet.initial_state = initial_state.pop('flow_sheet')
        except KeyError:
            pass

        for state_name, state_value in initial_state.items():
            if state_name not in self._initial_state:
                raise CADETProcessError('Not an valid state')
            setattr(self, state_name, state_value)

    @property
    def config(self):
        return Dict({'parameters': self.parameters,
                'initial_state': self.initial_state})

    @config.setter
    def config(self, config):
        self.parameters = config['parameters']
        self.initial_state = config['initial_state']

    @property
    def process_meta(self):
        """ProcessMeta: Meta information of the process.
        
        See Also
        --------
        ProcessResults
        Performance        
        """
        return ProcessMeta(
            cycle_time = self.cycle_time,
            m_feed = self.m_feed,
            V_solid = self.V_solid,
            V_eluent = self.V_eluent,
        )
    
    def add_inlet_profile(self, unit, time, c, component_index=None, s=1e-6):
        if not isinstance(unit, Source):
            raise TypeError('Expected Source')
        
        if max(time) > self.cycle_time:
            raise ValueError('Inlet profile exceeds cycle time')
        
        if component_index == -1:
            # Assume same profile for all components
            if c.ndim > 1:
                raise ValueError('Expected single concentration profile')

            c = np.column_stack([c]*2)
            
        elif component_index is None and c.shape[1] != self.n_comp:
            # Assume c is given for all components
            raise CADETProcessError('Number of components does not match')
            
        for comp in range(self.n_comp):
            tck = interpolate.splrep(time, c[:,comp], s=s)
            ppoly = interpolate.PPoly.from_spline(tck)
        
            for i, (t, sec) in enumerate(zip(ppoly.x, ppoly.c.T)):
                if i < 3:
                    continue
                elif i > len(ppoly.x) - 5:
                    continue
                evt = self.add_event(
                    f'{unit}_inlet_{comp}_{i-3}', f'flow_sheet.{unit}.c', 
                    np.flip(sec), t, comp
                )
            
    def __str__(self):
        return self.name
