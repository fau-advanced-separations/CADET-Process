import copy
from collections import defaultdict
import math

from addict import Dict
import numpy as np
from scipy import integrate

from CADETProcess import CADETProcessError
from CADETProcess.common import EventHandler
from CADETProcess.common import UnsignedInteger
from CADETProcess.common import ProcessMeta
from CADETProcess.common import Section, TimeLine
from CADETProcess.processModel import FlowSheet

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
    """
    _initial_states = ['system_state', 'system_state_derivative']
    _n_cycles = UnsignedInteger(default=1)

    def __init__(self, flow_sheet, name):
        self.flow_sheet = flow_sheet
        self.name = name

        self.system_state = None
        self.system_state_derivative = None

        super().__init__()

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
        """
        feed_all = []
        flow_rate_timelines = self.flow_rate_timelines
        for feed in self.flow_sheet.feed_sources:
            feed_sections = flow_rate_timelines[feed.name]
            m_i  = [integrate.quad(lambda t:
                    feed_sections.value(t) * feed.c[comp], 0, self.cycle_time,
                    points=feed_sections.section_times)[0]
                    for comp in range(self.n_comp)]
            feed_all.append(np.array(m_i))

        return sum(feed_all)


    @property
    def V_eluent(self):
        """float: Volume of the eluent entering the system in one cycle.
        """
        V_all = []
        flow_rate_timelines = self.flow_rate_timelines
        for eluent in self.flow_sheet.eluent_sources:
            eluent_sections = flow_rate_timelines[eluent.name]
            V_eluent = integrate.quad(lambda t:
                eluent_sections.value(t), 0, self.cycle_time,
                points=eluent_sections.section_times)[0]
            V_all.append(V_eluent)

        return sum(V_all)

    @property
    def V_solid(self):
        """float: Volume of all solid phase material used in flow sheet.
        """
        return sum([unit.volume_solid
                       for unit in self.flow_sheet.units_with_binding])


    @property
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

        times = self.event_times
        
        if len(times) == 0:
            enumerator = {0: []}.items()
        else:
            enumerator = self.timeline.items()

        for index, (time, events) in enumerate(enumerator):
            start = time
            if index < len(times)-1:
                end = times[index+1]
            else:
                end = self.cycle_time

            for evt in events:
                evt.perform()

            flow_rates = self.flow_sheet.flow_rates

            for unit, flow_rate in flow_rates.items():
                section = Section(start, end, flow_rate.total)
                flow_rate_timelines[unit]['total'].add_section(section)
                for dest, flow_rate_dest in flow_rate.destinations.items():
                    section = Section(start, end, flow_rate_dest)
                    flow_rate_timelines[unit]['destinations'][dest].add_section(section)

        return Dict(flow_rate_timelines)
    
    @property
    def flow_rate_section_states(self):
        """dict: Lists of events for every time, one or more events occur.
        """
        if len(self.event_times) == 0:
            event_times = [0]
        else:
            event_times = self.event_times
            
        section_states = {
            time: {
                unit.name: {
                    'total': [],
                    'destinations': defaultdict(dict)
                } for unit in self.flow_sheet.units
            } for time in event_times
        }
        
        for evt_time in event_times:
            for unit, unit_flow_rates in self.flow_rate_timelines.items():
                section_states[evt_time][unit]['total'] = \
                    unit_flow_rates['total'].coefficients(evt_time)[0,:]
                
                for dest, tl in unit_flow_rates.destinations.items():
                    section_states[evt_time][unit]['destinations'][dest] = \
                        tl.coefficients(evt_time)[0,:]

        return Dict(section_states)


    @property
    def unit_flow_rate_section_states(self):
        """dict: Lists of events for every time, one or more events occur.
        !!! Todo Not working yet!
        """
        section_states = {
            unit.name: {
                'total': {time: None for time in self.event_times},
                'destinations': defaultdict(dict)
            } for unit in self.flow_sheet.units
        }
        
        for evt_time in self.event_times:
            for param, tl in self.flow_rate_timelines.items():
                
                section_states[evt_time][param]['total'] = tl.coefficients(evt_time)

        return section_states


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
    def piecewise_polynomial_parameters(self):
        parameters = Dict()
        parameters.flow_sheet = self.flow_sheet.piecewise_polynomial_parameters

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
    
            
    def __str__(self):
        return self.name
