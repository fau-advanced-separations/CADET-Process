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
        """Sets the flow_sheet object for the process simulation.

        Checks the type of the flow_sheet object and sets it.

        Parameters
        ----------
        flow_sheet : FlowSheet
            Superstructure of the chromatographic process.

        Raises
        -------
        TypeError:
            If selected flow_sheet is no instance of FlowSheet.

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
    def flow_sheet_sections(self):
        flow_sheet_sections = []
        for cycle in range(0, self._n_cycles):
            for time_step, events in self.time_line.items():
                [evt.perform() for evt in events]
                flow_sheet_sections.append(copy.deepcopy(self.flow_sheet))
        return flow_sheet_sections


    @property
    def m_feed(self):
        """ndarray: Mass of the feed components entering the system in one cycle.
        """
        feed_all = []
        flow_rate_sections = self.flow_rate_sections
        for feed in self.flow_sheet.feed_sources:
            feed_sections = flow_rate_sections[feed.name]
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
        flow_rate_sections = self.flow_rate_sections
        for eluent in self.flow_sheet.eluent_sources:
            eluent_sections = flow_rate_sections[eluent.name]
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
    def flow_rate_sections(self):
        """dict: TimeLine of flow_rate for all unit_operations.
        """
        flow_rate_sections = defaultdict(TimeLine)

        times = list(self.time_line.keys())

        for index, (time, events) in enumerate(self.time_line.items()):
            start = time
            if index < len(times)-1:
                end = times[index+1]
            else:
                end = self.cycle_time

            for evt in events:
                evt.perform()

            flow_rates = self.flow_sheet.flow_rates

            for unit in self.flow_sheet.units:
                section = Section(start, end, flow_rates[unit.name].total)
                flow_rate_sections[unit.name].add_section(section)

        return Dict(flow_rate_sections)

    @property
    def time(self):
        """Returns time vetor for one cycle

        Returns
        -------
        time : ndarray
            Time vector of the chromatogramm.
        """
        cycle_time = self.cycle_time
        return np.linspace(0, cycle_time, math.ceil(cycle_time))

    @property
    def _time_complete(self):
        """Defines the complete time for simulation.

        First the number of cycles is set for evaluating the complete time of
        simulation by multiplication the value of the cycle time with the
        number of cycles. To get the number of steps for the linspace without
        start and ending point the indices are evaluated.

        Returns
        -------
        time_complete : ndarray
            array of the time vector with 1 decimal-point rounded values from
            zero to time complete with number of indices as steps.
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
        """Additional information required for calculating performance

        Returns
        -------
        process_meta : ProcessMeta
            Additional information required for calculating performance
        """
        return ProcessMeta(
                cycle_time = self.cycle_time,
                m_feed = self.m_feed,
                V_solid = self.V_solid,
                V_eluent = self.V_eluent,
                )

    def __str__(self):
        return self.name
