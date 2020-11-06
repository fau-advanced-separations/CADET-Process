from addict import Dict
import numpy as np
from collections import defaultdict

from CADETProcess import CADETProcessError
from CADETProcess.common import check_nested, generate_nested_dict, get_nested_value
from CADETProcess.common import StructMeta, UnsignedFloat
from CADETProcess.common import plotlib, PlotParameters
from CADETProcess.common import Section, TimeLine

class EventHandler(metaclass=StructMeta):
    """ Class for handling Events that change the property of an event performer.

    Events
    Attributes
    ----------
    event_performers : dict
        Dictionary with all objects whose attributes can be modified
    events : list
        list of events
    event_dict : dict
        Dictionary with the information abaout all added events of a process.
    durations_dict : dict
        Dictionary with the information abaout all added durations of a process.

    See also
    --------
    Events
    add_event
    add_event_dependency
    Duration
    """
    cycle_time = UnsignedFloat(default=10.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._events = []
        self._durations = []

    @property
    def events(self):
        """ Returns a sorted list with all events.

        Returns a sorted list for all events in a process with information
        about the performer, attribute, state and value. The list is ordered by
        value.

        Returns
        -------
        events : list
            List with all events in EventHandler ordered by time.
        """
        return sorted(self._events, key=lambda evt: evt.time)

    @property
    def events_dict(self):
        """Returns a dictionary with all events in a process.

        Returns
        -------
        events_dict : dict
            Dictionary with all events and durations, indexed by Event.name.
        """
        evts =  {evt.name: evt for evt in self.events}
        durs = {dur.name: dur for dur in self.durations}
        return {**evts, **durs}


    def add_event(self, name, parameter_path, state, time=0.0,
                  component_index=None):
        """Factory function for creating and adding events.

        Parameters
        ----------
        name : str
            Name of the event.
        parameter_path : str
            Path of the parameter that is changed in dot notation.
        state : float
            Value of the attribute that is changed at Event execution.
        time : float
            Time at which the event is executed.

        Raises
        ------
        CADETProcessError
            If Event already exists in the event_dict
        CADETProcessError
            If EventPerformer is not found in EventHandler

        See also
        --------
        Event
        remove_event
        add_event_dependency
        """
        if name in self.events_dict:
            raise CADETProcessError("Event already exists")
        evt = Event(name, self, parameter_path, state, time=time,
                    component_index=component_index)

        self._events.append(evt)
        super().__setattr__(name, evt)

    def remove_event(self, evt_name):
        """Remove event from the EventHandler.

        Parameters
        ----------
        evt_name : str
            Name of the event to be removed

        Raises
        ------
        CADETProcessError
            If Event is not found.

        Note
        ----
        !!! Check remove_event_dependencies

        See also
        --------
        add_event
        remove_event_dependency
        Event
        """
        try:
            evt = self.events_dict[evt_name]
        except KeyError:
            raise CADETProcessError("Event does not exist")

        self._events.remove(evt)
        self.__dict__.pop(evt_name)

    def add_duration(self, name, start_event, end_event, time=0.0):
        """Add duration to the EventHandler.

        Parameters
        ----------
        name: str
            Name of the event.
        start_event : str
            Name of exsiting event for starting duration.
        end_event : str
            Name of existing event for stopping duration.
        time : float
            Time point for perfoming the event.

        Raises
        ------
        CADETProcessError
            If Duration already exists.
            If Start event or End event does not exist.
            If Parameter paths don't match.

        See also
        --------
        durations
        remove_duration
        Duration
        add_event
        add_event_dependency
        """
        if name in self.events_dict:
            raise CADETProcessError("Duration already exists")

        try:
            start_event = self.events_dict[start_event]
        except KeyError:
            raise CADETProcessError("Start event does not exist")
        try:
            end_event = self.events_dict[end_event]
        except KeyError:
            raise CADETProcessError("End event does not exist")

        if start_event.parameter_path != end_event.parameter_path:
            raise CADETProcessError("Event parameters don't match")

        dur = Duration(name, self, start_event, end_event, time)

        self._durations.append(dur)
        super().__setattr__(name, dur)


    def remove_duration(self, duration_name):
        """Remove duration from list of durations.

        Parameters
        ----------
        duration : str
            Name of the duration be removed from the EventHandler.

        Raises
        ------
        CADETProcessError
            If Duration is not found.

        See also
        --------
        Duration
        add_duration
        remove_event_dependency
        """
        try:
            dur = self.events_dict[duration_name]
        except KeyError:
            raise CADETProcessError("Duration does not exist")

        self._durations.remove(dur)
        self.__dict__.pop(duration_name)


    @property
    def durations(self):
        """List of all durations in the process
        """
        return self._durations


    def add_event_dependency(
            self, dependent_event, independent_events, factors=None):
        """Add dependency between two events.

        First it combines the events in the events_dict and the durations_dict
        into one local variable combined_evt_dur. It raises a CADETProcessError
        if the given dependent_event is not in the combined_evt_dur dictionary.
        Also a CADETProcessErroris raised if the length of factors does not equal
        the length of given independent_events. Then it adds the dependency for
        the given dependent event by calling the method add_dependency from the
        event object.

        Parameters
        ---------
        dependent_event : str
            Name of the event whose value will depend on other events.
        independent_events : list
            List of independent event names.
        factors : list
            List of factors used for the relation with the independent events.
            Factors has to be integers of 1 or -1. The length of this list has
            to be equal the list of independent.

        Raises
        ------
        CADETProcessError
            If dependent_event OR independent_events are not in the
            combined_evt_dur dictionary.
            If length of factors does not equal length of independent events.

        See also
        --------
        Event
        add_dependency
        """
        try:
            evt = self.events_dict[dependent_event]
        except KeyError:
            raise CADETProcessError("Cannot find dependent Event")

        if not all(indep in self.events_dict for indep in independent_events):
            raise CADETProcessError("Cannot find one or more independent events")

        if factors is None:
            factors = [1]*len(independent_events)

        if len(factors) != len(independent_events):
            raise CADETProcessError("Length of factors must be equal to length of \
                                independent Events")

        for indep, fac in zip(independent_events, factors):
            indep = self.events_dict[indep]
            evt.add_dependency(indep, fac)


    def remove_event_dependency(self, dependent_event, independent_events):
        """Remove dependency between two events.

        First it checks if the dependent_event exists in list events and also
        if one or more independet event doesn't exist in list events and
        durations and raises a CADETProcessError if it do so. Otherwise the method
        remove_dependency from the event object is called to remove this
        dependency.

        Parameters
        ---------
        dependent_event : str
            Name of the event whose value will depend on other events.
        independent_events : list
            List of independent event names.

        Raises
        ------
        CADETProcessError
            If dependent_event is not in list events.
            If one or more independent event is not in list events and
            durations.

        See also:
        ---------
        remove_dependecy
        Event
        """
        if dependent_event not in self.events:
            raise CADETProcessError("Cannot find dependent Event")

        if not all(evt in self.events_dict for evt in independent_events):
            raise CADETProcessError("Cannot find one or more independent events")

        for indep in independent_events:
            self.events[dependent_event].remove_dependency(indep)


    @property
    def independent_events(self):
        """list: List of all independent events.
        """
        return list(filter(lambda evt: evt.isIndependent, self.events))


    @property
    def dependent_events(self):
        """list: List of all events with dependencies.
        """
        return list(
                filter(lambda evt: evt.isIndependent == False, self.events))

    @property
    def independent_durations(self):
        """list: List of all independent durations.
        """
        return list(filter(lambda dur: dur.isIndependent, self.durations))

    @property
    def dependent_durations(self):
        """list: List of all durations with dependencies.
        """
        return list(
                filter(lambda dur: dur.isIndependent == False, self.durations))

    @property
    def performer_event_lists(self):
        """dict: list of events for every event peformer.
        """
        performer_event_lists = defaultdict(list)
        for evt in self.events:
            performer_event_lists[evt.performer].append(evt)

        return Dict(performer_event_lists)

    @property
    def event_parameter_lists(self):
        """dict: list of events for every event parameter.
        """
        event_performers_dict = defaultdict(list)
        for evt in self.events:
            event_performers_dict[evt.parameter_path].append(evt)

        return Dict(event_performers_dict)

    @property
    def event_parameter_time_lines(self):
        """dict: TimeLine for every event parameter.
        """
        parameter_time_lines = defaultdict(TimeLine)
        for evt_parameter, events in self.event_parameter_lists.items():

            for index, evt in enumerate(events):
                section_start = evt.time

                if index < len(events) - 1:
                    section_end = events[index + 1].time
                    section = Section(section_start, section_end, evt.state)
                    parameter_time_lines[evt.parameter_path].add_section(section)
                else:
                    section_end = self.cycle_time
                    section = Section(section_start, section_end, evt.state)
                    parameter_time_lines[evt.parameter_path].add_section(section)

                    section_start = 0
                    section_end = events[0].time
                    if section_start != section_end:
                        section = Section(section_start, section_end, evt.state)
                        parameter_time_lines[evt.parameter_path].add_section(section)
        return Dict(parameter_time_lines)


    @property
    def time_line(self):
        """dict: Lists of events for every time, one or more events occur.
        """
        time_line = defaultdict(list)
        for evt in self.events:
            time_line[evt.time].append(evt)

        return dict(time_line)

    @property
    def parameters(self):
        parameters = Dict()

        events = {evt.name: evt.parameters for evt in self.independent_events}
        parameters.update(events)

        durations = {dur.name: dur.parameters for dur in self.independent_durations}
        parameters.update(durations)

        parameters['cycle_time'] = self.cycle_time

        return parameters

    @parameters.setter
    def parameters(self, parameters):
        try:
            self.cycle_time = parameters.pop('cycle_time')
        except KeyError:
            pass

        for evt_name, evt_parameters in parameters.items():
            try:
                evt = self.events_dict[evt_name]
            except AttributeError:
                raise CADETProcessError('Not a valid event')
            if evt not in self.independent_events + self.independent_durations:
                raise CADETProcessError('{} is not a valid event'.format(str(evt)))

            evt.parameters = evt_parameters

    def plot_events(self):
        """Plot state as afunctio of time for all performers.
        """
        for performer, events in self.event_performers_dict.items():
            x = self.time/60
            y = self.state_vector[performer]

            plot_parameters = PlotParameters()
            plot_parameters.x_label = '$time~/~min$'
            plot_parameters.y_label = '$state$'
            plot_parameters.title = '${}$'.format(str(performer))

            plotlib.plot(x, y, plot_parameters)


class Event():
    """Class for defining dynamic changes.

    An Event is defined by the performer whose attribute is to be changed to a
    certain state at a given time. The time can depende on other Events or
    Durations. To ensure cyclic behaviour, the time is returned modulo the cycle
    time of the EventHandler.

    Attributes
    ----------
    name : str
        Name of the event.
    event_handler : EventHandler
        Reference to the object holding the performers and the cycle time.
    parameter_path : str.
        Path of the evaluation_object parameter in dot notation.
    state : float
        Value of the attribute to be set by the event.
    time : float
        Time at which the event is performed.
    dependencies : list
        List of the events on which the event time depends.
    fatcors : List
        List with factors for linear combination of dependencies.
    component_index : int
        Index for component specific variables

    Raises
    ------
    CADETProcessError
        If performner does not have attribute.
        If state is not valid for attribute.

    See also:
    ---------
    EventHandler
    Duration
    """
    def __init__(self, name, event_handler, parameter_path, state, time=0.0,
                 component_index=None):

        self.event_handler = event_handler
        self.parameter_path = parameter_path
        self.component_index = component_index
        self.state = state

        self._dependencies = []
        self._factors = []

        self.name = name
        self.time = time

        self._parameters = ['time', 'state']

    @property
    def parameter_path(self):
        return self._parameter_path

    @parameter_path.setter
    def parameter_path(self, parameter_path):
        if not check_nested(self.event_handler.parameters, parameter_path):
            raise CADETProcessError('Not a valid event parameter')
        self._parameter_path = parameter_path

    @property
    def parameter_sequence(self):
        """tuple: Tuple of parameters path elements.
        """
        return tuple(self.parameter_path.split('.'))

    @property
    def component_index(self):
        return self._component_index

    @component_index.setter
    def component_index(self, component_index):
        if component_index is not None:
            parameter = get_nested_value(
                    self.event_handler.parameters, self.parameter_sequence)

            if component_index > len(parameter)-1:
                raise CADETProcessError('Index exceeds components')
        self._component_index = component_index

    def add_dependency(self, dependency, factor=1):
        """Add dependency of event time on other events.

        The time of an event can depend on other events or durationsin any
        linear combination.

        Parameters
        ----------
        dependency : Event
            Event object for adding a dependency.
        factor : int
            Factor of the dependency between to events, default value is set to
            one.

        Raises
        ------
        CADETProcessError
            If the dependency already exists in list dependencies.
        """
        if dependency in self._dependencies:
            raise CADETProcessError("Dependency already exists")

        self._dependencies.append(dependency)
        self._factors.append(factor)

    def remove_dependency(self, dependency):
        """Remove dependencies of events.

        Gets the index of the dependency, which has to be removed and deletes
        the entry for this index in the list factors and dependencies.

        Parameters
        ----------
        dependency : Event
            Event object for adding a dependency.

        Raises
        ------
        CADETProcessError
            If the dependency doesn't exists in list dependencies.
        """
        if dependency in self._dependencies:
            raise CADETProcessError("Dependency not found")

        index = self._dependencies(dependency)

        del(self._dependencies[index])
        del(self._factors[index])


    @property
    def dependencies(self):
        """list : List of events on which the Event depends.
        """
        return self._dependencies

    @property
    def isIndependent(self):
        """bool: True, if event is independent, False otherwise.
        """
        if len(self.dependencies) == 0:
            return True
        else:
            return False

    @property
    def factors(self):
        """list: List of factors for linear combination of dependent events.
        """
        return self._factors

    @property
    def time(self):
        """Returns time when the event is executed.

        Getter
        ------
            Returns
            -------
            time : float
                Returns the time modulo cycle_time, if the event is independent.
                Otherwise, the dependent value is calculated by a linear
                combination of its dependencies multiplied with the respective
                factors.

        Setter
        -----
            Raises
            ------
            CADETProcessError
                If the event is not independent.
        """
        if self.isIndependent:
            time = self._time
        else:
            time = np.dot(
                [dep.time for dep in self.dependencies], self._factors)
        cycle_time = getattr(self.event_handler, 'cycle_time')
        return time % cycle_time

    @time.setter
    def time(self, time):
        if not isinstance(time, (int, float)):
            raise TypeError("Expected {}".format(float))

        if self.isIndependent:
            self._time = time
        else:
            raise CADETProcessError("Cannot set time for dependent events")

    @property
    def current_state(self):
        return get_nested_value(self.event_handler.parameters, self.parameter_sequence)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        try:
            if self.component_index is not None:
                value_list = self.current_state
                value_list[self.component_index] = state
                parameters = generate_nested_dict(self.parameter_sequence, value_list)
            else:
                parameters = generate_nested_dict(self.parameter_sequence, state)
            self.event_handler.parameters = parameters
        except (TypeError, ValueError) as e:
            raise CADETProcessError('{}'.format(str(e)))
        self._state = get_nested_value(self.event_handler.parameters, self.parameter_path)

    @property
    def attribute(self):
        return self.parameter_sequence[-1]

    @property
    def performer(self):
        if len(self.parameter_sequence) == 1:
            return self.parameter_sequence[0]
        else:
            return ".".join(self.parameter_sequence[:-1])

    def perform(self):
        """Set the value to the attriubte of the performer.
        """
        if self.component_index is not None:
            value_list = get_nested_value(self.event_handler.parameters, self.parameter_sequence)
            value_list[self.component_index] = self.state
            parameters = generate_nested_dict(self.parameter_sequence, value_list)
        else:
            parameters = generate_nested_dict(self.parameter_sequence, self.state)
        self.event_handler.parameters = parameters

    @property
    def parameters(self):
        """Returns the parameters in a list.

        Returns
        -------
        parameters : dict
            list with all the parameters.
        """
        return Dict({param: getattr(self, param) for param in self._parameters})

    @parameters.setter
    def parameters(self, parameters):
        if isinstance(parameters, (float, int)):
            self.time = parameters
        else:
            for param, value in parameters.items():
                if param not in self._parameters:
                    raise CADETProcessError('Not a valid parameter')
                setattr(self, param, value)

    def __repr__(self):
        return '{}(name={}, parameter_path={}, state={}, time={}'.format(
                self.__class__.__name__, self.name, self.parameter_path,
                self.state, self.time)

class Duration(Event):
    """Class for representing a duration between two events in an Eventhandler.

    Attributes
    ----------
    start_event : str
        Name of the start event of a duration.
    end_event : str
        Name of the end event of a duration.
    """
    def __init__(self, name, event_handler, start_event, end_event, time=0.0):
        self.start_event = start_event
        self.end_event = end_event

        parameter_path = start_event.parameter_path
        state = start_event.state
        component_index = start_event.component_index

        super().__init__(name, event_handler, parameter_path, state, time,
             component_index)

        self._parameters = ['time']


    def __repr__(self):
        return '{}(name={}, parameter_path={}, state={}, time={}'.format(
                self.__class__.__name__, self.name, self.parameter_path,
                self.state, self.time)
