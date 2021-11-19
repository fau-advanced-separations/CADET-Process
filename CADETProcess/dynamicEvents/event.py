from abc import abstractmethod
import copy

from addict import Dict
import numpy as np
from collections import defaultdict

from CADETProcess import CADETProcessError

from CADETProcess.dataStructure import StructMeta, frozen_attributes
from CADETProcess.dataStructure import UnsignedFloat
from CADETProcess.dataStructure import CachedPropertiesMixin, cached_property_if_locked
from CADETProcess.dataStructure import check_nested, generate_nested_dict, get_nested_value

from CADETProcess import plotting
from .section import Section, TimeLine, MultiTimeLine

@frozen_attributes
class EventHandler(CachedPropertiesMixin, metaclass=StructMeta):
    """Baseclass for handling Events that change a property of an event performer.

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
        self._lock = False
        self._parameters = None

    @property
    def events(self):
        """list: All Events ordered by event time.

        See Also
        --------
        Event
        add_event
        remove_event
        Durations
        """
        return sorted(self._events, key=lambda evt: evt.time)

    @property
    def events_dict(self):
        """dict: Events and Durations orderd by name.
        """
        evts =  {evt.name: evt for evt in self.events}
        durs = {dur.name: dur for dur in self.durations}
        return {**evts, **durs}

    def add_event(
            self, name, parameter_path, state, time=0.0, entry_index=None,
            dependencies=None, factors=None, 
        ):
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
        evt = Event(
            name, self, parameter_path, state, time=time,
            entry_index=entry_index
        )

        self._events.append(evt)
        super().__setattr__(name, evt)
        
        if dependencies is not None:
            self.add_event_dependency(evt.name, dependencies, factors)

        return evt

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

    def add_duration(self, name, time=0.0):
        """Add duration to the EventHandler.

        Parameters
        ----------
        name: str
            Name of the event.
        time : float
            Time point for perfoming the event.

        Raises
        ------
        CADETProcessError
            If Duration already exists.

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

        dur = Duration(name, self, time)

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
            self, dependent_event, independent_events, factors=None
        ):
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

        if not isinstance(independent_events, list):
            independent_events = [independent_events]
        if not all(indep in self.events_dict for indep in independent_events):
            raise CADETProcessError("Cannot find one or more independent events")

        if factors is None:
            factors = [1]*len(independent_events)

        if not isinstance(factors, list):
            factors = [factors]
        if len(factors) != len(independent_events):
            raise CADETProcessError(
                "Length of factors must be equal to length of independent Events"
            )

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
        """list: Independent Events.
        """
        return list(filter(lambda evt: evt.isIndependent, self.events))

    @property
    def dependent_events(self):
        """list: Events with dependencies.
        """
        return list(
            filter(lambda evt: evt.isIndependent == False, self.events)
        )

    @property
    def event_parameters(self):
        """list: Event parameters.
        """
        return list({evt.parameter_path for evt in self.events})

    @property
    def event_performers(self):
        """list: Event peformers.
        """
        return list({evt.performer for evt in self.events})

    @property
    def event_times(self):
        """list: Time of events, sorted by Event time.
        """
        event_times = list({evt.time for evt in self.events})
        event_times.sort()

        return event_times

    @property
    def section_times(self):
        """list: Section times.

        Includes 0 and cycle_time if they do not coincide with event time.
        """
        if len(self.event_times) == 0:
            return [0, self.cycle_time]

        section_times = self.event_times

        if section_times[0] != 0:
            section_times = [0] + section_times
        if section_times[-1] != self.cycle_time:
            section_times = section_times + [self.cycle_time]

        return section_times

    @property
    def n_sections(self):
        """int: Number of sections.
        """
        return len(self.section_times) - 1

    @cached_property_if_locked
    def section_states(self):
        """dict: state of event parameters at every section.
        """
        parameter_timelines = self.parameter_timelines
        section_states = defaultdict(dict)

        for sec_time in self.section_times[0:-1]:
            for param, tl in parameter_timelines.items():
                section_states[sec_time][param] = tl.coefficients(sec_time)

        return Dict(section_states)

    @cached_property_if_locked
    def parameter_events(self):
        """dict: list of events for every event parameter.

        Notes
        -----
        For entry dependent events, a key is added per component.
        """
        parameter_events = defaultdict(list)
        for evt in self.events:
            if evt.entry_index is not None:
                parameter_events[
                    f'{evt.parameter_path}_{evt.entry_index}'
                ].append(evt)
            else:
                parameter_events[evt.parameter_path].append(evt)
        return Dict(parameter_events)

    @cached_property_if_locked
    def parameter_timelines(self):
        """dict: TimeLine for every event parameter.
        """
        parameter_timelines = {
            param: TimeLine() for param in self.event_parameters
            if param not in self.entry_dependent_parameters
            }

        parameters = self.parameters
        multi_timelines = {}
        for param in self.entry_dependent_parameters:
            base_state = get_nested_value(parameters, param)
            multi_timelines[param] = MultiTimeLine(base_state)

        for evt_parameter, events in self.parameter_events.items():
            for index, evt in enumerate(events):
                section_start = evt.time

                if index < len(events) - 1:
                    section_end = events[index + 1].time
                    section = Section(
                        section_start, section_end, evt.state,
                        evt.n_entries, evt.degree
                    )
                    self._add_section(
                        evt, section, parameter_timelines, multi_timelines
                    )
                else:
                    section_end = self.cycle_time
                    section = Section(
                        section_start, section_end, evt.state,
                        evt.n_entries, evt.degree
                    )
                    self._add_section(
                        evt, section, parameter_timelines, multi_timelines
                    )

                    if events[0].time != 0:
                        section = Section(
                            0.0, events[0].time, evt.state,
                            evt.n_entries, evt.degree
                        )
                        self._add_section(
                            evt, section, parameter_timelines, multi_timelines
                        )

        for param, tl in multi_timelines.items():
            parameter_timelines[param] = tl.combined_time_line()

        return Dict(parameter_timelines)

    def _add_section(self, evt, section, parameter_timelines, multi_timelines):
        """Helper function to add sections to timelines."""
        if evt.parameter_path in self.entry_dependent_parameters:
            multi_timelines[evt.parameter_path].add_section(
                section, evt.entry_index
            )
        else:
            parameter_timelines[evt.parameter_path].add_section(
                section
            )

    @property
    def performer_events(self):
        """dict: list of events for every event peformer.
        """
        performer_events = defaultdict(list)
        for evt in self.events:
            performer_events[evt.performer].append(evt)

        return Dict(performer_events)

    @cached_property_if_locked
    def performer_timelines(self):
        """dict: TimeLines for every event parameter of a performer.
        """
        performer_timelines = {performer: {} for performer in self.event_performers}

        for param, tl in self.parameter_timelines.items():
            performer, param = param.rsplit('.',1)
            performer_timelines[performer][param] = tl

        return performer_timelines


    @property
    def parameters(self):
        parameters = Dict()

        events = {evt.name: evt.parameters for evt in self.independent_events}
        parameters.update(events)

        durations = {dur.name: dur.parameters for dur in self.durations}
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
            if evt not in self.independent_events + self.durations:
                raise CADETProcessError('{} is not a valid event'.format(str(evt)))

            evt.parameters = evt_parameters

    @abstractmethod
    def section_dependent_parameters(self):
        return

    @property
    def entry_dependent_parameters(self):
        parameters = {
            evt.parameter_path for evt in self.events
            if evt.entry_index is not None
        }

        return parameters


    @abstractmethod
    def polynomial_parameters(self):
        return

    def plot_events(self):
        """Plot parameter state as function of time.
        """
        time = np.linspace(0, self.cycle_time, 1001)
        for parameter, tl in self.parameter_timelines.items():
            y = tl.value(time)
            
            fig, ax = plotting.setup_figure()
        
            layout = plotting.Layout()
            layout.title = str(parameter)
            layout.x_label = '$time~/~min$'
            layout.y_label = '$state$'
        
            ax.plot(time/60, y)

            plotting.set_layout(fig, ax, layout)        
        
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
    entry_index : int
        Index for events that only modify one entry of a parameter.

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
    def __init__(
            self, name, event_handler,
            parameter_path, state, time=0.0, entry_index=None
        ):
        self.event_handler = event_handler
        self.parameter_path = parameter_path
        self.entry_index = entry_index
        self.state = state

        self._is_polynomial = check_nested(
            self.event_handler.polynomial_parameters, self.parameter_path
        )

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
        if not check_nested(
            self.event_handler.section_dependent_parameters, parameter_path
        ):
            raise CADETProcessError('Not a valid event parameter')
        self._parameter_path = parameter_path

    @property
    def parameter_sequence(self):
        """tuple: Tuple of parameters path elements.
        """
        return tuple(self.parameter_path.split('.'))

    @property
    def parameter_shape(self):
        parameter = get_nested_value(
                self.event_handler.parameters, self.parameter_sequence
            )
        return np.array((parameter),ndmin=2).shape

    @property
    def entry_index(self):
        return self._entry_index

    @entry_index.setter
    def entry_index(self, entry_index):
        if entry_index is not None:
            parameter = get_nested_value(
                self.event_handler.parameters, self.parameter_sequence
            )

            if entry_index > len(parameter)-1:
                raise CADETProcessError('Index exceeds components')
        self._entry_index = entry_index

    @property
    def is_polynomial(self):
        return self._is_polynomial

    @property
    def degree(self):
        if self.is_polynomial:
            state = np.array(self.state, ndmin=2)
            return state.shape[1] - 1
        else:
            return 0

    @property
    def n_entries(self):
        if self.is_polynomial:
            state = np.array(self.state, ndmin=2)
            return state.shape[0]
        else:
            state = self.state
            if isinstance(state, (int, float, bool)):
                return 1
            else:
                state = list(self.state)
            return len(state)

    @property
    def is_entry_specific(self):
        if self.entry_index is not None:
            return True
        else:
            return False

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
        """list: Events on which the Event depends.
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
        """list: Linear coefficients for dependent events.
        """
        return self._factors

    @property
    def time(self):
        """float: Time when the event is executed.

        If the value is larger than the cycle time, the time modulo cycle time
        is returned.
        If the Event is not independent, the time is calculated from its
        dependencies.

        Raises
        ------
        CADETProcessError
            If the event is not independent.
        """
        if self.isIndependent:
            time = self._time
        else:
            time = np.dot(
                [dep.time for dep in self.dependencies], self._factors
            )
        cycle_time = getattr(self.event_handler, 'cycle_time')
        return time % cycle_time

    @time.setter
    def time(self, time):
        if not isinstance(time, (int, float, np.int64, np.float64)):
            raise TypeError("Expected {}".format(float))

        if self.isIndependent:
            self._time = time
        else:
            raise CADETProcessError("Cannot set time for dependent events")

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        current_value = get_nested_value(
            self.event_handler.parameters, self.parameter_path
        )
        try:
            if self.entry_index is not None:
                value_list = current_value
                if isinstance(value_list, np.ndarray):
                    value_list = value_list.tolist()
                value_list[self.entry_index] = state
                parameters = generate_nested_dict(self.parameter_sequence, value_list)
            else:
                parameters = generate_nested_dict(self.parameter_sequence, state)
            self.event_handler.parameters = parameters
        except (TypeError, ValueError) as e:
            raise CADETProcessError('{}'.format(str(e)))

        state = get_nested_value(
                self.event_handler.parameters, self.parameter_path
                )
        if self.entry_index is not None:
            state = state[self.entry_index]

        self._state = copy.deepcopy(state)

    @property
    def performer(self):
        if len(self.parameter_sequence) == 1:
            return self.parameter_sequence[0]
        else:
            return ".".join(self.parameter_sequence[:-1])

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


class Duration():
    """Class for representing a duration between two events in an Eventhandler.

    Attributes
    ----------
    start_event : str
        Name of the start event of a duration.
    end_event : str
        Name of the end event of a duration.
    """
    def __init__(self, name, event_handler, time=0.0):
        self.name = name
        self.time = time
        self._parameters = ['time']

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
        return f'{self.__class__.__name__}(name={self.name}, time={self.time}'
