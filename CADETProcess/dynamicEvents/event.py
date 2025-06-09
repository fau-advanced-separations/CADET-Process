from __future__ import annotations

import copy
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Type

import numpy as np
from addict import Dict
from matplotlib.axes import Axes

from CADETProcess import CADETProcessError, plotting
from CADETProcess.dataStructure import (
    Bool,
    CachedPropertiesMixin,
    Float,
    Integer,
    ParameterBase,
    Sized,
    Structure,
    Typed,
    UnsignedFloat,
    cached_property_if_locked,
    check_nested,
    frozen_attributes,
    generate_nested_dict,
    get_nested_attribute,
    get_nested_value,
)

from .section import MultiTimeLine, Section, TimeLine, generate_indices, unravel

__all__ = ["EventHandler", "Event", "Duration"]


@frozen_attributes
class EventHandler(CachedPropertiesMixin, Structure):
    """
    A handler for dynamic events that affect parameters in a process.

    The `EventHandler` class provides a framework to schedule and manage events
    that cause changes to parameters during a simulation or process. This includes
    single point events as well as durations, and it allows for events to be
    dependent on others, forming complex relationships. Events can be associated
    with transformations or factors that determine their effect.

    Primary functionalities:
    - Schedule events with specific timings and effects.
    - Establish dependencies between events.
    - Manage durations or continuous periods with specific characteristics.
    - Access sorted lists of independent and dependent events.

    Attributes
    ----------
    events : list
        A sorted list of scheduled events, ordered by their execution time.
    durations : list
        List of time durations with specific characteristics.
    event_dict : dict
        A dictionary containing detailed information about all scheduled events.
    durations_dict : dict
        A dictionary containing detailed information about all defined durations.
    independent_events : list
        A list of events that are not influenced by other events.
    dependent_events : list
        A list of events that rely on other events.
    event_performers : dict (not shown in provided code, description based on context)
        A mapping of objects that can perform or be affected by events.
    event_parameters : list
        A list of unique parameters that the events will affect.
    event_times : list
        A list of unique times when events are scheduled to occur, sorted chronologically.
    section_times : list
        A list of times demarcating sections based on event timings.
    n_sections : int
        Total number of sections derived from the section times.
    section_states : dict
        A dictionary providing the state of event parameters at the beginning of every section.
    parameter_events : dict
        A dictionary mapping each parameter to the list of events that affect it.

    Notes
    -----
    The class relies heavily on the concept of "events", which are instances
    of dynamic changes that can influence parameters in the system. These events
    can be independent or based on other events, creating intricate relationships
    to capture complex scenarios.

    See Also
    --------
    Event : Represents a single point change in the system's parameters.
    Duration : Represents a continuous time period with specific attributes or effects.
    """

    cycle_time = UnsignedFloat(default=np.inf)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct for EventHandler."""
        super().__init__(*args, **kwargs)

        self._events = []
        self._durations = []
        self._lock = False

    @cached_property_if_locked
    def events(self) -> list[Event]:
        """
        list: All Events ordered by event time.

        See Also
        --------
        Event
        add_event
        remove_event
        event_dependencies
        Durations

        """
        return sorted(self._events, key=lambda evt: evt.time)

    @cached_property_if_locked
    def events_dict(self) -> dict[str, Event | Duration]:
        """Return Events and Durations orderd by name."""
        evts = {evt.name: evt for evt in self.events}
        durs = {dur.name: dur for dur in self.durations}
        return {**evts, **durs}

    def add_event(
        self,
        name: str,
        parameter_path: str,
        state: float,
        time: float = 0.0,
        indices: int = None,
        dependencies: list = None,
        factors: list = None,
        transforms: Optional[list] = None,
    ) -> Event:
        """
        Add a new event that changes a parameter during the process.

        An event is a dynamic alteration that occurs at a specified time and can modify
        the attributes of specific objects involved in the process.

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
        dependencies : list
            List of the events on which the event time depends.
        factors : List
            List with factors for linear combination of dependencies.
        indices : int
            Index slices for events that modify an entry of a parameter array.
        transforms : list, optional
            List of functions used to transform the parameter value.
            Length must be equal the length of independent events.
            If None, no transform is applied.

        Returns
        -------
        Event:
            The new Event.

        Raises
        ------
        CADETProcessError
            If Event already exists in the event_dict
        CADETProcessError
            If EventPerformer is not found in EventHandler

        See Also
        --------
        events
        remove_event
        add_event_dependency
        Event
        Event.add_dependency
        add_duration
        """
        if name in self.events_dict:
            raise CADETProcessError("Event already exists")
        evt = Event(name, self, parameter_path, state, time=time, indices=indices)

        self._events.append(evt)
        super().__setattr__(name, evt)

        if dependencies is not None:
            self.add_event_dependency(evt.name, dependencies, factors, transforms)

        return evt

    def remove_event(self, evt_name: str) -> None:
        """
        Remove a specified event from the event handler.

        This method ensures that the specified event will no longer influence the
        process by dynamically changing any attributes.

        Parameters
        ----------
        evt_name : str
            Name of the event to be removed

        Raises
        ------
        CADETProcessError
            If Event is not found.

        Notes
        -----
        !!! Check remove_event_dependencies

        See Also
        --------
        add_event
        Event
        Event.remove_dependency
        """
        try:
            evt = self.events_dict[evt_name]
        except KeyError:
            raise CADETProcessError("Event does not exist")

        self._events.remove(evt)
        self.__dict__.pop(evt_name)

    def add_duration(self, name: str, time: float = 0.0) -> "Duration":
        """
        Register a new duration or time point of interest.

        Durations are specific moments in the process that do not necessarily modify
        attributes but are noteworthy or need to be tracked.

        Parameters
        ----------
        name: str
            Name of the event.
        time : float
            Time point for perfoming the event.

        Returns
        -------
        Event:
            The new Event.

        Raises
        ------
        CADETProcessError
            If Duration already exists.

        See Also
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

        return dur

    def remove_duration(self, duration_name: str) -> None:
        """
        Remove a specified duration or time point from tracking.

        This method ensures that the specified duration is no longer considered a point
        of interest in the process.

        Parameters
        ----------
        duration_name : str
            Name of the duration be removed from the EventHandler.

        Raises
        ------
        CADETProcessError
            If Duration is not found.

        See Also
        --------
        add_duration
        Duration
        remove_event_dependency
        """
        try:
            dur = self.events_dict[duration_name]
        except KeyError:
            raise CADETProcessError("Duration does not exist")

        self._durations.remove(dur)
        self.__dict__.pop(duration_name)

    @cached_property_if_locked
    def durations(self) -> list["Duration"]:
        """List of all durations in the process."""
        return self._durations

    def add_event_dependency(
        self,
        dependent_event: str,
        independent_events: list,
        factors: Optional[list] = None,
        transforms: Optional[list] = None,
    ) -> None:
        """
        Create a dependency relationship between events.

        This method establishes how one event (dependent) is influenced by one or more
        other events (independents) through factors and optional transformation
        functions. For example, the time of a dependent event could be determined by the
        sum of the times of independent events multiplied by their corresponding
        factors.

        Parameters
        ----------
        dependent_event : str
            Event whose value will depend on other events.
        independent_events : list
            List of independent event names.
        factors : list, optional
            List of factors used for the relation with the independent events.
            Length must be equal the length of independent events.
            If None, all factors are assumed to be 1.
        transforms : list, optional
            List of functions used to transform the parameter value.
            Length must be equal the length of independent events.
            If None, no transform is applied.

        Raises
        ------
        CADETProcessError
            If dependent_event OR independent_events are not found.
            If length of factors does not equal length of independent events.
            If length of transforms does not equal length of independent events.

        See Also
        --------
        Event
        add_event
        add_duration
        remove_event_dependency
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
            factors = [1] * len(independent_events)

        if not isinstance(factors, list):
            factors = [factors]
        if len(factors) != len(independent_events):
            raise CADETProcessError(
                "Length of factors must equal length of independent Events"
            )
        if transforms is None:
            transforms = [None] * len(independent_events)

        if not isinstance(transforms, list):
            transforms = [transforms]
        if len(transforms) != len(independent_events):
            raise CADETProcessError(
                "Length of transforms must equal length of independent Events"
            )

        for indep, fac, trans in zip(independent_events, factors, transforms):
            indep = self.events_dict[indep]
            evt.add_dependency(indep, fac, trans)

    def remove_event_dependency(
        self, dependent_event: str, independent_events: list
    ) -> None:
        """
        Remove a previously defined dependency between events.

        Parameters
        ----------
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

        See Also
        --------
        Event
        Event.remove_dependecy
        add_event_dependency

        """
        if dependent_event not in self.events:
            raise CADETProcessError("Cannot find dependent Event")

        if not all(evt in self.events_dict for evt in independent_events):
            raise CADETProcessError("Cannot find one or more independent events")

        for indep in independent_events:
            self.events[dependent_event].remove_dependency(indep)

    @cached_property_if_locked
    def independent_events(self) -> list[Event]:
        """list: All events that are not dependent on other events."""
        return list(filter(lambda evt: evt.is_independent, self.events))

    @cached_property_if_locked
    def dependent_events(self) -> list[Event]:
        """list: All events that are dependent on other events."""
        return list(filter(lambda evt: evt.is_independent is False, self.events))

    @cached_property_if_locked
    def event_parameters(self) -> list[str]:
        """list: Event parameters."""
        return list({evt.parameter_path for evt in self.events})

    @cached_property_if_locked
    def event_performers(self) -> list[str]:
        """list: Event peformers."""
        return list({evt.performer for evt in self.events})

    @cached_property_if_locked
    def event_times(self) -> list[float]:
        """list: Time of events, sorted by Event time."""
        event_times = list({evt.time for evt in self.events})
        event_times.sort()

        return event_times

    @cached_property_if_locked
    def section_times(self) -> list[float]:
        """
        list: Section times.

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
    def n_sections(self) -> int:
        """int: Number of sections."""
        return len(self.section_times) - 1

    @cached_property_if_locked
    def section_states(self) -> Dict[float, dict[float, np.ndarray]]:
        """Return state of event parameters at every section."""
        parameter_timelines = self.parameter_timelines
        section_states = defaultdict(dict)

        for sec_time in self.section_times[0:-1]:
            for param, tl in parameter_timelines.items():
                section_states[sec_time][param] = tl.coefficients(sec_time)

        return Dict(section_states)

    @cached_property_if_locked
    def parameter_events(self) -> dict[str, list[Event] | dict[int, list[Event]]]:
        """
        Return event parameters mapped to their corresponding events.

        This dictionary associates each event parameter with its list of events.
        For events that are index-specific, an inner dictionary is used, where
        each index maps to its list of events.

        Notes
        -----
        For index-dependent events, a separate key is added for each index.
        """
        parameter_events = defaultdict(list)
        for evt in self.events:
            if evt.is_index_specific:
                for index in evt.full_indices:
                    parameter_events[evt.parameter_path] = defaultdict(list)

        for evt in self.events:
            if evt.is_index_specific:
                for index in evt.full_indices:
                    parameter_events[evt.parameter_path][index].append(evt)
            else:
                parameter_events[evt.parameter_path].append(evt)
        return Dict(parameter_events)

    @cached_property_if_locked
    def parameter_timelines(self) -> Dict[str, TimeLine]:
        """
        Return Dict: TimeLine representation for every event parameter.

        This dictionary associates each event parameter with its TimeLine object.
        If an event parameter is considered as one of the 'sized parameters',
        it gets associated with a MultiTimeLine object, which handles
        multi-dimensional or indexed data.

        Each timeline, be it a regular or multi-timeline, consists of
        sections representing time intervals where the parameter holds
        a specific value or state.
        """
        parameter_timelines = {}
        multi_timelines = {}

        parameters = self.parameters
        for param in self.event_parameters:
            if param not in self.sized_parameters:
                parameter_timelines[param] = TimeLine()
            else:
                base_state = get_nested_value(parameters, param)
                is_polynomial = check_nested(self.polynomial_parameters, param)
                multi_timelines[param] = MultiTimeLine(base_state, is_polynomial)

        for evt_parameter, events in self.parameter_events.items():
            if not isinstance(events, dict):
                events = {None: events}

            for index, index_events in events.items():
                for i_evt, evt in enumerate(index_events):
                    section_start = evt.time

                    if i_evt < len(index_events) - 1:
                        section_end = index_events[i_evt + 1].time
                        self._create_and_add_sections(
                            section_start,
                            section_end,
                            evt,
                            index,
                            parameter_timelines,
                            multi_timelines,
                        )
                    else:
                        section_end = self.cycle_time
                        self._create_and_add_sections(
                            section_start,
                            section_end,
                            evt,
                            index,
                            parameter_timelines,
                            multi_timelines,
                        )

                        if index_events[0].time != 0:
                            section_start = 0.0
                            section_end = index_events[0].time
                            self._create_and_add_sections(
                                section_start,
                                section_end,
                                evt,
                                index,
                                parameter_timelines,
                                multi_timelines,
                            )

        for param, tl in multi_timelines.items():
            parameter_timelines[param] = tl.combined_time_line

        return Dict(parameter_timelines)

    def _create_and_add_sections(
        self,
        start: float,
        end: float,
        evt: Event,
        index: int | tuple[int],
        parameter_timelines: dict,
        multi_timelines: dict,
    ) -> None:
        """
        Create a new Section and integrate it into the correct TimeLine.

        This method forms a new `Section` object using the provided start and end times,
        and the state from the event `evt`. Depending on whether the event is index-
        specific, this section is then added to a regular TimeLine or a MultiTimeLine.

        Parameters
        ----------
        start : float
            Starting time of the Section.
        end : float
            Ending time of the Section.
        evt : Event
            Event associated with the Section.
            Determines the state for this time interval.
        index : int or tuple
            Index or indices pointing to specific entries in indexed event parameters.
        parameter_timelines : dict
            Dictionary mapping parameter names to their respective TimeLine objects.
        multi_timelines : dict
            Dictionary mapping parameter names to their respective MultiTimeLine objects.
        """
        if not evt.is_index_specific:
            section = Section(start, end, evt.full_state)
            parameter_timelines[evt.parameter_path].add_section(section)
        else:
            section = Section(
                start, end, evt.index_states[index], is_polynomial=evt.is_polynomial
            )
            if evt.degree > 0 and len(index) == 1:
                index = (0,) + index
            multi_timelines[evt.parameter_path].add_section(section, index)

    @property
    def performer_events(self) -> Dict[str, list[Event]]:
        """
        Return Dict: Event performer mapped to their corresponding list of events.

        For every event, this dictionary associates the event's performer
        with the event. This allows for easy retrieval of all events carried out
        by a specific performer.
        """
        performer_events = defaultdict(list)
        for evt in self.events:
            performer_events[evt.performer].append(evt)

        return Dict(performer_events)

    @cached_property_if_locked
    def performer_timelines(self) -> Dict[str, dict[str, TimeLine]]:
        """
        Return Dict: Each performer mapped to their TimeLines based on event parameters.

        This dictionary provides a representation of event parameters in the form
        of timelines for each performer. This hierarchical structure helps in
        quickly accessing the TimeLine of any event parameter specific to a performer.
        """
        performer_timelines = {performer: {} for performer in self.event_performers}

        for param, tl in self.parameter_timelines.items():
            performer, param = param.rsplit(".", 1)
            performer_timelines[performer][param] = tl

        return performer_timelines

    @property
    def parameters(self) -> dict:
        """
        dict: The EventHandler parameters.

        In addition to the standard parameters retrieved from the superclass,
        this property adds event parameters from independent events, parameters
        from durations, and the cycle time.
        """
        parameters = super().parameters

        events = {evt.name: evt.parameters for evt in self.independent_events}
        parameters.update(events)

        events = {evt.name: evt.parameters for evt in self.dependent_events}
        parameters.update(events)

        durations = {dur.name: dur.parameters for dur in self.durations}
        parameters.update(durations)

        parameters["cycle_time"] = self.cycle_time

        return parameters

    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        """Set event parameters based on provided dictionary."""
        try:
            self.cycle_time = parameters.pop("cycle_time")
        except KeyError:
            pass

        for evt_name, evt_parameters in parameters.items():
            try:
                evt = self.events_dict[evt_name]
            except AttributeError:
                raise CADETProcessError("Not a valid event")
            if (
                "time" in evt_parameters
                and evt not in self.independent_events + self.durations
            ):
                raise CADETProcessError(
                    f'Cannot set "time" for {str(evt)} because it is not an independent event.'
                )

            evt.parameters = evt_parameters

    @property
    def sized_parameters(self) -> dict:
        """
        dict: Compilation of parameters from events with indices.

        Besides the sized parameters fetched from the superclass, this property
        collects parameters from events that have associated indices.
        """
        parameters = super().sized_parameters

        events = {
            evt.parameter_path: evt.parameters
            for evt in self.events
            if evt.indices is not None
        }
        parameters.update(events)

        return parameters

    def check_config(self) -> bool:
        """
        Validate the event configuration.

        Ensure no duplicate events exist for a specific parameter and index and
        verify that constants are incorporated in polynomials.

        Returns
        -------
        bool
            True if all validations pass, False otherwise.
        """
        flag = True

        if not self.check_duplicate_events():
            flag = False

        if not self.check_uninitialized_indices():
            flag = False

        return flag

    def check_duplicate_events(self) -> bool:
        """
        Ensure no simulateneous events are scheduled for a specific parameter and index.

        Evaluates all events scheduled for each parameter and index combination.
        Raises a warning if multiple events are scheduled to occur simultaneously,
        as this can lead to unexpected system or simulation behavior.

        Returns
        -------
        bool
            True if no duplicate events are detected, False otherwise.

        Warnings
        --------
            If events are detected to occur at the same timestamp.
        """
        flag = True
        for evt_parameter, events in self.parameter_events.items():
            if not isinstance(events, dict):
                events = {None: events}

            for index, index_events in events.items():
                index_event_times = [evt.time for evt in index_events]

                duplicates = [
                    time
                    for time in set(index_event_times)
                    if index_event_times.count(time) > 1
                ]

                if duplicates:
                    duplicate_events = [
                        evt for evt in index_events if evt.time in duplicates
                    ]
                    warnings.warn(
                        f"Got multiple events at the same time: {duplicate_events}"
                    )
                    flag = False

        return flag

    def check_uninitialized_indices(self) -> bool:
        """
        Ensure all indices are specified when a parameter isn't initialized.

        Returns
        -------
        bool
            True if all indices are properly defined, False otherwise.

        Warnings
        --------
            If there are parameters with uninitialized entries for some indices.
        """
        flag = True
        for evt_parameter, events in self.parameter_events.items():
            current_value = get_nested_value(self.parameters, evt_parameter)
            current_value = np.array(current_value)

            if np.any(np.isnan(current_value)):
                warnings.warn(f"{evt_parameter} has entries which were not initialized")
                flag = False

        return flag

    def plot_events(self, x_axis_in_minutes: bool = True) -> list[Axes]:
        """
        Plot parameter state as a function of time.

        The method creates a plot for each parameter timeline and displays the state
        of the parameter against time. The time is represented on the x-axis, while
        the parameter state is shown on the y-axis.

        Parameters
        ----------
        x_axis_in_minutes: bool, optional
            If True, the x-axis will be plotted using minutes. The default is True.

        Returns
        -------
        list[Axes]
            List of Axes objects, each containing a plot of the parameter state.

        Notes
        -----
        The time is divided into 1001 linearly spaced points between 0 and the cycle
        time for the evaluation of the parameter state.
        """
        time_s = np.linspace(0, self.cycle_time, 1001)

        time_ax = time_s
        if x_axis_in_minutes:
            time_ax = time_ax / 60

        axs: list[Axes] = []

        for parameter, tl in self.parameter_timelines.items():
            fig, ax = plotting.setup_figure()

            y = tl.value(time_s)

            layout = plotting.Layout()
            layout.title = str(parameter)
            layout.x_label = "$time~/~s$"
            if x_axis_in_minutes:
                layout.x_label = "$time~/~min$"
            layout.y_label = "$state$"

            ax.plot(time_ax, y)

            plotting.set_layout(ax, layout)

            axs.append(ax)

        return axs


class Event:
    """
    Defines dynamic changes of model parameters based on events.

    An `Event` is a time-based modification to an attribute of a performer.
    Its execution time can depend on other Events or Durations. To handle
    cyclic behavior, times are computed modulo the cycle time of the EventHandler.

    Attributes
    ----------
    name : str
        The event's name.
    event_handler : EventHandler
        Object managing the performers and cycle time.
    parameter_path : str
        Dot notation path to the target parameter within the evaluation_object.
    state : float
        Desired attribute value when the event is executed.
    time : float, default=0.0
        The execution time of the event.
    indices : int or list, default=None
        Specific indices if the event modifies a parameter array entry.

    See Also
    --------
    EventHandler
    Duration
    """

    _parameters = ["time", "state"]

    def __init__(
        self,
        name: str,
        event_handler: EventHandler,
        parameter_path: str,
        state: float,
        time: float = 0.0,
        indices: Optional[int | list[int]] = None,
    ) -> None:
        """
        Initialize the Event object.

        Parameters
        ----------
        name : str
            The event's name.
        event_handler : EventHandler
            Object managing the performers and cycle time.
        parameter_path : str
            Dot notation path to the target parameter within the evaluation_object.
        state : float
            Desired attribute value when the event is executed.
        time : float, default=0.0
            The execution time of the event.
        indices : int or list of int, optional
            Specific indices if the event modifies a parameter array entry.
            Can also accept slices.
        """
        self.name = name

        self.event_handler = event_handler
        self.parameter_path = parameter_path

        self.indices = indices
        self.state = state

        self._dependencies = []
        self._factors = []
        self._transforms = []

        self.time = time

    @property
    def parameter_path(self) -> str:
        """str: Dot notation path to the target parameter within the evaluation_object."""
        return self._parameter_path

    @parameter_path.setter
    def parameter_path(self, parameter_path: str) -> None:
        if not check_nested(
            self.event_handler.section_dependent_parameters, parameter_path
        ):
            raise CADETProcessError("Not a valid event parameter")
        self._parameter_path = parameter_path

    @property
    def parameter_sequence(self) -> tuple[str, ...]:
        """tuple: Tuple of parameters path elements."""
        return tuple(self.parameter_path.split("."))

    @property
    def parameter_descriptor(self) -> Optional[ParameterBase]:
        """Return parameter descriptor."""
        performer_class = type(self.performer_obj)
        try:
            descriptor = getattr(performer_class, self.parameter_sequence[-1])
        except AttributeError:
            return None

        if not isinstance(descriptor, ParameterBase):
            return None

        return descriptor

    @property
    def parameter_type(self) -> Type[Any]:
        """type: Type of the parameter."""
        if isinstance(self.parameter_descriptor, Typed):
            return self.parameter_descriptor.ty

        if self.current_value is None:
            raise CADETProcessError(
                "Parameter is not initialized. Cannot determine parameter type."
            )

        return type(self.current_value)

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        """tuple: Shape of the parameter array."""
        param_descriptor = self.parameter_descriptor
        if isinstance(param_descriptor, (Float, Integer, Bool)):
            return ()

        if isinstance(param_descriptor, Sized):
            shape = param_descriptor.get_expected_size(self.performer_obj)
            if not isinstance(shape, tuple):
                shape = (shape,)

            return shape

        cur_value = self.current_value
        if cur_value is None:
            raise CADETProcessError(
                "Parameter is not initialized. Cannot determine parameter shape."
            )

        return np.array(cur_value).shape

    @property
    def is_sized(self) -> bool:
        """bool: True if descriptor is instance of Sized. False otherwise."""
        if isinstance(self.parameter_descriptor, (Float, Integer, Bool)):
            return False

        if isinstance(self.parameter_descriptor, Sized):
            return True

        if self.current_value is None:
            raise CADETProcessError(
                "Parameter is not initialized. "
                "Cannot determine dimensions required for setting index."
            )
        return np.array(self.current_value).size > 1

    @property
    def is_polynomial(self) -> bool:
        """bool: True if descriptor is instance of NdPolynomial. False otherwise."""
        return check_nested(
            self.event_handler.polynomial_parameters, self.parameter_path
        )

    @property
    def degree(self) -> int:
        """int: The degree of the polynomial event state."""
        if self.is_polynomial:
            shape = self.parameter_shape
            return shape[-1] - 1
        else:
            return 0

    @property
    def indices(self) -> list[tuple[int]] | None:
        """
        list: Indices for events that modifies only specific entries of a parameter.

        List of tuples for each entry. If parameter is scalar, None
        """
        param_shape = self.parameter_shape

        if len(param_shape) == 0:
            return

        indices = generate_indices(param_shape, self._indices)

        # Check if all indices unique:
        full_indices = unravel(param_shape, indices)
        duplicates = [
            index for index in set(full_indices) if full_indices.count(index) > 1
        ]

        if len(duplicates) > 0:
            raise ValueError(f"Got duplicate entries for indices {duplicates}")

        return indices

    @indices.setter
    def indices(self, indices: list[int]) -> None:
        """
        list: Indices of parameters to set Event state.

        Can be list of tuples. Including slicing.
        """
        if indices is not None and not self.is_sized:
            raise IndexError("Events for scalar parameters cannot have indices.")

        self._indices = indices

        # Since indices are constructed on `get`, call the property here:
        try:
            _ = self.indices
        except (ValueError, TypeError) as e:
            raise e

    @property
    def is_index_specific(self) -> bool:
        """bool: True if event modifies entry of a parameter array, False otherwise."""
        if len(self.full_indices) > 0:
            return True
        else:
            return False

    @property
    def full_indices(self) -> list[int]:
        """list: Full indices."""
        indices = self.indices
        if self.indices is None and len(self.parameter_shape) > 0:
            indices = generate_indices(self.parameter_shape)
        return unravel(self.parameter_shape, indices)

    @property
    def n_indices(self) -> int:
        """int: Number of (full) indices."""
        if len(self.parameter_shape) > 0:
            return len(self.full_indices)
        else:
            return 0

    @property
    def n_entries(self) -> int:
        """int: The number of entries in the event state."""
        if self.is_polynomial:
            return np.array(self.full_state).shape[0]
        else:
            if isinstance(self.full_state, (int, float, bool)):
                return 1
            else:
                return self.n_indices

    def add_dependency(
        self,
        dependency: Event,
        factor: float = 1,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Add a time dependency on another event.

        When an event is dependent, the time of the event is based on a linear
        combination of its dependencies.

        Parameters
        ----------
        dependency : Event
            Event that this event depends on.
        factor : float, default=1
            Weighting factor for the dependency.
        transform : callable, optional
            A function to transform the dependent event's time.

        Raises
        ------
        CADETProcessError
            If the dependency is already listed.
        """
        if dependency in self._dependencies:
            raise CADETProcessError("Dependency already exists")

        self._dependencies.append(dependency)
        self._factors.append(factor)
        if transform is None:

            def transform(t: Any) -> Any:
                return t

        self._transforms.append(transform)

    def remove_dependency(self, dependency: Event) -> None:
        """
        Remove dependencies of events.

        Parameters
        ----------
        dependency : Event
            Event object to remove from dependencies.

        Raises
        ------
        CADETProcessError
            If the dependency doesn't exists in list dependencies.
        """
        if dependency in self._dependencies:
            raise CADETProcessError("Dependency not found")

        index = self._dependencies(dependency)

        del self._dependencies[index]
        del self._factors[index]
        del self._transforms[index]

    @property
    def dependencies(self) -> list[Event]:
        """list: Events on which the Event depends."""
        return self._dependencies

    @property
    def is_independent(self) -> bool:
        """bool: True, if event is independent, False otherwise."""
        if len(self.dependencies) == 0:
            return True
        else:
            return False

    @property
    def factors(self) -> list[int]:
        """list: Linear coefficients for dependent events."""
        return self._factors

    @property
    def transforms(self) -> list[Callable]:
        """list: Transform functions for dependent events."""
        return self._transforms

    @property
    def time(self) -> float:
        """
        float: Time when the event is executed.

        If the value is larger than the cycle time, the time modulo cycle time
        is returned. If the Event is not independent, the time is calculated
        from its dependencies.

        Raises
        ------
        CADETProcessError
            If the event is not independent.

        """
        if self.is_independent:
            time = self._time
        else:
            transformed_time = [
                f(dep.time) for f, dep in zip(self.transforms, self.dependencies)
            ]
            time = np.dot(transformed_time, self._factors)
        cycle_time = getattr(self.event_handler, "cycle_time")
        return time % cycle_time

    @time.setter
    def time(self, time: float) -> None:
        if not np.isscalar(time):
            raise TypeError("Expected scalar value")

        if self.is_independent:
            self._time = time
        else:
            raise CADETProcessError("Cannot set time for dependent events")

    @property
    def state(self) -> float | np.ndarray:
        """
        Return the state of the parameter event.

        When retrieving, it returns the current state of the event.
        When setting, the internal state is updated.

        Returns
        -------
        float | np.ndarray
            The state of the parameter event.
        """
        return self._state

    @state.setter
    def state(self, state: float | np.ndarray) -> None:
        """
        Set the state of the event.

        If indices are not defined and there's no current value, it initializes
        the state with the provided value. The state is then updated with the
        calculated full state based on the indices and provided value.

        Parameters
        ----------
        state : float or np.ndarray
            Value to set as the new state of the event.

        Raises
        ------
        ValueError, TypeError
            If the updated state does not align with the expected data type or structure.
        """
        # Initialize value to get dimensions
        if self._indices is None and self.current_value is None:
            self.set_value(state)
            state = self.current_value

        self._state = state

        # Since event is constructed on `get`, call the property here:
        try:
            _ = self.full_state
        except (ValueError, TypeError) as e:
            raise e

    def _ensure_2D_for_slices(self, state: float | np.ndarray) -> None:
        """
        Ensure the state is 2D when dealing with slices.

        If there's only one set of indices and it contains a slice, it
        prepares the state to be handled as a 2D structure.

        Parameters
        ----------
        state : float or np.ndarray
            The state that might need to be converted.

        Returns
        -------
        float or np.ndarray
            Original state or a 2D structure depending on the indices.
        """
        if len(self.indices) == 1 and any(
            isinstance(i, slice) for i in self.indices[0]
        ):
            state = [state]

        return state

    @property
    def full_state(self) -> float | list:
        """
        Construct the full state based on indices and current value.

        This method reconstructs the state from the stored state value,.

        Returns
        -------
        float or list
            The computed full state, either as a scalar or an array.

        Raises
        ------
        ValueError
            If the length of the state does not match the length of the indices.
        """
        state = self._state
        indices = self.indices

        # Get new (full) parameter value
        if self._indices is None:
            new_value = state
        else:
            if self.current_value is None:
                new_value = np.full(self.parameter_shape, np.nan)
            else:
                new_value = np.array(self.current_value, ndmin=1)

            # Ensure state is list
            if not isinstance(state, list):
                state = [state]

            # Ensure state is 2D for indices that contain slices
            state = self._ensure_2D_for_slices(state)

            if len(state) != len(indices):
                raise ValueError(
                    f"Expected {len(self.indices)} entries for state. Got {len(state)}"
                )

            for i, ind in enumerate(indices):
                expected_shape = new_value[ind].shape
                if (
                    self.is_polynomial
                    and len(self.parameter_shape) > 1
                    and len(ind) == 1
                ):
                    new_slice = self.parameter_descriptor.fill_values(
                        expected_shape, state[i]
                    )
                else:
                    new_slice = np.array(state[i], ndmin=1)

                if any(isinstance(i, slice) for i in ind):
                    if new_slice.size != np.prod(expected_shape):
                        new_slice = np.broadcast_to(new_slice, expected_shape)
                    else:
                        new_slice = np.reshape(new_slice, expected_shape)

                if len(expected_shape) == 0:
                    new_slice = new_slice.squeeze()
                new_value[ind] = new_slice

            if self.parameter_type is not np.ndarray:
                new_value = self.parameter_type(new_value.tolist())

        # Set the value:
        self.set_value(new_value)
        new_value = self.current_value

        if indices is not None:
            new_value = np.array(new_value, ndmin=1)
            full_state = []
            for ind in indices:
                full_state += new_value[ind].flatten().tolist()
        else:
            full_state = new_value

        return full_state

    @property
    def index_states(self) -> dict[tuple, float]:
        """dict[tuple, float]: State values mapped to their respective indices."""
        index_states = {}
        for ind, state in zip(self.full_indices, self.full_state):
            index_states[ind] = state

        return index_states

    @property
    def performer(self) -> str:
        """str: The name of the performer of the event."""
        if len(self.parameter_sequence) == 1:
            return self.parameter_sequence[0]
        else:
            return ".".join(self.parameter_sequence[:-1])

    @property
    def performer_obj(self) -> Any:
        """any: Performer object from the event handler."""
        return get_nested_attribute(self.event_handler, self.performer)

    def set_value(self, state: float | np.ndarray) -> None:
        """Set the specified state to the associated event parameter."""
        state = copy.deepcopy(state)
        if self.parameter_descriptor is not None:
            setattr(self.performer_obj, self.parameter_sequence[-1], state)
        else:
            parameters = generate_nested_dict(self.parameter_sequence, state)
            self.event_handler.parameters = parameters

    @property
    def current_value(self) -> Any:
        """any: Current state of the associated event parameter."""
        if self.parameter_descriptor is not None:
            return getattr(self.performer_obj, self.parameter_sequence[-1])
        else:
            return get_nested_value(self.event_handler.parameters, self.parameter_path)

    @property
    def parameters(self) -> dict:
        """dict: list with all parameters."""
        return Dict({param: getattr(self, param) for param in self._parameters})

    @parameters.setter
    def parameters(self, parameters: float | int | dict) -> None:
        if isinstance(parameters, (float, int)):
            self.time = parameters
        else:
            for param, value in parameters.items():
                if param not in self._parameters:
                    raise CADETProcessError("Not a valid parameter")
                setattr(self, param, value)

    def __repr__(self) -> str:
        """str: String representation of the Event."""
        representation = (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"parameter_path={self.parameter_path}, "
            f"state={self.state}, "
            f"time={self.time}"
        )
        if self.indices is not None:
            representation += f", indices={self.indices}"

        representation += ")"

        return representation


class Duration:
    """
    Class for representing a duration between two events in an Eventhandler.

    Attributes
    ----------
    start_event : str
        Name of the start event of a duration.
    end_event : str
        Name of the end event of a duration.
    """

    def __init__(
        self,
        name: str,
        event_handler: EventHandler,
        time: float = 0.0,
    ) -> None:
        """Initialize Duration Object."""
        self.name = name
        self.time = time
        self._parameters = ["time"]

    @property
    def parameters(self) -> Dict:
        """dict: list with all parameters."""
        return Dict({param: getattr(self, param) for param in self._parameters})

    @parameters.setter
    def parameters(self, parameters: float | int | dict) -> None:
        if isinstance(parameters, (float, int)):
            self.time = parameters
        else:
            for param, value in parameters.items():
                if param not in self._parameters:
                    raise CADETProcessError("Not a valid parameter")
                setattr(self, param, value)

    def __repr__(self) -> str:
        """str: String representation of the Duration."""
        return f"{self.__class__.__name__}(name={self.name}, time={self.time})"
