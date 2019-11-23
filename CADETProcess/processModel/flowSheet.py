from addict import Dict

from CADETProcess import CADETProcessError
from CADETProcess.common import StructMeta, UnsignedInteger, String
from CADETProcess.processModel import UnitBaseClass, SourceMixin, SinkMixin
from CADETProcess.processModel import NoBinding

class FlowSheet(metaclass=StructMeta):
    """ Class to design a superstructure of a chromatographic process.

    This class defines the number of components and saves all units for
    generating the superstructure in a list. It can add and remove units from
    the list, defines sources and sinks and connections between the units. The
    flow_rate or states of each unit is adapted.

    Attributes
    ----------
    n_comp : UnsignedInteger
        Number of components in a system.
    name : String
        Name of the FlowSheet object.
    units : list
        List with all the unit_operations.
    """
    name = String()
    n_comp = UnsignedInteger()

    def __init__(self, n_comp, name):
        self.n_comp = n_comp
        self.name = name
        self._units = []
        self._feed_sources = []
        self._eluent_sources = []
        self._chromatogram_sinks = []

    @property
    def units(self):
        """list: list of all unit_operations in the flow sheet.
        """
        return self._units

    @property
    def units_dict(self):
        """dict: dictionary for access of units by name
        """
        return {unit.name: unit for unit in self.units}

    @property
    def number_of_units(self):
        """Returns the length respectively the number of unit_operations
        of the list units.

        Returns
        -------
        number_of_units : int
            Number of the units in the flow sheet.
        """
        return len(self._units)

    def get_unit_index(self, unit):
        """Return the unit index of the unit.

        Returns the unit index and raises a CADETProcessError if the unit doesn't
        exist in the current flow sheet object.

        Parameters
        ----------
        unit : UnitBaseClass
            UnitBaseClass object of which the index is to be returned.

        Raises
        ------
        CADETProcessError
            If unit does not exist in the current flow sheet.

        Returns
        -------
        unit_index : int
            Returns the unit index of the unit_operation.
        """
        if unit not in self.units:
            raise CADETProcessError('Unit not in flow sheet')

        return self.units.index(unit)


    @property
    def sources(self):
        """list: List of all units implementing the SourceMixin interface.
        """
        return [unit for unit in self._units if isinstance(unit, SourceMixin)]

    @property
    def sinks(self):
        """list: List of all units implementing the SinkMixin interface.
        """
        return [unit for unit in self._units if isinstance(unit, SinkMixin)]

    @property
    def units_with_binding(self):
        """list : List of units with binding behavior
        """
        return [unit for unit in self._units
                if not isinstance(unit.binding_model, NoBinding)]

    def add_unit(self, unit, feed_source=False, eluent_source=False,
                 chromatogram_sink=False):
        """Add units to the flow sheet.

        Adds units to the list and checks for correct type, number of
        components or if it already exists.

        Parameters
        ----------
        unit : UnitBaseClass
            UnitBaseClass object to be added to the flow sheet.
        feed_source : bool
            If True, add unit to feed sources.
        eluent_source : bool
            If True, add unit to eluent sources.
        chromatogram_sink : bool
            If True, add unit to chromatogram sinks.

        Raises
        ------
        TypeError
            If unit is no instance of UnitBaseClass.
        CADETProcessError
            If unit already exists in flow sheet.
            If number of components of the unit does not fit that of flow
            sheet.

        See also
        --------
        remove_unit

        Note
        -----
        can't add objects of Source or SorceMixin to FlowSheet.
        """
        if not isinstance(unit, UnitBaseClass):
            raise TypeError('Expected UnitOperation')

        if unit in self._units:
            raise CADETProcessError('Unit already part of System')

        if unit.n_comp != self.n_comp:
            raise CADETProcessError('Number of components does not match.')

        self._units.append(unit)
        super().__setattr__(unit.name, unit)

        if feed_source:
            self.add_feed_source(unit)
        if eluent_source:
            self.add_eluent_source(unit)
        if chromatogram_sink:
            self.add_chromatogram_sink(unit)

    def remove_unit(self, unit):
        """Remove unit from flow sheet.

        Removes unit from the list. Tries to remove units which are twice
        located as desinations. For this the origins and destinations are
        deleted for the unit. Raises a CADETProcessError if an ValueError is
        excepted. If the unit is specified as feed_source, eluent_source
        or chromatogram_sink, the corresponding attributes are deleted.

        Parameters
        ----------
        unit : UnitBaseClass
            UnitBaseClass object to be removed to the flow sheet.

        Raises
        ------
        CADETProcessError
            If unit does not exist in the flow sheet.

        See also:
        ---------
        add_unit
        feed_source
        eluent_source
        chromatogram_sink
        """
        if unit not in self.units:
            raise CADETProcessError('Unit not in flow sheet')

        if unit is self.feed_sources:
            self.remove_feed_source(unit.name)
        if unit is self.eluent_sources:
            self.remove_eluent_source(unit.name)
        if unit is self.chromatogram_sinks:
            self.remove_chromatogram_sink(unit.name)

        origins = unit.origins.copy()
        for origin in origins:
            self.remove_connection(origin, unit)

        destinations = unit.destinations.copy()
        for destination in destinations:
            self.remove_connection(unit, destination)

        self._units.remove(unit)
        self.__dict__.pop(unit.name)

    @property
    def connections(self):
        """Returns a dictionary with all connections fo each unit.

        Saves the the destinations as strings in a list for each unit. The
        connections are then saved in a dictionary.

        Returns
        -------
        connections : dict
            Dictionary with a list of all connections for each unit in list
            units.

        See Also
        --------
        remove_connection
        add_connection
        """
        return {str(unit): [str(dest) for dest in unit.destinations.keys()]
                for unit in self.units}

    def add_connection(self, origin, destination):
        """Add a connection between units 'origin' and 'destination'.

        First the origin and destination is checked of being objects in the
        list of units and also of an already existing connection in the flow
        sheet. For adding a connection a data_dict dictionary is generated with
        a default flow_rate. The destination of an origin.destination and
        conversely is assigned to the dictionary. A default vlaue for the
        origin output_state is set.

        Parameters
        ----------
        origin : UnitBaseClass
            UnitBaseClass from which the connection originates.
        destination : UnitBaseClass
            UnitBaseClass where the connection terminates.

        Raises
        ------
        CADETProcessError
            If origin OR destination do not exist in the current flow sheet.
            If connection already exists in the current flow sheet.

        See Also
        --------
        connections
        remove_connection
        """
        if origin not in self._units:
            raise CADETProcessError('Origin not in flow sheet')
        if destination not in self._units:
            raise CADETProcessError('Destination not in flow sheet')
        if destination in origin.destinations:
            raise CADETProcessError('Connection already exists')

        data_dict = dict()
        data_dict['flow_rate'] = 0.0

        origin.destinations[destination] = data_dict
        destination.origins[origin] = data_dict

        origin.output_state = 0

    def remove_connection(self, origin, destination):
        """Removes a connection between units 'origin' and 'destination'.

        First the origin and destination is checked of being objects in the
        list of units. Then it tries to delete the destinaton of an
        origin.destination and conversely. A CADETProcessError is raised, when a
        KeyError is excepted.

        Parameters
        ----------
        origin : UnitBaseClass
            UnitBaseClass from which the connection originates.
        destination : UnitBaseClass
            UnitBaseClass where the connection terminates.

        Raises
        ------
        CADETProcessError
            If origin OR destination do not exist in the current flow sheet.
            If Connection does not exist.

        See Also
        --------
        connections
        add_connection
        """
        if origin not in self._units:
            raise CADETProcessError('Origin not in flow sheet')
        if destination not in self._units:
            raise CADETProcessError('Destination not in flow sheet')

        try:
            del origin.destinations[destination]
            del destination.origins[origin]
        except KeyError:
            raise CADETProcessError('Connection does not exist.')

    @property
    def feed_sources(self):
        """list: List of sources considered for calculating recovery yield.

        Raises
        ------
        CADETProcessError
            If no feed source is set.
        """
        if len(self._feed_sources) == 0:
            raise CADETProcessError('No feed source set.')
        return self._feed_sources

    def add_feed_source(self, feed_source):
        """Add source to list of units to be considered for recovery.

        Parameters
        ----------
        feed_source : SourceMixin
            Unit to be added to list of feed sources

        Raises
        ------
        CADETProcessError
            If unit is not in a source object
            If unit is already marked as feed source
        """
        if feed_source not in self.sources:
            raise CADETProcessError('Expected Source')
        if feed_source in self._feed_sources:
            raise CADETProcessError('{} is already eluent source'.format(
                    feed_source))
        self._feed_sources.append(feed_source)

    def remove_feed_source(self, feed_source):
        """Remove source from list of units to be considered for recovery.

        Parameters
        ----------
        feed_source : SourceMixin
            Unit to be removed from list of feed sources.
        """
        if feed_source not in self._feed_sources:
            raise CADETProcessError('Unit \'{}\' is not a feed source.'.format(
                    feed_source))
        self._feed_sources.remove(feed_source)

    @property
    def eluent_sources(self):
        """list: List of sources to be considered for eluent consumption.

        Raises
        ------
        CADETProcessError
            If no eluent source is set.
        """
        if len(self._eluent_sources) == 0:
            raise CADETProcessError('No eluent source set')
        return self._eluent_sources

    def add_eluent_source(self, eluent_source):
        """Add source to list of units to be considered for eluent consumption.

        Parameters
        ----------
        eluent_source : SourceMixin
            Unit to be added to list of eluent sources.

        Raises
        ------
        CADETProcessError
            If unit is not in a source object
            If unit is already marked as eluent source
        """
        if eluent_source not in self.sources:
            raise CADETProcessError('Expected Source')
        if eluent_source in self._eluent_sources:
            raise CADETProcessError('{} is already eluent source'.format(
                    eluent_source))
        self._eluent_sources.append(eluent_source)

    def remove_eluent_source(self, eluent_source):
        """Remove source from list of units to be considered eluent consumption.

        Parameters
        ----------
        eluent_source : SourceMixin
            Unit to be added to list of eluent sources.
        Raises
        ------
        CADETProcessError
            If unit is not in eluent sources
        """
        if eluent_source not in self._eluent_sources:
            raise CADETProcessError('Unit \'{}\' is not an eluent source.'.format(
                    eluent_source))
        self._eluent_sources.remove(eluent_source)

    @property
    def chromatogram_sinks(self):
        """list: List of sinks to be considered for fractionation.

        Raises
        ------
        CADETProcessError
            If no chromatogram sink is set.
        """
        if len(self._chromatogram_sinks) == 0:
            raise CADETProcessError('No chromatogram sink set')
        return self._chromatogram_sinks

    def add_chromatogram_sink(self, chromatogram_sink):
        """Add sink to list of units to be considered for fractionation.

        Parameters
        ----------
        chromatogram_sink : SinkMixin
            Unit to be added to list of chromatogram sinks.

        Raises
        ------
        CADETProcessError
            If unit is not a sink object.
            If unit is already marked as chromatogram sink.
        """
        if chromatogram_sink not in self.sinks:
            raise CADETProcessError('Expected Sink')
        if chromatogram_sink in self._chromatogram_sinks:
            raise CADETProcessError('{} is already chomatogram sink'.format(
                    chromatogram_sink))
        self._chromatogram_sinks.append(chromatogram_sink)

    def remove_chromatogram_sink(self, chromatogram_sink):
        """Remove sink from list of units to be considered for fractionation.

        Parameters
        ----------
        chromatogram_sink : SinkMixin
            Unit to be added to list of chromatogram sinks.

        Raises
        ------
        CADETProcessError
            If unit is not a chromatogram sink.
        """
        if chromatogram_sink not in self._chromatogram_sinks:
            raise CADETProcessError('Unit \'{}\' is not a chromatogram sink.'.format(
                    chromatogram_sink))
        self._chromatogram_sinks.remove(chromatogram_sink)

    @property
    def flow_rates(self):
        """Updates the flow_rates and saves them in a dictionary.

        First all flow_rates are updated, then the flow_rates are saved for
        each destination and for each unit into a dictionary.

        Returns
        -------
        flow_rates : dict
            Dictionary containing for each unit the the flow_rate of each
            connection.

        See Also
        --------
        update_flow_rates
        reset_flow_rates
        """
        self.update_flow_rates()

        flow_rates = Dict()

        for unit in self.units:
            flow_rates[unit.name].destinations = {
                    dest.name: val['flow_rate']
                    for (dest, val) in unit.destinations.items()
                    }
            if isinstance(unit, SinkMixin) and not isinstance(unit, SourceMixin):
                flow_rates[unit.name].total = sum(
                    [val['flow_rate'] for val in unit.origins.values()]
                    )
            else:
                flow_rates[unit.name].total = sum(
                    [val['flow_rate'] for val in unit.destinations.values()]
                    )

        return flow_rates

    def reset_flow_rates(self):
        """Resets the values of the flow_rates to a default value and updates
        them.

        If the unit is no instance of class SourceMixin the value of the
        flow_rate is set to defeault value and updated.

        See Also
        --------
        update_flow_rates
        flow_rates
        """
        for unit in self._units:
            if not isinstance(unit, SourceMixin):
                unit.flow_rate = 0.0
            unit.update_flow_rate()

    def update_flow_rates(self):
        """Updates the flow_rates of each unit in a flow sheet.

        First the flow_rates are reset then a local method push_flow_rate is
        called. This method loops through the destinations of a unit and updates
        the flow_rates. Then also the flow_rates of the sources are adapted by
        calling the method push_flow_rate with source as parameter.

        See Also
        --------
        reset_flow_rates
        flow_rates
        """
        def push_flow_rate(unit, visited):
            """Update the flow_rates for succeeding units until sink is reached.

            Loop through destinations of a unit and updates the corresponding
            flow_rate depending on the output state.
            If the unit is not in sources the flow_rate is the sum
            of all flow_rates coming from an origin. If the destination is no
            part of list sinks and the flow_rate is unequal zero, the method
            push_flow_rate of the destination is called rekursiveley.

            Parameters
            ----------
            unit : UnitBaseClass
                 object to be added to the flow sheet.
            """
            # Loop through outgoing streams
            for index, destination in enumerate(unit.destinations):
                # Add ingoing streams if unit not source-like
                if unit not in self.sources:
                    unit.flow_rate = sum(
                        [orig['flow_rate'] for orig in unit.origins.values()])

                unit.update_flow_rate()

                # Push flow rates of connecting nodes
                if destination not in self.sinks \
                    and destination not in visited \
                    and unit.flow_rate != 0.0 \
                    and unit.output_state[index] > 0:
                    push_flow_rate(destination, visited)

                visited.append(destination)


        # Loop through inlets and push flow_rates
        for source in self.sources:
            visited = []
            push_flow_rate(source, visited)

        for unit in self.units:
            push_flow_rate(unit, visited)

    @property
    def parameters(self):
        parameters = {unit.name: unit.parameters for unit in self.units}

        return Dict(parameters)

    @parameters.setter
    def parameters(self, parameters):
        for unit, params in parameters.items():
            if unit not in self.units_dict:
                raise CADETProcessError('Not a valid unit')
            self.units_dict[unit].parameters = params

    @property
    def initial_state(self):
        initial_state = {unit.name: unit.initial_state for unit in self.units}

        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        for unit, st in initial_state.items():
            if unit not in self.units_dict:
                raise CADETProcessError('Not a valid unit')
            self.units_dict[unit].initial_state = st

    def __getitem__(self, unit_name):
        """Make FlowSheet substriptable s.t. units can be used as keys.

        Parameters
        ----------
        unit_name : str
            Name of the unit.

        Returns
        -------
        unit : UnitBaseClass
            UnitOperation of flowsheet.

        Raises
        ------
        KeyError
            If unit not in flowSheet
        """
        try:
            return self.units_dict[unit_name]
        except KeyError:
            raise KeyError('Not a valid unit')


    def __contains__(self, item):
        """Check if an item is part of units.

        Parameters
        ----------
        item : UnitBaseClass
            item to be searched

        Returns
        --------
        Bool : bool
            Return True if item is in list units, otherwise False.

        Note
        ----
        maybe deficient in documentation.
        """
        if item in self._units:
            return True
        else:
            return False