from collections import defaultdict

import sympy as sym
from addict import Dict

from CADETProcess import CADETProcessError
from CADETProcess.common import StructMeta, UnsignedInteger, String
from CADETProcess.processModel import UnitBaseClass, SourceMixin, SinkMixin, Sink
from CADETProcess.processModel import NoBinding

class FlowSheet(metaclass=StructMeta):
    """Class to design process flow sheet.

    In this class, UnitOperation models are added and connected in a flow
    sheet.

    Attributes
    ----------
    n_comp : UnsignedInteger
        Number of components of the units in the flow sheet.
    name : String
        Name of the FlowSheet.
    units : list
        UnitOperations in the FlowSheet.
    connections_in : dict
        Connections of UnitOperations.
    connections_out : dict
        Connections of UnitOperations.
    output_states : dict
        Split ratios of outgoing streams of UnitOperations.
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
        self._connections_in = dict()
        self._connections_out = dict()
        self._output_states = dict()

    def _unit_name_decorator(func):
        def inner(self, unit, *args, **kwargs):
            """Do stuff before and/or after execution of function
            """
            if isinstance(unit, str):
                try:
                    unit = self.units_dict[unit]
                except KeyError:
                    raise CADETProcessError('Not a valid unit')
            func(self, unit, *args, **kwargs)

        return inner


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
        self._connections_in[unit] = []
        self._connections_out[unit] = []
        self._output_states[unit] = []
        
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

        See Also
        --------
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
        self._connections_in.pop(unit)
        self._connections_out.pop(unit)
        self._output_states.pop(unit)
        self.__dict__.pop(unit.name)

    @property
    def connections_in(self):
        """Returns dictionary with all ingoing connections for each unit.

        Saves the the destinations as strings in a list for each unit. The
        connections are then saved in a dictionary.

        Returns
        -------
        connections : dict
            Dictionary with a list of all connections for each unit in list
            units.

        See Also
        --------
        connections_out
        remove_connection
        add_connection
        """
        return self._connections_in
    
    @property
    def connections_out(self):
        """Returns dictionary with all outgoing connections for each unit.

        Saves the the destinations as strings in a list for each unit. The
        connections are then saved in a dictionary.

        Returns
        -------
        connections : dict
            Dictionary with a list of all connections for each unit in list
            units.

        See Also
        --------
        connections_in
        remove_connection
        add_connection
        """
        return self._connections_out

    def add_connection(self, origin, destination):
        """Add a connection between units 'origin' and 'destination'.

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
        remove_connection
        connections_in
        connections_out
        output_state
        """
        if origin not in self._units:
            raise CADETProcessError('Origin not in flow sheet')
        if destination not in self._units:
            raise CADETProcessError('Destination not in flow sheet')
        
        if destination in self._connections_out[origin]:
            raise CADETProcessError('Connection already exists')

        self._connections_out[origin].append(destination)
        self._connections_in[destination].append(origin)
        self.set_output_state(origin, 0)

    def remove_connection(self, origin, destination):
        """Removes a connection between units 'origin' and 'destination'.

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
            If connection does not exists in the current flow sheet.

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
            self._connections_out[origin].remove(destination)
            self._connections_in[destination].remove(origin)
        except KeyError:
            raise CADETProcessError('Connection does not exist.')

    @property
    def output_states(self):
        return self._output_states
    
    @_unit_name_decorator
    def set_output_state(self, unit, state):
        """Set split ratio of outgoing streams for UnitOperation.
        
        Parameters
        ----------
        unit : UnitBaseClass
            UnitOperation of flowsheet.
        state : int or list of floats
            new output state of the unit. 

        Raises
        ------
        CADETProcessError
            If unit not in flowSheet
            If state is integer and the state >= the state_length.
            If the length of the states is unequal the state_length.
            If the sum of the states is not equal to 1.
        """
        if unit not in self._units:
            raise CADETProcessError('Unit not in flow sheet')
            
        state_length = len(self.connections_out[unit])

        if state_length == 0:
            output_state = []

        if type(state) is int:
            if state >= state_length:
                raise CADETProcessError('Index exceeds destinations')

            output_state = [0] * state_length
            output_state[state] = 1

        else:
            if len(state) != state_length:
                raise CADETProcessError(
                    'Expected length {}.'.format(state_length))

            elif sum(state) != 1:
                raise CADETProcessError('Sum of fractions must be 1')

            output_state = state

        self._output_states[unit] = output_state                    


    @property
    def feed_sources(self):
        """list: List of sources considered for calculating recovery yield.
        """
        return self._feed_sources

    @_unit_name_decorator
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

    @_unit_name_decorator
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
        """
        return self._eluent_sources

    @_unit_name_decorator
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

    @_unit_name_decorator
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
        """
        return self._chromatogram_sinks

    @_unit_name_decorator
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

    @_unit_name_decorator
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
        """Returns outgoing flow rates for each unit.

        Because a simple 'push' algorithm cannot be used when closed loops are
        present in a FlowSheet (e.g. SMBs), sympy is used to set up and solve 
        the system of equations.
        
        Todo
        ----
        Implement dynamic flow rates!
        Make flow rates function of time and / or return polynomials.

        Returns
        -------
        flow_rates : dict
            Outgoing flow rates for each unit.

        """
        # Setup lists for symbols
        unit_total_flow_symbols = sym.symbols(
            'Q_total_0:{}'.format(self.number_of_units)
        )
        unit_inflow_symbols = []
        unit_outflow_symbols = []
        
        unit_total_flow_eq = []
        unit_outflow_eq = []
        
        # Setup symbolic equations
        for unit_index, unit in enumerate(self.units):
            if isinstance(unit, SourceMixin):
                unit_total_flow_eq.append(
                    sym.Add(unit_total_flow_symbols[unit_index], -unit.flow_rate)
                )
            else:
                unit_i_inflow_symbols = []
                
                for origin in self.connections_in[unit]:
                    origin_index = self.get_unit_index(origin)
                    unit_i_inflow_symbols.append(
                        sym.symbols('Q_{}_{}'.format(origin_index, unit_index))
                    )
                    
                unit_i_total_flow_eq = sym.Add(
                    *unit_i_inflow_symbols, -unit_total_flow_symbols[unit_index]
                )
                
                unit_inflow_symbols += unit_i_inflow_symbols
                unit_total_flow_eq.append(unit_i_total_flow_eq)
                
            if not isinstance(unit, Sink):
                output_state = self.output_states[unit]
                unit_i_outflow_symbols = []
                
                for destination in self.connections_out[unit]:
                    destination_index = self.get_unit_index(destination)
                    unit_i_outflow_symbols.append(
                        sym.symbols('Q_{}_{}'.format(unit_index, destination_index))
                    )
                    
                unit_i_outflow_eq = [
                    sym.Add(
                        unit_i_outflow_symbols[dest],
                        -unit_total_flow_symbols[unit_index]*output_state[dest]
                    )
                    for dest in range(len(self.connections_out[unit]))
                ]
                                     
                unit_outflow_symbols += unit_i_outflow_symbols
                unit_outflow_eq += unit_i_outflow_eq
                
        # Solve system of equations
        solution = sym.solve(
            unit_total_flow_eq + unit_outflow_eq, 
            (*unit_total_flow_symbols, *unit_inflow_symbols, *unit_outflow_symbols)
        )
        solution = {str(key): value for key, value in solution.items()}
        
        # Assign values to flow_rates
        flow_rates = Dict()
        
        for unit_index, unit in enumerate(self.units):
            flow_rates[unit.name].total = \
                float(solution['Q_total_{}'.format(unit_index)])
            
            for destination in self.connections_out[unit]:
                destination_index = self.get_unit_index(destination)
                flow_rates[unit.name].destinations[destination.name] = \
                    float(solution['Q_{}_{}'.format(unit_index, destination_index)])

        return flow_rates

    @property
    def parameters(self):
        parameters = {unit.name: unit.parameters for unit in self.units}
        parameters['output_states'] = {
            unit.name: self.output_states[unit] for unit in self.units}

        return Dict(parameters)

    @parameters.setter
    def parameters(self, parameters):
        output_states = parameters.pop('output_states')
        for unit, state in output_states.items():
            unit = self.units_dict[unit]
            self.set_output_state(unit, state)
        
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
        -------
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
