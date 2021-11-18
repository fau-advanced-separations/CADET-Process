from collections import defaultdict

import numpy as np
import sympy as sym
from addict import Dict

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import frozen_attributes
from CADETProcess.dataStructure import StructMeta, UnsignedInteger, String
from .componentSystem import ComponentSystem
from .unitOperation import UnitBaseClass, SourceMixin, SinkMixin, Sink
from .binding import NoBinding


@frozen_attributes
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
    connections : dict
        Connections of UnitOperations.
    output_states : dict
        Split ratios of outgoing streams of UnitOperations.
    """

    name = String()
    n_comp = UnsignedInteger()
    
    def __init__(self, component_system, name=None):
        self.component_system = component_system
        self.name = name
        self._units = []
        self._feed_sources = []
        self._eluent_sources = []
        self._chromatogram_sinks = []
        self._connections = Dict()
        self._output_states = Dict()
        self._flow_rates = Dict()
        self._parameters = Dict()
        self._section_dependent_parameters = Dict()
        self._polynomial_parameters = Dict()

    @property
    def component_system(self):
        return self._component_system

    @component_system.setter
    def component_system(self, component_system):
        if not isinstance(component_system, ComponentSystem):
            raise TypeError('Expected ComponentSystem')
        self._component_system = component_system
        
    @property
    def n_comp(self):
        return self.component_system.n_comp
            
    def _unit_name_decorator(func):
        def wrapper(self, unit, *args, **kwargs):
            """Enable calling functions with unit object or unit name.
            """
            if isinstance(unit, str):
                try:
                    unit = self.units_dict[unit]
                except KeyError:
                    raise CADETProcessError('Not a valid unit')
            return func(self, unit, *args, **kwargs)

        return wrapper
    
    def update_parameters(self):
        for unit in self.units:
            self._parameters[unit.name] = unit.parameters
            self._section_dependent_parameters[unit.name] = \
                unit.section_dependent_parameters
            self._polynomial_parameters[unit.name] = unit.polynomial_parameters
        
        self._parameters['output_states'] = {
            unit.name: self.output_states[unit] for unit in self.units
        }
        
        self._section_dependent_parameters['output_states'] = {
            unit.name: self.output_states[unit] 
            for unit in self.units
        }
        
    def update_parameters_decorator(func):
        def wrapper(self, *args, **kwargs):
            """Update parameters dict to save time.
            """
            results = func(self, *args, **kwargs)
            self.update_parameters()
            
            return results
        return wrapper

    @property
    def units(self):
        """list: list of all unit_operations in the flow sheet.
        """
        return self._units

    @property
    def units_dict(self):
        """dict: Unit names and objects.
        """
        return {unit.name: unit for unit in self.units}
    
    @property
    def unit_names(self):
        """list: Unit names
        """
        return [unit.name for unit in self.units]
    
    @property
    def number_of_units(self):
        """int: Number of unit operations in the FlowSheet.
        """
        return len(self._units)

    @_unit_name_decorator
    def get_unit_index(self, unit):
        """Return the unit index of the unit.

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
        """list: All UnitOperations implementing the SourceMixin interface.
        """
        return [unit for unit in self._units if isinstance(unit, SourceMixin)]

    @property
    def sinks(self):
        """list: All UnitOperations implementing the SinkMixin interface.
        """
        return [unit for unit in self._units if isinstance(unit, SinkMixin)]

    @property
    def units_with_binding(self):
        """list: UnitOperations with binding models.
        """
        return [unit for unit in self._units
                if not isinstance(unit.binding_model, NoBinding)]

    @update_parameters_decorator
    def add_unit(
            self, unit, 
            feed_source=False, eluent_source=False, chromatogram_sink=False
        ):
        """Add unit to the flow sheet.

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
            If n_comp does not match with FlowSheet.

        See Also
        --------
        remove_unit
        """
        if not isinstance(unit, UnitBaseClass):
            raise TypeError('Expected UnitOperation')

        if unit in self._units:
            raise CADETProcessError('Unit already part of System')

        if unit.component_system is not self.component_system:
            raise CADETProcessError('Component systems do not match.')

        self._units.append(unit)
        self._connections[unit] = Dict({
            'origins': [], 
            'destinations': [],
        })
        self._output_states[unit] = []
        self._flow_rates[unit] = []
        
        super().__setattr__(unit.name, unit)

        if feed_source:
            self.add_feed_source(unit)
        if eluent_source:
            self.add_eluent_source(unit)
        if chromatogram_sink:
            self.add_chromatogram_sink(unit)


    @update_parameters_decorator
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
            self.remove_feed_source(unit)
        if unit is self.eluent_sources:
            self.remove_eluent_source(unit)
        if unit is self.chromatogram_sinks:
            self.remove_chromatogram_sink(unit)

        origins = self.connections[unit].origins.copy()
        for origin in origins:
            self.remove_connection(origin, unit)

        destinations = self.connections[unit].destinations.copy()
        for destination in destinations:
            self.remove_connection(unit, destination)

        self._units.remove(unit)
        self._connections.pop(unit)
        self._output_states.pop(unit)
        self.__dict__.pop(unit.name)

    @property
    def connections(self):
        """dict: In- and outgoing connections for each unit.

        See Also
        --------
        add_connection
        remove_connection
        """
        return self._connections
    
    @update_parameters_decorator
    def add_connection(self, origin, destination):
        """Add connection between units 'origin' and 'destination'.

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
        output_state
        """
        if origin not in self._units:
            raise CADETProcessError('Origin not in flow sheet')
        if destination not in self._units:
            raise CADETProcessError('Destination not in flow sheet')
        
        if destination in self.connections[origin].destinations:
            raise CADETProcessError('Connection already exists')

        self._connections[origin].destinations.append(destination)
        self._connections[destination].origins.append(origin)        
        
        self.set_output_state(origin, 0)

    @update_parameters_decorator
    def remove_connection(self, origin, destination):
        """Remove connection between units 'origin' and 'destination'.

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
            self._connections[origin].destinations.remove(destination)
            self._connections[destination].origins.remove(origin)
        except KeyError:
            raise CADETProcessError('Connection does not exist.')

    @property
    def output_states(self):
        return self._output_states
    
    @_unit_name_decorator
    @update_parameters_decorator
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
            
        state_length = len(self.connections[unit].destinations)

        if state_length == 0:
            output_state = []

        if isinstance(state, (int, np.int64)):
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

    
    def get_flow_rates(self, state=None):
        """Calculate flow rate for all connections.unit operation flow rates.
        
        If an additional state is passed, it will b
        
        Parameters
        ----------
        state : Dict, optional
            Output states

        Returns
        -------
        flow_rates : Dict
            Volumetric flow rate of each unit operation.

        """
        flow_rates = {
            unit.name: unit.flow_rate for unit in self.sources
        }
        output_states = self.output_states

        if state is not None:
            for param, value in state.items():
                param = param.split('.')
                unit_name = param[1]
                param_name = param[-1]
                if param_name == 'flow_rate':
                    flow_rates[unit_name] = value[0]
                elif unit_name == 'output_states':
                    unit = self.units_dict[param_name]
                    output_states[unit] = list(value.ravel())
        
        def list_factory():
            return [0,0,0,0]
        
        total_flow_rates = {unit.name: list_factory() for unit in self.units}
        destination_flow_rates = {
            unit.name: defaultdict(list_factory) for unit in self.units
        }
        
        for i in range(4):
            solution = self.solve_flow_rates(flow_rates, output_states, i)
            if solution is not None:
                for unit_index, unit in enumerate(self.units):
                    total_flow_rates[unit.name][i] = \
                        float(solution['Q_total_{}'.format(unit_index)])
                    
                    for destination in self.connections[unit].destinations:
                        destination_index = self.get_unit_index(destination)
                        destination_flow_rates[unit.name][destination.name][i] = \
                            float(solution['Q_{}_{}'.format(unit_index, destination_index)])
        
        flow_rates = Dict()
        for unit in self.units:
            flow_rates[unit.name].total = np.array(total_flow_rates[unit.name])
            for destination, flow_rate in destination_flow_rates[unit.name].items():
                flow_rates[unit.name].destinations[destination] = np.array(flow_rate)
        
        return flow_rates                

    def solve_flow_rates(self, source_flow_rates, output_states, coeff=0):
        """Solve flow rates of system using sympy.
        
        Because a simple 'push' algorithm cannot be used when closed loops are
        present in a FlowSheet (e.g. SMBs), sympy is used to set up and solve 
        the system of equations.
    
        Parameters
        ----------
        source_flow_rates: dict
            Flow rates of Source UnitOperations.
        output_states: dict
            Output states of all UnitOperations.
        coeff: int
            Polynomial coefficient of flow rates to be solved.
    
        Returns
        -------
        solution : dict
            Solution of the flow rates in the system
            
        Note
        ----
        Since dynamic flow rates can be described as cubic polynomials, the
        flow rates are solved individually for all coefficients.
        """
        coeffs = np.array(
            [source_flow_rates[unit.name][coeff] for unit in self.sources]
        )
        if not np.any(coeffs):
            return None
        
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
                    sym.Add(
                        unit_total_flow_symbols[unit_index], 
                        - float(source_flow_rates[unit.name][coeff])
                    )
                )
            else:
                unit_i_inflow_symbols = []
                
                for origin in self.connections[unit].origins:
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
                output_state = output_states[unit]
                unit_i_outflow_symbols = []
                
                for destination in self.connections[unit].destinations:
                    destination_index = self.get_unit_index(destination)
                    unit_i_outflow_symbols.append(
                        sym.symbols('Q_{}_{}'.format(unit_index, destination_index))
                    )
                    
                unit_i_outflow_eq = [
                    sym.Add(
                        unit_i_outflow_symbols[dest],
                        -unit_total_flow_symbols[unit_index]*output_state[dest]
                    )
                    for dest in range(len(self.connections[unit].destinations))
                ]
                                     
                unit_outflow_symbols += unit_i_outflow_symbols
                unit_outflow_eq += unit_i_outflow_eq
                
        # Solve system of equations
        solution = sym.solve(
            unit_total_flow_eq + unit_outflow_eq, 
            (*unit_total_flow_symbols, *unit_inflow_symbols, *unit_outflow_symbols)
        )
        solution = {str(key): value for key, value in solution.items()}
        
        return solution
    
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
            raise CADETProcessError(
                '{} is already eluent source'.format(feed_source)
            )
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
            raise CADETProcessError(
                '{} is already chomatogram sink'.format(chromatogram_sink)
            )
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
            raise CADETProcessError(
                'Unit \'{}\' is not a chromatogram sink.'.format(chromatogram_sink)
            )
        self._chromatogram_sinks.remove(chromatogram_sink)


    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        try:
            output_states = parameters.pop('output_states')
            for unit, state in output_states.items():
                unit = self.units_dict[unit]
                self.set_output_state(unit, state)
        except KeyError:
            pass
        
        for unit, params in parameters.items():
            if unit not in self.units_dict:
                raise CADETProcessError('Not a valid unit')
            self.units_dict[unit].parameters = params
            
        self.update_parameters()

    @property
    def section_dependent_parameters(self):
        return self._section_dependent_parameters

    @property
    def polynomial_parameters(self):
        return self._polynomial_parameters

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
        Bool : True if item is in units, otherwise False.

        Note
        ----
        maybe deficient in documentation.
        """
        if (item in self._units) or (item in self.unit_names):
            return True
        else:
            return False
