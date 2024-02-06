from functools import wraps
from warnings import warn
from collections import defaultdict

import numpy as np
from addict import Dict

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import frozen_attributes
from CADETProcess.dataStructure import Structure, UnsignedInteger, String
from .componentSystem import ComponentSystem
from .unitOperation import UnitBaseClass
from .unitOperation import Inlet, Outlet, Cstr
from .binding import NoBinding


@frozen_attributes
class FlowSheet(Structure):
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

    def __init__(self, component_system, name=None):
        self.component_system = component_system
        self.name = name
        self._units = []
        self._feed_inlets = []
        self._eluent_inlets = []
        self._product_outlets = []
        self._connections = Dict()
        self._output_states = Dict()
        self._flow_rates = Dict()
        self._parameters = Dict()
        self._sized_parameters = Dict()
        self._polynomial_parameters = Dict()
        self._section_dependent_parameters = Dict()

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

    def unit_name_decorator(func):
        @wraps(func)
        def wrapper(self, unit, *args, **kwargs):
            """Enable calling functions with unit object or unit name."""
            if isinstance(unit, str):
                try:
                    unit = self.units_dict[unit]
                except KeyError:
                    raise CADETProcessError('Not a valid unit')
            return func(self, unit, *args, **kwargs)

        return wrapper

    def origin_destination_name_decorator(func):
        @wraps(func)
        def wrapper(self, origin, destination, *args, **kwargs):
            """Enable calling origin and destination using unit names."""
            if isinstance(origin, str):
                try:
                    origin = self.units_dict[origin]
                except KeyError:
                    raise CADETProcessError('Not a valid unit')

            if isinstance(destination, str):
                try:
                    destination = self.units_dict[destination]
                except KeyError:
                    raise CADETProcessError('Not a valid unit')

            return func(self, origin, destination, *args, **kwargs)

        return wrapper

    def update_parameters(self):
        for unit in self.units:
            self._parameters[unit.name] = unit.parameters
            self._section_dependent_parameters[unit.name] = \
                unit.section_dependent_parameters
            self._polynomial_parameters[unit.name] = unit.polynomial_parameters
            self._sized_parameters[unit.name] = unit.sized_parameters

        self._parameters['output_states'] = {
            unit.name: self.output_states[unit] for unit in self.units
        }

        self._sized_parameters['output_states'] = {
            unit.name: self.output_states[unit]
            for unit in self.units
        }

        self._section_dependent_parameters['output_states'] = {
            unit.name: self.output_states[unit]
            for unit in self.units
        }

    def update_parameters_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            """Update parameters dict to save time."""
            results = func(self, *args, **kwargs)
            self.update_parameters()

            return results
        return wrapper

    @property
    def units(self):
        """list: list of all unit_operations in the flow sheet."""
        return self._units

    @property
    def units_dict(self):
        """dict: Unit operation names and objects."""
        return {unit.name: unit for unit in self.units}

    @property
    def unit_names(self):
        """list: Names of unit operations."""
        return [unit.name for unit in self.units]

    @property
    def number_of_units(self):
        """int: Number of unit operations in the FlowSheet."""
        return len(self._units)

    @unit_name_decorator
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
    def inlets(self):
        """list: All Inlets in the system."""
        return [unit for unit in self._units if isinstance(unit, Inlet)]

    @property
    def outlets(self):
        """list: All Outlets in the system."""
        return [unit for unit in self._units if isinstance(unit, Outlet)]

    @property
    def cstrs(self):
        """list: All Cstrs in the system."""
        return [unit for unit in self._units if isinstance(unit, Cstr)]

    @property
    def units_with_binding(self):
        """list: UnitOperations with binding models."""
        return [unit for unit in self._units
                if not isinstance(unit.binding_model, NoBinding)]

    @update_parameters_decorator
    def add_unit(
            self, unit,
            feed_inlet=False, eluent_inlet=False, product_outlet=False):
        """Add unit to the flow sheet.

        Parameters
        ----------
        unit : UnitBaseClass
            UnitBaseClass object to be added to the flow sheet.
        feed_inlet : bool
            If True, add unit to feed inlets.
        eluent_inlet : bool
            If True, add unit to eluent inlets.
        product_outlet : bool
            If True, add unit to product outlets.

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

        if unit in self._units or unit.name in self.unit_names:
            raise CADETProcessError('Unit already part of System')

        if unit.component_system is not self.component_system:
            raise CADETProcessError('Component systems do not match.')

        self._units.append(unit)

        for i in range(unit.n_ports):

            if isinstance(unit, Inlet):
                self._connections[unit]['origins'] = None
                self._connections[unit]['destinations'][i] = defaultdict(list)

            elif isinstance(unit, Outlet):
                self._connections[unit]['origins'][i] = defaultdict(list)
                self._connections[unit]['destinations'] = None

            else:
                self._connections[unit]['origins'][i] = defaultdict(list)
                self._connections[unit]['destinations'][i] = defaultdict(list)



        # TODO: Ports must also be implemented for output states and flow rates.

        self._output_states[unit] = Dict()
        self._flow_rates[unit] = []

        super().__setattr__(unit.name, unit)

        if feed_inlet:
            self.add_feed_inlet(unit)
        if eluent_inlet:
            self.add_eluent_inlet(unit)
        if product_outlet:
            self.add_product_outlet(unit)

    @unit_name_decorator
    @update_parameters_decorator
    def remove_unit(self, unit):
        """Remove unit from flow sheet.

        Removes unit from the list. Tries to remove units which are twice
        located as desinations. For this the origins and destinations are
        deleted for the unit. Raises a CADETProcessError if an ValueError is
        excepted. If the unit is specified as feed_inlet, eluent_inlet
        or product_outlet, the corresponding attributes are deleted.

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
        feed_inlet
        eluent_inlet
        product_outlet

        """
        if unit not in self.units:
            raise CADETProcessError('Unit not in flow sheet')

        if unit is self.feed_inlets:
            self.remove_feed_inlet(unit)
        if unit is self.eluent_inlets:
            self.remove_eluent_inlet(unit)
        if unit is self.product_outlets:
            self.remove_product_outlet(unit)
        # TODO: Connections must be removed for ports @hannah
        #Unit has to be removed from every port of every origin and every destination

        origins = []
        destinations = []

        if self._connections[unit]['origins'] is not None:
            origins = [origin for ports in self._connections[unit]['origins'] for origin in self._connections[unit]['origins'][ports]]

        if self._connections[unit]['destinations'] is not None:
            destinations = [destination for ports in self._connections[unit]['destinations'] for destination in self._connections[unit]['destinations'][ports]].copy()

        for origin in origins:
            for origin_port in self._connections[unit]['origins']:
                for unit_port in self._connections[unit]['origins'][origin_port][origin]:
                    self.remove_connection(origin, unit, origin_port, unit_port)

        for destination in destinations:
            for destination_port in self._connections[unit]['destinations']:
                for unit_port in self._connections[unit]['destinations'][destination_port][destination]:
                    self.remove_connection(unit, destination, unit_port, destination_port)

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

    @origin_destination_name_decorator
    @update_parameters_decorator
    def add_connection(
            self, origin, destination, origin_port=None, destination_port=None):
        """Add connection between units 'origin' and 'destination'.

        Parameters
        ----------
        origin : UnitBaseClass
            UnitBaseClass from which the connection originates.
        destination : UnitBaseClass
            UnitBaseClass where the connection terminates.
        origin_port : int
            Port from which connection originates.
        destination_port : int
            Port where connection terminates.

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
        if origin.n_ports != 1 and origin_port is None:
            raise CADETProcessError('Missing `origin_port`')
        if origin.n_ports == 1:
            origin_port = 0
        if origin_port > origin.n_ports-1:
            raise CADETProcessError('Origin port exceeds number of ports.')
        if origin_port in self._connections[destination]['origins'][destination_port][origin]:
            raise CADETProcessError("Connection already exists")

        if destination not in self._units:
            raise CADETProcessError('Destination not in flow sheet')
        if destination.n_ports !=1 and origin_port is None:
            raise CADETProcessError('Missing `destination_port`')
        if destination.n_ports == 1:
            destination_port = 0
        if destination_port > destination.n_ports-1:
            raise CADETProcessError('Destination port exceeds number of ports.')
        if destination_port in self._connections[origin]['destinations'][origin_port][destination]:
            raise CADETProcessError("Connection already exists")

        # TOOD: Add tests
        # TODO: How to store connections with ports
        # e.g. self._connections[origin][port].destinations.append(destination)
        # e.g. self._connections[destination][port].origins.append(destination)
        # TODO: How to expose discretization n_rad to flow_sheet/ports
        # TODO: What happens if port index exceeds n_rad
        # TODO: Add check methods (e.g. for unconnected ports, flow rate balance)
        # TODO: Can ports have individual output states?

        self._connections[destination]['origins'][destination_port][origin].append(origin_port)
        self._connections[origin]['destinations'][origin_port][destination].append(destination_port)

        # TODO: Set output state for ports @hannah
        self.set_output_state(origin, 0)

    @origin_destination_name_decorator
    @update_parameters_decorator
    def remove_connection( self, origin, destination, origin_port=None, destination_port=None):
        """Remove connection between units 'origin' and 'destination'.

        Parameters
        ----------
        origin : UnitBaseClass
            UnitBaseClass from which the connection originates.
        destination : UnitBaseClass
            UnitBaseClass where the connection terminates.
        origin_port : int
            Port from which connection originates.
        destination_port : int
            Port where connection terminates.

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
        if origin.n_ports != 1 and origin_port is None:
            raise CADETProcessError('Missing `origin_port`')
        if origin.n_ports == 1:
            origin_port = 0
        if origin_port > origin.n_ports-1:
            raise CADETProcessError('Origin port exceeds number of ports.')

        if destination not in self._units:
            raise CADETProcessError('Destination not in flow sheet')
        if destination.n_ports !=1 and origin_port is None:
            raise CADETProcessError('Missing `destination_port`')
        if destination.n_ports == 1:
            destination_port = 0
        if destination_port > destination.n_ports-1:
            raise CADETProcessError('Destination port exceeds number of ports.')

        try:

            self._connections[destination]['origins'][destination_port].pop(origin)
            self._connections[origin]['destinations'][origin_port].pop(destination)
        except KeyError:
            raise CADETProcessError('Connection does not exist.')



    @origin_destination_name_decorator
    def connection_exists(self, origin, destination, origin_port=0, destination_port=0):
        """bool: check if connection exists in flow sheet.

        Parameters
        ----------
        origin : UnitBaseClass
            UnitBaseClass from which the connection originates.
        destination : UnitBaseClass
            UnitBaseClass where the connection terminates.

        Notes
        -----
        This method checks if the units destination equals any origin in the connected unit (and vice versa) while neglecting ports.

        """
        if destination in self._connections[origin].destinations[origin_port]\
                and destination_port in self._connections[origin].destinations[origin_port][destination]\
                and origin in self._connections[destination].origins[destination_port]\
                and origin_port in self._connections[destination].origins[destination_port][origin]:
            return True

        return False

    def check_connections(self):
        """Validate that units are connected correctly.

        Raises
        ------
        Warning
            If Inlets have ingoing streams.
            If Outlets have outgoing streams.
            If Units (other than Cstr) are not fully connected.

        Returns
        -------
        flag : bool
            True if all units are connected correctly. False otherwise.

        """
        flag = True
        for unit, connections in self.connections.items():
            if isinstance(unit, Inlet):
                if connections.origins != None:
                    flag = False
                    warn("Inlet unit cannot have ingoing stream.")
                if len(connections.destinations) == 0:
                    flag = False
                    warn(f" Unit '{unit.name}' does not have outgoing stream.")
            elif isinstance(unit, Outlet):
                if connections.destinations != None:
                    flag = False
                    warn("Outlet unit cannot have outgoing stream.")
                if len(connections.origins) == 0:
                    flag = False
                    warn(f"Unit '{unit.name}' does not have ingoing stream.")
            elif isinstance(unit, Cstr):
                if unit.flow_rate is not None and len(connections.destinations) == 0:
                    flag = False
                    warn("Cstr cannot have flow rate without outgoing stream.")
            else:
                if all(len(port_list) == 0 for port_list in connections.origins.values()):
                    flag = False
                    warn(f"Unit '{unit.name}' does not have ingoing stream.")
                if all(len(port_list) == 0 for port_list in connections.destinations.values()):
                    flag = False
                    warn(f" Unit '{unit.name}' does not have outgoing stream.")

        return flag

    @property
    def missing_parameters(self):
        missing_parameters = []

        for unit in self.units:
            missing_parameters += [
                f'{unit.name}.{param}' for param in unit.missing_parameters
            ]

        return missing_parameters

    def check_units_config(self):
        """Check if units are configured correctly.

        Returns
        -------
        flag : bool
            True if units are configured correctly. False otherwise.

        """
        flag = True
        for unit in self.units:
            if not unit.check_required_parameters():
                flag = False
        return flag

    @property
    def output_states(self):
        return self._output_states

    @unit_name_decorator
    @update_parameters_decorator
    def set_output_state(self, unit, state, port=0):
        """Set split ratio of outgoing streams for UnitOperation.

        Parameters
        ----------
        unit : UnitBaseClass
            UnitOperation of flowsheet.
        state : int or list of floats or dict
            new output state of the unit.
        port : int
            Port for which to set the output state.

        Raises
        ------
        CADETProcessError
            If unit not in FlowSheet
            If state is integer and the state >= the state_length.
            If the length of the states is unequal the state_length.
            If the sum of the states is not equal to 1.

        """
        def get_port_index(unit_connection_dict, destination, destination_port):
            """helper function to classify the index of a connection for your outputstate

            Parameters
            ----------

            unit_connection_dict : Defaultdict
                contains dict with connected units and their respective ports
            destination : UnitBaseClass
                destination object
            destination_port : int
                destination Port

            """

            ret_index = 0
            for unit_destination in unit_connection_dict:
                if unit_destination is destination:
                    ret_index+=unit_connection_dict[unit_destination].index(destination_port)
                    break
                ret_index+=len(unit_connection_dict[unit_destination])
            return ret_index


        if unit not in self._units:
            raise CADETProcessError('Unit not in flow sheet')

        state_length = sum([len(self.connections[unit].destinations[port][unit_name]) for unit_name in self.connections[unit].destinations[port]])

        if state_length == 0:
            output_state = []

        if isinstance(state, (int, np.integer)):
            if state >= state_length:
                raise CADETProcessError('Index exceeds destinations')

            output_state = [0] * state_length
            output_state[state] = 1

        elif isinstance(state, dict):
            output_state = [0] * state_length
            for dest, value in state.items():
                try:
                    for destination_port, output_value in value.items():
                        try:
                            assert self.connection_exists(unit, dest, port, destination_port)
                        except AssertionError:
                            raise CADETProcessError(f'{unit} on port {port} does not connect to {dest} on port {destination_port}.')
                        dest = self[dest]
                        index = get_port_index(self.connections[unit].destinations[port], dest, destination_port)
                        output_state[index] = output_value
                except AttributeError:
                    destination_port = 0
                    try:
                        assert self.connection_exists(unit, dest, port, destination_port)
                    except AssertionError:
                        raise CADETProcessError(f'{unit} on port {port} does not connect to {dest} on port {destination_port}.')
                    dest = self[dest]
                    index = get_port_index(self.connections[unit].destinations[port], dest, destination_port)
                    output_state[index] = value

        elif isinstance(state, list):
            if len(state) != state_length:
                raise CADETProcessError(f'Expected length {state_length}.')

            output_state = state

        else:
            raise TypeError("Output state must be integer, list or dict.")

        if state_length != 0 and not np.isclose(sum(output_state), 1):
            raise CADETProcessError('Sum of fractions must be 1')

        self._output_states[unit][port] = output_state

    def get_flow_rates(self, port=None, state=None):
        """Calculate flow rate for all connections.

        Optionally, an additional output state can be passed to update the
        current output states.

        Parameters
        ----------
        state : Dict, optional
            Updated flow rates and output states for process sections.
            Default is None.

        Returns
        -------
        Dict
            Volumetric flow rate for each unit operation.

        Notes
        -----
        To calculate the flow rates, a system of equations is set up:

        .. math::

            Q_i = \sum_{j=1}^{n_{units}} q_{ji} = \sum_{j=1}^{n_{units}} Q_j * w_{ji},

        where :math:`Q_i` is the total flow exiting unit :math:`i`, and :math:`w_{ij}`
        is the percentile of the total flow of unit :math:`j` directed to unit
        :math:`i`. If the unit is an `Inlet` or a `Cstr` with a given flow rate,
        :math:`Q_i` is given and the system is simplified. This system is solved using
        `numpy.linalg.solve`. Then, the individual flows :math:`q_{ji}` are extracted.

        Raises
        ------
        CADETProcessError
            If flow sheet connectivity matrix is singular, indicating a potential issue
            in flow sheet configuration.

        References
        ----------
        Forum discussion on flow rate calculation:
        https://forum.cadet-web.de/t/improving-the-flowrate-calculation/795
        """
        flow_rates = {
            unit.name: unit.flow_rate
            for unit in (self.inlets + self.cstrs)
            if unit.flow_rate is not None
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

        n_units = self.number_of_units

        # Setup matrix with output states.
        w_out = np.zeros((n_units, n_units))


        #TODO: add Ports
        for unit in self.units:
            unit_index = self.get_unit_index(unit)
            if unit.name in flow_rates:
                w_out[unit_index, unit_index] = 1
            else:
                for origin in self.connections[unit]['origins']:
                    o_index = self.get_unit_index(origin)
                    local_d_index = self.connections[origin].destinations.index(unit)
                    w_out[unit_index, o_index] = output_states[origin][local_d_index]
                w_out[unit_index, unit_index] += -1

        # Check for a singular matrix before the loop
        if np.linalg.cond(w_out) == np.inf:
            raise CADETProcessError(
                "Flow sheet connectivity matrix is singular, which may be due to "
                "unconnected units or missing flow rates. Please ensure all units are "
                "correctly connected and all necessary flow rates are set."
            )

        # Solve system of equations for each polynomial coefficient
        total_flow_rate_coefficents = np.zeros((4, n_units))
        for i in range(4):
            coeffs = np.array(list(flow_rates.values()), ndmin=2)[:, i]
            if not np.any(coeffs):
                continue

            Q_vec = np.zeros(n_units)
            for unit_name in flow_rates:
                unit_index = self.get_unit_index(self.units_dict[unit_name])
                Q_vec[unit_index] = flow_rates[unit_name][i]
            try:
                total_flow_rate_coefficents[i, :] = np.linalg.solve(w_out, Q_vec)
            except np.linalg.LinAlgError:
                raise CADETProcessError(
                    "Unexpected error in flow rate calculation. "
                    "Please check the flow sheet setup."
                )

        # w_out_help is the same as w_out but it contains the origin flow for every unit
        w_out_help = np.zeros((n_units, n_units))

        for unit in self.connections:
            unit_index = self.get_unit_index(unit)
            for origin in self.connections[unit]['origins']:
                o_index = self.get_unit_index(origin)
                local_d_index = self.connections[origin].destinations.index(unit)
                w_out_help[unit_index, o_index] = output_states[origin][local_d_index]

        # Calculate total_in as a matrix in "one" step rather than iterating manually.
        total_in_matrix = w_out_help @ total_flow_rate_coefficents.T

        # Generate output dict
        return_flow_rates = Dict()
        for index, unit in enumerate(self.units):
            unit_solution_dict = Dict()

            if not isinstance(unit, Inlet):
                unit_solution_dict['total_in'] = list(total_in_matrix[index])

            if not isinstance(unit, Outlet):
                unit_solution_dict['total_out'] = list(total_flow_rate_coefficents[:, index])

            if not isinstance(unit, Inlet):
                unit_solution_dict['origins'] = Dict(
                    {
                        origin.name: list(
                            total_flow_rate_coefficents[:, self.get_unit_index(origin)]
                            * w_out_help[index, self.get_unit_index(origin)]
                        )
                        for origin in self.connections[unit].origins
                    }
                )

            if not isinstance(unit, Outlet):
                unit_solution_dict['destinations'] = Dict(
                    {
                        destination.name: list(
                            total_flow_rate_coefficents[:, index]
                            * w_out_help[self.get_unit_index(destination), index]
                        )
                        for destination in self.connections[unit].destinations
                    }
                )

            return_flow_rates[unit.name] = unit_solution_dict

        return return_flow_rates

    def check_flow_rates(self, state=None):
        flow_rates = self.get_flow_rates(state)
        for unit, q in flow_rates.items():
            if isinstance(unit, (Inlet, Outlet)):
                continue
            elif isinstance(unit, Cstr) and Cstr.flow_rate is not None:
                continue

            if not np.all(q.total_in == q.total_out):
                raise CADETProcessError(f"Unbalanced flow rate for unit '{unit}'.")

    @property
    def feed_inlets(self):
        """list: Inlets considered for calculating recovery yield."""
        return self._feed_inlets

    @unit_name_decorator
    def add_feed_inlet(self, feed_inlet):
        """Add inlet to list of units to be considered for recovery.

        Parameters
        ----------
        feed_inlet : SourceMixin
            Unit to be added to list of feed inlets.

        Raises
        ------
        CADETProcessError
            If unit is not an Inlet.
            If unit is already marked as feed inlet.

        """
        if feed_inlet not in self.inlets:
            raise CADETProcessError('Expected Inlet')
        if feed_inlet in self._feed_inlets:
            raise CADETProcessError(
                f'Unit \'{feed_inlet}\' is already a feed inlet.'
            )
        self._feed_inlets.append(feed_inlet)

    @unit_name_decorator
    def remove_feed_inlet(self, feed_inlet):
        """Remove inlet from list of units to be considered for recovery.

        Parameters
        ----------
        feed_inlet : SourceMixin
            Unit to be removed from list of feed inlets.

        """
        if feed_inlet not in self._feed_inlets:
            raise CADETProcessError(
                f'Unit \'{feed_inlet}\' is not a feed inlet.'
            )
        self._feed_inlets.remove(feed_inlet)

    @property
    def eluent_inlets(self):
        """list: Inlets to be considered for eluent consumption."""
        return self._eluent_inlets

    @unit_name_decorator
    def add_eluent_inlet(self, eluent_inlet):
        """Add inlet to list of units to be considered for eluent consumption.

        Parameters
        ----------
        eluent_inlet : SourceMixin
            Unit to be added to list of eluent inlets.

        Raises
        ------
        CADETProcessError
            If unit is not an Inlet.
            If unit is already marked as eluent inlet.

        """
        if eluent_inlet not in self.inlets:
            raise CADETProcessError('Expected Inlet')
        if eluent_inlet in self._eluent_inlets:
            raise CADETProcessError(
                f'Unit \'{eluent_inlet}\' is already an eluent inlet'
            )
        self._eluent_inlets.append(eluent_inlet)

    @unit_name_decorator
    def remove_eluent_inlet(self, eluent_inlet):
        """Remove inlet from list of units considered for eluent consumption.

        Parameters
        ----------
        eluent_inlet : SourceMixin
            Unit to be added to list of eluent inlets.

        Raises
        ------
        CADETProcessError
            If unit is not in eluent inlets.

        """
        if eluent_inlet not in self._eluent_inlets:
            raise CADETProcessError(
                f'Unit \'{eluent_inlet}\' is not an eluent inlet.'
            )
        self._eluent_inlets.remove(eluent_inlet)

    @property
    def product_outlets(self):
        """list: Outlets to be considered for fractionation."""
        return self._product_outlets

    @unit_name_decorator
    def add_product_outlet(self, product_outlet):
        """Add outlet to list of units considered for fractionation.

        Parameters
        ----------
        product_outlet : Outlet
            Unit to be added to list of product outlets.

        Raises
        ------
        CADETProcessError
            If unit is not an Outlet.
            If unit is already marked as product outlet.

        """
        if product_outlet not in self.outlets:
            raise CADETProcessError('Expected Outlet')
        if product_outlet in self._product_outlets:
            raise CADETProcessError(
                f'Unit \'{product_outlet}\' is already a product outlet'
            )
        self._product_outlets.append(product_outlet)

    @unit_name_decorator
    def remove_product_outlet(self, product_outlet):
        """Remove outlet from list of units to be considered for fractionation.

        Parameters
        ----------
        product_outlet : Outlet
            Unit to be added to list of product outlets.

        Raises
        ------
        CADETProcessError
            If unit is not a product outlet.

        """
        if product_outlet not in self._product_outlets:
            raise CADETProcessError(
                f'Unit \'{product_outlet}\' is not a product outlet.'
            )
        self._product_outlets.remove(product_outlet)

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
    def sized_parameters(self):
        return self._sized_parameters

    @property
    def polynomial_parameters(self):
        return self._polynomial_parameters

    @property
    def section_dependent_parameters(self):
        return self._section_dependent_parameters

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
            UnitOperation of FlowSheet.

        Raises
        ------
        KeyError
            If unit not in FlowSheet

        """
        try:
            return self.units_dict[unit_name]
        except KeyError:
            raise KeyError('Not a valid unit')

    def __contains__(self, item):
        """Check if UnitOperation is part of the FlowSheet.

        Parameters
        ----------
        item : UnitBaseClass
            item to be searched

        Returns
        -------
        Bool : True if item is in units, otherwise False.

        """
        if (item in self._units) or (item in self.unit_names):
            return True
        else:
            return False

        def __iter__(self):
            yield from self.units
