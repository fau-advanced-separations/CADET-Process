from __future__ import annotations

from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Iterator, Optional
from warnings import warn

import numpy as np
from addict import Dict

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import String, Structure, frozen_attributes

from .binding import NoBinding
from .componentSystem import ComponentSystem
from .unitOperation import Cstr, Inlet, Outlet, SourceMixin, UnitBaseClass


@frozen_attributes
class FlowSheet(Structure):
    """
    Class to design process flow sheet.

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

    def __init__(
        self,
        component_system: ComponentSystem,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize flow sheet."""
        super().__init__(*args, **kwargs)

        self.component_system = component_system

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
    def component_system(self) -> ComponentSystem:
        """ComponentSystem: The component system of the flow sheet."""
        return self._component_system

    @component_system.setter
    def component_system(self, component_system: ComponentSystem) -> None:
        if not isinstance(component_system, ComponentSystem):
            raise TypeError("Expected ComponentSystem")
        self._component_system = component_system

    @property
    def n_comp(self) -> int:
        """int: The number of components."""
        return self.component_system.n_comp

    def unit_name_decorator(func: Callable) -> Callable:
        """Wrap methods to enable calling functions with unit object or unit name."""

        @wraps(func)
        def unit_name_wrapper(
            self: FlowSheet,
            unit: str | UnitBaseClass,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            """Enable calling functions with unit object or unit name."""
            if isinstance(unit, str):
                try:
                    unit = self.units_dict[unit]
                except KeyError:
                    raise CADETProcessError("Not a valid unit")
            return func(self, unit, *args, **kwargs)

        return unit_name_wrapper

    def origin_destination_name_decorator(func: Callable) -> Callable:
        """Wrap methods to enable calling functions using origin and destination units."""

        @wraps(func)
        def origin_destination_name_wrapper(
            self: FlowSheet,
            origin: str | UnitBaseClass,
            destination: str | UnitBaseClass,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            """Enable calling origin and destination using unit names."""
            if isinstance(origin, str):
                try:
                    origin = self.units_dict[origin]
                except KeyError:
                    raise CADETProcessError("Not a valid unit")

            if isinstance(destination, str):
                try:
                    destination = self.units_dict[destination]
                except KeyError:
                    raise CADETProcessError("Not a valid unit")

            return func(self, origin, destination, *args, **kwargs)

        return origin_destination_name_wrapper

    def update_parameters(self) -> None:
        """Update current parameters."""
        for unit in self.units:
            self._parameters[unit.name] = unit.parameters
            self._section_dependent_parameters[unit.name] = (
                unit.section_dependent_parameters
            )
            self._polynomial_parameters[unit.name] = unit.polynomial_parameters
            self._sized_parameters[unit.name] = unit.sized_parameters

        self._parameters["output_states"] = {
            unit.name: self.output_states[unit] for unit in self.units
        }

        self._sized_parameters["output_states"] = {
            unit.name: self.output_states[unit] for unit in self.units
        }

        self._section_dependent_parameters["output_states"] = {
            unit.name: self.output_states[unit] for unit in self.units
        }

    def update_parameters_decorator(func: Callable) -> Callable:
        """Wrap method s.t. parameters dict is automatically updated."""

        @wraps(func)
        def wrapper(self: FlowSheet, *args: Any, **kwargs: Any) -> Any:
            """Update parameters dict to save time."""
            results = func(self, *args, **kwargs)
            self.update_parameters()

            return results

        return wrapper

    @property
    def units(self) -> list[UnitBaseClass]:
        """list: list of all unit_operations in the flow sheet."""
        return self._units

    @property
    def units_dict(self) -> dict[str, UnitBaseClass]:
        """dict: Unit operation names and objects."""
        return {unit.name: unit for unit in self.units}

    @property
    def unit_names(self) -> list[str]:
        """list: Names of unit operations."""
        return [unit.name for unit in self.units]

    @property
    def number_of_units(self) -> int:
        """int: Number of unit operations in the FlowSheet."""
        return len(self._units)

    @unit_name_decorator
    def get_unit_index(self, unit: UnitBaseClass) -> int:
        """
        Return the unit index of the unit.

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
            raise CADETProcessError("Unit not in flow sheet")

        return self.units.index(unit)

    def get_port_index(self, unit: UnitBaseClass, port: str) -> int:
        """
        Return the port index of a unit.

        Parameters
        ----------
        unit : UnitBaseClass
            UnitBaseClass object of wich port index is to be returned
        port : string
            Name of port which index is to be returned

        Raises
        ------
        CADETProcessError
            If unit or port is not in the current flow sheet.

        Returns
        -------
        port_index : int
            Returns the port index of the port of the unit_operation.
        """
        if unit not in self.units:
            raise CADETProcessError("Unit not in flow sheet")

        port_index = unit.ports.index(port)

        return port_index

    @property
    def inlets(self) -> list[Inlet]:
        """list: All Inlets in the system."""
        return [unit for unit in self._units if isinstance(unit, Inlet)]

    @property
    def outlets(self) -> list[Outlet]:
        """list: All Outlets in the system."""
        return [unit for unit in self._units if isinstance(unit, Outlet)]

    @property
    def cstrs(self) -> list[Cstr]:
        """list: All Cstrs in the system."""
        return [unit for unit in self._units if isinstance(unit, Cstr)]

    @property
    def units_with_binding(self) -> list[UnitBaseClass]:
        """list: UnitOperations with binding models."""
        return [
            unit
            for unit in self._units
            if not isinstance(unit.binding_model, NoBinding)
        ]

    @update_parameters_decorator
    def add_unit(
        self,
        unit: UnitBaseClass,
        feed_inlet: bool = False,
        eluent_inlet: bool = False,
        product_outlet: bool = False,
    ) -> None:
        """
        Add unit to the flow sheet.

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
            raise TypeError("Expected UnitOperation")

        if unit in self._units or unit.name in self.unit_names:
            raise CADETProcessError("Unit already part of System")

        if unit.component_system is not self.component_system:
            raise CADETProcessError("Component systems do not match.")

        self._units.append(unit)

        for port in unit.ports:
            if isinstance(unit, Inlet):
                self._connections[unit]["origins"] = None
                self._connections[unit]["destinations"][port] = defaultdict(list)

            elif isinstance(unit, Outlet):
                self._connections[unit]["origins"][port] = defaultdict(list)
                self._connections[unit]["destinations"] = None

            else:
                self._connections[unit]["origins"][port] = defaultdict(list)
                self._connections[unit]["destinations"][port] = defaultdict(list)

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
    def remove_unit(self, unit: UnitBaseClass) -> None:
        """
        Remove unit from flow sheet.

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
            raise CADETProcessError("Unit not in flow sheet")

        if unit is self.feed_inlets:
            self.remove_feed_inlet(unit)
        if unit is self.eluent_inlets:
            self.remove_eluent_inlet(unit)
        if unit is self.product_outlets:
            self.remove_product_outlet(unit)

        origins = []
        destinations = []

        if self._connections[unit]["origins"] is not None:
            origins = [
                origin
                for ports in self._connections[unit]["origins"]
                for origin in self._connections[unit]["origins"][ports]
            ]

        if self._connections[unit]["destinations"] is not None:
            destinations = [
                destination
                for ports in self._connections[unit]["destinations"]
                for destination in self._connections[unit]["destinations"][ports]
            ].copy()

        for origin in origins:
            for origin_port in self._connections[unit]["origins"]:
                for unit_port in self._connections[unit]["origins"][origin_port][origin]:
                    self.remove_connection(origin, unit, origin_port, unit_port)

        for destination in destinations:
            for destination_port in self._connections[unit]["destinations"]:
                for unit_port in self._connections[unit]["destinations"][destination_port][destination]:  # noqa: E501
                    self.remove_connection(unit, destination, unit_port, destination_port)

        self._units.remove(unit)
        self._connections.pop(unit)
        self._output_states.pop(unit)
        self.__dict__.pop(unit.name)

    @property
    def connections(self) -> Dict:
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
        self,
        origin: UnitBaseClass,
        destination: UnitBaseClass,
        origin_port: Optional[str] = None,
        destination_port: Optional[str] = None,
    ) -> None:
        """
        Add connection between units 'origin' and 'destination'.

        Parameters
        ----------
        origin : UnitBaseClass
            UnitBaseClass from which the connection originates.
        destination : UnitBaseClass
            UnitBaseClass where the connection terminates.
        origin_port : str, optional
            Port from which connection originates.
        destination_port : str, optional
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
            raise CADETProcessError("Origin not in flow sheet")
        if isinstance(origin, Outlet):
            raise CADETProcessError("Outlet unit cannot have outgoing stream.")

        if destination not in self._units:
            raise CADETProcessError("Destination not in flow sheet")
        if isinstance(destination, Inlet):
            raise CADETProcessError("Inlet unit cannot have ingoing stream.")

        if origin.has_ports and origin_port is None:
            raise CADETProcessError("Missing `origin_port`")
        if not origin.has_ports and origin_port is not None:
            raise CADETProcessError("Origin unit does not support ports.")
        if origin.has_ports and origin_port not in origin.ports:
            raise CADETProcessError(
                f'Origin port "{origin_port}" not found in ports: {origin.ports}.'
            )
        if (
            origin_port
            in self._connections[destination]["origins"][destination_port][origin]
        ):
            raise CADETProcessError("Connection already exists")

        if destination.has_ports and destination_port is None:
            raise CADETProcessError("Missing `destination_port`")
        if not destination.has_ports and destination_port is not None:
            raise CADETProcessError("Destination unit does not support ports.")
        if destination.has_ports and destination_port not in destination.ports:
            raise CADETProcessError(
                f'destination port "{destination_port}" not found in ports: {destination.ports}.'
            )

        if (
            destination_port
            in self._connections[origin]["destinations"][origin_port][destination]
        ):
            raise CADETProcessError("Connection already exists")

        if not destination.has_ports:
            destination_port = destination.ports[0]
        if not origin.has_ports:
            origin_port = origin.ports[0]

        self._connections[destination]["origins"][destination_port][origin].append(
            origin_port
        )
        self._connections[origin]["destinations"][origin_port][destination].append(
            destination_port
        )

        self.set_output_state(origin, 0, origin_port)

    @origin_destination_name_decorator
    @update_parameters_decorator
    def remove_connection(
        self,
        origin: UnitBaseClass,
        destination: UnitBaseClass,
        origin_port: Optional[str] = None,
        destination_port: Optional[str] = None,
    ) -> None:
        """
        Remove connection between units 'origin' and 'destination'.

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
            raise CADETProcessError("Origin not in flow sheet")
        if origin.has_ports and origin_port is None:
            raise CADETProcessError("Missing `origin_port`")

        if origin_port not in origin.ports:
            raise CADETProcessError(
                f"Origin port {origin_port} not in Unit {origin.name}."
            )

        if destination not in self._units:
            raise CADETProcessError("Destination not in flow sheet")
        if destination.has_ports and origin_port is None:
            raise CADETProcessError("Missing `destination_port`")

        if destination_port not in destination.ports:
            raise CADETProcessError(
                f"Origin port {destination_port} not in Unit {destination.name}."
            )

        try:
            self._connections[destination]["origins"][destination_port].pop(origin)
            self._connections[origin]["destinations"][origin_port].pop(destination)
        except KeyError:
            raise CADETProcessError("Connection does not exist.")

    @origin_destination_name_decorator
    def connection_exists(
        self,
        origin: UnitBaseClass,
        destination: UnitBaseClass,
        origin_port: Optional[str] = None,
        destination_port: Optional[str] = None,
    ) -> bool:
        """
        bool: check if connection exists in flow sheet.

        Parameters
        ----------
        origin : UnitBaseClass
            UnitBaseClass from which the connection originates.
        destination : UnitBaseClass
            UnitBaseClass where the connection terminates.
        origin_port : Port, optional
            If origin unit operation has ports, origin port can be specified.
        destination_port : Port optional
            if destination unit operation has ports, destination port can be specified.

        """
        if origin.has_ports and not origin_port:
            raise CADETProcessError(
                "Origin port needs to be specified for unit operation with ports "
                f"{origin.name}."
            )

        if destination.has_ports and not destination_port:
            raise CADETProcessError(
                "Destination port needs to be specified for unit operation with ports "
                f"{destination.name}."
            )

        if origin_port not in origin.ports:
            raise CADETProcessError(f"{origin.name} does not have port {origin_port}")

        if destination_port not in destination.ports:
            raise CADETProcessError(
                f"{destination.name} does not have port {destination_port}"
            )

        if (
            destination in self._connections[origin].destinations[origin_port]
            and
            destination_port in self._connections[origin].destinations[origin_port][destination]
            and
            origin in self._connections[destination].origins[destination_port]
            and
            origin_port in self._connections[destination].origins[destination_port][origin]
        ):
            return True

        return False

    def check_connections(self) -> bool:
        """
        Validate that units are connected correctly.

        Warning:
        -------
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
                if connections.origins is not None:
                    flag = False
                    warn("Inlet unit cannot have ingoing stream.")
                if len(connections.destinations) == 0:
                    flag = False
                    warn(f" Unit '{unit.name}' does not have outgoing stream.")
            elif isinstance(unit, Outlet):
                if connections.destinations is not None:
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
                if all(
                    len(port_list) == 0 for port_list in connections.origins.values()
                ):
                    flag = False
                    warn(f"Unit '{unit.name}' does not have ingoing stream.")
                if all(
                    len(port_list) == 0
                    for port_list in connections.destinations.values()
                ):
                    flag = False
                    warn(f" Unit '{unit.name}' does not have outgoing stream.")

        return flag

    @property
    def missing_parameters(self) -> list[str]:
        """dict: Missing parameters of the flow sheet."""
        missing_parameters = []

        for unit in self.units:
            missing_parameters += [
                f"{unit.name}.{param}" for param in unit.missing_parameters
            ]

        return missing_parameters

    def check_units_config(self) -> bool:
        """
        Check if units are configured correctly.

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
    def output_states(self) -> Dict:
        """dict: Output states of the unit operations."""
        output_states_dict = self._output_states.copy()

        for unit, ports in output_states_dict.items():
            for port in ports:
                if port is None:
                    output_states_dict[unit] = output_states_dict[unit][port]

        return output_states_dict

    @unit_name_decorator
    @update_parameters_decorator
    def set_output_state(
        self,
        unit: UnitBaseClass,
        state: int | list[float] | dict,
        port: Optional[str] = None,
    ) -> None:
        """
        Set split ratio of outgoing streams for UnitOperation.

        Parameters
        ----------
        unit : UnitBaseClass
            UnitOperation of flowsheet.
        state : int or list of floats or dict
            new output state of the unit.
        port : str
            Port for which to set the output state.

        Raises
        ------
        CADETProcessError
            If unit not in FlowSheet
            If state is integer and the state >= the state_length.
            If the length of the states is unequal the state_length.
            If the sum of the states is not equal to 1.
            If port cannot be found in the unit operation.
        """

        def get_port_index(
            unit_connection_dict: Dict,
            destination: UnitBaseClass,
            destination_port: str,
        ) -> int:
            """
            Compute the index of a connection for the output state.

            Parameters
            ----------
            unit_connection_dict : Defaultdict
                contains dict with connected units and their respective ports
            destination : UnitBaseClass
                destination object
            destination_port : str
                destination Port
            """
            ret_index = 0
            for unit_destination in unit_connection_dict:
                if unit_destination is destination:
                    ret_index += unit_connection_dict[unit_destination].index(
                        destination_port
                    )
                    break
                ret_index += len(unit_connection_dict[unit_destination])
            return ret_index

        if unit not in self._units:
            raise CADETProcessError("Unit not in flow sheet")

        if port not in unit.ports:
            raise CADETProcessError(f"Port {port} is not a port of Unit {unit.name}")

        state_length = sum([
            len(self._connections[unit].destinations[port][unit_name])
            for unit_name in self._connections[unit].destinations[port]
        ])

        if state_length == 0:
            output_state = []

        if isinstance(state, (int, np.integer)):
            if state >= state_length:
                raise CADETProcessError("Index exceeds destinations")

            output_state = [0.0] * state_length
            output_state[state] = 1.0

        elif isinstance(state, dict):
            output_state = [0.0] * state_length
            for dest, value in state.items():
                try:
                    for destination_port, output_value in value.items():
                        if not self.connection_exists(unit, dest, port, destination_port):
                            raise CADETProcessError(
                                f"{unit} on port {port} does not connect to {dest} on "
                                f"port {destination_port}."
                            )
                        inner_dest = self[dest]
                        index = get_port_index(
                            self._connections[unit].destinations[port],
                            inner_dest,
                            destination_port,
                        )
                        output_state[index] = output_value
                except AttributeError:
                    destination_port = None

                    if not self.connection_exists(unit, dest, port, destination_port):
                        raise CADETProcessError(
                            f"{unit} on port {port} does not connect to {dest} on port "
                            f"{destination_port}."
                        )
                    dest = self[dest]
                    index = get_port_index(
                        self.connections[unit].destinations[port],
                        dest,
                        destination_port,
                    )
                    output_state[index] = value

        elif isinstance(state, (list)):
            if len(state) != state_length:
                raise CADETProcessError(f"Expected length {state_length}.")

            output_state = state
        elif isinstance(state, np.ndarray):
            if len(state) != state_length:
                raise CADETProcessError(f"Expected length {state_length}.")

            output_state = list(state)

        else:
            raise TypeError("Output state must be integer, list or dict.")

        if state_length != 0 and not np.isclose(sum(output_state), 1):
            raise CADETProcessError("Sum of fractions must be 1")

        self._output_states[unit][port] = output_state

    def get_flow_rates(self, state: Optional[Dict] = None, eps: float = 5.9e16) -> Dict:
        r"""
        Calculate flow rate for all connections.

        Optionally, an additional output state can be passed to update the
        current output states.

        Parameters
        ----------
        state : Dict, optional
            Updated flow rates and output states for process sections.
            Default is None.
        eps : float, optional
            eps as an upper boarder for condition of flow_rate calculation

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
        port_index_list = []
        port_number = 0

        for unit in self.units:
            for port in unit.ports:
                port_index_list.append((unit, port))
                port_number += 1

        flow_rates = {
            unit.name: unit.flow_rate
            for unit in (self.inlets + self.cstrs)
            if unit.flow_rate is not None
        }

        output_states = self._output_states

        if state is not None:
            for param, value in state.items():
                param = param.split(".")
                param_name = param[-1]
                if param_name == "flow_rate":
                    unit_name = param[1]
                    flow_rates[unit_name] = value[0]
                elif param[1] == "output_states":
                    unit_name = param[2]
                    unit = self.units_dict[unit_name]
                    if unit.has_ports:
                        port = param[2]
                    else:
                        port = None
                    output_states[unit][port] = list(value.ravel())

        # Setup matrix with output states.
        w_out = np.zeros((port_number, port_number))

        for unit in self.units:
            if unit.name in flow_rates:
                unit_index = port_index_list.index((unit, None))
                w_out[unit_index, unit_index] = 1
            else:
                for port in self._connections[unit]["origins"]:
                    port_index = port_index_list.index((unit, port))

                    for origin_unit in self._connections[unit]["origins"][port]:
                        for origin_port in self._connections[unit]["origins"][port][origin_unit]:
                            o_index = port_index_list.index((origin_unit, origin_port))
                            local_d_index = 0

                            for inner_unit in self._connections[origin_unit][
                                "destinations"
                            ][origin_port]:
                                if inner_unit == unit:
                                    break
                                local_d_index += len(list(
                                    self._connections[origin_unit]["destinations"][origin_port][inner_unit]
                                ))

                            local_d_index += \
                                self._connections[origin_unit]["destinations"][origin_port][unit].index(port)

                            if output_states[origin_unit][origin_port][local_d_index]:
                                w_out[port_index, o_index] = \
                                    output_states[origin_unit][origin_port][local_d_index]

                    w_out[port_index, port_index] += -1

        # Check for a singular matrix before the loop
        if np.linalg.cond(w_out) > eps:
            raise CADETProcessError(
                "Flow sheet connectivity matrix is singular, which may be due to "
                "unconnected units or missing flow rates. Please ensure all units are "
                "correctly connected and all necessary flow rates are set."
            )

        # Solve system of equations for each polynomial coefficient
        total_flow_rate_coefficents = np.zeros((4, port_number))
        for i in range(4):
            if len(flow_rates) == 0:
                continue

            coeffs = np.array(list(flow_rates.values()), ndmin=2)[:, i]
            if not np.any(coeffs):
                continue

            Q_vec = np.zeros(port_number)
            for unit_name in flow_rates:
                port_index = port_index_list.index((self.units_dict[unit_name], None))
                Q_vec[port_index] = flow_rates[unit_name][i]
            try:
                total_flow_rate_coefficents[i, :] = np.linalg.solve(w_out, Q_vec)
            except np.linalg.LinAlgError:
                raise CADETProcessError(
                    "Unexpected error in flow rate calculation. "
                    "Please check the flow sheet setup."
                )

        # w_out_help is the same as w_out but it contains the origin flow for every unit
        w_out_help = np.zeros((port_number, port_number))

        for unit in self.units:
            if self._connections[unit]["origins"]:
                for port in self._connections[unit]["origins"]:
                    port_index = port_index_list.index((unit, port))

                    for origin_unit in self._connections[unit]["origins"][port]:
                        for origin_port in self._connections[unit]["origins"][port][origin_unit]:
                            o_index = port_index_list.index((origin_unit, origin_port))
                            local_d_index = 0

                            for inner_unit in self._connections[origin_unit]["destinations"][origin_port]:  # noqa: E501
                                if inner_unit == unit:
                                    break
                                local_d_index += len(list(
                                    self._connections[origin_unit]["destinations"][origin_port][inner_unit]
                                ))

                            local_d_index += \
                                self._connections[origin_unit]["destinations"][origin_port][unit].index(port)

                            if not output_states[origin_unit][origin_port][local_d_index]:
                                w_out_help[port_index, o_index] = 0
                            else:
                                w_out_help[port_index, o_index] = \
                                    output_states[origin_unit][origin_port][local_d_index]

        # Calculate total_in as a matrix in "one" step rather than iterating manually.
        total_in_matrix = w_out_help @ total_flow_rate_coefficents.T

        # Generate output dict
        return_flow_rates = Dict()
        for unit in self.units:
            unit_solution_dict = Dict()

            if not isinstance(unit, Inlet):
                unit_solution_dict["total_in"] = Dict()

                for unit_port in unit.ports:
                    index = port_index_list.index((unit, unit_port))
                    unit_solution_dict["total_in"][unit_port] = \
                        list(total_in_matrix[index])

            if not isinstance(unit, Outlet):
                unit_solution_dict["total_out"] = Dict()

                for unit_port in unit.ports:
                    index = port_index_list.index((unit, unit_port))
                    unit_solution_dict["total_out"][unit_port] = \
                        list(total_flow_rate_coefficents[:, index])

            if not isinstance(unit, Inlet):
                unit_solution_dict["origins"] = Dict()

                for unit_port in self._connections[unit]["origins"]:
                    if self._connections[unit]["origins"][unit_port]:
                        unit_solution_dict["origins"][unit_port] = Dict()

                        for origin_unit in self._connections[unit]["origins"][unit_port]:
                            unit_solution_dict["origins"][unit_port][origin_unit.name] = Dict()

                            for origin_port in self._connections[unit]["origins"][unit_port][origin_unit]:  # noqa: E501
                                origin_port_index = \
                                    port_index_list.index((origin_unit, origin_port))
                                unit_port_index = port_index_list.index((unit, unit_port))
                                flow_list = list(
                                    total_flow_rate_coefficents[:, origin_port_index]
                                    * w_out_help[unit_port_index, origin_port_index]
                                )
                                unit_solution_dict["origins"][unit_port][origin_unit.name][origin_port] = flow_list  # noqa: E501

            if not isinstance(unit, Outlet):
                unit_solution_dict["destinations"] = Dict()

                for unit_port in self._connections[unit]["destinations"]:
                    if self._connections[unit]["destinations"][unit_port]:
                        unit_solution_dict["destinations"][unit_port] = Dict()

                        for destination_unit in self._connections[unit]["destinations"][unit_port]:
                            unit_solution_dict["destinations"][unit_port][destination_unit.name] = Dict()  # noqa 501

                            for destination_port in self._connections[unit]["destinations"][unit_port][destination_unit]:  # noqa E501
                                destination_port_index = port_index_list.index((destination_unit, destination_port))  # noqa 501
                                unit_port_index = port_index_list.index((unit, unit_port))
                                flow_list = list(
                                    total_flow_rate_coefficents[:, unit_port_index]
                                    * w_out_help[destination_port_index, unit_port_index]
                                )
                                unit_solution_dict["destinations"][unit_port][destination_unit.name][destination_port] = flow_list  # noqa 501

            return_flow_rates[unit.name] = unit_solution_dict

        return return_flow_rates

    def check_flow_rates(self, state: Optional[dict] = None) -> None:
        """Check if in and outgoing flow rates of unit operations are balanced."""
        flow_rates = self.get_flow_rates(state)
        for unit, q in flow_rates.items():
            if isinstance(unit, (Inlet, Outlet)):
                continue
            elif isinstance(unit, Cstr) and Cstr.flow_rate is not None:
                continue

            if not np.all(q.total_in == q.total_out):
                raise CADETProcessError(f"Unbalanced flow rate for unit '{unit}'.")

    @property
    def feed_inlets(self) -> list[UnitBaseClass]:
        """list: Inlets considered for calculating recovery yield."""
        return self._feed_inlets

    @unit_name_decorator
    def add_feed_inlet(self, feed_inlet: SourceMixin) -> None:
        """
        Add inlet to list of units to be considered for recovery.

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
            raise CADETProcessError("Expected Inlet")
        if feed_inlet in self._feed_inlets:
            raise CADETProcessError(f"Unit '{feed_inlet}' is already a feed inlet.")
        self._feed_inlets.append(feed_inlet)

    @unit_name_decorator
    def remove_feed_inlet(self, feed_inlet: SourceMixin) -> None:
        """
        Remove inlet from list of units to be considered for recovery.

        Parameters
        ----------
        feed_inlet : SourceMixin
            Unit to be removed from list of feed inlets.
        """
        if feed_inlet not in self._feed_inlets:
            raise CADETProcessError(f"Unit '{feed_inlet}' is not a feed inlet.")
        self._feed_inlets.remove(feed_inlet)

    @property
    def eluent_inlets(self) -> list[UnitBaseClass]:
        """list: Inlets to be considered for eluent consumption."""
        return self._eluent_inlets

    @unit_name_decorator
    def add_eluent_inlet(self, eluent_inlet: SourceMixin) -> None:
        """
        Add inlet to list of units to be considered for eluent consumption.

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
            raise CADETProcessError("Expected Inlet")
        if eluent_inlet in self._eluent_inlets:
            raise CADETProcessError(f"Unit '{eluent_inlet}' is already an eluent inlet")
        self._eluent_inlets.append(eluent_inlet)

    @unit_name_decorator
    def remove_eluent_inlet(self, eluent_inlet: SourceMixin) -> None:
        """
        Remove inlet from list of units considered for eluent consumption.

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
            raise CADETProcessError(f"Unit '{eluent_inlet}' is not an eluent inlet.")
        self._eluent_inlets.remove(eluent_inlet)

    @property
    def product_outlets(self) -> list[UnitBaseClass]:
        """list: Outlets to be considered for fractionation."""
        return self._product_outlets

    @unit_name_decorator
    def add_product_outlet(self, product_outlet: Outlet) -> None:
        """
        Add outlet to list of units considered for fractionation.

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
            raise CADETProcessError("Expected Outlet")
        if product_outlet in self._product_outlets:
            raise CADETProcessError(
                f"Unit '{product_outlet}' is already a product outlet"
            )
        self._product_outlets.append(product_outlet)

    @unit_name_decorator
    def remove_product_outlet(self, product_outlet: Outlet) -> None:
        """
        Remove outlet from list of units to be considered for fractionation.

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
            raise CADETProcessError(f"Unit '{product_outlet}' is not a product outlet.")
        self._product_outlets.remove(product_outlet)

    @property
    def parameters(self) -> dict:
        """dict: Parameters of the flow sheet and associated unit operations."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        try:
            output_states = parameters.pop("output_states")
            for unit, state in output_states.items():
                unit = self.units_dict[unit]
                # Hier, if unit.has_ports: iterate.
                if not unit.has_ports:
                    self.set_output_state(unit, state)
                else:
                    for port_i, state_i in state.items():
                        self.set_output_state(unit, state_i, port_i)
        except KeyError:
            pass

        for unit, params in parameters.items():
            if unit not in self.units_dict:
                raise CADETProcessError("Not a valid unit")
            self.units_dict[unit].parameters = params

        self.update_parameters()

    @property
    def sized_parameters(self) -> list[str]:
        """list: List of sized parameters."""
        return self._sized_parameters

    @property
    def polynomial_parameters(self) -> list[str]:
        """list: List of polynomial parameters."""
        return self._polynomial_parameters

    @property
    def section_dependent_parameters(self) -> list[str]:
        """list: List of section dependent parameters."""
        return self._section_dependent_parameters

    @property
    def initial_state(self) -> dict:
        """dict: Initial state of the unit oeprations."""
        initial_state = {unit.name: unit.initial_state for unit in self.units}

        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state: dict) -> None:
        for unit, st in initial_state.items():
            if unit not in self.units_dict:
                raise CADETProcessError("Not a valid unit")
            self.units_dict[unit].initial_state = st

    def __len__(self) -> int:
        """int: The number of unit operations in the system."""
        return self.number_of_units

    def __iter__(self) -> Iterator[UnitBaseClass]:
        """Iterate over the unit operations in the system."""
        return iter(self.units)

    def __getitem__(self, unit_name: str) -> UnitBaseClass:
        """
        Make FlowSheet substriptable s.t. units can be used as keys.

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
            raise KeyError("Not a valid unit")

    def __contains__(self, item: UnitBaseClass | str) -> bool:
        """
        Check if UnitOperation is part of the FlowSheet.

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
