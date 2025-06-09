import warnings
from copy import deepcopy
from functools import wraps
from typing import Any, NoReturn, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from addict import Dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from CADETProcess import CADETProcessError, SimulationResults, plotting
from CADETProcess.dataStructure import (
    Integer,
    Structure,
    UnsignedFloat,
    UnsignedInteger,
)
from CADETProcess.processModel import (
    BindingBaseClass,
    ComponentSystem,
    Cstr,
    FlowSheet,
    Inlet,
    Langmuir,
    Linear,
    Outlet,
    Process,
    TubularReactor,
    TubularReactorBase,
    UnitBaseClass,
)
from CADETProcess.solution import SolutionBase

__all__ = [
    "SerialZone",
    "ParallelZone",
    "CarouselBuilder",
    "SMBBuilder",
    "LinearSMBBuilder",
    "LangmuirSMBBuilder",
]


class ZoneBaseClass(UnitBaseClass):
    """
    Base class for a multi-column zone with configurable columns and flow directions.

    Attributes
    ----------
    n_columns : UnsignedInteger
        The number of columns in the zone.
    flow_direction : Integer
        The flow direction in the zone, where 1 indicates normal and -1 indicates
        reverse flow.
    """

    n_columns = UnsignedInteger()
    flow_direction = Integer(default=1)

    def __init__(
        self,
        component_system: ComponentSystem,
        name: str,
        n_columns: int = 1,
        flow_direction: int = 1,
        initial_state: Optional[list] = None,
        valve_parameters: Optional[dict] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a ZoneBaseClass instance.

        Parameters
        ----------
        component_system : Any
            The component system associated with this zone.
        name : str
            The name of the zone.
        n_columns : int, optional
            Number of columns in the zone.
        flow_direction : int, optional
            Flow direction in the zone.
        initial_state : list, optional
            Initial state of the zone.
        valve_parameters : dict
            Additional parameters to setup mixer/splitter valves.
        *args : Optional
            Additional Parameters passed down to UnitBaseClass
        **kwargs : Optional
            Additional Parameters passed down to UnitBaseClass
        """
        self.n_columns = n_columns
        self.flow_direction = flow_direction
        self.initial_state = initial_state

        valve_parameters = valve_parameters or {}

        self._inlet_unit = self._setup_valve(
            component_system, f"{name}_inlet", **valve_parameters
        )
        self._outlet_unit = self._setup_valve(
            component_system, f"{name}_outlet", **valve_parameters
        )

        super().__init__(component_system, name, *args, **kwargs)

    @property
    def initial_state(self) -> list[dict[str, list]]:
        """list: The initial state of the columns in the zone."""
        return self._initial_state

    @initial_state.setter
    def initial_state(
        self,
        initial_state: list[dict[str, list]] | dict[str, list],
    ) -> NoReturn:
        if initial_state is None:
            self._initial_state = initial_state
            return

        if not isinstance(initial_state, list):
            initial_state = self.n_columns * [initial_state]

        if len(initial_state) != self.n_columns:
            raise CADETProcessError(f"Expected size {self.n_columns}")

        self._initial_state = initial_state

    def _setup_valve(
        self,
        component_system: ComponentSystem,
        name: str,
        unit_type: Union[Cstr, TubularReactor] = Cstr,
        **valve_parameters: dict,
    ) -> None:
        if unit_type not in (Cstr, TubularReactor):
            raise ValueError(
                "Unknown unit type. Must be one of `Cstr`, `TubularReactor`."
            )

        valve = unit_type(component_system, name)

        valve_dead_volume = valve_parameters.get("valve_dead_volume", 1e-9)

        if isinstance(valve, Cstr):
            valve.init_liquid_volume = valve_dead_volume
        elif isinstance(valve, TubularReactor):
            valve.discretization.ncol = 1

            length_diameter_ratio = valve_parameters.get("length_diameter_ratio", 1)
            valve.diameter = (
                4 / np.pi * valve_dead_volume / length_diameter_ratio
            ) ** (1 / 3)
            valve.length = length_diameter_ratio * valve.diameter

            valve.axial_dispersion = valve_parameters.get("axial_dispersion", 0)

        return valve

    @property
    def inlet_unit(self) -> Cstr:
        """Cstr: The inlet CSTR unit of the zone."""
        return self._inlet_unit

    @property
    def outlet_unit(self) -> Cstr:
        """Cstr: The outlet CSTR unit of the zone."""
        return self._outlet_unit


class SerialZone(ZoneBaseClass):
    """Zone with columns connected in series."""

    pass


class ParallelZone(ZoneBaseClass):
    """Zone with columns connected in parallel."""

    pass


class CarouselBuilder(Structure):
    """
    Configurator for multi-column processes.

    Attributes
    ----------
    component_system : Any
        The system of components for which the carousel is configured.
    name : str
        Name of the carousel system.
    switch_time : float
        Column switch time.
    valve_parameters : dict
        Additional parameters to setup mixer/splitter valves.
    """

    switch_time = UnsignedFloat()

    def __init__(
        self,
        component_system: ComponentSystem,
        name: str,
        valve_parameters: dict = None,
    ) -> NoReturn:
        """
        Initialize a CarouselBuilder instance.

        Parameters
        ----------
        component_system : Any
            The system of components that will be used in the carousel.
        name : str
            The carousel name.
        valve_parameters : dict
            Additional parameters to setup mixer/splitter valves.
        """
        self.component_system = component_system
        self.name = name
        self._flow_sheet = FlowSheet(component_system, name)
        self._column = None
        self.valve_parameters = valve_parameters

    @property
    def flow_sheet(self) -> FlowSheet:
        """FlowSheet: The flow sheet instance associated with the carousel."""
        return self._flow_sheet

    @property
    def column(self) -> TubularReactorBase:
        """TubularReactorBase: The column template for all zones."""
        return self._column

    @column.setter
    def column(self, column: TubularReactorBase) -> NoReturn:
        if not isinstance(column, TubularReactorBase):
            raise TypeError("Column must be an instance of TubularReactorBase.")
        if self.component_system is not column.component_system:
            raise CADETProcessError("Number of components does not match.")
        self._column = column

    @wraps(FlowSheet.add_unit)
    def add_unit(self, *args: Any, **kwargs: Any) -> None:
        """Wrap FlowSheet.add_unit to add a unit to the flow sheet."""
        self.flow_sheet.add_unit(*args, **kwargs)

    @wraps(FlowSheet.add_connection)
    def add_connection(self, *args: Any, **kwargs: Any) -> None:
        """Wrap FlowSheet.add_connection to add a connection between units."""
        self.flow_sheet.add_connection(*args, **kwargs)

    @wraps(FlowSheet.set_output_state)
    def set_output_state(self, *args: Any, **kwargs: Any) -> None:
        """Wrap FlowSheet.set_output_state to set the output state of a unit."""
        self.flow_sheet.set_output_state(*args, **kwargs)

    @property
    def zones(self) -> list[ZoneBaseClass]:
        """
        Get all zones in the carousel system.

        Returns
        -------
        list
            A list of all zones in the carousel system.
        """
        return [
            unit for unit in self.flow_sheet.units if isinstance(unit, ZoneBaseClass)
        ]

    @property
    def zone_names(self) -> list[str]:
        """list: Zone names."""
        return [zone.name for zone in self.zones]

    @property
    def zones_dict(self) -> dict[str, ZoneBaseClass]:
        """dict: Zone names and objects."""
        return {zone.name: zone for zone in self.zones}

    @property
    def n_zones(self) -> int:
        """int: Number of zones in the Carousel System."""
        return len(self.zones)

    @property
    def n_columns(self) -> int:
        """int: Number of columns in the Carousel System."""
        return sum([zone.n_columns for zone in self.zones])

    def build_flow_sheet(self) -> FlowSheet:
        """
        Assemble the flow sheet.

        Returns
        -------
        FlowSheet
            The assembled flow sheet.
        """
        if self.column is None:
            raise CADETProcessError("No column associated with Carousel.")

        flow_sheet = FlowSheet(self.component_system, self.name)

        self._add_units(flow_sheet)
        self._add_inter_zone_connections(flow_sheet)
        self._add_intra_zone_connections(flow_sheet)
        self._set_output_states(flow_sheet)

        return flow_sheet

    def _add_units(self, flow_sheet: FlowSheet) -> None:
        """Add units to flow_sheet."""
        col_index = 0
        for unit in self.flow_sheet.units:
            if not isinstance(unit, ZoneBaseClass):
                is_feed_inlet = unit in self.flow_sheet.feed_inlets
                is_eluent_inlet = unit in self.flow_sheet.eluent_inlets
                is_output_outlet = unit in self.flow_sheet.product_outlets
                flow_sheet.add_unit(
                    unit,
                    feed_inlet=is_feed_inlet,
                    eluent_inlet=is_eluent_inlet,
                    product_outlet=is_output_outlet,
                )
            else:
                flow_sheet.add_unit(unit.inlet_unit)
                flow_sheet.add_unit(unit.outlet_unit)
                for i_col in range(unit.n_columns):
                    col = deepcopy(self.column)
                    col.component_system = self.component_system
                    col.name = f"column_{col_index}"
                    if unit.initial_state is not None:
                        col.initial_state = unit.initial_state[i_col]
                    flow_sheet.add_unit(col)
                    col_index += 1

    def _add_inter_zone_connections(self, flow_sheet: FlowSheet) -> NoReturn:
        """Add connections between zones."""
        for unit, connections in self.flow_sheet.connections.items():
            if isinstance(unit, ZoneBaseClass):
                origin = unit.outlet_unit
            else:
                origin = unit
            if connections.destinations:
                for destination in connections.destinations[None]:
                    if isinstance(destination, ZoneBaseClass):
                        destination = destination.inlet_unit

                    flow_sheet.add_connection(origin, destination)

    def _add_intra_zone_connections(self, flow_sheet: FlowSheet) -> NoReturn:
        """Add connections within zones."""
        for zone in self.zones:
            for col_index in range(self.n_columns):
                col = flow_sheet[f"column_{col_index}"]
                flow_sheet.add_connection(zone.inlet_unit, col)
                col = flow_sheet[f"column_{col_index}"]
                flow_sheet.add_connection(col, zone.outlet_unit)

        for col_index in range(self.n_columns):
            col_orig = flow_sheet[f"column_{col_index}"]
            if col_index < self.n_columns - 1:
                col_dest = flow_sheet[f"column_{col_index + 1}"]
            else:
                col_dest = flow_sheet[f"column_{0}"]
            flow_sheet.add_connection(col_orig, col_dest)

    def _set_output_states(self, flow_sheet: FlowSheet) -> NoReturn:
        for unit in self.flow_sheet.output_states:
            output_state = self.flow_sheet.output_states[unit]

            if output_state == {}:
                continue

            if isinstance(unit, ZoneBaseClass):
                flow_sheet.set_output_state(unit.outlet_unit, output_state)
            else:
                flow_sheet.set_output_state(unit, output_state)

    def build_process(self) -> Process:
        """
        Assemble the process object.

        Returns
        -------
        Process
            The assembled process object.
        """
        flow_sheet = self.build_flow_sheet()
        process = Process(flow_sheet, self.name)

        self._add_events(process)

        return process

    @property
    def cycle_time(self) -> float:
        """float: cycle time of the process."""
        return self.n_columns * self.switch_time

    def _add_events(self, process: Process) -> None:
        """Add events to process."""
        process.cycle_time = self.n_columns * self.switch_time
        process.add_duration("switch_time", self.switch_time)

        for carousel_state in range(self.n_columns):
            position_counter = 0
            for i_zone, zone in enumerate(self.zones):
                col_indices = np.arange(zone.n_columns)
                col_indices += position_counter
                col_indices = self.column_indices_at_state(col_indices, carousel_state)

                if isinstance(zone, SerialZone):
                    evt = process.add_event(
                        f"{zone.name}_{carousel_state}",
                        f"flow_sheet.output_states.{zone.inlet_unit}",
                        col_indices[0],
                    )
                    process.add_event_dependency(
                        evt.name, "switch_time", [carousel_state]
                    )
                    for i, col in enumerate(col_indices):
                        if i < (zone.n_columns - 1):
                            evt = process.add_event(
                                f"column_{col}_{carousel_state}",
                                f"flow_sheet.output_states.column_{col}",
                                self.n_zones,
                            )
                        else:
                            evt = process.add_event(
                                f"column_{col}_{carousel_state}",
                                f"flow_sheet.output_states.column_{col}",
                                i_zone,
                            )
                        process.add_event_dependency(
                            evt.name, "switch_time", [carousel_state]
                        )
                elif isinstance(zone, ParallelZone):
                    output_state = self.n_columns * [0]
                    for col in col_indices:
                        output_state[col] = 1 / zone.n_columns

                    evt = process.add_event(
                        f"{zone.name}_{carousel_state}",
                        f"flow_sheet.output_states.{zone.inlet_unit}",
                        output_state,
                    )
                    process.add_event_dependency(
                        evt.name, "switch_time", [carousel_state]
                    )

                    for col in col_indices:
                        evt = process.add_event(
                            f"column_{col}_{carousel_state}",
                            f"flow_sheet.output_states.column_{col}",
                            i_zone,
                        )
                        process.add_event_dependency(
                            evt.name, "switch_time", [carousel_state]
                        )

                for i, col in enumerate(col_indices):
                    evt = process.add_event(
                        f"column_{col}_{carousel_state}_velocity",
                        f"flow_sheet.column_{col}.flow_direction",
                        zone.flow_direction,
                    )
                    process.add_event_dependency(
                        evt.name, "switch_time", [carousel_state]
                    )

                position_counter += zone.n_columns

    def carousel_state(self, t: float) -> int:
        """
        Return carousel state at given time.

        Parameters
        ----------
        t: float
            Time

        Returns
        -------
        int:
             Carousel state at given time.
        """
        return int(np.floor((t % self.cycle_time) / self.switch_time))

    def column_indices_at_state(
        self,
        carousel_positions: np.typing.NDArray[int],
        carousel_state: int,
    ) -> np.ndarray[int]:
        """
        Determine index of column unit at given carousel position and state.

        Parameters
        ----------
        carousel_positions: np.typing.NDArray[np.int_]
            Carousel position indices (e.g. wash position, elute position).
        carousel_state : int
            Curent state of the carousel system.

        Returns
        -------
        np.ndarray[int]
            Indices of column units at given carousel positions and state.
        """
        carousel_positions = np.array(carousel_positions, dtype=int)

        return (carousel_positions + carousel_state) % self.n_columns

    def column_indices_at_time(
        self,
        t: float,
        carousel_positions: np.typing.NDArray[int],
    ) -> int:
        """
        Determine index of column unit at given carousel position and time.

        Parameters
        ----------
        t: float
            Time
        carousel_position : int
            Carousel position.

        Returns
        -------
        np.ndarray[int]
            Indices of column units at given carousel positions and time.
        """
        carousel_positions = np.array(carousel_positions, dtype=int)

        carousel_state = self.carousel_state(t)
        column_indices = self.column_indices_at_state(
            carousel_positions, carousel_state
        )

        return column_indices


class SMBBuilder(CarouselBuilder):
    """
    Configurator for 4 Zone SMB systems.

    Attributes
    ----------
    binding_model_type : Type[BindingBaseClass]
        Specifies the type of binding model used in the SMB process.
    """

    binding_model_type = BindingBaseClass

    def __init__(
        self,
        feed: Inlet,
        eluent: Inlet,
        column: TubularReactorBase,
        name: str = "SMB",
        valve_parameters: dict = None,
    ) -> NoReturn:
        """
        Initialize an SMBBuilder instance.

        Parameters
        ----------
        feed : Inlet
            The feed inlet to the SMB system.
        eluent : Inlet
            The eluent inlet to the SMB system.
        column : TubularReactorBase
            The column equipment in the SMB system.
        name : str, optional
            The name of the SMB setup, by default 'SMB'.

        Raises
        ------
        CADETProcessError
            If component systems of feed, eluent, and column do not match.
        TypeError
            If feed, eluent, or column are not of the expected type.
        """
        component_system = feed.component_system
        if not (component_system is eluent.component_system is column.component_system):
            raise CADETProcessError("ComponentSystems do not match.")
        if not isinstance(feed, Inlet):
            raise TypeError(f"Expected inlet object. Got {type(feed)}.")
        if not isinstance(eluent, Inlet):
            raise TypeError(f"Expected inlet object. Got {type(eluent)}.")

        if not isinstance(column, TubularReactorBase):
            raise TypeError(f"Expected Column object. Got {type(column)}.")

        self._validate_binding_model(column.binding_model)

        super().__init__(component_system, name, valve_parameters)

        raffinate = Outlet(component_system, name="raffinate")
        extract = Outlet(component_system, name="extract")

        zone_I = SerialZone(
            component_system, "zone_I", 1, valve_parameters=self.valve_parameters
        )
        zone_II = SerialZone(
            component_system, "zone_II", 1, valve_parameters=self.valve_parameters
        )
        zone_III = SerialZone(
            component_system, "zone_III", 1, valve_parameters=self.valve_parameters
        )
        zone_IV = SerialZone(
            component_system, "zone_IV", 1, valve_parameters=self.valve_parameters
        )

        # Carousel Builder
        self.column = column
        self.add_unit(feed, feed_inlet=True)
        self.add_unit(eluent, eluent_inlet=True)

        self.add_unit(raffinate, product_outlet=True)
        self.add_unit(extract, product_outlet=True)

        self.add_unit(zone_I)
        self.add_unit(zone_II)
        self.add_unit(zone_III)
        self.add_unit(zone_IV)

        self.add_connection(eluent, zone_I)

        self.add_connection(zone_I, extract)
        self.add_connection(zone_I, zone_II)

        self.add_connection(zone_II, zone_III)

        self.add_connection(feed, zone_III)

        self.add_connection(zone_III, raffinate)
        self.add_connection(zone_III, zone_IV)

        self.add_connection(zone_IV, zone_I)

    def _validate_binding_model(self, binding_model: BindingBaseClass) -> None:
        """
        Validate that the provided binding model matches the required type.

        Parameters
        ----------
        binding_model : BindingBaseClass
            The binding model to be validated.

        Raises
        ------
        TypeError
            If the binding model does not match the expected type.
        """
        if not isinstance(binding_model, self.binding_model_type):
            raise TypeError(
                f"Invalid binding model. Expected {self.binding_model_type}."
            )

    def _get_zone_flow_rates(self, m: list, switch_time: float) -> list[float]:
        m1, m2, m3, m4 = m

        Vc = self.column.volume
        et = self.column.total_porosity

        Q_I = Vc * (m1 * (1 - et) + et) / switch_time     # Flow rate Zone I
        Q_II = Vc * (m2 * (1 - et) + et) / switch_time    # Flow rate Zone II
        Q_III = Vc * (m3 * (1 - et) + et) / switch_time   # Flow rate Zone III
        Q_IV = Vc * (m4 * (1 - et) + et) / switch_time    # Flow rate Zone IV

        return [Q_I, Q_II, Q_III, Q_IV]

    def _get_unit_flow_rates(self, Q_zones: list[float]) -> list[float]:
        Q_I, Q_II, Q_III, Q_IV = Q_zones

        Q_feed = Q_III - Q_II
        Q_eluent = Q_I - Q_IV
        Q_raffinate = Q_III - Q_IV
        Q_extract = Q_I - Q_II

        return [Q_feed, Q_eluent, Q_raffinate, Q_extract]

    def _get_split_ratios(
        self, Q_zones: list[float], Q_units: list[float]
    ) -> tuple[float, float]:
        Q_I, Q_II, Q_III, Q_IV = Q_zones
        Q_feed, Q_eluent, Q_raffinate, Q_extract = Q_units

        w_r = Q_raffinate / Q_III
        w_e = Q_extract / Q_I

        return w_r, w_e

    def get_design_parameters(
        self,
        binding_model: BindingBaseClass,
        c_feed: np.ndarray,
    ) -> Any:
        """
        Retrieve design parameters based on the binding model.

        Parameters
        ----------
        binding_model : BindingBaseClass
            The binding model used to calculate design parameters.
        c_feed : np.ndarray
            The feed concentration.

        Returns
        -------
        Any
            The design parameters computed from the binding model.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def calculate_m_opt(self, *design_parameters: Any) -> list:
        """
        Calculate the optimal zone flow rates based on provided design parameters.

        Parameters
        ----------
        design_parameters : Any
            A variable number of parameters defining the design.

        Returns
        -------
        list
            The optimal zone flow rates based on the design parameters.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def apply_safety_factor(
        self,
        m_opt: list,
        *design_parameters: Any,
        gamma: float | list[float],
    ) -> Any:
        """
        Apply a safety factor to the optimal zone flow rates.

        Parameters
        ----------
        m_opt : Any
            The optimal zone flow rates calculated without safety considerations.
        design_parameters : Any
            A variable number of parameters defining the design.
        gamma : float | list[float]
            The safety factor(s) to apply to the zone flow rates. If float is provided,
            the same value is applied to all zone flow rates.

        Returns
        -------
        list
            The adjusted zone flow rates after applying the safety factor.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def triangle_design(
        self,
        binding_model: Optional[BindingBaseClass] = None,
        c_feed: Optional[np.ndarray] = None,
        switch_time: Optional[float] = None,
        gamma: float | list[float] = 1,
        set_values: bool = True,
    ) -> list[float]:
        """
        Design the SMB process according to the triangle theory.

        Parameters
        ----------
        binding_model : BindingBaseClass, optional
            The binding model to be used.
            If None, uses the binding model from the column attribute.
        c_feed : np.ndarray, optional
            Feed concentration matrix.
            If None, uses current feed concentrations.
        switch_time : float, optional
            Switching time for the SMB zones.
            If None, uses the existing switch_time attribute.
        gamma : float | list[float]
            The safety factor(s) to apply to the zone flow rates. If float is provided,
            the same value is applied to all zone flow rates.
            The default is 1.
        set_values : bool, default True
            If True, sets the calculated values to the process model.

        Returns
        -------
        list
            A list containing the feed flow rate, eluent flow rate, raffinate and
            extract split ratios.

        Raises
        ------
        Warning
            If binding_model is provided and set_values is True, values cannot be set.
        """
        if binding_model is None:
            binding_model = self.column.binding_model
        elif set_values is True:
            warnings.warn("Cannot set values if binding_model is given.")
            set_values = False

        self._validate_binding_model(binding_model)

        if c_feed is None:
            c_feed = self.flow_sheet.feed.c[:, 0]
        elif set_values is True:
            self.flow_sheet.feed.c = c_feed

        design_parameters = self.get_design_parameters(binding_model, c_feed)
        m_opt = self.calculate_m_opt(*design_parameters)
        m = self.apply_safety_factor(m_opt, *design_parameters, gamma=gamma)

        if switch_time is None:
            switch_time = self.switch_time
        elif set_values is True:
            self.switch_time = switch_time

        Q_zones = self._get_zone_flow_rates(m, switch_time)
        Q_units = self._get_unit_flow_rates(Q_zones)
        Q_feed, Q_eluent, Q_raffinate, Q_extract = Q_units
        w_r, w_e = self._get_split_ratios(Q_zones, Q_units)

        if set_values:
            self.flow_sheet.feed.flow_rate = Q_feed
            self.flow_sheet.eluent.flow_rate = Q_eluent
            self.set_output_state("zone_I", [w_e, 1 - w_e])
            self.set_output_state("zone_III", [w_r, 1 - w_r])

        return [Q_feed, Q_eluent, w_r, w_e]

    def plot_triangle(
        self,
        binding_model: Optional[BindingBaseClass] = None,
        c_feed: Optional[np.ndarray] = None,
        gamma: float | list[float] = 1,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the triangle diagram for the SMB process with the operating point marked.

        Parameters
        ----------
        binding_model : BindingBaseClass, optional
            The binding model to use for the plot.
            If None, uses the column's binding model.
        c_feed : np.ndarray, optional
            The feed concentration array.
            If None, uses the current feed concentration from the flow sheet.
        gamma : float | list[float]
            The safety factor(s) to apply to the zone flow rates. If float is provided,
            the same value is applied to all zone flow rates.
        fig : plt.Figure, optional
            The matplotlib figure object. If None, a new figure will be created.
        ax : plt.Axes, optional
            The matplotlib axes object.
            If None, a new axes will be created in the new or specified figure.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The figure and axes objects containing the triangle diagram.

        Notes
        -----
        This method uses internal methods to fetch design parameters and calculate
        optimal zone flow rates, which should be defined in a subclass.
        """
        if binding_model is None:
            binding_model = self.column.binding_model
        self._validate_binding_model(binding_model)

        if c_feed is None:
            c_feed = self.flow_sheet.feed.c[:, 0]

        # Operating Point
        design_parameters = self.get_design_parameters(binding_model, c_feed)
        m_opt = self.calculate_m_opt(*design_parameters)
        m1, m2, m3, m4 = self.apply_safety_factor(
            m_opt, *design_parameters, gamma=gamma
        )

        # Setup figure
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Plot Triangle
        self._plot_triangle(ax, *design_parameters)

        ax.set_xlabel("$m_{II}$")
        ax.set_ylabel("$m_{III}$")

        # Operating point
        ax.scatter(m2, m3, c="k", marker="x", zorder=3)
        ax.annotate(
            "operating point",
            xy=(m2, m3),
            xytext=(1.1 * m2, 0.9 * m3),
            arrowprops=dict(facecolor="black", shrink=0.01),
        )

        return fig, ax

    def _plot_triangle(self, ax: Axes, *design_parameters: Any) -> None:
        """
        Plot a theoretical triangle diagram for SMB processes.

        This method needs to be implemented by subclasses to define the specific way the
        triangle is visualized based on provided design parameters.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axes on which the triangle will be plotted.
        design_parameters : Any
            Variable-length argument list of design parameters that define the triangle.

        Raises
        ------
        NotImplementedError
            This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError()


class LinearSMBBuilder(SMBBuilder):
    """
    Configure a 4-zone SMB system with linear isotherms.

    Note, this is currently only supported for 2-Component systems.

    Attributes
    ----------
    binding_model_type : type
        Specifies that this builder uses the Linear binding model.
    """

    binding_model_type = Linear

    def _validate_binding_model(self, binding_model: BindingBaseClass) -> None:
        """
        Validate the binding model for compatibility with Langmuir SMB systems.

        Parameters
        ----------
        binding_model : BindingBaseClass
            The binding model to be validated.

        Raises
        ------
        CADETProcessError
            If the binding model does not contain exactly two components.

        Warns
        -----
        RuntimeWarning
            If the binding model is kinetic, which conflicts with the assumption of
            instant equilibrium.
        """
        super()._validate_binding_model(binding_model)

        if binding_model.n_comp != 2:
            raise CADETProcessError("This only works for 2-Component Systems.")

        if binding_model.is_kinetic:
            warnings.warn(
                "Isotherm uses kinetic binding, "
                "however, triangle theory assumes instant equilibrium."
            )

    def get_design_parameters(
        self,
        binding_model: BindingBaseClass,
        c_feed: np.ndarray,
    ) -> tuple[float, float]:
        """
        Calculate Henry's constants (H) based on adsorption and desorption rates.

        Parameters
        ----------
        binding_model : BindingBaseClass
            The binding model containing adsorption and desorption rate data.
        c_feed : np.ndarray
            The concentration feed.

        Returns
        -------
        Tuple[float, float]
            Henry's constants for the strongly and weakly adsorbing components.
        """
        k_ads = np.array(binding_model.adsorption_rate)
        k_des = np.array(binding_model.desorption_rate)
        H = k_ads / k_des

        if H[1] < H[0]:
            HA, HB = H
        else:
            HB, HA = H

        return HA, HB

    def calculate_m_opt(
        self,
        HA: float,
        HB: float,
    ) -> list[float]:
        """
        Calculate the optimal flow rates for SMB zones based on Henry's constants.

        Parameters
        ----------
        HA : float
            Henry's constant for strongly binding component.
        HB : float
            Henry's constant for weakly binding component.

        Returns
        -------
        list[float]
            List of optimal flow rates for zones 1 to 4, respectively.
        """
        m1 = HA
        m2 = HB
        m3 = HA
        m4 = HB

        return [m1, m2, m3, m4]

    def apply_safety_factor(
        self,
        m_opt: list[float],
        *design_parameters: Any,
        gamma: float | list[float] = 1,
    ) -> list[float]:
        """
        Adjust the optimal flow rates by applying safety factors to each zone.

        Parameters
        ----------
        m_opt : list[float]
            A list containing the optimal flow rates for zones 1 to 4.
        design_parameters : Union[float, List[float]]
            Additional design parameters that might influence safety factor application.
        gamma : float | list[float]
            Safety factors to apply. Can be a single float applied to all zones, or a
            list of floats for each zone. The default is 1.

        Returns
        -------
        list[float]
            Adjusted flow rates for zones 1 to 4 after applying the safety factors.

        Raises
        ------
        ValueError
            If the product of gamma for zones 2 and 3 is not smaller than the ratio of
            m3_opt to m2_opt, which is a critical operational constraint.
        """
        m1_opt, m2_opt, m3_opt, m4_opt = m_opt

        if np.isscalar(gamma):
            gamma = 4 * [gamma]

        gamma_1, gamma_2, gamma_3, gamma_4 = gamma

        if gamma_2 * gamma_3 >= m3_opt / m2_opt:
            raise ValueError("gamma_2 * gamma_3 must be smaller than HA / HB ")

        m1 = gamma_1 * m1_opt
        m2 = gamma_2 * m2_opt
        m3 = m3_opt / gamma_3
        m4 = m4_opt / gamma_4

        return [m1, m2, m3, m4]

    def _plot_triangle(
        self,
        ax: Axes,
        HA: float,
        HB: float,
    ) -> NoReturn:
        """
        Plot SMB triangle for linear isotherm.

        Notable points:
        - a: [HA, HA]
        - b: [HB, HB]
        - w_opt: [HB, HA]

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        HA : float
            Henry coefficient of strongly binding component.
        HB : float
            Henry coefficient of strongly binding component.
        """
        # Bounds
        lb = HB - 0.3 * (HA - HB)
        ub = HA + 0.3 * (HA - HB)

        ax.set_xlim(lb, ub)
        ax.set_ylim(lb, ub)

        # Diagonal
        ax.plot((lb, ub), (lb, ub), "k")

        # Henry coefficients
        for h in [HB, HA]:
            ax.hlines(h, 0, h, "k", "dashed")
            ax.vlines(h, h, ub, "k", "dashed")

        # Triangle
        ax.hlines(HA, HB, HA, "k")
        ax.vlines(HB, HB, HA, "k")

        # Label regions
        ax.text(
            (HB + (HA - HB) / 2), (0.95 * ub), "Pure extract", ha="center", va="center"
        )
        ax.text(
            (1.05 * lb),
            (HB + (HA - HB) / 2),
            "Pure raffinate",
            ha="center",
            va="center",
            rotation="vertical",
        )


class LangmuirSMBBuilder(SMBBuilder):
    """
    Configure a 4-zone SMB system with Langmuir isotherms.

    Note, this is currently only supported for 2-Component systems.

    Attributes
    ----------
    binding_model_type : type
        Specifies that this builder uses the Linear binding model.
    """

    binding_model_type = Langmuir

    def _validate_binding_model(self, binding_model: BindingBaseClass) -> NoReturn:
        """
        Validate the binding model for compatibility with Langmuir SMB systems.

        Parameters
        ----------
        binding_model : BindingBaseClass
            The binding model to be validated.

        Raises
        ------
        CADETProcessError
            If the binding model does not contain exactly two components.

        Warns
        -----
        RuntimeWarning
            If the binding model is kinetic, which conflicts with the assumption of
            instant equilibrium.
        """
        super()._validate_binding_model(binding_model)

        if binding_model.n_comp != 2:
            raise CADETProcessError("This only works for 2-Component Systems.")

        if binding_model.is_kinetic:
            warnings.warn(
                "Isotherm uses kinetic binding, "
                "however, triangle theory assumes instant equilibrium."
            )

    def get_design_parameters(
        self,
        binding_model: BindingBaseClass,
        c_feed: np.ndarray,
    ) -> tuple[float, float]:
        """
        Calculate the optimal flow rates for SMB zones based on the provided parameters.

        Parameters
        ----------
        binding_model : BindingBaseClass
            The binding model parameters.
        c_feed : np.ndarray
            The concentration feed matrix.

        Returns
        -------
        tuple
            A tuple containing Henry's constants HA, HB, binding coefficients bA, bB,
            feed concentrations cFA, cFB, and quadratic nulls wG, wF.
        """
        k_ads = np.array(binding_model.adsorption_rate)
        k_des = np.array(binding_model.desorption_rate)
        k_eq = k_ads / k_des
        q_sat = np.array(binding_model.capacity)
        H = [k_eq * q_s for k_eq, q_s in zip(k_eq, q_sat)]

        if H[1] < H[0]:
            HA, HB = H
            bA, bB = k_eq
            cFA, cFB = c_feed
        else:
            HB, HA = H
            bB, bA = k_eq
            cFB, cFA = c_feed

        a = -(HA * (1 + bB * cFB) + HB * (1 + bA * cFA)) / (1 + bB * cFB + bA * cFA)
        b = HA * HB / (1 + bB * cFB + bA * cFA)
        wG = -a / 2 + np.sqrt((-a / 2) ** 2 - b)
        wF = -a / 2 - np.sqrt((-a / 2) ** 2 - b)

        return HA, HB, bA, bB, cFA, cFB, wG, wF

    def calculate_m_opt(
        self,
        HA: float,
        HB: float,
        bA: float,
        bB: float,
        cFA: float,
        cFB: float,
        wG: float,
        wF: float,
    ) -> list[float]:
        """
        Calculate optimal zone flow rates based on Langmuir isotherm parameters.

        Parameters
        ----------
        HA : float
            Henry's constant for strongly binding component.
        HB : float
            Henry's constant for weakly binding component.
        bA : float
            Binding coefficient for strongly binding component.
        bB : float
            Binding coefficient for weakly binding component.
        cFA : float
            Feed concentration for strongly binding component.
        cFB : float
            Feed concentration for weakly binding component.
        wG : float
            First quadratic null.
        wF : float
            Second quadratic null.

        Returns
        -------
        list[float]
            List of optimal flow rates for zones 1 to 4.
        """
        m1 = HA
        m2 = HB / HA * wG
        m3 = wG * (wF * (HA - HB) + HB * (HB - wF)) / (HB * (HA - wF))
        m4 = (
            1
            / 2
            * (
                HB
                + m3
                + bB * cFB * (m3 - m2)
                - np.sqrt((HB + m3 + bB * cFB * (m3 - m2)) ** 2 - 4 * HB * m3)
            )
        )

        return [m1, m2, m3, m4]

    def apply_safety_factor(
        self,
        m_opt: list[float],
        HA: float,
        HB: float,
        bA: float,
        bB: float,
        cFA: float,
        cFB: float,
        wG: float,
        wF: float,
        gamma: float | list[float] = 1,
    ) -> list[float]:
        """
        Apply a safety factor to the optimal zone flow rates.

        Parameters
        ----------
        m_opt : list[float]
            List of optimal flow rates for zones 1 to 4.
        HA : float
            Henry's constant for strongly binding component.
        HB : float
            Henry's constant for weakly binding component.
        bA : float
            Binding coefficient for strongly binding component.
        bB : float
            Binding coefficient for weakly binding component.
        cFA : float
            Feed concentration for strongly binding component.
        cFB : float
            Feed concentration for weakly binding component.
        wG : float
            First quadratic null.
        wF : float
            Second quadratic null.
        gamma : float | list[float]
            The safety factor(s) to apply to the zone flow rates. If float is provided,
            the same value is applied to all zone flow rates.

        Returns
        -------
        list[float]
            Adjusted flow rates for zones 1 to 4 after applying the safety factors.
        """
        m1_opt, m2_opt, m3_opt, m4_opt = m_opt

        if np.isscalar(gamma):
            W_opt = np.array([m2_opt, m3_opt])
            B = np.array([HB, HB])
            R = [
                wG**2 / HA,
                wG
                * (wF * (HA - wG) * (HA - HB) + HB * wG * (HA - wF))
                / (HA * HB * (HA - wF)),
            ]

            # Calculating vectors WB and WA
            WB = B - W_opt
            WR = R - W_opt

            # Normalizing vectors
            norm_WB = WB / np.linalg.norm(WB)
            norm_WR = WR / np.linalg.norm(WR)

            # Calculating the bisector WB / WR
            bisector = norm_WB + norm_WR
            norm_dir_vec_bisector = bisector / np.linalg.norm(bisector)

            # Calculating the new point W' using the safety factor gamma
            W = W_opt + (gamma - 1) * norm_dir_vec_bisector

            m1 = gamma * m1_opt
            m2 = W[0]
            m3 = W[1]
            m4 = m4_opt / gamma

        else:
            gamma_1, gamma_2, gamma_3, gamma_4 = gamma

            m1 = gamma_1 * m1_opt
            m2 = gamma_2 * m2_opt
            m3 = m3_opt / gamma_3
            m4 = m4_opt / gamma_4

        return [m1, m2, m3, m4]

    def _plot_triangle(
        self,
        ax: Axes,
        HA: float,
        HB: float,
        bA: float,
        bB: float,
        cFA: float,
        cFB: float,
        wG: float,
        wF: float,
    ) -> None:
        """
        Plot SMB triangle for Langmuir isotherm.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        HA : float
            Henry's constant for strongly binding component.
        HB : float
            Henry's constant for weakly binding component.
        bA : float
            Binding coefficient for strongly binding component.
        bB : float
            Binding coefficient for weakly binding component.
        cFA : float
            Feed concentration for strongly binding component.
        cFB : float
            Feed concentration for weakly binding component.
        wG : float
            First quadratic null.
        wF : float
            Second quadratic null.
        """
        m1, m2, m3, m4 = self.calculate_m_opt(HA, HB, bA, bB, cFA, cFB, wG, wF)
        W = [m2, m3]

        R = [
            wG**2 / HA,
            wG
            * (wF * (HA - wG) * (HA - HB) + HB * wG * (HA - wF))
            / (HA * HB * (HA - wF)),
        ]

        # Bounds
        lb = W[0] - 0.3 * (HA - W[0])
        ub = HA + 0.3 * (HA - W[0])

        ax.set_xlim(lb, ub)
        ax.set_ylim(lb, ub)

        # Diagonal
        ax.plot((lb, ub), (lb, ub), "k")

        # Plot [W -> R]
        m2WR = np.linspace(W[0], R[0], 50)
        m3WR = (
            1 / (bA * cFA * wG) * (wG * (HA - wG) - (HA - wG * (1 + bA * cFA)) * m2WR)
        )
        ax.plot(m2WR, m3WR, "k-")

        # plot [W -> HB]
        m2WHB = np.linspace(W[0], HB, 10)
        m3WHB = (
            1 / (bA * cFA * HB) * (HB * (HA - HB) - (HA - HB * (1 + bA * cFA)) * m2WHB)
        )
        ax.plot(m2WHB, m3WHB, "k-")

        # plot [R -> HA]
        m2RHA = np.linspace(R[0], HA, 10)
        m3RHA = m2RHA + (np.sqrt(HA) - np.sqrt(m2RHA)) ** 2 / (bA * cFA)
        ax.plot(m2RHA, m3RHA, "k-")

        # TODO: Equations that plot regions of pure extract / raffinate not clear yet.


class CarouselSolutionBulk(SolutionBase):
    """
    Solution at unit inlet or outlet.

    N_COLUMNS * NCOL * NRAD
    """

    _coordinates = ["axial_coordinates", "radial_coordinates"]

    def __init__(
        self,
        builder: CarouselBuilder,
        simulation_results: SimulationResults,
    ) -> None:
        if not builder.column.solution_recorder.write_solution_bulk:
            raise CADETProcessError(
                "Cannot instantiate CarouselSolutionBulk if solution is not stored. "
                "Please set "
                "`builder.column.solution_recorder.write_solution_bulk = True`."
            )
        self.builder = builder
        self.simulation_results = simulation_results

    @property
    def component_system(self) -> ComponentSystem:
        return self.builder.component_system

    @property
    def solution(self) -> Dict:
        return self.simulation_results.solution

    @property
    def axial_coordinates(self) -> npt.ArrayLike:
        return self.simulation_results.solution.column_0.bulk.axial_coordinates

    @property
    def radial_coordinates(self) -> npt.ArrayLike:
        radial_coordinates = \
            self.simulation_results.solution.column_0.bulk.radial_coordinates
        if radial_coordinates is not None and len(radial_coordinates) == 1:
            radial_coordinates = None

        return radial_coordinates

    @property
    def time(self) -> float:
        return self.simulation_results.solution.column_0.bulk.time

    def plot_at_time(
        self,
        t: float,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        axs: Optional[Axes] = None,
    ) -> tuple[Figure, Axes]:
        """Plot bulk solution over space at given time.

        Parameters
        ----------
        t : float
            time for plotting
        y_min : float, optional
            Set minimum for plotting purposes.
            If None, y_min is set to minimum of the data.
        y_max : float, optional
            Set maximum for plotting purposes.
            If None, y_max is set to minimum of the data
        ax : Axes
            Axes to plot on.

        See Also
        --------
        CADETProcess.plotting
        """
        n_cols = self.builder.n_columns
        if axs is None:
            fig, axs = plt.subplots(
                ncols=n_cols,
                figsize=(n_cols * 4, 6),
                gridspec_kw=dict(wspace=0.0, hspace=0.0),
                sharey="row",
            )
        else:
            fig = axs[0].figure

        t_i = np.where(t <= self.time)[0][0]

        x = self.axial_coordinates

        y_min_data = 0
        y_max_data = 0
        zone_counter = 0
        column_counter = 0

        for position, ax in enumerate(axs):
            col_index = self.builder.column_indices_at_time(t, position)

            y = self.solution[f"column_{col_index}"].bulk.solution[t_i, :]

            y_min_data = min(y_min_data, min(0, np.min(y)))
            y_max_data = max(y_max_data, 1.1 * np.max(y))

            ax.plot(x, y)

            zone = self.builder.zones[zone_counter]

            if zone.n_columns > 1:
                ax.set_title(f"{zone.name}, position {column_counter}")
            else:
                ax.set_title(f"{zone.name}")

            if column_counter < (zone.n_columns - 1):
                column_counter += 1
            else:
                zone_counter += 1
                column_counter = 0

        plotting.add_text(ax, f"time = {t:.2f} s")

        if y_min is None:
            y_min = y_min_data
        if y_min is None:
            y_min = y_max_data

        for position, ax in enumerate(axs):
            ax.set_ylim((y_min, y_max))

        return fig, axs
