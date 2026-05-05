import copy
from typing import Optional

from CADETProcess.processModel import (
    ChromatographicColumnBase,
    FlowSheet,
    Inlet,
    Outlet,
    Process,
)


def _copy_column(
    column: ChromatographicColumnBase, own_binding_model: bool
) -> ChromatographicColumnBase:
    col = copy.copy(column)
    if own_binding_model:
        col._binding_model = copy.deepcopy(column.binding_model)
        col._binding_model.component_system = col.component_system
    return col


class SerialColumns(Process):
    """
    Serial columns process.

    The flowsheet is configured with the following unit operations:
    - feed: Inlet
    - eluent_1: Inlet
    - eluent_2: Inlet
    - column_1: ChromatographicColumnBase
    - column_2: ChromatographicColumnBase
    - outlet_1: Outlet for column 1
    - outlet_2: Outlet for column 2

    --- Injection ---
    - `feed_on`: Sets `feed.flow_rate` to `flow_rate`.
    - `eluent_1_off`: Sets `eluent_1.flow_rate` to 0.0; triggered by `feed_on`.
    - `serial_on`: Connects column 1 to column 2.
    - `eluent_2_off`: Sets `eluent_2.flow_rate` to 0.0; triggered by `serial_on`.

    --- Elution ---
    - `feed_off`: Sets `feed.flow_rate` to 0.0; triggered by `feed_on` and `feed_duration`.
    - `eluent_1_on`: Sets `eluent_1.flow_rate` to `flow_rate`; triggered by `feed_off`.

    --- Serial Connection ---
    - `serial_off`: Disconnects column 1 from column 2 (sets `output_states.column_1` to 0).
    - `eluent_2_on`: Sets `eluent_2.flow_rate` to `flow_rate`; triggered by `serial_off`.
    """

    def __init__(
        self,
        column: ChromatographicColumnBase,
        split_ratio: float,
        c_feed: list[float],
        flow_rate: float,
        feed_duration: float,
        t_serial_on: float,
        t_serial_off: float,
        cycle_time: float,
        c_eluent: Optional[list[float] | float] = 0.0,
        own_binding_model: bool = False,
    ) -> None:
        """
        Initialize batch elution process.

        Parameters
        ----------
        column : ChromatographicColumnBase
            Template chromatographic column object.
            Both columns are derived from this template.
        split_ratio : float
            Split ratio for column lengths (must be between 0 and 1).
            column_1.length = split_ratio * column.length
            column_2.length = (1 - split_ratio) * column.length
        c_feed : list[float]
            Feed concentration.
        flow_rate : float
            Flow rate.
        feed_duration : float
            Feed duration.
        t_serial_on: float
            Time at which to connect columns serially.
        t_serial_off: float
            Time at which to cut serial connection.
        cycle_time : float
            Cycle time.
        c_eluent : list[float] | float, optional
            Eluent concentration. The default is 0.0.
        own_binding_model : bool, optional
            If True, each column gets its own deep-copied binding model.
            If False (default), all columns share the same binding model instance.
        """
        if not isinstance(column, ChromatographicColumnBase):
            raise TypeError("Expected ChromatographicColumnBase.")

        flow_sheet = self._build_flow_sheet(
            column,
            split_ratio,
            c_feed,
            c_eluent,
            own_binding_model,
        )

        super().__init__(flow_sheet, "Serial Columns")
        self.cycle_time = cycle_time

        # Durations
        self.add_duration("feed_duration", feed_duration)

        # Injection
        self.add_event("feed_on", "flow_sheet.feed.flow_rate", flow_rate, time=0.0)
        self.add_event("eluent_1_off", "flow_sheet.eluent_1.flow_rate", 0.0)
        self.add_event_dependency("eluent_1_off", ["feed_on"])

        # Serial Connection
        self.add_event(
            'serial_on',
            'flow_sheet.output_states.column_1',
            {"column_2": 1},
            time=t_serial_on,
        )
        self.add_event('eluent_2_off', 'flow_sheet.eluent_2.flow_rate', 0.0)
        self.add_event_dependency('eluent_2_off', ['serial_on'], [1])

        # Elution
        self.add_event("feed_off", "flow_sheet.feed.flow_rate", 0.0)
        self.add_event_dependency("feed_off", ["feed_on", "feed_duration"], [1, 1])
        self.add_event("eluent_1_on", "flow_sheet.eluent_1.flow_rate", flow_rate)
        self.add_event_dependency("eluent_1_on", ["feed_off"])

        # Cut serial connection
        self.add_event(
            'serial_off',
            'flow_sheet.output_states.column_1',
            {"outlet_1": 1},
            time=t_serial_off,
        )
        self.add_event('eluent_2_on', 'flow_sheet.eluent_2.flow_rate', flow_rate)
        self.add_event_dependency('eluent_2_on', ['serial_off'], [1])

    def _build_flow_sheet(
        self,
        column: ChromatographicColumnBase,
        split_ratio: float,
        c_feed: list[float],
        c_eluent: list[float] | float | None = 0.0,
        own_binding_model: bool = False,
    ) -> FlowSheet:
        """Build and return the flow sheet for batch elution process."""
        component_system = column.component_system

        # Unit Operations
        feed = Inlet(component_system, name="feed")
        feed.c = c_feed

        eluent_1 = Inlet(component_system, name="eluent_1")
        eluent_1.c = c_eluent

        eluent_2 = Inlet(component_system, name="eluent_2")
        eluent_2.c = c_eluent

        column_1 = _copy_column(column, own_binding_model)
        column_1.name = "column_1"
        column_1.length = split_ratio * column.length

        column_2 = _copy_column(column, own_binding_model)
        column_2.name = "column_2"
        column_2.length = (1 - split_ratio) * column.length

        outlet_1 = Outlet(component_system, name="outlet_1")
        outlet_2 = Outlet(component_system, name="outlet_2")

        # Flow Sheet
        flow_sheet = FlowSheet(component_system)

        flow_sheet.add_unit(feed, feed_inlet=True)
        flow_sheet.add_unit(eluent_1, eluent_inlet=True)
        flow_sheet.add_unit(eluent_2, eluent_inlet=True)
        flow_sheet.add_unit(column_1)
        flow_sheet.add_unit(column_2)
        flow_sheet.add_unit(outlet_1, product_outlet=True)
        flow_sheet.add_unit(outlet_2, product_outlet=True)

        flow_sheet.add_connection(feed, column_1)
        flow_sheet.add_connection(eluent_1, column_1)
        flow_sheet.add_connection(column_1, outlet_1)
        flow_sheet.add_connection(column_1, column_2)
        flow_sheet.add_connection(eluent_2, column_2)
        flow_sheet.add_connection(column_2, outlet_2)

        return flow_sheet
