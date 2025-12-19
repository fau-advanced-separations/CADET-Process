import copy
import warnings

from CADETProcess.processModel import (
    ChromatographicColumnBase,
    Cstr,
    FlowSheet,
    Inlet,
    Outlet,
    Process,
)


class MRSSR(Process):
    """
    Mixed-recycle steady-state recycling (MRSSR) process.

    The flowsheet includes:
    - feed: Inlet
    - eluent: Inlet
    - tank: Cstr (mixing tank)
    - column: ChromatographicColumnBase
    - outlet: Outlet

    The process is configured with the following events:
    - Feed injection (duration: `feed_duration`), coupled to elution flow rate.
    - Recycling (starts at `recycle_on`, ends at `recycle_off`).
    """

    def __init__(
        self,
        column: ChromatographicColumnBase,
        c_feed: list[float],
        flow_rate: float,
        feed_duration: float,
        recycle_on: float,
        recycle_off: float,
        cycle_time: float,
        V_tank: float,
        c_eluent: list[float] | float = 0.0,
        c_tank_init: list[float] | float | None = None,
    ) -> None:
        """
        Initialize MRSSR process.

        Parameters
        ----------
        column : ChromatographicColumnBase
            Chromatographic column object.
        c_feed : list[float]
            Feed concentration.
        flow_rate : float
            Flow rate.
        feed_duration : float
            Feed injection duration.
        recycle_on : float
            Time at which to start recycling.
        recycle_off : float
            Time at which to end recycling.
        cycle_time : float
            Total cycle time.
        V_tank : float
            Volume of the mixing tank.
        c_eluent : list[float] | float, optional
            Eluent concentration. Defaults to 0.0.
        c_tank_init : list[float] | float | None, optional
            Initial tank concentration. If None, feed concentration is used.
            Defaults to None.

        """
        if not isinstance(column, ChromatographicColumnBase):
            raise TypeError("Expected ChromatographicColumnBase.")

        column = copy.deepcopy(column)
        if not column.name == "column":
            warnings.warn("Renaming column to `column` for consistency")
            column.name = "column"

        flow_sheet = self._build_flow_sheet(
            column=column,
            c_feed=c_feed,
            c_eluent=c_eluent,
            V_tank=V_tank,
            c_tank_init=c_tank_init,
        )

        super().__init__(flow_sheet, "MRSSR")

        self.cycle_time = cycle_time

        # Durations
        self.add_duration("feed_duration", feed_duration)

        # Events
        self.add_event("inject_on", "flow_sheet.tank.flow_rate", flow_rate, 0)
        self.add_event("inject_off", "flow_sheet.tank.flow_rate", 0.0)
        self.add_event("eluent_on", "flow_sheet.eluent.flow_rate", flow_rate)
        self.add_event("eluent_off", "flow_sheet.eluent.flow_rate", 0.0)
        self.add_event("feed_on", "flow_sheet.feed.flow_rate", flow_rate)
        self.add_event("feed_off", "flow_sheet.feed.flow_rate", 0.0)
        self.add_event(
            "recycle_on",
            f"flow_sheet.output_states.{column.name}",
            {"tank": 1},
            recycle_on,
        )
        self.add_event(
            "recycle_off",
            f"flow_sheet.output_states.{column.name}",
            {"outlet": 1},
            recycle_off,
        )

        # Dependencies
        self.add_event_dependency("eluent_off", ["inject_on"])
        self.add_event_dependency("eluent_on", ["inject_off"])
        self.add_event_dependency("feed_on", ["inject_off"])
        self.add_event_dependency("feed_off", ["feed_on", "feed_duration"], [1, 1])
        self.add_event_dependency(
            "inject_off",
            ["inject_on", "feed_duration", "recycle_off", "recycle_on"],
            [1, 1, 1, -1]
        )

    def _build_flow_sheet(
        self,
        column: ChromatographicColumnBase,
        c_feed: list[float],
        c_eluent: list[float] | float = 0.0,
        V_tank: float = 1e-6,
        c_tank_init: list[float] | float | None = None,
    ) -> FlowSheet:
        """Build and return the flow sheet for MRSSR process."""
        component_system = column.component_system

        # Unit Operations
        feed = Inlet(component_system, name="feed")
        feed.c = c_feed

        eluent = Inlet(component_system, name="eluent")
        eluent.c = c_eluent

        tank = Cstr(component_system, name="tank")
        tank.V = V_tank
        if c_tank_init is None:
            c_tank_init = c_feed
        tank.c = c_tank_init

        outlet = Outlet(component_system, name="outlet")

        # Flow Sheet
        flow_sheet = FlowSheet(component_system)

        flow_sheet.add_unit(feed, feed_inlet=True)
        flow_sheet.add_unit(eluent, eluent_inlet=True)
        flow_sheet.add_unit(tank)
        flow_sheet.add_unit(column)
        flow_sheet.add_unit(outlet, product_outlet=True)

        flow_sheet.add_connection(feed, tank)
        flow_sheet.add_connection(tank, column)
        flow_sheet.add_connection(eluent, column)
        flow_sheet.add_connection(column, tank)
        flow_sheet.add_connection(column, outlet)

        return flow_sheet
