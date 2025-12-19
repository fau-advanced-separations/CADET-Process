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


class CLR(Process):
    """
    Closed loop recycling (CLR) process.

    The flowsheet includes:
    - feed: Inlet
    - eluent: Inlet
    - column: ChromatographicColumnBase
    - pump: Cstr (auxiliary pump unit)
    - outlet: Outlet

    The process is configured with the following events:
    - Feed injection (duration: `feed_duration`).
    - Recycling (ends at `recycle_off`).
    - Elution after recycling.
    """

    def __init__(
        self,
        column: ChromatographicColumnBase,
        c_feed: list[float],
        flow_rate: float,
        feed_duration: float,
        recycle_off: float,
        cycle_time: float,
        c_eluent: list[float] | float = 0.0,
        pump_volume: float = 1e-9,
    ) -> None:
        """
        Initialize CLR process.

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
        recycle_off : float
            Time at which recycling ends.
        cycle_time : float
            Total cycle time.
        c_eluent : list[float] | float | None, optional
            Eluent concentration. Defaults to 0.0.
        pump_volume : float, optional
            Volume of the auxiliary pump unit. Defaults to 1e-9.
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
            pump_volume=pump_volume,
        )
        super().__init__(flow_sheet, "CLR")
        self.cycle_time = cycle_time

        # 0: Start feeding, no eluent, no recycle
        self.add_event("feed_on", "flow_sheet.feed.flow_rate", flow_rate, 0.0)
        self.add_event("eluent_off", "flow_sheet.eluent.flow_rate", 0.0)
        self.add_event_dependency("eluent_off", ["feed_on"])

        # t_inj / t_rec_start: End feeding, no eluent, start recycle
        self.add_event("feed_off", "flow_sheet.feed.flow_rate", 0.0)
        self.add_duration("feed_duration", feed_duration)
        self.add_event_dependency("feed_off", ["feed_on", "feed_duration"], [1, 1])
        self.add_event(
            "recycle_on_output_state",
            f"flow_sheet.output_states.{column.name}",
            {"pump": 1},
        )
        self.add_event_dependency("recycle_on_output_state", ["feed_off"])
        self.add_event("recycle_on_pump", "flow_sheet.pump.flow_rate", flow_rate)
        self.add_event_dependency("recycle_on_pump", ["recycle_on_output_state"])

        # t_rec_end: End recycle, start eluent
        self.add_event(
            "recycle_off_output_state",
            f"flow_sheet.output_states.{column.name}",
            {"outlet": 1},
            recycle_off,
        )
        self.add_event("recycle_off_pump", "flow_sheet.pump.flow_rate", 0)
        self.add_event_dependency("recycle_off_pump", ["recycle_off_output_state"])
        self.add_event("eluent_on", "flow_sheet.eluent.flow_rate", flow_rate)
        self.add_event_dependency("eluent_on", ["recycle_off_output_state"])

    def _build_flow_sheet(
        self,
        column: ChromatographicColumnBase,
        c_feed: list[float],
        c_eluent: list[float] | float = 0.0,
        pump_volume: float = 1e-9,
    ) -> FlowSheet:
        """Build and return the flow sheet for CLR process."""
        component_system = column.component_system

        # Unit Operations
        feed = Inlet(component_system, name="feed")
        feed.c = c_feed

        eluent = Inlet(component_system, name="eluent")
        eluent.c = c_eluent if c_eluent is not None else 0.0

        pump = Cstr(component_system, name="pump")
        pump.V = pump_volume

        outlet = Outlet(component_system, name="outlet")

        # Flow Sheet
        flow_sheet = FlowSheet(component_system)

        flow_sheet.add_unit(feed, feed_inlet=True)
        flow_sheet.add_unit(eluent, eluent_inlet=True)
        flow_sheet.add_unit(pump)
        flow_sheet.add_unit(column)
        flow_sheet.add_unit(outlet, product_outlet=True)

        flow_sheet.add_connection(feed, column)
        flow_sheet.add_connection(eluent, column)
        flow_sheet.add_connection(column, outlet)
        flow_sheet.add_connection(column, pump)
        flow_sheet.add_connection(pump, column)

        return flow_sheet
