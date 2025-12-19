import copy
import warnings
from typing import Optional

from CADETProcess.processModel import (
    ChromatographicColumnBase,
    FlowSheet,
    Inlet,
    Outlet,
    Process,
)


class BatchElution(Process):
    """
    Batch elution process.

    The flowsheet is configured with the following unit operations:
    - feed: Inlet
    - eluent: Inlet
    - column: ChromatographicColumnBase
    - outlet: Outlet

    The following durations/events are configured:
    - `feed_duration`: The length of the feed injection.
    - `feed_on`: Set `feed.flow_rate` to `flow_rate`.
    - `eluent_off`: Set `eluent.flow_rate` to 0.0; triggered by `feed_on`.
    - `feed_off`: Set `feed.flow_rate` to 0.0; triggered by `feed_on` and `feed_duration`.
    - `eluent_on`: Set `eluent.flow_rate` to `flow_rate`; triggered by `feed_off`.
    """

    def __init__(
        self,
        column: ChromatographicColumnBase,
        c_feed: list[float],
        flow_rate: float,
        feed_duration: float,
        cycle_time: float,
        c_eluent: Optional[list[float] | float] = 0.0,
    ) -> None:
        """
        Initialize batch elution process.

        Parameters
        ----------
        column : ChromatographicColumnBase
            Chromatographic column object to be used in process.
        c_feed : list[float]
            Feed concentration.
        flow_rate : float
            Flow rate.
        feed_duration : float
            Feed duration.
        cycle_time : float
            Cycle time.
        c_eluent : list[float] | float, optional
            Eluent concentration. The default is 0.0.
        """
        if not isinstance(column, ChromatographicColumnBase):
            raise TypeError("Expected ChromatographicColumnBase.")

        column = copy.deepcopy(column)
        if not column.name == "column":
            warnings.warn("Renaming column to `column` for consistency")
            column.name = "column"

        flow_sheet = self._build_flow_sheet(
            column,
            c_feed,
            c_eluent,
        )

        super().__init__(flow_sheet, "Batch Elution")
        self.cycle_time = cycle_time

        # Durations
        self.add_duration("feed_duration", feed_duration)

        # Injection
        self.add_event("feed_on", "flow_sheet.feed.flow_rate", flow_rate)
        self.add_event("eluent_off", "flow_sheet.eluent.flow_rate", 0.0)
        self.add_event_dependency("eluent_off", ["feed_on"])

        # Elution
        self.add_event("feed_off", "flow_sheet.feed.flow_rate", 0.0)
        self.add_event_dependency("feed_off", ["feed_on", "feed_duration"], [1, 1])
        self.add_event("eluent_on", "flow_sheet.eluent.flow_rate", flow_rate)
        self.add_event_dependency("eluent_on", ["feed_off"])

    def _build_flow_sheet(
        self,
        column: ChromatographicColumnBase,
        c_feed: list[float],
        c_eluent: list[float] | float | None = 0.0,
    ) -> FlowSheet:
        """Build and return the flow sheet for batch elution process."""
        component_system = column.component_system

        # Unit Operations
        feed = Inlet(component_system, name="feed")
        feed.c = c_feed

        eluent = Inlet(component_system, name="eluent")
        eluent.c = c_eluent

        outlet = Outlet(component_system, name="outlet")

        # Flow Sheet
        flow_sheet = FlowSheet(component_system)

        flow_sheet.add_unit(feed, feed_inlet=True)
        flow_sheet.add_unit(eluent, eluent_inlet=True)
        flow_sheet.add_unit(column)
        flow_sheet.add_unit(outlet, product_outlet=True)

        flow_sheet.add_connection(feed, column)
        flow_sheet.add_connection(eluent, column)
        flow_sheet.add_connection(column, outlet)

        return flow_sheet
