import warnings

from CADETProcess.processModel import (
    ChromatographicColumnBase,
    FlowSheet,
    Inlet,
    Outlet,
    Process,
)


class FlipFlop(Process):
    """
    Flip-flop process.

    The flowsheet includes:
    - feed: Inlet
    - eluent: Inlet
    - column: ChromatographicColumnBase
    - outlet: Outlet

    The process is configured with two injections and column flow direction flips:
    - Injection 1 (duration: `feed_duration`)
    - Flip column flow direction after `delay_flip`
    - Injection 2 (delayed by `delay_injection`)
    - Flip column flow direction back
    """

    def __init__(
        self,
        column: ChromatographicColumnBase,
        c_feed: list[float],
        flow_rate: float,
        feed_duration: float,
        delay_flip: float,
        delay_injection: float,
        c_eluent: list[float] | float = 0.0,
    ) -> None:
        """
        Initialize flip-flop process.

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
        delay_flip : float
            Time delay before flipping column flow direction.
        delay_injection : float
            Time delay before second injection.
        c_eluent : list[float] | float | None, optional
            Eluent concentration. Defaults to 0.0.
        """
        if not isinstance(column, ChromatographicColumnBase):
            raise TypeError("Expected ChromatographicColumnBase.")

        if not column.name == "column":
            warnings.warn("Renaming column to `column` for consistency")
            column.name = "column"

        flow_sheet = self._build_flow_sheet(
            column=column,
            c_feed=c_feed,
            c_eluent=c_eluent,
        )
        super().__init__(flow_sheet, "FlipFlop")
        self.cycle_time = 2 * (feed_duration + delay_flip + delay_injection)

        # Durations
        self.add_duration("feed_duration", feed_duration)
        self.add_duration("delay_flip", delay_flip)
        self.add_duration("delay_injection", delay_injection)

        # Injection 1
        self.add_event("feed_on_1", "flow_sheet.feed.flow_rate", flow_rate, 0)
        self.add_event("eluent_off_1", "flow_sheet.eluent.flow_rate", 0.0)
        self.add_event_dependency("eluent_off_1", ["feed_on_1"])
        self.add_event("feed_off_1", "flow_sheet.feed.flow_rate", 0.0)
        self.add_event_dependency("feed_off_1", ["feed_on_1", "feed_duration"], [1, 1])
        self.add_event("eluent_on_1", "flow_sheet.eluent.flow_rate", flow_rate)
        self.add_event_dependency("eluent_on_1", ["feed_off_1"])

        # Flip backwards
        self.add_event(
            "backward_flow",
            f"flow_sheet.{column.name}.flow_direction",
            -1,
        )
        self.add_event_dependency("backward_flow", ["feed_off_1", "delay_flip"], [1, 1])

        # Injection 2
        self.add_event("feed_on_2", "flow_sheet.feed.flow_rate", flow_rate)
        self.add_event_dependency("feed_on_2", ["backward_flow", "delay_injection"], [1, 1])
        self.add_event("eluent_off_2", "flow_sheet.eluent.flow_rate", 0.0)
        self.add_event_dependency("eluent_off_2", ["feed_on_2"])
        self.add_event("feed_off_2", "flow_sheet.feed.flow_rate", 0.0)
        self.add_event_dependency("feed_off_2", ["feed_on_2", "feed_duration"], [1, 1])
        self.add_event("eluent_on_2", "flow_sheet.eluent.flow_rate", flow_rate)
        self.add_event_dependency("eluent_on_2", ["feed_off_2"])

        # Flip forwards
        self.add_event(
            "forward_flow",
            f"flow_sheet.{column.name}.flow_direction",
            1,
        )
        self.add_event_dependency("forward_flow", ["eluent_on_2", "delay_flip"], [1, 1])

    def _build_flow_sheet(
        self,
        column: ChromatographicColumnBase,
        c_feed: list[float],
        c_eluent: list[float] | float = 0.0,
    ) -> FlowSheet:
        """Build and return the flow sheet for flip-flop process."""
        component_system = column.component_system

        # Unit Operations
        feed = Inlet(component_system, name="feed")
        feed.c = c_feed

        eluent = Inlet(component_system, name="eluent")
        eluent.c = c_eluent if c_eluent is not None else 0.0

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
