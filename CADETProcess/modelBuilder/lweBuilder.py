import copy
import warnings

from CADETProcess.processModel import (
    ChromatographicColumnBase,
    FlowSheet,
    Inlet,
    Outlet,
    Process,
)


class LWE(Process):
    """
    Load-Wash-Elute (LWE) process.

    The flowsheet includes:
    - load: Inlet (feed)
    - salt_low: Inlet (low salt buffer)
    - salt_high: Inlet (high salt buffer)
    - column: ChromatographicColumnBase
    - outlet: Outlet

    The process is configured with the following phases:
    - Load (duration: `load_duration`)
    - Wash (duration: `wash_duration`)
    - Elute (duration: `gradient_duration`, with gradient)
    - Optional final wash (duration: `final_wash_duration`)
    """

    def __init__(
        self,
        column: ChromatographicColumnBase,
        c_load: list[float],
        c_salt_low: list[float],
        c_salt_high: list[float],
        flow_rate: float,
        load_duration: float,
        wash_duration: float,
        gradient_duration: float,
        final_wash_duration: float = 0.0,
        set_initial_conditions: bool = True,
    ) -> None:
        """
        Initialize LWE process.

        Parameters
        ----------
        column : ChromatographicColumnBase
            Chromatographic column object.
        c_load : list[float]
            Feed concentration.
        c_salt_low : list[float]
            Low salt buffer concentration.
        c_salt_high : list[float]
            High salt buffer concentration.
        flow_rate : float
            Flow rate.
        load_duration : float
            Loading duration.
        wash_duration : float
            Wash duration.
        gradient_duration : float
            Gradient duration.
        final_wash_duration : float, optional
            Optional final wash step with high salt buffer. Defaults to 0.0.
        set_initial_conditions : bool, optional
            Whether to set initial column conditions. Defaults to True.
        """
        if not isinstance(column, ChromatographicColumnBase):
            raise TypeError("Expected ChromatographicColumnBase.")

        column = copy.deepcopy(column)
        if not column.name == "column":
            warnings.warn("Renaming column to `column` for consistency")
            column.name = "column"

        flow_sheet = self._build_flow_sheet(
            column=column,
            c_load=c_load,
            c_salt_low=c_salt_low,
            c_salt_high=c_salt_high,
            set_initial_conditions=set_initial_conditions,
        )
        super().__init__(flow_sheet, 'LWE')

        self.cycle_time = (
            load_duration
            + wash_duration
            + gradient_duration
            + final_wash_duration
        )
        gradient_slope = flow_rate / gradient_duration

        # Durations
        self.add_duration('load_duration', load_duration)
        self.add_duration('wash_duration', wash_duration)
        self.add_duration('gradient_duration', gradient_duration)
        if final_wash_duration > 0:
            self.add_duration('final_wash_duration', final_wash_duration)

        # Events
        # Load
        self.add_event('load_on', 'flow_sheet.load.flow_rate', flow_rate, 0)
        # Wash
        self.add_event('load_off', 'flow_sheet.load.flow_rate', 0.0)
        self.add_event_dependency('load_off', ['load_on', 'load_duration'], [1, 1])
        self.add_event('wash_on', 'flow_sheet.salt_low.flow_rate', flow_rate)
        self.add_event_dependency('wash_on', ['load_off'])
        # Elute
        self.add_event(
            'gradient_salt_low_start',
            'flow_sheet.salt_low.flow_rate',
            [flow_rate, -gradient_slope],
            0,
        )
        self.add_event_dependency(
            'gradient_salt_low_start',
            ['wash_on', 'wash_duration'],
            [1, 1],
        )
        self.add_event(
            'gradient_salt_high_start',
            'flow_sheet.salt_high.flow_rate',
            [0, gradient_slope],
            0,
        )
        self.add_event_dependency('gradient_salt_high_start', ['gradient_salt_low_start'])
        # Final Wash
        self.add_event('gradient_salt_low_end', 'flow_sheet.salt_low.flow_rate', 0.0)
        self.add_event_dependency(
            'gradient_salt_low_end',
            ['gradient_salt_low_start', 'gradient_duration'],
            [1, 1],
        )
        if final_wash_duration > 0:
            self.add_event(
                'final_wash_on',
                'flow_sheet.salt_high.flow_rate',
                flow_rate,
            )
            self.add_event_dependency('final_wash_on', ['gradient_salt_low_end'])
            self.add_event('final_wash_off', 'flow_sheet.salt_high.flow_rate', 0.0)
            self.add_event_dependency(
                'final_wash_off',
                ['final_wash_on', 'final_wash_duration'],
                [1, 1],
            )

    def _build_flow_sheet(
        self,
        column: ChromatographicColumnBase,
        c_load: list[float],
        c_salt_low: list[float],
        c_salt_high: list[float],
        set_initial_conditions: bool = True,
    ) -> FlowSheet:
        """Build and return the flow sheet for LWE process."""
        component_system = column.component_system

        # Unit Operations
        load = Inlet(component_system, name='load')
        load.c = c_load
        salt_low = Inlet(component_system, name='salt_low')
        salt_low.c = c_salt_low
        salt_high = Inlet(component_system, name='salt_high')
        salt_high.c = c_salt_high
        if set_initial_conditions:
            column.c = c_salt_low
            if "cp" in column.parameters:
                column.cp = c_salt_low
        outlet = Outlet(component_system, name='outlet')

        # Flow Sheet
        flow_sheet = FlowSheet(component_system)

        flow_sheet.add_unit(load, feed_inlet=True)
        flow_sheet.add_unit(salt_low, eluent_inlet=True)
        flow_sheet.add_unit(salt_high, eluent_inlet=True)
        flow_sheet.add_unit(column)
        flow_sheet.add_unit(outlet, product_outlet=True)

        flow_sheet.add_connection(load, column)
        flow_sheet.add_connection(salt_low, column)
        flow_sheet.add_connection(salt_high, column)
        flow_sheet.add_connection(column, outlet)

        return flow_sheet
