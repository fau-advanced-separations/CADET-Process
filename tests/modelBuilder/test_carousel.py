import unittest

import numpy as np
from CADETProcess.modelBuilder import CarouselBuilder, ParallelZone, SerialZone
from CADETProcess.processModel import (
    ComponentSystem,
    Inlet,
    Linear,
    LumpedRateModelWithoutPores,
    Outlet,
)
from CADETProcess.simulator import Cadet


class Test_Carousel(unittest.TestCase):
    def setUp(self):
        self.component_system = ComponentSystem(2)

        self.binding_model = Linear(self.component_system)
        self.binding_model.adsorption_rate = [6, 8]
        self.binding_model.desorption_rate = [1, 1]

        self.column = LumpedRateModelWithoutPores(
            self.component_system, name="master_column"
        )
        self.column.length = 0.6
        self.column.diameter = 0.024
        self.column.axial_dispersion = 4.7e-7
        self.column.total_porosity = 0.7

        self.column.binding_model = self.binding_model

    def create_serial(self):
        source = Inlet(self.component_system, name="source")
        source.c = [10, 10]
        source.flow_rate = 2e-7

        sink = Outlet(self.component_system, name="sink")

        serial_zone = SerialZone(self.component_system, "serial", 2, flow_direction=1)

        builder = CarouselBuilder(self.component_system, "serial")
        builder.column = self.column

        builder.add_unit(source)
        builder.add_unit(sink)
        builder.add_unit(serial_zone)

        builder.add_connection(source, serial_zone)
        builder.add_connection(serial_zone, sink)

        builder.switch_time = 300

        return builder

    def create_parallel(self):
        source = Inlet(self.component_system, name="source")
        source.c = [10, 10]
        source.flow_rate = 2e-7

        sink = Outlet(self.component_system, name="sink")

        parallel_zone = ParallelZone(
            self.component_system, "parallel", 2, flow_direction=1
        )

        builder = CarouselBuilder(self.component_system, "parallel")
        builder.column = self.column

        builder.add_unit(source)
        builder.add_unit(sink)
        builder.add_unit(parallel_zone)

        builder.add_connection(source, parallel_zone)
        builder.add_connection(parallel_zone, sink)

        builder.switch_time = 300

        return builder

    def create_smb(self):
        feed = Inlet(self.component_system, name="feed")
        feed.c = [10, 10]
        feed.flow_rate = 2e-7

        eluent = Inlet(self.component_system, name="eluent")
        eluent.c = [0, 0]
        eluent.flow_rate = 6e-7

        raffinate = Outlet(self.component_system, name="raffinate")
        extract = Outlet(self.component_system, name="extract")

        zone_I = SerialZone(self.component_system, "zone_I", 1)
        zone_II = SerialZone(self.component_system, "zone_II", 1)
        zone_III = SerialZone(self.component_system, "zone_III", 1)
        zone_IV = SerialZone(self.component_system, "zone_IV", 1)

        builder = CarouselBuilder(self.component_system, "smb")
        builder.column = self.column
        builder.add_unit(feed)
        builder.add_unit(eluent)

        builder.add_unit(raffinate)
        builder.add_unit(extract)

        builder.add_unit(zone_I)
        builder.add_unit(zone_II)
        builder.add_unit(zone_III)
        builder.add_unit(zone_IV)

        builder.add_connection(eluent, zone_I)

        builder.add_connection(zone_I, extract)
        builder.add_connection(zone_I, zone_II)
        w_e = 0.15
        builder.set_output_state(zone_I, [w_e, 1 - w_e])

        builder.add_connection(zone_II, zone_III)

        builder.add_connection(feed, zone_III)

        builder.add_connection(zone_III, raffinate)
        builder.add_connection(zone_III, zone_IV)
        w_r = 0.15
        builder.set_output_state(zone_III, [w_r, 1 - w_r])

        builder.add_connection(zone_IV, zone_I)

        builder.switch_time = 300

        return builder

    def create_multi_zone(self):
        source_serial = Inlet(self.component_system, name="source_serial")
        source_serial.c = [10, 10]
        source_serial.flow_rate = 2e-7

        sink_serial = Outlet(self.component_system, name="sink_serial")

        serial_zone = SerialZone(self.component_system, "serial", 2, flow_direction=1)

        source_parallel = Inlet(self.component_system, name="source_parallel")
        source_parallel.c = [10, 10]
        source_parallel.flow_rate = 2e-7

        sink_parallel = Outlet(self.component_system, name="sink_parallel")

        parallel_zone = ParallelZone(
            self.component_system, "parallel", 2, flow_direction=-1
        )

        builder = CarouselBuilder(self.component_system, "multi_zone")
        builder.column = self.column
        builder.add_unit(source_serial)
        builder.add_unit(source_parallel)

        builder.add_unit(sink_serial)
        builder.add_unit(sink_parallel)

        builder.add_unit(serial_zone)
        builder.add_unit(parallel_zone)

        builder.add_connection(source_serial, serial_zone)
        builder.add_connection(serial_zone, sink_serial)
        builder.add_connection(serial_zone, parallel_zone)
        builder.set_output_state(serial_zone, [0.5, 0.5])

        builder.add_connection(source_parallel, parallel_zone)
        builder.add_connection(parallel_zone, sink_parallel)

        builder.switch_time = 300

        return builder

    def test_units(self):
        """Check if all units are added properly in the FlowSheet"""
        # Serial
        builder = self.create_serial()
        flow_sheet = builder.build_flow_sheet()

        units_expected = [
            "source",
            "sink",
            "serial_inlet",
            "serial_outlet",
            "column_0",
            "column_1",
        ]

        self.assertEqual(units_expected, flow_sheet.unit_names)

        # Parallel
        builder = self.create_parallel()
        flow_sheet = builder.build_flow_sheet()

        units_expected = [
            "source",
            "sink",
            "parallel_inlet",
            "parallel_outlet",
            "column_0",
            "column_1",
        ]

        self.assertEqual(units_expected, flow_sheet.unit_names)

        # SMB
        builder = self.create_smb()
        flow_sheet = builder.build_flow_sheet()

        units_expected = [
            "feed",
            "eluent",
            "raffinate",
            "extract",
            "zone_I_inlet",
            "zone_I_outlet",
            "column_0",
            "zone_II_inlet",
            "zone_II_outlet",
            "column_1",
            "zone_III_inlet",
            "zone_III_outlet",
            "column_2",
            "zone_IV_inlet",
            "zone_IV_outlet",
            "column_3",
        ]

        self.assertEqual(units_expected, flow_sheet.unit_names)

    def test_connections(self):
        """Check if all units are connected properly in the FlowSheet"""
        # Serial
        builder = self.create_serial()
        flow_sheet = builder.build_flow_sheet()

        self.assertTrue(flow_sheet.connection_exists("source", "serial_inlet"))

        self.assertTrue(flow_sheet.connection_exists("serial_inlet", "column_0"))
        self.assertTrue(flow_sheet.connection_exists("serial_inlet", "column_1"))

        self.assertTrue(flow_sheet.connection_exists("column_0", "column_1"))
        self.assertTrue(flow_sheet.connection_exists("column_0", "serial_outlet"))

        self.assertTrue(flow_sheet.connection_exists("column_1", "column_0"))
        self.assertTrue(flow_sheet.connection_exists("column_1", "serial_outlet"))

        # Parallel
        builder = self.create_parallel()
        flow_sheet = builder.build_flow_sheet()

        self.assertTrue(flow_sheet.connection_exists("source", "parallel_inlet"))

        self.assertTrue(flow_sheet.connection_exists("parallel_inlet", "column_0"))

        self.assertTrue(flow_sheet.connection_exists("parallel_inlet", "column_1"))

        self.assertTrue(flow_sheet.connection_exists("column_0", "parallel_outlet"))

        self.assertTrue(flow_sheet.connection_exists("column_1", "parallel_outlet"))

        # SMB
        builder = self.create_smb()
        flow_sheet = builder.build_flow_sheet()

        self.assertTrue(flow_sheet.connection_exists("eluent", "zone_I_inlet"))
        self.assertTrue(flow_sheet.connection_exists("feed", "zone_III_inlet"))

        self.assertTrue(flow_sheet.connection_exists("zone_I_outlet", "extract"))
        self.assertTrue(flow_sheet.connection_exists("zone_I_outlet", "zone_II_inlet"))

        self.assertTrue(
            flow_sheet.connection_exists("zone_II_outlet", "zone_III_inlet")
        )

        self.assertTrue(flow_sheet.connection_exists("zone_III_outlet", "raffinate"))
        self.assertTrue(
            flow_sheet.connection_exists("zone_III_outlet", "zone_IV_inlet")
        )

        self.assertTrue(flow_sheet.connection_exists("zone_IV_outlet", "zone_I_inlet"))

        self.assertTrue(flow_sheet.connection_exists("zone_I_inlet", "column_0"))
        self.assertTrue(flow_sheet.connection_exists("zone_I_inlet", "column_1"))
        self.assertTrue(flow_sheet.connection_exists("zone_I_inlet", "column_2"))
        self.assertTrue(flow_sheet.connection_exists("zone_I_inlet", "column_3"))

        self.assertTrue(flow_sheet.connection_exists("zone_II_inlet", "column_0"))
        self.assertTrue(flow_sheet.connection_exists("zone_II_inlet", "column_1"))
        self.assertTrue(flow_sheet.connection_exists("zone_II_inlet", "column_2"))
        self.assertTrue(flow_sheet.connection_exists("zone_II_inlet", "column_3"))

        self.assertTrue(flow_sheet.connection_exists("zone_III_inlet", "column_0"))
        self.assertTrue(flow_sheet.connection_exists("zone_III_inlet", "column_1"))
        self.assertTrue(flow_sheet.connection_exists("zone_III_inlet", "column_2"))
        self.assertTrue(flow_sheet.connection_exists("zone_III_inlet", "column_3"))

        self.assertTrue(flow_sheet.connection_exists("zone_IV_inlet", "column_0"))
        self.assertTrue(flow_sheet.connection_exists("zone_IV_inlet", "column_1"))
        self.assertTrue(flow_sheet.connection_exists("zone_IV_inlet", "column_2"))
        self.assertTrue(flow_sheet.connection_exists("zone_IV_inlet", "column_3"))

        self.assertTrue(flow_sheet.connection_exists("column_0", "zone_I_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_0", "zone_II_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_0", "zone_III_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_0", "zone_IV_outlet"))

        self.assertTrue(flow_sheet.connection_exists("column_1", "zone_I_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_1", "zone_II_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_1", "zone_III_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_1", "zone_IV_outlet"))

        self.assertTrue(flow_sheet.connection_exists("column_2", "zone_I_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_2", "zone_II_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_2", "zone_III_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_2", "zone_IV_outlet"))

        self.assertTrue(flow_sheet.connection_exists("column_3", "zone_I_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_3", "zone_II_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_3", "zone_III_outlet"))
        self.assertTrue(flow_sheet.connection_exists("column_3", "zone_IV_outlet"))

    def test_column_position_indices(self):
        """Test column position indices."""
        builder = self.create_smb()

        # Initial state, position 0
        carousel_position = 0
        carousel_state = 0
        indices_expected = 0

        indices = builder.column_indices_at_state(carousel_position, carousel_state)
        self.assertEqual(indices_expected, indices)

        time = carousel_state * builder.switch_time
        indices = builder.column_indices_at_time(time, carousel_position)
        self.assertEqual(indices_expected, indices)

        # Initial state, position 1
        carousel_position = 1
        carousel_state = 0
        indices_expected = 1

        indices = builder.column_indices_at_state(carousel_position, carousel_state)
        self.assertEqual(indices_expected, indices)

        time = carousel_state * builder.switch_time
        indices = builder.column_indices_at_time(time, carousel_position)
        self.assertEqual(indices_expected, indices)

        # First state, position 0
        carousel_position = 0
        carousel_state = 1
        indices_expected = 1

        indices = builder.column_indices_at_state(carousel_position, carousel_state)
        self.assertEqual(indices_expected, indices)

        time = carousel_state * builder.switch_time
        indices = builder.column_indices_at_time(time, carousel_position)
        self.assertEqual(indices_expected, indices)

        # First state, position 1
        carousel_position = 1
        carousel_state = 1
        indices_expected = 2

        indices = builder.column_indices_at_state(carousel_position, carousel_state)
        self.assertEqual(indices_expected, indices)

        time = carousel_state * builder.switch_time
        indices = builder.column_indices_at_time(time, carousel_position)
        self.assertEqual(indices_expected, indices)

        # 4th state (back to initial state), position 0
        carousel_position = 0
        carousel_state = 4
        indices_expected = 0

        indices = builder.column_indices_at_state(carousel_position, carousel_state)
        self.assertEqual(indices_expected, indices)

        time = carousel_state * builder.switch_time
        indices = builder.column_indices_at_time(time, carousel_position)
        self.assertEqual(indices_expected, indices)

    def test_carousel_state(self):
        """Test carousel state."""

        builder = self.create_smb()

        # Initial state
        time = 0
        state_expected = 0

        state = builder.carousel_state(time)

        self.assertEqual(state_expected, state)

        # Position 0
        time = builder.switch_time / 2
        state_expected = 0

        state = builder.carousel_state(time)

        self.assertEqual(state_expected, state)

        # Position 1
        time = builder.switch_time
        state_expected = 1

        state = builder.carousel_state(time)

        self.assertEqual(state_expected, state)

        # Back to initial state; position 0
        time = 4 * builder.switch_time
        state_expected = 0

        state = builder.carousel_state(time)

        self.assertEqual(state_expected, state)

    def test_flow_rates(self):
        # Serial
        builder = self.create_serial()
        process = builder.build_process()

        serial_inlet = process.flow_rate_timelines["serial_inlet"]
        serial_outlet = process.flow_rate_timelines["serial_outlet"]
        column_0 = process.flow_rate_timelines["column_0"]
        column_1 = process.flow_rate_timelines["column_1"]

        flow_rate = serial_inlet.total_in[None].value(0)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        flow_rate = serial_outlet.total_in[None].value(0)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        flow_rate = column_0.total_in[None].value(0)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        flow_rate = column_1.total_in[None].value(0)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        # Parallel (flow is split between columns)
        builder = self.create_parallel()
        process = builder.build_process()

        parallel_inlet = process.flow_rate_timelines["parallel_inlet"]
        column_0 = process.flow_rate_timelines["column_0"]
        column_1 = process.flow_rate_timelines["column_1"]

        flow_rate = parallel_inlet.total_in[None].value(0)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        flow_rate = parallel_inlet.total_in[None].value(0)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        flow_rate = column_0.total_in[None].value(0)
        flow_rate_expected = 1e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        flow_rate = column_1.total_in[None].value(0)
        flow_rate_expected = 1e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        # Multi-Zone (Side streams between zones)
        builder = self.create_multi_zone()
        process = builder.build_process()

        serial_inlet = process.flow_rate_timelines["serial_inlet"]
        parallel_inlet = process.flow_rate_timelines["parallel_inlet"]
        column_0 = process.flow_rate_timelines["column_0"]
        column_2 = process.flow_rate_timelines["column_2"]

        flow_rate = serial_inlet.total_in[None].value(0)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        flow_rate = parallel_inlet.total_in[None].value(0)
        flow_rate_expected = 3e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        # Initial state
        t = 0
        flow_rate = column_0.total_in[None].value(t)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        flow_rate = column_2.total_in[None].value(t)
        flow_rate_expected = 1.5e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        # First position
        t = builder.switch_time
        flow_rate = column_0.total_in[None].value(t)
        flow_rate_expected = 1.5e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        flow_rate = column_2.total_in[None].value(t)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

    def test_flow_direction(self):
        # Multi-Zone (Side streams between zones)
        builder = self.create_multi_zone()
        process = builder.build_process()

        # Initial state
        t = 0

        tl = process.parameter_timelines["flow_sheet.column_0.flow_direction"]
        flow_direction = tl.value(t)
        flow_direction_expected = 1
        np.testing.assert_almost_equal(flow_direction, flow_direction_expected)

        tl = process.parameter_timelines["flow_sheet.column_2.flow_direction"]
        flow_direction = tl.value(t)
        flow_direction_expected = -1
        np.testing.assert_almost_equal(flow_direction, flow_direction_expected)

        # First position
        t = builder.switch_time

        tl = process.parameter_timelines["flow_sheet.column_0.flow_direction"]
        flow_direction = tl.value(t)
        flow_direction_expected = -1
        np.testing.assert_almost_equal(flow_direction, flow_direction_expected)

        tl = process.parameter_timelines["flow_sheet.column_2.flow_direction"]
        flow_direction = tl.value(t)
        flow_direction_expected = 1
        np.testing.assert_almost_equal(flow_direction, flow_direction_expected)

    def test_simulation(self):
        builder = self.create_serial()
        process = builder.build_process()

        process_simulator = Cadet()
        simulation_results = process_simulator.simulate(process)

        self.assertEqual(simulation_results.exit_flag, 0)


if __name__ == "__main__":
    unittest.main()
