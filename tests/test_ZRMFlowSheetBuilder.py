import unittest

import numpy as np
from CADETProcess.modelBuilder import ZRMFlowSheetBuilder
from CADETProcess.processModel import (
    ComponentSystem,
    Inlet,
    Langmuir,
    LumpedRateModelWithoutPores,
    Outlet,
    Process,
)
from CADETProcess.simulator import Cadet

from tests.processModel.test_flow_sheet import assert_almost_equal_dict


class TestZRMFlowsheetBuilder(unittest.TestCase):
    def setUp(self):
        # --- Base components ---
        self.component_system = ComponentSystem(2)
        self.flow_rate = 1.0
        self.init_c = 1.0

        # Define binding model
        self.binding_model = Langmuir(self.component_system)
        self.binding_model.is_kinetic = True
        self.binding_model.adsorption_rate = [1, 1]
        self.binding_model.desorption_rate = [1, 1]
        self.binding_model.capacity = [2, 2]

        # --- Template zone ---
        self.zone_template = LumpedRateModelWithoutPores(
            self.component_system,
            "zone_template",
            total_porosity=0.6,
            length=4.2e-3,
            axial_dispersion=7e-9,
        )

        # --- Simple Configuration (single zone) ---
        simple_configuration = "axial"
        simple_segmented_area = [1.0]
        simple_volume_in = [1.0]
        simple_volume_out = [1.0]

        self.simple_builder = ZRMFlowSheetBuilder(
            configuration=simple_configuration,
            zone_template=self.zone_template,
            segments_area=simple_segmented_area,
            void_in_volumes=simple_volume_in,
            void_out_volumes=simple_volume_out,
        )

        # Build flow sheet with inlet/outlet
        self.simple_flow_sheet = self.simple_builder.build_flow_sheet()
        inlet = Inlet(self.component_system, "inlet")
        inlet.c = self.init_c
        inlet.flow_rate = self.flow_rate
        outlet = Outlet(self.component_system, "outlet")
        self.simple_flow_sheet.add_unit(inlet)
        self.simple_flow_sheet.add_unit(outlet)
        self.simple_flow_sheet.add_connection(inlet, self.simple_builder.inlet_port)
        self.simple_flow_sheet.add_connection(self.simple_builder.outlet_port, outlet)

        # Wrap into a process
        self.simple_process = Process(self.simple_flow_sheet, 'simple_process')
        self.complex_process = Process(self.simple_flow_sheet, 'complex_process')

        # --- Complex Configuration (two zones, radial) ---
        complex_configuration = "radial"
        complex_segmented_area = [1.0, 1.0]
        complex_volume_in = [1.0, 2.0]
        complex_volume_out = [1.5, 2.5]

        self.complex_builder = ZRMFlowSheetBuilder(
            configuration=complex_configuration,
            zone_template=self.zone_template,
            segments_area=complex_segmented_area,
            void_in_volumes=complex_volume_in,
            void_out_volumes=complex_volume_out,
        )

        self.complex_flow_sheet = self.complex_builder.build_flow_sheet()
        self.complex_flow_sheet.add_unit(inlet)
        self.complex_flow_sheet.add_unit(outlet)
        self.complex_flow_sheet.add_connection(inlet, self.complex_builder.inlet_port)
        self.complex_flow_sheet.add_connection(self.complex_builder.outlet_port, outlet)

    # --- Boundary Tests ---
    def test_invalid_inputs(self):
        # Negative or mismatched volumes or areas
        with self.assertRaises(ValueError):
            ZRMFlowSheetBuilder(
                configuration="axial",
                zone_template=self.zone_template,
                segments_area=[1.0, -1.0],
                void_in_volumes=[1.0, 2.0],
                void_out_volumes=[1.0, 2.0],
            )

        with self.assertRaises(ValueError):
            ZRMFlowSheetBuilder(
                configuration="axial",
                zone_template=self.zone_template,
                segments_area=[1.0, 1.0],
                void_in_volumes=[1.0, -2.0],
                void_out_volumes=[1.0, 2.0],
            )

        with self.assertRaises(ValueError):
            ZRMFlowSheetBuilder(
                configuration="diagonal",
                zone_template=self.zone_template,
                segments_area=[1.0, 1.0],
                void_in_volumes=[1.0, 2.0],
                void_out_volumes=[1.0, 2.0],
            )

    # --- Structural Tests ---
    def test_binding(self):
        self.zone_template.binding_model = self.binding_model
        self.assertIsInstance(self.complex_builder.zone_template.binding_model, Langmuir)

    # --- Connectivity Tests ---
    def test_complex_flow_connections(self):
        flow_rates = self.complex_flow_sheet.get_flow_rates().to_dict()

        flow_rates_expected = {
            "inlet": {
                "total_out": {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                "destinations": {
                    None: {"void_in_0": {None: [np.float64(1.0), 0.0, 0.0, 0.0]}}
                },
            },
            "outlet": {
                "total_in": {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                "origins": {
                    None: {"void_out_1": {None: [np.float64(1.0), 0.0, 0.0, 0.0]}}
                },
            },
            "zone_0": {
                "total_in": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                "total_out": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                "origins": {
                    None: {"void_in_0": {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}
                },
                "destinations": {
                    None: {"void_out_0": {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}
                },
            },
            "zone_1": {
                "total_in": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                "total_out": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                "origins": {
                    None: {"void_in_1": {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}
                },
                "destinations": {
                    None: {"void_out_1": {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}
                },
            },
            "void_in_0": {
                "total_in": {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                "total_out": {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                "origins": {None: {"inlet": {None: [np.float64(1.0), 0.0, 0.0, 0.0]}}},
                "destinations": {
                    None: {
                        "void_in_1": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                        "zone_0": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                    }
                },
            },
            "void_out_0": {
                "total_in": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                "total_out": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                "origins": {None: {"zone_0": {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}},
                "destinations": {
                    None: {"void_out_1": {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}
                },
            },
            "void_in_1": {
                "total_in": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                "total_out": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                "origins": {None: {"void_in_0": {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}},
                "destinations": {None: {"zone_1": {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}},
            },
            "void_out_1": {
                "total_in": {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                "total_out": {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                "origins": {
                    None: {
                        "zone_1": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                        "void_out_0": {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                    }
                },
                "destinations": {None: {"outlet": {None: [np.float64(1.0), 0.0, 0.0, 0.0]}}},
            },
        }

        assert_almost_equal_dict(flow_rates, flow_rates_expected)

    # --- Simulation Test (lightweight) ---
    def test_process_simulation_runs(self):
        simulator = Cadet()
        simple_process = Process(self.simple_flow_sheet, 'simple_process')
        simple_process.cycle_time = 10.0

        result = simulator.simulate(simple_process)
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "solution"))


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
