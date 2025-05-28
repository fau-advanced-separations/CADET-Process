
import unittest
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem, Langmuir, LumpedRateModelWithoutPores
from CADETProcess.simulator import Cadet

from tests.test_flow_sheet import assert_almost_equal_dict

        
class TestZRMBuilder(unittest.TestCase):
    def setUp(self):
        self.component_system = ComponentSystem(2)
        self.flow_rate = 1
        self.binding_model = Langmuir(self.component_system)
        self.init_c = 1

        self.binding_model.is_kinetic = True
        self.binding_model.adsorption_rate = [1,1]
        self.binding_model.desorption_rate = [1,1]
        self.binding_model.capacity = [2,2]

        # --- Simple Configuration ---
        self.simple_configuration = "Axial"
        self.simple_segmented_area = [1]
        self.simple_volume_in = [1]
        self.simple_volume_out = []  # Default for symmetric problem

        self.simple_builder = ZRMBuilder(
            self.component_system,
            self.flow_rate,
            self.simple_segmented_area,
            self.simple_configuration,
            init_c=0,
            binding_model=self.binding_model
        )

        self.simple_builder.build(
            rate_model=LumpedRateModelWithoutPores,
            length=1.0,
            porosity=1,
            axial_dispersion=1,
            volumes_in=self.simple_volume_in,
            volumes_out=self.simple_volume_out
        )
        self.simple_builder.cycle_time = 1000

        # --- Complex Configuration ---
        self.complex_configuration = "radial"
        self.complex_segmented_area = [1, 1]
        self.complex_volume_in = [1, 2]
        self.complex_volume_out = [1.5, 2.5]

        self.complex_builder = ZRMBuilder(
            self.component_system,
            self.flow_rate,
            self.complex_segmented_area,
            self.complex_configuration,
            init_c=self.init_c,
            binding_model=self.binding_model
        )

        self.complex_builder.build(
            rate_model=LumpedRateModelWithoutPores,
            length=1,
            porosity=1,
            axial_dispersion=1,
            volumes_in=self.complex_volume_in,
            volumes_out=self.complex_volume_out
        )
        self.complex_builder.cycle_time = 1000

    def test_binding_model_is_set(self):
        self.simple_builder.binding_model = self.binding_model
        self.assertIsInstance(self.simple_builder.binding_model, Langmuir)
        self.assertIsInstance(self.simple_builder.zones[0].binding_model, Langmuir)

    
    def test_boundary_conditions(self):
        self.extra_volume_in=[1,2,3]
        self.negative_volume_in=[1,-2]
        self.fake_volume_out = [1.5]
        self.negative_segmeted_area=[1,-1]
        self.nonvalid_configuration = "diagonal"

        #Too many inputs for volume_in
        with self.assertRaises(ValueError):
            self.complex_builder.build(
            rate_model=LumpedRateModelWithoutPores,
            length=1,
            porosity=1,
            axial_dispersion=1,
            volumes_in=self.extra_volume_in,
            volumes_out=self.simple_volume_out
        )

        #negative values in cstrs volume    
        with self.assertRaises(ValueError):
            self.complex_builder.build(
            rate_model=LumpedRateModelWithoutPores,
            length=1,
            porosity=1,
            axial_dispersion=1,
            volumes_in=self.negative_volume_in,
            volumes_out=self.simple_volume_out
        )

        #mismatched volumes_out length    
        with self.assertRaises(ValueError):
            self.complex_builder.build(
            rate_model=LumpedRateModelWithoutPores,
            length=1,
            porosity=1,
            axial_dispersion=1,
            volumes_in=self.complex_volume_in,
            volumes_out=self.fake_volume_out
        )

        # Negative segment area       
        with self.assertRaises(ValueError):
            dismatched_builder = ZRMBuilder(
            self.component_system,
            self.flow_rate,
            self.negative_segmeted_area,
            self.complex_configuration,
            init_c=self.init_c,
            binding_model=self.binding_model
        )

        #Non-valid filter configuration    
        with self.assertRaises(ValueError):
            dismatched_builder = ZRMBuilder(
            self.component_system,
            self.flow_rate,
            self.negative_segmeted_area,
            self.nonvalid_configuration,
            init_c=self.init_c,
            binding_model=self.binding_model
        )




    def test_complex_flow_connections(self):
        flow_rates = self.complex_builder.flow_sheet.get_flow_rates().to_dict()

        flow_rates_expected = {
            'inlet': {'total_out': {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                      'destinations': {None: {'cstr_in_1': {None: [np.float64(1.0), 0.0, 0.0, 0.0]}}}},
            'outlet': {'total_in': {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                       'origins': {None: {'cstr_out_2': {None: [np.float64(1.0), 0.0, 0.0, 0.0]}}}},
            'zone_1': {'total_in': {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                       'total_out': {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                       'origins': {None: {'cstr_in_1': {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}},
                       'destinations': {None: {'cstr_out_1': {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}}},
            'zone_2': {'total_in': {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                       'total_out': {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                       'origins': {None: {'cstr_in_2': {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}},
                       'destinations': {None: {'cstr_out_2': {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}}},
            'cstr_in_1': {'total_in': {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                          'total_out': {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                          'origins': {None: {'inlet': {None: [np.float64(1.0), 0.0, 0.0, 0.0]}}},
                          'destinations': {None: {
                              'cstr_in_2': {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                              'zone_1': {None: [np.float64(0.5), 0.0, 0.0, 0.0]}
                          }}},
            'cstr_out_1': {'total_in': {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                           'total_out': {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                           'origins': {None: {'zone_1': {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}},
                           'destinations': {None: {'cstr_out_2': {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}}},
            'cstr_in_2': {'total_in': {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                          'total_out': {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                          'origins': {None: {'cstr_in_1': {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}},
                          'destinations': {None: {'zone_2': {None: [np.float64(0.5), 0.0, 0.0, 0.0]}}}},
            'cstr_out_2': {'total_in': {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                           'total_out': {None: [np.float64(1.0), 0.0, 0.0, 0.0]},
                           'origins': {None: {
                               'zone_2': {None: [np.float64(0.5), 0.0, 0.0, 0.0]},
                               'cstr_out_1': {None: [np.float64(0.5), 0.0, 0.0, 0.0]}
                           }},
                           'destinations': {None: {'outlet': {None: [np.float64(1.0), 0.0, 0.0, 0.0]}}}}
        }

        assert_almost_equal_dict(flow_rates, flow_rates_expected)
    
    def test_process(self):
        simulator = Cadet()
        current_process=self.simple_builder.process
        current_process.cycle_time = 1000
        simulation_results = simulator.simulate(current_process)
        




if __name__ == "__main__":
    unittest.main()
