import unittest
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Langmuir
from CADETProcess.modelBuilder import CompartmentBuilder


class Test_CompartmentBuilder(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.component_system = ComponentSystem(2)

        self.binding_model = Langmuir(self.component_system)

        self.volume_simple = [1, 2, 3, 4, 5]
        self.matrix_simple = [
            0,   0.1, 0.2, 0.3, 0.4,
            0.1, 0,   0,   0,   0,
            0.2, 0,   0,   0,   0,
            0.3, 0,   0,   0,   0,
            0.4, 0,   0,   0,   0,
        ]

        self.builder_simple = CompartmentBuilder(
            self.component_system,
            self.volume_simple, self.matrix_simple,
        )
        self.builder_simple.cycle_time = 1000

        self.volume_complex = ['inlet', 2, 1, 3, 1, 2, 1, 4, 1, 'outlet']
        self.matrix_complex = [
        #   0    1    2    3    4    5    6    7    8    9
            0,   0.1, 0,   0,   0,   0,   0,   0,   0,   0,    # 0
            0,   0,   0.3, 0,   0,   0,   0,   0.1, 0,   0,    # 1
            0,   0.1, 0,   0.1, 0,   0,   0,   0.1, 0,   0,    # 2
            0,   0.2, 0,   0,   0,   0.5, 0,   0,   0.1, 0,    # 3
            0,   0,   0,   0.1, 0,   0.1, 0,   0.1, 0,   0,    # 4
            0,   0,   0,   0,   0.3, 0,   0.2, 0.1, 0,   0,    # 5
            0,   0,   0,   0,   0,   0,   0,   0.1, 0,   0.1,  # 6
            0,   0,   0,   0.5, 0,   0,   0,   0,   0,   0,    # 7
            0,   0,   0,   0.1, 0,   0,   0,   0,   0,   0,    # 8
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    # 9
        ]

        self.builder_complex = CompartmentBuilder(
            self.component_system,
            self.volume_complex, self.matrix_complex,
            init_c=1,
        )
        self.builder_complex.cycle_time = 1000

    def test_binding(self):
        self.builder_simple.binding_model = self.binding_model
        self.assertIsInstance(self.builder_simple.binding_model, Langmuir)
        compartment = self.builder_simple.flow_sheet.compartment_2
        self.assertIsInstance(compartment.binding_model, Langmuir)

    def test_complex(self):
        self.builder_complex.validate_flow_rates()

    def test_connections(self):
        flow_rates_expected = {
            'compartment_0': {
                'total_in': np.array([1., 0., 0., 0.]),
                'total_out': np.array([1., 0., 0., 0.]),
                'origins': {
                    'compartment_1': np.array([0.1, 0., 0., 0.]),
                    'compartment_2': np.array([0.2, 0., 0., 0.]),
                    'compartment_3': np.array([0.3, 0., 0., 0.]),
                    'compartment_4': np.array([0.4, 0., 0., 0.])
                },
                'destinations': {
                    'compartment_1': np.array([0.1, 0., 0., 0.]),
                    'compartment_2': np.array([0.2, 0., 0., 0.]),
                    'compartment_3': np.array([0.3, 0., 0., 0.]),
                    'compartment_4': np.array([0.4, 0., 0., 0.])
                },
            },
            'compartment_1': {
                'total_in': np.array([0.1, 0., 0., 0.]),
                'total_out': np.array([0.1, 0., 0., 0.]),
                'origins': {'compartment_0': np.array([0.1, 0., 0., 0.])},
                'destinations': {'compartment_0': np.array([0.1, 0., 0., 0.])},
            },
            'compartment_2': {
                'total_in': np.array([0.2, 0., 0., 0.]),
                'total_out': np.array([0.2, 0., 0., 0.]),
                'origins': {'compartment_0': np.array([0.2, 0., 0., 0.])},
                'destinations': {'compartment_0': np.array([0.2, 0., 0., 0.])},
            },
            'compartment_3': {
                'total_in': np.array([0.3, 0., 0., 0.]),
                'total_out': np.array([0.3, 0., 0., 0.]),
                'origins': {'compartment_0': np.array([0.3, 0., 0., 0.])},
                'destinations': {'compartment_0': np.array([0.3, 0., 0., 0.])},
            },
            'compartment_4': {
                'total_in': np.array([0.4, 0., 0., 0.]),
                'total_out': np.array([0.4, 0., 0., 0.]),
                'origins': {'compartment_0': np.array([0.4, 0., 0., 0.])},
                'destinations': {'compartment_0': np.array([0.4, 0., 0., 0.])},
            }
        }
        flow_rates = self.builder_simple.flow_sheet.get_flow_rates().to_dict()
        np.testing.assert_equal(flow_rates, flow_rates_expected)

    def test_validate_flow_rates(self):
        self.builder_complex.flow_sheet.compartment_2.flow_rate = 0

        with self.assertRaises(CADETProcessError):
            self.builder_complex.validate_flow_rates()

    def test_initial_conditions(self):
        # All to zero (default)
        builder = CompartmentBuilder(
            self.component_system,
            self.volume_complex, self.matrix_complex,
        )
        c_expected = [0, 0]
        c = builder.flow_sheet.compartment_2.c
        np.testing.assert_almost_equal(c, c_expected)

        # All components, all compartments same value
        builder = CompartmentBuilder(
            self.component_system,
            self.volume_complex, self.matrix_complex,
            init_c=1,
        )
        c_expected = [1, 1]
        c = builder.flow_sheet.compartment_2.c
        np.testing.assert_almost_equal(c, c_expected)

        # All compartments same value
        builder = CompartmentBuilder(
            self.component_system,
            self.volume_complex, self.matrix_complex,
            init_c=[1, 2],
        )
        c_expected = [1, 2]
        c = builder.flow_sheet.compartment_2.c
        np.testing.assert_almost_equal(c, c_expected)

        # Different per zone
        init_c = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4]
        ])

        builder = CompartmentBuilder(
            self.component_system,
            self.volume_simple, self.matrix_simple,
            init_c=init_c,
        )
        c = builder.flow_sheet.compartment_2.c
        np.testing.assert_almost_equal(c, init_c[2].tolist())

    def test_tracer(self):
        """
        ..todo::
            flow rate validation not working for flow rate filter
        """
        self.builder_complex.add_tracer(4, [1, 1], 10, 0.1)

    def test_process(self):
        self.builder_complex.cycle_time = 1000

        self.builder_complex.add_tracer(4, [1, 1], 10, 0.1)

        from CADETProcess.simulator import Cadet
        process_simulator = Cadet()

        proc_results = process_simulator.simulate(self.builder_complex.process)


if __name__ == '__main__':
    unittest.main()
