import unittest
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Source, Sink, Cstr
from CADETProcess.modelBuilder import CompartmentBuilder

class Test_CompartmentBuilder(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.component_system = ComponentSystem(2)

        self.volume_simple = [1, 2, 3, 4, 5]
        self.matrix_simple = [
            0,   0.1, 0.2, 0.3, 0.4,
            0.1, 0,   0,   0,   0,
            0.2, 0,   0,   0,   0,
            0.3, 0,   0,   0,   0,
            0.4, 0,   0,   0,   0,
        ]

        self.volume_complex = ['inlet', 2, 1, 3, 1, 2, 1, 4, 1, 'outlet']
        self.matrix_complex = [
        #   0    1    2    3    4    5    6    7    8    9
            0,   0.1, 0,   0,   0,   0,   0,   0,   0,   0,   # 0
            0,   0,   0.3, 0,   0,   0,   0,   0.1, 0,   0,   # 1
            0,   0.1, 0,   0.1, 0,   0,   0,   0.1, 0,   0,   # 2
            0,   0.2, 0,   0,   0,   0.5, 0,   0,   0.1, 0,   # 3
            0,   0,   0,   0.1, 0,   0.1, 0,   0.1, 0,   0,   # 4
            0,   0,   0,   0,   0.3, 0,   0.2, 0.1, 0,   0,   # 5
            0,   0,   0,   0,   0,   0,   0,   0.1, 0,   0.1, # 6
            0,   0,   0,   0.5, 0,   0,   0,   0,   0,   0,   # 7
            0,   0,   0,   0.1, 0,   0,   0,   0,   0,   0,   # 8
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   # 9
        ]


    def test_complex(self):
        builder = CompartmentBuilder(
            self.component_system, 'complex',
            self.volume_complex, self.matrix_complex,
            cycle_time=1000
        )
        builder.validate_flow_rates()

    def test_connections(self):
        builder = CompartmentBuilder(
            self.component_system, 'simple',
            self.volume_simple, self.matrix_simple,
            cycle_time=1000
        )
        flow_rates_expected =  {
            'compartment_0': {
                'total_in': np.array([1., 0., 0., 0.]),
                'total_out': np.array([1., 0., 0., 0.]),
                'origins': {
                    'compartment_1': np.array([0.1, 0. , 0. , 0. ]),
                    'compartment_2': np.array([0.2, 0. , 0. , 0. ]),
                    'compartment_3': np.array([0.3, 0. , 0. , 0. ]),
                    'compartment_4': np.array([0.4, 0. , 0. , 0. ])
                },
                'destinations': {
                    'compartment_1': np.array([0.1, 0. , 0. , 0. ]),
                    'compartment_2': np.array([0.2, 0. , 0. , 0. ]),
                    'compartment_3': np.array([0.3, 0. , 0. , 0. ]),
                    'compartment_4': np.array([0.4, 0. , 0. , 0. ])
                },
            },
            'compartment_1': {
                'total_in': np.array([0.1, 0. , 0. , 0. ]),
                'total_out': np.array([0.1, 0. , 0. , 0. ]),
                'origins': {'compartment_0': np.array([0.1, 0. , 0. , 0. ])},
                'destinations': {'compartment_0': np.array([0.1, 0. , 0. , 0. ])},
            },
            'compartment_2': {
                'total_in': np.array([0.2, 0. , 0. , 0. ]),
                'total_out': np.array([0.2, 0. , 0. , 0. ]),
                'origins': {'compartment_0': np.array([0.2, 0. , 0. , 0. ])},
                'destinations': {'compartment_0': np.array([0.2, 0. , 0. , 0. ])},
            },
            'compartment_3': {
                'total_in': np.array([0.3, 0. , 0. , 0. ]),
                'total_out': np.array([0.3, 0. , 0. , 0. ]),
                'origins': {'compartment_0': np.array([0.3, 0. , 0. , 0. ])},
                'destinations': {'compartment_0': np.array([0.3, 0. , 0. , 0. ])},
            },
            'compartment_4': {
                'total_in': np.array([0.4, 0. , 0. , 0. ]),
                'total_out': np.array([0.4, 0. , 0. , 0. ]),
                'origins': {'compartment_0': np.array([0.4, 0. , 0. , 0. ])},
                'destinations': {'compartment_0': np.array([0.4, 0. , 0. , 0. ])},
            }
        }
        flow_rates = builder.flow_sheet.get_flow_rates().to_dict()
        np.testing.assert_equal(flow_rates, flow_rates_expected)

    def test_validate_flow_rates(self):
        builder = CompartmentBuilder(
            self.component_system, 'complex',
            self.volume_complex, self.matrix_complex,
            cycle_time=1000
        )
        builder.flow_sheet.compartment_2.flow_rate = 0

        with self.assertRaises(CADETProcessError):
            builder.validate_flow_rates()

    def test_initial_conditions(self):
        # All to zero
        builder = CompartmentBuilder(
            self.component_system, 'complex',
            self.volume_complex, self.matrix_complex,
            cycle_time=1000
        )
        c_expected = [0, 0]
        c = builder.flow_sheet.compartment_2.c
        np.testing.assert_almost_equal(c, c_expected)

        # All same value
        builder = CompartmentBuilder(
            self.component_system, 'complex',
            self.volume_complex, self.matrix_complex,
            init_c=1,
            cycle_time=1000
        )
        c_expected = [1, 1]
        c = builder.flow_sheet.compartment_2.c
        np.testing.assert_almost_equal(c, c_expected)

        # Different per zone
        init_c = np.array([
            [0,0],
            [1,1],
            [2,2],
            [3,3],
            [4,4]
        ])

        builder = CompartmentBuilder(
            self.component_system, 'complex',
            self.volume_simple, self.matrix_simple,
            init_c=init_c,
            cycle_time=1000
        )
        c = builder.flow_sheet.compartment_2.c
        np.testing.assert_almost_equal(c, init_c[2].tolist())

    def test_tracer(self):
        """
        To do
        ------
            flow rate validation not working for flow rate filter
        """
        builder = CompartmentBuilder(
            self.component_system, 'complex',
            self.volume_complex, self.matrix_complex,
            init_c=1,
            cycle_time=1000
        )

        builder.add_tracer(4, [1, 1], 10, 0.1)

    def test_process(self):
        builder = CompartmentBuilder(
            self.component_system, 'complex',
            self.volume_complex, self.matrix_complex,
            init_c=1,
            cycle_time=1000
        )

        builder.add_tracer(4, [1, 1], 10, 0.1)

        from CADETProcess.simulation import Cadet
        process_simulator = Cadet()

        proc_results = process_simulator.run(builder.process)
        for comp in range(builder.n_compartments):
            proc_results.solution[f'compartment_{comp}'].outlet.plot()





if __name__ == '__main__':
    unittest.main()