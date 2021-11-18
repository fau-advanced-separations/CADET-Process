import unittest

import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import (
    Source, Cstr, LumpedRateModelWithoutPores, Sink
) 
from CADETProcess.processModel import FlowSheet

class Test_flow_sheet(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
    
    def setUp(self):
        self.component_system = ComponentSystem()
        self.component_system.add_component('A')
        self.component_system.add_component('B')

    def create_ssr_flow_sheet(self):
        flow_sheet = FlowSheet(self.component_system)

        feed = Source(self.component_system, name='feed')
        eluent = Source(self.component_system, name='eluent')
        cstr = Cstr(self.component_system, name='cstr')
        column = LumpedRateModelWithoutPores(self.component_system, name='column')
        outlet = Sink(self.component_system, name='outlet')

        flow_sheet.add_unit(feed)
        flow_sheet.add_unit(eluent)
        flow_sheet.add_unit(cstr)
        flow_sheet.add_unit(column)
        flow_sheet.add_unit(outlet)

        flow_sheet.add_connection(feed, cstr)
        flow_sheet.add_connection(cstr, column)
        flow_sheet.add_connection(eluent, column)
        flow_sheet.add_connection(column, cstr)
        flow_sheet.add_connection(column, outlet)

        flow_sheet.add_eluent_source(eluent)
        flow_sheet.add_feed_source(feed)
        flow_sheet.add_chromatogram_sink(outlet)

        return flow_sheet

    def test_unit_names(self):
        flow_sheet = self.create_ssr_flow_sheet()
        unit_names = ['feed', 'eluent', 'cstr', 'column', 'outlet']

        self.assertEqual(list(flow_sheet.units_dict.keys()), unit_names)


    def test_sources(self):
        flow_sheet = self.create_ssr_flow_sheet()

        self.assertIn(flow_sheet.feed, flow_sheet.sources)
        self.assertIn(flow_sheet.eluent, flow_sheet.sources)
        self.assertIn(flow_sheet.cstr, flow_sheet.sources)


    def test_sinks(self):
        flow_sheet = self.create_ssr_flow_sheet()

        self.assertIn(flow_sheet.cstr, flow_sheet.sinks)
        self.assertIn(flow_sheet.outlet, flow_sheet.sinks)


    def test_connections(self):
        flow_sheet = self.create_ssr_flow_sheet()

        expected_connections = {
                'feed': ['cstr'],
                'eluent': ['column'],
                'cstr': ['column'],
                'column': ['cstr', 'outlet'],
                'outlet': []}

        # self.assertDictEqual(flow_sheet.connections_out, expected_connections)


    def test_ssr_flow_rates(self):
        flow_sheet = self.create_ssr_flow_sheet()

        # Injection
        flow_sheet.feed.flow_rate = 0
        flow_sheet.eluent.flow_rate = 0
        flow_sheet.cstr.flow_rate = 1
        flow_sheet.set_output_state('column', 1)

        expected_flow_rates = {
            'feed': {
                'total': (0, 0, 0, 0),
                'destinations': {
                    'cstr': (0, 0, 0, 0),
                },
            },
            'eluent': {
                'total': (0, 0, 0, 0),
                'destinations': {
                    'column': (0, 0, 0, 0),
                },
            },
            'cstr': {
                'total': (1.0, 0, 0, 0),
                'destinations': {
                    'column': (1.0, 0, 0, 0),
                },
            },            
            'column': {
                'total': (1.0, 0, 0, 0),
                'destinations': {
                    'cstr': (0, 0, 0, 0),
                    'outlet': (1.0, 0, 0, 0),
                },
            },
            'outlet': {
                'total': (1.0, 0, 0, 0),
                },
        }        

        np.testing.assert_equal(flow_sheet.get_flow_rates(), expected_flow_rates)

        # Elution and Feed
        flow_sheet.feed.flow_rate = 1
        flow_sheet.eluent.flow_rate = 1
        flow_sheet.cstr.flow_rate = 0
        flow_sheet.set_output_state('column', 1)

        expected_flow_rates = {
            'feed': {
                'total': (1.0, 0, 0, 0),
                'destinations': {
                    'cstr': (1.0, 0, 0, 0),
                },
            },
            'eluent': {
                'total': (1.0, 0, 0, 0),
                'destinations': {
                    'column': (1.0, 0, 0, 0),
                },
            },
            'cstr': {
                'total': (0, 0, 0, 0),
                'destinations': {
                    'column': (0, 0, 0, 0),
                },
            },            
            'column': {
                'total': (1.0, 0, 0, 0),
                'destinations': {
                    'cstr': (0, 0, 0, 0),
                    'outlet': (1.0, 0, 0, 0),
                },
            },
            'outlet': {
                'total': (1.0, 0, 0, 0),
                },
        }        
        
        np.testing.assert_equal(flow_sheet.get_flow_rates(), expected_flow_rates)

        # Elution
        flow_sheet.feed.flow_rate = 0
        flow_sheet.eluent.flow_rate = 1
        flow_sheet.cstr.flow_rate = 0
        flow_sheet.set_output_state('column', 1)

        expected_flow_rates = {
            'feed': {
                'total': (0, 0, 0, 0),
                'destinations': {
                    'cstr': (0, 0, 0, 0),
                },
            },
            'eluent': {
                'total': (1.0, 0, 0, 0),
                'destinations': {
                    'column': (1.0, 0, 0, 0),
                },
            },
            'cstr': {
                'total': (0, 0, 0, 0),
                'destinations': {
                    'column': (0, 0, 0, 0),
                },
            },            
            'column': {
                'total': (1.0, 0, 0, 0),
                'destinations': {
                    'cstr': (0, 0, 0, 0),
                    'outlet': (1.0, 0, 0, 0),
                },
            },
            'outlet': {
                'total': (1.0, 0, 0, 0),
                },
        }

        np.testing.assert_equal(flow_sheet.get_flow_rates(), expected_flow_rates)

        # Recycle
        flow_sheet.feed.flow_rate = 0
        flow_sheet.eluent.flow_rate = 1
        flow_sheet.cstr.flow_rate = 0
        flow_sheet.set_output_state('column', 0)

        expected_flow_rates = {
            'feed': {
                'total': (0, 0, 0, 0),
                'destinations': {
                    'cstr': (0, 0, 0, 0),
                },
            },
            'eluent': {
                'total': (1.0, 0, 0, 0),
                'destinations': {
                    'column': (1.0, 0, 0, 0),
                },
            },
            'cstr': {
                'total': (0, 0, 0, 0),
                'destinations': {
                    'column': (0, 0, 0, 0),
                },
            },            
            'column': {
                'total': (1.0, 0, 0, 0),
                'destinations': {
                    'cstr': (1.0, 0, 0, 0),
                    'outlet': (0, 0, 0, 0),
                },
            },
            'outlet': {
                'total': (0, 0, 0, 0),
                },
        }

        np.testing.assert_equal(flow_sheet.get_flow_rates(), expected_flow_rates)

    def create_clr_flow_sheet(self):
        flow_sheet = FlowSheet(n_comp=2, name='test')

        feed = Source(n_comp=2, name='feed')
        column = LumpedRateModelWithoutPores(n_comp=2, name='column')
        outlet = Sink(n_comp=2, name='outlet')

        flow_sheet.add_unit(feed)
        flow_sheet.add_unit(column)
        flow_sheet.add_unit(outlet)

        flow_sheet.add_connection(feed, column)
        flow_sheet.add_connection(column, outlet)
        flow_sheet.add_connection(column, column)

        return flow_sheet

    def test_clr_flow_rates(self):
        """Currently not working in CADET-Process; Must be implemented with 
        CSTR/Recycle Pump
        """
        # flow_sheet = self.create_clr_flow_sheet()

        # # Injection
        # flow_sheet.feed.flow_rate = 1
        # flow_sheet.set_output_state('column', 0)

        # expected_flow_rates = {
        #     'feed': {
        #         'total': (1.0, 0, 0, 0),
        #         'destinations': {
        #             'column': (1.0, 0, 0, 0),
        #         },
        #     },
        #     'column': {
        #         'total': (1.0, 0, 0, 0),
        #         'destinations': {
        #             'outlet': (1.0, 0, 0, 0),
        #             'column': (0, 0, 0, 0),                    
        #         },
        #     },
        #     'outlet': {
        #         'total': (1.0, 0, 0, 0),
        #         },
        # }

        # np.testing.assert_equal(flow_sheet.get_flow_rates(), expected_flow_rates)

        # # Recycle
        # flow_sheet.feed.flow_rate = 0
        # flow_sheet.set_output_state('column', [0, 1])

        # expected_flow_rates = {
        #     'feed': {
        #         'total': (0, 0, 0, 0),
        #         'destinations': {
        #             'column': (0, 0, 0, 0),
        #         },
        #     },
        #     'column': {
        #         'total': (1.0, 0, 0, 0),
        #         'destinations': {
        #             'outlet': (0, 0, 0, 0),
        #             'column': (1.0, 0, 0, 0),                    
        #         },
        #     },
        #     'outlet': {
        #         'total': (0, 0, 0, 0),
        #         },
        # }

        # np.testing.assert_equal(flow_sheet.get_flow_rates(), expected_flow_rates)

        # # Elution
        # flow_sheet.feed.flow_rate = 1
        # flow_sheet.set_output_state('column', 0)

        # expected_flow_rates = {
        #     'feed': {
        #         'total': (1.0, 0, 0, 0),
        #         'destinations': {
        #             'column': (1.0, 0, 0, 0),
        #         },
        #     },
        #     'column': {
        #         'total': (1.0, 0, 0, 0),
        #         'destinations': {
        #             'outlet': (1.0, 0, 0, 0),
        #             'column': (0, 0, 0, 0),                    
        #         },
        #     },
        #     'outlet': {
        #         'total': (1.0, 0, 0, 0),
        #         },
        # } 

        # np.testing.assert_equal(flow_sheet.get_flow_rates(), expected_flow_rates)


if __name__ == '__main__':
    unittest.main()