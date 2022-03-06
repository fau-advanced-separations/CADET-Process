import unittest

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import (
    Source, Cstr, LumpedRateModelWithoutPores, Sink
)
from CADETProcess.processModel import FlowSheet

class Test_flow_sheet(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        # Batch
        self.component_system = ComponentSystem(2)

        flow_sheet = FlowSheet(self.component_system)

        feed = Source(self.component_system, name='feed')
        eluent = Source(self.component_system, name='eluent')
        column = LumpedRateModelWithoutPores(self.component_system, name='column')
        outlet = Sink(self.component_system, name='outlet')

        flow_sheet.add_unit(feed)
        flow_sheet.add_unit(eluent)
        flow_sheet.add_unit(column)
        flow_sheet.add_unit(outlet)

        flow_sheet.add_connection(feed, column)
        flow_sheet.add_connection(eluent, column)
        flow_sheet.add_connection(column, outlet)

        self.batch_flow_sheet = flow_sheet

        # SSR
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

        self.ssr_flow_sheet = flow_sheet

    def test_unit_names(self):
        unit_names = ['feed', 'eluent', 'cstr', 'column', 'outlet']

        self.assertEqual(list(self.ssr_flow_sheet.units_dict.keys()), unit_names)

    def test_sources(self):
        self.assertIn(self.ssr_flow_sheet.feed, self.ssr_flow_sheet.sources)
        self.assertIn(self.ssr_flow_sheet.eluent, self.ssr_flow_sheet.sources)
        self.assertIn(self.ssr_flow_sheet.cstr, self.ssr_flow_sheet.sources)

    def test_sinks(self):
        self.assertIn(self.ssr_flow_sheet.cstr, self.ssr_flow_sheet.sinks)
        self.assertIn(self.ssr_flow_sheet.outlet, self.ssr_flow_sheet.sinks)

    def test_connections(self):
        feed = self.ssr_flow_sheet['feed']
        eluent = self.ssr_flow_sheet['eluent']
        cstr = self.ssr_flow_sheet['cstr']
        column = self.ssr_flow_sheet['column']
        outlet = self.ssr_flow_sheet['outlet']
        expected_connections = {
            feed: {
                'origins': [],
                'destinations': [cstr],
            },
            eluent: {
                'origins': [],
                'destinations': [column],
            },
            cstr: {
                'origins': [feed, column],
                'destinations': [column],
            },
            column: {
                'origins': [cstr, eluent],
                'destinations': [cstr, outlet],
            },
            outlet: {
                'origins': [column],
                'destinations': [],
            },
        }

        self.assertDictEqual(self.ssr_flow_sheet.connections, expected_connections)
        
        self.assertTrue(self.ssr_flow_sheet.connection_exists(feed, cstr))
        self.assertTrue(self.ssr_flow_sheet.connection_exists(eluent, column))
        self.assertTrue(self.ssr_flow_sheet.connection_exists(column, outlet))
        
        self.assertFalse(self.ssr_flow_sheet.connection_exists(feed, eluent))
                        
    def test_name_decorator(self):
        feed = Source(self.component_system, name='feed')
        eluent = Source(self.component_system, name='eluent')
        cstr = Cstr(self.component_system, name='cstr')
        column = LumpedRateModelWithoutPores(self.component_system, name='column')
        outlet = Sink(self.component_system, name='outlet')
        
        flow_sheet = FlowSheet(self.component_system)
        
        flow_sheet.add_unit(feed)
        flow_sheet.add_unit(eluent)
        flow_sheet.add_unit(cstr)
        flow_sheet.add_unit(column)
        flow_sheet.add_unit(outlet)
        
        flow_sheet.add_connection('feed', 'cstr')
        flow_sheet.add_connection(cstr, column)
        flow_sheet.add_connection(eluent, 'column')
        flow_sheet.add_connection(column, cstr)
        flow_sheet.add_connection('column', outlet)
        
        expected_connections = {
            feed: {
                'origins': [],
                'destinations': [cstr],
            },
            eluent: {
                'origins': [],
                'destinations': [column],
            },
            cstr: {
                'origins': [feed, column],
                'destinations': [column],
            },
            column: {
                'origins': [cstr, eluent],
                'destinations': [cstr, outlet],
            },
            outlet: {
                'origins': [column],
                'destinations': [],
            },
        }

        self.assertDictEqual(flow_sheet.connections, expected_connections)
        
        # Connection already exists
        with self.assertRaises(CADETProcessError):
            flow_sheet.add_connection('column', 'outlet')
            
        # Origin not found
        with self.assertRaises(CADETProcessError):
            flow_sheet.add_connection('wrong_origin', cstr)
        
        # Destination not found
        with self.assertRaises(CADETProcessError):
            flow_sheet.add_connection('wrong_origin', cstr)
       
    def test_flow_rates(self):
        # Injection
        self.ssr_flow_sheet.feed.flow_rate = 0
        self.ssr_flow_sheet.eluent.flow_rate = 0
        self.ssr_flow_sheet.cstr.flow_rate = 1
        self.ssr_flow_sheet.set_output_state('column', 1)

        expected_flow_rates = {
            'feed': {
                'total_out': (0, 0, 0, 0),
                'destinations': {
                    'cstr': (0, 0, 0, 0),
                },
            },
            'eluent': {
                'total_out': (0, 0, 0, 0),
                'destinations': {
                    'column': (0, 0, 0, 0),
                },
            },
            'cstr': {
                'total_in': (0.0, 0, 0, 0),
                'total_out': (1.0, 0, 0, 0),
                'origins': {
                    'feed': (0, 0, 0, 0),
                    'column': (0, 0, 0, 0),
                },
                'destinations': {
                    'column': (1.0, 0, 0, 0),
                },
            },
            'column': {
                'total_in': (1.0, 0, 0, 0),
                'total_out': (1.0, 0, 0, 0),
                'origins': {
                    'cstr': (1.0, 0, 0, 0),
                    'eluent': (0, 0, 0, 0),
                },
                 'destinations': {
                    'cstr': (0, 0, 0, 0),
                    'outlet': (1.0, 0, 0, 0),
                },
            },
            'outlet': {
                'origins': {
                    'column': (1.0, 0, 0, 0),
                },
                'total_in': (1.0, 0, 0, 0),
                },
        }

        np.testing.assert_equal(
            self.ssr_flow_sheet.get_flow_rates(), expected_flow_rates
        )

        # Elution and Feed
        self.ssr_flow_sheet.feed.flow_rate = 1
        self.ssr_flow_sheet.eluent.flow_rate = 1
        self.ssr_flow_sheet.cstr.flow_rate = 0
        self.ssr_flow_sheet.set_output_state('column', 1)

        expected_flow_rates = {
            'feed': {
                'total_out': (1, 0, 0, 0),
                'destinations': {
                    'cstr': (1, 0, 0, 0),
                },
            },
            'eluent': {
                'total_out': (1, 0, 0, 0),
                'destinations': {
                    'column': (1, 0, 0, 0),
                },
            },
            'cstr': {
                'total_in': (1.0, 0, 0, 0),
                'total_out': (0.0, 0, 0, 0),
                'origins': {
                    'feed': (1, 0, 0, 0),
                    'column': (0, 0, 0, 0),
                },
                'destinations': {
                    'column': (0.0, 0, 0, 0),
                },
            },
            'column': {
                'total_in': (1.0, 0, 0, 0),
                'total_out': (1.0, 0, 0, 0),
                'origins': {
                    'cstr': (0, 0, 0, 0),
                    'eluent': (1, 0, 0, 0),
                },
                 'destinations': {
                    'cstr': (0, 0, 0, 0),
                    'outlet': (1.0, 0, 0, 0),
                },
            },
            'outlet': {
                'origins': {
                    'column': (1.0, 0, 0, 0),
                },
                'total_in': (1.0, 0, 0, 0),
                },
        }
        np.testing.assert_equal(
            self.ssr_flow_sheet.get_flow_rates(), expected_flow_rates
        )

        # Elution
        self.ssr_flow_sheet.feed.flow_rate = 0
        self.ssr_flow_sheet.eluent.flow_rate = 1
        self.ssr_flow_sheet.cstr.flow_rate = 0
        self.ssr_flow_sheet.set_output_state('column', 1)

        expected_flow_rates = {
            'feed': {
                'total_out': (0, 0, 0, 0),
                'destinations': {
                    'cstr': (0, 0, 0, 0),
                },
            },
            'eluent': {
                'total_out': (1, 0, 0, 0),
                'destinations': {
                    'column': (1, 0, 0, 0),
                },
            },
            'cstr': {
                'total_in': (0.0, 0, 0, 0),
                'total_out': (0.0, 0, 0, 0),
                'origins': {
                    'feed': (0, 0, 0, 0),
                    'column': (0, 0, 0, 0),
                },
                'destinations': {
                    'column': (0.0, 0, 0, 0),
                },
            },
            'column': {
                'total_in': (1.0, 0, 0, 0),
                'total_out': (1.0, 0, 0, 0),
                'origins': {
                    'cstr': (0, 0, 0, 0),
                    'eluent': (1, 0, 0, 0),
                },
                 'destinations': {
                    'cstr': (0, 0, 0, 0),
                    'outlet': (1.0, 0, 0, 0),
                },
            },
            'outlet': {
                'origins': {
                    'column': (1.0, 0, 0, 0),
                },
                'total_in': (1.0, 0, 0, 0),
                },
        }

        np.testing.assert_equal(
            self.ssr_flow_sheet.get_flow_rates(), expected_flow_rates
        )

        # Recycle
        self.ssr_flow_sheet.feed.flow_rate = 0
        self.ssr_flow_sheet.eluent.flow_rate = 1
        self.ssr_flow_sheet.cstr.flow_rate = 0
        self.ssr_flow_sheet.set_output_state('column', 0)


        expected_flow_rates = {
            'feed': {
                'total_out': (0, 0, 0, 0),
                'destinations': {
                    'cstr': (0, 0, 0, 0),
                },
            },
            'eluent': {
                'total_out': (1, 0, 0, 0),
                'destinations': {
                    'column': (1, 0, 0, 0),
                },
            },
            'cstr': {
                'total_in': (1.0, 0, 0, 0),
                'total_out': (0.0, 0, 0, 0),
                'origins': {
                    'feed': (0, 0, 0, 0),
                    'column': (1, 0, 0, 0),
                },
                'destinations': {
                    'column': (0.0, 0, 0, 0),
                },
            },
            'column': {
                'total_in': (1.0, 0, 0, 0),
                'total_out': (1.0, 0, 0, 0),
                'origins': {
                    'cstr': (0, 0, 0, 0),
                    'eluent': (1, 0, 0, 0),
                },
                 'destinations': {
                    'cstr': (1, 0, 0, 0),
                    'outlet': (0.0, 0, 0, 0),
                },
            },
            'outlet': {
                'origins': {
                    'column': (0, 0, 0, 0),
                },
                'total_in': (0, 0, 0, 0),
                },
        }

        np.testing.assert_equal(
            self.ssr_flow_sheet.get_flow_rates(), expected_flow_rates
        )
        
    def test_connectivity(self):
        self.batch_flow_sheet.remove_unit('outlet')
        
        with self.assertRaises(CADETProcessError):
            self.batch_flow_sheet.check_connections()
            
if __name__ == '__main__':
    unittest.main()