import unittest

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import (
    Inlet, Cstr, LumpedRateModelWithoutPores, Outlet
)
from CADETProcess.processModel import FlowSheet


def setup_single_cstr_flow_sheet(component_system=None):
    """
    Set up a simple `FlowSheet` with a single Continuous Stirred Tank Reactor (CSTR).

    Parameters
    ----------
    component_system : ComponentSystem, optional
        The component system for the CSTR.
        Defaults to a system with two components if None.

    Returns
    -------
    FlowSheet
        A flow sheet with a single CSTR unit.

    """
    if component_system is None:
        component_system = ComponentSystem(2)

    cstr = Cstr(component_system, 'cstr')

    flow_sheet = FlowSheet(component_system)
    flow_sheet.add_unit(cstr)

    return flow_sheet


def setup_batch_elution_flow_sheet(component_system=None):
    """
    Set up a `FlowSheet` for a typical batch elution process.

    Parameters
    ----------
    component_system : ComponentSystem, optional
        The component system for the batch elution process.
        Defaults to a system with two components if None.

    Returns
    -------
    FlowSheet
        A flow sheet configured for batch elution processes, including feed and eluent
        inlets, a column, and an outlet.

    """
    if component_system is None:
        component_system = ComponentSystem(2)

    flow_sheet = FlowSheet(component_system)

    feed = Inlet(component_system, name='feed')
    eluent = Inlet(component_system, name='eluent')
    column = LumpedRateModelWithoutPores(component_system, name='column')
    outlet = Outlet(component_system, name='outlet')

    flow_sheet.add_unit(feed)
    flow_sheet.add_unit(eluent)
    flow_sheet.add_unit(column)
    flow_sheet.add_unit(outlet)

    flow_sheet.add_connection(feed, column)
    flow_sheet.add_connection(eluent, column)
    flow_sheet.add_connection(column, outlet)

    return flow_sheet


def setup_ssr_flow_sheet(component_system=None):
    """
    Set up a `FlowSheet` for a steady state recycling (SSR) process.

    Parameters
    ----------
    component_system : ComponentSystem, optional
        The component system for the SSR process.
        Defaults to a system with two components if None.

    Returns
    -------
    FlowSheet
        A flow sheet configured for SSR elution, including feed and eluent inlets, a
        mixer tank, a column, and an outlet.

    """
    if component_system is None:
        component_system = ComponentSystem(2)

    flow_sheet = FlowSheet(component_system)

    feed = Inlet(component_system, name='feed')
    eluent = Inlet(component_system, name='eluent')
    cstr = Cstr(component_system, name='cstr')
    column = LumpedRateModelWithoutPores(component_system, name='column')
    outlet = Outlet(component_system, name='outlet')

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

    flow_sheet.add_eluent_inlet(eluent)
    flow_sheet.add_feed_inlet(feed)
    flow_sheet.add_product_outlet(outlet)

    return flow_sheet


class TestFlowSheet(unittest.TestCase):
    """Test general functionatlity of `FlowSheet` class."""

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.component_system = ComponentSystem(2)

        # Single Cstr
        self.single_cstr_flow_sheet = setup_single_cstr_flow_sheet(self.component_system)

        # Batch
        self.batch_flow_sheet = setup_batch_elution_flow_sheet(self.component_system)

        # SSR
        self.ssr_flow_sheet = setup_ssr_flow_sheet(self.component_system)

    def test_unit_names(self):
        unit_names_expected = ['feed', 'eluent', 'cstr', 'column', 'outlet']

        unit_names = list(self.ssr_flow_sheet.units_dict.keys())

        self.assertEqual(self.ssr_flow_sheet.unit_names, unit_names_expected)
        self.assertEqual(unit_names, unit_names_expected)

        # Connection already exists
        duplicate_unit_name = Inlet(self.component_system, 'feed')
        with self.assertRaises(CADETProcessError):
            self.batch_flow_sheet.add_unit(duplicate_unit_name)

    def test_inlets(self):
        self.assertIn(self.ssr_flow_sheet.feed, self.ssr_flow_sheet.inlets)
        self.assertIn(self.ssr_flow_sheet.eluent, self.ssr_flow_sheet.inlets)

    def test_outlets(self):
        self.assertIn(self.ssr_flow_sheet.outlet, self.ssr_flow_sheet.outlets)

    def test_cstrs(self):
        self.assertIn(self.ssr_flow_sheet.cstr, self.ssr_flow_sheet.cstrs)

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

        self.assertDictEqual(
            self.ssr_flow_sheet.connections, expected_connections
        )

        self.assertTrue(self.ssr_flow_sheet.connection_exists(feed, cstr))
        self.assertTrue(self.ssr_flow_sheet.connection_exists(eluent, column))
        self.assertTrue(self.ssr_flow_sheet.connection_exists(column, outlet))

        self.assertFalse(self.ssr_flow_sheet.connection_exists(feed, eluent))

    def test_name_decorator(self):
        feed = Inlet(self.component_system, name='feed')
        eluent = Inlet(self.component_system, name='eluent')
        cstr = Cstr(self.component_system, name='cstr')
        column = LumpedRateModelWithoutPores(
            self.component_system, name='column'
        )
        outlet = Outlet(self.component_system, name='outlet')

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
            flow_sheet.add_connection(cstr, 'wrong_destination')

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

        # Single Cstr
        expected_flow_rates = {
            'cstr': {
                'total_in': [0.0, 0.0, 0.0, 0.0],
                'total_out': [0.0, 0.0, 0.0, 0.0],
                'origins': {},
                'destinations': {}
            }
        }

        np.testing.assert_equal(
            self.single_cstr_flow_sheet.get_flow_rates(), expected_flow_rates
        )

    def test_check_connectivity(self):
        self.assertTrue(self.single_cstr_flow_sheet.check_connections())
        self.assertTrue(self.batch_flow_sheet.check_connections())
        self.assertTrue(self.ssr_flow_sheet.check_connections())

        self.batch_flow_sheet.remove_unit('outlet')

        with self.assertWarns(Warning):
            self.batch_flow_sheet.check_connections()


    def test_output_state(self):
        column = self.ssr_flow_sheet.column

        output_state_expected = [1, 0]
        output_state = self.ssr_flow_sheet.output_states[column]
        np.testing.assert_equal(output_state, output_state_expected)

        self.ssr_flow_sheet.set_output_state(column, [0,  1])
        output_state_expected = [0, 1]
        output_state = self.ssr_flow_sheet.output_states[column]
        np.testing.assert_equal(output_state, output_state_expected)

        self.ssr_flow_sheet.set_output_state(column, 0)
        output_state_expected = [1, 0]
        output_state = self.ssr_flow_sheet.output_states[column]
        np.testing.assert_equal(output_state, output_state_expected)

        self.ssr_flow_sheet.set_output_state(column, [0.5, 0.5])
        output_state_expected = [0.5, 0.5]
        output_state = self.ssr_flow_sheet.output_states[column]
        np.testing.assert_equal(output_state, output_state_expected)

        self.ssr_flow_sheet.set_output_state(
            column,
            {
                'cstr': 0.1,
                'outlet': 0.9,
            }
        )
        output_state_expected = [0.1, 0.9]
        output_state = self.ssr_flow_sheet.output_states[column]
        np.testing.assert_equal(output_state, output_state_expected)

        with self.assertRaises(TypeError):
            self.ssr_flow_sheet.set_output_state(column, 'unknown_type')

        with self.assertRaises(CADETProcessError):
            self.ssr_flow_sheet.set_output_state(column, [1, 1])

        with self.assertRaises(CADETProcessError):
            self.ssr_flow_sheet.set_output_state(
                column,
                {
                    'column': 0.1,
                    'outlet': 0.9,
                }
            )
    
    def test_add_connection_error(self):
        """
        Test for all raised exceptions of add_connections. 
        """
        inlet = self.ssr_flow_sheet['eluent']
        column = self.ssr_flow_sheet['column']
        outlet = self.ssr_flow_sheet['outlet']
        external_unit = Cstr(self.component_system, name='external_unit')

        # Inlet can't be a destination
        with self.assertRaises(CADETProcessError):
            self.ssr_flow_sheet.add_connection(column, inlet)

        # Outlet can't be an origin
        with self.assertRaises(CADETProcessError):
            self.ssr_flow_sheet.add_connection(outlet, column)

        # Destination not part of flow_sheet
        with self.assertRaises(CADETProcessError):
            self.ssr_flow_sheet.add_connection(inlet, external_unit)

        # Origin not part of flow_sheet
        with self.assertRaises(CADETProcessError):
            self.ssr_flow_sheet.add_connection(external_unit, outlet)

        # Connection already exists
        with self.assertRaises(CADETProcessError):
            self.ssr_flow_sheet.add_connection(inlet, column)


class TestCstrFlowRate(unittest.TestCase):
    """
    Test `Cstr` behaviour.

    Notes
    -----
    When the `flow_rate` parameter of the `Cstr` is not explicitly set, it is treated
    just like any other `UnitOperation`. I.e., q_in == q_out. In contrast, when a value
    is set, it has properties similar to an `Inlet`.
    """

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.component_system = ComponentSystem(2)

        flow_sheet = FlowSheet(self.component_system)

        inlet = Inlet(self.component_system, name='inlet')
        cstr = Cstr(self.component_system, name='cstr')
        outlet = Outlet(self.component_system, name='outlet')

        flow_sheet.add_unit(inlet)
        flow_sheet.add_unit(cstr)
        flow_sheet.add_unit(outlet)

        flow_sheet.add_connection(inlet, cstr)
        flow_sheet.add_connection(cstr, outlet)

        self.flow_sheet = flow_sheet

    def test_continuous_flow(self):
        self.flow_sheet.inlet.flow_rate = 1

        flow_rates = self.flow_sheet.get_flow_rates()

        cstr_in = flow_rates['cstr']['total_in']
        cstr_in_expected = [1., 0., 0., 0.]
        np.testing.assert_almost_equal(cstr_in, cstr_in_expected)

        cstr_out = flow_rates['cstr']['total_out']
        cstr_out_expected = [1., 0., 0., 0.]
        np.testing.assert_almost_equal(cstr_out, cstr_out_expected)

        # Test dynamic flow
        self.flow_sheet.inlet.flow_rate = [1, 1]

        flow_rates = self.flow_sheet.get_flow_rates()

        cstr_in = flow_rates['cstr']['total_in']
        cstr_in_expected = [1., 1., 0., 0.]
        np.testing.assert_almost_equal(cstr_in, cstr_in_expected)

        cstr_out = flow_rates['cstr']['total_out']
        cstr_out_expected = [1., 1., 0., 0.]
        np.testing.assert_almost_equal(cstr_out, cstr_out_expected)

    def test_no_flow(self):
        self.flow_sheet.inlet.flow_rate = 0

        flow_rates = self.flow_sheet.get_flow_rates()

        cstr_in = flow_rates['cstr']['total_in']
        cstr_in_expected = [0., 0., 0., 0.]
        np.testing.assert_almost_equal(cstr_in, cstr_in_expected)

        cstr_out = flow_rates['cstr']['total_out']
        cstr_out_expected = [0., 0., 0., 0.]
        np.testing.assert_almost_equal(cstr_out, cstr_out_expected)

        self.flow_sheet.inlet.flow_rate = 0
        self.flow_sheet.cstr.flow_rate = 0

        flow_rates = self.flow_sheet.get_flow_rates()

        cstr_in = flow_rates['cstr']['total_in']
        cstr_in_expected = [0., 0., 0., 0.]
        np.testing.assert_almost_equal(cstr_in, cstr_in_expected)

        cstr_out = flow_rates['cstr']['total_out']
        cstr_out_expected = [0., 0., 0., 0.]
        np.testing.assert_almost_equal(cstr_out, cstr_out_expected)

    def test_holdup(self):
        self.flow_sheet.inlet.flow_rate = 1
        self.flow_sheet.cstr.flow_rate = 0

        flow_rates = self.flow_sheet.get_flow_rates()

        cstr_in = flow_rates['cstr']['total_in']
        cstr_in_expected = [1., 0., 0., 0.]
        np.testing.assert_almost_equal(cstr_in, cstr_in_expected)

        cstr_out = flow_rates['cstr']['total_out']
        cstr_out_expected = [0., 0., 0., 0.]
        np.testing.assert_almost_equal(cstr_out, cstr_out_expected)

    def test_state_update(self):
        self.flow_sheet.inlet.flow_rate = [1, 1]

        state = {
            'flow_sheet.cstr.flow_rate': np.array([2, 2, 0, 0], ndmin=2)
        }

        flow_rates = self.flow_sheet.get_flow_rates(state)

        cstr_in = flow_rates['cstr']['total_in']
        cstr_in_expected = [1., 1., 0., 0.]
        np.testing.assert_almost_equal(cstr_in, cstr_in_expected)

        cstr_out = flow_rates['cstr']['total_out']
        cstr_out_expected = [2., 2., 0., 0.]
        np.testing.assert_almost_equal(cstr_out, cstr_out_expected)


class TestFlowRateMatrix(unittest.TestCase):
    """Test calculation of flow rates with another simple testcase by @daklauss"""

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.component_system = ComponentSystem(1)

        flow_sheet = FlowSheet(self.component_system)

        inlet = Inlet(self.component_system, name='inlet')
        cstr1 = Cstr(self.component_system, name='cstr1')
        cstr2 = Cstr(self.component_system, name='cstr2')
        outlet1 = Outlet(self.component_system, name='outlet1')
        outlet2 = Outlet(self.component_system, name='outlet2')

        flow_sheet.add_unit(inlet)
        flow_sheet.add_unit(cstr1)
        flow_sheet.add_unit(cstr2)

        flow_sheet.add_unit(outlet1)
        flow_sheet.add_unit(outlet2)

        flow_sheet.add_connection(inlet, cstr1)
        flow_sheet.add_connection(inlet, cstr2)

        flow_sheet.add_connection(cstr1, outlet1)
        flow_sheet.add_connection(cstr2, outlet2)

        flow_sheet.add_connection(cstr2, cstr1)

        flow_sheet.set_output_state(inlet, [0.3, 0.7])
        flow_sheet.set_output_state(cstr2, [0.5, 0.5])

        self.flow_sheet = flow_sheet

    def test_matrix_example(self):
        self.flow_sheet.inlet.flow_rate = [1]

        expected_flow_rates = {
            'inlet': {
                'total_out': (1, 0, 0, 0),
                'destinations': {
                    'cstr1': (0.3, 0, 0, 0),
                    'cstr2': (0.7, 0, 0, 0),
                },
            },
            'cstr1': {
                'total_in': (0.65, 0, 0, 0),
                'total_out': (0.65, 0, 0, 0),
                'origins': {
                    'inlet': (0.3, 0, 0, 0),
                    'cstr2': (0.35, 0, 0, 0),
                },
                'destinations': {
                    'outlet1': (0.65, 0, 0, 0),
                },
            },
            'cstr2': {
                'total_in': (0.7, 0, 0, 0),
                'total_out': (0.7, 0, 0, 0),
                'origins': {
                    'inlet': (0.7, 0, 0, 0),
                },
                'destinations': {
                    'cstr1': (0.35, 0, 0, 0),
                    'outlet2': (0.35, 0, 0, 0),
                },
            },
            'outlet1': {
                'origins': {
                    'cstr1': (0.65, 0, 0, 0),
                },
                'total_in': (0.65, 0, 0, 0),
            },

            'outlet2': {
                'origins': {
                    'cstr2': (0.35, 0, 0, 0),
                },
                'total_in': (0.35, 0, 0, 0),
                }
        }

        calc_flow_rate = self.flow_sheet.get_flow_rates()

        def assert_almost_equal_dict(
                dict_actual, dict_expected, decimal=7, verbose=True):
            """Helper function to assert nested dicts are (almost) equal.

            Because of floating point calculations, it is necessary to use
            `np.assert_almost_equal` to check the flow rates. However, this does not
            work well with nested dicts which is why this helper function was written.

            Parameters
            ----------
            dict_actual : dict
                The object to check.
            dict_expected : dict
                The expected object.
            decimal : int, optional
                Desired precision, default is 7.
            err_msg : str, optional
                The error message to be printed in case of failure.
            verbose : bool, optional
                If True, the conflicting values are appended to the error message.

            """
            for key in dict_actual:
                if isinstance(dict_actual[key], dict):
                    assert_almost_equal_dict(dict_actual[key], dict_expected[key])
                else:
                    np.testing.assert_almost_equal(
                        dict_actual[key], dict_expected[key],
                        decimal=decimal,
                        err_msg=f'Dicts are not equal in key {key}.',
                        verbose=verbose
                    )

        assert_almost_equal_dict(calc_flow_rate, expected_flow_rates)


class TestFlowRateSelfMatrix(unittest.TestCase):
    """Test special case where one unit is connected to itself."""
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.component_system = ComponentSystem(1)

        flow_sheet = FlowSheet(self.component_system)

        inlet = Inlet(self.component_system, name='inlet')
        cstr = Cstr(self.component_system, name='cstr')
        outlet = Outlet(self.component_system, name='outlet')

        flow_sheet.add_unit(inlet)
        flow_sheet.add_unit(cstr)
        flow_sheet.add_unit(outlet)

        flow_sheet.add_connection(inlet, cstr)
        flow_sheet.add_connection(cstr, outlet)

        flow_sheet.add_connection(cstr, cstr)

        flow_sheet.set_output_state(cstr, [0.5, 0.5])

        self.flow_sheet = flow_sheet

    def test_matrix_self_example(self):
        self.flow_sheet.inlet.flow_rate = [1]

        expected_flow_rates = {
            'inlet': {
                'total_out': (1, 0, 0, 0),
                'destinations': {
                    'cstr': (1, 0, 0, 0)
                },
            },
            'cstr': {
                'total_in': (2, 0, 0, 0),
                'total_out': (2, 0, 0, 0),
                'origins': {
                    'inlet': (1, 0, 0, 0),
                    'cstr': (1, 0, 0, 0),
                },
                'destinations': {
                    'outlet': (1, 0, 0, 0),
                    'cstr': (1, 0, 0, 0)
                },
            },
            'outlet': {
                'total_in': (1, 0, 0, 0),
                'origins': {
                    'cstr': (1, 0, 0, 0)
                }
            }
        }
        calc_flow_rate = self.flow_sheet.get_flow_rates()
        np.testing.assert_equal(calc_flow_rate, expected_flow_rates)


class TestSingularFlowMatrix(unittest.TestCase):
    """Test cases with disconnected flow circles

    Notes
    -----
    `FlowSheet`: `Inlet` connected to an `Outlet` and two `Cstr`s connected to each
    other but not with the other units.

    """

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):

        self.component_system = ComponentSystem(1)

        flow_sheet = FlowSheet(self.component_system)

        inlet = Inlet(self.component_system, name='inlet')
        cstr1 = Cstr(self.component_system, name='cstr1')
        cstr2 = Cstr(self.component_system, name='cstr2')
        outlet = Outlet(self.component_system, name='outlet')

        flow_sheet.add_unit(inlet)
        flow_sheet.add_unit(cstr1)
        flow_sheet.add_unit(cstr2)
        flow_sheet.add_unit(outlet)

        flow_sheet.add_connection(inlet, outlet)

        flow_sheet.add_connection(cstr1, cstr2)
        flow_sheet.add_connection(cstr2, cstr1)

        self.flow_sheet = flow_sheet

    def test_expelled_cicuit(self):
        # Throw error even without flow, because system is singular
        flow_sheet = self.flow_sheet

        with self.assertRaises(CADETProcessError):
            flow_sheet.get_flow_rates()

        flow_sheet.inlet.flow_rate = [1]

        with self.assertRaises(CADETProcessError):
            flow_sheet.get_flow_rates()

    def test_expelled_circuit_with_flow(self):
        # Solvable because both disconnected circles have their own flow rates
        expected_flow_rates = {
            'inlet': {
                'total_out': (1, 0, 0, 0),
                'destinations': {
                    'outlet': (1, 0, 0, 0)
                },
            },
            'cstr1': {
                'total_in': (1, 0, 0, 0),
                'total_out': (1, 0, 0, 0),
                'origins': {
                    'cstr2': (1, 0, 0, 0),
                },
                'destinations': {
                    'cstr2': (1, 0, 0, 0)
                },
            },
            'cstr2': {
                'total_in': (1, 0, 0, 0),
                'total_out': (1, 0, 0, 0),
                'origins': {
                    'cstr1': (1, 0, 0, 0),
                },
                'destinations': {
                    'cstr1': (1, 0, 0, 0)
                },
            },
            'outlet': {
                'total_in': (1, 0, 0, 0),
                'origins': {
                    'inlet': (1, 0, 0, 0)
                }
            }
        }

        flow_sheet = self.flow_sheet
        flow_sheet.inlet.flow_rate = [1]
        flow_sheet.cstr1.flow_rate = [1]

        np.testing.assert_equal(flow_sheet.get_flow_rates(), expected_flow_rates)


if __name__ == '__main__':
    unittest.main()
