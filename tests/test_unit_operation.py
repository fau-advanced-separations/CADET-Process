"""
Todo
----
Add tests for
- section dependent parameters, polynomial parameters
"""
import unittest

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import (
    Inlet, Cstr,
    TubularReactor, LumpedRateModelWithPores, LumpedRateModelWithoutPores, MCT
)

length = 0.6
diameter = 0.024

cross_section_area = np.pi/4 * diameter**2
volume_liquid = cross_section_area * length
volume = cross_section_area * length

bed_porosity = 0.3
particle_porosity = 0.6
total_porosity = bed_porosity + (1 - bed_porosity) * particle_porosity
const_solid_volume = volume * (1 - total_porosity)
init_liquid_volume = volume * total_porosity

axial_dispersion = 4.7e-7

channel_cross_section_areas = [0.1,0.1,0.1]
exchange_matrix = np.array([
                             [[0.0],[0.01],[0.0]],
                             [[0.02],[0.0],[0.03]],
                             [[0.0],[0.0],[0.0]]
                             ])
flow_direction = 1


class Test_Unit_Operation(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.component_system = ComponentSystem(2)

    def create_source(self):
        source = Inlet(self.component_system, name='test')

        return source

    def create_cstr(self):
        cstr = Cstr(self.component_system, name='test')

        cstr.const_solid_volume = const_solid_volume
        cstr.init_liquid_volume = init_liquid_volume

        cstr.flow_rate = 1

        return cstr

    def create_tubular_reactor(self):
        tube = TubularReactor(self.component_system, name='test')

        tube.length = length
        tube.diameter = diameter
        tube.axial_dispersion = axial_dispersion

        return tube

    def create_MCT(self, components):
        mct = MCT(ComponentSystem(components), nchannel=3, name='test')

        mct.length = length
        mct.channel_cross_section_areas = channel_cross_section_areas
        mct.axial_dispersion = 0
        mct.flow_direction = flow_direction

        return mct

    def create_lrmwop(self):
        lrmwop = LumpedRateModelWithoutPores(
            self.component_system, name='test'
        )

        lrmwop.length = length
        lrmwop.diameter = diameter
        lrmwop.axial_dispersion = axial_dispersion
        lrmwop.total_porosity = total_porosity

        return lrmwop

    def create_lrmwp(self):
        lrmwp = LumpedRateModelWithPores(
            self.component_system, name='test'
        )

        lrmwp.length = length
        lrmwp.diameter = diameter
        lrmwp.axial_dispersion = axial_dispersion
        lrmwp.bed_porosity = bed_porosity
        lrmwp.particle_porosity = particle_porosity

        return lrmwp

    def test_geometry(self):
        cstr = self.create_cstr()
        lrmwop = self.create_lrmwop()
        lrmwp = self.create_lrmwp()

        self.assertEqual(lrmwop.cross_section_area, cross_section_area)
        self.assertEqual(lrmwp.cross_section_area, cross_section_area)

        self.assertEqual(lrmwop.total_porosity, total_porosity)
        self.assertEqual(lrmwp.total_porosity, total_porosity)

        self.assertEqual(lrmwop.volume, volume)
        self.assertEqual(lrmwp.volume, volume)

        volume_interstitial = total_porosity * volume
        self.assertAlmostEqual(lrmwop.volume_interstitial, volume_interstitial)
        volume_interstitial = bed_porosity * volume
        self.assertAlmostEqual(lrmwp.volume_interstitial, volume_interstitial)

        volume_liquid = total_porosity * volume
        self.assertAlmostEqual(cstr.volume_liquid, volume_liquid)
        self.assertAlmostEqual(lrmwop.volume_liquid, volume_liquid)
        self.assertAlmostEqual(lrmwp.volume_liquid, volume_liquid)

        volume_solid = (1 - total_porosity) * volume
        self.assertAlmostEqual(cstr.volume_solid, volume_solid)
        self.assertAlmostEqual(lrmwop.volume_solid, volume_solid)
        self.assertAlmostEqual(lrmwp.volume_solid, volume_solid)

        lrmwop.cross_section_area = cross_section_area/2
        self.assertAlmostEqual(lrmwop.diameter, diameter/(2**0.5))

    def test_convection_dispersion(self):
        tube = self.create_tubular_reactor()
        lrmwp = self.create_lrmwp()

        flow_rate = 0
        tube.length = 1
        tube.cross_section_area = 1
        tube.axial_dispersion = 0

        with self.assertRaises(ZeroDivisionError):
            tube.calculate_interstitial_velocity(flow_rate)
        with self.assertRaises(ZeroDivisionError):
            tube.calculate_superficial_velocity(flow_rate)
        with self.assertRaises(ZeroDivisionError):
            tube.NTP(flow_rate)

        flow_rate = 2
        tube.axial_dispersion = 3
        self.assertAlmostEqual(tube.calculate_interstitial_velocity(flow_rate), 2)
        self.assertAlmostEqual(tube.calculate_interstitial_rt(flow_rate), 0.5)
        self.assertAlmostEqual(tube.calculate_superficial_velocity(flow_rate), 2)
        self.assertAlmostEqual(tube.calculate_superficial_rt(flow_rate), 0.5)
        self.assertAlmostEqual(tube.NTP(flow_rate), 1/3)

        tube.set_axial_dispersion_from_NTP(1/3, 2)
        self.assertAlmostEqual(tube.axial_dispersion, 3)

        flow_rate = 2
        lrmwp.length = 1
        lrmwp.bed_porosity = 0.5
        lrmwp.cross_section_area = 1
        self.assertAlmostEqual(lrmwp.calculate_interstitial_velocity(flow_rate), 4)
        self.assertAlmostEqual(lrmwp.calculate_interstitial_rt(flow_rate), 0.25)
        self.assertAlmostEqual(lrmwp.calculate_superficial_velocity(flow_rate), 2)
        self.assertAlmostEqual(lrmwp.calculate_superficial_rt(flow_rate), 0.5)

    def test_poly_properties(self):
        source = self.create_source()

        ref = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
        source.c = 1
        np.testing.assert_equal(source.c, ref)
        source.c = [1, 1]
        np.testing.assert_equal(source.c, ref)

        ref = np.array([[1, 0, 0, 0], [2, 0, 0, 0]])
        source.c = [1, 2]
        np.testing.assert_equal(source.c, ref)
        source.c = [[1, 0], [2, 0]]
        np.testing.assert_equal(source.c, ref)

        ref = np.array([[1, 2, 0, 0], [3, 4, 0, 0]])
        source.c = [[1, 2], [3, 4]]
        np.testing.assert_equal(source.c, ref)
        source.c = ref
        np.testing.assert_equal(source.c, ref)

        cstr = self.create_cstr()

        ref = np.array([1, 0, 0, 0])
        cstr.flow_rate = 1
        np.testing.assert_equal(cstr.flow_rate, ref)
        cstr.flow_rate = [1, 0]
        np.testing.assert_equal(cstr.flow_rate, ref)

        ref = np.array([1, 1, 0, 0])
        cstr.flow_rate = [1, 1]
        np.testing.assert_equal(cstr.flow_rate, ref)
        cstr.flow_rate = ref
        np.testing.assert_equal(cstr.flow_rate, ref)

    def test_parameters(self):
        """
        Notes
        -----
            Currently, only getting parameters is tested. Should also test if
            setting works. For this, adsorption parameters should be provided.
        """
        cstr = self.create_cstr()
        parameters_expected = {
                'flow_rate': np.array([1, 0, 0, 0]),
                'init_liquid_volume': init_liquid_volume,
                'flow_rate_filter': 0,
                'c': [0, 0],
                'q': [],
                'const_solid_volume': const_solid_volume,
        }

        np.testing.assert_equal(parameters_expected, cstr.parameters)

        sec_dep_parameters_expected = {
                'flow_rate': np.array([1, 0, 0, 0]),
                'flow_rate_filter': 0,
        }
        np.testing.assert_equal(
            sec_dep_parameters_expected, cstr.section_dependent_parameters
        )

        poly_parameters = {
            'flow_rate': np.array([1, 0, 0, 0]),
        }
        np.testing.assert_equal(
            poly_parameters, cstr.polynomial_parameters
        )

        self.assertEqual(cstr.required_parameters, ['init_liquid_volume'])


    def test_MCT(self):
        """
        Notes
        -----
            Tests Parameters, Volumes and Attributes depending on nchannel. Should be later integrated into general testing workflow.
        """
        total_porosity = 1

        mct = self.create_MCT(1)

        mct.exchange_matrix = exchange_matrix

        parameters_expected = {
        'c': np.array([[0., 0., 0.]]),
        'axial_dispersion' : 0,
        'channel_cross_section_areas' : channel_cross_section_areas,
        'length' : length,
        'exchange_matrix': exchange_matrix,
        'flow_direction' : 1,
        'nchannel' : 3
        }
        np.testing.assert_equal(parameters_expected, {key: value for key, value in mct.parameters.items() if key != 'discretization'})

        volume = length*sum(channel_cross_section_areas)
        volume_liquid = volume*total_porosity
        volume_solid = (total_porosity-1)*volume

        self.assertAlmostEqual(mct.volume_liquid, volume_liquid)
        self.assertAlmostEqual(mct.volume_solid, volume_solid)

        with self.assertRaises(ValueError):
            mct.exchange_matrix =  np.array([[
                             [0.0, 0.01, 0.0],
                             [0.02, 0.0, 0.03],
                             [0.0, 0.0, 0.0]
                             ]])

        mct.nchannel = 2
        with self.assertRaises(ValueError):
            mct.exchange_matrix
            mct.channel_cross_section_areas

        self.assertTrue(mct.nchannel*mct.component_system.n_comp == mct.c.size)

        mct2 = self.create_MCT(2)

        with self.assertRaises(ValueError):
            mct2.exchange_matrix =  np.array([[
                            [0.0, 0.01, 0.0],
                            [0.02, 0.0, 0.03],
                            [0.0, 0.0, 0.0]
                            ],

                            [
                            [0.0, 0.01, 0.0],
                            [0.02, 0.0, 0.03],
                            [0.0, 0.0, 0.0]
                            ]])


if __name__ == '__main__':
    unittest.main()
