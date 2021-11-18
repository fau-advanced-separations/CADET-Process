"""
Todo
----
Add tests for
- section dependent parameters, polynomial parameters
"""
import unittest

import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import (
    Source, Cstr,
    TubularReactor, LumpedRateModelWithPores, LumpedRateModelWithoutPores
)

length = 0.6
diameter = 0.024

cross_section_area = np.pi/4 * diameter**2
volume_liquid = cross_section_area * length
volume = cross_section_area * length

bed_porosity = 0.3
particle_porosity = 0.6
total_porosity = bed_porosity + (1 - bed_porosity) * particle_porosity

axial_dispersion = 4.7e-7

class Test_Unit_Operation(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        
    def setUp(self):
        self.component_system = ComponentSystem()
        self.component_system.add_component('A')
        self.component_system.add_component('B')

    def create_source(self):
        source = Source(self.component_system, name='test')

        return source

    def create_cstr(self):
        cstr = Cstr(self.component_system, name='test')

        cstr.porosity = total_porosity
        cstr.V = volume

        cstr.flow_rate = 1

        return cstr

    def create_tubular_reactor(self):
        tube = TubularReactor(self.component_system, name='test')

        tube.length = length
        tube.diameter = diameter
        tube.axial_dispersion = axial_dispersion

        return tube

    def create_lrmwop(self):
        lrmwop = LumpedRateModelWithoutPores(self.component_system, name='test')

        lrmwop.length = length
        lrmwop.diameter = diameter
        lrmwop.axial_dispersion = axial_dispersion
        lrmwop.total_porosity = total_porosity

        return lrmwop

    def create_lrmwp(self):
        lrmwp = LumpedRateModelWithPores(self.component_system, name='test')

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
            tube.u0(flow_rate)
            tube.NTP(flow_rate)

        flow_rate = 2
        tube.axial_dispersion = 3
        self.assertAlmostEqual(tube.u0(flow_rate), 2)
        self.assertAlmostEqual(tube.t0(flow_rate), 0.5)
        self.assertAlmostEqual(tube.NTP(flow_rate), 1/3)

        tube.set_axial_dispersion_from_NTP(1/3, 2)
        self.assertAlmostEqual(tube.axial_dispersion, 3)

        flow_rate = 2
        lrmwp.length = 1
        lrmwp.bed_porosity = 0.5
        lrmwp.cross_section_area = 1
        self.assertAlmostEqual(lrmwp.u0(flow_rate), 4)
        self.assertAlmostEqual(lrmwp.t0(flow_rate), 0.25)


    def test_poly_properties(self):
        source = self.create_source()

        ref = np.array([[1,0,0,0], [1,0,0,0]])
        source.c = 1
        np.testing.assert_equal(source.c, ref)
        source.c = [1,1]
        np.testing.assert_equal(source.c, ref)

        ref = np.array([[1,0,0,0], [2,0,0,0]])
        source.c = [1,2]
        np.testing.assert_equal(source.c, ref)
        source.c = [[1,0], [2,0]]
        np.testing.assert_equal(source.c, ref)

        ref = np.array([[1,2,0,0], [3,4,0,0]])
        source.c = [[1,2], [3,4]]
        np.testing.assert_equal(source.c, ref)
        source.c = ref
        np.testing.assert_equal(source.c, ref)

        cstr = self.create_cstr()

        ref = np.array([1,0,0,0])
        cstr.flow_rate = 1
        cstr.flow_rate_filter = 1
        np.testing.assert_equal(cstr.flow_rate, ref)
        np.testing.assert_equal(cstr.flow_rate_filter, ref)
        cstr.flow_rate = [1,0]
        cstr.flow_rate_filter = [1,0]
        np.testing.assert_equal(cstr.flow_rate, ref)
        np.testing.assert_equal(cstr.flow_rate_filter, ref)

        ref = np.array([1,1,0,0])
        cstr.flow_rate = [1,1]
        cstr.flow_rate_filter = [1,1]
        np.testing.assert_equal(cstr.flow_rate, ref)
        np.testing.assert_equal(cstr.flow_rate_filter, ref)
        cstr.flow_rate = ref
        cstr.flow_rate_filter = ref
        np.testing.assert_equal(cstr.flow_rate, ref)
        np.testing.assert_equal(cstr.flow_rate_filter, ref)


    def test_parameters(self):
        """
        Note
        ----
        Currently, only getting parameters is tested. Should also test if
        setting works. For this, adsorption parameters should be provided.
        """
        cstr = self.create_cstr()
        parameters_expected = {
                'flow_rate': np.array([1,0,0,0]),
                'porosity': total_porosity,
                'flow_rate_filter': np.array([0,0,0,0]),
        }
        np.testing.assert_equal(parameters_expected, cstr.parameters)

        sec_dep_parameters_expected = {
                'flow_rate': np.array([1,0,0,0]),
                'flow_rate_filter': np.array([0,0,0,0]),
        }
        np.testing.assert_equal(
            sec_dep_parameters_expected, cstr.section_dependent_parameters
        )

        poly_parameters = {
                'flow_rate': np.array([1,0,0,0]),
                'flow_rate_filter': np.array([0,0,0,0]),
        }
        np.testing.assert_equal(
            poly_parameters, cstr.polynomial_parameters
        )


if __name__ == '__main__':
    unittest.main()
