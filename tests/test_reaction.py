"""
Todo
----
Add more tests for checking manual setting of exponents.
"""

import unittest

import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import MassActionLaw, MassActionLawParticle

class Test_Reaction(unittest.TestCase):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName)


    def create_simple_bulk_reaction(self, is_kinetic=True, k_fwd_min=100):
        # 0: NH4+(aq) <=> NH3(aq) + H+(aq)
        component_system = ComponentSystem()
        component_system.add_component('H+', charges=[1])
        component_system.add_component(
            'Ammonia', species=['NH4+', 'NH3'], charges=[1,0]
        )
        reaction_model = MassActionLaw(component_system, 'simple')
        reaction_model.add_reaction(
            [1, 2, 0], [-1, 1, 1], 10**(-9.2)*1e3,
            is_kinetic=is_kinetic, k_fwd_min=k_fwd_min
        )
        return reaction_model

    def create_complex_bulk_reaction(self, is_kinetic=True, k_fwd_min=1):
        # Reactions
        # 0: NH4+(aq) <=> NH3(aq) + H+(aq)
        # 1: Lys2+(aq) <=> Lys+(aq) + H+(aq)
        # 2: Lys+(aq) <=> Lys(aq) + H+(aq)
        # 3: Lys(aq) <=> Lys-(aq) + H+(aq)
        # 4: Arg2+ <=> Arg+ + H+
        # 5: Arg+ <=> Arg + H+
        # 6: Arg <=> Arg- + H+
        component_system = ComponentSystem()
        component_system.add_component('H+', charges=[1])
        component_system.add_component(
            'Ammonia', species=['NH4+', 'NH3'], charges=[1,0]
        )
        component_system.add_component(
            'Lysine',
            species=['Lys2+', 'Lys+', 'Lys', 'Lys-'],
            charges=[2,1,0,-1]
        )
        component_system.add_component(
            'Arginine',
            species=['Arg2+', 'Arg+', 'Arg', 'Arg-'],
            charges=[2,1,0,-1]
        )
        reaction_model = MassActionLaw(component_system, 'complex')
        reaction_model.add_reaction(
            [1, 2, 0], [-1, 1, 1], 10**(-9.2)*1e3,
            is_kinetic=is_kinetic, k_fwd_min=k_fwd_min
        )
        reaction_model.add_reaction(
            [3, 4, 0], [-1, 1, 1], 10**(-2.20)*1e3,
            is_kinetic=is_kinetic, k_fwd_min=k_fwd_min
        )
        reaction_model.add_reaction(
            [4, 5, 0], [-1, 1, 1], 10**(-8.90)*1e3,
            is_kinetic=is_kinetic, k_fwd_min=k_fwd_min
        )
        reaction_model.add_reaction(
            [5, 6, 0], [-1, 1, 1], 10**(-10.28)*1e3,
            is_kinetic=is_kinetic, k_fwd_min=k_fwd_min
        )
        reaction_model.add_reaction(
            [7, 8, 0], [-1, 1, 1], 10**(-2.18)*1e3,
            is_kinetic=is_kinetic, k_fwd_min=k_fwd_min
        )
        reaction_model.add_reaction(
            [8, 9, 0], [-1, 1, 1], 10**(-9.09)*1e3,
            is_kinetic=is_kinetic, k_fwd_min=k_fwd_min
        )
        reaction_model.add_reaction(
            [9, 10, 0], [-1, 1, 1], 10**(-13.2)*1e3,
            is_kinetic=is_kinetic, k_fwd_min=k_fwd_min
        )

        return reaction_model

    def test_stoich_bulk(self):
        simple_model = self.create_simple_bulk_reaction()
        stoich_expected = np.array([
            [1.0],
            [-1.0],
            [1.0]
        ])
        stoich_expected = np.array([1,-1,1], ndmin=2).T
        np.testing.assert_array_equal(stoich_expected, simple_model.stoich)

        complex_model = self.create_complex_bulk_reaction()
        stoich_expected = np.array([
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
            [-1.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 1.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  1., -1.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  1., -1.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  1.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0., -1.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  1., -1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  1., -1.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  1.]
        ])

        np.testing.assert_array_equal(stoich_expected, complex_model.stoich)

    def test_reaction_rates(self):
        simple_model_kin = self.create_simple_bulk_reaction(is_kinetic=True)
        k_fwd_expected = [6.309573444801942e-07]
        self.assertEqual(k_fwd_expected, simple_model_kin.k_fwd)
        k_bwd_expected = [1]
        self.assertEqual(k_bwd_expected, simple_model_kin.k_bwd)
        k_eq_expeced = [6.309573444801942e-07]
        self.assertEqual(k_eq_expeced, simple_model_kin.k_eq)

        simple_model_eq = self.create_simple_bulk_reaction(
            is_kinetic=False, k_fwd_min=1
        )
        k_fwd_expected = [1]
        self.assertEqual(k_fwd_expected, simple_model_eq.k_fwd)
        k_bwd_expected = [1584893.1924611111]
        self.assertEqual(k_bwd_expected, simple_model_eq.k_bwd)
        k_eq_expeced = [6.309573444801942e-07]
        self.assertEqual(k_eq_expeced, simple_model_eq.k_eq)


        simple_model_eq = self.create_simple_bulk_reaction(
            is_kinetic=False, k_fwd_min=100
        )
        k_fwd_expected = [100]
        self.assertEqual(k_fwd_expected, simple_model_eq.k_fwd)
        k_bwd_expected = [158489319.2461111]
        self.assertEqual(k_bwd_expected, simple_model_eq.k_bwd)
        k_eq_expeced = [6.309573444801942e-07]
        self.assertEqual(k_eq_expeced, simple_model_eq.k_eq)

    def create_cross_phase_reaction(self):
        component_system = ComponentSystem()
        component_system.add_component(
            'Ammonia', species=['NH4+', 'NH3'], charges=[1,0]
        )
        component_system.add_component(
            'Lysine',
            species=['Lys2+', 'Lys+', 'Lys', 'Lys-'],
            charges=[2,1,0,-1]
        )
        component_system.add_component('H+', charges=[1])

        reaction_model = MassActionLawParticle(component_system, 'complex')

        # Pore Liquid
        # 0: NH4+(aq) <=> NH3(aq) + H+(aq)
        # 1: Lys2+(aq) <=> Lys+(aq) + H+(aq)
        # 2: Lys+(aq) <=> Lys(aq) + H+(aq)
        # 3: Lys(aq) <=> Lys-(aq) + H+(aq)
        reaction_model.add_liquid_reaction(
            [1, 2, 0], [-1, 1, 1], 10**(-9.2)*1e3, is_kinetic=False
        )
        reaction_model.add_liquid_reaction(
            [3, 4, 0], [-1, 1, 1], 10**(-2.20)*1e3, is_kinetic=False
        )
        reaction_model.add_liquid_reaction(
            [4, 5, 0], [-1, 1, 1], 10**(-8.90)*1e3, is_kinetic=False
        )
        reaction_model.add_liquid_reaction(
            [5, 6, 0], [-1, 1, 1], 10**(-10.28)*1e3, is_kinetic=False
        )

        # Adsorption
        k_fwd_min_ads = 100
        # 0: NH4+(aq) + H+(s) <=> NH4+(s) + H+(aq)
        # 1: NH4+(s) <=> NH3(aq) + H+(s)
        reaction_model.add_cross_phase_reaction(
            [1, 0, 1, 0], [-1, -1, 1, 1], [0, 1, 1, 0], 1/1.5,
            k_fwd_min=k_fwd_min_ads
        )
        reaction_model.add_cross_phase_reaction(
            [1, 2, 0], [-1, 1, 1], [1, 0, 1], 1.5, k_fwd_min=k_fwd_min_ads
        )
        # 2: Lys2+(aq) + 2H+(s) <=> Lys2+(s) + 2H+(aq)
        # 3: Lys2+(s) <=> Lys+(aq) + H+(s)
        reaction_model.add_cross_phase_reaction(
            [3, 0, 3, 0], [-1, -2, 1, 2], [0, 1, 1, 0], 1/5,
            k_fwd_min=k_fwd_min_ads
        )
        reaction_model.add_cross_phase_reaction(
            [3, 2, 0], [-1, 1, 1], [1, 0, 1], 5,
            k_fwd_min=k_fwd_min_ads
        )
        # 4: Lys+(aq) + H+(s) <=> Lys+(s) + H+(aq)
        # 5: Lys+(s) <=> Lys(aq) + H+(s)
        reaction_model.add_cross_phase_reaction(
            [4, 0, 4, 0], [-1, -1, 1, 1], [0, 1, 1, 0], 1/0.75,
            k_fwd_min=k_fwd_min_ads
        )
        reaction_model.add_cross_phase_reaction(
            [4, 5, 0], [-1, 1, 1], [1, 0, 1], 0.75,
            k_fwd_min=k_fwd_min_ads
        )

        return reaction_model

    def test_stoich_cross_phase(self):
        cross_phase_model = self.create_cross_phase_reaction()

        stoich_liquid_expected = np.array([
            [ 1.,  1.,  1.,  1.,  1.,  0.,  2.,  0.,  1.,  0.],
            [-1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
            [ 1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
            [ 0.,  1., -1.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  1.],
            [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        ])
        np.testing.assert_array_equal(
            stoich_liquid_expected, cross_phase_model.stoich_liquid
        )

        stoich_solid_expected = np.array([
            [-1.,  1., -2.,  1., -1.,  1.],
            [ 1., -1.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  1., -1.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  1., -1.],
            [ 0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.],
        ])
        np.testing.assert_array_equal(
            stoich_solid_expected, cross_phase_model.stoich_solid
        )


if __name__ == '__main__':
    unittest.main()