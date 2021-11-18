import unittest

import numpy as np

from CADETProcess.processModel import MassActionLaw
from CADETProcess.processModel import ComponentSystem

from CADETProcess import equilibria

class TestReactionEquilibrium(unittest.TestCase):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName)
        
    def setUp(self):
        component_system = ComponentSystem()
        component_system.add_component('A')
        component_system.add_component('B')
        
        self.single_0 = MassActionLaw(component_system, name='simple')
        self.single_0.add_reaction(
            [0, 1], [-1, 1], 2, k_bwd=2
        )
        
        self.single_1 = MassActionLaw(component_system, name='simple')
        self.single_1.add_reaction(
            [0, 1], [-1, 1], 1, k_bwd=1
        )
        
        component_system = ComponentSystem()
        component_system.add_component('A')
        component_system.add_component('B')
        component_system.add_component('C')
        self.single_2 = MassActionLaw(n_comp=3, name='simple')
        self.single_2.add_reaction(
            [0, 1], [-1, 1], 1, k_bwd=1
        )
        
        self.multi = MassActionLaw(n_comp=3, name='simple')
        self.multi.add_reaction(
            [0, 1], [-1, 1], 1, k_bwd=1
        )
        self.multi.add_reaction(
            [1, 2], [-1, 1], 1, k_bwd=1
        )
        
        self.multi = MassActionLaw(n_comp=3, name='simple')
        self.multi.add_reaction(
            [0, 1], [-1, 1], 1, k_bwd=1
        )
        self.multi.add_reaction(
            [1, 2], [-1, 1], 1, k_bwd=1
        )
        
        self.lysine = MassActionLaw(n_comp=5, name='Lysine')
        self.lysine.add_reaction(
            [0, 1, -1], [-1, 1, 1], 10**(-2.20)*1e3, is_kinetic=False
        )
        self.lysine.add_reaction(
            [1, 2, -1], [-1, 1, 1], 10**(-8.90)*1e3, is_kinetic=False
        )
        self.lysine.add_reaction(
            [2, 3, -1], [-1, 1, 1], 10**(-10.28)*1e3, is_kinetic=False
        )

    def test_dydx(self):
        buffer = [1,0]
        dydx = equilibria.dydx_mal(buffer, self.single_1)
        dydx_expected = [-1, 1]
        np.testing.assert_almost_equal(dydx, dydx_expected)
        
        buffer = [1,0,0]
        dydx = equilibria.dydx_mal(buffer, self.single_2)
        dydx_expected = [-1, 1, 0]
        np.testing.assert_almost_equal(dydx, dydx_expected)
        
        buffer = [1,0,0]
        dydx = equilibria.dydx_mal(buffer, self.single_2, constant_indices=[0])
        dydx_expected = [0, 1, 0]
        np.testing.assert_almost_equal(dydx, dydx_expected)
        
        buffer = [1,0,0]
        buffer_init = [2,0,0]
        dydx = equilibria.dydx_mal(
            buffer, self.single_2, constant_indices=[0], c_init=buffer_init)
        dydx_expected = [0, 2, 0]
        np.testing.assert_almost_equal(dydx, dydx_expected)
        
        buffer = [1,0,0]
        dydx = equilibria.dydx_mal(buffer, self.multi)
        dydx_expected = [-1, 1, 0]
        np.testing.assert_almost_equal(dydx, dydx_expected)
        
        buffer = [1,1,0]
        dydx = equilibria.dydx_mal(buffer, self.multi)
        dydx_expected = [0, -1, 1]
        np.testing.assert_almost_equal(dydx, dydx_expected)
        
        buffer = [1,1,1]
        dydx = equilibria.dydx_mal(buffer, self.multi)
        dydx_expected = [0, 0, 0]
        np.testing.assert_almost_equal(dydx, dydx_expected)
        
        buffer = [0,1,0]
        dydx = equilibria.dydx_mal(buffer, self.multi)
        dydx_expected = [1, -2, 1]
        np.testing.assert_almost_equal(dydx, dydx_expected)
    
    def test_jac(self):
        buffer = [1,0]
        jac = equilibria.jac_mal(buffer, self.single_0)
        jac_expected = [
            [-2, 2],
            [ 2, -2]
        ]
        np.testing.assert_almost_equal(jac, jac_expected)
        
    def test_reaction_equilibrium(self):
        buffer = [1, 0]
        
        eq = equilibria.calculate_buffer_equilibrium(buffer, self.single_1)
        eq_expected = [0.5, 0.5]
        np.testing.assert_almost_equal(eq, eq_expected)
        
        buffer = [0, 0, 0, 1, 0]
        
        pH = 0
        buffer[-1] = 10**(-pH+3)
        eq = equilibria.calculate_buffer_equilibrium(
            buffer, self.lysine, constant_indices=[-1]
        )
        eq_expected = [
            0.99373000042703, 0.006270012422, 7.89e-12, 0.0,
            buffer[-1]
        ]
        pH = 3
        buffer[-1] = 10**(-pH+3)
        eq = equilibria.calculate_buffer_equilibrium(
            buffer, self.lysine, constant_indices=[-1]
        )
        eq_expected = [
            0.13680673998909, 0.86319217370507, 1.08669456e-06, 6e-14,
            buffer[-1]
        ]
        np.testing.assert_almost_equal(eq, eq_expected)
        pH = 7
        buffer[-1] = 10**(-pH+3)
        eq = equilibria.calculate_buffer_equilibrium(
            buffer, self.lysine, constant_indices=[-1]
        )
        eq_expected = [
            1.565153925e-05, 0.98754536426943, 0.01243245954378, 6.52464752e-06,
            buffer[-1]
        ]
        np.testing.assert_almost_equal(eq, eq_expected)
        pH = 11
        buffer[-1] = 10**(-pH+3)
        eq = equilibria.calculate_buffer_equilibrium(
            buffer, self.lysine, constant_indices=[-1]
        )
        eq_expected = [
            2.01e-12, 0.00126970262768, 0.15984609034134, 0.83888420702896,
            buffer[-1]
        ]
        np.testing.assert_almost_equal(eq, eq_expected)
        pH = 14
        buffer[-1] = 10**(-pH+3)
        eq = equilibria.calculate_buffer_equilibrium(
            buffer, self.lysine, constant_indices=[-1]
        )
        eq_expected = [
            0.0, 0.0, 1e-11, 0.99999999999,
            buffer[-1]
        ]
        
from CADETProcess.processModel import Linear, Langmuir, StericMassAction
class TestAdsorptionEquilibrium(unittest.TestCase):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName)
        
    def setUp(self):
        self.linear = Linear(1, 'linear')
        self.linear.adsorption_rate = [1]
        
        self.langmuir = Langmuir(2, 'langmuir')
        self.langmuir.adsorption_rate = [2,1]
        self.langmuir.desorption_rate = [1,1]
        self.langmuir.saturation_capacity = [10, 10]
        
        self.sma = StericMassAction(2, 'SMA')
        self.sma.adsorption_rate = [1, 2]
        self.sma.desorption_rate = [1,1]
        self.sma.characteristic_charge = [1, 1]
        self.sma.steric_factor = [0, 0]
        self.sma.capacity = 10
        
    
    def test_adsorption(self):
        buffer = [1]
        eq = equilibria.simulate_solid_equilibria(self.linear, buffer)
        eq_expected = [1]
        np.testing.assert_almost_equal(eq, eq_expected)
        
        buffer = [1, 1]
        eq = equilibria.simulate_solid_equilibria(self.langmuir, buffer)
        eq_expected = [5, 2.5]
        np.testing.assert_almost_equal(eq, eq_expected)
        
        buffer = [1, 1]
        eq = equilibria.simulate_solid_equilibria(self.sma, buffer)
        eq_expected = [10/3, 2*10/3]
        np.testing.assert_almost_equal(eq, eq_expected)
        
if __name__ == '__main__':
    unittest.main()    
