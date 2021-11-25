import unittest

import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import MassActionLaw
from CADETProcess import equilibria

class TestBufferCapacity(unittest.TestCase):

    def __init__(self, methodName = 'runTest'):
        super().__init__(methodName)
        
    def setUp(self):
        self.components_simple = ComponentSystem(2, charges=[1,2])
        
        self.components_ammonia = ComponentSystem()
        self.components_ammonia.add_component(
            'Ammonia', 
            species=['NH4+', 'NH3'],
            charge=[1, 0]
        )
        self.components_ammonia.add_component(
            'H+', 
            charge=1
        )
        self.reaction_ammonia = MassActionLaw(self.components_ammonia)
        self.reaction_ammonia.add_reaction(
            [0, 1, 2], [-1, 1, 1], 10**(-9.2)*1e3, is_kinetic=False
        )
        
        self.components_lys = ComponentSystem()
        self.components_lys.add_component(
            'Lysine', 
            species=['Lys2+', 'Lys+', 'Lys', 'Lys-'],
            charge=[2, 1, 0, -1]
        )
        self.components_lys.add_component(
            'H+', 
            charge=1
        )
        self.reaction_lys = MassActionLaw(self.components_lys)
        self.reaction_lys.add_reaction(
            [0, 1, -1], [-1, 1, 1], 10**(-2.20)*1e3, is_kinetic=False
        )
        self.reaction_lys.add_reaction(
            [1, 2, -1], [-1, 1, 1], 10**(-8.90)*1e3, is_kinetic=False
        )
        self.reaction_lys.add_reaction(
            [2, 3, -1], [-1, 1, 1], 10**(-10.28)*1e3, is_kinetic=False
        )
        
        self.components_ammonia_lys = ComponentSystem()
        self.components_ammonia_lys.add_component(
            'Ammonia', 
            species=['NH4+', 'NH3'],
            charge=[1, 0]
        )
        self.components_ammonia_lys.add_component(
            'Lysine', 
            species=['Lys2+', 'Lys+', 'Lys', 'Lys-'],
            charge=[2, 1, 0, -1]
        )
        self.components_ammonia_lys.add_component(
            'H+', 
            charge=1
        )
        self.reaction_ammonia_lys = MassActionLaw(self.components_ammonia_lys)
        self.reaction_ammonia_lys.add_reaction(
            [0, 1, -1], [-1, 1, 1], 10**(-9.2)*1e3, is_kinetic=False
        )
        self.reaction_ammonia_lys.add_reaction(
            [2, 3, -1], [-1, 1, 1], 10**(-2.20)*1e3, is_kinetic=False
        )
        self.reaction_ammonia_lys.add_reaction(
            [3, 4, -1], [-1, 1, 1], 10**(-8.90)*1e3, is_kinetic=False
        )
        self.reaction_ammonia_lys.add_reaction(
            [4, 5, -1], [-1, 1, 1], 10**(-10.28)*1e3, is_kinetic=False
        )
        

    def test_ionic_strength(self):
        c = [1,1]
        i_excpected = 2.5
        i = equilibria.ionic_strength(self.components_simple, c)
        np.testing.assert_almost_equal(i, i_excpected)
        
        c = [1,2]
        i_excpected = 4.5
        i = equilibria.ionic_strength(self.components_simple, c)
        np.testing.assert_almost_equal(i, i_excpected)
        
        c = [1,0,1]
        i_excpected = 1
        i = equilibria.ionic_strength(self.components_ammonia, c)
        np.testing.assert_almost_equal(i, i_excpected)
        
        # Uncharged species should have no effect
        c = [1,1,1]
        i_excpected = 1
        i = equilibria.ionic_strength(self.components_ammonia, c)
        np.testing.assert_almost_equal(i, i_excpected)
        
        c = [2,0,1]
        i_excpected = 1.5
        i = equilibria.ionic_strength(self.components_ammonia, c)
        np.testing.assert_almost_equal(i, i_excpected)
        
        
        
    # def test_
    def test_buffer_capacity(self):
        pH = np.linspace(0,14,101)
        
        buffer = [1000, 0, 0]
        b = equilibria.buffer_capacity(self.reaction_ammonia, buffer, pH)
        plt.figure()
        plt.plot(pH, b)
        
        buffer = [0, 0, 1000, 0, 0]
        b = equilibria.buffer_capacity(self.reaction_lys, buffer, pH)
        plt.figure()
        plt.plot(pH, b)
        
        buffer = [1000, 0, 0, 0, 1000, 0, 0]
        b = equilibria.buffer_capacity(self.reaction_ammonia_lys, buffer, pH)
        plt.figure()
        plt.plot(pH, b)
        
        # c = [1,1]
        # i_excpected = 1
        # i = equilibria.ionic_strength(self.component_system_2, c)
        # np.testing.assert_almost_equal(i, i_excpected)
        
        # c = [1,1]
        # i_excpected = 1.5
        # i = equilibria.ionic_strength(self.component_system_2, c)
        # np.testing.assert_almost_equal(i, i_excpected)
        
        

        
        
        
if __name__ == '__main__':
    unittest.main()    
