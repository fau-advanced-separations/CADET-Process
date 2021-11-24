import unittest
import numpy as np

from CADETProcess.processModel import ComponentSystem

class TestComponents(unittest.TestCase):
    def setUp(self):
        self.component_system_1 = ComponentSystem()
        self.component_system_1.add_component(
            'Ammonia', 
            species=['NH4+', 'NH3'],
            charge=[1,0]
        )
        self.component_system_1.add_component(
            'Lysine', 
            ['Lys2+', 'Lys+', 'Lys', 'Lys'],
            [2,1,0,-1]
        )
        self.component_system_1.add_component(
            'H+', 
            charge=1
        )
        self.component_system_2 = ComponentSystem(['A', ['B', 'C']])

        self.component_system_3 = ComponentSystem(2)
        self.component_system_3.add_component('manual_label')
        
        self.component_system_4 = ComponentSystem(2, charges=[1,1])
        self.component_system_4 = ComponentSystem(['A', ['B', 'C']], charges=[1,[1,1]])
        
    def test_labels(self):
        labels_expected = [
            'NH4+', 'NH3', 'Lys2+', 'Lys+', 'Lys', 'Lys', 'H+'
        ]
        labels = self.component_system_1.labels
        np.testing.assert_equal(labels, labels_expected)
        
        labels_expected = ['A', 'B', 'C']
        labels = self.component_system_2.labels
        np.testing.assert_equal(labels, labels_expected)
        
        labels_expected = ['0', '1', 'manual_label']
        labels = self.component_system_3.labels
        np.testing.assert_equal(labels, labels_expected)
        
    def test_indices(self):
        indices_expected = {
            'Ammonia': [0, 1],
            'Lysine': [2,3,4,5],
            'H+': [6],
        }
        indices = self.component_system_1.indices
        np.testing.assert_equal(indices, indices_expected)
        
    def test_n_comp(self):
        n_components_expected = 3
        n_components = self.component_system_1.n_components
        self.assertEqual(n_components_expected, n_components)
        
        n_comp_expected = 7
        n_comp = self.component_system_1.n_comp
        self.assertEqual(n_comp_expected, n_comp)

        n_components_expected = 2
        n_components = self.component_system_2.n_components
        self.assertEqual(n_components_expected, n_components)
        
        n_comp_expected = 3
        n_comp = self.component_system_2.n_comp
        self.assertEqual(n_comp_expected, n_comp)

        n_components_expected = 2
        n_components = self.component_system_2.n_components
        self.assertEqual(n_components_expected, n_components)
        
        n_comp_expected = 3
        n_comp = self.component_system_2.n_comp
        self.assertEqual(n_comp_expected, n_comp)
        
        n_components_expected = 3
        n_components = self.component_system_3.n_components
        self.assertEqual(n_components_expected, n_components)
        
        n_comp_expected = 3
        n_comp = self.component_system_3.n_comp
        self.assertEqual(n_comp_expected, n_comp)
        
    def test_charge(self):
        charges_expected = [1,0,2,1,0,-1,1]
        charges = self.component_system_1.charges
        np.testing.assert_equal(charges_expected, charges)
        
    def test_total_concentration(self):
        pass

if __name__ == '__main__':
    unittest.main()
    