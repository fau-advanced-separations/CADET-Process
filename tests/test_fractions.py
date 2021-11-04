
import math
import unittest

import numpy as np
import CADETProcess


class Test_Fractions(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        
    def create_fractions(self):
        m_0 = np.array([0,0])
        frac0 = CADETProcess.fractionation.Fraction(m_0,1)
        m_1 = np.array([3,0])
        frac1 = CADETProcess.fractionation.Fraction(m_1,2)
        m_2 = np.array([1,2])
        frac2 = CADETProcess.fractionation.Fraction(m_2,3)
        m_3 = np.array([0,2])
        frac3 = CADETProcess.fractionation.Fraction(m_3,3)
        m_4 = np.array([0,0])
        frac4 = CADETProcess.fractionation.Fraction(m_4,0)
        
        return frac0, frac1, frac2, frac3, frac4
        
    def create_pools(self):
        fractions = self.create_fractions()
        
        pool_waste = CADETProcess.fractionation.FractionPool(n_comp=2)
        pool_waste.add_fraction(fractions[0])
        pool_waste.add_fraction(fractions[1])
        pool_waste.add_fraction(fractions[2])
        
        pool_1 = CADETProcess.fractionation.FractionPool(n_comp=2)
        pool_1.add_fraction(fractions[3])
        
        pool_2 = CADETProcess.fractionation.FractionPool(n_comp=2)
        pool_2.add_fraction(fractions[4])
        
        return pool_waste, pool_1, pool_2
            
    def test_fraction_mass(self):
        fractions = self.create_fractions()
        
        self.assertEqual(fractions[0].fraction_mass, 0)
        self.assertEqual(fractions[1].fraction_mass, 3)
        self.assertEqual(fractions[2].fraction_mass, 3)
        self.assertEqual(fractions[3].fraction_mass, 2)
        self.assertEqual(fractions[4].fraction_mass, 0)
        
    def test_fraction_concentration(self):
        fractions = self.create_fractions()
        
        np.testing.assert_equal(fractions[0].concentration, np.array([0., 0.]))
        np.testing.assert_equal(fractions[1].concentration, np.array([1.5, 0]))
        np.testing.assert_equal(fractions[2].concentration, np.array([1/3, 2/3]))
        np.testing.assert_equal(fractions[3].concentration, np.array([0, 2/3]))
        np.testing.assert_equal(fractions[4].concentration, np.array([0, 0]))
    
    def test_fraction_purity(self):
        fractions = self.create_fractions()
        
        np.testing.assert_equal(fractions[0].purity, np.array([0., 0.]))
        np.testing.assert_equal(fractions[1].purity, np.array([1, 0]))
        np.testing.assert_equal(fractions[2].purity, np.array([1/3, 2/3]))
        np.testing.assert_equal(fractions[3].purity, np.array([0, 1]))
        np.testing.assert_equal(fractions[4].purity, np.array([0, 0]))
        
    def test_n_comp(self):
        pools = self.create_pools()

        self.assertEqual(pools[0].n_comp, 2)
        self.assertEqual(pools[0].fractions[0].n_comp, 2)
        
        m_wrong_n_comp = np.array([0,0,0])
        frac_wrong_n_comp = CADETProcess.fractionation.Fraction(m_wrong_n_comp,1)
        
        with self.assertRaises(CADETProcess.CADETProcessError):
            pools[0].add_fraction(frac_wrong_n_comp)        
    
    def test_pool_mass(self):
        pools = self.create_pools()

        np.testing.assert_equal(pools[0].mass, np.array([4., 2.]))
        np.testing.assert_equal(pools[1].mass, np.array([0, 2]))
        np.testing.assert_equal(pools[2].mass, np.array([0, 0]))

    def test_pool_pool_mass(self):
        pools = self.create_pools()

        self.assertEqual(pools[0].pool_mass, 6)
        self.assertEqual(pools[1].pool_mass, 2)
        self.assertEqual(pools[2].pool_mass, 0)

    def test_pool_volume(self):
        pools = self.create_pools()

        self.assertEqual(pools[0].volume, 6)
        self.assertEqual(pools[1].volume, 3)
        self.assertEqual(pools[2].volume, 0)

    def test_pool_concentration(self):
        pools = self.create_pools()

        np.testing.assert_equal(pools[0].concentration, np.array([2/3, 1/3]))
        np.testing.assert_equal(pools[1].concentration, np.array([0, 2/3]))
        np.testing.assert_equal(pools[2].concentration, np.array([0, 0]))

    def test_pool_purity(self):
        pools = self.create_pools()

        np.testing.assert_equal(pools[0].purity, np.array([2/3, 1/3]))
        np.testing.assert_equal(pools[1].purity, np.array([0, 1]))
        np.testing.assert_equal(pools[2].purity, np.array([0, 0]))
        
        
if __name__ == '__main__':
    unittest.main()
