import unittest

import numpy as np

from CADETProcess.common import StructMeta, \
    String, List, RangedFloat, UnsignedInteger, \
    DependentlySizedUnsignedNdArray, DependentlySizedUnsignedList, \
    Polynomial, NdPolynomial

class TestParameters(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        
    def setUp(self):
        class DummyModel(metaclass=StructMeta):
            string_var = String()
            list_var = List()
            unsigned_var = UnsignedInteger()
            bound_var = RangedFloat(lb=-1, ub=1)
            
            dep_1 = UnsignedInteger(default=2, description='foo')
            single_dep_var = DependentlySizedUnsignedList(dep='dep_1', default=1)
            dep_2 = UnsignedInteger()
            double_dep_var = DependentlySizedUnsignedNdArray(
                dep=('dep_1', 'dep_2'), default=1
            )
            
            n_coeff = 4
            poly_single_hard = Polynomial(n_coeff=2)
            poly_single_dep = Polynomial(dep=('n_coeff'))

            entries = 3
            poly_nd_hard = NdPolynomial(n_coeff=2, dep=('entries'))
            poly_nd_dep = NdPolynomial(dep=('entries', 'n_coeff'))
                                 
        self.dummy = DummyModel()
        
    def test_values(self):
        self.dummy.string_var = 'sting_var'
        self.assertEqual(self.dummy.string_var, 'sting_var')

    def test_default(self):
        self.assertEqual(self.dummy.dep_1, 2)
        
        with self.assertRaises(TypeError):
            class WrongDefaultDummy_1(metaclass=StructMeta):
                param_1 = UnsignedInteger(default='string')
        
        with self.assertRaises(ValueError):
            class WrongDefaultDummy_2(metaclass=StructMeta):
                param_1 = UnsignedInteger(default=-1)

        
    def test_type_checking(self):
        with self.assertRaises(TypeError):
            self.dummy.string_var = 1
        with self.assertRaises(TypeError):
            self.dummy.string_var = [1]
        with self.assertRaises(TypeError):
            self.dummy.string_var = 1.6        
            
    
    def test_range_checking(self):
        with self.assertRaises(ValueError):
            self.dummy.unsigned_var = -1
        with self.assertRaises(ValueError):
            self.dummy.bound_var = -2
        with self.assertRaises(ValueError):
            self.dummy.bound_var = 2    

    def test_dependencies(self):
        with self.assertRaises(ValueError):
            self.dummy.unsigned_var = -1
            
        self.assertListEqual(self.dummy.single_dep_var, [1,1])
        
        self.dummy.dep_2 = 3
        np.testing.assert_array_equal(self.dummy.double_dep_var, np.ones((2,3)))
            
        # wrong shape
        with self.assertRaises(ValueError):
            self.dummy.double_dep_var = np.ones((3,3))
        
        # wrong range
        with self.assertRaises(ValueError):
            self.dummy.double_dep_var = -1 * np.ones((2,3))
    
    def test_description(self):
        self.assertEqual(type(self.dummy).dep_1.description, 'foo')
        
    def test_modified_descriptor(self):
        try:
            type(self.dummy).bound_var.ub = 2
            self.dummy.bound_var = 2
        except ValueError as e:
            self.fail(str(e))
            
    def test_poly(self):
        self.dummy.poly_single_hard = 1
        np.testing.assert_equal(self.dummy.poly_single_hard, [1,0])
        self.dummy.poly_single_dep = 1
        np.testing.assert_equal(self.dummy.poly_single_dep, [1,0,0,0])
        
        self.dummy.poly_single_hard = [1,2]
        np.testing.assert_equal(self.dummy.poly_single_hard, [1,2])
        self.dummy.poly_single_dep = [1,2]
        np.testing.assert_equal(self.dummy.poly_single_dep, [1,2,0,0])    
        
        with self.assertRaises(ValueError):
            class TestDuplicateCoeff(metaclass=StructMeta):
                n_coeff = 2
                faulty = Polynomial(n_coeff=2, dep=('n_coeff'))
        
        self.dummy.poly_nd_hard = 1
        np.testing.assert_equal(self.dummy.poly_nd_hard, [[1,0], [1,0], [1,0]])
        self.dummy.poly_nd_dep = 1
        np.testing.assert_equal(
            self.dummy.poly_nd_dep, [[1,0,0,0], [1,0,0,0], [1,0,0,0]],
        )
        
        self.dummy.poly_nd_hard = [1,2,3]
        np.testing.assert_almost_equal(
            self.dummy.poly_nd_hard, [[1,0], [2,0], [3,0]]
        )
        self.dummy.poly_nd_dep = [1,2,3]
        np.testing.assert_almost_equal(
            self.dummy.poly_nd_dep, [[1,0,0,0], [2,0,0,0], [3,0,0,0]]
        )
        
        self.dummy.poly_nd_dep = [[1,1],[2,-2],[3,3]]
        np.testing.assert_almost_equal(
            self.dummy.poly_nd_dep, [[1,1,0,0], [2,-2,0,0], [3,3,0,0]]
        )    
        
        self.dummy.poly_nd_dep = [[1],[2,-2],[3,3,0,0]]
        np.testing.assert_almost_equal(
            self.dummy.poly_nd_dep, [[1,0,0,0], [2,-2,0,0], [3,3,0,0]]
        )            
        
        with self.assertRaises(ValueError):
            self.dummy.poly_nd_hard = [1]
        with self.assertRaises(ValueError):
            self.dummy.poly_nd_hard = [1,1,1,1]
        
        with self.assertRaises(ValueError):
            # Missing number of entries
            class TestDuplicateCoeff(metaclass=StructMeta):
                n_coeff = 2
                faulty = NdPolynomial(dep=('n_coeff'))                
        
        with self.assertRaises(ValueError):
            class TestDuplicateCoeff(metaclass=StructMeta):
                n_coeff = 2
                entries = 2                
                faulty = NdPolynomial(n_entries=2, n_coeff=2, dep=('n_coeff'))                
        

if __name__ == '__main__':
    unittest.main()