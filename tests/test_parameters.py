import unittest

import numpy as np

from CADETProcess.common import StructMeta, \
    String, List, RangedFloat, UnsignedInteger, \
    DependentlySizedUnsignedNdArray, DependentlySizedUnsignedList

class TestParameters(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        
    def setUp(self):
        self.dummy = DummyModel()
        
    def test_values(self):
        
        with self.assertRaises(ValueError):
            self.dummy.string_var
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
        
class DummyModel(metaclass=StructMeta):
    string_var = String()
    list_var = List()
    unsigned_var = UnsignedInteger()
    bound_var = RangedFloat(lb=-1, ub=1)
    
    dep_1 = UnsignedInteger(default=2)
    single_dep_var = DependentlySizedUnsignedList(dep='dep_1', default=1)
    dep_2 = UnsignedInteger()
    double_dep_var = DependentlySizedUnsignedNdArray(
        dep=('dep_1', 'dep_2'), default=1)


if __name__ == '__main__':
    unittest.main()