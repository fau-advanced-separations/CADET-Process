import unittest

import numpy as np
from scipy import integrate

from CADETProcess import CADETProcessError
from CADETProcess.common import Section, TimeLine


class TestSection(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def create_constant_section_single(self):
        return Section(0,1,1)
    
    def create_constant_section_multi(self):
        return Section(1,2,[1,2])
    
    def create_poly_section_single(self):
        return Section(0, 1, [0,1,0,0], n_entries=1, degree=3)
    
    def create_poly_section_multi(self):
        return Section(0,1, [[0,1],[1,-1]], n_entries=2, degree=3)
    
    def test_section_value(self):
        const_single = self.create_constant_section_single()
        self.assertEqual(const_single.value(0), 1)
        with self.assertRaises(ValueError):
            # Exceed section times
            val = const_single.value(2)

        const_multi = self.create_constant_section_multi()
        np.testing.assert_equal(const_multi.value(1), [1,2])
        with self.assertRaises(ValueError):
            # Exceed section times
            val = const_multi.value(0)
        
        poly_single = self.create_poly_section_single()
        np.testing.assert_equal(poly_single.value(0), 0)
        np.testing.assert_equal(poly_single.value(0.5), 0.5)
        np.testing.assert_equal(poly_single.value(1), 1)

        poly_multi = self.create_poly_section_multi()
        np.testing.assert_equal(poly_multi.value(0), [0,1])
        np.testing.assert_equal(poly_multi.value(0.5), [0.5,0.5])
        np.testing.assert_equal(poly_multi.value(1), [1,0])        
        
    def test_section_integral(self):
        const_single = self.create_constant_section_single()
        self.assertEqual(const_single.integral(0,0), 0)
        self.assertEqual(const_single.integral(0,1), 1)
        with self.assertRaises(ValueError):
            # Exceed section times
            val = const_single.value(2)
        
        const_multi = self.create_constant_section_multi()
        np.testing.assert_equal(const_multi.integral(1,2), [1,2])
        with self.assertRaises(ValueError):
            # Exceed section times
            val = const_multi.value(0)
        
        poly_single = self.create_poly_section_single()
        np.testing.assert_equal(poly_single.integral(0, 0.5), 0.125)
        np.testing.assert_equal(poly_single.integral(0, 1), 0.5)
        
        poly_multi = self.create_poly_section_multi()
        np.testing.assert_equal(poly_multi.integral(0, 0.5), [0.125, 0.375])
        np.testing.assert_equal(poly_multi.integral(0.5, 1), [0.375, 0.125])
        np.testing.assert_equal(poly_multi.integral(0, 1), [0.5, 0.5])   
        
class TestTimeLine(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)        
        
    def create_time_line_constant_single(self):
        section_0 = Section(0,1,1.5)
        section_1 = Section(1,2,0)
        section_2 = Section(2,3,0)
        section_3 = Section(3,4,1)
        section_4 = Section(4,5,2)
        section_5 = Section(5,6,2)
    
        tl = TimeLine()
        tl.add_section(section_0)
        tl.add_section(section_1)
        tl.add_section(section_2)
        tl.add_section(section_3)
        tl.add_section(section_4)
        tl.add_section(section_5)
    
        return tl

    def create_time_line_constant_multi(self):
        section_0 = Section(0,1,(1.5,0))
        section_1 = Section(1,2,(0,0))
        section_2 = Section(2,3,(0,1))
        section_3 = Section(3,4,(1,0))
        section_4 = Section(4,5,(2,0))
        section_5 = Section(5,6,(2,-2))
    
        tl = TimeLine()
        tl.add_section(section_0)
        tl.add_section(section_1)
        tl.add_section(section_2)
        tl.add_section(section_3)
        tl.add_section(section_4)
        tl.add_section(section_5)
    
        return tl    
        
    def create_time_line_poly_single(self):
        section_0 = Section(0,1,(1.5,0), n_entries=1, degree=2)
        section_1 = Section(1,2,(0,0), n_entries=1, degree=2)
        section_2 = Section(2,3,(0,1), n_entries=1, degree=2)
        section_3 = Section(3,4,(1,0), n_entries=1, degree=2)
        section_4 = Section(4,5,(2,0), n_entries=1, degree=2)
        section_5 = Section(5,6,(2,-2), n_entries=1, degree=2)
        
        tl = TimeLine()
        tl.add_section(section_0)
        tl.add_section(section_1)
        tl.add_section(section_2)
        tl.add_section(section_3)
        tl.add_section(section_4)
        tl.add_section(section_5)
    
        return tl
    
    def create_time_line_constant_multi(self):
        section_0 = Section(0,1,(1.5,0))
        section_1 = Section(1,2,(0,0))
        section_2 = Section(2,3,(0,1))
        section_3 = Section(3,4,(1,0))
        section_4 = Section(4,5,(2,0))
        section_5 = Section(5,6,(2,-2))
    
        tl = TimeLine()
        tl.add_section(section_0)
        tl.add_section(section_1)
        tl.add_section(section_2)
        tl.add_section(section_3)
        tl.add_section(section_4)
        tl.add_section(section_5)
    
        return tl    
    
    def create_time_line_poly_multi(self):
        section_0 = Section(0,1, [[1.5, 0], [0, 1]], n_entries=2, degree=3)
        section_1 = Section(1,2, [[0, 0], [1, 1]], n_entries=2, degree=3)
        section_2 = Section(2,3, [[0, 1], [1, 1]], n_entries=2, degree=3)
        section_3 = Section(3,4, [[1, 0], [0, 0]], n_entries=2, degree=3)
        section_4 = Section(4,5, [[2, 0], [0, 0]], n_entries=2, degree=3)
        section_5 = Section(5,6, [[2, -2], [0, 0, 1]], n_entries=2, degree=3)
    
        tl = TimeLine()
        tl.add_section(section_0)
        tl.add_section(section_1)
        tl.add_section(section_2)
        tl.add_section(section_3)
        tl.add_section(section_4)
        tl.add_section(section_5)
    
        return tl    
    
    def test_timeline_value(self):
        tl = self.create_time_line_constant_single()
        
        np.testing.assert_equal(tl.value(0), 1.5)
        np.testing.assert_equal(tl.value(1), 0.0)
        np.testing.assert_equal(tl.value(2), 0.0)
        np.testing.assert_equal(tl.value(3), 1.0)
        np.testing.assert_equal(tl.value(4), 2.0)
        np.testing.assert_equal(tl.value(5), 2.0)
        np.testing.assert_equal(tl.value(5.5), 2.0)
        np.testing.assert_equal(tl.value(6), 2.0)
        
        tl = self.create_time_line_constant_multi()
    
        np.testing.assert_equal(tl.value(0), [1.5, 0])
        np.testing.assert_equal(tl.value(1), [0.0, 0])
        np.testing.assert_equal(tl.value(2), [0.0, 1])
        np.testing.assert_equal(tl.value(3), [1.0, 0])
        np.testing.assert_equal(tl.value(4), [2.0, 0])
        np.testing.assert_equal(tl.value(5), [2.0, -2])
        np.testing.assert_equal(tl.value(5.5), [2.0, -2])
        np.testing.assert_equal(tl.value(6), [2.0, -2])        
    
        tl = self.create_time_line_poly_single()
        
        np.testing.assert_equal(tl.value(2), 0.0)
        np.testing.assert_equal(tl.value(2.5), 0.5)
        np.testing.assert_equal(tl.value(5), 2.0)
        np.testing.assert_equal(tl.value(5.5), 1.0)
        np.testing.assert_equal(tl.value(6), 0.0)
        
        tl = self.create_time_line_poly_multi()
        np.testing.assert_equal(tl.value(0), [1.5, 0])
        np.testing.assert_equal(tl.value(2.5), [0.5, 1.5])
        np.testing.assert_equal(tl.value(5), [2.0, 0])
        np.testing.assert_equal(tl.value(5.5), [1.0, 0.25])
        np.testing.assert_equal(tl.value(6), [0.0, 1])
        
    def test_timeline_integral(self):
        tl = self.create_time_line_poly_single()

        np.testing.assert_equal(tl.integral(0,0), 0.0)
        np.testing.assert_equal(tl.integral(0,0.5), 1.5/2)
        np.testing.assert_equal(tl.integral(0,1), 1.5)
        np.testing.assert_equal(tl.integral(0,2), 1.5)
        np.testing.assert_equal(tl.integral(2,2.5), 0.5/2/2)
        np.testing.assert_equal(tl.integral(2,3), 0.5)
        
        tl = self.create_time_line_poly_multi()
        
        np.testing.assert_equal(tl.integral(0,0), [0.0, 0.0])
        np.testing.assert_equal(tl.integral(0,0.5), [1.5/2, 0.125])
        np.testing.assert_equal(tl.integral(0,1), [1.5, 0.5])
        np.testing.assert_equal(tl.integral(0,2), [1.5, 2.0])
        np.testing.assert_equal(tl.integral(5,6), [1, 1/3])
        
    def test_timeline_coeff(self):
        tl = self.create_time_line_poly_multi()
        
        np.testing.assert_equal(
            tl.coefficients(0.0), [[1.5, 0, 0, 0], [0, 1, 0, 0]]
        )

        np.testing.assert_equal(
            tl.coefficients(5.5), [[1, -2, 0, 0], [0.25, 1, 1, 0]]
        )
        
    def test_section_times(self):
        tl = self.create_time_line_constant_single()
        
        self.assertEqual(tl.section_times, [0, 1, 2, 3, 4, 5, 6])
        

if __name__ == '__main__':
    unittest.main()