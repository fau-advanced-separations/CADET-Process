import unittest

import numpy as np
from CADETProcess.dynamicEvents import MultiTimeLine, Section, TimeLine
from CADETProcess.dynamicEvents.section import generate_indices


class TestGenerateIndices(unittest.TestCase):
    def test_generate_indices(self):
        shape = (3, 3)

        indices = [[0, 1], [1, 2]]
        indices_tuple_expected = [(0, 1), (1, 2)]
        indices_tuple = generate_indices(shape, indices)
        np.testing.assert_equal(indices_tuple, indices_tuple_expected)

        indices = np.s_[:]
        indices_tuple_expected = [(slice(None, None, None),)]
        indices_tuple = generate_indices(shape, indices)
        np.testing.assert_equal(indices_tuple, indices_tuple_expected)

        indices = np.s_[0, :]
        indices_tuple_expected = [(0, slice(None, None, None))]
        indices_tuple = generate_indices(shape, indices)
        np.testing.assert_equal(indices_tuple, indices_tuple_expected)

        indices = [np.s_[0, :], [1, 1]]
        indices_tuple_expected = [(0, slice(None, None, None)), (1, 1)]
        indices_tuple = generate_indices(shape, indices)
        np.testing.assert_equal(indices_tuple, indices_tuple_expected)

        with self.assertRaises(IndexError):
            _ = generate_indices(shape, indices=[3, 3])

        with self.assertRaises(ValueError):
            _ = generate_indices((), indices=[[0, 1], [1, 2]])


class TestSection(unittest.TestCase):
    def setUp(self):
        self.constant_section_single = Section(0, 1, 1)
        self.constant_section_multi = Section(1, 2, [1, 2])
        self.poly_section_single = Section(0, 1, [0, 1, 0, 0], is_polynomial=True)
        self.poly_section_multi = Section(0, 1, [[0, 1], [1, -1]], is_polynomial=True)

    def test_coeffs(self):
        np.testing.assert_equal(self.constant_section_single.coeffs, [[1]])
        np.testing.assert_equal(self.constant_section_multi.coeffs, [[1], [2]])
        np.testing.assert_equal(self.poly_section_single.coeffs, [[0, 1, 0, 0]])
        np.testing.assert_equal(self.poly_section_multi.coeffs, [[0, 1], [1, -1]])

    def test_section_value(self):
        const_single = self.constant_section_single
        self.assertEqual(const_single.value(0), 1)

        # Exceed section times
        with self.assertRaises(ValueError):
            val = const_single.value(2)

        const_multi = self.constant_section_multi
        np.testing.assert_equal(const_multi.value(1), [1, 2])

        # Exceed section times
        with self.assertRaises(ValueError):
            val = const_multi.value(0)

        poly_single = self.poly_section_single
        np.testing.assert_equal(poly_single.value(0), 0)
        np.testing.assert_equal(poly_single.value(0.5), 0.5)
        np.testing.assert_equal(poly_single.value(1), 1)

        poly_multi = self.poly_section_multi
        np.testing.assert_equal(poly_multi.value(0), [0, 1])
        np.testing.assert_equal(poly_multi.value(0.5), [0.5, 0.5])
        np.testing.assert_equal(poly_multi.value(1), [1, 0])

    def test_section_integral(self):
        const_single = self.constant_section_single
        self.assertEqual(const_single.integral(0, 0), 0)
        self.assertEqual(const_single.integral(0, 1), 1)

        # Exceed section times
        with self.assertRaises(ValueError):
            val = const_single.value(2)

        const_multi = self.constant_section_multi
        np.testing.assert_equal(const_multi.integral(1, 2), [1, 2])

        # Exceed section times
        with self.assertRaises(ValueError):
            val = const_multi.value(0)

        poly_single = self.poly_section_single
        np.testing.assert_equal(poly_single.integral(0, 0.5), 0.125)
        np.testing.assert_equal(poly_single.integral(0, 1), 0.5)

        poly_multi = self.poly_section_multi
        np.testing.assert_equal(poly_multi.integral(0, 0.5), [0.125, 0.375])
        np.testing.assert_equal(poly_multi.integral(0.5, 1), [0.375, 0.125])
        np.testing.assert_equal(poly_multi.integral(0, 1), [0.5, 0.5])


class TestTimeLine(unittest.TestCase):
    def create_timeline_constant_single(self):
        """Piecewise constant sections with single entry."""
        section_0 = Section(0, 1, 1.5)
        section_1 = Section(1, 2, 0)
        section_2 = Section(2, 3, 0)
        section_3 = Section(3, 4, 1)
        section_4 = Section(4, 5, 2)
        section_5 = Section(5, 6, 2)

        tl = TimeLine()
        tl.add_section(section_0)
        tl.add_section(section_1)
        tl.add_section(section_2)
        tl.add_section(section_3)
        tl.add_section(section_4)
        tl.add_section(section_5)

        return tl

    def create_timeline_constant_multi(self):
        """Piecewise constant sections with multiple entries (differing in value)."""
        section_0 = Section(0, 1, (1.5, 0))
        section_1 = Section(1, 2, (0, 0))
        section_2 = Section(2, 3, (0, 1))
        section_3 = Section(3, 4, (1, 0))
        section_4 = Section(4, 5, (2, 0))
        section_5 = Section(5, 6, (2, -2))

        tl = TimeLine()
        tl.add_section(section_0)
        tl.add_section(section_1)
        tl.add_section(section_2)
        tl.add_section(section_3)
        tl.add_section(section_4)
        tl.add_section(section_5)

        return tl

    def create_timeline_poly_single(self):
        """Polynomial sections with single entry."""
        section_0 = Section(0, 1, (1.5, 0), is_polynomial=True)
        section_1 = Section(1, 2, (0, 0), is_polynomial=True)
        section_2 = Section(2, 3, (0, 1), is_polynomial=True)
        section_3 = Section(3, 4, (1, 0), is_polynomial=True)
        section_4 = Section(4, 5, (2, 0), is_polynomial=True)
        section_5 = Section(5, 6, (2, -2), is_polynomial=True)

        tl = TimeLine()
        tl.add_section(section_0)
        tl.add_section(section_1)
        tl.add_section(section_2)
        tl.add_section(section_3)
        tl.add_section(section_4)
        tl.add_section(section_5)

        return tl

    def create_timeline_poly_multi(self):
        """Polynomial sections with multiple entries (differing in value)."""
        section_0 = Section(0, 1, [[1.5, 0, 0], [0, 1, 0]], is_polynomial=True)
        section_1 = Section(1, 2, [[0, 0, 0], [1, 1, 0]], is_polynomial=True)
        section_2 = Section(2, 3, [[0, 1, 0], [1, 1, 0]], is_polynomial=True)
        section_3 = Section(3, 4, [[1, 0, 0], [0, 0, 0]], is_polynomial=True)
        section_4 = Section(4, 5, [[2, 0, 0], [0, 0, 0]], is_polynomial=True)
        section_5 = Section(5, 6, [[2, -2, 0], [0, 0, 1]], is_polynomial=True)

        tl = TimeLine()
        tl.add_section(section_0)
        tl.add_section(section_1)
        tl.add_section(section_2)
        tl.add_section(section_3)
        tl.add_section(section_4)
        tl.add_section(section_5)

        return tl

    def test_timeline_value(self):
        """Test values at specific times."""
        tl = self.create_timeline_constant_single()

        np.testing.assert_equal(tl.value(0), 1.5)
        np.testing.assert_equal(tl.value(1), 0.0)
        np.testing.assert_equal(tl.value(2), 0.0)
        np.testing.assert_equal(tl.value(3), 1.0)
        np.testing.assert_equal(tl.value(4), 2.0)
        np.testing.assert_equal(tl.value(5), 2.0)
        np.testing.assert_equal(tl.value(5.5), 2.0)
        np.testing.assert_equal(tl.value(6), 2.0)

        tl = self.create_timeline_constant_multi()

        np.testing.assert_equal(tl.value(0), [1.5, 0])
        np.testing.assert_equal(tl.value(1), [0.0, 0])
        np.testing.assert_equal(tl.value(2), [0.0, 1])
        np.testing.assert_equal(tl.value(3), [1.0, 0])
        np.testing.assert_equal(tl.value(4), [2.0, 0])
        np.testing.assert_equal(tl.value(5), [2.0, -2])
        np.testing.assert_equal(tl.value(5.5), [2.0, -2])
        np.testing.assert_equal(tl.value(6), [2.0, -2])

        tl = self.create_timeline_poly_single()

        np.testing.assert_equal(tl.value(2), 0.0)
        np.testing.assert_equal(tl.value(2.5), 0.5)
        np.testing.assert_equal(tl.value(5), 2.0)
        np.testing.assert_equal(tl.value(5.5), 1.0)
        np.testing.assert_equal(tl.value(6), 0.0)

        tl = self.create_timeline_poly_multi()
        np.testing.assert_equal(tl.value(0), [1.5, 0])
        np.testing.assert_equal(tl.value(2.5), [0.5, 1.5])
        np.testing.assert_equal(tl.value(5), [2.0, 0])
        np.testing.assert_equal(tl.value(5.5), [1.0, 0.25])
        np.testing.assert_equal(tl.value(6), [0.0, 1])

    def test_timeline_integral(self):
        """Test definite integral values for specific time intervals."""
        tl = self.create_timeline_poly_single()

        np.testing.assert_equal(tl.integral(0, 0), 0.0)
        np.testing.assert_equal(tl.integral(0, 0.5), 1.5 / 2)
        np.testing.assert_equal(tl.integral(0, 1), 1.5)
        np.testing.assert_equal(tl.integral(0, 2), 1.5)
        np.testing.assert_equal(tl.integral(2, 2.5), 0.5 / 2 / 2)
        np.testing.assert_equal(tl.integral(2, 3), 0.5)

        tl = self.create_timeline_poly_multi()

        np.testing.assert_equal(tl.integral(0, 0), [0.0, 0.0])
        np.testing.assert_equal(tl.integral(0, 0.5), [1.5 / 2, 0.125])
        np.testing.assert_equal(tl.integral(0, 1), [1.5, 0.5])
        np.testing.assert_equal(tl.integral(0, 2), [1.5, 2.0])
        np.testing.assert_equal(tl.integral(5, 6), [1, 1 / 3])

    def test_timeline_coeff(self):
        """Test coefficient values at given times."""
        tl = self.create_timeline_poly_single()
        np.testing.assert_equal(tl.coefficients(0.0), [1.5, 0])
        np.testing.assert_equal(tl.coefficients(5.5), [1, -2])

        tl = self.create_timeline_poly_multi()

        np.testing.assert_equal(tl.coefficients(0.0), [[1.5, 0, 0], [0, 1, 0]])

        np.testing.assert_equal(tl.coefficients(5.5), [[1, -2, 0], [0.25, 1, 1]])

    def test_section_times(self):
        """Test section times."""
        tl = self.create_timeline_constant_single()

        self.assertEqual(tl.section_times, [0, 1, 2, 3, 4, 5, 6])

    def test_tl_from_profile(self):
        """Test creation of time line from time series profile."""
        time = np.linspace(0, 100, 1001)
        y = np.sin(time / 10)

        tl = TimeLine.from_profile(time, y)
        np.testing.assert_almost_equal(tl.value(time)[:, 0], y, decimal=3)


class TestMultiTimeLine(unittest.TestCase):
    def create_timeline_constant_multi(self):
        """Piecewise constant sections with multiple entries managed by MultiTimeline."""
        section_0_0 = Section(0, 1, 1.5)
        section_0_1 = Section(1, 3, 0)
        section_0_2 = Section(3, 4, 1)
        section_0_3 = Section(4, 6, 2)

        section_1_0 = Section(0, 2, 0)
        section_1_1 = Section(2, 3, 1)
        section_1_2 = Section(3, 5, 0)
        section_1_3 = Section(5, 6, -2)

        tl = MultiTimeLine([0, 0])
        tl.add_section(section_0_0, 0)
        tl.add_section(section_0_1, 0)
        tl.add_section(section_0_2, 0)
        tl.add_section(section_0_3, 0)

        tl.add_section(section_1_0, 1)
        tl.add_section(section_1_1, 1)
        tl.add_section(section_1_2, 1)
        tl.add_section(section_1_3, 1)

        return tl

    def create_timeline_poly_multi(self):
        """Polynomial sections with multiple entries managed by MultiTimeline."""
        # Entry 0
        # Const. Coeff.
        section_0_0_0 = Section(0, 1, 1.5)
        section_0_0_1 = Section(1, 3, 0)
        section_0_0_2 = Section(3, 4, 1)
        section_0_0_3 = Section(4, 6, 2)

        # Lin. Coeff.
        section_0_1_0 = Section(0, 2, 0)
        section_0_1_1 = Section(2, 3, 1)
        section_0_1_2 = Section(3, 5, 0)
        section_0_1_3 = Section(5, 6, -2)

        # Entry 1
        # Const. Coeff.
        section_1_0_0 = Section(0, 1, 0)
        section_1_0_1 = Section(1, 3, 1)
        section_1_0_2 = Section(3, 6, 0)

        # Lin. Coeff.
        section_1_1_0 = Section(0, 3, 1)
        section_1_1_1 = Section(3, 6, 0)

        # Cubic Coeff
        section_1_2_0 = Section(0, 5, 0)
        section_1_2_1 = Section(5, 6, 1)

        tl = MultiTimeLine(base_state=[[0, 0, 0, 0], [0, 0, 0, 0]], is_polynomial=True)
        tl.add_section(section_0_0_0, entry_index=(0, 0))
        tl.add_section(section_0_0_1, entry_index=(0, 0))
        tl.add_section(section_0_0_2, entry_index=(0, 0))
        tl.add_section(section_0_0_3, entry_index=(0, 0))

        tl.add_section(section_0_1_0, entry_index=(0, 1))
        tl.add_section(section_0_1_1, entry_index=(0, 1))
        tl.add_section(section_0_1_2, entry_index=(0, 1))
        tl.add_section(section_0_1_3, entry_index=(0, 1))

        tl.add_section(section_1_0_0, entry_index=(1, 0))
        tl.add_section(section_1_0_1, entry_index=(1, 0))
        tl.add_section(section_1_0_2, entry_index=(1, 0))

        tl.add_section(section_1_1_0, entry_index=(1, 1))
        tl.add_section(section_1_1_1, entry_index=(1, 1))

        tl.add_section(section_1_2_0, entry_index=(1, 2))
        tl.add_section(section_1_2_1, entry_index=(1, 2))

        return tl

    # TODO: Test when one base state is not modified
    # TODO: Test exceeding index
    # TODO: Test missing sections

    def test_multi_timeline_value(self):
        multi_tl = self.create_timeline_constant_multi()
        tl = multi_tl.combined_time_line

        np.testing.assert_equal(tl.value(0), [1.5, 0])
        np.testing.assert_equal(tl.value(1), [0.0, 0])
        np.testing.assert_equal(tl.value(2), [0.0, 1])
        np.testing.assert_equal(tl.value(3), [1.0, 0])
        np.testing.assert_equal(tl.value(4), [2.0, 0])
        np.testing.assert_equal(tl.value(5), [2.0, -2])
        np.testing.assert_equal(tl.value(5.5), [2.0, -2])
        np.testing.assert_equal(tl.value(6), [2.0, -2])

        multi_tl = self.create_timeline_poly_multi()
        tl = multi_tl.combined_time_line

        np.testing.assert_equal(tl.value(0), [1.5, 0])
        np.testing.assert_equal(tl.value(2.5), [0.5, 2.5])
        np.testing.assert_equal(tl.value(5), [2.0, 0])
        np.testing.assert_equal(tl.value(5.5), [1.0, 0.25])
        np.testing.assert_equal(tl.value(6), [0.0, 1])

    def test_timeline_integral(self):
        """Test definite integral values for specific time intervals."""
        multi_tl = self.create_timeline_poly_multi()
        tl = multi_tl.combined_time_line

        np.testing.assert_equal(tl.integral(0, 0), [0.0, 0.0])
        np.testing.assert_equal(tl.integral(0, 0.5), [1.5 / 2, 0.125])
        np.testing.assert_equal(tl.integral(0, 1), [1.5, 0.5])
        np.testing.assert_equal(tl.integral(0, 2), [1.5, 2.0])
        np.testing.assert_equal(tl.integral(5, 6), [1, 1 / 3])

    def test_timeline_coeff(self):
        """Test coefficient values at given times."""
        multi_tl = self.create_timeline_poly_multi()
        tl = multi_tl.combined_time_line

        np.testing.assert_equal(tl.coefficients(0.0), [[1.5, 0, 0, 0], [0, 1, 0, 0]])
        np.testing.assert_equal(tl.coefficients(5.5), [[1, -2, 0, 0], [0.25, 1, 1, 0]])

    def test_section_times(self):
        """Test section times."""
        tl = self.create_timeline_constant_multi()

        self.assertEqual(tl.section_times, [0, 1, 2, 3, 4, 5, 6])

    def test_multi_timeline(self):
        base_state = [1, 2, 3]
        multi_tl_list = MultiTimeLine(base_state)
        # multi_tl_list.combined_time_line

        base_state = [[1, 2, 3], [4, 5, 6]]
        multi_tl_array = MultiTimeLine(base_state)
        # multi_tl_array.combined_time_line

        base_state = [1, 2, 3]
        multi_tl_list_poly = MultiTimeLine(base_state, is_polynomial=True)
        # multi_tl_list_poly.combined_time_line

        base_state = [[1, 2, 3], [4, 5, 6]]
        multi_tl_poly = MultiTimeLine(base_state, is_polynomial=True)
        # multi_tl_poly.combined_time_line

    def test_combined_timeline(self):
        pass


if __name__ == "__main__":
    unittest.main()
