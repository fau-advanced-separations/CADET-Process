"""
..todo::
    add/remove events (check component indices)
    add/remove durations
    add/remove dependencies

- event times (especially considerung time modulo cycle time)
- event /parameter/ performer timelines
- section states (especially piecewise poly times)

Notes
-----
Since the EventHandler defines an interface, that requires the
implementation of some methods, a HandlerFixture class is defined.

Maybe this is too complicated, just use Process instead?
"""

import unittest

import CADETProcess
import numpy as np
from addict import Dict
from CADETProcess.dataStructure import (
    Float,
    NdPolynomial,
    Polynomial,
    SizedList,
    SizedNdArray,
    SizedTuple,
    Structure,
    Switch,
)

plot = True


class PerformerFixture(Structure):
    n_entries = 4
    n_coeff = 4

    scalar_float = Float(default=0)
    switch = Switch(valid=[-1, 1], default=1)
    sized_tuple = SizedTuple(size=2, default=(1, 1))
    array_1d = SizedList(size=4, default=0)
    array_1d_single = SizedList(size=1, default=0)
    ndarray = SizedNdArray(size=(2, 4), default=0)
    ndarray_no_default = SizedNdArray(size=(2, 4))
    array_1d_poly = Polynomial(n_coeff=4, default=0)
    ndarray_poly = NdPolynomial(n_entries=2, n_coeff=4, default=0)
    ndarray_poly_dep = NdPolynomial(size=("n_entries", "n_coeff"), default=0)

    _parameters = [
        "scalar_float",
        "sized_tuple",
        "switch",
        "array_1d",
        "array_1d_single",
        "ndarray",
        "ndarray_no_default",
        "array_1d_poly",
        "ndarray_poly",
        "ndarray_poly_dep",
    ]
    _section_dependent_parameters = [
        "scalar_float",
        "sized_tuple",
        "array_1d",
        "array_1d_single",
        "ndarray",
        "ndarray_no_default",
        "array_1d_poly",
        "ndarray_poly",
        "ndarray_poly_dep",
    ]

    @property
    def section_dependent_parameters(self):
        parameters = {
            param: getattr(self, param) for param in self._section_dependent_parameters
        }
        return parameters


class HandlerFixture(CADETProcess.dynamicEvents.EventHandler):
    def __init__(self):
        self.name = None
        self.performer = PerformerFixture()
        super().__init__()

    @property
    def parameters(self):
        parameters = super().parameters

        parameters["performer"] = self.performer.parameters

        return Dict(parameters)

    @parameters.setter
    def parameters(self, parameters):
        try:
            self.performer.parameters = parameters.pop("performer")
        except KeyError:
            pass

        super(HandlerFixture, self.__class__).parameters.fset(self, parameters)

    @property
    def section_dependent_parameters(self):
        parameters = Dict()
        parameters["performer"] = self.performer.section_dependent_parameters
        return parameters

    @property
    def polynomial_parameters(self):
        parameters = Dict()
        parameters["performer"] = self.performer.polynomial_parameters
        return parameters


class Test_Events(unittest.TestCase):
    def setup_event_handler(self, add_events=False):
        event_handler = HandlerFixture()
        event_handler.cycle_time = 20

        if add_events:
            evt = event_handler.add_event("evt0", "performer.scalar_float", 0)
            evt = event_handler.add_event("evt1", "performer.scalar_float", 1)
            evt = event_handler.add_event("evt2", "performer.sized_tuple", (2, 1))
            evt = event_handler.add_event("evt3", "performer.sized_tuple", (3, 2))
            evt = event_handler.add_event("evt4", "performer.sized_tuple", (3, 3))

        return event_handler

    def setUp(self):
        self.event_handler = self.setup_event_handler()

    def test_exceptions(self):
        event_handler = self.setup_event_handler()

        # Invalid path
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event("wrong_path", "performer.wrong", 1)

        # Invalid state
        with self.assertRaises(TypeError):
            event_handler.add_event("wrong_value", "performer.scalar_float", "wrong")

        # Duplicate name
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event("duplicate", "performer.scalar_float", 1)
            event_handler.add_event("duplicate", "performer.scalar_float", 1)

        # Not section dependent
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event("not_sec_dependent", "performer.switch", 1)

    def test_event_scalar(self):
        event_handler = self.event_handler

        evt = event_handler.add_event("trivial", "performer.scalar_float", 1, time=0)
        self.assertEqual(evt.state, 1)
        self.assertEqual(evt.full_state, 1)
        self.assertEqual(evt.n_entries, 1)
        self.assertEqual(evt.indices, None)
        self.assertEqual(evt.full_indices, [])
        self.assertEqual(evt.n_indices, 0)
        self.assertEqual(event_handler.performer.scalar_float, 1)

        # Raise Error for indices
        with self.assertRaises(IndexError):
            evt = event_handler.add_event(
                "param_has_no_indices", "performer.scalar_float", 1, time=0, indices=1
            )

    def test_event_array_1d_single(self):
        """
        Extra test for array with single entry.

        See also: https://github.com/fau-advanced-separations/CADET-Process/pull/68
        """
        event_handler = self.event_handler

        # No index
        evt = event_handler.add_event("trivial", "performer.array_1d_single", 1, time=0)
        self.assertEqual(evt.state, 1)
        self.assertEqual(evt.full_state, [1])
        self.assertEqual(evt.n_entries, 1)
        self.assertEqual(evt.indices, [(slice(None, None, None),)])
        self.assertEqual(evt.full_indices, [(0,)])
        self.assertEqual(evt.n_indices, 1)
        self.assertEqual(event_handler.performer.array_1d_single, [1])

        # Explicit Index
        evt = event_handler.add_event(
            "trivial_1", "performer.array_1d_single", 2, time=0, indices=0
        )
        self.assertEqual(evt.state, 2)
        self.assertEqual(evt.full_state, [2])
        self.assertEqual(evt.n_entries, 1)
        self.assertEqual(evt.indices, [(0,)])
        self.assertEqual(evt.full_indices, [(0,)])
        self.assertEqual(evt.n_indices, 1)
        self.assertEqual(event_handler.performer.array_1d_single, [2])

        # Raise Error for exceeding indices
        with self.assertRaises(IndexError):
            evt = event_handler.add_event(
                "param_has_no_indices",
                "performer.array_1d_single",
                1,
                time=0,
                indices=1,
            )

    def test_event_1D(self):
        # TODO: Check too many / few indices
        event_handler = self.event_handler

        # Add event for single entry in 1D list / array
        event_handler.performer.array_1d
        evt = event_handler.add_event(
            "1D_single", "performer.array_1d", 1, indices=0, time=0
        )
        self.assertEqual(evt.state, 1)
        self.assertEqual(evt.full_state, [1])
        self.assertEqual(evt.n_entries, 1)
        self.assertEqual(evt.indices, [(0,)])
        self.assertEqual(evt.full_indices, [(0,)])
        self.assertEqual(evt.n_indices, 1)
        np.testing.assert_equal(event_handler.performer.array_1d, [1, 0, 0, 0])

        # Add event for multiple entries in 1D list / array
        # TODO: When creating section, missing parameters must be read at that time!
        evt = event_handler.add_event(
            "1D_multi", "performer.array_1d", [0, 1], indices=[0, 1], time=1
        )
        np.testing.assert_equal(evt.state, [0, 1])
        self.assertEqual(evt.full_state, [0, 1])
        self.assertEqual(evt.n_entries, 2)
        np.testing.assert_equal(evt.indices, [(0,), (1,)])
        self.assertEqual(evt.full_indices, [(0,), (1,)])
        self.assertEqual(evt.n_indices, 2)
        np.testing.assert_equal(event_handler.performer.array_1d, [0, 1, 0, 0])

        # Add event for multiple entries in 1D list / array with custom order
        evt = event_handler.add_event(
            "1D_multi_order", "performer.array_1d", [2, 3], indices=[1, 2], time=2
        )
        np.testing.assert_equal(evt.state, [2, 3])
        np.testing.assert_equal(evt.full_state, [2, 3])
        self.assertEqual(evt.n_entries, 2)
        np.testing.assert_equal(evt.indices, [(1,), (2,)])
        np.testing.assert_equal(evt.full_indices, [(1,), (2,)])
        self.assertEqual(evt.n_indices, 2)
        np.testing.assert_equal(event_handler.performer.array_1d, [0, 2, 3, 0])

        # Add event for all entries in 1D list / array
        evt = event_handler.add_event(
            "1D_all", "performer.array_1d", [0, 0, 1], indices=[0, 1, 2], time=3
        )
        np.testing.assert_equal(evt.state, [0, 0, 1])
        np.testing.assert_equal(evt.full_state, [0, 0, 1])
        self.assertEqual(evt.n_entries, 3)
        np.testing.assert_equal(evt.indices, [(0,), (1,), (2,)])
        np.testing.assert_equal(evt.full_indices, [(0,), (1,), (2,)])
        self.assertEqual(evt.n_indices, 3)
        np.testing.assert_equal(event_handler.performer.array_1d, [0, 0, 1, 0])

        # Set all values without indices
        evt = event_handler.add_event(
            "1D_all_slice_all_no_indices", "performer.array_1d", [0, 1, 2, 3], time=4
        )
        np.testing.assert_equal(evt.state, [0, 1, 2, 3])
        np.testing.assert_equal(evt.full_state, [0, 1, 2, 3])
        self.assertEqual(evt.n_entries, 4)
        np.testing.assert_equal(evt.indices, [(slice(None, None, None),)])
        np.testing.assert_equal(evt.full_indices, [(0,), (1,), (2,), (3,)])
        self.assertEqual(evt.n_indices, 4)
        np.testing.assert_equal(event_handler.performer.array_1d, [0, 1, 2, 3])

        # Set all values using slicing notation.
        evt = event_handler.add_event(
            "1D_all_slice", "performer.array_1d", [2, 3, 4, 5], indices=np.s_[:], time=5
        )
        np.testing.assert_equal(evt.state, [2, 3, 4, 5])
        np.testing.assert_equal(evt.full_state, [2, 3, 4, 5])
        self.assertEqual(evt.n_entries, 4)
        np.testing.assert_equal(evt.indices, [(slice(None, None, None),)])
        np.testing.assert_equal(evt.full_indices, [(0,), (1,), (2,), (3,)])
        self.assertEqual(evt.n_indices, 4)
        np.testing.assert_equal(event_handler.performer.array_1d, [2, 3, 4, 5])

        # Set all values to one value using slicing notation
        evt = event_handler.add_event(
            "1D_all_slice_one_value", "performer.array_1d", 1, indices=np.s_[:], time=6
        )
        np.testing.assert_equal(evt.state, 1)
        np.testing.assert_equal(evt.full_state, [1, 1, 1, 1])
        self.assertEqual(evt.n_entries, 4)
        np.testing.assert_equal(evt.indices, [(slice(None, None, None),)])
        np.testing.assert_equal(evt.full_indices, [(0,), (1,), (2,), (3,)])
        self.assertEqual(evt.n_indices, 4)
        np.testing.assert_equal(event_handler.performer.array_1d, [1, 1, 1, 1])

        # Set all values from starting_index to one value using slice
        evt = event_handler.add_event(
            "1D_partial_slice", "performer.array_1d", 2, indices=np.s_[1:], time=7
        )
        np.testing.assert_equal(evt.state, 2)
        np.testing.assert_equal(evt.full_state, [2, 2, 2])
        self.assertEqual(evt.n_entries, 3)
        np.testing.assert_equal(evt.indices, [(slice(1, None, None),)])
        np.testing.assert_equal(evt.full_indices, [(1,), (2,), (3,)])
        self.assertEqual(evt.n_indices, 3)
        np.testing.assert_equal(event_handler.performer.array_1d, [1, 2, 2, 2])

        if plot:
            event_handler.plot_events()

        # Number of indices
        with self.assertRaises(ValueError):
            event_handler.add_event(
                "not_enough_entries", "performer.array_1d", [1, 2], indices=[2]
            )
        with self.assertRaises(ValueError):
            event_handler.add_event(
                "too_many_indices", "performer.array_1d", [1, 2], indices=[1, 2, 2]
            )

        # Index exceeds shape
        with self.assertRaises(IndexError):
            event_handler.add_event(
                "index_exceeds_shape", "performer.array_1d", [1, 2], indices=[1, 4]
            )

        # Duplicate entries for indices
        with self.assertRaises(ValueError):
            event_handler.add_event(
                "duplicate_entries", "performer.array_1d", [1, 2], indices=[1, 1]
            )

    def test_event_ndarray(self):
        event_handler = self.event_handler

        t = 0

        # Add event for single entry in ndarray
        t += 1
        evt = event_handler.add_event(
            "ndarray_single", "performer.ndarray", 1, indices=(0, 0), time=t
        )
        np.testing.assert_equal(evt.state, 1)
        self.assertEqual(evt.n_entries, 1)
        np.testing.assert_equal(evt.indices, [(0, 0)])
        np.testing.assert_equal(evt.full_indices, [(0, 0)])
        self.assertEqual(evt.n_indices, 1)
        np.testing.assert_equal(
            event_handler.performer.ndarray, [[1, 0, 0, 0], [0, 0, 0, 0]]
        )

        # Add event for multiple entries in array
        t += 1
        evt = event_handler.add_event(
            "ndarray_multiple",
            "performer.ndarray",
            [0, 1],
            indices=[(0, 0), (0, 1)],
            time=t,
        )
        np.testing.assert_equal(evt.state, [0, 1])
        np.testing.assert_equal(evt.full_state, [0, 1])
        self.assertEqual(evt.n_entries, 2)
        np.testing.assert_equal(evt.indices, [(0, 0), (0, 1)])
        np.testing.assert_equal(evt.full_indices, [(0, 0), (0, 1)])
        self.assertEqual(evt.n_indices, 2)
        np.testing.assert_equal(
            event_handler.performer.ndarray, [[0, 1, 0, 0], [0, 0, 0, 0]]
        )

        # Add event for multiple entries in array with custom order
        t += 1
        evt = event_handler.add_event(
            "ndarray_multiple_order",
            "performer.ndarray",
            [1, 0],
            indices=[(1, 0), (0, 1)],
            time=t,
        )
        np.testing.assert_equal(evt.state, [1, 0])
        np.testing.assert_equal(evt.full_state, [1, 0])
        self.assertEqual(evt.n_entries, 2)
        np.testing.assert_equal(evt.indices, [(1, 0), (0, 1)])
        np.testing.assert_equal(evt.full_indices, [(1, 0), (0, 1)])
        self.assertEqual(evt.n_indices, 2)
        np.testing.assert_equal(
            event_handler.performer.ndarray, [[0, 0, 0, 0], [1, 0, 0, 0]]
        )

        # Add event for all entries in array
        t += 1
        evt = event_handler.add_event(
            "ndarray_all",
            "performer.ndarray",
            [0, 1, 2, 3, 4, 5, 6, 7],
            indices=[(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
            time=t,
        )
        np.testing.assert_equal(evt.state, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(evt.n_entries, 8)
        np.testing.assert_equal(
            evt.indices,
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        )
        np.testing.assert_equal(
            evt.full_indices,
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        )
        self.assertEqual(evt.n_indices, 8)
        np.testing.assert_equal(
            event_handler.performer.ndarray, [[0, 1, 2, 3], [4, 5, 6, 7]]
        )

        # Set all values without indices
        t += 1
        evt = event_handler.add_event(
            "ndarray_all_no_indices",
            "performer.ndarray",
            [[8, 7, 6, 5], [4, 3, 2, 1]],
            time=t,
        )
        np.testing.assert_equal(evt.state, [[8, 7, 6, 5], [4, 3, 2, 1]])
        np.testing.assert_equal(evt.full_state, [8, 7, 6, 5, 4, 3, 2, 1])
        self.assertEqual(evt.n_entries, 8)
        np.testing.assert_equal(evt.indices, [(slice(None, None, None),)])
        np.testing.assert_equal(
            evt.full_indices,
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        )
        self.assertEqual(evt.n_indices, 8)
        np.testing.assert_equal(
            event_handler.performer.ndarray, [[8, 7, 6, 5], [4, 3, 2, 1]]
        )

        # Set all values using slicing notation.
        t += 1
        evt = event_handler.add_event(
            "ndarray_all_slice",
            "performer.ndarray",
            [[0, 0, 0, 0], [1, 1, 1, 1]],
            indices=np.s_[:],
            time=t,
        )
        np.testing.assert_equal(evt.state, [[0, 0, 0, 0], [1, 1, 1, 1]])
        np.testing.assert_equal(evt.full_state, [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(evt.n_entries, 8)
        np.testing.assert_equal(evt.indices, [(slice(None, None, None),)])
        np.testing.assert_equal(
            evt.full_indices,
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        )
        self.assertEqual(evt.n_indices, 8)
        np.testing.assert_equal(
            event_handler.performer.ndarray, [[0, 0, 0, 0], [1, 1, 1, 1]]
        )

        # Set all values to one value using slicing notation
        t += 1
        evt = event_handler.add_event(
            "ndarray_all_slice_one_value",
            "performer.ndarray",
            1,
            indices=np.s_[:],
            time=t,
        )
        np.testing.assert_equal(evt.state, 1)
        np.testing.assert_equal(evt.full_state, [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(evt.n_entries, 8)
        np.testing.assert_equal(evt.indices, [(slice(None, None, None),)])
        np.testing.assert_equal(
            evt.full_indices,
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        )
        self.assertEqual(evt.n_indices, 8)
        np.testing.assert_equal(
            event_handler.performer.ndarray, [[1, 1, 1, 1], [1, 1, 1, 1]]
        )

        # TODO: Check slicing for rows/columns
        # Set all values of one row using slicing notation
        t += 1
        evt = event_handler.add_event(
            "ndarray_row_slice",
            "performer.ndarray",
            [0, 0, 0, 0],
            indices=np.s_[0, :],
            time=t,
        )
        np.testing.assert_equal(evt.state, [0, 0, 0, 0])
        np.testing.assert_equal(evt.full_state, [0, 0, 0, 0])
        self.assertEqual(evt.n_entries, 4)
        np.testing.assert_equal(
            evt.full_indices,
            [(0, 0), (0, 1), (0, 2), (0, 3)],
        )
        self.assertEqual(evt.n_indices, 4)
        np.testing.assert_equal(
            event_handler.performer.ndarray, [[0, 0, 0, 0], [1, 1, 1, 1]]
        )

        # Set all values of one row to one value using slicing notation
        t += 1
        evt = event_handler.add_event(
            "ndarray_row_slice_one_value",
            "performer.ndarray",
            2,
            indices=np.s_[0, :],
            time=t,
        )
        np.testing.assert_equal(evt.state, 2)
        np.testing.assert_equal(evt.full_state, [2, 2, 2, 2])
        self.assertEqual(evt.n_entries, 4)
        np.testing.assert_equal(
            evt.indices,
            [(0, slice(None, None, None))],
        )
        np.testing.assert_equal(
            evt.full_indices,
            [(0, 0), (0, 1), (0, 2), (0, 3)],
        )
        self.assertEqual(evt.n_indices, 4)
        np.testing.assert_equal(
            event_handler.performer.ndarray,
            [[2, 2, 2, 2], [1, 1, 1, 1]],
        )

        # Set all values of one column from starting_index to one value using slice
        t += 1
        evt = event_handler.add_event(
            "ndarray_column_partial_slice",
            "performer.ndarray",
            3,
            indices=np.s_[1:, 1],
            time=t,
        )
        np.testing.assert_equal(evt.state, 3)
        np.testing.assert_equal(evt.full_state, [3])
        self.assertEqual(evt.n_entries, 1)
        np.testing.assert_equal(evt.indices, [(slice(1, None, None), 1)])
        np.testing.assert_equal(evt.full_indices, [(1, 1)])
        self.assertEqual(evt.n_indices, 1)
        np.testing.assert_equal(
            event_handler.performer.ndarray,
            [[2, 2, 2, 2], [1, 3, 1, 1]],
        )

        # Check multiple slices
        t += 1
        evt = event_handler.add_event(
            "ndarray_multiple_slices",
            "performer.ndarray",
            [4, 5],
            indices=[np.s_[0, 1:], np.s_[1, :]],
            time=t,
        )
        np.testing.assert_equal(evt.state, [4, 5])
        np.testing.assert_equal(evt.full_state, [4, 4, 4, 5, 5, 5, 5])
        self.assertEqual(evt.n_entries, 7)
        np.testing.assert_equal(
            evt.indices,
            [(0, slice(1, None, None)), (1, slice(None, None, None))],
        )
        np.testing.assert_equal(
            evt.full_indices,
            [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        )
        self.assertEqual(evt.n_indices, 7)
        np.testing.assert_equal(
            event_handler.performer.ndarray,
            [[2, 4, 4, 4], [5, 5, 5, 5]],
        )

        if plot:
            event_handler.plot_events()

    def test_event_polynomial_1D(self):
        event_handler = self.event_handler

        # Add event relying on filling missing coefficients
        evt = event_handler.add_event(
            "1D_single_fill", "performer.array_1d_poly", 1, time=0
        )
        np.testing.assert_equal(evt.state, 1)
        np.testing.assert_equal(evt.full_state, [1, 0, 0, 0])
        self.assertEqual(evt.n_entries, 4)
        np.testing.assert_equal(evt.indices, [(slice(None, None, None),)])
        np.testing.assert_equal(evt.full_indices, [(0,), (1,), (2,), (3,)])
        self.assertEqual(evt.n_indices, 4)
        np.testing.assert_equal(
            event_handler.performer.array_1d_poly,
            [1, 0, 0, 0],
        )

        # Fill all values to specific value
        evt = event_handler.add_event(
            "1D_multi_no_fill", "performer.array_1d_poly", 2, indices=np.s_[:], time=1
        )
        np.testing.assert_equal(evt.state, 2)
        np.testing.assert_equal(evt.full_state, [2, 2, 2, 2])
        self.assertEqual(evt.n_entries, 4)
        np.testing.assert_equal(evt.indices, [(slice(None, None, None),)])
        np.testing.assert_equal(evt.full_indices, [(0,), (1,), (2,), (3,)])
        self.assertEqual(evt.n_indices, 4)
        np.testing.assert_equal(event_handler.performer.array_1d_poly, [2, 2, 2, 2])

        # Add event relying on filling missing coefficients
        evt = event_handler.add_event(
            "1D_multi_fill", "performer.array_1d_poly", [3, 3], time=2
        )
        np.testing.assert_equal(evt.state, [3, 3])
        np.testing.assert_equal(evt.full_state, [3, 3, 0, 0])
        self.assertEqual(evt.n_entries, 4)
        np.testing.assert_equal(evt.indices, [(slice(None, None, None),)])
        np.testing.assert_equal(evt.full_indices, [(0,), (1,), (2,), (3,)])
        self.assertEqual(evt.n_indices, 4)
        np.testing.assert_equal(event_handler.performer.array_1d_poly, [3, 3, 0, 0])

        # Add event for single entry in array
        evt = event_handler.add_event(
            "1D_single", "performer.array_1d_poly", 1, indices=0, time=3
        )
        np.testing.assert_equal(evt.state, 1)
        np.testing.assert_equal(evt.full_state, [1])
        self.assertEqual(evt.n_entries, 1)
        np.testing.assert_equal(evt.indices, [(0,)])
        np.testing.assert_equal(evt.full_indices, [(0,)])
        self.assertEqual(evt.n_indices, 1)
        np.testing.assert_equal(event_handler.performer.array_1d_poly, [1, 3, 0, 0])

        # Add event for multiple entry in array
        evt = event_handler.add_event(
            "1D_multi", "performer.array_1d_poly", [2, 2], indices=[0, 1], time=4
        )
        np.testing.assert_equal(evt.state, [2, 2])
        np.testing.assert_equal(evt.full_state, [2, 2])
        self.assertEqual(evt.n_entries, 2)
        np.testing.assert_equal(evt.indices, [(0,), (1,)])
        np.testing.assert_equal(evt.full_indices, [(0,), (1,)])
        self.assertEqual(evt.n_indices, 2)
        np.testing.assert_equal(event_handler.performer.array_1d_poly, [2, 2, 0, 0])

        # Add event for single entry in array to check if list notation works
        evt = event_handler.add_event(
            "1D_multi_flat", "performer.array_1d_poly", [0], indices=[1], time=5
        )
        np.testing.assert_equal(evt.state, [0])
        np.testing.assert_equal(evt.full_state, [0])
        self.assertEqual(evt.n_entries, 1)
        np.testing.assert_equal(evt.indices, [(1,)])
        np.testing.assert_equal(evt.full_indices, [(1,)])
        self.assertEqual(evt.n_indices, 1)
        np.testing.assert_equal(event_handler.performer.array_1d_poly, [2, 0, 0, 0])

        if plot:
            event_handler.plot_events()

    def test_event_polynomial_ndarray(self):
        event_handler = self.event_handler

        t = 0

        # Add event relying for individual coefficient
        t += 1
        evt = event_handler.add_event(
            "single_coeff", "performer.ndarray_poly", 1, indices=(0, 1), time=t
        )
        np.testing.assert_equal(evt.state, 1)
        np.testing.assert_equal(evt.full_state, [1])
        self.assertEqual(evt.n_entries, 1)
        np.testing.assert_equal(evt.indices, [(0, 1)])
        np.testing.assert_equal(evt.full_indices, [(0, 1)])
        self.assertEqual(evt.n_indices, 1)
        np.testing.assert_equal(
            event_handler.performer.ndarray_poly,
            [
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ],
        )

        # Add event for multiple coefficients / indices
        t += 1
        evt = event_handler.add_event(
            "multi_coeffs",
            "performer.ndarray_poly",
            [1, 2],
            indices=[(0, 0), (1, 1)],
            time=t,
        )
        np.testing.assert_equal(evt.state, [1, 2])
        np.testing.assert_equal(evt.full_state, [1, 2])
        self.assertEqual(evt.n_entries, 2)
        np.testing.assert_equal(evt.indices, [(0, 0), (1, 1)])
        np.testing.assert_equal(evt.full_indices, [(0, 0), (1, 1)])
        self.assertEqual(evt.n_indices, 2)
        np.testing.assert_equal(
            event_handler.performer.ndarray_poly,
            [
                [1, 1, 0, 0],
                [0, 2, 0, 0],
            ],
        )

        # Add event relying on filling missing coefficients
        t += 1
        evt = event_handler.add_event("fill_all", "performer.ndarray_poly", 1, time=t)
        np.testing.assert_equal(evt.state, 1)
        np.testing.assert_equal(
            evt.full_state,
            [1, 0, 0, 0, 1, 0, 0, 0],
        )
        self.assertEqual(evt.n_entries, 8)
        np.testing.assert_equal(evt.indices, [(slice(None, None, None),)])
        np.testing.assert_equal(
            evt.full_indices,
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        )
        self.assertEqual(evt.n_indices, 8)
        np.testing.assert_equal(
            event_handler.performer.ndarray_poly,
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ],
        )
        # Add event relying on filling missing coefficients for single entry
        t += 1
        evt = event_handler.add_event(
            "multi_fill_single", "performer.ndarray_poly", 3, indices=0, time=t
        )
        np.testing.assert_equal(evt.state, 3)
        np.testing.assert_equal(
            evt.full_state,
            [3, 0, 0, 0],
        )
        self.assertEqual(evt.n_entries, 4)
        np.testing.assert_equal(evt.indices, [(0,)])
        np.testing.assert_equal(
            evt.full_indices,
            [(0, 0), (0, 1), (0, 2), (0, 3)],
        )
        self.assertEqual(evt.n_indices, 4)
        np.testing.assert_equal(
            event_handler.performer.ndarray_poly,
            [
                [3, 0, 0, 0],
                [1, 0, 0, 0],
            ],
        )

        # Add event relying on filling missing coefficients with inhomogeneous shape
        t += 1
        evt = event_handler.add_event(
            "multi_fill_inhomogeneous", "performer.ndarray_poly", [2, [0, 1]], time=t
        )
        np.testing.assert_equal(evt.state, [2, [0, 1]])
        np.testing.assert_equal(
            evt.full_state,
            [2, 0, 0, 0, 0, 1, 0, 0],
        )
        self.assertEqual(evt.n_entries, 8)
        np.testing.assert_equal(evt.indices, [(slice(None, None, None),)])
        np.testing.assert_equal(
            evt.full_indices,
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        )
        self.assertEqual(evt.n_indices, 8)
        np.testing.assert_equal(
            event_handler.performer.ndarray_poly,
            [
                [2, 0, 0, 0],
                [0, 1, 0, 0],
            ],
        )

        # Add event using slicing (should work just like a regular ndarray)
        t += 1
        evt = event_handler.add_event(
            "multi_slice", "performer.ndarray_poly", [1], indices=[np.s_[0, :]], time=t
        )

        np.testing.assert_equal(evt.state, [1])
        np.testing.assert_equal(evt.full_state, [1, 1, 1, 1])
        self.assertEqual(evt.n_entries, 4)
        np.testing.assert_equal(evt.indices, [(0, slice(None, None, None))])
        np.testing.assert_equal(
            evt.full_indices,
            [(0, 0), (0, 1), (0, 2), (0, 3)],
        )
        self.assertEqual(evt.n_indices, 4)
        np.testing.assert_equal(
            event_handler.performer.ndarray_poly,
            [
                [1, 1, 1, 1],
                [0, 1, 0, 0],
            ],
        )

        if plot:
            event_handler.plot_events()

    def test_lwe(self):
        """This reproduces the situation from the LWE / SMA model in the examples."""
        event_handler = self.setup_event_handler()
        event_handler.name = "LWE"

        cycle_time = 110 * 60
        event_handler.cycle_time = cycle_time

        t_flush = 20 * 60

        t_gradient_start = 90
        gradient_duration = cycle_time - t_gradient_start - t_flush

        c_load = np.array([5, 1, 1, 1])
        c_wash = np.array([5, 0, 0, 0])
        c_elute = np.array([1000, 0, 0, 0])
        gradient_slope = (c_elute - c_wash) / gradient_duration
        c_gradient_poly = np.array(list(zip(c_wash, gradient_slope)))
        c_gradient_poly[0][1] = 0.4
        c_gradient_poly = c_gradient_poly.tolist()

        event_handler.add_event("load", "performer.ndarray_poly_dep", c_load, time=0)
        event_handler.add_event(
            "wash", "performer.ndarray_poly_dep", c_wash, time=t_flush
        )
        event_handler.add_event(
            "grad_start",
            "performer.ndarray_poly_dep",
            c_gradient_poly,
            time=t_gradient_start,
        )
        event_handler.add_event(
            "Flush",
            "performer.ndarray_poly_dep",
            c_elute,
            time=t_gradient_start + gradient_duration,
        )

        if plot:
            event_handler.plot_events()

    def test_event_times(self):
        event_handler = self.setup_event_handler(add_events=True)

        self.assertEqual(event_handler.event_times, [0])

        event_handler.evt0.time = 1
        self.assertEqual(event_handler.event_times, [0, 1])

        event_handler.cycle_time = 10
        event_handler.evt0.time = 11
        self.assertEqual(event_handler.event_times, [0, 1])

    def test_dependencies(self):
        event_handler = self.setup_event_handler(add_events=True)

        event_handler.add_event_dependency("evt1", "evt0")
        self.assertEqual(event_handler.event_times, [0])

        event_handler.evt0.time = 1
        self.assertEqual(event_handler.event_times, [0, 1])

        event_handler.add_event_dependency("evt2", "evt1", 2)
        self.assertEqual(event_handler.event_times, [0, 1, 2])

        event_handler.add_event_dependency("evt3", ["evt1", "evt0"], [2, 1])
        self.assertEqual(event_handler.event_times, [0, 1, 2, 3])

        # Dependent event
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.evt1.time = 1

        # Event does not exist
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event_dependency("evt3", "evt0")

        # Duplicate dependency
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event_dependency("evt1", "evt0")

        # Linear factors not matching
        with self.assertRaises(CADETProcess.CADETProcessError):
            event_handler.add_event_dependency("evt1", "evt0", [1, 1])

    def test_section_states(self):
        pass

    def test_timelines(self):
        pass

    def test_duplicate_event_times(self):
        event_handler = self.setup_event_handler()

        event_handler.add_event("evt0", "performer.scalar_float", 0, 0)
        event_handler.add_event("evt1", "performer.scalar_float", 1, 1)

        event_handler.check_config()

        event_handler.evt1.time = 0

        self.assertFalse(event_handler.check_duplicate_events())

    def test_uninitialized_indices(self):
        event_handler = self.setup_event_handler()

        event_handler.add_event(
            "evt0", "performer.ndarray_no_default", 0, indices=(0, 0)
        )

        self.assertFalse(event_handler.check_uninitialized_indices())


if __name__ == "__main__":
    plot = True
    import matplotlib.pyplot as plt

    plt.close("all")
    unittest.main()
