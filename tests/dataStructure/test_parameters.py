import unittest

import numpy as np
from CADETProcess.dataStructure import (
    Aggregator,
    Callable,
    Constant,
    DependentlyModulatedUnsignedList,
    Float,
    FloatList,
    Integer,
    IntegerList,
    List,
    Matrix,
    NdPolynomial,
    Polynomial,
    RangedFloat,
    SizedAggregator,
    SizedList,
    SizedNdArray,
    SizedUnsignedList,
    SizedUnsignedNdArray,
    String,
    Structure,
    Switch,
    Typed,
    UnsignedInteger,
    Vector,
)


class TestDescription(unittest.TestCase):
    def setUp(self):
        class Model(Structure):
            param_with_description = Integer(description="foo")

        self.model = Model()

    def test_description(self):
        self.assertEqual(type(self.model).param_with_description.description, "foo")

    def test_modified_descriptor(self):
        type(self.model).param_with_description.description = "bar"
        self.assertEqual(type(self.model).param_with_description.description, "bar")


class TestParameterDictionaries(unittest.TestCase):
    def setUp(self):
        class Model(Structure):
            param = Integer()
            param_default = Integer(default=1)
            no_param = None

            _parameters = ["param", "param_default"]

        self.model = Model()

    def test_parameters_dict_getter(self):
        np.testing.assert_equal(
            self.model._parameters_dict, {"param": None, "param_default": 1}
        )

        self.model.param = 1
        np.testing.assert_equal(
            self.model._parameters_dict, {"param": 1, "param_default": 1}
        )

        self.model.param = None
        np.testing.assert_equal(
            self.model._parameters_dict, {"param": None, "param_default": 1}
        )

        self.model.param_default = 2
        np.testing.assert_equal(
            self.model._parameters_dict, {"param": None, "param_default": 2}
        )

        self.model.param_default = None
        np.testing.assert_equal(
            self.model._parameters_dict, {"param": None, "param_default": 1}
        )

    def test_parameters_dict_setter(self):
        self.model.parameters = {"param": 1, "param_default": 2}
        np.testing.assert_equal(
            self.model._parameters_dict, {"param": 1, "param_default": 2}
        )

        self.model.parameters = {"param": 2}
        np.testing.assert_equal(
            self.model._parameters_dict, {"param": 2, "param_default": 2}
        )

        with self.assertRaises(ValueError):
            self.model.parameters = {"not_a_valid_param": 1}


class TestConstant(unittest.TestCase):
    def setUp(self):
        class Model(Structure):
            const_int = Constant(value=0)
            const_list = Constant(value=[0, 1])

        self.model = Model()

    def test_constant(self):
        self.assertEqual(self.model.const_int, 0)

        with self.assertRaises(ValueError):
            self.model.const_int = 1

        self.assertEqual(self.model.const_list, [0, 1])

        with self.assertRaises(ValueError):
            self.model.const_list = [1, 2]

    def test_error_when_no_value(self):
        with self.assertRaises(TypeError):

            class NoValue(Structure):
                const = Constant()


class TestSwitch(unittest.TestCase):
    def setUp(self):
        class Model(Structure):
            switch = Switch(valid=["foo", "bar"], default="foo")

        self.model = Model()

    def test_value(self):
        self.model.switch = "bar"
        self.assertEqual(self.model.switch, "bar")

        with self.assertRaises(ValueError):
            self.model.switch = "spam"

    def test_default(self):
        self.assertEqual(self.model.switch, "foo")

        with self.assertRaises(ValueError):

            class InvalidDefault(Structure):
                switch = Switch(valid=["foo", "bar"], default="spam")


class TestTyped(unittest.TestCase):
    def setUp(self):
        class Model(Structure):
            string_param = String(default="foo")
            list_param = List(default=[1, 2])
            dynamic_type_param = Typed(ty=list, default=[0, 1])

        self.model = Model()
        self.model_other = Model()

    def test_values(self):
        self.model.string_param = "string_param"
        self.assertEqual(self.model.string_param, "string_param")

        with self.assertRaises(TypeError):
            self.model.string_param = 0

        self.model.list_param = [
            0,
        ]
        self.assertEqual(
            self.model.list_param,
            [
                0,
            ],
        )

        with self.assertRaises(TypeError):
            self.model.list_param = 0

        self.model.dynamic_type_param = [0]
        with self.assertRaises(TypeError):
            self.model.dynamic_type_param = 0

    def test_default(self):
        self.assertEqual(self.model.string_param, "foo")
        self.assertEqual(self.model.list_param, [1, 2])
        self.assertEqual(self.model.dynamic_type_param, [0, 1])

        self.assertFalse(self.model.list_param is self.model_other.list_param)
        self.assertFalse(self.model.dynamic_type_param is self.model_other.list_param)

        with self.assertRaises(TypeError):

            class WrongDefaultType(Structure):
                param_1 = List(default="string")

        with self.assertRaises(TypeError):

            class WrongDefaultType(Structure):
                param_1 = List(default=1)

            model = WrongDefaultType()
            model.param_1


class TestCallable(unittest.TestCase):
    def setUp(self):
        def default_method(x):
            return x

        class Model(Structure):
            method = Callable()
            method_default = Callable(default=default_method)

        self.model = Model()

    def test_values(self):
        def method(x):
            return 2 * x

        self.model.method = method
        assert callable(self.model.method)

        self.assertEqual(self.model.method(2), 4)

        with self.assertRaises(TypeError):
            self.model.method = 2

    def test_default(self):
        assert callable(self.model.method_default)
        self.assertEqual(self.model.method_default(2), 2)


class TestDtype(unittest.TestCase):
    def setUp(self):
        class Model(Structure):
            integer_list_param = IntegerList()
            float_list_param = FloatList()

        self.model = Model()
        self.model_other = Model()

    def test_values(self):
        self.model.integer_list_param = [1, 2]
        self.assertEqual(self.model.integer_list_param, [1, 2])

        with self.assertRaises(ValueError):
            self.model.integer_list_param = [1, "foo"]

        self.model.float_list_param = [1.0, 2]
        self.assertEqual(self.model.float_list_param, [1.0, 2.0])


class TestRanged(unittest.TestCase):
    def setUp(self):
        class Model(Structure):
            bound_float_param = RangedFloat(lb=-1, ub=1, default=0)
            unsigned_integer_param = UnsignedInteger(default=1)

        self.model = Model()

    def test_values(self):
        self.model.bound_float_param = 0.5
        self.assertEqual(self.model.bound_float_param, 0.5)

        with self.assertRaises(ValueError):
            self.model.bound_float_param = -2

        with self.assertRaises(ValueError):
            self.model.bound_float_param = 2

        self.model.unsigned_integer_param = 10
        with self.assertRaises(ValueError):
            self.model.unsigned_integer_param = -2

    def test_default(self):
        self.assertEqual(self.model.bound_float_param, 0.0)
        self.assertEqual(self.model.unsigned_integer_param, 1)

        with self.assertRaises(ValueError):

            class InvalidDefault(Structure):
                param_1 = UnsignedInteger(default=-1)


class TestSizedUnified(unittest.TestCase):
    def setUp(self):
        class Model(Structure):
            sized_list = SizedList(size=4, default=0)
            sized_array = SizedNdArray(size=(2, 4), default=0)

            sized_unsigned_list = SizedUnsignedList(size=4, default=1)
            sized_unsigned_array = SizedUnsignedNdArray(size=(2, 4), default=1)

            sized_list_full_default = SizedList(size=4, default=[1, 2, 3, 4])
            sized_array_full_default = SizedNdArray(
                size=(2, 4), default=[[1, 2, 3, 4], [5, 6, 7, 8]]
            )

        self.model = Model()

    def test_values(self):
        self.model.sized_list = [1, 2, 3, 4]
        np.testing.assert_equal(self.model.sized_list, [1, 2, 3, 4])

        self.model.sized_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        np.testing.assert_equal(self.model.sized_array, [[1, 2, 3, 4], [5, 6, 7, 8]])

        self.model.sized_array = [[0, 1, 2, 3], [4, 5, 6, 7]]
        np.testing.assert_equal(self.model.sized_array, [[0, 1, 2, 3], [4, 5, 6, 7]])

        self.model.sized_unsigned_list = [1, 2, 3, 4]
        np.testing.assert_equal(self.model.sized_unsigned_list, [1, 2, 3, 4])

        with self.assertRaises(ValueError):
            self.model.sized_unsigned_list = [-1, 2, 3, 4]

        self.model.sized_unsigned_array = [[1, 2, 3, 4], [5, 6, 7, 8]]
        np.testing.assert_equal(
            self.model.sized_unsigned_array, [[1, 2, 3, 4], [5, 6, 7, 8]]
        )

        with self.assertRaises(ValueError):
            self.model.sized_unsigned_array = [[-1, 2, 3, 4], [5, 6, 7, 8]]

    def test_default(self):
        np.testing.assert_equal(self.model.sized_list, [0, 0, 0, 0])
        np.testing.assert_equal(self.model.sized_array, [[0, 0, 0, 0], [0, 0, 0, 0]])

        np.testing.assert_equal(self.model.sized_unsigned_list, [1, 1, 1, 1])
        np.testing.assert_equal(
            self.model.sized_unsigned_array, [[1, 1, 1, 1], [1, 1, 1, 1]]
        )

        # Full default
        # List
        np.testing.assert_equal(self.model.sized_list_full_default, [1, 2, 3, 4])

        with self.assertRaises(ValueError):

            class InvalidDefaultSize(Structure):
                param_1 = SizedList(size=4, default=[1, 2, 3])

        # Array
        np.testing.assert_equal(
            self.model.sized_array_full_default, [[1, 2, 3, 4], [5, 6, 7, 8]]
        )
        with self.assertRaises(ValueError):

            class InvalidDefaultSize(Structure):
                param_1 = SizedNdArray(
                    size=(4, 2), default=[[1, 2, 3, 4], [5, 6, 7, 8], [10, 11, 12, 13]]
                )

        # Invalid range
        with self.assertRaises(ValueError):

            class InvalidDefaultRange(Structure):
                param_1 = SizedUnsignedList(size=4, default=-1)


class TestSizedDependent(unittest.TestCase):
    """Previous test methods for dependently sized parameters."""

    def setUp(self):
        class Model(Structure):
            dep_1 = UnsignedInteger()
            dep_2 = UnsignedInteger(default=3)

            list_single_dep_param = SizedUnsignedList(size="dep_1", default=1)
            array_single_dep_param = SizedUnsignedNdArray(size="dep_1", default=1)

            list_double_dep_param = SizedUnsignedList(
                size=("dep_1", "dep_2"), default=1
            )
            array_double_dep_param = SizedUnsignedNdArray(
                size=("dep_1", "dep_2"), default=1
            )

            list_double_dep_with_int = SizedUnsignedList(size=("dep_1", 2), default=2)
            array_double_dep_with_int = SizedUnsignedNdArray(
                size=("dep_1", 2), default=2
            )

        self.model = Model()

    def test_values(self):
        # Missing dependency value
        with self.assertRaises(ValueError):
            self.model.list_single_dep_param = [2, 2]

        with self.assertRaises(ValueError):
            self.model.array_single_dep_param = [2, 2]

        self.model.dep_1 = 2

        # Single dependency
        self.model.list_single_dep_param = [2, 2]
        np.testing.assert_equal(self.model.list_single_dep_param, [2, 2])

        self.model.array_single_dep_param = [2, 2]
        np.testing.assert_equal(self.model.array_single_dep_param, [2, 2])

        # Wrong shape
        with self.assertRaises(ValueError):
            self.model.list_single_dep_param = [2, 2, 2]

        # Multiple dependencies
        np.testing.assert_array_equal(self.model.list_double_dep_param, np.ones((6,)))
        np.testing.assert_array_equal(
            self.model.array_double_dep_param, np.ones((2, 3))
        )

        # Size error
        with self.assertRaises(ValueError):
            self.model.list_double_dep_param = np.ones((3, 3)).flatten().tolist()
        with self.assertRaises(ValueError):
            self.model.array_double_dep_param = np.ones((3, 3))

        # Range error
        with self.assertRaises(ValueError):
            self.model.list_double_dep_param = (-1 * np.ones((2, 3))).flatten().tolist()
        with self.assertRaises(ValueError):
            self.model.array_double_dep_param = -1 * np.ones((2, 3))

        # Dependencies with integer dimension
        np.testing.assert_equal(self.model.list_double_dep_with_int, [2, 2, 2, 2])
        np.testing.assert_equal(self.model.array_double_dep_with_int, [[2, 2], [2, 2]])

    def test_default(self):
        self.model.dep_1 = 2

        np.testing.assert_equal(self.model.list_single_dep_param, [1, 1])
        np.testing.assert_equal(self.model.array_single_dep_param, [1, 1])

        # Full default
        with self.assertRaises(ValueError):

            class InvalidDefaultSizeList(Structure):
                dep_1 = UnsignedInteger()
                param_1 = SizedUnsignedList(size="dep_1", default=[1, 2, 3])

        with self.assertRaises(ValueError):

            class InvalidDefaultSizeArray(Structure):
                param_1 = SizedUnsignedNdArray(
                    size=(4, "dep_1"), default=[[1, 2, 3, 4], [5, 6, 7, 8]]
                )

        # Range error
        with self.assertRaises(ValueError):

            class InvalidDefault(Structure):
                list_single_dep_param = SizedUnsignedList(size="dep_1", default=-1)


class TestPolynomial(unittest.TestCase):
    def setUp(self):
        class Model(Structure):
            poly_param = Polynomial(n_coeff=2, default=0)
            ndpoly_param = NdPolynomial(n_entries=2, n_coeff=4)

            n_coeff = 4
            poly_param_dep = Polynomial(size=("n_coeff"))

            n_entries = 3
            ndpoly_param_dep = NdPolynomial(size=("n_entries", "n_coeff"))

        self.model = Model()

    def test_parameter(self):
        with self.assertRaises(ValueError):

            class TestDuplicateCoeffs(Structure):
                n_coeff = 2
                faulty = Polynomial(n_coeff=2, size=("n_coeff"))

        with self.assertRaises(ValueError):

            class TestMissingEntries(Structure):
                n_coeff = 2
                faulty = NdPolynomial(size=("n_coeff"))

        with self.assertRaises(ValueError):

            class TestDuplicateCoeff(Structure):
                n_coeff = 2
                entries = 2
                faulty = NdPolynomial(n_entries=2, n_coeff=2, size=("n_coeff"))

        with self.assertRaises(ValueError):

            class TestInvalidDefault(Structure):
                faulty = NdPolynomial(n_entries=2, n_coeff=2, default=1)

    def test_values(self):
        # Single entry n_coeff
        self.model.poly_param = 2
        np.testing.assert_equal(self.model.poly_param, [2, 0])

        self.model.poly_param = [3, 2]
        np.testing.assert_equal(self.model.poly_param, [3, 2])

        with self.assertRaises(ValueError):
            self.model.poly_param = [3, 2, 1]

        # Single entry with dependency
        self.model.n_coeff = 3

        with self.assertRaises(ValueError):
            self.model.poly_param_dep = [1, 2, 3, 4]

        self.model.poly_param_dep = [1, 2, 3]
        np.testing.assert_equal(self.model.poly_param_dep, [1, 2, 3])

        self.model.n_coeff = 4
        self.model.poly_param_dep = [1, 2, 3, 4]
        np.testing.assert_equal(self.model.poly_param_dep, [1, 2, 3, 4])

        # Multiple entries
        self.model.ndpoly_param = 2
        np.testing.assert_equal(self.model.ndpoly_param, [[2, 0, 0, 0], [2, 0, 0, 0]])

        self.model.ndpoly_param = [3, 2]
        np.testing.assert_equal(self.model.ndpoly_param, [[3, 0, 0, 0], [2, 0, 0, 0]])

        self.model.ndpoly_param = [[1, 2], [0, 1, 2]]
        np.testing.assert_equal(
            self.model.ndpoly_param, [[1, 2, 0.0, 0.0], [0, 1, 2, 0]]
        )

        with self.assertRaises(ValueError):
            self.model.ndpoly_param = [3, 2, 1]

        with self.assertRaises(ValueError):
            self.model.ndpoly_param = [[1], [1], [1]]

        # Multiple entries with dependency
        self.model.n_coeff = 3

        with self.assertRaises(ValueError):
            self.model.ndpoly_param_dep = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]

        self.model.ndpoly_param_dep = 2
        np.testing.assert_equal(
            self.model.ndpoly_param_dep, [[2, 0, 0], [2, 0, 0], [2, 0, 0]]
        )
        self.model.n_coeff = 4

        self.model.ndpoly_param_dep = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        np.testing.assert_equal(
            self.model.ndpoly_param_dep, [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        )

        self.model.ndpoly_param_dep = [1, 2, 3]
        np.testing.assert_equal(
            self.model.ndpoly_param_dep, [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]
        )

        self.model.ndpoly_param_dep = [[1, 2], [0, 1, 2], [0]]
        np.testing.assert_equal(
            self.model.ndpoly_param_dep, [[1, 2, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0]]
        )

        with self.assertRaises(ValueError):
            self.model.ndpoly_param_dep = [1, 2, 3, 4]

        with self.assertRaises(ValueError):
            self.model.ndpoly_param_dep = [[1], [1], [1], [1]]

    def test_default(self):
        """
        Currently, only `0` is allowed as default value.

        Notes
        -----
        Technically, default values would be possible from point of the NdPolynomial
        class. However, due to its use as an Event Parameter, this is currently diabled.
        """
        np.testing.assert_equal(self.model.poly_param, [0, 0])

        np.testing.assert_equal(self.model.ndpoly_param, None)


class TestModulated(unittest.TestCase):
    def setUp(self):
        class Model(Structure):
            n_mod = 4

            modulated_list = DependentlyModulatedUnsignedList(size="n_mod")

        self.model = Model()

    def test_value(self):
        self.model.modulated_list = [1, 2, 3, 4]

        np.testing.assert_equal(self.model.modulated_list, [1, 2, 3, 4])

        self.model.modulated_list = [1, 2, 3, 4, 5, 6, 7, 8]

        with self.assertRaises(ValueError):
            self.model.modulated_list = [1, 2, 3, 4, 5]

        self.model.n_mod = 5
        with self.assertRaises(ValueError):
            self.model.modulated_list


class TestDimensionalized(unittest.TestCase):
    def setUp(self):
        class Model(Structure):
            vector = Vector()
            matrix = Matrix()

        self.model = Model()

    def test_value(self):
        self.model.vector = [1, 2, 3]

        with self.assertRaises(ValueError):
            self.model.vector = [[1, 2, 3], [4, 5, 6]]

        with self.assertRaises(ValueError):
            self.model.matrix = [1, 2]


class TestAggregator(unittest.TestCase):
    def setUp(self):
        class DummyInstance(Structure):
            float_param = Float(default=1.0)
            sized_param = SizedNdArray(size=4)
            sized_param_transposed = SizedNdArray(size=2)

        class Model(Structure):
            aggregator = Aggregator("float_param", "container")
            sized_aggregator = SizedAggregator("sized_param", "container")
            transposed_sized_aggregator = SizedAggregator(
                "sized_param_transposed", "container", transpose=True
            )

            def __init__(self):
                self.container = [
                    DummyInstance(
                        float_param=i,
                        sized_param=[float(i * j) for j in range(4)],
                        sized_param_transposed=[float(i * j) for j in range(2)],
                    )
                    for i in range(3)
                ]

        self.model = Model()

    def test_value(self):
        # Aggregator
        self.assertAlmostEqual(self.model.aggregator, [0.0, 1.0, 2.0])

        new_value = [1, 2, 3]
        self.model.aggregator = new_value

        self.assertAlmostEqual(self.model.aggregator, new_value)
        for con, val in zip(self.model.container, new_value):
            self.assertAlmostEqual(con.float_param, val)

        # Test setting incorrect types or dimensions:
        with self.assertRaises((ValueError, TypeError)):
            self.model.aggregator = 3

        # Test setting slices and indexes
        self.model.aggregator[0] = 3.0
        self.assertAlmostEqual(self.model.aggregator[0], 3.0)

        # SizedAggregator
        np.testing.assert_almost_equal(
            self.model.sized_aggregator,
            [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 4, 6],
            ],
        )

        new_value = [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]
        self.model.sized_aggregator = new_value

        np.testing.assert_almost_equal(self.model.sized_aggregator, new_value)
        for con, val in zip(self.model.container, new_value):
            np.testing.assert_almost_equal(con.sized_param, val)

        # Test setting incorrect types or dimensions:
        with self.assertRaises((ValueError, TypeError)):
            self.model.sized_aggregator = 3

        with self.assertRaises((ValueError, TypeError)):
            self.model.sized_aggregator = [1.0, 2.0, 3.0, 4.0]

        with self.assertRaises(IndexError):
            self.model.sized_aggregator[3] = 3.0

        # Test setting slices and indexes
        new_value = [
            [1.5, 2.5, 3.5, 4.5],
            [2.5, 3.5, 4.5, 5.5],
            [3.5, 4.5, 5.5, 6.5],
        ]

        self.model.sized_aggregator[0] = new_value[0]
        np.testing.assert_almost_equal(self.model.sized_aggregator[0], new_value[0])

        self.model.sized_aggregator[0, 0] = 7.5
        np.testing.assert_almost_equal(self.model.sized_aggregator[0, 0], 7.5)

        # Transposed SizedAggregator
        np.testing.assert_almost_equal(
            self.model.transposed_sized_aggregator,
            [
                [0, 0, 0],
                [0, 1, 2],
            ],
        )
        new_value = [
            [1, 2, 3],
            [2, 3, 4],
        ]
        self.model.transposed_sized_aggregator = new_value

        np.testing.assert_almost_equal(
            self.model.transposed_sized_aggregator, new_value
        )
        for con, val in zip(self.model.container, np.array(new_value).T):
            np.testing.assert_almost_equal(con.sized_param_transposed, val)


if __name__ == "__main__":
    unittest.main()
