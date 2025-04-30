import shutil
import time
import unittest
from abc import ABC, abstractmethod

import numpy as np
from addict import Dict
from CADETProcess import settings
from CADETProcess.dataStructure import (
    Float,
    List,
    NdPolynomial,
    Polynomial,
    SizedList,
    SizedNdArray,
    Structure,
)
from CADETProcess.optimization import OptimizationProblem

from tests.optimization_problem_fixtures import (
    LinearConstraintsSooTestProblem2,
    LinearEqualityConstraintsSooTestProblem,
)


class EvaluationObject(Structure):
    uninitialized = None
    scalar_param = Float(default=1)
    scalar_param_2 = Float(default=1)
    list_param = List()
    sized_list_param = SizedList(size=2, default=[1, 2])
    sized_list_param_single = SizedList(size=1, default=1)
    sized_list_param_no_default = SizedList(size=2)
    nd_array = SizedNdArray(size=(2, 2))
    polynomial_param = Polynomial(n_coeff=2, default=0)
    polynomial_param_no_default = Polynomial(n_coeff=2)
    nd_polynomial_param = NdPolynomial(size=(2, 4), default=0)

    _parameters = [
        "uninitialized",
        "scalar_param",
        "scalar_param_2",
        "list_param",
        "sized_list_param",
        "sized_list_param_single",
        "sized_list_param_no_default",
        "nd_array",
        "polynomial_param",
        "polynomial_param_no_default",
        "nd_polynomial_param",
    ]

    def __init__(self, name="Dummy"):
        self.name = name
        super().__init__()

    def __str__(self):
        return self.name


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, request):
        pass

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def __str__(self):
        return self.__class__.__name__


class ExpensiveEvaluator(Evaluator):
    """Mockup for an expensive evaluator"""

    def evaluate(self, evaluation_object):
        print("Run expensive evaluator; this takes forever...")

        expensive_results = Dict()
        expensive_results.result = np.array(2 * [evaluation_object.scalar_param])

        time.sleep(1)

        return expensive_results


class CheapEvaluator(Evaluator):
    """Mockup for a cheap evaluator"""

    def evaluate(self, expensive_results):
        print("Run cheap evaluator.")
        cheap_results = Dict()
        cheap_results.result_1 = expensive_results.result * 2
        cheap_results.result_2 = expensive_results.result * 2

        return cheap_results


def sum_x(x):
    print("run sum(x)")
    return [sum(x)]


def min_results_1(cheap_results):
    print("run MinResult1")
    score = cheap_results.result_1 * 2
    return score.tolist()


def min_results_2(cheap_results):
    print("run MinResult2")
    score = cheap_results.result_2 * 3
    return score.tolist()


from CADETProcess import CADETProcessError
from CADETProcess.optimization import OptimizationVariable


class Test_OptimizationVariable(unittest.TestCase):
    def setUp(self):
        self.evaluation_object = EvaluationObject()

    def test_index(self):
        var = OptimizationVariable(
            "no_index_required",
            evaluation_objects=[self.evaluation_object],
            parameter_path="scalar_param",
        )
        var.value = 1
        np.testing.assert_equal(var.value, 1)
        np.testing.assert_equal(self.evaluation_object.scalar_param, 1)

        with self.assertRaises(IndexError):
            var = OptimizationVariable(
                "parameter_not_sized",
                evaluation_objects=[self.evaluation_object],
                parameter_path="scalar_param",
                indices=1,
            )

        var = OptimizationVariable(
            "list_index",
            evaluation_objects=[self.evaluation_object],
            parameter_path="sized_list_param",
            indices=1,
        )
        var.value = 2
        np.testing.assert_equal(var.value, 2)
        np.testing.assert_equal(var.indices, [[(1,)]])
        np.testing.assert_equal(var.full_indices, [[(1,)]])
        np.testing.assert_equal(self.evaluation_object.sized_list_param, [1, 2])

        with self.assertRaises(IndexError):
            var = OptimizationVariable(
                "index_exceeds_list_size",
                evaluation_objects=[self.evaluation_object],
                parameter_path="sized_list_param",
                indices=2,
            )

        # Test uninitialized value
        with self.assertRaises(CADETProcessError):
            var = OptimizationVariable(
                "uninitialized_attribute",
                evaluation_objects=[self.evaluation_object],
                parameter_path="uninitialized",
                indices=2,
            )

    def test_significant_digits(self):
        var_no_precision = OptimizationVariable(
            "not_rounded_variable",
            evaluation_objects=[self.evaluation_object],
            parameter_path="scalar_param",
            significant_digits=None,
        )
        var_no_precision.value = 1.1234
        np.testing.assert_equal(var_no_precision.value, 1.1234)
        np.testing.assert_equal(self.evaluation_object.scalar_param, 1.1234)

        var_with_precision = OptimizationVariable(
            "rounded_variable",
            evaluation_objects=[self.evaluation_object],
            parameter_path="scalar_param",
            significant_digits=2,
        )
        var_with_precision.value = 1.1234
        np.testing.assert_equal(var_with_precision.value, 1.1)
        np.testing.assert_equal(self.evaluation_object.scalar_param, 1.1)

    def test_sized_list_single(self):
        """
        Extra test for array with single entry.

        See also: https://github.com/fau-advanced-separations/CADET-Process/pull/68
        """
        # No indices
        var = OptimizationVariable(
            "sized_list_param_single",
            evaluation_objects=[self.evaluation_object],
            parameter_path="sized_list_param_single",
        )
        np.testing.assert_equal(var.indices, [[(slice(None, None, None),)]])
        var.value = 1
        np.testing.assert_equal(var.value, 1)
        np.testing.assert_equal(self.evaluation_object.sized_list_param_single, [1])

        # Explicit index
        var = OptimizationVariable(
            "sized_list_param_single",
            evaluation_objects=[self.evaluation_object],
            parameter_path="sized_list_param_single",
            indices=0,
        )
        np.testing.assert_equal(var.indices, [[(0,)]])
        var.value = 2
        np.testing.assert_equal(var.value, 2)
        np.testing.assert_equal(self.evaluation_object.sized_list_param_single, [2])

        # Raise Exception for exceeding index
        with self.assertRaises(IndexError):
            var = OptimizationVariable(
                "sized_list_param_single",
                evaluation_objects=[self.evaluation_object],
                parameter_path="sized_list_param_single",
                indices=1,
            )

    def test_nd_array(self):
        var = OptimizationVariable(
            "nd_array_index",
            evaluation_objects=[self.evaluation_object],
            parameter_path="nd_array",
            indices=(0, 0),
        )
        var.value = 1
        np.testing.assert_equal(var.value, 1)
        np.testing.assert_equal(var.indices, [[(0, 0)]])
        np.testing.assert_equal(var.full_indices, [[(0, 0)]])
        np.testing.assert_equal(
            self.evaluation_object.nd_array, [[1, np.nan], [np.nan, np.nan]]
        )

        var = OptimizationVariable(
            "nd_array_slice",
            evaluation_objects=[self.evaluation_object],
            parameter_path="nd_array",
            indices=np.s_[:, 0],
        )
        var.value = 2
        np.testing.assert_equal(var.value, 2)
        np.testing.assert_equal(var.indices, [[(slice(None, None, None), 0)]])
        np.testing.assert_equal(var.full_indices, [[(0, 0), (1, 0)]])
        np.testing.assert_equal(
            self.evaluation_object.nd_array, [[2, np.nan], [2, np.nan]]
        )

        var = OptimizationVariable(
            "nd_array_all",
            evaluation_objects=[self.evaluation_object],
            parameter_path="nd_array",
        )
        var.value = 3
        np.testing.assert_equal(var.value, 3)
        np.testing.assert_equal(var.indices, [[(slice(None, None, None),)]])
        np.testing.assert_equal(var.full_indices, [[(0, 0), (0, 1), (1, 0), (1, 1)]])
        np.testing.assert_equal(self.evaluation_object.nd_array, [[3, 3], [3, 3]])

    def test_polynomial(self):
        var = OptimizationVariable(
            "polynomial",
            evaluation_objects=self.evaluation_object,
            parameter_path="polynomial_param",
            indices=0,
        )

        var.value = 1
        np.testing.assert_equal(var.value, 1)
        np.testing.assert_equal(var.indices, [[(0,)]])
        np.testing.assert_equal(var.full_indices, [[(0,)]])
        np.testing.assert_equal(self.evaluation_object.scalar_param, 1)

        # Check convenience function; No index should modify constant coefficient and
        # and set rest to 0
        var = OptimizationVariable(
            "polynomial_convenience_1D",
            evaluation_objects=self.evaluation_object,
            parameter_path="polynomial_param",
        )
        var.value = 1
        np.testing.assert_equal(var.value, 1)
        np.testing.assert_equal(self.evaluation_object.polynomial_param, [1, 0])

        var = OptimizationVariable(
            "polynomial_convenience_ND_all",
            evaluation_objects=self.evaluation_object,
            parameter_path="nd_polynomial_param",
        )
        var.value = 1
        np.testing.assert_equal(var.value, 1)
        np.testing.assert_equal(var.indices, [[(slice(None, None, None),)]])
        np.testing.assert_equal(
            var.full_indices,
            [[(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]],
        )
        np.testing.assert_equal(
            self.evaluation_object.nd_polynomial_param, [[1, 0, 0, 0], [1, 0, 0, 0]]
        )

        var = OptimizationVariable(
            "polynomial_convenience_ND_single",
            evaluation_objects=self.evaluation_object,
            parameter_path="nd_polynomial_param",
            indices=0,
        )
        var.value = 2
        np.testing.assert_equal(var.value, 2)
        np.testing.assert_equal(var.indices, [[(0,)]])
        np.testing.assert_equal(var.full_indices, [[(0, 0), (0, 1), (0, 2), (0, 3)]])
        np.testing.assert_equal(
            self.evaluation_object.nd_polynomial_param, [[2, 0, 0, 0], [1, 0, 0, 0]]
        )

        var = OptimizationVariable(
            "polynomial_convenience_ND_single",
            evaluation_objects=self.evaluation_object,
            parameter_path="nd_polynomial_param",
            indices=np.s_[0, :],
        )
        var.value = 3
        np.testing.assert_equal(var.value, 3)
        np.testing.assert_equal(var.indices, [[(0, slice(None, None, None))]])
        np.testing.assert_equal(var.full_indices, [[(0, 0), (0, 1), (0, 2), (0, 3)]])
        np.testing.assert_equal(
            self.evaluation_object.nd_polynomial_param, [[3, 3, 3, 3], [1, 0, 0, 0]]
        )

    def test_transform(self):
        var = OptimizationVariable(
            "scalar_param",
            evaluation_objects=[self.evaluation_object],
            parameter_path="scalar_param",
        )

        # Missing bounds
        with self.assertRaises(CADETProcessError):
            var = OptimizationVariable(
                "scalar_param",
                evaluation_objects=[self.evaluation_object],
                parameter_path="scalar_param",
                transform="auto",
            )


from tests.test_events import HandlerFixture


class Test_OptimizationVariableEvents(unittest.TestCase):
    def setUp(self):
        evaluation_object = HandlerFixture()

        evt = evaluation_object.add_event(
            "single_index_1D", "performer.array_1d", 1, indices=0, time=1
        )
        evt = evaluation_object.add_event(
            "single_index_ND", "performer.ndarray", 1, indices=(0, 0), time=2
        )

        evt = evaluation_object.add_event(
            "multi_index_1D", "performer.array_1d", [0, 1], indices=[0, 1], time=3
        )
        evt = evaluation_object.add_event(
            "multi_index_ND",
            "performer.ndarray",
            [0, 1],
            indices=[(0, 0), (0, 1)],
            time=4,
        )

        evt = evaluation_object.add_event(
            "nd_evt_state", "performer.ndarray", [[8, 7, 6, 5], [4, 3, 2, 1]], time=5
        )

        evt = evaluation_object.add_event(
            "array_1d_poly_full", "performer.array_1d_poly", 1, time=6
        )
        evt = evaluation_object.add_event(
            "array_1d_poly_indices",
            "performer.array_1d_poly",
            [1, 1],
            indices=[1, 2],
            time=7,
        )

        evt = evaluation_object.add_event(
            "ndarray_poly_full", "performer.ndarray_poly", 1, time=8
        )
        evt = evaluation_object.add_event(
            "ndarray_poly_indices",
            "performer.ndarray_poly",
            [1, 1],
            indices=[(0, 1), (1, -1)],
            time=9,
        )

        self.evaluation_object = evaluation_object

    def test_index(self):
        # Single Event.state entry
        var = OptimizationVariable(
            "single_index_1D",
            evaluation_objects=[self.evaluation_object],
            parameter_path="single_index_1D.state",
        )
        var.value = 2
        np.testing.assert_equal(var.value, 2)
        np.testing.assert_equal(self.evaluation_object.single_index_1D.state, 2)
        np.testing.assert_equal(self.evaluation_object.performer.array_1d, [2, 1, 0, 0])

        with self.assertRaises(IndexError):
            var = OptimizationVariable(
                "event_state_not_sized",
                evaluation_objects=[self.evaluation_object],
                parameter_path="single_index_1D.state",
                indices=1,
            )

        var = OptimizationVariable(
            "single_index_ND",
            evaluation_objects=[self.evaluation_object],
            parameter_path="single_index_ND.state",
        )
        var.value = 2
        np.testing.assert_equal(var.value, 2)
        np.testing.assert_equal(self.evaluation_object.single_index_ND.state, 2)
        np.testing.assert_equal(
            self.evaluation_object.performer.ndarray, [[2, 7, 6, 5], [4, 3, 2, 1]]
        )

        with self.assertRaises(IndexError):
            var = OptimizationVariable(
                "event_state_not_sized",
                evaluation_objects=[self.evaluation_object],
                parameter_path="single_index_ND.state",
                indices=1,
            )

        # 1D Event.state with multiple entries
        var = OptimizationVariable(
            "multi_index_1D",
            evaluation_objects=[self.evaluation_object],
            parameter_path="multi_index_1D.state",
            indices=0,
        )
        var.value = 3
        np.testing.assert_equal(var.value, 3)
        np.testing.assert_equal(self.evaluation_object.multi_index_1D.state, [3, 1])
        np.testing.assert_equal(self.evaluation_object.performer.array_1d, [3, 1, 0, 0])

        with self.assertRaises(IndexError):
            var = OptimizationVariable(
                "index_exceeds_parameter",
                evaluation_objects=[self.evaluation_object],
                parameter_path="multi_index_1D.state",
                indices=2,
            )
        # ND Event.state with multiple entries (in 1D)
        var = OptimizationVariable(
            "multi_index_ND",
            evaluation_objects=[self.evaluation_object],
            parameter_path="multi_index_ND.state",
            indices=0,
        )
        var.value = 4
        np.testing.assert_equal(var.value, 4)
        np.testing.assert_equal(self.evaluation_object.multi_index_ND.state, [4, 1])
        np.testing.assert_equal(
            self.evaluation_object.performer.ndarray, [[4, 1, 6, 5], [4, 3, 2, 1]]
        )

        with self.assertRaises(IndexError):
            var = OptimizationVariable(
                "index_exceeds_parameter",
                evaluation_objects=[self.evaluation_object],
                parameter_path="multi_index_ND.state",
                indices=2,
            )

        with self.assertRaises(IndexError):
            var = OptimizationVariable(
                "index_exceeds_parameter_dimension",
                evaluation_objects=[self.evaluation_object],
                parameter_path="multi_index_ND.state",
                indices=(1, 1),
            )

        # ND Event.state with multiple entries in ND
        var = OptimizationVariable(
            "nd_evt_state",
            evaluation_objects=[self.evaluation_object],
            parameter_path="nd_evt_state.state",
            indices=(0, 0),
        )
        var.value = 5
        np.testing.assert_equal(var.value, 5)
        np.testing.assert_equal(
            self.evaluation_object.nd_evt_state.state, [[5, 7, 6, 5], [4, 3, 2, 1]]
        )
        np.testing.assert_equal(
            self.evaluation_object.performer.ndarray, [[5, 7, 6, 5], [4, 3, 2, 1]]
        )

        with self.assertRaises(IndexError):
            var = OptimizationVariable(
                "index_exceeds_parameter",
                evaluation_objects=[self.evaluation_object],
                parameter_path="nd_evt_state.state",
                indices=(2, 2),
            )

    def test_polynomial(self):
        # Event state modifies entire 1D polyomial coefficients using `fill_values`
        var = OptimizationVariable(
            "array_1d_poly_full",
            evaluation_objects=self.evaluation_object,
            parameter_path="array_1d_poly_full.state",
        )

        var.value = 2
        np.testing.assert_equal(var.value, 2)
        np.testing.assert_equal(self.evaluation_object.array_1d_poly_full.state, 2)
        np.testing.assert_equal(
            self.evaluation_object.performer.array_1d_poly, [2, 0, 0, 0]
        )

        # Event state modifies individual 1D polyomial coefficients
        var = OptimizationVariable(
            "array_1d_poly_indices",
            evaluation_objects=self.evaluation_object,
            parameter_path="array_1d_poly_indices.state",
            indices=0,
        )

        var.value = 2
        np.testing.assert_equal(var.value, 2)
        np.testing.assert_equal(
            self.evaluation_object.array_1d_poly_indices.state, [2, 1]
        )
        np.testing.assert_equal(
            self.evaluation_object.performer.array_1d_poly, [2, 2, 1, 0]
        )

        # with self.assertRaises(ValueError):
        #     var = OptimizationVariable(
        #         'missing_index',
        #         evaluation_objects=self.evaluation_object,
        #         parameter_path='array_1d_poly_indices.state',
        #     )

        # Event state modifies entire ND polyomial coefficients using `fill_values`
        var = OptimizationVariable(
            "ndarray_poly_full",
            evaluation_objects=self.evaluation_object,
            parameter_path="ndarray_poly_full.state",
        )

        var.value = 2
        np.testing.assert_equal(var.value, 2)
        np.testing.assert_equal(self.evaluation_object.ndarray_poly_full.state, 2)
        np.testing.assert_equal(
            self.evaluation_object.performer.ndarray_poly, [[2, 0, 0, 0], [2, 0, 0, 0]]
        )

        # Event state modifies individual 1D polyomial coefficients
        var = OptimizationVariable(
            "ndarray_poly_indices",
            evaluation_objects=self.evaluation_object,
            parameter_path="ndarray_poly_indices.state",
            indices=0,
        )

        var.value = 2
        np.testing.assert_equal(var.value, 2)
        np.testing.assert_equal(
            self.evaluation_object.ndarray_poly_indices.state, [2, 1]
        )
        np.testing.assert_equal(
            self.evaluation_object.performer.ndarray_poly, [[2, 2, 0, 0], [2, 0, 0, 1]]
        )

        # with self.assertRaises(ValueError):
        #     var = OptimizationVariable(
        #         'missing_index',
        #         evaluation_objects=self.evaluation_object,
        #         parameter_path='ndarray_poly_indices.state',
        #     )

    def test_multi_eval_obj(self):
        """Test setting indexed variables for multiple evaluation objects."""
        evaluation_object_2 = HandlerFixture()

        evt = evaluation_object_2.add_event(
            "multi_index_ND",
            "performer.ndarray",
            [0, 1, 2],
            indices=[(0, 0), (0, 1), (1, 1)],
            time=0,
        )

        var = OptimizationVariable(
            "multi_index_ND",
            evaluation_objects=[self.evaluation_object, evaluation_object_2],
            parameter_path="multi_index_ND.state",
            indices=0,
        )

        with self.assertRaises(IndexError):
            var = OptimizationVariable(
                "index_exceeds_one_eval_obj_state",
                evaluation_objects=[self.evaluation_object, evaluation_object_2],
                parameter_path="multi_index_ND.state",
                indices=2,
            )


def setup_dummy_eval_fun(n_metrics, rng=None):
    if rng is None:
        rng = np.random.default_rng(12345)

    def dummy_eval_fun(x):
        return rng.random(n_metrics)

    return dummy_eval_fun


def dummy_meta_score(f):
    return np.sum(f)


def setup_optimization_problem(
    n_vars=2,
    n_obj=1,
    n_lincon=0,
    n_lineqcon=0,
    n_nonlincon=0,
    n_meta=0,
    bounds=None,
    obj_fun=None,
    nonlincon_fun=None,
    lincons=None,
    lineqcons=None,
    use_diskcache=False,
):
    optimization_problem = OptimizationProblem("simple", use_diskcache=use_diskcache)

    for i_var in range(n_vars):
        if bounds is None:
            lb, ub = (0, 1)
        else:
            lb, ub = bounds[i_var]

        optimization_problem.add_variable(f"var_{i_var}", lb=lb, ub=ub)

    if n_lincon > 0:
        if lincons is None:
            lincons = [
                ([f"var_{i_var}", f"var_{i_var + 1}"], [1, -1], 0)
                for i_var in range(n_lincon)
            ]
        for opt_vars, lhs, b in lincons:
            optimization_problem.add_linear_constraint(opt_vars, lhs, b)

    if n_lineqcon > 0:
        if lineqcons is None:
            lineqcons = [
                ([f"var_{i_var}", f"var_{i_var + 1}"], [1, -1], 0)
                for i_var in range(n_lineqcon)
            ]
        for opt_vars, lhs, beq in lineqcons:
            optimization_problem.add_linear_equality_constraint(opt_vars, lhs, beq)

    if obj_fun is None:
        obj_fun = setup_dummy_eval_fun(n_obj)

    optimization_problem.add_objective(
        obj_fun, n_objectives=n_obj, labels=[f"f_{i}" for i in range(n_obj)]
    )

    if n_nonlincon > 0:
        if nonlincon_fun is None:
            nonlincon_fun = setup_dummy_eval_fun(n_nonlincon)
        optimization_problem.add_nonlinear_constraint(
            nonlincon_fun,
            n_nonlinear_constraints=n_nonlincon,
            labels=[f"g_{i}" for i in range(n_nonlincon)],
            bounds=0.5,
        )

    if n_meta > 0:
        optimization_problem.add_meta_score(dummy_meta_score)

    return optimization_problem


class Test_OptimizationProblemSimple(unittest.TestCase):
    def setUp(self):
        optimization_problem = OptimizationProblem("simple", use_diskcache=False)

        optimization_problem.add_variable("var_0", lb=0, ub=1)
        optimization_problem.add_variable("var_1", lb=0, ub=10)

        self.optimization_problem = optimization_problem

    def tearDown(self):
        shutil.rmtree("./results_simple", ignore_errors=True)
        settings.working_directory = None

    def test_variable_names(self):
        names_expected = ["var_0", "var_1"]
        names = self.optimization_problem.variable_names
        self.assertEqual(names_expected, names)

        # Variable already exists
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable("var_0")

    def test_bounds(self):
        lb_expected = [0, 0]
        lb = self.optimization_problem.lower_bounds
        np.testing.assert_almost_equal(lb_expected, lb)

        ub_expected = [1, 10]
        ub = self.optimization_problem.upper_bounds
        np.testing.assert_almost_equal(ub_expected, ub)

        # lb >= ub
        with self.assertRaises(ValueError):
            self.optimization_problem.add_variable("spam", lb=0, ub=0)


class Test_OptimizationProblemLinCon(unittest.TestCase):
    def setUp(self):
        optimization_problem = setup_optimization_problem(
            n_lincon=1, use_diskcache=False
        )

        self.optimization_problem = optimization_problem

    def test_add_linear_constraints(self):
        self.optimization_problem.add_linear_constraint("var_0")
        self.optimization_problem.add_linear_constraint(["var_0", "var_1"])

        self.optimization_problem.add_linear_constraint(["var_0", "var_1"], [2, 2])
        self.optimization_problem.add_linear_constraint(["var_0", "var_1"], [3, 3], 1)
        self.optimization_problem.add_linear_constraint(["var_0", "var_1"], 4)

        A_expected = np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        )

        A = self.optimization_problem.A
        np.testing.assert_almost_equal(A, A_expected)

        b_expected = np.array([0, 0, 0, 0, 1, 0])
        b = self.optimization_problem.b
        np.testing.assert_almost_equal(b, b_expected)

        # Variable does not exist
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_linear_constraint("inexistent")

        # Incorrect shape
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_linear_constraint("var_0", [])

    def test_add_linear_equality_constraints(self):
        self.optimization_problem.add_linear_equality_constraint("var_0")
        self.optimization_problem.add_linear_equality_constraint(["var_0", "var_1"])

        self.optimization_problem.add_linear_equality_constraint(
            ["var_0", "var_1"], [2, 2]
        )
        self.optimization_problem.add_linear_equality_constraint(
            ["var_0", "var_1"], [3, 3], 1
        )
        self.optimization_problem.add_linear_equality_constraint(["var_0", "var_1"], 4)

        Aeq_expected = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        )

        Aeq = self.optimization_problem.Aeq
        np.testing.assert_almost_equal(Aeq, Aeq_expected)

        beq_expected = np.array([0, 0, 0, 1, 0])
        beq = self.optimization_problem.beq
        np.testing.assert_almost_equal(beq, beq_expected)

        # Variable does not exist
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_linear_equality_constraint("inexistent")

        # Incorrect shape
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_linear_equality_constraint("var_0", [])

    def test_initial_values(self):
        x0_chebyshev_expected = [0.29289322, 0.70710678]
        x0_chebyshev = self.optimization_problem.get_chebyshev_center(
            include_dependent_variables=True
        )
        np.testing.assert_almost_equal(x0_chebyshev, x0_chebyshev_expected)

        x0_seed_1_expected = [[0.3448127, 0.9138096]]
        x0_seed_1 = self.optimization_problem.create_initial_values(1, seed=1)
        np.testing.assert_almost_equal(x0_seed_1, x0_seed_1_expected)

        x0_seed_1_random = self.optimization_problem.create_initial_values(1)

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_1_random, x0_seed_1_expected)

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_1_random, x0_chebyshev_expected)

        x0_seed_10_expected = [
            [0.34481272, 0.91380963],
            [0.13935918, 0.7377129],
            [0.5219434, 0.63404393],
            [0.23933603, 0.8127269],
            [0.30483107, 0.980534],
            [0.11605111, 0.86206301],
            [0.1869498, 0.99896377],
            [0.55986142, 0.65237364],
            [0.49963092, 0.98791549],
            [0.6839696, 0.92668444],
        ]
        x0_seed_10 = self.optimization_problem.create_initial_values(10, seed=1)
        np.testing.assert_almost_equal(x0_seed_10, x0_seed_10_expected)

        x0_seed_10_random = self.optimization_problem.create_initial_values(10)

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_10_random, x0_seed_10_expected)


class Test_OptimizationProblemDepVar(unittest.TestCase):
    def setUp(self):
        optimization_problem = OptimizationProblem("simple", use_diskcache=False)

        optimization_problem.add_variable("foo", lb=0, ub=1)
        optimization_problem.add_variable("bar", lb=0, ub=1)
        optimization_problem.add_variable("spam", lb=0, ub=1)
        optimization_problem.add_variable("eggs", lb=0, ub=1)

        optimization_problem.add_linear_constraint(["foo", "spam"], [-1, 1])
        optimization_problem.add_linear_constraint(["foo", "eggs"], [-1, 1])
        optimization_problem.add_linear_constraint(["eggs", "spam"], [-1, 1])

        def copy_var(var):
            return var

        optimization_problem.add_variable_dependency("spam", "bar", copy_var)

        self.optimization_problem = optimization_problem

    def test_dependencies(self):
        independent_variables_expected = ["foo", "bar", "eggs"]
        independent_variables = self.optimization_problem.independent_variable_names
        self.assertEqual(independent_variables_expected, independent_variables)

        def transform():
            pass

        # Variable does not exist
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable_dependency(
                "inexistent", ["bar", "spam"], transform
            )

        # Dependent Variable does not exist
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable_dependency(
                "foo", ["inexistent", "spam"], transform
            )

        # Variable is already dependent
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable_dependency(
                "spam", ["bar", "spam"], transform
            )

        # Transform is not callable
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable_dependency(
                "spam", ["bar", "spam"], transform=None
            )

        independent_variables = self.optimization_problem.independent_variable_names
        self.assertEqual(independent_variables_expected, independent_variables)

        dependent_variables_expected = ["spam"]
        dependent_variables = self.optimization_problem.dependent_variable_names
        self.assertEqual(dependent_variables_expected, dependent_variables)

        variables_expected = ["foo", "bar", "spam", "eggs"]
        variables = self.optimization_problem.variable_names
        self.assertEqual(variables_expected, variables)

    def test_initial_values_without_dependencies(self):
        x0_chebyshev_expected = [0.70710678, 0.29289322, 0.29289322]
        x0_chebyshev = self.optimization_problem.get_chebyshev_center(
            include_dependent_variables=False
        )
        np.testing.assert_almost_equal(x0_chebyshev, x0_chebyshev_expected)

        variables_expected = [0.70710678, 0.29289322, 0.29289322, 0.29289322]
        variables = self.optimization_problem.get_dependent_values(x0_chebyshev)
        np.testing.assert_almost_equal(variables, variables_expected)

        x0_seed_1_expected = [[0.8324936, 0.279713, 0.6649623]]
        x0_seed_1 = self.optimization_problem.create_initial_values(
            1, seed=1, include_dependent_variables=False
        )
        np.testing.assert_almost_equal(x0_seed_1, x0_seed_1_expected)
        self.assertTrue(
            self.optimization_problem.check_linear_constraints(
                x0_seed_1[0], get_dependent_values=True
            )
        )

        x0_seed_1_random = self.optimization_problem.create_initial_values(
            1, include_dependent_variables=False
        )

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_1_random, x0_seed_1_expected)

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_1_random, x0_chebyshev_expected)

        x0_seed_10_expected = [
            [0.8324936, 0.27971297, 0.66496231],
            [0.73128644, 0.17275154, 0.5115499],
            [0.95771264, 0.19106333, 0.79216119],
            [0.81089326, 0.29062957, 0.6673637],
            [0.76359474, 0.20575487, 0.59221373],
            [0.9596961, 0.6718687, 0.90182574],
            [0.89945427, 0.01653708, 0.49987409],
            [0.92682306, 0.07106034, 0.8258856],
            [0.77868893, 0.4824452, 0.77038195],
            [0.59375802, 0.03129941, 0.22275326],
        ]
        x0_seed_10 = self.optimization_problem.create_initial_values(
            10, seed=1, include_dependent_variables=False
        )
        np.testing.assert_almost_equal(x0_seed_10, x0_seed_10_expected)

        x0_seed_10_random = self.optimization_problem.create_initial_values(
            10, include_dependent_variables=False
        )

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_10_random, x0_seed_10_expected)

    def test_initial_values(self):
        x0_chebyshev_expected = [0.70710678, 0.29289322, 0.29289322, 0.29289322]
        x0_chebyshev = self.optimization_problem.get_chebyshev_center(
            include_dependent_variables=True
        )
        np.testing.assert_almost_equal(x0_chebyshev, x0_chebyshev_expected)

        independent_variables_expected = [0.70710678, 0.29289322, 0.29289322]
        independent_variables = self.optimization_problem.get_independent_values(
            x0_chebyshev
        )
        np.testing.assert_almost_equal(
            independent_variables, independent_variables_expected
        )

        x0_seed_1_expected = [[0.8324936, 0.279713, 0.279713, 0.6649623]]
        x0_seed_1 = self.optimization_problem.create_initial_values(
            1, seed=1, include_dependent_variables=True
        )
        np.testing.assert_almost_equal(x0_seed_1, x0_seed_1_expected)
        self.assertTrue(
            self.optimization_problem.check_linear_constraints(x0_seed_1[0])
        )

        x0_seed_1_random = self.optimization_problem.create_initial_values(
            1, include_dependent_variables=True
        )[0]

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_1_random, x0_seed_1_expected)

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_1_random, x0_chebyshev_expected)

        x0_seed_10_expected = [
            [0.8324936, 0.27971297, 0.27971297, 0.66496231],
            [0.73128644, 0.17275154, 0.17275154, 0.5115499],
            [0.95771264, 0.19106333, 0.19106333, 0.79216119],
            [0.81089326, 0.29062957, 0.29062957, 0.6673637],
            [0.76359474, 0.20575487, 0.20575487, 0.59221373],
            [0.9596961, 0.6718687, 0.6718687, 0.90182574],
            [0.89945427, 0.01653708, 0.01653708, 0.49987409],
            [0.92682306, 0.07106034, 0.07106034, 0.8258856],
            [0.77868893, 0.4824452, 0.4824452, 0.77038195],
            [0.59375802, 0.03129941, 0.03129941, 0.22275326],
        ]
        x0_seed_10 = self.optimization_problem.create_initial_values(
            10, seed=1, include_dependent_variables=True
        )
        np.testing.assert_almost_equal(x0_seed_10, x0_seed_10_expected)

        x0_seed_10_random = self.optimization_problem.create_initial_values(
            10, include_dependent_variables=True
        )

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_10_random, x0_seed_10_expected)


class Test_OptimizationProblemJacobian(unittest.TestCase):
    def setUp(self):
        def single_obj_single_var(x):
            return x[0] ** 2

        name = "single_obj_single_var"
        optimization_problem = OptimizationProblem(name, use_diskcache=False)
        optimization_problem.add_variable("x")
        optimization_problem.add_objective(single_obj_single_var)
        self.single_obj_single_var = optimization_problem

        def single_obj_two_vars(x):
            return x[1] ** 2 - x[0] ** 2

        name = "single_obj_two_vars"
        optimization_problem = OptimizationProblem(name, use_diskcache=False)
        optimization_problem.add_variable("x_1")
        optimization_problem.add_variable("x_2")
        optimization_problem.add_objective(single_obj_two_vars)
        self.single_obj_two_vars = optimization_problem

        def two_obj_single_var(x):
            return [x[0] ** 2, x[0] ** 2]

        name = "two_obj_single_var"
        optimization_problem = OptimizationProblem(name, use_diskcache=False)
        optimization_problem.add_variable("x")
        optimization_problem.add_objective(two_obj_single_var, n_objectives=2)
        self.two_obj_single_var = optimization_problem

        def two_obj_two_var(x):
            return [x[0] ** 2, x[1] ** 2]

        name = "two_obj_two_var"
        optimization_problem = OptimizationProblem(name, use_diskcache=False)
        optimization_problem.add_variable("x_1")
        optimization_problem.add_variable("x_2")
        optimization_problem.add_objective(two_obj_two_var, n_objectives=2)
        self.two_obj_two_var = optimization_problem

    def test_jacobian(self):
        jac_expected = [[4.001]]
        jac = self.single_obj_single_var.objective_jacobian([2])
        np.testing.assert_almost_equal(jac, jac_expected)

        jac_expected = [[0.001]]
        jac = self.single_obj_single_var.objective_jacobian([0])
        np.testing.assert_almost_equal(jac, jac_expected)

        jac_expected = [[-4.001, 4.001]]
        jac = self.single_obj_two_vars.objective_jacobian([2, 2])
        np.testing.assert_almost_equal(jac, jac_expected)

        jac_expected = [[-0.001, 0.001]]
        jac = self.single_obj_two_vars.objective_jacobian([0, 0])
        np.testing.assert_almost_equal(jac, jac_expected)

        jac_expected = [[4.001], [4.001]]
        jac = self.two_obj_single_var.objective_jacobian([2])
        np.testing.assert_almost_equal(jac, jac_expected)

        jac_expected = [[0.001], [0.001]]
        jac = self.two_obj_single_var.objective_jacobian([0])
        np.testing.assert_almost_equal(jac, jac_expected)

        jac_expected = [[4.001, 0], [0, 4.001]]
        jac = self.two_obj_two_var.objective_jacobian([2, 2])
        np.testing.assert_almost_equal(jac, jac_expected)

        jac_expected = [[0.001, 0], [0, 0.001]]
        jac = self.two_obj_two_var.objective_jacobian([0, 0])
        np.testing.assert_almost_equal(jac, jac_expected)


class Test_OptimizationProblemConstraintTransforms(unittest.TestCase):
    """
    for linear transformation of constraints in an `OptimizationProblem`,
    tests if `A_transformed` and `b_transformed` properties are correctly
    computed.
    """

    def setup(self):
        settings.working_directory = "./test_problem"

    def tearDown(self):
        shutil.rmtree("./test_problem", ignore_errors=True)
        settings.working_directory = None

    @staticmethod
    def check_inequality_constraints(X, problem, transformed_space=False):
        """
        evaluates constraints in transformed space. This behavior is using
        the `A_transformed` and `b_transformed` properties of
        `OptimizationProblem`
        """

        if transformed_space:
            A = problem.A_transformed
            b = problem.b_transformed
        else:
            A = problem.A
            b = problem.b

        evaluate_constraints = lambda x: A.dot(x) - b

        lhs = np.array(list(map(evaluate_constraints, X)))
        rhs = 0
        CV = np.all(lhs <= rhs, axis=1)

        return CV

    @staticmethod
    def check_equality_constraints(X, problem, transformed_space=False):
        """
        evaluates constraints in transformed space. This behavior is using
        the `A_transformed` and `b_transformed` properties of
        `OptimizationProblem`
        """

        if transformed_space:
            Aeq = problem.Aeq_transformed
            beq = problem.beq_transformed
        else:
            Aeq = problem.Aeq
            beq = problem.beq

        evaluate_constraints = lambda x: Aeq.dot(x) - beq

        lhs = np.array(list(map(evaluate_constraints, X)))
        rhs = problem.eps_lineq
        CV = np.all(np.abs(lhs) <= rhs, axis=1)

        return CV

    @staticmethod
    def check_constraint_transform(problem, check_constraint_func):
        nvars = problem.n_independent_variables

        rng = np.random.default_rng(seed=72729)
        X = rng.uniform(0, 1, size=(100000, nvars))

        CV = check_constraint_func(X=X, problem=problem, transformed_space=True)

        # extract valid X and untransform then check constraints in
        # untransformed space
        X_valid = X[CV, :]
        X_valid = problem.untransform(X_valid)

        CV_test_valid = check_constraint_func(
            X=X_valid, problem=problem, transformed_space=False
        )

        # extract invalid X and untransform, then check constraints in
        # untransformed space
        X_invalid = X[~CV, :]
        X_invalid = problem.untransform(X_invalid)

        CV_test_invalid = check_constraint_func(
            X=X_invalid, problem=problem, transformed_space=False
        )

        # tests if all valid X remain valid in untransformed space
        assert np.all(CV_test_valid)

        # tests if all invalid X remain invalid in untransformed space
        assert np.all(~CV_test_invalid)

    def test_linear_inequality_constrained_transform(self):
        problem = LinearConstraintsSooTestProblem2(
            transform="linear",
            use_diskcache=False,
        )

        self.check_constraint_transform(problem, self.check_inequality_constraints)

    def test_linear_equality_constrained_transform(self):
        problem = LinearEqualityConstraintsSooTestProblem(
            transform="linear",
            use_diskcache=False,
        )

        self.check_constraint_transform(problem, self.check_equality_constraints)


class Test_OptimizationProblemEvaluator(unittest.TestCase):
    """
    Szenarien für Variablen:
        - Bound Constraints
        - Linear Constraints
        - Dependencies
        - Dependencies with Constraints

    Szenarien für EvaluationObjects:
        - Verschiedene Variablen für verschiedene EvalObjects.
        - Verschiedene Objectives/Constraints für verschiedene EvalObjects.
        - Verschiedene Evaluatoren für verschiedene Objectives/Constraints.
        - Caching nur für ausgewählte Evaluatoren.

    """

    def setUp(self):
        eval_obj = EvaluationObject()

        # Simple case
        optimization_problem = OptimizationProblem(
            "with_evaluator", use_diskcache=False
        )

        optimization_problem.add_evaluation_object(eval_obj)

        optimization_problem.add_variable("scalar_param")
        optimization_problem.add_variable("sized_list_param", lb=0, ub=10, indices=0)

        self.optimization_problem = optimization_problem

        def copy_var(var):
            return var

        optimization_problem.add_variable_dependency(
            "sized_list_param", "scalar_param", copy_var
        )

    def test_variable_names(self):
        names_expected = ["scalar_param", "sized_list_param"]
        names = self.optimization_problem.variable_names
        self.assertEqual(names_expected, names)

        # Variable does not exist in Evaluator
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable("bar", lb=0, ub=1)

        # Check that adding dummy variables still works
        self.optimization_problem.add_variable(
            "bar", evaluation_objects=None, lb=0, ub=1
        )
        names_expected = ["scalar_param", "sized_list_param", "bar"]
        names = self.optimization_problem.variable_names
        self.assertEqual(names_expected, names)

    def test_set_variables(self):
        pass

    def test_indices(self):
        pass

    def test_duplicate_variables(self):
        self.optimization_problem.check_duplicate_variables()

        # Duplicate name
        self.optimization_problem.add_variable("foo", evaluation_objects=None)

        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable("foo", evaluation_objects=None)

        # Duplicate scalar parameter variable
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable(
                name="another_scalar_parameter",
                parameter_path="scalar_param",
            )

        # Duplicate index parameter variable
        self.optimization_problem.add_variable(
            name="another_sized_list_param_var",
            parameter_path="sized_list_param",
            lb=0,
            ub=10,
            indices=1,
        )

        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable(
                name="yet_another_sized_list_param_var",
                parameter_path="sized_list_param",
                lb=0,
                ub=10,
                indices=1,
            )

    def test_cache(self):
        pass


class Test_MultiEvaluationObjects(unittest.TestCase):
    def setUp(self):
        eval_obj_1 = EvaluationObject(name="foo")
        eval_obj_2 = EvaluationObject(name="bar")

        # Simple case
        optimization_problem = OptimizationProblem(
            "with_evaluator", use_diskcache=False
        )

        optimization_problem.add_evaluation_object(eval_obj_1)
        optimization_problem.add_evaluation_object(eval_obj_2)

        optimization_problem.add_variable(
            name="scalar_eval_obj_1",
            parameter_path="scalar_param",
            lb=0,
            ub=1,
            evaluation_objects=[eval_obj_1],
        )
        optimization_problem.add_variable(
            name="scalar_eval_obj_2",
            parameter_path="scalar_param",
            lb=0,
            ub=1,
            evaluation_objects=[eval_obj_2],
        )

        optimization_problem.add_variable(
            name="scalar_both_eval_obj",
            parameter_path="scalar_param_2",
            lb=0,
            ub=1,
        )

        def single_obj_1(eval_obj):
            return 0

        optimization_problem.add_objective(single_obj_1)

        def single_obj_2(eval_obj):
            return 1

        optimization_problem.add_objective(single_obj_2, evaluation_objects=eval_obj_1)

        def multi_obj(eval_obj):
            return [2, 3]

        optimization_problem.add_objective(multi_obj, n_objectives=2)

        self.optimization_problem = optimization_problem

    def test_evaluation(self):
        f_expected = [0, 0, 1, 2, 3, 2, 3]
        f = self.optimization_problem.evaluate_objectives([1, 1, 1])
        np.testing.assert_allclose(f, f_expected)

        f_expected = [[0, 0, 1, 2, 3, 2, 3], [0, 0, 1, 2, 3, 2, 3]]
        f = self.optimization_problem.evaluate_objectives([[1, 1, 1], [1, 1, 1]])

        np.testing.assert_allclose(f, f_expected)

    def test_names(self):
        names_expected = ["single_obj_1", "single_obj_2", "multi_obj"]
        names = self.optimization_problem.objective_names
        self.assertEqual(names, names_expected)

    def test_labels(self):
        labels_expected = [
            "foo_single_obj_1",
            "bar_single_obj_1",
            "single_obj_2",
            "foo_multi_obj_0",
            "bar_multi_obj_0",
            "foo_multi_obj_1",
            "bar_multi_obj_1",
        ]
        labels = self.optimization_problem.objective_labels
        self.assertEqual(labels, labels_expected)


if __name__ == "__main__":
    unittest.main()
