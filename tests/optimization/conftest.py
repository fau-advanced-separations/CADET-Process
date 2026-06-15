import warnings
from functools import partial
from typing import NoReturn

import numpy as np
import pytest
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


class EvaluationObject(Structure):
    """Minimal evaluation object covering all mapper types used in OP tests."""

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


@pytest.fixture
def eval_obj():
    return EvaluationObject()


@pytest.fixture
def op_scalar(eval_obj):
    """OP with a single bounded scalar_param [0, 2]; used by write-through tests."""
    op = OptimizationProblem("rt", use_diskcache=False)
    op.add_evaluation_object(eval_obj)
    op.add_variable("scalar_param", lb=0, ub=2)
    return op


@pytest.fixture
def op_with_dep_var(eval_obj):
    """OP with scalar_param + sized_list_param[0] as a dependent variable."""
    op = OptimizationProblem("with_evaluator", use_diskcache=False)
    op.add_evaluation_object(eval_obj)
    op.add_variable("scalar_param")
    op.add_variable("sized_list_param", lb=0, ub=10, indices=0)
    op.add_variable_dependency("sized_list_param", "scalar_param", lambda var: var)
    return op


# ── Shared factories ─────────────────────────────────────────────────────────


def make_dummy_eval_fun(n_metrics, rng=None):
    if rng is None:
        rng = np.random.default_rng(12345)

    def dummy_eval_fun(x):
        return rng.random(n_metrics)

    return dummy_eval_fun


def dummy_meta_score(f):
    return np.sum(f)


def make_optimization_problem(
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
    """Factory for a configurable OptimizationProblem.

    Used across test_optimization_problem, test_optimization_results,
    test_pymoo, and test_parallelization_adapter.
    """
    optimization_problem = OptimizationProblem("simple", use_diskcache=use_diskcache)

    for i_var in range(n_vars):
        lb, ub = bounds[i_var] if bounds is not None else (0, 1)
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
        obj_fun = make_dummy_eval_fun(n_obj)

    optimization_problem.add_objective(
        obj_fun, n_objectives=n_obj, labels=[f"f_{i}" for i in range(n_obj)]
    )

    if n_nonlincon > 0:
        if nonlincon_fun is None:
            nonlincon_fun = make_dummy_eval_fun(n_nonlincon)
        optimization_problem.add_nonlinear_constraint(
            nonlincon_fun,
            n_nonlinear_constraints=n_nonlincon,
            labels=[f"g_{i}" for i in range(n_nonlincon)],
            bounds=0.5,
        )

    if n_meta > 0:
        optimization_problem.add_meta_score(dummy_meta_score)

    return optimization_problem


# ── Benchmark optimization problems ──────────────────────────────────────────
#
# OptimizationProblem subclasses with known analytical optima, used by
# test_optimizer_behavior.py (convergence tests) and test_optimization_problem.py
# (Jacobian parametrization).


error = "Optimizer did not approach solution close enough."
default_test_kwargs = {"rtol": 0.01, "atol": 0.0001, "err_msg": error}


def allow_test_failure_percentage(test_function, test_kwargs, mismatch_tol=0.0):
    """Allow a fraction of element-wise comparisons to fail.

    Used in multi-objective convergence tests where building the full Pareto
    front is slow and a small fraction of dominated solutions is acceptable.
    """
    assert 0.0 <= mismatch_tol <= 1.0, "mismatch_tol must be between 0 and 1."
    try:
        test_function(**test_kwargs)
    except AssertionError as e:
        msg = e.args[0].split("\n")
        lnum, mismatch_line = [
            (i, l) for i, l in enumerate(msg) if "Mismatched elements:" in l  # noqa: E741
        ][0]
        mismatch_percent = float(mismatch_line.split("(")[1].split("%")[0])
        if mismatch_percent / 100 > mismatch_tol:
            err_line = (
                "---> "
                + mismatch_line
                + f" exceeded tolerance ({mismatch_percent}% > {mismatch_tol * 100}%)"
            )
            msg[lnum] = err_line
            raise AssertionError("\n".join(msg))
        else:
            warn_line = (
                mismatch_line
                + f" below tolerance ({mismatch_percent}% <= {mismatch_tol * 100}%)"
            )
            warnings.warn(f"Equality test passed with {warn_line}")


class TestProblem(OptimizationProblem):
    __test__ = False

    @property
    def optimal_solution(self):
        raise NotImplementedError

    def test_if_solved(self, results):
        raise NotImplementedError

    @property
    def x0(self):
        raise NotImplementedError


class Rosenbrock(TestProblem):
    def __init__(self, *args, n_var=2, **kwargs):
        super().__init__("rosenbrock", *args, **kwargs)

        if n_var not in [1, 2, 3, 4, 5, 6, 7]:
            raise ValueError("n_var must be 1 or 2")

        self.add_variable("var_0", lb=-10, ub=10)
        if n_var == 2:
            self.add_variable("var_1", lb=-10, ub=10)

        self.add_objective(self._objective_function)

    def _objective_function(self, x):
        if self.n_variables == 1:
            return self.rosen_1D(x)
        if self.n_variables > 1 and self.n_variables <= 7:
            return self.rosen(x)

    @staticmethod
    def rosen(x):
        """Rosenbrock function for 2-7D optimization problems."""
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    @classmethod
    def rosen_1D(cls, x):
        x_ = np.array([x, 1])
        return cls.rosen(x_)

    @property
    def optimal_solution(self):
        x = np.repeat(1, self.n_variables).reshape(1, self.n_variables)
        f = 0
        return x, f

    @property
    def x0(self):
        return np.repeat(0.9, self.n_variables)

    def test_if_solved(
        self, optimization_results, test_kwargs=default_test_kwargs
    ) -> NoReturn:
        x_true, f_true = self.optimal_solution
        x = optimization_results.x
        f = optimization_results.f

        test_kwargs["err_msg"] = error
        np.testing.assert_allclose(f, f_true, **test_kwargs)
        np.testing.assert_allclose(x, x_true, **test_kwargs)


class LinearConstraintsSooTestProblem(TestProblem):
    def __init__(
        self,
        transform=None,
        has_evaluator=False,
        significant_digits=None,
        *args,
        **kwargs,
    ):
        self.test_abs_tol = 0.1
        super().__init__("linear_constraints_single_objective", *args, **kwargs)
        self.setup_variables(transform=transform, significant_digits=significant_digits)
        self.setup_linear_constraints()
        if has_evaluator:
            eval_fun = lambda x: x
            self.add_evaluator(eval_fun)
            self.add_objective(self._objective_function, requires=eval_fun)
        else:
            self.add_objective(self._objective_function)

    def setup_variables(self, transform, significant_digits=None):
        self.add_variable(
            "var_0", lb=-2, ub=2,
            transform=transform, significant_digits=significant_digits,
        )
        self.add_variable(
            "var_1", lb=-2, ub=2,
            transform=transform, significant_digits=significant_digits,
        )
        self.add_variable(
            "var_2", lb=0, ub=2,
            transform="log", significant_digits=significant_digits,
        )

    def setup_linear_constraints(self):
        self.add_linear_constraint(["var_0", "var_1"], [-1, -0.5], 0)

    def _objective_function(self, x):
        return x[0] - x[1] + x[2]

    @property
    def optimal_solution(self):
        x = np.array([-1, 2, 0.0]).reshape(1, self.n_variables)
        f = -3
        return x, f

    @property
    def x0(self):
        return [-0.5, 1.5, 0.1]

    @property
    def conditional_minima(self):
        f_x0 = lambda x0: x0 - 2
        f_x1 = lambda x1: x1 * -3 / 2
        f_x2 = lambda x2: x2
        return f_x0, f_x1, f_x2

    def test_if_solved(
        self, optimization_results, test_kwargs=default_test_kwargs
    ):
        x_true, f_true = self.optimal_solution
        x = optimization_results.x
        f = optimization_results.f

        test_kwargs["err_msg"] = error
        np.testing.assert_allclose(f, f_true, **test_kwargs)
        np.testing.assert_allclose(x, x_true, **test_kwargs)


class NonlinearConstraintsSooTestProblem(TestProblem):
    def __init__(self, transform=None, has_evaluator=False, *args, **kwargs):
        self.fixture_evaluator = None
        super().__init__("linear_constraints_single_objective", *args, **kwargs)
        self.setup_variables(transform=transform)
        self.setup_evaluator(has_evaluator=has_evaluator)
        self.setup_nonlinear_constraints()
        self.setup_objectives()

    def setup_evaluator(self, has_evaluator):
        if has_evaluator:
            self.fixture_evaluator = lambda x: x
            self.add_evaluator(self.fixture_evaluator)
        else:
            self.fixture_evaluator = None

    def setup_objectives(self):
        self.add_objective(self._objective_function, requires=self.fixture_evaluator)

    def setup_variables(self, transform):
        self.add_variable("var_0", lb=-2, ub=0, transform=transform)
        self.add_variable("var_1", lb=-2, ub=2, transform=transform)

    def setup_nonlinear_constraints(self):
        nlc_fun_0 = lambda x: -1 * x[0] - 0.5 * x[1]
        self.add_nonlinear_constraint(
            nlc_fun_0, bounds=0, n_nonlinear_constraints=1,
            requires=self.fixture_evaluator,
        )

        def nlc_fun_1(x):
            return -0.01 / (1 + np.exp(x[0])) + 0.005, x[1]

        self.add_nonlinear_constraint(
            nlc_fun_1, bounds=[0.001, 2], n_nonlinear_constraints=2,
            requires=self.fixture_evaluator,
        )

    @property
    def x0(self):
        return [-0.5, 1.5]

    def _objective_function(self, x):
        return x[0] - x[1]

    @property
    def optimal_solution(self):
        x = np.array([-1, 2]).reshape(1, self.n_variables)
        f = -3
        return x, f

    def test_if_solved(
        self, optimization_results, test_kwargs=default_test_kwargs
    ):
        x_true, f_true = self.optimal_solution
        x = optimization_results.x
        f = optimization_results.f

        test_kwargs["err_msg"] = error
        np.testing.assert_allclose(f, f_true, **test_kwargs)
        np.testing.assert_allclose(x, x_true, **test_kwargs)


class LinearConstraintsSooTestProblem2(TestProblem):
    def __init__(self, transform=None, *args, **kwargs):
        super().__init__("linear_constraints_single_objective_2", *args, **kwargs)
        self.setup_variables(transform=transform)
        self.setup_linear_constraints()
        self.add_objective(self._objective_function)

    def setup_variables(self, transform=None):
        self.add_variable("var_0", lb=-5, ub=5, transform=transform)
        self.add_variable("var_1", lb=-5, ub=5, transform=transform)
        self.add_variable("var_2", lb=-5, ub=5, transform=transform)

    def setup_linear_constraints(self):
        self.add_linear_constraint(["var_0", "var_1"], [1, 2], 8)
        self.add_linear_constraint(["var_0", "var_1", "var_2"], [-1, -1, -0.5], 0)
        self.add_linear_constraint(["var_1", "var_2"], [0.5, -2], 4)

    def _objective_function(self, x):
        return 2 * x[0] - x[1] + 0.5 * x[2]

    @property
    def x0(self):
        return [-4, 4, 1]

    @property
    def optimal_solution(self):
        x = np.array([-5.0, 5.0, 0.0]).reshape(1, self.n_variables)
        f = -15.0
        return x, f

    def test_if_solved(
        self, optimization_results, test_kwargs=default_test_kwargs
    ):
        x_true, f_true = self.optimal_solution
        x = optimization_results.x
        f = optimization_results.f

        test_kwargs["err_msg"] = error
        np.testing.assert_allclose(f, f_true, **test_kwargs)
        np.testing.assert_allclose(x, x_true, **test_kwargs)


class LinearEqualityConstraintsSooTestProblem(TestProblem):
    def __init__(self, transform=None, *args, **kwargs):
        super().__init__(
            "linear_equality_constraints_single_objective", *args, **kwargs
        )
        self.setup_variables(transform=transform)
        self.setup_linear_constraints()
        self.add_objective(self._objective_function)

    def setup_variables(self, transform=None, significant_digits=None):
        self.add_variable(
            "var_0", lb=-5, ub=5,
            transform=transform, significant_digits=significant_digits,
        )
        self.add_variable(
            "var_1", lb=-5, ub=5,
            transform=transform, significant_digits=significant_digits,
        )
        self.add_variable(
            "var_2", lb=-5, ub=5,
            transform=transform, significant_digits=significant_digits,
        )

    def setup_linear_constraints(self):
        self.add_linear_equality_constraint(["var_0", "var_1"], [1.0, 2.0], 8)

    @property
    def x0(self):
        return np.array([-1.0, 4.5, -4.0])

    def _objective_function(self, x):
        return 2 * x[0] - x[1] + 0.5 * x[2]

    @property
    def optimal_solution(self):
        x = np.array([-2, 5, -5])
        f = self._objective_function(x)
        return x.reshape(1, self.n_variables), f

    def test_if_solved(
        self, optimization_results, test_kwargs=default_test_kwargs
    ):
        x_true, f_true = self.optimal_solution
        x = optimization_results.x
        f = optimization_results.f

        test_kwargs["err_msg"] = error
        np.testing.assert_allclose(f, f_true, **test_kwargs)
        np.testing.assert_allclose(x, x_true, **test_kwargs)


class NonlinearLinearConstraintsSooTestProblem(TestProblem):
    def __init__(self, transform=None, *args, **kwargs):
        self.test_tol = 0.1
        super().__init__(
            "nonlinear_linear_constraints_single_objective", *args, **kwargs
        )
        self.setup_variables(transform=transform)
        self.setup_linear_constraints()
        self.setup_nonlinear_constraints()
        self.add_objective(self._objective_function)

    def setup_variables(self, transform):
        self.add_variable("var_0", lb=-2, ub=2, transform=transform)
        self.add_variable("var_1", lb=-2, ub=2, transform=transform)

    def setup_linear_constraints(self):
        self.add_linear_constraint(["var_0", "var_1"], [-1, -0.5], 0)

    def setup_nonlinear_constraints(self):
        f_nonlinconc = lambda x: np.array([(x[0] + x[1]) ** 2])
        self.add_nonlinear_constraint(f_nonlinconc, "nonlincon_0", bounds=4)

    def _objective_function(self, x):
        return x[0] - x[1]

    @property
    def x0(self):
        return [-0.5, 1.5]

    @property
    def optimal_solution(self):
        x = np.array([-1, 2]).reshape(1, self.n_variables)
        f = -3
        return x, f

    def test_if_solved(
        self, optimization_results, test_kwargs=default_test_kwargs
    ):
        x_true, f_true = self.optimal_solution
        x = optimization_results.x
        f = optimization_results.f

        test_kwargs["err_msg"] = error
        np.testing.assert_allclose(f, f_true, **test_kwargs)
        np.testing.assert_allclose(x, x_true, **test_kwargs)


class LinearConstraintsMooTestProblem(TestProblem):
    """Function courtesy of Florian Schunck and Samuel Leweke."""

    def __init__(self, transform=None, *args, **kwargs):
        self.test_abs_tol = 0.1
        super().__init__("linear_constraints_multi_objective", *args, **kwargs)
        self.setup_variables(transform=transform)
        self.setup_linear_constraints()
        self.setup_objectives()

    def setup_variables(self, transform=None):
        self.add_variable("var_0", lb=1, ub=5, transform=transform)
        self.add_variable("var_1", lb=0, ub=3, transform=transform)

    def setup_linear_constraints(self):
        self.add_linear_constraint(["var_0", "var_1"], [-1, -1], -3)
        self.add_linear_constraint(["var_0", "var_1"], [1, -1], 5)

    @staticmethod
    def _objective_function(x):
        f1 = x[0]
        f2 = (1 + x[1]) / x[0]
        return f1, f2

    def setup_objectives(self):
        def f1(x):
            return self._objective_function(x)[0]

        def f2(x):
            return self._objective_function(x)[1]

        self.add_objective(f1, n_objectives=1)
        self.add_objective(f2, n_objectives=1)

    def find_corresponding_x2(self, x1):
        return np.where(x1 <= 3, 3 - x1, 0)

    @property
    def conditional_minima(self):
        def f_x0(x0):
            f1 = x0
            f2 = np.where(x0 <= 3, (1 + -x0 + 3) / x0, (1 + 0) / x0)
            return np.array([f1, f2])

        def f_x1(x1):
            f1 = np.where(x1 <= 2, -x1 + 3, 1)
            f2 = (1 + x1) / 5
            return np.array([f1, f2])

        return f_x0, f_x1

    @property
    def x0(self):
        return [1.6, 1.4]

    @property
    def optimal_solution(self):
        x1 = np.linspace(1, 5, 101)
        x2 = self.find_corresponding_x2(x1=x1)
        X = np.column_stack([x1, x2])
        F = np.array(list(map(self._objective_function, X)))
        return X, F

    def test_if_solved(
        self, optimization_results, test_kwargs=default_test_kwargs
    ) -> NoReturn:
        X = optimization_results.x
        x1, x2 = X.T
        x2_test = np.where(x1 <= 3, 3 - x1, 0)

        test_kwargs_ = test_kwargs.copy()
        test_kwargs_["err_msg"] = error
        mismatch_tol = test_kwargs_.pop("mismatch_tol", 0.0)
        test_func = partial(np.testing.assert_allclose, actual=x2, desired=x2_test)

        allow_test_failure_percentage(
            test_function=test_func, test_kwargs=test_kwargs_, mismatch_tol=mismatch_tol
        )


class LinearNonlinearConstraintsMooTestProblem(TestProblem):
    """Function courtesy of Florian Schunck and Samuel Leweke."""

    def __init__(self, has_evaluator=False, *args, **kwargs):
        super().__init__("linear_constraints_multi_objective", *args, **kwargs)
        self.setup_variables()
        self.setup_linear_constraints()
        self.setup_nonlinear_constraints()
        self.setup_objectives(has_evaluator=has_evaluator)

    def setup_variables(self):
        self.add_variable("var_0", lb=1, ub=5)
        self.add_variable("var_1", lb=0, ub=3)

    def setup_linear_constraints(self):
        self.add_linear_constraint(["var_0", "var_1"], [-1, -1], -2)
        self.add_linear_constraint(["var_0", "var_1"], [1, -1], 5)

    def setup_nonlinear_constraints(self):
        f_nonlinconc_0 = lambda x: np.array([x[0] ** 2, x[1] ** 2])
        f_nonlinconc_1 = lambda x: np.array([x[0] ** 1.1, x[1] ** 1.1])

        self.add_nonlinear_constraint(
            nonlincon=f_nonlinconc_0, name="nonlincon_0",
            bounds=4, n_nonlinear_constraints=2,
        )
        self.add_nonlinear_constraint(
            nonlincon=f_nonlinconc_1, name="nonlincon_1",
            bounds=3, n_nonlinear_constraints=2,
        )

    def setup_objectives(self, has_evaluator):
        if has_evaluator:
            eval_fun = lambda x: x
            self.add_evaluator(eval_fun)
            self.add_objective(
                objective=self._objective_function,
                requires=eval_fun, n_objectives=2,
            )
        else:
            self.add_objective(self._objective_function, n_objectives=2)

    @staticmethod
    def _objective_function(x):
        f1 = x[0]
        f2 = (1 + x[1]) / x[0]
        return f1, f2

    def find_corresponding_x2(self, x1):
        return np.where(x1 <= 2, 2 - x1, 0)

    @property
    def x0(self):
        return [1.6, 1.4]

    @property
    def optimal_solution(self):
        x1 = np.linspace(1, 5, 101)
        x2 = self.find_corresponding_x2(x1=x1)
        X = np.column_stack([x1, x2])
        F = np.array(list(map(self._objective_function, X)))
        return X, F

    def test_if_solved(
        self, optimization_results, test_kwargs=default_test_kwargs
    ):
        X = optimization_results.x
        x1, x2 = X.T
        x2_test = self.find_corresponding_x2(x1)

        test_kwargs_ = test_kwargs.copy()
        test_kwargs_["err_msg"] = error
        mismatch_tol = test_kwargs_.pop("mismatch_tol", 0.0)
        test_func = partial(np.testing.assert_allclose, actual=x2, desired=x2_test)

        allow_test_failure_percentage(
            test_function=test_func, test_kwargs=test_kwargs_, mismatch_tol=mismatch_tol
        )


class NonlinearConstraintsMooTestProblem(TestProblem):
    def __init__(self, has_evaluator=False, *args, **kwargs):
        from pymoo.problems.multi import SRN

        self._problem = SRN()
        self.fixture_evaluator = None
        super().__init__("nonlinear_constraints_multi_objective", *args, **kwargs)

        self.add_variable("var_0", lb=-20, ub=20)
        self.add_variable("var_1", lb=-20, ub=20)
        self.setup_evaluator(has_evaluator=has_evaluator)
        self.setup_nonlinear_constraints()
        self.setup_objectives()

    def setup_evaluator(self, has_evaluator):
        if has_evaluator:
            self.fixture_evaluator = lambda x: x
            self.add_evaluator(self.fixture_evaluator)
        else:
            self.fixture_evaluator = None

    def setup_nonlinear_constraints(self):
        self.add_nonlinear_constraint(
            nonlincon=self._nonlincon_fun,
            requires=self.fixture_evaluator,
            n_nonlinear_constraints=2,
        )

    def setup_objectives(self):
        self.add_objective(
            objective=self._objective_function,
            requires=self.fixture_evaluator,
            n_objectives=2,
        )

    def _objective_function(self, x):
        return self._problem.evaluate(x)[0]

    def _nonlincon_fun(self, x):
        return self._problem.evaluate(x)[1]

    @property
    def x0(self):
        return [-2.4, 5.0]

    @property
    def optimal_solution(self):
        X = self._problem.pareto_set()
        F = self._problem.pareto_front()
        return X, F

    def test_if_solved(
        self, optimization_results, test_kwargs=default_test_kwargs
    ) -> NoReturn:
        X = optimization_results.x_transformed
        x1, x2 = X.T

        test_kwargs_ = test_kwargs.copy()
        mismatch_tol = test_kwargs_.pop("mismatch_tol", 0.0)
        test_kwargs_["err_msg"] = error

        test_func_1 = partial(np.testing.assert_allclose, actual=x1, desired=-2.5)
        test_func_2 = partial(np.testing.assert_array_less, x=x2, y=14.7902)
        test_func_3 = partial(np.testing.assert_array_less, x=-x2, y=-2.5)

        allow_test_failure_percentage(
            test_function=test_func_1, test_kwargs=test_kwargs_,
            mismatch_tol=mismatch_tol,
        )
        allow_test_failure_percentage(
            test_function=test_func_2, test_kwargs={}, mismatch_tol=mismatch_tol
        )
        allow_test_failure_percentage(
            test_function=test_func_3, test_kwargs={}, mismatch_tol=mismatch_tol
        )
