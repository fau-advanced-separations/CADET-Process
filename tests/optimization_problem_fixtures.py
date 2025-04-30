"""

TODO:
- [ ] Add documentation / scope of tests (e.g. link to scipy/pymoo)

"""

import warnings
from functools import partial
from typing import NoReturn

import numpy as np
from CADETProcess.optimization import OptimizationProblem, OptimizationResults

__all__ = [
    "Rosenbrock",
    "LinearConstraintsSooTestProblem",
    "LinearConstraintsSooTestProblem2",
    "LinearEqualityConstraintsSooTestProblem",
    "NonlinearConstraintsSooTestProblem",
    "NonlinearLinearConstraintsSooTestProblem",
    "LinearConstraintsMooTestProblem",
    "LinearNonlinearConstraintsMooTestProblem",
    "NonlinearConstraintsMooTestProblem",
]


error = "Optimizer did not approach solution close enough."
default_test_kwargs = {"rtol": 0.01, "atol": 0.0001, "err_msg": error}


def allow_test_failure_percentage(test_function, test_kwargs, mismatch_tol=0.0):
    """When two arrays are compared, allow a certain fraction of the comparisons
    to fail. This behaviour is explicitly accepted in convergence tests of
    multi-objective tests. The reason behind it is that building the pareto
    front takes time and removing dominated solutions from the frontier can
    take a long time. Hence accepting a certain fraction of dominated-solutions
    is acceptable, when the majority points lies on the pareto front.

    While of course for full convergence checks the mismatch tolerance should
    be reduced to zero, for normal testing the fraction can be raised to
    say 0.25. This value can be adapted for easy or difficult cases.
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
    # To prevent Pytest interpreting this class as test:
    __test__ = False

    @property
    def optimal_solution(self):
        """Must return X, F, and if it has, G."""
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
        """Rosenbrock function for 2-D, 3-D and 4-7-D optimization problems.

        for 2-D and 3-D it has exactly 1 optimal solution which is
        f(x=(1,..,1)) = 0

        for 4-7 D it has the same global minimum but in addition a local optimum
        at x=(-1, 1, ..., 1)
        """
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    @classmethod
    def rosen_1D(cls, x):
        """Wrapper to ensure x is 2D before passing to actual function."""

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
        self, optimization_results: OptimizationResults, test_kwargs=default_test_kwargs
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
            "var_0",
            lb=-2,
            ub=2,
            transform=transform,
            significant_digits=significant_digits,
        )
        self.add_variable(
            "var_1",
            lb=-2,
            ub=2,
            transform=transform,
            significant_digits=significant_digits,
        )
        self.add_variable(
            "var_2",
            lb=0,
            ub=2,
            transform="log",
            significant_digits=significant_digits,
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
        self, optimization_results: OptimizationResults, test_kwargs=default_test_kwargs
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
        """
        these should reproduce the same results as above only with nonlinear
        constraints.
        TODO: Bounds are probably redundant
        """
        nlc_fun_0 = lambda x: -1 * x[0] - 0.5 * x[1]
        self.add_nonlinear_constraint(
            nlc_fun_0,
            bounds=0,
            n_nonlinear_constraints=1,
            requires=self.fixture_evaluator,
        )

        def nlc_fun_1(x):
            return -0.01 / (1 + np.exp(x[0])) + 0.005, x[1]

        self.add_nonlinear_constraint(
            nlc_fun_1,
            bounds=[0.001, 2],
            n_nonlinear_constraints=2,
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
        self, optimization_results: OptimizationResults, test_kwargs=default_test_kwargs
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

    def setup_variables(self: OptimizationProblem, transform=None):
        self.add_variable("var_0", lb=-5, ub=5, transform=transform)
        self.add_variable("var_1", lb=-5, ub=5, transform=transform)
        self.add_variable("var_2", lb=-5, ub=5, transform=transform)

    def setup_linear_constraints(self):
        # cuts off upper right corner of var 0 and var 1
        self.add_linear_constraint(["var_0", "var_1"], [1, 2], 8)
        # halfs the cube along the diagonal plane and allows only values above
        self.add_linear_constraint(["var_0", "var_1", "var_2"], [-1, -1, -0.5], 0)
        # ???
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
        self, optimization_results: OptimizationResults, test_kwargs=default_test_kwargs
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

    def setup_variables(
        self: OptimizationProblem, transform=None, significant_digits=None
    ):
        self.add_variable(
            "var_0",
            lb=-5,
            ub=5,
            transform=transform,
            significant_digits=significant_digits,
        )
        self.add_variable(
            "var_1",
            lb=-5,
            ub=5,
            transform=transform,
            significant_digits=significant_digits,
        )
        self.add_variable(
            "var_2",
            lb=-5,
            ub=5,
            transform=transform,
            significant_digits=significant_digits,
        )

    def setup_linear_constraints(self):
        self.add_linear_equality_constraint(["var_0", "var_1"], [1.0, 2.0], 8, eps=1e-3)

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
        self, optimization_results: OptimizationResults, test_kwargs=default_test_kwargs
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
        self, optimization_results: OptimizationResults, test_kwargs=default_test_kwargs
    ):
        x_true, f_true = self.optimal_solution
        x = optimization_results.x
        f = optimization_results.f

        test_kwargs["err_msg"] = error
        np.testing.assert_allclose(f, f_true, **test_kwargs)
        np.testing.assert_allclose(x, x_true, **test_kwargs)


class LinearConstraintsMooTestProblem(TestProblem):
    """Function curtesy of Florian Schunck and Samuel Leweke."""

    def __init__(self, transform=None, *args, **kwargs):
        self.test_abs_tol = 0.1

        super().__init__("linear_constraints_multi_objective", *args, **kwargs)
        self.setup_variables(transform=transform)
        self.setup_linear_constraints()
        self.setup_objectives()

    def setup_variables(self: OptimizationProblem, transform=None):
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
        """
        in a point x in a pareto set
        """
        return np.where(x1 <= 3, 3 - x1, 0)

    @property
    def conditional_minima(self):
        def f_x0(x0):
            f1 = x0
            # solve constraints with respect to x1 and substitute in a way
            # that minimizes f2
            # when x0 <= 3 the first linear constraint is dominating,
            # when x0 > 3 the boundary constraint of x1 is dominating
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
        self, optimization_results: OptimizationResults, test_kwargs=default_test_kwargs
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
    """Function curtesy of Florian Schunck and Samuel Leweke."""

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
            nonlincon=f_nonlinconc_0,
            name="nonlincon_0",
            bounds=4,
            n_nonlinear_constraints=2,
        )

        self.add_nonlinear_constraint(
            nonlincon=f_nonlinconc_1,
            name="nonlincon_1",
            bounds=3,
            n_nonlinear_constraints=2,
        )

    def setup_objectives(self, has_evaluator):
        if has_evaluator:
            eval_fun = lambda x: x
            self.add_evaluator(eval_fun)
            self.add_objective(
                objective=self._objective_function,
                requires=eval_fun,
                n_objectives=2,
            )
        else:
            self.add_objective(self._objective_function, n_objectives=2)

    @staticmethod
    def _objective_function(x):
        f1 = x[0]
        f2 = (1 + x[1]) / x[0]

        return f1, f2

    def find_corresponding_x2(self, x1):
        """
        in a point x in a pareto set
        """
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
        self, optimization_results: OptimizationResults, test_kwargs=default_test_kwargs
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
        # TODO: test nonlinear constraints as well.

        return X, F  # G ???

    def test_if_solved(
        self, optimization_results: OptimizationResults, test_kwargs=default_test_kwargs
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
            test_function=test_func_1,
            test_kwargs=test_kwargs_,
            mismatch_tol=mismatch_tol,
        )

        allow_test_failure_percentage(
            test_function=test_func_2, test_kwargs={}, mismatch_tol=mismatch_tol
        )

        allow_test_failure_percentage(
            test_function=test_func_3, test_kwargs={}, mismatch_tol=mismatch_tol
        )
