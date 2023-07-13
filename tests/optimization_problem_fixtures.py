"""

TODO:
- [ ] Add documentation / scope of tests (e.g. link to scipy/pymoo)

"""

import numpy as np
from CADETProcess.optimization import OptimizationProblem, OptimizationResults
from CADETProcess.transform import NormLinearTransform, NormLogTransform

__all__ = [
    'Rosenbrock',
    'LinearConstraintsSooTestProblem',
    'LinearConstraintsMooTestProblem',
    'NonlinearConstraintsMooTestProblem',
]


class TestProblem(OptimizationProblem):
    @property
    def optimal_solution(self):
        """Must return X, F, and if it has, G."""
        raise NotImplementedError

    def test_if_solved(self):
        raise NotImplementedError

    @property
    def x0(self):
        raise NotImplementedError


class Rosenbrock(TestProblem):
    def __init__(self, *args, n_var=2, **kwargs):
        super().__init__('rosenbrock', *args, **kwargs)

        if n_var not in [1, 2, 3, 4, 5, 6, 7]:
            raise ValueError('n_var must be 1 or 2')

        self.add_variable('var_0', lb=-10, ub=10)
        if n_var == 2:
            self.add_variable('var_1', lb=-10, ub=10)

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
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    @classmethod
    def rosen_1D(cls, x):
        """Wrapper to ensure x is 2D before passing to actual function."""

        x_ = np.array([x, 1])
        return cls.rosen(x_)

    @property
    def optimal_solution(self):
        x = np.repeat(1, self.n_variables)
        f = 0

        return x, f

    @property
    def x0(self):
        return np.repeat(0.5, self.n_variables)


class LinearConstraintsSooTestProblem(TestProblem):
    def __init__(self, transform=None, *args, **kwargs):
        super().__init__('linear_constraints_single_objective', *args, **kwargs)
        self.setup_variables(transform=transform)
        self.setup_linear_constraints()
        self.add_objective(self._objective_function)

    def setup_variables(self, transform):
        self.add_variable('var_0', lb=-2, ub=2, transform=transform)
        self.add_variable('var_1', lb=-2, ub=2, transform=transform)

    def setup_linear_constraints(self):
        self.add_linear_constraint(['var_0', 'var_1'], [-1, -0.5], 0)

    def _objective_function(self, x):
        return x[0] - x[1]

    def optimal_solution(self):
        x = [-1, 2]
        f = -3

        return x, f

    def test_if_solved(self, optimization_results: OptimizationResults, decimal=7):
        x_true, f_true = self.optimal_solution()
        x = optimization_results.x
        f = optimization_results.f

        np.testing.assert_almost_equal(f-f_true, 0, decimal=decimal)
        np.testing.assert_almost_equal(x-x_true, 0, decimal=decimal)


class LinearConstraintsSooTestProblem2(TestProblem):
    def __init__(
            self,
            transform=None,
            *args, **kwargs
        ):
        super().__init__(
            "linear_constraints_single_objective_2",
            *args, **kwargs
        )

        self.setup_variables(transform=transform)
        self.setup_linear_constraints()
        self.add_objective(self._objective_function)

    def setup_variables(self: OptimizationProblem, transform=None):
        self.add_variable('var_0', lb=-5, ub=5, transform=transform)
        self.add_variable('var_1', lb=-5, ub=5, transform=transform)
        self.add_variable('var_2', lb=-5, ub=5, transform=transform)

    def setup_linear_constraints(self):
        # cuts off upper right corner of var 0 and var 1
        self.add_linear_constraint(['var_0', 'var_1'], [1, 2], 8)
        # halfs the cube along the diagonal plane and allows only values above
        self.add_linear_constraint(['var_0', 'var_1', 'var_2'], [-1, -1, -0.5], 0)
        # ???
        self.add_linear_constraint(['var_1', 'var_2'], [0.5, -2], 4)

    def _objective_function(self, x):
        return 2 * x[0] - x[1] + 0.5 * x[2]


class LinearEqualityConstraintsSooTestProblem(TestProblem):
    def __init__(
            self,
            transform=None,
            *args, **kwargs
        ):
        super().__init__(
            "linear_equality_constraints_single_objective",
            *args, **kwargs
        )

        self.setup_variables(transform=transform)
        self.setup_linear_constraints()
        self.add_objective(self._objective_function)

    def setup_variables(self: OptimizationProblem, transform=None):
        self.add_variable('var_0', lb=-5, ub=5, transform=transform)
        self.add_variable('var_1', lb=-5, ub=5, transform=transform)
        self.add_variable('var_2', lb=-5, ub=5, transform=transform)

    def setup_linear_constraints(self):
        self.add_linear_equality_constraint(
            ['var_0', 'var_1'], [1.0,  2.0], 8, eps=1e-3
        )

    @property
    def x0(self):
        return np.array([2.0, 3.0, 0.0])

    def _objective_function(self, x):
        return 2 * x[0] - x[1] + 0.5 * x[2]

    @property
    def optimal_solution(self):
        x = np.array([-2, 5, -5])
        f = self._objective_function(x)

        return x, f

    def test_if_solved(self, optimization_results: OptimizationResults, decimal=7):
        x_true, f_true = self.optimal_solution
        x = optimization_results.x
        f = optimization_results.f

        np.testing.assert_almost_equal(f-f_true, 0, decimal=decimal)
        np.testing.assert_almost_equal(x-x_true, 0, decimal=decimal)


class LinearConstraintsMooTestProblem(TestProblem):
    """Function curtesy of Florian Schunck and Samuel Leweke."""

    def __init__(self, *args, **kwargs):
        super().__init__('linear_constraints_multi_objective', *args, **kwargs)

        self.add_variable('var_0', lb=1, ub=5)
        self.add_variable('var_1', lb=0, ub=3)

        self.add_linear_constraint(['var_0', 'var_1'], [-1, -1], -3)
        self.add_linear_constraint(['var_0', 'var_1'], [ 1, -1],  5)

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
        return np.where(x1 <= 3, 3 - x1, 0)

    def optimal_solution(self):
        x1 = np.linspace(1, 5, 101)
        x2 = self.find_corresponding_x2(x1=x1)
        X = np.column_stack([x1, x2])

        F = np.array(list(map(self._objective_function, X)))

        return X, F

    def test_if_solved(self, optimization_results, decimal=7):
        flag = False

        X = optimization_results.x

        x1, x2 = X.T
        x2_test = np.where(x1 <= 3, 3 - x1, 0)

        np.testing.assert_almost_equal(x2, x2_test, decimal=decimal)


class NonlinearConstraintsMooTestProblem(TestProblem):

    def __init__(self, *args, **kwargs):
        from pymoo.problems.multi import SRN
        self._problem = SRN()

        super().__init__('nonlinear_constraints_multi_objective', *args, **kwargs)

        self.add_variable('var_0', lb=-20, ub=20)
        self.add_variable('var_1', lb=-20, ub=20)

        self.add_objective(self._objective_function, n_objectives=2)
        self.add_nonlinear_constraint(self._nonlincon_fun, n_nonlinear_constraints=2)

    def _objective_function(self, x):
        return self._problem.evaluate(x)[0]

    def _nonlincon_fun(self, x):
        return self._problem.evaluate(x)[1]

    def optimal_solution(self):
        X = self._problem.pareto_set()
        F = self._problem.pareto_front()
        # G = ???

        return X, F     # G ???

    def test_if_solved(self, optimization_results, decimal=7):
        X = optimization_results.x_transformed
        x1, x2 = X.T

        np.testing.assert_almost_equal(x1, -2.5, decimal=decimal)
        assert np.all(2.5 <= x2 <= 14.7902)
