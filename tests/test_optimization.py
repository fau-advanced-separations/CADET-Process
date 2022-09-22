from abc import ABC, abstractmethod
import math
import time
import unittest

from addict import Dict
import numpy as np

from CADETProcess.optimization import OptimizationProblem


class EvaluationObject():
    def __init__(self, name='Dummy'):
        self.name = name
        self.dummy_parameter = 1
        self.component_parameter = [1, 2]
        self.polynomial_parameter = [1, 1]

    def __str__(self):
        return self.name

    @property
    def parameters(self):
        return {
            'dummy_parameter': self.dummy_parameter,
            'component_parameter': self.component_parameter,
            'polynomial_parameter': self.polynomial_parameter,
        }

    @property
    def polynomial_parameters(self):
        return {
            'polynomial_parameter': self.polynomial_parameter,
        }

    @parameters.setter
    def parameters(self, parameters):
        for key, value in parameters.items():
            setattr(self, key, value)


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
        print('Run expensive evaluator; this takes forever...')

        expensive_results = Dict()
        expensive_results.result = \
            np.array(2*[evaluation_object.dummy_parameter])

        time.sleep(1)

        return expensive_results


class CheapEvaluator(Evaluator):
    """Mockup for a cheap evaluator"""

    def evaluate(self, expensive_results):
        print('Run cheap evaluator.')
        cheap_results = Dict()
        cheap_results.result_1 = expensive_results.result * 2
        cheap_results.result_2 = expensive_results.result * 2

        return cheap_results


def sum_x(x):
    print("run sum(x)")
    return [sum(x)]


def min_results_1(cheap_results):
    print('run MinResult1')
    score = cheap_results.result_1 * 2
    return score.tolist()


def min_results_2(cheap_results):
    print('run MinResult2')
    score = cheap_results.result_2 * 3
    return score.tolist()


from CADETProcess.optimization import OptimizationVariable
from CADETProcess import CADETProcessError


class Test_OptimizationVariable(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.evaluation_object = EvaluationObject()

    def test_component_index(self):
        var = OptimizationVariable(
            'component_parameter',
            evaluation_objects=[self.evaluation_object],
            parameter_path='dummy_parameter',
        )
        with self.assertRaises(CADETProcessError):
            var = OptimizationVariable(
                'dummy_parameter',
                evaluation_objects=[self.evaluation_object],
                parameter_path='dummy_parameter',
                component_index=1
            )

    def test_polynomial_index(self):
        var = OptimizationVariable(
            'dummy_parameter',
            evaluation_objects=[self.evaluation_object],
            parameter_path='polynomial_parameter',
        )
        with self.assertRaises(CADETProcessError):
            var = OptimizationVariable(
                'dummy_parameter',
                evaluation_objects=[self.evaluation_object],
                parameter_path='dummy_parameter',
                polynomial_index=1
            )

    def test_transform(self):
        var = OptimizationVariable(
            'dummy_parameter',
            evaluation_objects=[self.evaluation_object],
            parameter_path='dummy_parameter',
        )

        # Missing bounds
        with self.assertRaises(CADETProcessError):
            var = OptimizationVariable(
                'dummy_parameter',
                evaluation_objects=[self.evaluation_object],
                parameter_path='dummy_parameter',
                transform='auto'
            )


class Test_OptimizationProblemSimple(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        optimization_problem = OptimizationProblem('simple')

        optimization_problem.add_variable('foo', lb=0, ub=1)
        optimization_problem.add_variable('bar', lb=0, ub=10)

        self.optimization_problem = optimization_problem

    def test_variable_names(self):
        names_expected = ['foo', 'bar']
        names = self.optimization_problem.variable_names
        self.assertEqual(names_expected, names)

        # Variable already exists
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable('foo')

    def test_bounds(self):
        lb_expected = [0, 0]
        lb = self.optimization_problem.lower_bounds
        self.assertAlmostEqual(lb_expected, lb)

        ub_expected = [1, 10]
        ub = self.optimization_problem.upper_bounds
        self.assertAlmostEqual(ub_expected, ub)

        # lb >= ub
        with self.assertRaises(ValueError):
            self.optimization_problem.add_variable('spam', lb=0, ub=0)


class Test_OptimizationProblemLinCon(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        optimization_problem = OptimizationProblem('simple')

        optimization_problem.add_variable('foo', lb=0, ub=1)
        optimization_problem.add_variable('bar', lb=0, ub=1)

        optimization_problem.add_linear_constraint(
            ['foo', 'bar'], [1, -1]
        )

        self.optimization_problem = optimization_problem

    def test_add_linear_constraints(self):
        self.optimization_problem.add_linear_constraint('foo')
        self.optimization_problem.add_linear_constraint(['foo', 'bar'])

        self.optimization_problem.add_linear_constraint(['foo', 'bar'], [2, 2])
        self.optimization_problem.add_linear_constraint(
            ['foo', 'bar'], [3, 3], 1
        )

        A_expected = np.array([
            [1., -1.],
            [1., 0.],
            [1., 1.],
            [2., 2.],
            [3., 3.]
        ])

        A = self.optimization_problem.A
        np.testing.assert_almost_equal(A, A_expected)

        b_expected = np.array([0, 0, 0, 0, 1])
        b = self.optimization_problem.b
        np.testing.assert_almost_equal(b, b_expected)

        # Variable does not exist
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_linear_constraint('inexistent')

        # Incorrect shape
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_linear_constraint('foo', [])

    def test_initial_values(self):
        x0_chebyshev_expected = [[0.29289322, 0.70710678]]
        x0_chebyshev = self.optimization_problem.create_initial_values(1, method='chebyshev')
        np.testing.assert_almost_equal(x0_chebyshev, x0_chebyshev_expected)

        x0_seed_1_expected = [[0.07521520978385349, 0.23740528714055203]]
        x0_seed_1 = self.optimization_problem.create_initial_values(
            1, method='random', seed=1
        )
        np.testing.assert_almost_equal(x0_seed_1, x0_seed_1_expected)

        x0_seed_1_random = self.optimization_problem.create_initial_values(
            1, method='random'
        )

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_1_random, x0_seed_1_expected)

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_1_random, x0_chebyshev_expected)

        x0_seed_10_expected = [
            [0.07521520978385349, 0.23740528714055203],
            [0.08901561623633439, 0.8848459849050998],
            [0.6056437344482468, 0.6375645157271195],
            [0.04650251461049663, 0.42015215322147276],
            [0.028714508997914612, 0.5356851486115922],
            [0.07026419443990661, 0.3729188120964332],
            [0.874277202425165, 0.9514913878453785],
            [0.8309271836922101, 0.9528420604182982],
            [0.6599706433015226, 0.9975899348112521],
            [0.10123102816054416, 0.11216937234725632]
        ]
        x0_seed_10 = self.optimization_problem.create_initial_values(
            10, method='random', seed=1
        )
        np.testing.assert_almost_equal(x0_seed_10, x0_seed_10_expected)

        x0_seed_10_random = self.optimization_problem.create_initial_values(
            10, method='random'
        )

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_10_random, x0_seed_10_expected)


class Test_OptimizationProblemDepVar(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        optimization_problem = OptimizationProblem('simple')

        optimization_problem.add_variable('foo', lb=0, ub=1)
        optimization_problem.add_variable('bar', lb=0, ub=1)
        optimization_problem.add_variable('spam', lb=0, ub=1)
        optimization_problem.add_variable('eggs', lb=0, ub=1)

        optimization_problem.add_linear_constraint(['foo', 'spam'], [-1, 1])
        optimization_problem.add_linear_constraint(['foo', 'eggs'], [-1, 1])
        optimization_problem.add_linear_constraint(['eggs', 'spam'], [-1, 1])

        def copy_var(var):
            return var

        optimization_problem.add_variable_dependency('spam', 'bar', copy_var)

        self.optimization_problem = optimization_problem

    def test_dependencies(self):
        independent_variables_expected = ['foo', 'bar', 'eggs']
        independent_variables = self.optimization_problem.independent_variable_names
        self.assertEqual(independent_variables_expected, independent_variables)

        def transform():
            pass

        # Variable does not exist
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable_dependency(
                'inexistent', ['bar', 'spam'], transform
            )

        # Dependent Variable does not exist
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable_dependency(
                'foo', ['inexistent', 'spam'], transform
            )

        # Variable is already dependent
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable_dependency(
                'spam', ['bar', 'spam'], transform
            )

        # Transform is not callable
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable_dependency(
                'spam', ['bar', 'spam'], transform=None
            )

    def test_initial_values(self):
        x0_chebyshev_expected = [[0.79289322, 0.20710678, 0.5]]
        x0_chebyshev = self.optimization_problem.create_initial_values(
            1, method='chebyshev'
        )
        np.testing.assert_almost_equal(x0_chebyshev, x0_chebyshev_expected)

        variables_expected = [0.7928932188134523, 0.2071067811865475, 0.2071067811865475, 0.4999999999999999]
        variables = self.optimization_problem.get_dependent_values(
            x0_chebyshev[0, :]
        )

        self.assertTrue(
            self.optimization_problem.check_linear_constraints(
                x0_chebyshev[0, :], get_dependent_values=True
            )
        )
        self.assertTrue(
            self.optimization_problem.check_linear_constraints(variables)
        )

        x0_seed_1_expected = [[0.683406761128623, 0.20035083186671915, 0.5365875027254742]]
        x0_seed_1 = self.optimization_problem.create_initial_values(
            1, method='random', seed=1
        )
        np.testing.assert_almost_equal(x0_seed_1, x0_seed_1_expected)

        x0_seed_1_random = self.optimization_problem.create_initial_values(
            1, method='random'
        )

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_1_random, x0_seed_1_expected)

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_1_random, x0_chebyshev_expected)

        x0_seed_10_expected = [
           [0.683406761128623, 0.20035083186671915, 0.5365875027254742],
           [0.9980060138045826, 0.08011370410257823, 0.8842271946376223],
           [0.6557173734540802, 0.012250693630149918, 0.5857067436411985],
           [0.898766251850674, 0.5424404067145038, 0.8285357386637899],
           [0.8053261672692843, 0.12771130209522807, 0.17024926604391552],
           [0.8544093697283153, 0.4307907650523974, 0.7573221052037831],
           [0.7159255710200734, 0.042814189490454346, 0.4597295311958366],
           [0.9974163319578235, 0.2959419753378878, 0.4187164993542842],
           [0.6414185523923226, 0.029990684218219377, 0.19992187974530876],
           [0.9522902967426273, 0.5001476668542261, 0.5591670050427271]
        ]
        x0_seed_10 = self.optimization_problem.create_initial_values(
            10, method='random', seed=1
        )
        np.testing.assert_almost_equal(x0_seed_10, x0_seed_10_expected)

        x0_seed_10_random = self.optimization_problem.create_initial_values(
            10, method='random'
        )

        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(x0_seed_10_random, x0_seed_10_expected)


class Test_OptimizationProblemJacobian(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        def single_obj_single_var(x):
            return x[0]**2

        name = 'single_obj_single_var'
        optimization_problem = OptimizationProblem(name)
        optimization_problem.add_variable('x')
        optimization_problem.add_objective(single_obj_single_var)
        self.single_obj_single_var = optimization_problem

        def single_obj_two_vars(x):
            return x[1]**2 - x[0]**2

        name = 'single_obj_two_vars'
        optimization_problem = OptimizationProblem(name)
        optimization_problem.add_variable('x_1')
        optimization_problem.add_variable('x_2')
        optimization_problem.add_objective(single_obj_two_vars)
        self.single_obj_two_vars = optimization_problem

        def two_obj_single_var(x):
            return [x[0]**2, x[0]**2]

        name = 'two_obj_single_var'
        optimization_problem = OptimizationProblem(name)
        optimization_problem.add_variable('x')
        optimization_problem.add_objective(two_obj_single_var, n_objectives=2)
        self.two_obj_single_var = optimization_problem

        def two_obj_two_var(x):
            return [x[0]**2, x[1]**2]

        name = 'two_obj_two_var'
        optimization_problem = OptimizationProblem(name)
        optimization_problem.add_variable('x_1')
        optimization_problem.add_variable('x_2')
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

        jac_expected = [
            [4.001, 0],
            [0, 4.001]
        ]
        jac = self.two_obj_two_var.objective_jacobian([2, 2])
        np.testing.assert_almost_equal(jac, jac_expected)

        jac_expected = [
            [0.001, 0],
            [0, 0.001]
        ]
        jac = self.two_obj_two_var.objective_jacobian([0, 0])
        np.testing.assert_almost_equal(jac, jac_expected)


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

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        eval_obj = EvaluationObject()

        # Simple case
        optimization_problem = OptimizationProblem('with_evaluator')

        optimization_problem.add_evaluation_object(eval_obj)

        optimization_problem.add_variable('dummy_parameter')
        optimization_problem.add_variable('component_parameter', lb=0, ub=10)

        self.optimization_problem = optimization_problem

        def copy_var(var):
            return var

        optimization_problem.add_variable_dependency(
            'component_parameter', 'dummy_parameter', copy_var
        )

    def test_variable_names(self):
        names_expected = ['dummy_parameter', 'component_parameter']
        names = self.optimization_problem.variable_names
        self.assertEqual(names_expected, names)

        # Variable does not exist in Evaluator
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable('bar', lb=0, ub=1)

        # Check that adding dummy variables still works
        self.optimization_problem.add_variable(
            'bar', evaluation_objects=None, lb=0, ub=1
        )
        names_expected = ['dummy_parameter', 'component_parameter', 'bar']
        names = self.optimization_problem.variable_names
        self.assertEqual(names_expected, names)

    def test_set_variables(self):
        pass

    def test_cache(self):
        pass


if __name__ == '__main__':
    unittest.main()
