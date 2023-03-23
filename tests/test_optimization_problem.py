from abc import ABC, abstractmethod
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


def setup_dummy_eval_fun(n_metrics):
    def dummy_eval_fun(x):
        return np.random(n_metrics)

    return dummy_eval_fun


def dummy_meta_score(f):
    return np.sum(f)


def setup_optimization_problem(
        n_vars=2, n_obj=1, n_nonlincon=0, n_meta=0, use_diskcache=False):
    optimization_problem = OptimizationProblem('simple', use_diskcache=use_diskcache)

    for i_var in range(n_vars):
        optimization_problem.add_variable(f'var_{i_var}', lb=0, ub=1)

    optimization_problem.add_objective(setup_dummy_eval_fun(n_obj), n_objectives=n_obj)

    if n_nonlincon > 0:
        optimization_problem.add_nonlinear_constraint(
            setup_dummy_eval_fun(n_nonlincon), n_nonlinear_constraints=n_nonlincon
        )

    if n_meta > 0:
        optimization_problem.add_meta_score(dummy_meta_score)

    return optimization_problem


class Test_OptimizationProblemSimple(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        optimization_problem = OptimizationProblem('simple', use_diskcache=False)

        optimization_problem.add_variable('var_0', lb=0, ub=1)
        optimization_problem.add_variable('var_1', lb=0, ub=10)

        self.optimization_problem = optimization_problem

    def test_variable_names(self):
        names_expected = ['var_0', 'var_1']
        names = self.optimization_problem.variable_names
        self.assertEqual(names_expected, names)

        # Variable already exists
        with self.assertRaises(CADETProcessError):
            self.optimization_problem.add_variable('var_0')

    def test_bounds(self):
        lb_expected = [0, 0]
        lb = self.optimization_problem.lower_bounds
        np.testing.assert_almost_equal(lb_expected, lb)

        ub_expected = [1, 10]
        ub = self.optimization_problem.upper_bounds
        np.testing.assert_almost_equal(ub_expected, ub)

        # lb >= ub
        with self.assertRaises(ValueError):
            self.optimization_problem.add_variable('spam', lb=0, ub=0)


class Test_OptimizationProblemLinCon(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        optimization_problem = setup_optimization_problem(use_diskcache=False)

        optimization_problem.add_linear_constraint(
            ['var_0', 'var_1'], [1, -1]
        )

        self.optimization_problem = optimization_problem

    def test_add_linear_constraints(self):
        self.optimization_problem.add_linear_constraint('var_0')
        self.optimization_problem.add_linear_constraint(['var_0', 'var_1'])

        self.optimization_problem.add_linear_constraint(['var_0', 'var_1'], [2, 2])
        self.optimization_problem.add_linear_constraint(
            ['var_0', 'var_1'], [3, 3], 1
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
            self.optimization_problem.add_linear_constraint('var_0', [])

    def test_initial_values(self):
        x0_chebyshev_expected = [[0.2928932, 0.7071068]]
        x0_chebyshev = self.optimization_problem.create_initial_values(
            1, method='chebyshev'
        )
        np.testing.assert_almost_equal(x0_chebyshev, x0_chebyshev_expected)

        x0_seed_1_expected = [[0.5666524, 0.8499365]]
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
            [0.566652437432882, 0.8499365450601801],
            [0.16209066464206898, 0.7722304506499793],
            [0.4265526985487967, 0.609778640412627],
            [0.36121046800146495, 0.9264826793698916],
            [0.7875352350909218, 0.8881166149888506],
            [0.2685012584445143, 0.9067761747715452],
            [0.9701611935982989, 0.9809160496933532],
            [0.35086424227614005, 0.4668187637599064],
            [0.8928778441932161, 0.9360696751348305],
            [0.5365699848069944, 0.6516012021958184]
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
        optimization_problem = OptimizationProblem('simple', use_diskcache=False)

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

        independent_variables = self.optimization_problem.independent_variable_names
        self.assertEqual(independent_variables_expected, independent_variables)

        dependent_variables_expected = ['spam']
        dependent_variables = self.optimization_problem.dependent_variable_names
        self.assertEqual(dependent_variables_expected, dependent_variables)

        variables_expected = ['foo', 'bar', 'spam', 'eggs']
        variables = self.optimization_problem.variable_names
        self.assertEqual(variables_expected, variables)

    def test_initial_values(self):
        x0_chebyshev_expected = [[0.79289322, 0.20710678, 0.5]]
        x0_chebyshev = self.optimization_problem.create_initial_values(
            1, method='chebyshev'
        )
        np.testing.assert_almost_equal(x0_chebyshev, x0_chebyshev_expected)

        variables_expected = [
            0.7928932188134523,
            0.2071067811865475,
            0.2071067811865475,
            0.4999999999999999
        ]
        variables = self.optimization_problem.get_dependent_values(
            x0_chebyshev[0, :]
        )
        np.testing.assert_almost_equal(variables, variables_expected)

        self.assertTrue(
            self.optimization_problem.check_linear_constraints(
                x0_chebyshev[0, :], get_dependent_values=True
            )
        )
        self.assertTrue(
            self.optimization_problem.check_linear_constraints(variables)
        )

        x0_seed_1_expected = [[0.7311044, 0.1727515, 0.1822629]]
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
            [0.7311043824888657, 0.1727515432673712, 0.18226293643057073],
            [0.9836918383919191, 0.8152389217047241, 0.8560016844195478],
            [0.7358144798470049, 0.2574714423019172, 0.49387609464567295],
            [0.34919171897183954, 0.05751800197656948, 0.3237260675631758],
            [0.9265061673265441, 0.4857572549618687, 0.8149444448089398],
            [0.9065669851023331, 0.1513817591204391, 0.7710992332649812],
            [0.8864554240066591, 0.4771068979697068, 0.5603893963194555],
            [0.6845940550232432, 0.2843172686185149, 0.6792904559788712],
            [0.923735889273789, 0.6890814170651027, 0.7366940211809302],
            [0.8359314486227345, 0.39493879515319996, 0.8128182754300088]
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
        optimization_problem = OptimizationProblem(name, use_diskcache=False)
        optimization_problem.add_variable('x')
        optimization_problem.add_objective(single_obj_single_var)
        self.single_obj_single_var = optimization_problem

        def single_obj_two_vars(x):
            return x[1]**2 - x[0]**2

        name = 'single_obj_two_vars'
        optimization_problem = OptimizationProblem(name, use_diskcache=False)
        optimization_problem.add_variable('x_1')
        optimization_problem.add_variable('x_2')
        optimization_problem.add_objective(single_obj_two_vars)
        self.single_obj_two_vars = optimization_problem

        def two_obj_single_var(x):
            return [x[0]**2, x[0]**2]

        name = 'two_obj_single_var'
        optimization_problem = OptimizationProblem(name, use_diskcache=False)
        optimization_problem.add_variable('x')
        optimization_problem.add_objective(two_obj_single_var, n_objectives=2)
        self.two_obj_single_var = optimization_problem

        def two_obj_two_var(x):
            return [x[0]**2, x[1]**2]

        name = 'two_obj_two_var'
        optimization_problem = OptimizationProblem(name, use_diskcache=False)
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
        optimization_problem = OptimizationProblem(
            'with_evaluator', use_diskcache=False
        )

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
