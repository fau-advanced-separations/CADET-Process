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

    def test_dependencies(self):
        pass


class Test_OptimizationProblem(unittest.TestCase):
    """
    Szenarien für Variablen:
        - Bound Constraints
        - Linear Constraints
        - Dependencies
        - Dependencies with Constraints

    Szenarien für EaluationObjects:
        - Verschiedene Variablen für verschiedene EvalObjects.
        - Verschiedene Objectives/Constraints für verschiedene EvalObjects.
        - Verschiedene Evaluatoren für verschiedene Objectives/Constraints.
        - Caching nur für ausgewählte Evaluatoren.

    """

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.evaluation_object = EvaluationObject()

    def test_variables(self):
        pass

    def test_dependencies(self):
        pass

    def test_bounds(self):
        pass

    def test_linear_constraints(self):
        pass

    def test_initial_values(self):
        pass

    def test_cache(self):
        pass

    def test_jacobian(self):
        pass


eval_obj = EvaluationObject()

opt = OptimizationProblem('test')
opt.add_evaluation_object(eval_obj)

opt.add_variable('dummy_parameter')

expensive_evaluator = ExpensiveEvaluator()
opt.add_evaluator(expensive_evaluator)

cheap_evaluator = CheapEvaluator()
opt.add_evaluator(cheap_evaluator, cache=True)

opt.add_objective(sum_x)

opt.add_objective(min_results_1, requires=[expensive_evaluator, cheap_evaluator])
opt.add_objective(min_results_2, requires=[expensive_evaluator, cheap_evaluator])


# x = [1]
# f = opt.evaluate_objectives(x)
# print(f"objectives: {f}")
# g = opt.evaluate_nonlinear_constraints(x)
# print(f"constraints: {g}")
# f = opt.evaluate_objectives(x)
# print(f"objectives: {f}")

### Test set variables
# print(f"{eval_obj[0] is process}")
# print(eval_obj[0].cycle_time)
# print(eval_obj[0].feed_duration.time)

# print('Start evaluating objectives')
# f = optimization_problem.evaluate_objectives(x)
# print(f)
# print('Start evaluating constraints')
# g = optimization_problem.evaluate_nonlinear_constraints(x)
# print(g)


if __name__ == '__main__':
    unittest.main()
