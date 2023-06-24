try:
    from ax import RangeParameter, ParameterType
    ax_imported = True
except ImportError:
    ax_imported = False

import numpy as np
import sys
import unittest

from CADETProcess.optimization import OptimizationProblem
from CADETProcess.comparison import Comparator
from CADETProcess.optimization.axAdapater import AxInterface
from matplotlib import pyplot as plt
from scipy.signal import peak_widths, find_peaks


sys.path.insert(0, '../')


def create_parameters():
    var_0 = RangeParameter(
        "var_0", parameter_type=ParameterType.FLOAT,
        lower=0.0, upper=1.0
    )
    var_1 = RangeParameter(
        "var_1", parameter_type=ParameterType.FLOAT,
        lower=0.0, upper=1.0
    )

    return {"var_0": var_0, "var_1": var_1}


def setup_simple_simulator(theta_true):
    # generate simulation results with single known global optimum
    x = np.linspace(0, 50, 1000)

    def simulator(theta):
        """a gaussian function"""
        return x, theta[0] * np.exp(- (x - theta[1]) ** 2 / (2 * theta[2] ** 2))

    _, y = simulator(theta=theta_true)

    return simulator, x, y


def setup_2metric_comparator(reference):
    comparator = Comparator()
    comparator.add_reference(reference)
    comparator.add_difference_metric(
        'PeakHeight', reference, 'outlet.outlet',
    )

    comparator.add_difference_metric(
        'PeakPosition', reference, 'outlet.outlet',
    )

    return comparator


def setup_optimization_problem(
        theta_true, theta_bounds, metrics, n_lincon=0, results_directory="work"):

    optimization_problem = OptimizationProblem('simple', use_diskcache=False)

    simulator, x, y = setup_simple_simulator(theta_true)
    optimization_problem.add_evaluator(simulator, name="test-simulator")

    for i, (lb, ub) in enumerate(theta_bounds):
        optimization_problem.add_variable(
            f'var_{i}',
            lb=lb, ub=ub,
            transform="auto"
        )

    if n_lincon > 0:
        lincons = [
            ([f'var_{i_var}', f'var_{i_var+1}'], [1, -1], 0)
            for i_var in range(n_lincon)
        ]
        for opt_vars, lhs, b in lincons:
            optimization_problem.add_linear_constraint(opt_vars, lhs, b)

    objective_functions = [metric((x, y)) for metric in metrics]
    for obj_fun in objective_functions:
        optimization_problem.add_objective(
            obj_fun,
            n_objectives=1,
            requires=[simulator]
        )

    def callback_plot(
            simulation_results, individual, evaluation_object, callbacks_dir='./'):
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, y, label="true")
        x_sim, y_sim = simulation_results
        ax.plot(x_sim, y_sim, label="sim")
        ax.legend()
        fig.savefig(f"{callbacks_dir}/comparison.png")
        fig.savefig(f"{callbacks_dir}/comparison_{individual.id}.png")
        plt.close()

    optimization_problem.add_callback(
        callback_plot,
        requires=[simulator],
        callbacks_dir=f"{results_directory}/callbacks"
    )

    return optimization_problem


def setup_ax():
    optimizer = AxInterface()
    optimization_problem = setup_optimization_problem()

    return optimizer, optimization_problem


class Metric:
    def __init__(self, reference) -> None:
        self.target = self._evaluate(reference)

    def __call__(self, results):
        metric = self._evaluate(results)

        return np.sum((metric - self.target) ** 2)

    def __repr__(self):
        return str(type(self).__name__)

    @staticmethod
    def _evaluate(results):
        raise NotImplementedError


class PeakHeight(Metric):
    """calculates the maximum of a peak, assuming that there is only one peak"""

    @staticmethod
    def _evaluate(results):
        x, y = results
        return np.max(y)


class PeakPosition(Metric):
    """calculates the maximum of a peak, assuming that there is only one peak"""

    @staticmethod
    def _evaluate(results):
        x, y = results
        return x[np.argmax(y)]


class PeakWidth(Metric):
    """calculates the maximum of a peak, assuming that there is only one peak"""

    @staticmethod
    def _evaluate(results):
        x, y = results
        wscale = x[-1] / len(x)

        peaks, _ = find_peaks(y)
        if len(peaks) == 1:
            width, height, ipl, ipr = peak_widths(y, peaks, rel_height=.5)
            scaled_width = width * wscale
        elif len(peaks) > 1:
            raise ValueError("there should only be one peak so far")
        elif len(peaks) == 0:
            scaled_width = x[-1]-x[0]
        else:
            raise NotImplementedError
        # plt.plot(x,y)
        # plt.hlines(height, ipl*wscale, ipr*wscale)

        return scaled_width


class SSE(Metric):

    @staticmethod
    def _evaluate(results):
        x, y = results
        return y


# Set flags for which test cases to run
@unittest.skipUnless(ax_imported, "ax package is not installed")
class TestAxInterface(unittest.TestCase):

    def test_bounds(self):
        optimization_problem = OptimizationProblem('simple')

        optimization_problem.add_variable(
            'var_negpos',
            lb=-100, ub=100,
            transform="auto"
        )

        optimization_problem.add_objective(lambda x: x**2, "test")

        optimization_problem.evaluate_objectives([0.5], untransform=True)

    def test_single_objective(self):

        # plt.switch_backend(newbackend="qtagg")
        results_directory = "work/test_single_objective_ax"
        metrics = [SSE]
        theta_true = [0.5, 25, 2]
        theta_bounds = [(0, 1), (0, 50), (0.1, 10)]
        op = setup_optimization_problem(
            theta_true=theta_true,
            theta_bounds=theta_bounds,
            metrics=metrics,
            results_directory=results_directory
        )

        optimizer = AxInterface()
        optimizer.n_init_evals = 50
        optimizer.n_max_evals = 100
        optimizer.progress_frequency = 1
        optimizer.optimize(
            optimization_problem=op,
            save_results=True,
            results_directory=results_directory,
        )

        results = optimizer.results

        # test if theta_true was recovered by ax
        theta_true_transformed = op.transform(np.array([theta_true]))
        self.assertTrue(np.all(
            np.abs(results.x - theta_true_transformed) < 1e-2
        ))

    def test_single_objective_linear_constraints(self):

        results_directory = "work/test_single_objective_linear_constraints_ax"
        metrics = [SSE]
        theta_true = [0.5, 25, 2]
        theta_bounds = [(0, 1), (0, 50), (0.1, 10)]
        op = setup_optimization_problem(
            theta_true=theta_true,
            theta_bounds=theta_bounds,
            metrics=metrics,
            n_lincon=1,
            results_directory=results_directory,
        )

        optimizer = AxInterface()
        optimizer.n_init_evals = 50
        optimizer.n_max_evals = 100
        optimizer.progress_frequency = 1
        optimizer.optimize(
            optimization_problem=op,
            save_results=True,
            results_directory=results_directory,
        )

        results = optimizer.results

        # test if theta_true was recovered by ax
        theta_true_transformed = op.transform(np.array([theta_true]))
        self.assertTrue(np.all(
            np.abs(results.x - theta_true_transformed) < 1e-2
        ))

    def test_multi_objective(self):

        results_directory = "work/test_multi_objective_ax"
        metrics = [PeakHeight, PeakPosition, PeakWidth]
        theta_true = [0.5, 25, 2]
        theta_bounds = [(0, 1), (0, 50), (0.1, 10)]

        op = setup_optimization_problem(
            theta_true=theta_true,
            theta_bounds=theta_bounds,
            metrics=metrics,
            results_directory=results_directory
        )

        optimizer = AxInterface()
        optimizer.n_init_evals = 50
        optimizer.n_max_evals = 70
        optimizer.progress_frequency = 1
        optimizer.optimize(
            optimization_problem=op,
            save_results=True,
            results_directory=results_directory,
        )

        results = optimizer.results

        # test if theta_true was recovered by ax
        theta_true_transformed = op.transform(np.array([theta_true]))
        self.assertTrue(np.all(
            np.abs(results.x - theta_true_transformed) < 1e-2
        ))

        # TODO: write test that makes sure that the output on pareto front is
        #       deterministic
        # pareto_x_untransformed = np.array([
        #     [1.00000000e-02, 1.63233402e+01, 1.08042268e+01],
        #     [1.00000000e-02, 3.22857285e+01, 1.00000000e+02],
        #     [1.74158672e+01, 1.70892595e+01, 1.00000000e-02],
        #     [1.00000000e-02, 2.00016740e+01, 1.00000000e-02],
        #     [1.00000000e-02, 2.10689475e+01, 1.00000000e-02],
        #     [1.00000000e-02, 2.09363678e+01, 2.79707491e+01],
        #     [1.00000000e-02, 2.25172334e+01, 1.00000000e-02],
        #     [1.00000000e-02, 3.25891462e+01, 3.63959667e+01]
        # ])


if __name__ == '__main__':
    # Run the tests
    unittest.main()
