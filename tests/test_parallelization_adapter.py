import datetime
import importlib
import multiprocessing
import shutil
import time
import unittest

from CADETProcess import settings
from CADETProcess.optimization import U_NSGA3, SequentialBackend
from CADETProcess.simulator import Cadet

from tests.test_cadet_adapter import detect_cadet
from tests.test_optimization_problem import setup_optimization_problem

parallel_backends_module = importlib.import_module("CADETProcess.optimization")

_parallel_backends = [
    "Joblib",
    "Pathos",
]

parallel_backends = []
for Backend in _parallel_backends:
    try:
        Backend = getattr(parallel_backends_module, Backend)
        parallel_backends.append(Backend)
    except AttributeError:
        print(f"Couldn't import backend {Backend}")

backends = [SequentialBackend] + parallel_backends

found_cadet, install_path = detect_cadet()

n_cores = 2
cpu_count = multiprocessing.cpu_count()


class TestParallelizationBackend(unittest.TestCase):
    """Test initializing parallelization backends and n_cores attribute."""

    def test_n_cores(self):
        with self.assertRaises(ValueError):
            sequential_backend = SequentialBackend(n_cores=n_cores)

        with self.assertRaises(ValueError):
            sequential_backend = SequentialBackend()
            sequential_backend.n_cores = n_cores

        for Backend in parallel_backends:
            backend = Backend(n_cores=n_cores)
            backend.n_cores = n_cores

    def test_max_cores(self):
        for Backend in parallel_backends:
            with self.assertRaises(ValueError):
                backend = Backend(n_cores=cpu_count + 1)

    def test_negative_n_cores(self):
        for Backend in parallel_backends:
            backend = Backend(n_cores=0)
            self.assertEqual(backend._n_cores, cpu_count)

            backend.n_cores = -1
            self.assertEqual(backend._n_cores, cpu_count)

            backend.n_cores = -2
            if cpu_count > 1:
                self.assertEqual(backend._n_cores, cpu_count - 1)
            else:
                self.assertEqual(backend._n_cores, 1)

            with self.assertRaises(ValueError):
                backend.n_cores = -cpu_count - 1


class TestParallelEvaluation(unittest.TestCase):
    """Test evaluating function in parallel.

    Note
    ----
    Since initializing CADET in parallel can be an issue, this is also tested.

    """

    def tearDown(self):
        shutil.rmtree("./tmp", ignore_errors=True)
        shutil.rmtree("./test_parallelization", ignore_errors=True)

    def test_parallelization_backend(self):
        def evaluation_function(sleep_time=0.0):
            print(sleep_time)
            time.sleep(sleep_time)
            return True

        for Backend in backends:
            backend = Backend()
            if not isinstance(backend, SequentialBackend):
                backend.n_cores = n_cores
            results = backend.evaluate(evaluation_function, [0.01] * 4)
            self.assertTrue(all(results))

    @unittest.skipIf(found_cadet is False, "Skip if CADET is not installed.")
    def test_parallel_cadet_initialization(self):
        def evaluation_function(x):
            simulator = Cadet()

            file_name = simulator.get_tempfile_name()
            cwd = simulator.temp_dir
            sim = simulator.create_lwe(cwd / file_name)
            if hasattr(sim, "run_simulation"):
                return_information = sim.run_simulation()
            else:
                return_information = sim.run_load()

            return True

        for Backend in backends:
            backend = Backend()
            if not isinstance(backend, SequentialBackend):
                backend.n_cores = n_cores

            results = backend.evaluate(evaluation_function, [0.01] * 4)

            self.assertTrue(all(results))


class TestOptimizerParallelizationBackend(unittest.TestCase):
    """Test parallel backends in optimizer.

    Note
    ----
    Consider moving this to Optimizer tests.

    """

    def tearDown(self):
        settings.working_directory = None

        shutil.rmtree("./test_parallelization", ignore_errors=True)
        shutil.rmtree("./diskcache_simple", ignore_errors=True)

    def test_parallel_optimization(self):
        def run_optimization(backend=None):
            def dummy_objective_function(x):
                y = sum((x - 0.5) ** 2)
                time.sleep(0.05)
                return y

            settings.working_directory = "./test_parallelization"

            optimization_problem = setup_optimization_problem(
                use_diskcache=True, obj_fun=dummy_objective_function
            )

            optimizer = U_NSGA3()
            optimizer.pop_size = 16
            optimizer.n_max_gen = 1

            if backend is not None:
                optimizer.parallelization_backend = backend
                backend_string = str(backend)
            else:
                backend_string = "Default backend"

            start_time = datetime.datetime.now()
            opt_results = optimizer.optimize(
                optimization_problem,
                save_results=True,
                use_checkpoint=False,
            )
            end_time = datetime.datetime.now()

            print(
                f"======> {backend_string} with "
                f"{optimizer.parallelization_backend.n_cores} cores took "
                f"{end_time - start_time} <=========="
            )

        run_optimization(backend=None)

        for Backend in backends:
            backend = Backend()
            if not isinstance(backend, SequentialBackend):
                backend.n_cores = n_cores
            run_optimization(backend=backend)


if __name__ == "__main__":
    unittest.main()
