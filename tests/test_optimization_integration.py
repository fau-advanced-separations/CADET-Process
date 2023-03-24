from pathlib import Path
import time
import shutil
import sys
import unittest

from CADETProcess import settings

sys.path.insert(0, '../')

# Set flags for which test cases to run
test_batch_elution_single_objective_single_core = True
test_batch_elution_single_objective_multi_core = True
test_batch_elution_multi_objective = True
test_fit_column_parameters = True


class TestBatchElutionOptimizationSingleObjective(unittest.TestCase):
    def setUp(self):
        settings.working_directory = './test_batch'
        from examples.batch_elution.optimization_single import (
            optimization_problem, optimizer
        )

        self.optimization_problem = optimization_problem
        self.optimizer = optimizer

    def tearDown(self):
        shutil.rmtree('./test_batch', ignore_errors=True)
        shutil.rmtree('./tmp', ignore_errors=True)
        shutil.rmtree('./diskcache', ignore_errors=True)

    def test_single_core(self):
        if not test_batch_elution_single_objective_single_core:
            self.skipTest("Skipping test_batch_elution_single_objective_single_core")

        self.optimizer.n_cores = 1
        self.optimizer.pop_size = 4
        self.optimizer.n_max_gen = 4

        print("start test_batch_elution_single_objective_single_core")

        start = time.time()
        results = self.optimizer.optimize(
            self.optimization_problem,
            use_checkpoint=False,
        )
        end = time.time()

        print(f"Finalized test_batch_elution_single_objective_single_core in {end - start} s")

    @unittest.skipIf(__name__ != "__main__", "Only run test if test is run as __main__")
    def test_multi_core(self):
        if not test_batch_elution_single_objective_multi_core:
            self.skipTest("Skipping test_batch_elution_single_objective_multi_core")

        self.optimizer.n_cores = 4
        self.optimizer.pop_size = 16
        self.optimizer.n_max_gen = 4

        print("start test_batch_elution_single_objective_multi_core")

        start = time.time()
        results = self.optimizer.optimize(
            self.optimization_problem,
            use_checkpoint=False,
        )
        end = time.time()

        print(f"Finalized test_batch_elution_single_objective_multi_core in {end - start} s")
        print(f"Equivalent CPU time: {(end - start)* self.optimizer.n_cores} s")


class TestBatchElutionOptimizationMultiObjective(unittest.TestCase):
    def setUp(self):
        settings.working_directory = './test_batch'
        from examples.batch_elution.optimization_multi import (
            optimization_problem, optimizer
        )

        self.optimization_problem = optimization_problem
        self.optimizer = optimizer

    def tearDown(self):
        shutil.rmtree('./test_batch', ignore_errors=True)
        shutil.rmtree('./tmp', ignore_errors=True)
        shutil.rmtree('./diskcache', ignore_errors=True)

    @unittest.skipIf(__name__ != "__main__", "Only run test if test is run as __main__")
    def test_optimization(self):
        if not test_batch_elution_multi_objective:
            self.skipTest("Skipping test_batch_elution_multi_objective")

        self.optimizer.n_cores = 4
        self.optimizer.pop_size = 16
        self.optimizer.n_max_gen = 4

        print("start test_batch_elution_multi_objective")

        start = time.time()
        results = self.optimizer.optimize(
            self.optimization_problem,
            use_checkpoint=False,
        )
        end = time.time()

        print(f"Finalized test_batch_elution_multi_objective in {end - start} s")
        print(f"Equivalent CPU time: {(end - start)* self.optimizer.n_cores} s")

        print(results.f)
        print(results.x)


class TestFitColumnParameters(unittest.TestCase):
    def setUp(self):
        settings.working_directory = './test_fit_column_parameters'

        data_dir = Path(__file__).parent.parent / 'examples/characterize_chromatographic_system/experimental_data'
        shutil.rmtree('./experimental_data/', ignore_errors=True)
        shutil.copytree(data_dir, './experimental_data')

        from examples.characterize_chromatographic_system.column_transport_parameters import (
            optimization_problem, optimizer
        )

        self.optimization_problem = optimization_problem
        self.optimizer = optimizer

    def tearDown(self):
        shutil.rmtree('./test_fit_column_parameters', ignore_errors=True)
        shutil.rmtree('./tmp', ignore_errors=True)
        shutil.rmtree('./diskcache', ignore_errors=True)
        shutil.rmtree('./experimental_data/', ignore_errors=True)

    def test_optimization(self):
        if not test_fit_column_parameters:
            self.skipTest("Skipping test_fit_column_parameters")

        self.optimizer.n_cores = 4
        self.optimizer.pop_size = 16
        self.optimizer.n_max_gen = 4

        print("start test_fit_column_parameters")

        start = time.time()
        results = self.optimizer.optimize(
            self.optimization_problem,
            use_checkpoint=False,
        )
        end = time.time()

        print(f"Finalized test_fit_column_parameters in {end - start} s")
        print(f"Equivalent CPU time: {(end - start)* self.optimizer.n_cores} s")



if __name__ == '__main__':
    # Run the tests
    unittest.main()
