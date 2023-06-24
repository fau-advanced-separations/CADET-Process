from typing import Any, Dict
from functools import partial

import pandas as pd
import numpy as np

import ax
from ax import core
from ax.core.metric import MetricFetchResult
from ax.core.runner import Runner
from ax.core.metric import Metric, MetricFetchE
from ax.modelbridge.registry import Models
from ax.service.utils.report_utils import exp_to_df
from ax.utils.common.result import Err, Ok
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.acquisition import Acquisition
from botorch.utils.sampling import manual_seed
from botorch.models.gp_regression import FixedNoiseGP
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.analytic import UpperConfidenceBound, NoisyExpectedImprovement
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.modelbridge import TorchModelBridge
from ax.modelbridge.registry import Cont_X_trans, Y_trans

from CADETProcess.dataStructure import UnsignedInteger, UnsignedFloat, Integer, Typed
from CADETProcess.optimization.optimizationProblem import OptimizationProblem
from CADETProcess.optimization import OptimizerBase

__all__ = ['AxInterface']


class CADETProcessMetric(Metric):
    def __init__(
            self,
            name: str,
            lower_is_better: bool | None = None,
            properties: Dict[str, Any] | None = None) -> None:
        super().__init__(name, lower_is_better, properties)

    def fetch_trial_data(
            self,
            trial: core.base_trial.BaseTrial,
            **kwargs: Any) -> MetricFetchResult:
        try:
            trial_results = trial.run_metadata
            records = []
            for arm_name, arm in trial.arms_by_name.items():

                results_dict = {
                    "trial_index": trial.index,
                    "arm_name": arm_name,
                    "metric_name": self.name,
                }

                # this looks up the value of the objective function
                # generated in the runner
                arm_results = trial_results["arms"][arm]
                results_dict.update({"mean": arm_results[self.name]})
                results_dict.update({"sem": 0.0})

                records.append(results_dict)

            return Ok(ax.Data(df=pd.DataFrame.from_records(records)))
        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )


class CADETProcessRunner(Runner):
    def __init__(
            self,
            optimization_problem: OptimizationProblem,
            n_cores=1) -> None:
        self.optimization_problem = optimization_problem
        self.n_cores = n_cores

    def staging_required(self) -> bool:
        # TODO: change that to True
        return False

    def run(self, trial: core.base_trial.BaseTrial) -> Dict[str, Any]:
        # Get X from arms.
        X = []
        for arm in trial.arms:
            x = np.array(list(arm.parameters.values()))
            X.append(x)

        X = np.row_stack(X)

        # adjust the number of cores to the number of batch trials
        # TODO: Adapt new parallelization backend once #40 is merged
        # See: https://github.com/fau-advanced-separations/CADET-Process/pull/40

        n_cores = min(self.n_cores, len(X))
        # Calculate objectives
        # TODO: add noise to the result
        objective_labels = self.optimization_problem.objective_labels
        obj_fun = self.optimization_problem.evaluate_objectives_population

        F = obj_fun(X, untransform=True, n_cores=n_cores)

        # Calculate nonlinear constraints
        # TODO: add noise to the result
        nonlincon_fun = self.optimization_problem.evaluate_nonlinear_constraints_population
        nonlincon_labels = self.optimization_problem.nonlinear_constraint_labels

        G = nonlincon_fun(X, untransform=True, n_cores=n_cores)

        # Update trial information with results.
        trial_metadata = self.get_metadata(
            trial, F, objective_labels, G, nonlincon_labels
        )

        return trial_metadata

    @staticmethod
    def get_metadata(trial, F, objective_labels, G, nonlincon_labels):
        trial_metadata = {"name": str(trial.index)}
        trial_metadata.update({"arms": {}})

        # TODO: Maybe more elegant: check if G is None
        G = np.array(G, ndmin=2)
        for i, arm in enumerate(trial.arms):
            f_dict = {
                metric: f_metric[i]
                for metric, f_metric in zip(objective_labels, F.T)
            }
            g_dict = {
                metric: g_metric[i]
                for metric, g_metric in zip(nonlincon_labels, G.T)
            }
            trial_metadata["arms"].update({arm: {**f_dict, **g_dict}})

        return trial_metadata


class AxInterface(OptimizerBase):
    """Wrapper around Ax's bayesian optimization API."""

    supports_multi_objective = True
    supports_linear_constraints = True
    supports_linear_equality_constraints = True
    supports_nonlinear_constraints = True

    n_init_evals = UnsignedInteger(default=50)
    seed = UnsignedInteger(default=12345)

    acquisition_fn = Typed(default=qNoisyExpectedImprovement)
    surrogate_model = Typed(default=FixedNoiseGP)

    _specific_options = [
        'n_init_evals', 'seed', 'acquisition_fn', 'surrogate_model'
    ]

    @staticmethod
    def _setup_parameters(optimizationProblem: OptimizationProblem):
        # TODO: can parameters have other choices except continuous
        # integer erstmal nicht nötig
        parameters = []
        for var in optimizationProblem.independent_variables:
            lb, ub = var.transformed_bounds
            param = ax.RangeParameter(
                name=var.name,
                parameter_type=ax.ParameterType.FLOAT,
                lower=lb,
                upper=ub,
                log_scale=False,
                # TODO: find out what fidelity means here
                is_fidelity=False,
            )
            parameters.append(param)

        return parameters

    @staticmethod
    def _setup_linear_constraints(optimizationProblem: OptimizationProblem):
        # TODO: Only independent variables
        # TODO: Should be transformed space (e.g. linear_constraints_transformed)
        parameter_constraints = []
        for lincon in optimizationProblem.linear_constraints:
            constr = ax.ParameterConstraint(
                constraint_dict={
                    var: lhs for var, lhs in zip(lincon['opt_vars'], lincon['lhs'])
                },
                bound=lincon['b']
            )
            parameter_constraints.append(constr)

        return parameter_constraints

    @classmethod
    def _setup_searchspace(cls, optimizationProblem):
        return ax.SearchSpace(
            parameters=cls._setup_parameters(optimizationProblem),
            parameter_constraints=cls._setup_linear_constraints(optimizationProblem)

        )

    def _setup_objectives(self):
        """Parse objective functions from optimization problem."""
        objective_names = self.optimization_problem.objective_labels

        objectives = []
        for i, obj_name in enumerate(objective_names):
            ax_metric = CADETProcessMetric(
                name=obj_name,
                lower_is_better=True,
            )

            obj = ax.Objective(metric=ax_metric, minimize=True)
            objectives.append(obj)

        return objectives

    def _setup_outcome_constraints(self):
        """Parse nonliear constraint functions from optimization problem."""
        nonlincon_names = self.optimization_problem.nonlinear_constraint_labels
        bounds = self.optimization_problem.nonlinear_constraints_bounds

        outcome_constraints = []
        for name, bound in zip(nonlincon_names, bounds):
            ax_metric = CADETProcessMetric(name=name)

            nonlincon = ax.OutcomeConstraint(
                metric=ax_metric,
                op=ax.ComparisonOp.LEQ,
                bound=bound,
                relative=False,
            )
            outcome_constraints.append(nonlincon)

        return outcome_constraints

    def _create_manual_trial(self, X, F, G=None):
        """Create trial from pre-evaluated data."""
        variables = self.optimization_problem.independent_variable_names

        objective_labels = self.optimization_problem.objective_labels
        nonlincon_labels = self.optimization_problem.nonlinear_constraint_labels
        metric_names = objective_labels + nonlincon_labels

        if G is None:
            M = F
        else:
            M = np.hstack((F, G))

        for i, (x, m) in enumerate(zip(X, M)):
            trial_data = {
                "input": {var: x_i for var, x_i in zip(variables, x)},
                "output": {
                    metric: {"mean": m_i, "sem": 0.0}
                    for metric, m_i in zip(metric_names, m)},
            }

            arm_name = f"{i}_0"
            trial = self.ax_experiment.new_trial()
            trial.add_arm(ax.Arm(parameters=trial_data["input"], name=arm_name))
            data = ax.Data(df=pd.DataFrame.from_records([
                {
                      "arm_name": arm_name,
                      "metric_name": metric_name,
                      "mean": output["mean"],
                      "sem": output["sem"],
                      "trial_index": i,
                 }
                 for metric_name, output in trial_data["output"].items()
              ])
            )
            self.ax_experiment.attach_data(data)

            trial.run_metadata.update(CADETProcessRunner.get_metadata(
                trial, F, objective_labels, G, nonlincon_labels
            ))
            trial.mark_running(no_runner_required=True)
            trial.mark_completed()


    def _post_processing(self, trial):
        """
        ax holds the data of the model in a dataframe
        an experiment consists of trials which consist of arms
        in a sequential experiment, each trial only has one arm.
        Arms are evaluated. These hold the parameters.
        """

        # get the trial level data as a dataframe
        trial_data = self.ax_experiment.fetch_trials_data([trial.index])
        data = trial_data.df

        # TODO: Update for multi-processing. If n_cores > 1: len(arms) > 1 (oder @Flo?)
        x = list(trial.arms[0].parameters.values())

        # Get objective values
        objective_labels = self.optimization_problem.objective_labels
        f_data = data[data['metric_name'].isin(objective_labels)]
        assert f_data["metric_name"].values.tolist() == objective_labels
        f = f_data["mean"].values

        # Get nonlinear constraint values
        nonlincon_labels = self.optimization_problem.nonlinear_constraint_labels
        g_data = data[data['metric_name'].isin(nonlincon_labels)]
        assert g_data["metric_name"].values.tolist() == nonlincon_labels
        g = g_data["mean"].values

        nonlincon_cv_fun = self.optimization_problem.evaluate_nonlinear_constraints_violation
        cv = nonlincon_cv_fun(x, untransform=True)

        # übergeben von besten werten (xopt) - die müsste von ax kommen
        # wenn das verfügbar ist,
        # TODO: kann ax das auch?

        ## Here, I'm trying to get the "best" values ever evaluated for each objective.
        # This is interesting for logging but not quite yet what I'm looking for.
        # I'd like to get the full "pareto" (i.e. non-dominated) front at that point.
        # Try: exp_to_df!
        # self._data = ax.Data.from_multiple_data([self._data, trial.fetch_data()])
        # new_value = trial.fetch_data().df["mean"].min()

        # print(
        #     f"Iteration: Best in iteration {new_value:.3f}, "
        #     f"Best so far: {self._data.df['mean'].min():.3f}"
        # )

        self.run_post_evaluation_processing(
            x=x,
            f=f,
            g=g,
            cv=cv,
            current_evaluation=self.ax_experiment.num_trials,
            x_opt=None,
        )

    def setup_model(self):
        pass

    def run(self, optimization_problem, x0):
        search_space = self._setup_searchspace(self.optimization_problem)
        objectives = self._setup_objectives()
        outcome_constraints = self._setup_outcome_constraints()

        # TODO: Move to setup_model function.
        # Subclass interface and parametrize using model registries.
        if len(objectives) > 1:
            is_moo = True
        else:
            is_moo = False

        if is_moo:
            optimization_config = ax.MultiObjectiveOptimizationConfig(
                objective=ax.MultiObjective(objectives),
                outcome_constraints=outcome_constraints
            )
            Model = Models.MOO

        else:
            optimization_config = ax.OptimizationConfig(
                objective=objectives[0],
                outcome_constraints=outcome_constraints
            )
            Model = partial(
                Models.BOTORCH_MODULAR,
                surrogate=Surrogate(self.surrogate_model),        # Optional, will use default if unspecified
                botorch_acqf_class=self.acquisition_fn,  # Optional, will use default if unspecified
            )

        # Alternative: Use a model bridge directly with a botorch model.
        # This allows for more control and does not introduce another "magic"
        # middle layer in between. But currently I can't get this to work.
        # The above is as suggested in the PR
        # Model = BoTorchModel(
        #   acquisition_class=UpperConfidenceBound,
        #   surrogate=Surrogate(FixedNoiseGP)
        # )

        # model = TorchModelBridge(
        #     experiment=self.ax_experiment,
        #     search_space=search_space,
        #     data=self.ax_experiment.fetch_data(),
        #     model=Model,
        #     transforms=Cont_X_trans + Y_trans
        # )

        runner = CADETProcessRunner(optimization_problem=self.optimization_problem)

        self.ax_experiment = ax.Experiment(
            search_space=search_space,
            name=self.optimization_problem.name,
            optimization_config=optimization_config,
            runner=runner,
        )

        # Internal storage for tracking data
        self._data = self.ax_experiment.fetch_data()

        # Restore previous results from checkpoint
        if len(self.results.populations) > 0:
            for pop in self.results.populations:
                X, F, G = pop.x, pop.f, pop.g
                self._create_manual_trial(X, F, G)
        else:
            # Create initial samples
            X_init = self.optimization_problem.create_initial_values(
                n_samples=self.n_init_evals,
                method="chebyshev",
                seed=self.seed + 5641,
            )

            F_init = self.optimization_problem.evaluate_objectives_population(
                X_init, n_cores=self.n_cores
            )
            G_init = self.optimization_problem.evaluate_nonlinear_constraints_population(
                X_init, n_cores=self.n_cores
            )
            CV_init = self.optimization_problem.evaluate_nonlinear_constraints_violation_population(
                X_init, n_cores=self.n_cores
            )

            self._create_manual_trial(X_init, F_init, G_init)
            self.run_post_generation_processing(
                X_init, F_init, G_init, CV=CV_init, current_generation=1
            )

        n_iter = self.results.n_gen
        n_evals = self.results.n_evals

        # TODO: termination criteria (check tutorial "early stopping strategies")
        with manual_seed(seed=self.seed):
            while not (n_evals >= self.n_max_evals or n_iter >= self.n_max_iter):
                # Reinitialize GP+EI model at each step with updated data.
                model = Model(
                    experiment=self.ax_experiment,
                    data=self.ax_experiment.fetch_data(),
                )

                if n_evals == self.n_init_evals:
                    if is_moo:
                        srgm = model.model.model._get_name()
                        acqf = model.model.acqf_constructor.__name__.split("_")[1]
                    else:
                        srgm = model.model.surrogate.model._get_name()
                        acqf = model.model.botorch_acqf_class.__name__
                    print(f"Starting bayesian optimization loop...")
                    print(f"Surrogate model: {srgm}")
                    print(f"Acquisition function: {acqf}")

                print(f"Running optimization trial {n_evals+1}/{self.n_max_evals}...")

                # samples can be accessed here by sample_generator.arms:
                sample_generator = model.gen(n=1)

                # so this can be a staging environment
                # TODO: here optimization-problem could be used to reject
                #       samples based on non-linear constraints / dependent vars

                trial = self.ax_experiment.new_trial(generator_run=sample_generator)
                trial.run()

                trial.mark_running()

                # TODO: speed up post processing
                trial.mark_completed()
                self._post_processing(trial)

                n_iter += 1
                n_evals += len(trial.arms)

        print(exp_to_df(self.ax_experiment))
        print("finished")


class ModularBoTorch(AxInterface):
    pass


if __name__ == "__main__":

    from tests.test_optimizer_behavior import TestAxBehavior
    test = TestAxBehavior()
    test.test_constrained_moo()
    test.test_constrained_soo()
    test.test_single_objective()

    from tests.test_optimizer_ax import TestAxInterface
    test = TestAxInterface()
    test.test_single_objective()
    test.test_single_objective_linear_constraints()
    test.test_multi_objective()

    #x: [0.23783216 0.50267604 0.43198473], f: [4.81493504]
