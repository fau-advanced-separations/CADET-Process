from typing import Union, Dict, Any

import pandas as pd
import numpy as np

from ax import (
    Runner, Data, Metric, Models,
    OptimizationConfig, MultiObjectiveOptimizationConfig,
    SearchSpace, MultiObjective, Experiment, Arm,
    ComparisonOp, OutcomeConstraint,
    RangeParameter, ParameterType, ParameterConstraint, Objective,

)
from ax.core.metric import MetricFetchResult, MetricFetchE
from ax.core.base_trial import BaseTrial
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.registry import Models
from ax.utils.common.result import Err, Ok
from ax.service.utils.report_utils import exp_to_df
from botorch.utils.sampling import manual_seed
from botorch.models.gp_regression import FixedNoiseGP
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

from CADETProcess.dataStructure import UnsignedInteger, Typed
from CADETProcess.optimization.optimizationProblem import OptimizationProblem
from CADETProcess.optimization import OptimizerBase

__all__ = [
    'GPEI',
    'BotorchModular',
    'NEHVI',
]


class CADETProcessMetric(Metric):
    def __init__(
            self,
            name: str,
            lower_is_better: Union[bool, None] = None,
            properties: Union[Dict[str, Any], None] = None) -> None:
        super().__init__(name, lower_is_better, properties)

    def fetch_trial_data(
            self,
            trial: BaseTrial,
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

            return Ok(Data(df=pd.DataFrame.from_records(records)))
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

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
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
        if self.optimization_problem.n_nonlinear_constraints > 0:
            nonlincon_fun = self.optimization_problem.evaluate_nonlinear_constraints_population
            nonlincon_labels = self.optimization_problem.nonlinear_constraint_labels

            G = nonlincon_fun(X, untransform=True, n_cores=n_cores)

        else:
            G = None
            nonlincon_labels = None

        # Update trial information with results.
        trial_metadata = self.get_metadata(
            trial, F, objective_labels, G, nonlincon_labels
        )

        return trial_metadata

    # TODO: rename: write_metadata
    @staticmethod
    def get_metadata(trial, F, objective_labels, G, nonlincon_labels):
        trial_metadata = {"name": str(trial.index)}
        trial_metadata.update({"arms": {}})

        # TODO: Maybe more elegant: check if G is None
        for i, arm in enumerate(trial.arms):
            f_dict = {
                metric: f_metric[i]
                for metric, f_metric in zip(objective_labels, F.T)
            }
            g_dict = {}
            if G is not None:
                g_dict = {
                    metric: g_metric[i]
                    for metric, g_metric in zip(nonlincon_labels, G.T)
                }
            trial_metadata["arms"].update({arm: {**f_dict, **g_dict}})

        return trial_metadata


class AxInterface(OptimizerBase):
    """Wrapper around Ax's bayesian optimization API."""

    supports_multi_objective = False
    supports_linear_constraints = True
    supports_linear_equality_constraints = False
    supports_nonlinear_constraints = True

    n_init_evals = UnsignedInteger(default=50)
    seed = UnsignedInteger(default=12345)

    _specific_options = ['n_init_evals', 'seed',]

    @staticmethod
    def _setup_parameters(optimizationProblem: OptimizationProblem):
        # TODO: can parameters have other choices except continuous
        # integer erstmal nicht nötig
        parameters = []
        for var in optimizationProblem.independent_variables:
            lb, ub = var.transformed_bounds
            param = RangeParameter(
                name=var.name,
                parameter_type=ParameterType.FLOAT,
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
            constr = ParameterConstraint(
                constraint_dict={
                    var: lhs for var, lhs in zip(lincon['opt_vars'], lincon['lhs'])
                },
                bound=lincon['b']
            )
            parameter_constraints.append(constr)

        return parameter_constraints

    @classmethod
    def _setup_searchspace(cls, optimizationProblem):
        return SearchSpace(
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

            obj = Objective(metric=ax_metric, minimize=True)
            objectives.append(obj)

        return objectives

    def _setup_outcome_constraints(self):
        """Parse nonliear constraint functions from optimization problem."""
        nonlincon_names = self.optimization_problem.nonlinear_constraint_labels
        bounds = self.optimization_problem.nonlinear_constraints_bounds

        outcome_constraints = []
        for name, bound in zip(nonlincon_names, bounds):
            ax_metric = CADETProcessMetric(name=name)

            nonlincon = OutcomeConstraint(
                metric=ax_metric,
                op=ComparisonOp.LEQ,
                bound=bound,
                relative=False,
            )
            outcome_constraints.append(nonlincon)

        return outcome_constraints

    def _create_manual_data(self, trial, F, G=None):

        objective_labels = self.optimization_problem.objective_labels
        nonlincon_labels = self.optimization_problem.nonlinear_constraint_labels
        return CADETProcessRunner.get_metadata(trial, F, objective_labels, G, nonlincon_labels)

    def _create_manual_trial(self, X):
        """Create trial from pre-evaluated data."""
        variables = self.optimization_problem.independent_variable_names

        trial = self.ax_experiment.new_batch_trial()

        for i, x in enumerate(X):
            trial_data = {
                "input": {var: x_i for var, x_i in zip(variables, x)},
            }

            arm_name = f"{trial.index}_{i}"
            trial.add_arm(Arm(parameters=trial_data["input"], name=arm_name))

        return trial

    def _post_processing(self, trial):
        """
        ax holds the data of the model in a dataframe
        an experiment consists of trials which consist of arms
        in a sequential experiment, each trial only has one arm.
        Arms are evaluated. These hold the parameters.
        """
        op = self.optimization_problem

        # get the trial level data as a dataframe
        trial_data = self.ax_experiment.fetch_trials_data([trial.index])
        data = trial_data.df

        # DONE: Update for multi-processing. If n_cores > 1: len(arms) > 1 (oder @Flo?)
        X = np.array([list(arm.parameters.values()) for arm in trial.arms])
        objective_labels = op.objective_labels

        n_ind = len(X)

        # Get objective values
        F_data = data[data['metric_name'].isin(objective_labels)]
        assert np.all(F_data["metric_name"].values == np.repeat(objective_labels, len(X)).astype(object))
        F = F_data["mean"].values.reshape((op.n_objectives, n_ind)).T

        # Get nonlinear constraint values
        if op.n_nonlinear_constraints > 0:
            nonlincon_labels = op.nonlinear_constraint_labels
            G_data = data[data['metric_name'].isin(nonlincon_labels)]
            assert np.all(G_data["metric_name"].values.tolist() == np.repeat(nonlincon_labels, len(X)))
            G = G_data["mean"].values.reshape((op.n_nonlinear_constraints, n_ind)).T

            nonlincon_cv_fun = op.evaluate_nonlinear_constraints_violation_population
            CV = nonlincon_cv_fun(X, untransform=True)
        else:
            G = None
            CV = None

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

        self.run_post_generation_processing(
            X=X,
            F=F,
            G=G,
            CV=CV,
            current_generation=self.ax_experiment.num_trials,
            X_opt=None,
        )

    def _setup_model(self):
        """constructs a pre-instantiated `Model` class that specifies the
        surrogate model (e.g. Gaussian Process) and acquisition function,
        which are used in the bayesian optimization algorithm.
        """
        raise NotImplementedError

    def _setup_optimization_config(self, objectives, outcome_constraints):
        """instantiates an optimization configuration for Ax for single objective
        or multi objective optimization
        """
        raise NotImplementedError

    def run(self, optimization_problem, x0):
        search_space = self._setup_searchspace(self.optimization_problem)
        objectives = self._setup_objectives()
        outcome_constraints = self._setup_outcome_constraints()
        optimization_config = self._setup_optimization_config(
            objectives=objectives,
            outcome_constraints=outcome_constraints
        )

        runner = CADETProcessRunner(
            optimization_problem=self.optimization_problem,
            n_cores=self.n_cores
        )

        self.ax_experiment = Experiment(
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
                trial = self._create_manual_trial(X)
                trial.mark_running(no_runner_required=True)

                trial_data = self._create_manual_data(trial, F, G)
                trial.run_metadata.update(trial_data)
                trial.mark_completed()

        else:
            # Create initial samples
            X_init = self.optimization_problem.create_initial_values(
                n_samples=self.n_init_evals,
                method="chebyshev",
                seed=self.seed + 5641,
            )

            trial = self._create_manual_trial(X_init)
            trial.run()
            trial.mark_running()
            trial.mark_completed()
            self._post_processing(trial)
            print(exp_to_df(self.ax_experiment))

        n_iter = self.results.n_gen
        n_evals = self.results.n_evals

        # TODO: termination criteria (check tutorial "early stopping strategies")
        with manual_seed(seed=self.seed):
            while not (n_evals >= self.n_max_evals or n_iter >= self.n_max_iter):
                # Reinitialize GP+EI model at each step with updated data.
                data = self.ax_experiment.fetch_data()
                modelbridge = self._train_model(data=data)

                print(f"Running optimization trial {n_evals+1}/{self.n_max_evals}...")

                # samples can be accessed here by sample_generator.arms:
                sample_generator = modelbridge.gen(n=1)

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


class SingleObjectiveAxInterface(AxInterface):
    def _setup_optimization_config(self, objectives, outcome_constraints):
        return OptimizationConfig(
            objective=objectives[0],
            outcome_constraints=outcome_constraints
        )

class MultiObjectiveAxInterface(AxInterface):
    supports_multi_objective = True

    def _setup_optimization_config(self, objectives, outcome_constraints):
        return MultiObjectiveOptimizationConfig(
            objective=MultiObjective(objectives),
            outcome_constraints=outcome_constraints
        )


class GPEI(SingleObjectiveAxInterface):
    """
    Bayesian optimization algorithm with a Gaussian Process (GP) surrogate model
    and a Expected Improvement (EI) acquisition function for single objective
    """

    def __repr__(self):
        return 'GPEI'

    def _train_model(self, data):
        return Models.GPEI(
            experiment=self.ax_experiment,
            data=data
        )

class BotorchModular(SingleObjectiveAxInterface):
    """
    implements a modular single objective bayesian optimization algorithm.
    It takes 2 optional arguments and uses the BOTORCH_MODULAR API of Ax
    to construct a Model, which connects both componenns with the respective
    transforms necessary

    acquisition_fn: AcquisitionFunction class
    surrogate_model: Model class
    """
    acquisition_fn = Typed(default=qNoisyExpectedImprovement)
    surrogate_model = Typed(default=FixedNoiseGP)

    _specific_options = [
        'acquisition_fn', 'surrogate_model'
    ]


    def __repr__(self):
        afn = self.acquisition_fn.__name__
        smn = self.surrogate_model.__name__

        return f'BotorchModular({smn}+{afn})'

    def _train_model(self, data):
        return Models.BOTORCH_MODULAR(
            experiment=self.ax_experiment,
            surrogate=Surrogate(self.surrogate_model),
            botorch_acqf_class=self.acquisition_fn,
            data=data
        )

class NEHVI(MultiObjectiveAxInterface):
    """
    Multi objective Bayesian optimization algorithm, which acquires new points
    with noisy expected hypervolume improvement (NEHVI) and approximates the
    model with a Fixed Noise Gaussian Process
    """

    def __repr__(self):
        smn = 'FixedNoiseGP'
        afn = 'NEHVI'

        return f'{smn}+{afn}'

    def _train_model(self, data):
        return Models.MOO(
            experiment=self.ax_experiment,
            data=data
        )
