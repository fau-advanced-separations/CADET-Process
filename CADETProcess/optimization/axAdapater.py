from collections import defaultdict
from typing import Union, Dict, Any, Iterable, Set, Callable
import warnings
from logging import Logger

import pandas as pd
import numpy as np

from ax import (
    Runner, Data, Metric, Models,
    OptimizationConfig, MultiObjectiveOptimizationConfig,
    SearchSpace, MultiObjective, Experiment, Arm,
    ComparisonOp, OutcomeConstraint,
    RangeParameter, ParameterType, ParameterConstraint, Objective,
)

from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.core.metric import MetricFetchResult, MetricFetchE
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.models.torch.botorch_defaults import get_qLogNEI
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.result import Err, Ok
from ax.service.utils.report_utils import exp_to_df
from ax.service.utils.scheduler_options import TrialType
from ax.modelbridge.dispatch_utils import (
    calculate_num_initialization_trials,
    choose_generation_strategy
)
from ax.utils.common.logger import get_logger

from botorch.utils.sampling import manual_seed
from botorch.models.gp_regression import FixedNoiseGP
from botorch.acquisition.analytic import (
    LogExpectedImprovement
)

from CADETProcess.dataStructure import UnsignedInteger, Typed, Float, Bool
from CADETProcess.optimization.optimizationProblem import OptimizationProblem
from CADETProcess.optimization import OptimizerBase
from CADETProcess.optimization.parallelizationBackend import (
    SequentialBackend,
    ParallelizationBackendBase
)

logger: Logger = get_logger(__name__)


__all__ = [
    "GPEI",
    "BotorchModular",
    "NEHVI",
    "qNParEGO",
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
    """The way Ax intends the use of the components, a runner should dispatch
    or launch the evaluation of a trial. This can be scheduled on a remote or
    launched
    """
    def __init__(
            self,
            optimization_problem: OptimizationProblem,
            parallelization_backend: ParallelizationBackendBase,
            post_processing: Callable,
    ) -> None:
        self.optimization_problem = optimization_problem
        self.parallelization_backend = parallelization_backend
        self.post_processing = post_processing

    @property
    def staging_required(self) -> bool:
        return False

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        """
        This just returns that all trials have been completed, because the runner
        directly runs the experiment. If the runner itself launches a job
        that may not be finished after the run method call has been completed,
        this needs to query whether the evaluations have been terminated
        https://ax.dev/api/_modules/ax/runners/synthetic.html#SyntheticRunner
        """
        return {TrialStatus.COMPLETED: {t.index for t in trials}}

    def run(self, trial: Union[BaseTrial,BatchTrial]) -> Dict[str, Any]:
        # Get X from arms.
        logger.info(
            f"Running {type(trial).__name__} with {len(trial.arms)} arms "+
            f"With the {self.parallelization_backend}"
        )
        X = []
        for arm in trial.arms:
            x = np.array(list(arm.parameters.values()))
            X.append(x)

        X = np.row_stack(X)

        # adjust the number of cores to the number of batch trials
        # See: https://github.com/fau-advanced-separations/CADET-Process/issues/53

        # Calculate objectives
        # Explore if adding a small amount of noise to the result helps BO
        objective_labels = self.optimization_problem.objective_labels
        obj_fun = self.optimization_problem.evaluate_objectives

        # If I understand runners, correctly, this could be a dispatched
        # call to a remote instance.
        F = obj_fun(
            X,
            untransform=True,
            ensure_minimization=True,
            parallelization_backend=self.parallelization_backend
        )


        # Calculate nonlinear constraints
        # Explore if adding a small amount of noise to the result helps BO
        if self.optimization_problem.n_nonlinear_constraints > 0:
            nonlincon_cv_fun = self.optimization_problem.evaluate_nonlinear_constraints_violation
            nonlincon_labels = self.optimization_problem.nonlinear_constraint_labels


            G = None

            CV = nonlincon_cv_fun(
                X,
                untransform=True,
                parallelization_backend=self.parallelization_backend
            )

            raise NotImplementedError(
                "Are CV the violations or the constraint functions?"+
                "This must be modified in the post_processing call" +
                "And potentially in the get_metadata function"
            )

        else:
            G = None
            CV = None
            nonlincon_labels = None

        for x, f in zip(X, F):
            logger.info(f"CADET-Process results: x={x}, f={f}")
        # TODO: Are there conditions where we should mark a trial as failed
        #       This will not be considered in the surrogate model fitting

        logger.info(f"CADET-Process post-processing results")
        # self._post_processing(trial, trial_metadata)
        self.post_processing(
            X_transformed=X,
            F_minimized=F,
            G=G,
            CV=CV,
            current_generation=trial.index,
            X_opt_transformed=None,
        )
        # Update trial information with results.
        # this would normally be retrieved differently I think
        trial_metadata = self.get_metadata(
            trial, F, objective_labels, CV, nonlincon_labels
        )

        return trial_metadata

    @staticmethod
    def get_metadata(trial, F, objective_labels, CV, nonlincon_labels):
        trial_metadata = {"name": str(trial.index)}
        trial_metadata.update({"arms": {}})

        for i, arm in enumerate(trial.arms):
            f_dict = {
                f"{metric}_axidx_{i_obj}": f_metric[i]
                for i_obj, (metric, f_metric) in enumerate(zip(objective_labels, F.T))
            }
            cv_dict = {}
            if CV is not None:
                cv_dict = {
                    f"{metric}_axidx_{i_constr}": cv_metric[i]
                    for i_constr, (metric, cv_metric) in enumerate(zip(nonlincon_labels, CV.T))
                }
            trial_metadata["arms"].update({arm: {**f_dict, **cv_dict}})

        return trial_metadata


class AxInterface(OptimizerBase):
    """Wrapper around Ax's bayesian optimization API."""

    supports_bounds = True
    supports_multi_objective = False
    supports_linear_constraints = True
    supports_linear_equality_constraints = False
    supports_nonlinear_constraints = True

    early_stopping_improvement_window = UnsignedInteger(default=1000)
    early_stopping_improvement_bar = Float(default=1e-10)
    n_init_evals = UnsignedInteger(default=10)
    n_max_evals = UnsignedInteger(default=100)
    batch_size = UnsignedInteger(default=1)
    seed = UnsignedInteger(default=12345)

    _specific_options = [
        'n_init_evals',
        'n_max_evals',
        'batch_trials',
        'seed',
        'early_stopping_improvement_window',
        'early_stopping_improvement_bar',
    ]

    @staticmethod
    def _setup_parameters(optimizationProblem: OptimizationProblem):
        parameters = []
        for var in optimizationProblem.independent_variables:
            lb, ub = var.transformed_bounds
            param = RangeParameter(
                name=var.name,
                parameter_type=ParameterType.FLOAT,
                lower=lb,
                upper=ub,
                log_scale=False,
                is_fidelity=False,
            )
            parameters.append(param)

        return parameters

    @staticmethod
    def _setup_linear_constraints(optimizationProblem: OptimizationProblem):
        A_transformed = optimizationProblem.A_independent_transformed
        b_transformed = optimizationProblem.b_transformed
        indep_vars = optimizationProblem.independent_variables
        parameter_constraints = []
        for a_t, b_t in zip(A_transformed, b_transformed):
            constr = ParameterConstraint(
                constraint_dict={
                    var.name: a for var, a in zip(indep_vars, a_t)
                },
                bound=b_t
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
                name=f"{obj_name}_axidx_{i}",
                lower_is_better=True,
            )

            obj = Objective(metric=ax_metric, minimize=True)
            objectives.append(obj)

        return objectives

    def _setup_outcome_constraints(self):
        """Parse nonliear constraint functions from optimization problem."""
        nonlincon_names = self.optimization_problem.nonlinear_constraint_labels

        outcome_constraints = []
        for i_constr, name in enumerate(nonlincon_names):
            ax_metric = CADETProcessMetric(name=f"{name}_axidx_{i_constr}")

            nonlincon = OutcomeConstraint(
                metric=ax_metric,
                op=ComparisonOp.LEQ,
                bound=0.0,
                relative=False,
            )
            outcome_constraints.append(nonlincon)

        return outcome_constraints

    def _create_manual_data(self, trial, F, G=None):

        objective_labels = self.optimization_problem.objective_labels
        nonlincon_labels = self.optimization_problem.nonlinear_constraint_labels
        return CADETProcessRunner.get_metadata(trial, F, objective_labels, G, nonlincon_labels)

    def _restore_trials_from_checkoint(self):
        for pop in self.results.populations:
            X, F, G = pop.x, pop.f, pop.g
            raise NotImplementedError(
                "This seems not right. Currently all trials are evaluated again "+
                "to add the data."
            )
            trial = self._create_manual_trial(X)
            trial.mark_running(no_runner_required=True)

            trial_data = self._create_manual_data(trial, F, G)
            trial.run_metadata.update(trial_data)
            trial.mark_completed()

    def _create_manual_trial(self, x):
        """Create trial from pre-evaluated data."""
        variables = self.optimization_problem.independent_variable_names

        # for i, x in enumerate(X):
        trial = self.ax_experiment.new_trial()
        trial_data = {
            "input": {var: x_i for var, x_i in zip(variables, x)},
        }

        arm_name = f"{trial.index}_{0}"
        trial.add_arm(Arm(parameters=trial_data["input"], name=arm_name))
        trial.run()
        trial.mark_completed()

        return trial

    def _create_initial_trials(self, x0):
        if x0 is not None:

            X0_init = np.array(x0, ndmin=2)

            if len(X0_init) < self.n_init_evals:
                warnings.warn(
                    "Initial population smaller than popsize. "+
                    "Creating missing entries."
                )
                n_remaining = self.n_init_evals - len(X0_init)
                X0_remaining = self.optimization_problem.create_initial_values(
                    n_remaining, seed=self.seed, include_dependent_variables=False
                )
                X0_init = np.vstack((X0_init, X0_remaining))
            elif len(X0_init) > self.n_init_evals:
                warnings.warn("Initial population larger than popsize. Omitting overhead.")
                X0_init = X0_init[0:self.n_init_evals]

        else:
            # Create initial samples if they are not provided
            X0_init = self.optimization_problem.create_initial_values(
                n_samples=self.n_init_evals,
                include_dependent_variables=False,
                seed=self.seed + 5641,
            )

        X0_init_transformed = np.array(self.optimization_problem.transform(X0_init))

        for x0_init in X0_init_transformed:
            self._create_manual_trial(x0_init)

        print(exp_to_df(self.ax_experiment))

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
            parallelization_backend=SequentialBackend(),
            post_processing=self.run_post_processing
        )

        self.global_stopping_strategy = ImprovementGlobalStoppingStrategy(
            min_trials=self.n_init_evals + self.early_stopping_improvement_window,
            window_size=self.early_stopping_improvement_window,
            improvement_bar=self.early_stopping_improvement_bar,
            inactive_when_pending_trials=True
        )

        self.ax_experiment = Experiment(
            search_space=search_space,
            name=self.optimization_problem.name,
            optimization_config=optimization_config,
            runner=runner,
        )

        suggested_init_trials = calculate_num_initialization_trials(
            num_tunable_parameters=len(search_space.tunable_parameters),
            num_trials=self.n_max_evals,
            use_batch_trials=True if self.batch_size > 1 else False
        )
        if suggested_init_trials < self.n_init_evals:
            warnings.warn(
                f"The number of initial evaluations (trials) "+
                f"n_init_evals={self.n_init_evals} to start the Bayesian "+
                "optimization algorithm is lower than the suggested number "+
                f"of initialization trials ({suggested_init_trials})."
            )

        # this could be interesting. This can also be manually defined with the
        # GenerationStrategy, which is a list of steps to be taken [Sobol, BO]
        # Here for instance the automated
        gs = choose_generation_strategy(
            search_space=search_space,
            optimization_config=optimization_config,
            num_trials=self.n_max_evals,
            num_initialization_trials=self.n_init_evals,
            use_batch_trials=True if self.batch_size > 1 else False,
            # reduce the number of trials by the number of init evals that is
            # computed
            num_completed_initialization_trials=self.n_init_evals,
            max_parallelism_cap=1,
        )

        from ax.service.scheduler import Scheduler, SchedulerOptions
        # https://pytorch.org/tutorials/intermediate/ax_multiobjective_nas_tutorial.html#configuring-the-scheduler
        scheduler = Scheduler(
            experiment=self.ax_experiment,
            generation_strategy=gs,
            options=SchedulerOptions(
                trial_type=TrialType.BATCH_TRIAL if self.batch_size > 1 else TrialType.TRIAL,
                batch_size=self.batch_size,
                total_trials=self.n_max_evals,
                global_stopping_strategy=self.global_stopping_strategy,
                # TODO: What is max_pending_trials responsible for (this was taken from the tutorial)
                max_pending_trials=4
            ),
        )

        # Restore previous results from checkpoint
        if len(self.results.populations) > 0:
            self._restore_trials_from_checkoint()

        else:
            self._create_initial_trials(x0=x0)


        # Internal storage for tracking data
        self._data = self.ax_experiment.fetch_data()


        result = scheduler.run_all_trials()

        print(exp_to_df(self.ax_experiment))

        self.results.success = True
        self.results.exit_flag = 0
        # self.results.exit_message = global_stopping_message


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

    def train_model(self):
        return Models.GPEI(
            experiment=self.ax_experiment,
            data=self.ax_experiment.fetch_data()
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
    acquisition_fn = Typed(ty=type, default=LogExpectedImprovement)
    surrogate_model = Typed(ty=type, default=FixedNoiseGP)

    _specific_options = [
        'acquisition_fn', 'surrogate_model'
    ]

    def __repr__(self):
        afn = self.acquisition_fn.__name__
        smn = self.surrogate_model.__name__

        return f'BotorchModular({smn}+{afn})'

    def train_model(self):
        raise NotImplementedError("This model is currently broken. Please use Only GPEI or NEHVI")
        return Models.BOTORCH_MODULAR(
            experiment=self.ax_experiment,
            surrogate=Surrogate(self.surrogate_model),
            botorch_acqf_class=self.acquisition_fn,
            data=self.ax_experiment.fetch_data()
        )


class NEHVI(MultiObjectiveAxInterface):
    """
    Multi objective Bayesian optimization algorithm, which acquires new points
    with noisy expected hypervolume improvement (NEHVI) and approximates the
    model with a Fixed Noise Gaussian Process
    """
    supports_single_objective = False

    def __repr__(self):
        smn = 'FixedNoiseGP'
        afn = 'NEHVI'

        return f'{smn}+{afn}'

    def train_model(self):
        return Models.MOO(
            experiment=self.ax_experiment,
            data=self.ax_experiment.fetch_data()
        )


class qNParEGO(MultiObjectiveAxInterface):
    """
    Multi objective Bayesian optimization algorithm with the qNParEGO acquisition function.
    ParEGO transforms the MOO problem into a single objective problem by applying a randomly weighted augmented
    Chebyshev scalarization to the objectives, and maximizing the expected improvement of that scalarized
    quantity (Knowles, 2006). Recently, Daulton et al. (2020) used a multi-output Gaussian process and compositional
    Monte Carlo objective to extend ParEGO to the batch setting (qParEGO), which proved to be a strong baseline for
    MOBO. Additionally, the authors proposed a noisy variant (qNParEGO), but the empirical evaluation of qNParEGO
    was limited. [Daulton et al. 2021 "Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected
    Hypervolume Improvement"]
    """
    supports_single_objective = False

    def __repr__(self):
        smn = 'FixedNoiseGP'
        afn = 'qNParEGO'

        return f'{smn}+{afn}'

    def train_model(self):
        return Models.MOO(
            experiment=self.ax_experiment,
            data=self.ax_experiment.fetch_data(),
            acqf_constructor=get_qLogNEI,
            default_model_gen_options={
                "acquisition_function_kwargs": {
                    "chebyshev_scalarization": True,
                }
            },
        )
