import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax import (
    Arm,
    ComparisonOp,
    Data,
    Experiment,
    Metric,
    Models,
    MultiObjective,
    MultiObjectiveOptimizationConfig,
    Objective,
    OptimizationConfig,
    OutcomeConstraint,
    ParameterConstraint,
    ParameterType,
    RangeParameter,
    Runner,
    SearchSpace,
)
from ax.core.base_trial import BaseTrial
from ax.core.metric import MetricFetchE, MetricFetchResult
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.models.torch.botorch_defaults import get_qLogNEI
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.utils.report_utils import exp_to_df
from ax.utils.common.result import Err, Ok
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.sampling import manual_seed

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Float, Typed, UnsignedInteger
from CADETProcess.optimization import OptimizerBase
from CADETProcess.optimization.optimizationProblem import OptimizationProblem
from CADETProcess.optimization.parallelizationBackend import (
    ParallelizationBackendBase,
    SequentialBackend,
)

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
        properties: Union[Dict[str, Any], None] = None,
    ) -> None:
        super().__init__(name, lower_is_better, properties)

    def fetch_trial_data(
        self,
        trial: BaseTrial,
        **kwargs: Any,
    ) -> MetricFetchResult:
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
        parallelization_backend: ParallelizationBackendBase,
    ) -> None:
        self.optimization_problem = optimization_problem
        self.parallelization_backend = parallelization_backend

    @property
    def staging_required(self) -> bool:
        return False

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        # Get X from arms.
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

        F = obj_fun(
            X,
            untransform=True,
            get_dependent_values=True,
            ensure_minimization=True,
            parallelization_backend=self.parallelization_backend,
        )

        # Calculate nonlinear constraints
        # Explore if adding a small amount of noise to the result helps BO
        if self.optimization_problem.n_nonlinear_constraints > 0:
            nonlincon_cv_fun = (
                self.optimization_problem.evaluate_nonlinear_constraints_violation
            )
            nonlincon_labels = self.optimization_problem.nonlinear_constraint_labels

            CV = nonlincon_cv_fun(
                X,
                untransform=True,
                get_dependent_values=True,
                parallelization_backend=self.parallelization_backend,
            )

        else:
            CV = None
            nonlincon_labels = None

        # Update trial information with results.
        trial_metadata = self.get_metadata(
            trial, F, objective_labels, CV, nonlincon_labels
        )

        return trial_metadata

    @staticmethod
    def get_metadata(
        trial: BaseTrial,
        F: np.ndarray,
        objective_labels: list[str],
        CV: np.ndarray,
        nonlincon_labels: list[str],
    ) -> dict:
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
                    for i_constr, (metric, cv_metric) in enumerate(
                        zip(nonlincon_labels, CV.T)
                    )
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
    seed = UnsignedInteger(default=12345)

    _specific_options = [
        "n_init_evals",
        "n_max_evals",
        "seed",
        "early_stopping_improvement_window",
        "early_stopping_improvement_bar",
    ]

    @staticmethod
    def _setup_parameters(optimizationProblem: OptimizationProblem) -> list:
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
    def _setup_linear_constraints(optimizationProblem: OptimizationProblem) -> list:
        A_transformed = optimizationProblem.A_independent_transformed
        b_transformed = optimizationProblem.b_transformed
        indep_vars = optimizationProblem.independent_variables
        parameter_constraints = []
        for a_t, b_t in zip(A_transformed, b_transformed):
            constr = ParameterConstraint(
                constraint_dict={var.name: a for var, a in zip(indep_vars, a_t)},
                bound=b_t,
            )
            parameter_constraints.append(constr)

        return parameter_constraints

    @classmethod
    def _setup_searchspace(
        cls, optimizationProblem: OptimizationProblem
    ) -> SearchSpace:
        return SearchSpace(
            parameters=cls._setup_parameters(optimizationProblem),
            parameter_constraints=cls._setup_linear_constraints(optimizationProblem),
        )

    def _setup_objectives(self) -> list:
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

    def _setup_outcome_constraints(self) -> list:
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

    def _create_manual_data(
        self, trial: BaseTrial, F: npt.ArrayLike, G: Optional[npt.ArrayLike] = None
    ) -> dict:
        objective_labels = self.optimization_problem.objective_labels
        nonlincon_labels = self.optimization_problem.nonlinear_constraint_labels
        return CADETProcessRunner.get_metadata(
            trial, F, objective_labels, G, nonlincon_labels
        )

    def _create_manual_trial(self, X: npt.ArrayLike) -> None:
        """Create trial from pre-evaluated data."""
        variables = self.optimization_problem.independent_variable_names

        for i, x in enumerate(X):
            trial = self.ax_experiment.new_trial()
            trial_data = {
                "input": {var: x_i for var, x_i in zip(variables, x)},
            }

            arm_name = f"{trial.index}_{0}"
            trial.add_arm(Arm(parameters=trial_data["input"], name=arm_name))
            trial.run()
            trial.mark_completed()
            self._post_processing(trial)

            # When returning to batch trials, the Arms can be initialized here
            # and then collectively returned. See commit history

    def _post_processing(self, trial: BaseTrial) -> None:
        """
        Run post processing.

        Ax holds the data of the model in a dataframe an experiment consists of trials
        which consist of arms in a sequential experiment, each trial only has one arm.

        Arms are evaluated. These hold the parameters.
        """
        op = self.optimization_problem

        # get the trial level data as a dataframe
        trial_data = self.ax_experiment.fetch_trials_data([trial.index])
        data = trial_data.df

        # DONE: Update for multi-processing. If n_cores > 1: len(arms) > 1 (oder @Flo?)
        X = np.array([list(arm.parameters.values()) for arm in trial.arms])
        objective_labels = [
            f"{obj_name}_axidx_{i}" for i, obj_name in enumerate(op.objective_labels)
        ]

        n_ind = len(X)

        # Get objective values
        F_data = data[data["metric_name"].isin(objective_labels)]
        assert np.all(
            F_data["metric_name"].values
            == np.repeat(objective_labels, len(X)).astype(object)
        )
        F = F_data["mean"].values.reshape((op.n_objectives, n_ind)).T

        # Get nonlinear constraint values
        if op.n_nonlinear_constraints > 0:
            nonlincon_labels = [
                f"{name}_axidx_{i}"
                for i, name in enumerate(op.nonlinear_constraint_labels)
            ]
            G_data = data[data["metric_name"].isin(nonlincon_labels)]
            assert np.all(
                G_data["metric_name"].values.tolist()
                == np.repeat(nonlincon_labels, len(X))
            )
            G = G_data["mean"].values.reshape((op.n_nonlinear_constraints, n_ind)).T

            nonlincon_cv_fun = op.evaluate_nonlinear_constraints_violation
            CV = nonlincon_cv_fun(X, untransform=True, get_dependent_values=True)
        else:
            G = None
            CV = None

        # Ideally, the current optimum w.r.t. single and multi objective can be
        # obtained at this point and passed to run_post_processing.
        # with X_opt_transformed. Implementation is pending.
        # See: https://github.com/fau-advanced-separations/CADET-Process/issues/53

        self.run_post_processing(
            X_transformed=X,
            F_minimized=F,
            G=G,
            CV_nonlincon=CV,
            current_generation=self.ax_experiment.num_trials,
            X_opt_transformed=None,
        )

    def _setup_model(self) -> None:
        """
        Initialize a pre-instantiated `Model` class.

        The class specifies the surrogate model (e.g. Gaussian Process) and acquisition function,
        which are used in the bayesian optimization algorithm.
        """
        raise NotImplementedError

        """Initialize an optimization configuration for Ax."""

    def _setup_optimization_config(
        self,
        objectives: list[Objective],
        outcome_constraints: OutcomeConstraint,
    ) -> None:
        raise NotImplementedError

    def _run(
        self, optimization_problem: OptimizationProblem, x0: npt.ArrayLike
    ) -> None:
        search_space = self._setup_searchspace(self.optimization_problem)
        objectives = self._setup_objectives()
        outcome_constraints = self._setup_outcome_constraints()
        optimization_config = self._setup_optimization_config(
            objectives=objectives, outcome_constraints=outcome_constraints
        )

        runner = CADETProcessRunner(
            optimization_problem=self.optimization_problem,
            parallelization_backend=SequentialBackend(),
        )

        self.global_stopping_strategy = ImprovementGlobalStoppingStrategy(
            min_trials=self.n_init_evals + self.early_stopping_improvement_window,
            window_size=self.early_stopping_improvement_window,
            improvement_bar=self.early_stopping_improvement_bar,
            inactive_when_pending_trials=True,
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
            if x0 is not None:
                x0_init = np.array(x0, ndmin=2)

                if len(x0_init) < self.n_init_evals:
                    warnings.warn(
                        "Initial population smaller than popsize. "
                        "Creating missing entries."
                    )
                    n_remaining = self.n_init_evals - len(x0_init)
                    x0_remaining = optimization_problem.create_initial_values(
                        n_remaining, seed=self.seed, include_dependent_variables=False
                    )
                    x0_init = np.vstack((x0_init, x0_remaining))
                elif len(x0_init) > self.n_init_evals:
                    warnings.warn(
                        "Initial population larger than popsize. Omitting overhead."
                    )
                    x0_init = x0_init[0 : self.n_init_evals]

            else:
                # Create initial samples if they are not provided
                x0_init = self.optimization_problem.create_initial_values(
                    n_samples=self.n_init_evals,
                    include_dependent_variables=False,
                    seed=self.seed + 5641,
                )

            x0_init_transformed = np.array(optimization_problem.transform(x0_init))
            self._create_manual_trial(x0_init_transformed)
            print(exp_to_df(self.ax_experiment))

        n_iter = self.results.n_gen
        n_evals = self.results.n_evals

        global_stopping_message = None

        if n_evals >= self.n_max_evals:
            raise CADETProcessError(
                f"Initial number of evaluations exceeds `n_max_evals` "
                f"({self.n_max_evals})."
            )

        with manual_seed(seed=self.seed):
            while not (n_evals >= self.n_max_evals or n_iter >= self.n_max_iter):
                # Reinitialize GP+EI model at each step with updated data.
                modelbridge = self.train_model()

                print(f"Running optimization trial {n_evals + 1}/{self.n_max_evals}...")

                # samples can be accessed here by sample_generator.arms:
                sample_generator = modelbridge.gen(n=1)

                # A staging phase can be implemented here if needed.
                # See: https://github.com/fau-advanced-separations/CADET-Process/issues/53

                # The strategy itself will check if enough trials have already been
                # completed.
                (
                    stop_optimization,
                    global_stopping_message,
                ) = self.global_stopping_strategy.should_stop_optimization(
                    experiment=self.ax_experiment
                )

                if stop_optimization:
                    print(global_stopping_message)
                    break

                trial = self.ax_experiment.new_trial(generator_run=sample_generator)
                trial.run()

                trial.mark_completed()
                self._post_processing(trial)

                n_iter += 1
                n_evals += len(trial.arms)

        print(exp_to_df(self.ax_experiment))

        self.results.success = True
        self.results.exit_flag = 0
        self.results.exit_message = global_stopping_message


class SingleObjectiveAxInterface(AxInterface):
    def _setup_optimization_config(
        self,
        objectives: list[Objective],
        outcome_constraints: OutcomeConstraint,
    ) -> OptimizationConfig:
        return OptimizationConfig(
            objective=objectives[0], outcome_constraints=outcome_constraints
        )


class MultiObjectiveAxInterface(AxInterface):
    supports_multi_objective = True

    def _setup_optimization_config(
        self,
        objectives: list[Objective],
        outcome_constraints: OutcomeConstraint,
    ) -> MultiObjectiveOptimizationConfig:
        return MultiObjectiveOptimizationConfig(
            objective=MultiObjective(objectives),
            outcome_constraints=outcome_constraints,
        )


class GPEI(SingleObjectiveAxInterface):
    """Gaussian Process with Expected Improvement for single objectives."""

    def __repr__(self) -> str:
        """str: String representation of the optimization algorithm."""
        return "GPEI"

    def train_model(self) -> Models:
        """Train model."""
        return Models.GPEI(
            experiment=self.ax_experiment, data=self.ax_experiment.fetch_data()
        )


class BotorchModular(SingleObjectiveAxInterface):
    """
    Modular bayesian optimization algorithm.

    BotorchModular takes 2 optional arguments and uses the BOTORCH_MODULAR API of Ax to construct
    a Model which connects both components with the respective transforms necessary.

    Attributes
    ----------
    acquisition_fn: type, optional
        AcquisitionFunction class. The default is LogExpectedImprovement.
    surrogate_model: type, optional
        Model class. The default is SingleTaskGP.
    """

    acquisition_fn = Typed(ty=type, default=LogExpectedImprovement)
    surrogate_model = Typed(ty=type, default=SingleTaskGP)

    _specific_options = ["acquisition_fn", "surrogate_model"]

    def __repr__(self) -> str:
        """str: String representation of the optimization algorithm."""
        afn = self.acquisition_fn.__name__
        smn = self.surrogate_model.__name__

        return f"BotorchModular({smn}+{afn})"

    def train_model(self) -> NotImplementedError:
        """Train model."""
        raise NotImplementedError(
            "This model is currently broken. Please use Only GPEI or NEHVI"
        )
        return Models.BOTORCH_MODULAR(
            experiment=self.ax_experiment,
            surrogate=Surrogate(self.surrogate_model),
            botorch_acqf_class=self.acquisition_fn,
            data=self.ax_experiment.fetch_data(),
        )


class NEHVI(MultiObjectiveAxInterface):
    """Noisy expected hypervolume improvement multi-objective algorithm."""

    supports_single_objective = False

    def __repr__(self) -> str:
        """str: String representation of the optimization algorithm."""
        smn = "SingleTaskGP"
        afn = "NEHVI"

        return f"{smn}+{afn}"

    def train_model(self) -> Models:
        """Train model."""
        return Models.MOO(
            experiment=self.ax_experiment, data=self.ax_experiment.fetch_data()
        )


class qNParEGO(MultiObjectiveAxInterface):
    """
    qNParEGO multi-objective algorithm.

    ParEGO transforms the MOO problem into a single objective problem by applying a
    randomly weighted augmented Chebyshev scalarization to the objectives, and
    maximizing the expected improvement of that scalarized quantity (Knowles, 2006).
    Recently, Daulton et al. (2020) used a multi-output Gaussian process and
    compositional Monte Carlo objective to extend ParEGO to the batch setting (qParEGO),
    which proved to be a strong baseline for MOBO. Additionally, the authors proposed a
    noisy variant (qNParEGO), but the empirical evaluation of qNParEGO was limited.
    [Daulton et al. 2021 "Parallel Bayesian Optimization of Multiple Noisy Objectives
    with Expected Hypervolume Improvement"]
    """

    supports_single_objective = False

    def __repr__(self) -> str:
        """str: String representation of the algorithm."""
        smn = "SingleTaskGP"
        afn = "qNParEGO"

        return f"{smn}+{afn}"

    def train_model(self) -> Models:
        """Train model."""
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
