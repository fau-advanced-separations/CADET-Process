import warnings
from typing import Any, Dict, Optional, Mapping

import numpy as np
import numpy.typing as npt
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core import (
    ComparisonOp,
    Objective,
    OutcomeConstraint,
    # Arm,
    # Data,
    # Experiment,
    # Models,
    # MultiObjective,
    # MultiObjectiveOptimizationConfig,
    # OptimizationConfig,
    # ParameterConstraint,
    # ParameterType,
    # RangeParameter,
    SearchSpace,
)
from ax.core.base_trial import BaseTrial
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
# from ax.models.torch.botorch_defaults import get_qLogNEI
# from ax.models.torch.botorch_modular.surrogate import Surrogate
# from ax.service.utils.report_utils import exp_to_df
# from ax.utils.common.result import Err, Ok
# from botorch.acquisition.analytic import LogExpectedImprovement
# from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.sampling import manual_seed
from botorch.exceptions.errors import CandidateGenerationError

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Float, UnsignedInteger
from CADETProcess.optimization import OptimizerBase
from CADETProcess.optimization.optimizationProblem import OptimizationProblem
from CADETProcess.optimization.parallelizationBackend import (
    ParallelizationBackendBase,
    SequentialBackend,
)

__all__ = [
    "GPEI",
]

# class CADETProcessMetric(IMetric):
#     def __init__(
#         self,
#         name: str,
#         lower_is_better: Union[bool, None] = None,
#         properties: Union[Dict[str, Any], None] = None,
#     ) -> None:
#         super().__init__(name, lower_is_better, properties)

#     def fetch_trial_data(
#         self,
#         trial: BaseTrial,
#         **kwargs: Any,
#     ) -> MetricFetchResult:
#         try:
#             trial_results = trial.run_metadata
#             records = []
#             for arm_name, arm in trial.arms_by_name.items():
#                 results_dict = {
#                     "trial_index": trial.index,
#                     "arm_name": arm_name,
#                     "metric_name": self.name,
#                 }

#                 # this looks up the value of the objective function
#                 # generated in the runner
#                 arm_results = trial_results["arms"][arm]
#                 results_dict.update({"mean": arm_results[self.name]})
#                 results_dict.update({"sem": 0.0})

#                 records.append(results_dict)

#             return Ok(Data(df=pd.DataFrame.from_records(records)))
#         except Exception as e:
#             return Err(
#                 MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
#             )


class CADETProcessRunner:
    def __init__(
        self,
        optimization_problem: OptimizationProblem,
        parallelization_backend: ParallelizationBackendBase,
    ) -> None:
        self.optimization_problem = optimization_problem
        self.parallelization_backend = parallelization_backend

    def run_trials(
        self,
        trials: Dict[str | int, Mapping[str, float]]
    ) -> Dict[str | int, Dict[str, float]]:
        # Get X from arms.
        X = []
        for trial_index, trial in trials.items():
            x = np.array(list(trial.values()))
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
            trials, F, objective_labels, CV, nonlincon_labels
        )

        return trial_metadata

    @staticmethod
    def get_metadata(
        trials: Dict[str | int, Mapping[str, float]],
        F: np.ndarray,
        objective_labels: list[str],
        CV: Optional[np.ndarray] = None,
        nonlincon_labels: Optional[list[str]] = None,
    ) -> dict:
        trial_metadata = {}

        for results_index, (trial_index, trial) in enumerate(trials.items()):
            f_dict = {
                metric: f_metric[results_index]
                for metric, f_metric in zip(objective_labels, F.T)
            }
            cv_dict = {}
            if CV is not None:
                assert nonlincon_labels is not None, (
                    "If CV are given, nonlinear-constraint labels must also be given."
                )
                cv_dict = {
                    metric: cv_metric[results_index]
                    for metric, cv_metric in zip(nonlincon_labels, CV.T)
                }
            trial_metadata.update({trial_index: {**f_dict, **cv_dict}})

        return trial_metadata


class AxInterface(OptimizerBase):
    """Wrapper around Ax's bayesian optimization API."""

    supports_bounds = True
    supports_multi_objective = True
    supports_linear_constraints = True
    supports_linear_equality_constraints = False
    supports_nonlinear_constraints = True

    early_stopping_improvement_window = UnsignedInteger(default=1000)
    early_stopping_improvement_bar = Float(default=1e-10)
    n_parallel_evals = UnsignedInteger(default=3)
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

            if "-" in var.name or "+" in var.name:
                raise CADETProcessError(
                    f"Bad parameter name: '{var.name}'. Ax does not " +
                    "support dashes ('-','+') in parameter names."
                )

            lb, ub = var.transformed_bounds
            param = RangeParameterConfig(
                name=var.name,
                parameter_type="float",
                bounds=(lb, ub),
                scaling="linear",
                step_size=None,
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
            lhs = " + ".join([f"{a} * {var.name}" for var, a in zip(indep_vars, a_t)])
            rhs = b_t
            constr = f"{lhs} <= {rhs}"
            parameter_constraints.append(constr)

        return parameter_constraints

    @classmethod
    def _setup_searchspace(
        cls, optimizationProblem: OptimizationProblem
    ) -> SearchSpace:
        parameters = cls._setup_parameters(optimizationProblem)
        parameter_constraints = cls._setup_linear_constraints(optimizationProblem)
        return parameters, parameter_constraints

    def _setup_objectives(self) -> str:
        """Parse objective functions from optimization problem."""
        objective_names = self.optimization_problem.objective_labels

        objectives = []
        for i, obj_name in enumerate(objective_names):
            # TODO: add +
            if "-" in obj_name:
                raise CADETProcessError(
                    f"Bad objective name: '{obj_name}'. Ax does not support " +
                    "dashes ('-') in objective names "
                )

            # minus is prepended to indicate minimization
            objectives.append(f"-{obj_name}")

        return ", ".join(objectives)

    def _setup_outcome_constraints(self) -> list:
        """Parse nonliear constraint functions from optimization problem."""
        outcome_constraints = []
        for constr_label, constr_bound in zip(
            self.optimization_problem.nonlinear_constraint_labels,
            self.optimization_problem.nonlinear_constraints_bounds,
        ):
            if "<lambda>" in constr_label:
                raise CADETProcessError(
                    "lambda functions are not allowed as nonlinear constraints " +
                    "under Ax usage."
                )
            # TODO: @Jo is it correct that the operator is always leq?
            nonlincon = "{constraint} <= {bound}".format(
                constraint=constr_label,
                bound=constr_bound,
            )
            outcome_constraints.append(nonlincon)

        return outcome_constraints

    def _create_manual_data(
        self, trial: Dict, F: np.ndarray, CV: Optional[np.ndarray] = None
    ) -> dict:
        objective_labels = self.optimization_problem.objective_labels
        nonlincon_labels = self.optimization_problem.nonlinear_constraint_labels
        return CADETProcessRunner.get_metadata(
            trial, F, objective_labels, CV, nonlincon_labels
        )

    def _create_manual_trials(self, X: np.ndarray) -> Dict[str | int, Mapping[str, float]]:
        """Create trial from pre-evaluated data."""
        variables = self.optimization_problem.independent_variable_names

        trials = {}
        for x in X:
            par = {var: x_i for var, x_i in zip(variables, x)}
            _id = self.client._experiment.num_trials

            trial_index = self.client.attach_trial(
                arm_name=_id,
                parameters=par
            )
            trials.update({trial_index: par})

        return trials

    def _complete_trials(
        self, trials: Dict[str | int, Mapping[str, float]],
        data: Dict[str | int, Dict[str, float]]
    ) -> None:
        for trial_index, _ in trials.items():
            print(f"Completed {trial_index=} with {data[trial_index]=}")
            self.client.complete_trial(
                trial_index=trial_index,
                raw_data=data[trial_index]
            )

    def _post_processing(
            self,
            trials: Dict[str | int, Mapping[str, float]],
            results: Dict[str | int, Dict[str, float]],
            generation: int,
        ) -> None:
        """
        Run post processing.

        Expects a Dictionary of Dicts that hold the variable names : values pairs
        the same for results (objective function, nonlinear constraints)

        Parameters
        ----------
        trials: Dict[int, Dict[str, float]]
            The values (variables) for which the problem was evaluated. Each index
            corresponds to a complete trial and holds its variable:value pairs
        results: Dict[int, Dict[str, float]]
            The values (objective function results) of the problem evaluated at the
            corresponding variable values. Each index corresponds to a complete trial
            and holds its variable:value pairs
        generation: int
            Index of the batch trial
        """
        op: OptimizationProblem = self.optimization_problem

        # Get variable value in the order of the optimization problem
        X = np.array([
            [trial[var] for var in op.variable_names]
            for _, trial in trials.items()
        ])

        # Get objective values
        F = np.array([
            [result[obj_label] for obj_label in op.objective_labels]
            for _, result in results.items()
        ])

        # Get nonlinear constraints values
        if op.n_nonlinear_constraints > 0:
            G = np.array([
                [result[obj_label] for obj_label in op.nonlinear_constraint_labels]
                for _, result in results.items()
            ])

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
            current_generation=generation,
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

        self.client = Client()

        parameters, constraints = self._setup_searchspace(self.optimization_problem)
        self.client.configure_experiment(
            parameters=parameters,
            parameter_constraints=constraints,
            name=str(optimization_problem),
            description=None,
            experiment_type=None,
            owner=None
        )

        objectives = self._setup_objectives()
        outcome_constraints = self._setup_outcome_constraints()
        self.client.configure_optimization(
            objective=objectives,
            outcome_constraints=outcome_constraints
        )

        self.runner = CADETProcessRunner(
            optimization_problem=self.optimization_problem,
            parallelization_backend=SequentialBackend(),
        )


        if False:
            # TODO: Earlier implementation. Needs to be migrated to Ax Client API
            self.global_stopping_strategy = ImprovementGlobalStoppingStrategy(
                min_trials=self.n_init_evals + self.early_stopping_improvement_window,
                window_size=self.early_stopping_improvement_window,
                improvement_bar=self.early_stopping_improvement_bar,
                inactive_when_pending_trials=True,
            )

        # Restore previous results from checkpoint
        if len(self.results.populations) > 0:
            for pop in self.results.populations:
                # TODO: @Jo: Is it correct to use transformed x here?
                X, F, CV = pop.x_transformed, pop.f, pop.cv_nonlincon
                trials = self._create_manual_trials(X)
                trial_data = self._create_manual_data(trials, F, CV)
                self._complete_trials(trials, trial_data)
                # this starts with N generations, depending how many generation
                # strategies completed in the previous run

        else:
            if x0 is not None:
                x0_init = np.array(x0, ndmin=2)

                if len(x0_init) < self.n_init_evals:
                    warnings.warn(
                        "Initial population smaller than popsize. " +
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
            trials = self._create_manual_trials(x0_init_transformed)

            # complete initial trials
            results = self.runner.run_trials(trials=trials)
            self._post_processing(trials=trials, results=results, generation=0)
            
            self._complete_trials(trials=trials, data=results)
            # this starts with 1 generation (the init trials)


        n_gen = self.results.n_gen  # first generation is the 0-th generation
        n_evals = self.results.n_evals

        msg = None
        exit_flag = 0
        success = True

        if n_evals >= self.n_max_evals:
            raise CADETProcessError(
                "Initial number of evaluations exceeds `n_max_evals` " +
                f"({self.n_max_evals})."
            )

        with manual_seed(seed=self.seed):
            # comparison against self.n_max_iter needs to be < (and not <=) because
            # the comparison is against the current generation that starts at 0. and the
            # states are updated at the end of the loop
            while n_evals < self.n_max_evals and n_gen < self.n_max_iter:
                print(f"Running optimization trial {n_evals + 1}/{self.n_max_evals}...")

                # ask
                # make sure the max_trials are not overfulfilled due to parallelism
                max_trials = min(self.n_parallel_evals, self.n_max_evals - n_evals)

                try:
                    trials = self.client.get_next_trials(max_trials=max_trials)
                except CandidateGenerationError as err:
                    # This is currently not 100% stable. The reason is that
                    # Ax might run into a situation where the acquisition fct. suggests
                    # values only close to the bounds, which are also the optimum.
                    # Then botorch accepts values near the linear constraints (bounds) with
                    # a precision of 1e-6. This limit is hardcoded and cannot be changed
                    # Repeating the process seems to help, but it is a hotfix.
                    # A more stable and guaranteed failsafe method would be desirable.
                    # Perhaps an update to a future version of Ax fixes this. I don't
                    # think it is good form to provide an optimizert which fails.
                    # Another option is to catch it and then exit the optimization early
                    # with success_code = 0
                    # TODO: @Jo: What do you prefer
                    msg = (
                        "Trials could not be generated due to too tight constraints. " +
                        "This could also indicate that optimization is close to the " +
                        "optimum and candidates are hard to find. " +
                        f"Retrying once, failing afterwards. {err}"
                    )
                    warnings.warn(msg)
                    trials = self.client.get_next_trials(max_trials=max_trials)

                # compute
                # Ax allows trials to be of type str, int, float, bool. This is not supported
                # by ax. Therefore typing suggests an error. We choose to ignore it.
                results = self.runner.run_trials(trials=trials)
                self._post_processing(results=results, trials=trials, generation=n_gen)

                # tell
                self._complete_trials(trials=trials, data=results)

                # # The strategy itself will check if enough trials have already been
                # # completed.
                # (
                #     stop_optimization,
                #     global_stopping_message,
                # ) = self.global_stopping_strategy.should_stop_optimization(
                #     experiment=self.ax_experiment
                # )

                # if stop_optimization:
                #     print(global_stopping_message)
                #     break

                n_gen += 1
                n_evals += len(trials)

        # pareto = self.client.get_pareto_frontier()
        # best_parameters, prediction, index, name = self.client.get_best_parameterization()
        # print("Best Parameters:", best_parameters)
        # print("Prediction (mean, variance):", prediction)

        self.results.success = success
        self.results.exit_flag = exit_flag
        self.results.exit_message = msg


# class SingleObjectiveAxInterface(AxInterface):
#     def _setup_optimization_config(
#         self,
#         objectives: list[Objective],
#         outcome_constraints: OutcomeConstraint,
#     ):
#         return OptimizationConfig(
#             objective=objectives[0], outcome_constraints=outcome_constraints
#         )


# class MultiObjectiveAxInterface(AxInterface):
#     supports_multi_objective = True

#     def _setup_optimization_config(
#         self,
#         objectives: list[Objective],
#         outcome_constraints: OutcomeConstraint,
#     ):
#         return MultiObjectiveOptimizationConfig(
#             objective=MultiObjective(objectives),
#             outcome_constraints=outcome_constraints,
#         )


class GPEI(AxInterface):
    """Gaussian Process with Expected Improvement for single objectives."""

    def __repr__(self) -> str:
        """str: String representation of the optimization algorithm."""
        return "BO"


# class BotorchModular(SingleObjectiveAxInterface):
#     """
#     Modular bayesian optimization algorithm.

#     BotorchModular takes 2 optional arguments and uses the BOTORCH_MODULAR API of Ax to construct
#     a Model which connects both components with the respective transforms necessary.

#     Attributes
#     ----------
#     acquisition_fn: type, optional
#         AcquisitionFunction class. The default is LogExpectedImprovement.
#     surrogate_model: type, optional
#         Model class. The default is SingleTaskGP.
#     """

#     acquisition_fn = Typed(ty=type)  # , default=LogExpectedImprovement)
#     surrogate_model = Typed(ty=type)  # , default=SingleTaskGP)

#     _specific_options = ["acquisition_fn", "surrogate_model"]

#     def __repr__(self) -> str:
#         """str: String representation of the optimization algorithm."""
#         afn = self.acquisition_fn.__name__
#         smn = self.surrogate_model.__name__

#         return f"BotorchModular({smn}+{afn})"

#     def train_model(self) -> NotImplementedError:
#         """Train model."""
#         raise NotImplementedError(
#             "This model is currently broken. Please use Only GPEI or NEHVI"
#         )
#         return Models.BOTORCH_MODULAR(
#             experiment=self.ax_experiment,
#             surrogate=Surrogate(self.surrogate_model),
#             botorch_acqf_class=self.acquisition_fn,
#             data=self.ax_experiment.fetch_data(),
#         )


# class NEHVI(MultiObjectiveAxInterface):
#     """Noisy expected hypervolume improvement multi-objective algorithm."""

#     supports_single_objective = False

#     def __repr__(self) -> str:
#         """str: String representation of the optimization algorithm."""
#         smn = "SingleTaskGP"
#         afn = "NEHVI"

#         return f"{smn}+{afn}"

#     def train_model(self):
#         """Train model."""
#         return Models.MOO(
#             experiment=self.ax_experiment, data=self.ax_experiment.fetch_data()
#         )


# class qNParEGO(MultiObjectiveAxInterface):
#     """
#     qNParEGO multi-objective algorithm.

#     ParEGO transforms the MOO problem into a single objective problem by applying a
#     randomly weighted augmented Chebyshev scalarization to the objectives, and
#     maximizing the expected improvement of that scalarized quantity (Knowles, 2006).
#     Recently, Daulton et al. (2020) used a multi-output Gaussian process and
#     compositional Monte Carlo objective to extend ParEGO to the batch setting (qParEGO),
#     which proved to be a strong baseline for MOBO. Additionally, the authors proposed a
#     noisy variant (qNParEGO), but the empirical evaluation of qNParEGO was limited.
#     [Daulton et al. 2021 "Parallel Bayesian Optimization of Multiple Noisy Objectives
#     with Expected Hypervolume Improvement"]
#     """

#     supports_single_objective = False

#     def __repr__(self) -> str:
#         """str: String representation of the algorithm."""
#         smn = "SingleTaskGP"
#         afn = "qNParEGO"

#         return f"{smn}+{afn}"

#     def train_model(self):
#         """Train model."""
#         return Models.MOO(
#             experiment=self.ax_experiment,
#             data=self.ax_experiment.fetch_data(),
#             acqf_constructor=get_qLogNEI,
#             default_model_gen_options={
#                 "acquisition_function_kwargs": {
#                     "chebyshev_scalarization": True,
#                 }
#             },
#         )
