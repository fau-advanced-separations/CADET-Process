"""
Characterization helpers for LC systems.

Provides :class:`CharacterizeBase` and its subclasses, which encode domain
knowledge about which parameters to fit for each type of characterization
experiment and what sensible bounds to use.
All subclasses are :class:`~CADETProcess.optimization.OptimizationProblem`
instances and can be handed directly to any
:class:`~CADETProcess.optimization.OptimizerBase`.

The standalone :func:`setup_comparators` helper builds one
:class:`~CADETProcess.comparison.Comparator` per process from a matching list
of :class:`~CADETProcess.reference.ReferenceIO` objects, for users who want the
convenience without building comparators by hand.
"""

import os
from typing import Optional

import numpy as np

from CADETProcess.comparison import Comparator
from CADETProcess.instruments import LCProcess
from CADETProcess.optimization import OptimizationProblem
from CADETProcess.reference import ReferenceIO
from CADETProcess.simulator import SimulatorBase


def setup_comparators(
    processes: LCProcess | list[LCProcess],
    references: ReferenceIO | list[ReferenceIO],
    solution_path: str,
    metrics: list[str],
    components: list[str] | None = None,
    start: float | list[float | None] | None = None,
    end: float | list[float | None] | None = None,
) -> list[Comparator]:
    """
    Build one Comparator per process from a matching list of references.

    Parameters
    ----------
    processes : LCProcess or list[LCProcess]
        Processes in the same order as `references`.
    references : ReferenceIO or list[ReferenceIO]
        One reference per process.
    solution_path : str
        Path into ``SimulationResults.solution_cycles`` used by every metric,
        e.g. ``"column.outlet.outlet[0]"``.
    metrics : list[str]
        Difference metric class names, e.g. ``["Shape"]`` or ``["NRMSE"]``.
    components : list[str], optional
        Component names passed to each metric.
    start : float or list[float or None], optional
        Start of the comparison window in seconds.
        A scalar applies to every process; a list supplies one value per process.
    end : float or list[float or None], optional
        End of the comparison window in seconds.
        A scalar applies to every process; a list supplies one value per process.

    Returns
    -------
    list[Comparator]
        One comparator per process, in the same order as `processes`.

    Raises
    ------
    ValueError
        If the number of references does not match the number of processes,
        or if ``start``/``end`` lists have the wrong length.
    """
    if not isinstance(processes, list):
        processes = [processes]
    if not isinstance(references, list):
        references = [references]

    n = len(processes)

    if len(references) != n:
        raise ValueError(
            f"Got {len(references)} references for {n} processes; "
            "supply one reference per process."
        )

    def _expand(value: object, label: str) -> list:
        if value is None or not isinstance(value, list):
            return [value] * n
        if len(value) != n:
            raise ValueError(
                f"`{label}` has {len(value)} entries for {n} processes."
            )
        return value

    starts = _expand(start, "start")
    ends = _expand(end, "end")

    comparators = []
    for process, reference, t0, t1 in zip(processes, references, starts, ends):
        comp = Comparator(process.name)
        comp.add_reference(reference)
        for metric in metrics:
            comp.add_difference_metric(
                metric,
                reference,
                solution_path,
                components=components,
                start=t0,
                end=t1,
            )
        comparators.append(comp)

    return comparators


class CharacterizeBase(OptimizationProblem):
    """
    Base class for LC system characterization optimization problems.

    Wires one or more :class:`~CADETProcess.instruments.LCProcess` objects,
    a simulator, and a comparator per process into an
    :class:`~CADETProcess.optimization.OptimizationProblem`.
    A plot callback is registered automatically so every evaluated individual
    produces a comparison figure.

    Subclasses define their default variable specifications by overriding
    :meth:`_default_variables`.
    Users can override bounds or transforms for individual variables by passing
    the variable name as a keyword argument with a dict of overrides, e.g.
    ``bed_porosity={"lb": 0.35, "ub": 0.45, "transform": "auto"}``.

    Parameters
    ----------
    name : str
        Name of the optimization problem.
    processes : LCProcess or list[LCProcess]
        Process(es) to optimize.
    comparators : Comparator or list[Comparator]
        One comparator per process, in the same order as `processes`.
        Pass a single :class:`~CADETProcess.comparison.Comparator` when there
        is only one process.
    simulator : SimulatorBase
        Configured simulator instance, e.g. :class:`~CADETProcess.simulator.Cadet`.
    cache_directory : path-like, optional
        Directory for the optimization result cache.
    **kwargs
        Per-variable override dicts keyed by variable name, plus any remaining
        keyword arguments forwarded to
        :class:`~CADETProcess.optimization.OptimizationProblem`.
    """

    def __init__(
        self,
        name: str,
        processes: LCProcess | list[LCProcess],
        comparators: Comparator | list[Comparator],
        simulator: SimulatorBase,
        cache_directory: Optional[os.PathLike] = None,
        **kwargs: object,
    ) -> None:
        if not isinstance(processes, list):
            processes = [processes]
        if isinstance(comparators, Comparator):
            comparators = [comparators]

        if len(comparators) != len(processes):
            raise ValueError(
                f"Got {len(comparators)} comparators for {len(processes)} processes; "
                "supply one comparator per process."
            )

        # Keyed internally for the callback lookup
        comparator_map = {p.name: c for p, c in zip(processes, comparators)}

        # Merge per-variable overrides supplied as kwargs
        defaults = self._default_variables()
        var_names = {v["name"] for v in defaults}
        var_overrides = {k: kwargs.pop(k) for k in list(kwargs) if k in var_names}
        variables = [{**v, **var_overrides.get(v["name"], {})} for v in defaults]

        super().__init__(name=name, cache_directory=cache_directory, **kwargs)

        for process in processes:
            self.add_evaluation_object(process)

        for var in variables:
            self.add_variable(**var)

        self.add_evaluator(simulator)

        for process, comparator in zip(processes, comparators):
            self.add_objective(
                comparator,
                n_objectives=comparator.n_metrics,
                requires=[simulator],
                evaluation_objects=process,
            )

        def callback(
            simulation_results: object,
            individual: object,
            evaluation_object: object,
            callbacks_dir: str = "./",
        ) -> None:
            comparator_map[evaluation_object.name].plot_comparison(
                simulation_results,
                file_name=(
                    f"{callbacks_dir}/{individual.id}"
                    f"_{evaluation_object}_comparison.png"
                ),
                show=False,
            )

        self.add_callback(callback, requires=[simulator], frequency=1)

    def _default_variables(self) -> list[dict]:
        """
        Return the default variable specifications for this characterization type.

        Subclasses override this method to define the parameters to fit,
        their parameter paths, and default bounds or transforms.
        The base implementation returns an empty list.
        """
        return []


class CharacterizeTubing(CharacterizeBase):
    """
    Fit the length and axial dispersion of a named tubing segment.

    Parameters
    ----------
    name : str
        Name of the optimization problem.
    processes : LCProcess or list[LCProcess]
        Process(es) to optimize.
    tubing : str
        Name of the tubing unit in the flow sheet, e.g.
        ``"tubing_pre_injection"`` or ``"tubing_post_column"``.
    comparators : Comparator or list[Comparator]
        One comparator per process.
    simulator : SimulatorBase
        Configured simulator instance.
    **kwargs
        Per-variable override dicts (e.g. ``tubing_pre_injection_length={"ub": 1.0}``)
        and remaining keyword arguments forwarded to :class:`CharacterizeBase`.
    """

    def __init__(
        self,
        name: str,
        processes: LCProcess | list[LCProcess],
        tubing: str,
        comparators: Comparator | list[Comparator],
        simulator: SimulatorBase,
        **kwargs: object,
    ) -> None:
        self._tubing = tubing
        super().__init__(
            name=name,
            processes=processes,
            comparators=comparators,
            simulator=simulator,
            **kwargs,
        )

    def _default_variables(self) -> list[dict]:
        t = self._tubing
        return [
            {
                "name": f"{t}_length",
                "parameter_path": f"flow_sheet.{t}.length",
                "lb": 1e-2,
                "ub": 2.0,
                "transform": "auto",
            },
            {
                "name": f"{t}_axial_dispersion",
                "parameter_path": f"flow_sheet.{t}.axial_dispersion",
                "lb": 1e-9,
                "ub": 1e-2,
                "transform": "auto",
            },
        ]


class CharacterizePreInjection(CharacterizeBase):
    """
    Fit the pre-injection tubing length and mixer volume.

    Parameters
    ----------
    name : str
        Name of the optimization problem.
    processes : LCProcess or list[LCProcess]
        Process(es) to optimize.
    comparators : Comparator or list[Comparator]
        One comparator per process.
    simulator : SimulatorBase
        Configured simulator instance.
    **kwargs
        Per-variable override dicts (e.g. ``mixer_volume={"ub": 1e-4}``)
        and remaining keyword arguments forwarded to :class:`CharacterizeBase`.
    """

    def _default_variables(self) -> list[dict]:
        return [
            {
                "name": "tubing_pre_injection_length",
                "parameter_path": "flow_sheet.tubing_pre_injection.length",
                "lb": 1e-2,
                "ub": 2.0,
                "transform": "auto",
            },
            {
                "name": "mixer_volume",
                "parameter_path": "flow_sheet.mixer.init_liquid_volume",
                "lb": 1e-8,
                "ub": 1e-5,
                "transform": "auto",
            },
        ]


class CharacterizeBed(CharacterizeBase):
    """
    Fit column bed porosity and axial dispersion.

    Parameters
    ----------
    name : str
        Name of the optimization problem.
    processes : LCProcess or list[LCProcess]
        Process(es) to optimize.
    comparators : Comparator or list[Comparator]
        One comparator per process.
    simulator : SimulatorBase
        Configured simulator instance.
    **kwargs
        Per-variable override dicts (e.g. ``bed_porosity={"lb": 0.35, "ub": 0.45}``)
        and remaining keyword arguments forwarded to :class:`CharacterizeBase`.
    """

    def _default_variables(self) -> list[dict]:
        return [
            {
                "name": "bed_porosity",
                "parameter_path": "flow_sheet.column.bed_porosity",
                "lb": 0.2,
                "ub": 0.6,
                "transform": "auto",
            },
            {
                "name": "axial_dispersion",
                "parameter_path": "flow_sheet.column.axial_dispersion",
                "lb": 1e-11,
                "ub": 1e-3,
                "transform": "auto",
            },
        ]


class CharacterizeParticles(CharacterizeBase):
    """
    Fit particle-phase transport parameters.

    At least one of the ``include_*`` flags must be set to ``True``.

    Parameters
    ----------
    name : str
        Name of the optimization problem.
    processes : LCProcess or list[LCProcess]
        Process(es) to optimize.
    comparators : Comparator or list[Comparator]
        One comparator per process.
    simulator : SimulatorBase
        Configured simulator instance.
    include_axial_dispersion : bool
        Fit axial dispersion for ``component_index``.
    include_particle_porosity : bool
        Fit particle porosity (scalar).
    include_film_diffusion : bool
        Fit film diffusion coefficient for ``component_index``.
    include_pore_diffusion : bool
        Fit pore diffusion coefficient for ``component_index``.
    component_index : int
        Column index in parameter arrays for per-component transport parameters.
        Default 0.
    **kwargs
        Per-variable override dicts and remaining keyword arguments forwarded
        to :class:`CharacterizeBase`.

    Raises
    ------
    ValueError
        If no ``include_*`` flag is set.
    """

    def __init__(
        self,
        name: str,
        processes: LCProcess | list[LCProcess],
        comparators: Comparator | list[Comparator],
        simulator: SimulatorBase,
        include_axial_dispersion: bool = False,
        include_particle_porosity: bool = False,
        include_film_diffusion: bool = False,
        include_pore_diffusion: bool = False,
        component_index: int = 0,
        **kwargs: object,
    ) -> None:
        self._include_axial_dispersion = include_axial_dispersion
        self._include_particle_porosity = include_particle_porosity
        self._include_film_diffusion = include_film_diffusion
        self._include_pore_diffusion = include_pore_diffusion
        self._component_index = component_index
        super().__init__(
            name=name,
            processes=processes,
            comparators=comparators,
            simulator=simulator,
            **kwargs,
        )

    def _default_variables(self) -> list[dict]:
        ci = self._component_index
        variables = []
        if self._include_axial_dispersion:
            variables.append({
                "name": "axial_dispersion",
                "parameter_path": "flow_sheet.column.axial_dispersion",
                "lb": 1e-9,
                "ub": 1e-5,
                "indices": [ci],
                "transform": "auto",
            })
        if self._include_particle_porosity:
            variables.append({
                "name": "particle_porosity",
                "parameter_path": "flow_sheet.column.particle_porosity",
                "lb": 0.6,
                "ub": 0.9,
                "transform": "auto",
            })
        if self._include_film_diffusion:
            variables.append({
                "name": "film_diffusion",
                "parameter_path": "flow_sheet.column.film_diffusion",
                "lb": 1e-7,
                "ub": 1e-3,
                "indices": [ci],
                "transform": "auto",
            })
        if self._include_pore_diffusion:
            variables.append({
                "name": "pore_diffusion",
                "parameter_path": "flow_sheet.column.pore_diffusion",
                "lb": 1e-12,
                "ub": 1e-6,
                "indices": [ci],
                "transform": "auto",
            })

        if not variables:
            raise ValueError(
                "Specify at least one of: include_axial_dispersion, "
                "include_particle_porosity, include_film_diffusion, "
                "include_pore_diffusion."
            )

        return variables


class CharacterizeCapacity(CharacterizeBase):
    """
    Fit the binding model capacity.

    Parameters
    ----------
    name : str
        Name of the optimization problem.
    processes : LCProcess or list[LCProcess]
        Process(es) to optimize.
    comparators : Comparator or list[Comparator]
        One comparator per process.
    simulator : SimulatorBase
        Configured simulator instance.
    **kwargs
        Per-variable override dicts (e.g. ``capacity={"ub": 5.0}``)
        and remaining keyword arguments forwarded to :class:`CharacterizeBase`.
    """

    def _default_variables(self) -> list[dict]:
        return [
            {
                "name": "capacity",
                "parameter_path": "flow_sheet.column.binding_model.capacity",
                "lb": 1e-2,
                "ub": 2.0,
                "transform": "auto",
            },
        ]


class CharacterizeAdsorptionParameters(CharacterizeBase):
    r"""
    Fit steric-mass-action adsorption parameters.

    In equilibrium mode (``is_kinetic=False``) the desorption rate is fixed at
    1 and the adsorption rate is the equilibrium constant directly.
    In kinetic mode two independent variables (``equilibrium_constant`` and
    ``kinetic_constant``) are linked to the adsorption and desorption rates via
    variable dependencies, which preserves the thermodynamic constraint
    $k_\\text{des} = k_\\text{ads} / K_\\text{eq}$.

    Parameters
    ----------
    name : str
        Name of the optimization problem.
    processes : LCProcess or list[LCProcess]
        Process(es) to optimize.
        The binding model must already be attached to
        ``process.flow_sheet.column``.
    comparators : Comparator or list[Comparator]
        One comparator per process.
    simulator : SimulatorBase
        Configured simulator instance.
    is_kinetic : bool
        Use kinetic (True) or rapid-equilibrium (False) binding mode.
        Default False.
    include_film_diffusion : bool
        Also fit film diffusion for ``component_index``. Default False.
    include_pore_diffusion : bool
        Also fit pore diffusion for ``component_index``. Default False.
    component_index : int
        Column index in parameter arrays. Default 0.
    **kwargs
        Per-variable override dicts and remaining keyword arguments forwarded
        to :class:`CharacterizeBase`.
    """

    def __init__(
        self,
        name: str,
        processes: LCProcess | list[LCProcess],
        comparators: Comparator | list[Comparator],
        simulator: SimulatorBase,
        is_kinetic: bool = False,
        include_film_diffusion: bool = False,
        include_pore_diffusion: bool = False,
        component_index: int = 0,
        **kwargs: object,
    ) -> None:
        self._is_kinetic = is_kinetic
        self._include_film_diffusion = include_film_diffusion
        self._include_pore_diffusion = include_pore_diffusion
        self._component_index = component_index

        if not isinstance(processes, list):
            processes = [processes]

        binding_model = processes[0].flow_sheet.column.binding_model
        lambda_ = binding_model.capacity

        for process in processes:
            bm = process.flow_sheet.column.binding_model
            bm.is_kinetic = is_kinetic
            if is_kinetic:
                bm.reference_liquid_phase_conc = lambda_
                bm.reference_solid_phase_conc = lambda_
            else:
                bm.desorption_rate = 1

        super().__init__(
            name=name,
            processes=processes,
            comparators=comparators,
            simulator=simulator,
            **kwargs,
        )

        if is_kinetic:
            self.add_variable_dependency(
                dependent_variable="desorption_rate",
                independent_variables=["kinetic_constant"],
                transform=lambda k_kin: 1 / k_kin,
            )
            self.add_variable_dependency(
                dependent_variable="adsorption_rate",
                independent_variables=["kinetic_constant", "equilibrium_constant"],
                transform=lambda k_kin, k_eq: k_eq / k_kin,
            )

            def _select_best(pareto_population: object) -> object:
                f_sum = np.sum(pareto_population.f_minimized, axis=1)
                x = pareto_population.x[f_sum.argsort()]
                return x[:1]

            self.add_multi_criteria_decision_function(_select_best)

    def _default_variables(self) -> list[dict]:
        ci = self._component_index
        k_eq_lb, k_eq_ub = 1e-3, 1.0

        variables = [
            {
                "name": "characteristic_charge",
                "parameter_path": (
                    "flow_sheet.column.binding_model.characteristic_charge"
                ),
                "lb": 4,
                "ub": 9,
                "indices": [ci],
                "transform": "auto",
            },
        ]

        if not self._is_kinetic:
            variables.append({
                "name": "adsorption_rate",
                "parameter_path": (
                    "flow_sheet.column.binding_model.adsorption_rate"
                ),
                "lb": k_eq_lb,
                "ub": k_eq_ub,
                "indices": [ci],
                "transform": "auto",
            })
        else:
            variables += [
                {
                    "name": "adsorption_rate",
                    "parameter_path": (
                        "flow_sheet.column.binding_model.adsorption_rate"
                    ),
                    "indices": [ci],
                },
                {
                    "name": "desorption_rate",
                    "parameter_path": (
                        "flow_sheet.column.binding_model.desorption_rate"
                    ),
                    "indices": [ci],
                },
                {
                    "name": "equilibrium_constant",
                    "lb": k_eq_lb,
                    "ub": k_eq_ub,
                    "transform": "auto",
                    "evaluation_objects": None,
                },
                {
                    "name": "kinetic_constant",
                    "lb": 1e-9,
                    "ub": 1.0,
                    "transform": "auto",
                    "evaluation_objects": None,
                },
            ]

        if self._include_film_diffusion:
            variables.append({
                "name": "film_diffusion",
                "parameter_path": "flow_sheet.column.film_diffusion",
                "lb": 1e-7,
                "ub": 1e-4,
                "indices": [ci],
                "transform": "auto",
            })
        if self._include_pore_diffusion:
            variables.append({
                "name": "pore_diffusion",
                "parameter_path": "flow_sheet.column.pore_diffusion",
                "lb": 1e-12,
                "ub": 1e-6,
                "indices": [ci],
                "transform": "auto",
            })

        return variables
