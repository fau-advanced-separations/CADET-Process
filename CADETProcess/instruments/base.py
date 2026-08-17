"""
LC instrument templates for CADET-Process.

Event compilation model
-----------------------

All process templates compile declarative protocol descriptions into
CADET-Process ``Process`` events that drive ``flow_rate`` parameters on
individual buffer inlets.

:class:`Phase` is the basic unit: it declares a duration, a total flow rate,
and buffer-fraction compositions at the start and (optionally) end of the
phase.  Fractions are proportional; the flow rate for each buffer is
``flow_rate × fraction``.  Equal start and end compositions produce a constant-composition segment;
differing compositions produce a linear gradient.

:class:`PhasedProcess` is the canonical event compiler.  It collects every
buffer key that appears across *all* phases (``all_keys``) and emits one event
per key at the start of every phase.  Collecting keys across all phases (not
just the current one) ensures that a key active in an earlier phase is
explicitly zeroed when it is absent from a later phase, preventing residual
values from bleeding through.

Valve events (:meth:`LCProcess.add_valve_event`) are orthogonal to phase
events.  They model the 6-port injection valve via ``output_state`` changes on
``tubing_pre_injection`` and ``sample_loop``.  Each call generates one primary
event and one or more dependent events linked via ``add_event_dependency``, so
that moving the primary time automatically propagates to all coupled changes.
"""
import math
from dataclasses import dataclass
from typing import Literal

from CADETProcess.dataStructure import get_nested_value
from CADETProcess.processModel import (
    BindingBaseClass,
    ChromatographicColumnBase,
    ComponentSystem,
    Cstr,
    FlowSheet,
    Inlet,
    Outlet,
    Process,
    TubularReactor,
)
from CADETProcess.reference import ReferenceIO
from CADETProcess.simulator import Cadet
from CADETProcess.solution import slice_solution

_BUFFER_KEYS = {
    "A": "buffer_a",
    "B": "buffer_b",
    "C": "buffer_c",
    "D": "buffer_d",
    "F": "feed_inlet",
}

#: Units that may be named in bypass_units.
#: Bypassing "mixer" sets a near-zero volume instead of removing it (it is
#: always needed as a 4-to-1 buffer junction).
#: Bypassing "tubing_pre_injection" removes it; the mixer becomes the valve
#: junction directly.
_BYPASSABLE = frozenset({
    "mixer",
    "tubing_pre_injection",
    "tubing_pre_column",
    "column",
    "tubing_post_column",
    "tubing_detectors",
})

#: Valid 6-port valve positions for :meth:`LCProcess.add_valve_event`.
ValvePosition = Literal[
    "run",
    "load",
    "inject",
    "sample_pump_waste",
    "direct_inject",
    "system_pump_waste",
]


@dataclass(frozen=True)
class Phase:
    """
    A single phase in a phased LC process.

    Parameters
    ----------
    duration : float
        Phase duration in seconds. Must be positive.
    flow_rate : float
        Total flow rate in m³/s. Must be non-negative.
    composition_start : dict[str, float]
        Buffer fractions at the start of the phase. Keys are ``"A"``–``"D"``
        or ``"F"``. Missing keys default to 0. Fractions must sum to 1.
    composition_end : dict[str, float] | None
        Buffer fractions at the end of the phase. ``None`` (default) means a
        step (constant composition); a value different from
        ``composition_start`` produces a linear gradient.

    Notes
    -----
    Buffer F (feed inlet) cannot coexist with buffers A–D in a single
    composition dict.

    Examples
    --------
    Wash phase — 100 % buffer A at constant flow rate:

    >>> Phase(200.0, 8.3e-9, {"A": 1.0}).duration
    200.0

    Linear gradient from A to B over 400 s:

    >>> Phase(400.0, 8.3e-9, {"A": 1.0}, {"B": 1.0}).composition_end
    {'B': 1.0}

    Continuous sample delivery via feed inlet:

    >>> Phase(600.0, 8.3e-9, {"F": 1.0}).flow_rate
    8.3e-09
    """

    duration: float
    flow_rate: float
    composition_start: dict[str, float]
    composition_end: dict[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate and normalize composition dicts after dataclass construction."""
        # Defensive copies before the dataclass freezes the fields.
        object.__setattr__(self, "composition_start", dict(self.composition_start))
        if self.composition_end is None:
            object.__setattr__(self, "composition_end", dict(self.composition_start))
        else:
            object.__setattr__(self, "composition_end", dict(self.composition_end))

        if self.duration <= 0:
            raise ValueError(f"Phase duration must be positive, got {self.duration!r}")
        if self.flow_rate < 0:
            raise ValueError(
                f"Phase flow_rate must be non-negative, got {self.flow_rate!r}"
            )

        valid_keys = set(_BUFFER_KEYS)
        for label, fractions in (
            ("composition_start", self.composition_start),
            ("composition_end", self.composition_end),
        ):
            invalid = set(fractions) - valid_keys
            if invalid:
                raise ValueError(
                    f"Unknown buffer keys in {label}: {sorted(invalid)}"
                )
            if "F" in fractions and fractions["F"] > 0:
                other_nonzero = [k for k, v in fractions.items() if k != "F" and v > 0]
                if other_nonzero:
                    raise ValueError(
                        f"Buffer F cannot coexist with other buffers in {label}. "
                        f"Found: {other_nonzero}"
                    )
            if self.flow_rate > 0 and not fractions:
                raise ValueError(
                    f"{label} must not be empty when flow_rate > 0."
                )
            for k, v in fractions.items():
                if not (0.0 <= v <= 1.0):
                    raise ValueError(
                        f"Fraction for {k!r} in {label} must be in [0, 1], got {v!r}"
                    )
            if fractions:
                total = sum(fractions.values())
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(
                        f"Fractions in {label} must sum to 1, got {total!r}"
                    )


@dataclass(frozen=True)
class ValveEvent:
    """
    An instantaneous valve position change in a phased LC protocol.

    Unlike :class:`Phase`, a ``ValveEvent`` has no duration.
    Its absolute time is determined by the cumulative duration of all
    :class:`Phase` objects preceding it in the step sequence.

    Parameters
    ----------
    position : ValvePosition
        Target valve position. See :meth:`LCProcess.add_valve_event` for
        the available positions and their flow paths.

    Examples
    --------
    Inject at t=0, return to run after a 30 s injection:

    >>> steps = [
    ...     ValveEvent("inject"),
    ...     Phase(30.0, 8.3e-9, {"A": 1.0}),
    ...     ValveEvent("run"),
    ...     Phase(570.0, 8.3e-9, {"A": 1.0}),
    ... ]
    """

    position: ValvePosition


class LCFlowSheet(FlowSheet):
    """
    Flow sheet template for a standard LC system.

    Models the full instrument topology: four buffer inlets (A–D) feeding a
    mixer, a feed inlet (F), pre-injection tubing, an optional sample loop,
    optional post-injection path (pre-column tubing, column, post-column
    tubing, detector tubing), product outlet, and waste outlet.

    The default routing sends flow through the sample loop (inject position)
    when a loop is present, or directly to the first post-injection unit when
    it is not.
    Use :meth:`LCProcess.add_valve_event` to switch routing at runtime.

    Parameters
    ----------
    component_system : ComponentSystem
        Shared component system; passed to every unit in the flow sheet.
    sample_loop_volume : float | None, optional
        Sample loop volume in m³. ``None`` (default) omits the loop entirely;
        ``feed_inlet`` then connects directly to the first post-injection unit.
    sample_loop_diameter : float | None, optional
        Sample loop inner diameter in m. Used to compute loop length from the
        cross-sectional area. If ``None`` and a loop volume is given, the
        diameter is derived from the volume assuming a loop length of 1 m,
        i.e. ``d = sqrt(4 * volume / π)``.
    ColumnModel : type, optional
        Column unit operation class (e.g. ``LumpedRateModelWithPores``).
        Required unless ``"column"`` is listed in ``bypass_units``.
    BindingModel : type, optional
        Binding model class to attach to the column. Instantiated with
        ``component_system``; no parameters are set.
    bypass_units : list[str], optional
        Units to exclude or suppress. Valid names: ``"mixer"``,
        ``"tubing_pre_injection"``, ``"tubing_pre_column"``, ``"column"``,
        ``"tubing_post_column"``, ``"tubing_detectors"``.
        ``"mixer"`` is never truly removed (it is always needed as a buffer
        junction) but gets ``init_liquid_volume = 1e-9`` m³.
        ``"tubing_pre_injection"`` is removed; the mixer becomes the injection
        valve junction.

    Notes
    -----
    When all post-injection units are bypassed, ``first_unit`` falls back to
    ``outlet``, so flow goes directly from the loop/tubing to the product
    outlet.

    Without a sample loop, the ``"inject"`` and ``"load"`` valve positions
    degrade to ``"run"`` (no loop to switch in or out of the flow path).

    TODO: support multiple feed inlets.
    """

    def __init__(
        self,
        component_system: ComponentSystem,
        sample_loop_volume: float | None = None,
        sample_loop_diameter: float | None = None,
        ColumnModel: type[ChromatographicColumnBase] | None = None,
        BindingModel: type[BindingBaseClass] | None = None,
        bypass_units: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(component_system, **kwargs)

        if sample_loop_diameter is not None and sample_loop_volume is None:
            raise ValueError("sample_loop_diameter requires sample_loop_volume.")

        # Validate bypass_units upfront before any unit is added.
        bypass_set = frozenset(bypass_units) if bypass_units is not None else frozenset()
        unexpected = bypass_set - _BYPASSABLE
        if unexpected:
            raise ValueError(f"Unexpected bypass unit(s): {sorted(unexpected)}")
        if "column" not in bypass_set and ColumnModel is None:
            raise ValueError(
                "ColumnModel must be specified when column is not bypassed."
            )

        # Buffer inlets A–D (feed the mixer) and feed_inlet (direct to sample loop).
        for name in ("buffer_a", "buffer_b", "buffer_c", "buffer_d", "feed_inlet"):
            self.add_unit(Inlet(component_system, name=name))

        mixer = Cstr(component_system, "mixer")
        if "mixer" in bypass_set:
            mixer.init_liquid_volume = 1e-9
        self.add_unit(mixer)

        if "tubing_pre_injection" not in bypass_set:
            tubing_pre_injection = TubularReactor(
                component_system, name="tubing_pre_injection"
            )
            self.add_unit(tubing_pre_injection)
            valve_junction = tubing_pre_injection
        else:
            valve_junction = mixer

        # Sample loop (optional).
        has_loop = sample_loop_volume is not None
        if has_loop:
            if sample_loop_diameter is None:
                # Derive diameter from volume with unit length: d = sqrt(4V/π).
                sample_loop_diameter = math.sqrt(4 * sample_loop_volume / math.pi)
            sample_loop = TubularReactor(component_system, name="sample_loop")
            sample_loop.diameter = sample_loop_diameter
            sample_loop.length = sample_loop_volume / sample_loop.cross_section_area
            sample_loop.axial_dispersion = 0
            self.add_unit(sample_loop)

        # Post-injection units: add the ones not in bypass_set; track first_unit.
        first_unit = None

        if "tubing_pre_column" not in bypass_set:
            tubing_pre_column = TubularReactor(
                component_system, name="tubing_pre_column"
            )
            self.add_unit(tubing_pre_column)
            if first_unit is None:
                first_unit = tubing_pre_column

        if "column" not in bypass_set:
            column = ColumnModel(component_system, "column")
            if BindingModel is not None:
                column.binding_model = BindingModel(component_system)
            self.add_unit(column)
            if first_unit is None:
                first_unit = column

        if "tubing_post_column" not in bypass_set:
            tubing_post_column = TubularReactor(
                component_system, name="tubing_post_column"
            )
            self.add_unit(tubing_post_column)
            if first_unit is None:
                first_unit = tubing_post_column

        if "tubing_detectors" not in bypass_set:
            tubing_detectors = TubularReactor(
                component_system, name="tubing_detectors"
            )
            self.add_unit(tubing_detectors)
            if first_unit is None:
                first_unit = tubing_detectors

        outlet = Outlet(component_system, name="outlet")
        self.add_unit(outlet)
        # All post-injection units bypassed: route directly to outlet.
        if first_unit is None:
            first_unit = outlet

        waste = Outlet(component_system, name="waste")
        self.add_unit(waste)

        # Store topology flags; FlowSheet uses Structure descriptors that
        # block arbitrary attribute assignment, so bypass via object.__setattr__.
        object.__setattr__(self, "_first_unit_name", first_unit.name)
        object.__setattr__(self, "_has_sample_loop", has_loop)
        object.__setattr__(self, "_valve_junction_name", valve_junction.name)

        # Pre-injection connections.
        for key in ("A", "B", "C", "D"):
            self.add_connection(self[_BUFFER_KEYS[key]], mixer)
        if valve_junction is not mixer:
            self.add_connection(mixer, valve_junction)

        if has_loop:
            # feed_inlet → loop; valve_junction → loop OR first_unit OR waste.
            # Default: run position (loop out of line).
            self.add_connection(self["feed_inlet"], sample_loop)
            self.add_connection(valve_junction, sample_loop)
            self.add_connection(valve_junction, first_unit)
            self.add_connection(valve_junction, waste)
            self.set_output_state(valve_junction, {first_unit.name: 1})

            self.add_connection(sample_loop, first_unit)
            self.add_connection(sample_loop, waste)
            self.set_output_state(sample_loop, {first_unit.name: 1})
        else:
            # feed_inlet and valve_junction both go directly to first_unit.
            # Default: run position.
            self.add_connection(self["feed_inlet"], first_unit)
            self.add_connection(valve_junction, first_unit)
            self.add_connection(valve_junction, waste)
            self.set_output_state(valve_junction, {first_unit.name: 1})

        # Post-injection connections (sequential through non-bypassed units).
        connection_order = [
            "tubing_pre_column",
            "column",
            "tubing_post_column",
            "tubing_detectors",
            "outlet",
        ]
        origin = None
        for destination in connection_order:
            if origin is None:
                if destination == first_unit.name:
                    origin = first_unit.name
                continue
            if destination in self:
                self.add_connection(self[origin], self[destination])
                origin = destination

    @property
    def has_sample_loop(self) -> bool:
        """bool: Whether a sample loop is present in the flow path."""
        return self._has_sample_loop

    @property
    def first_unit_name(self) -> str:
        """str: Name of the first post-injection unit in the flow path."""
        return self._first_unit_name

    @property
    def valve_junction_name(self) -> str:
        """str: Name of the unit that controls injection valve routing.

        ``"tubing_pre_injection"`` when present; ``"mixer"`` when
        ``tubing_pre_injection`` is bypassed.
        """
        return self._valve_junction_name

    def get_system_dead_volume(
        self,
        exclude: list[str] | None = None,
        ignore_missing: bool = False,
    ) -> float:
        """Return total liquid volume of system units, excluding the given names."""
        if exclude is None:
            exclude = []
        if not isinstance(exclude, list):
            exclude = [exclude]

        def _get_unit_volume(name: str) -> float:
            try:
                vol = self.units_dict[name].volume_liquid
            except KeyError:
                if not ignore_missing:
                    raise KeyError(f"Cannot find unit: {name}.")
                return 0
            except TypeError:
                return 0
            return vol if vol is not None else 0

        system_units = [
            "mixer",
            "tubing_pre_injection",
            "column",
            "tubing_post_column",
            "tubing_detectors",
        ]
        return sum(
            _get_unit_volume(u) for u in system_units if u not in exclude
        )

    @property
    def system_dead_volume(self) -> float:
        """float: Total liquid volume of all system units."""
        return self.get_system_dead_volume(ignore_missing=True)


class LCProcess(Process):
    """
    Base process class for experiments run on an LC system.

    Construct an :class:`LCFlowSheet` first, then pass it here — the same
    pattern used for plain :class:`~CADETProcess.processModel.Process` objects
    in CADET-Process.

    Parameters
    ----------
    name : str
        Unique name for the process.
    flow_sheet : LCFlowSheet
        Configured LC flow sheet topology.
    """

    def __init__(
        self,
        name: str,
        flow_sheet: LCFlowSheet,
    ) -> None:
        super().__init__(flow_sheet, name)

    def _zero_buffer_flow_rates(self) -> None:
        """Set all buffer inlet flow rates to zero."""
        for name in _BUFFER_KEYS.values():
            if name in self.flow_sheet:
                self.flow_sheet[name].flow_rate = 0

    def add_valve_event(
        self,
        position: ValvePosition,
        t: float,
    ) -> None:
        """
        Schedule a valve position change at time ``t``.

        A valve position change affects multiple units simultaneously (e.g. both
        ``tubing_pre_injection`` and ``sample_loop`` switch when the 6-port valve
        turns). This method generates one ``output_state`` event per affected
        unit. The first event is the primary; all others are linked to it via
        ``add_event_dependency``, so moving the primary event's time (e.g. during
        optimisation) automatically propagates to all coupled events.

        Parameters
        ----------
        position : ValvePosition
            ``"run"``
                Normal column operation: SyP → ``first_unit``, SaP → waste.
                Loop is out of the active flow path. Available with or without
                loop.
            ``"load"``
                Loop loading: SyP → ``first_unit`` (column continues running);
                SaP → loop → waste (sample pump fills the loop). Requires a
                loop; degrades to ``"run"`` if none is present.
            ``"inject"``
                Loop injection: SyP → loop → ``first_unit``; SaP → waste.
                Requires a loop; degrades to ``"run"`` if none is present.
                Alias: ``"sample_pump_waste"`` (same flow path, emphasises
                that the sample pump output is routed to waste).
            ``"direct_inject"``
                Direct sample injection without loop: SyP → waste; SaP →
                ``first_unit``. Available with or without loop.
                Alias: ``"system_pump_waste"`` (same flow path, emphasises
                that the system pump output is routed to waste).
        t : float
            Time in seconds at which the position change takes effect.

        Examples
        --------
        Single injection at t=0, then switch back to run after the loop
        is pushed through:

        >>> proc.add_valve_event("inject", t=0.0)   # doctest: +SKIP
        >>> proc.add_valve_event("run",    t=30.0)  # doctest: +SKIP

        Successive injections — reload the loop during the column run:

        >>> proc.add_valve_event("inject", t=0.0)    # doctest: +SKIP
        >>> proc.add_valve_event("load",   t=30.0)   # doctest: +SKIP
        >>> proc.add_valve_event("inject", t=700.0)  # doctest: +SKIP
        >>> proc.add_valve_event("load",   t=730.0)  # doctest: +SKIP
        """
        first = self.flow_sheet.first_unit_name
        vjn = self.flow_sheet.valve_junction_name
        if self.flow_sheet.has_sample_loop:
            _run = {vjn: {first: 1.0}}
            _load = {vjn: {first: 1.0}, "sample_loop": {"waste": 1.0}}
            _inject = {vjn: {"sample_loop": 1.0}, "sample_loop": {first: 1.0}}
            _direct_inject = {vjn: {"waste": 1.0}, "sample_loop": {first: 1.0}}
            all_states = {
                "run":               _run,
                "load":              _load,
                "inject":            _inject,
                "sample_pump_waste": _inject,        # alias: same flow path
                "direct_inject":     _direct_inject,
                "system_pump_waste": _direct_inject,  # alias: same flow path
            }
        else:
            # No loop: load/inject degrade to run; direct_inject available natively.
            _run = {vjn: {first: 1.0}}
            _direct_inject = {vjn: {"waste": 1.0}}
            all_states = {
                "run":               _run,
                "load":              _run,           # degrades to run (no loop to fill)
                "inject":            _run,           # degrades to run (no loop to push)
                "sample_pump_waste": _run,           # degrades to run
                "direct_inject":     _direct_inject,
                "system_pump_waste": _direct_inject,  # alias
            }
        if position not in all_states:
            raise ValueError(
                f"Unknown valve position {position!r}. "
                f"Valid positions: {list(all_states)}"
            )
        output_states = all_states[position]
        existing = {e.name for e in self.events}
        primary_name = None
        for unit_name, state in output_states.items():
            base = f"valve_{unit_name}_{position}"
            name = base
            i = 1
            while name in existing:
                name = f"{base}_{i}"
                i += 1
            self.add_event(name, f"flow_sheet.output_states.{unit_name}", state, t)
            existing.add(name)
            if primary_name is None:
                primary_name = name
            else:
                self.add_event_dependency(name, primary_name)

    def generate_synthetic_data(
        self,
        solution_path: str = "outlet.outlet",
        components: list[str] | None = None,
    ) -> ReferenceIO:
        """Simulate and return the outlet solution as a ReferenceIO."""
        simulator = Cadet()
        simulation_results = simulator.simulate(self)
        solution = get_nested_value(simulation_results.solution, solution_path)
        solution = slice_solution(solution, components=components)
        return solution


class PhasedProcess(LCProcess):
    """
    LC process defined by an explicit sequence of phases.

    Each phase specifies a duration, flow rate, and buffer composition
    (fractions of A–D or F). Compositions are translated to per-buffer
    flow rate events with linear slopes for gradients or constant values
    for steps.

    Parameters
    ----------
    name : str
        Unique name for the process.
    flow_sheet : LCFlowSheet
        Configured LC flow sheet topology.
    steps : list[Phase | ValveEvent]
        Ordered sequence of phases and instantaneous valve events.
        ``cycle_time`` is set to the sum of all phase durations.
        :class:`ValveEvent` items fire at the cumulative time of all
        preceding phases. The default valve state before the first step
        is ``"run"``.

    Examples
    --------
    A two-phase step process — wash with buffer A, then step to buffer B:

    >>> from CADETProcess.processModel import ComponentSystem
    >>> from CADETProcess.instruments import LCFlowSheet, Phase, PhasedProcess
    >>> cs = ComponentSystem(["Salt"])
    >>> fs = LCFlowSheet(
    ...     cs, sample_loop_volume=50e-9, sample_loop_diameter=0.75e-3,
    ...     bypass_units=["tubing_pre_column", "column",
    ...                   "tubing_post_column", "tubing_detectors"],
    ... )
    >>> Q = 8.3e-9
    >>> proc = PhasedProcess("step", fs, steps=[
    ...     Phase(200.0, Q, {"A": 1.0}),
    ...     Phase(400.0, Q, {"B": 1.0}),
    ... ])
    >>> proc.cycle_time
    600.0
    """

    def __init__(
        self,
        name: str,
        flow_sheet: LCFlowSheet,
        steps: list[Phase | ValveEvent],
    ) -> None:
        super().__init__(name, flow_sheet)
        self._zero_buffer_flow_rates()
        self._build_events_from_steps(steps)

    def _build_events_from_steps(self, steps: list[Phase | ValveEvent]) -> None:
        # Validate: two consecutive ValveEvents fire at the same time, which
        # CADET-Process rejects for the same parameter path.
        for i in range(len(steps) - 1):
            if isinstance(steps[i], ValveEvent) and isinstance(steps[i + 1], ValveEvent):
                raise ValueError(
                    f"Two consecutive ValveEvents at positions {i} and {i + 1} "
                    f"would fire at the same time "
                    f"({steps[i].position!r} and {steps[i + 1].position!r})."
                )

        # Collect buffer keys across all phases so that every phase emits an
        # event for every key active anywhere in the protocol.  This ensures a
        # buffer active in phase i is explicitly zeroed in phase i+1 when absent,
        # preventing residual values from bleeding through.
        all_keys: set[str] = set()
        for step in steps:
            if isinstance(step, Phase):
                all_keys.update(step.composition_start)
                all_keys.update(step.composition_end)

        t = 0.0
        phase_index = 0
        for step in steps:
            if isinstance(step, ValveEvent):
                self.add_valve_event(step.position, t)
            else:
                for key in all_keys:
                    buf_name = _BUFFER_KEYS[key]
                    path = f"flow_sheet.{buf_name}.flow_rate"
                    v_start = step.flow_rate * step.composition_start.get(key, 0.0)
                    v_end = step.flow_rate * step.composition_end.get(key, 0.0)
                    if not math.isclose(v_start, v_end):
                        slope = (v_end - v_start) / step.duration
                        value = [v_start, slope]
                    else:
                        value = v_start
                    self.add_event(f"phase_{phase_index}_{key}", path, value, t)
                t += step.duration
                phase_index += 1

        self.cycle_time = t


class Breakthrough(PhasedProcess):
    """
    Breakthrough experiment on an LC system.

    Sample flows continuously from t=0 through the column at constant flow
    rate and composition. Pre-equilibration is assumed to be captured in
    the initial conditions.

    Parameters
    ----------
    name : str
        Unique name for the process.
    flow_sheet : LCFlowSheet
        Configured LC flow sheet topology.
    c_sample : list[float]
        Concentration of the continuously loaded sample.
    flow_rate : float
        Volumetric flow rate in m³/s.
    cycle_time : float
        Experiment duration in seconds.
    sample_buffer : str, optional
        Buffer key carrying the sample. Default ``"F"`` (feed inlet).
        Any key ``"A"``–``"D"`` or ``"F"`` is valid.
    delta_t_equilibration : float, optional
        Equilibration phase duration in seconds, run at ``flow_rate`` with
        buffer A before the breakthrough phase. Default 0 omits this phase.
        Added to ``cycle_time``.
    """

    def __init__(
        self,
        name: str,
        flow_sheet: LCFlowSheet,
        c_sample: list[float],
        flow_rate: float,
        cycle_time: float,
        sample_buffer: str = "F",
        delta_t_equilibration: float = 0.0,
    ) -> None:
        phases = []
        if delta_t_equilibration > 0:
            phases.append(Phase(delta_t_equilibration, flow_rate, {"A": 1.0}))
        phases.append(Phase(cycle_time, flow_rate, {sample_buffer: 1.0}))
        super().__init__(name, flow_sheet, phases)
        buf_name = _BUFFER_KEYS[sample_buffer]
        getattr(self.flow_sheet, buf_name).c = c_sample


class StepElution(PhasedProcess):
    """
    Load/Wash/Elute process with a step to high salt instead of a gradient.

    The sample loop is injected at t=0, then three phases follow: wash
    (100 % buffer A), step to elute (100 % buffer B, constant), final wash
    (100 % buffer A).

    Parameters
    ----------
    name : str
        Unique name for the process.
    flow_sheet : LCFlowSheet
        Configured LC flow sheet topology. Must include a sample loop.
    c_buffer_a : list[float]
        Running / equilibration buffer concentration.
    c_buffer_b : list[float]
        Elution buffer concentration.
    c_sample : list[float]
        Sample loop concentration.
    delta_t_wash : float
        Wash phase duration in seconds.
    delta_t_elute : float
        Elution phase duration in seconds.
    delta_t_final_wash : float
        Final wash phase duration in seconds.
    flow_rate_wash : float
        Flow rate during wash in m³/s.
    flow_rate_elute : float, optional
        Flow rate during elution. Defaults to ``flow_rate_wash``.
    flow_rate_final_wash : float, optional
        Flow rate during final wash. Defaults to ``flow_rate_elute``.
    delta_t_equilibration : float, optional
        Equilibration phase duration in seconds, run at ``flow_rate_wash``
        with buffer A before the sample loop is injected. Default 0 omits
        this phase. Added to ``cycle_time``.
    """

    def __init__(
        self,
        name: str,
        flow_sheet: LCFlowSheet,
        c_buffer_a: list[float],
        c_buffer_b: list[float],
        c_sample: list[float],
        delta_t_wash: float,
        delta_t_elute: float,
        delta_t_final_wash: float,
        flow_rate_wash: float,
        flow_rate_elute: float | None = None,
        flow_rate_final_wash: float | None = None,
        delta_t_equilibration: float = 0.0,
    ) -> None:
        if not flow_sheet.has_sample_loop:
            raise ValueError(
                "StepElution requires a sample loop. "
                "Pass sample_loop_volume to LCFlowSheet."
            )
        if flow_rate_elute is None:
            flow_rate_elute = flow_rate_wash
        if flow_rate_final_wash is None:
            flow_rate_final_wash = flow_rate_elute

        steps = []
        if delta_t_equilibration > 0:
            steps.append(Phase(delta_t_equilibration, flow_rate_wash, {"A": 1.0}))
        steps += [
            ValveEvent("inject"),
            Phase(delta_t_wash, flow_rate_wash, {"A": 1.0}),
            Phase(delta_t_elute, flow_rate_elute, {"B": 1.0}),
            Phase(delta_t_final_wash, flow_rate_final_wash, {"A": 1.0}),
        ]
        super().__init__(name, flow_sheet, steps)

        for unit in self.flow_sheet.units:
            if "c" in unit.parameters:
                unit.c = c_buffer_a
            if "cp" in unit.parameters:
                unit.cp = c_buffer_a

        self.flow_sheet.buffer_b.c = c_buffer_b
        self.flow_sheet.sample_loop.c = c_sample


class PulseInjection(PhasedProcess):
    """
    Pulse injection process on an LC system.

    The system is pre-equilibrated with buffer A at constant flow rate.
    At t=0 the sample loop is in the flow path (inject position).

    Parameters
    ----------
    name : str
        Unique name for the process.
    flow_sheet : LCFlowSheet
        Configured LC flow sheet topology. Must include a sample loop.
    c_buffer_a : list[float]
        Equilibration buffer concentration.
    c_sample : list[float]
        Sample loop concentration.
    cycle_time : float
        Experiment duration in seconds.
    flow_rate : float
        Volumetric flow rate in m³/s.
    delta_t_equilibration : float, optional
        Equilibration phase duration in seconds, run at ``flow_rate`` with
        buffer A before the sample loop is injected. Default 0 omits this
        phase. Added to ``cycle_time``.
    """

    def __init__(
        self,
        name: str,
        flow_sheet: LCFlowSheet,
        c_buffer_a: list[float],
        c_sample: list[float],
        cycle_time: float,
        flow_rate: float,
        delta_t_equilibration: float = 0.0,
    ) -> None:
        if not flow_sheet.has_sample_loop:
            raise ValueError(
                "PulseInjection requires a sample loop. "
                "Pass sample_loop_volume to LCFlowSheet."
            )
        steps = []
        if delta_t_equilibration > 0:
            steps.append(Phase(delta_t_equilibration, flow_rate, {"A": 1.0}))
        steps += [ValveEvent("inject"), Phase(cycle_time, flow_rate, {"A": 1.0})]
        super().__init__(name, flow_sheet, steps)

        for unit in self.flow_sheet.units:
            if "c" in unit.parameters:
                unit.c = c_buffer_a
            if "cp" in unit.parameters:
                unit.cp = c_buffer_a

        self.flow_sheet.sample_loop.c = c_sample


class Step(PhasedProcess):
    """
    Step-change process on an LC system.

    The system is pre-equilibrated with buffer A. At t=0 it switches to buffer B.

    Parameters
    ----------
    name : str
        Unique name for the process.
    flow_sheet : LCFlowSheet
        Configured LC flow sheet topology.
    c_buffer_a : list[float]
        Equilibration buffer concentration.
    c_buffer_b : list[float]
        Step buffer concentration.
    cycle_time : float
        Experiment duration in seconds.
    flow_rate : float
        Volumetric flow rate in m³/s.
    delta_t_equilibration : float, optional
        Equilibration phase duration in seconds, run at ``flow_rate`` with
        buffer A before the step to buffer B. Default 0 omits this phase.
        Added to ``cycle_time``.
    """

    def __init__(
        self,
        name: str,
        flow_sheet: LCFlowSheet,
        c_buffer_a: list[float],
        c_buffer_b: list[float],
        cycle_time: float,
        flow_rate: float,
        delta_t_equilibration: float = 0.0,
    ) -> None:
        phases = []
        if delta_t_equilibration > 0:
            phases.append(Phase(delta_t_equilibration, flow_rate, {"A": 1.0}))
        phases.append(Phase(cycle_time, flow_rate, {"B": 1.0}))
        super().__init__(name, flow_sheet, phases)

        for unit in self.flow_sheet.units:
            if "c" in unit.parameters:
                unit.c = c_buffer_a

        self.flow_sheet.buffer_b.c = c_buffer_b


class LWE(PhasedProcess):
    """
    Load/Wash/Elute process on an LC system.

    The system is pre-equilibrated with buffer A. The sample loop is injected
    at t=0, then the wash phase runs; after it, a linear salt gradient ramps
    buffer A down to zero while buffer B ramps up. A final wash phase holds
    at 100 % buffer B.

    Parameters
    ----------
    name : str
        Unique name for the process.
    flow_sheet : LCFlowSheet
        Configured LC flow sheet topology. Must include a sample loop.
    c_buffer_a : list[float]
        Equilibration / running buffer concentration.
    c_buffer_b : list[float]
        Elution buffer concentration.
    c_sample : list[float]
        Sample loop concentration.
    delta_t_wash : float
        Wash phase duration in seconds.
    delta_t_elute : float
        Gradient elution phase duration in seconds.
    delta_t_final_wash : float
        Final wash phase duration in seconds.
    flow_rate_wash : float
        Flow rate during wash in m³/s.
    flow_rate_elute : float, optional
        Flow rate during elution. Defaults to ``flow_rate_wash``.
    flow_rate_final_wash : float, optional
        Flow rate during final wash. Defaults to ``flow_rate_elute``.
    delta_t_equilibration : float, optional
        Equilibration phase duration in seconds, run at ``flow_rate_wash``
        with buffer A before the sample loop is injected. Default 0 omits
        this phase. Added to ``cycle_time``.
    """

    def __init__(
        self,
        name: str,
        flow_sheet: LCFlowSheet,
        c_buffer_a: list[float],
        c_buffer_b: list[float],
        c_sample: list[float],
        delta_t_wash: float,
        delta_t_elute: float,
        delta_t_final_wash: float,
        flow_rate_wash: float,
        flow_rate_elute: float | None = None,
        flow_rate_final_wash: float | None = None,
        delta_t_equilibration: float = 0.0,
    ) -> None:
        if not flow_sheet.has_sample_loop:
            raise ValueError(
                "LWE requires a sample loop. "
                "Pass sample_loop_volume to LCFlowSheet."
            )
        if flow_rate_elute is None:
            flow_rate_elute = flow_rate_wash
        if flow_rate_final_wash is None:
            flow_rate_final_wash = flow_rate_elute

        steps = []
        if delta_t_equilibration > 0:
            steps.append(Phase(delta_t_equilibration, flow_rate_wash, {"A": 1.0}))
        steps += [
            ValveEvent("inject"),
            Phase(delta_t_wash, flow_rate_wash, {"A": 1.0}),
            Phase(delta_t_elute, flow_rate_elute, {"A": 1.0}, {"B": 1.0}),
            Phase(delta_t_final_wash, flow_rate_final_wash, {"B": 1.0}),
        ]
        super().__init__(name, flow_sheet, steps)

        for unit in self.flow_sheet.units:
            if "c" in unit.parameters:
                unit.c = c_buffer_a
            if "cp" in unit.parameters:
                unit.cp = c_buffer_a

        self.flow_sheet.buffer_b.c = c_buffer_b
        self.flow_sheet.sample_loop.c = c_sample
