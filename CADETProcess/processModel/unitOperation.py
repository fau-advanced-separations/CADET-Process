import math
import warnings
from abc import abstractmethod
from typing import Any, Optional

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import (
    Constant,
    NdPolynomial,
    Polynomial,
    SizedFloatList,
    SizedNdArray,
    SizedUnsignedList,
    String,
    Structure,
    Switch,
    UnsignedFloat,
    frozen_attributes,
)

from .binding import BindingBaseClass, NoBinding
from .componentSystem import ComponentSystem
from .discretization import (
    DiscretizationParametersBase,
    GRMDiscretizationDG,
    GRMDiscretizationFV,
    LRMDiscretizationDG,
    LRMDiscretizationFV,
    LRMPDiscretizationDG,
    LRMPDiscretizationFV,
    MCTDiscretizationFV,
    NoDiscretization,
)
from .reaction import BulkReactionBase, NoReaction, ParticleReactionBase
from .solutionRecorder import (
    CSTRRecorder,
    GRMRecorder,
    IORecorder,
    LRMPRecorder,
    LRMRecorder,
    MCTRecorder,
    TubularReactorRecorder,
)

__all__ = [
    "UnitBaseClass",
    "SourceMixin",
    "SinkMixin",
    "Inlet",
    "Outlet",
    "Cstr",
    "TubularReactorBase",
    "TubularReactor",
    "ChromatographicColumnBase",
    "LumpedRateModelWithoutPores",
    "LumpedRateModelWithPores",
    "GeneralRateModel",
    "MCT",
]


@frozen_attributes
class UnitBaseClass(Structure):
    """
    Base class for all UnitOperation classes.

    A UnitOperation object stores model parameters and states of a unit.
    Every unit operation can be assotiated with a binding behavior and a
    reaction model.
    UnitOperations can be connected in a FlowSheet.

    Attributes
    ----------
    n_comp : UnsignedInteger
        Number of components in a system.
    parameters : list
        list of parameter names.
    name : String
        name of the unit operation.
    has_ports : bool
        flag if unit has ports. The default is False.
    binding_model : BindingBaseClass
        binding behavior of the unit. Defaults to NoBinding.
    solution_recorder : IORecorder
        Solution recorder for the unit operation.

    See Also
    --------
    CADETProcess.processModel.binding
    CADETProcess.processModel.reaction
    CADETProcess.processModel.FlowSheet
    """

    name = String()

    _parameters = []
    _section_dependent_parameters = []
    _initial_state = []

    has_ports = False
    supports_binding = False
    supports_bulk_reaction = False
    supports_particle_reaction = False
    discretization_schemes = ()

    def __init__(
        self,
        component_system: ComponentSystem,
        name: str,
        *args: Optional[tuple],
        binding_model: Optional[BindingBaseClass] = None,
        bulk_reaction_model: Optional[BulkReactionBase] = None,
        particle_reaction_model: Optional[ParticleReactionBase] = None,
        discretization: Optional[DiscretizationParametersBase] = None,
        solution_recorder: Optional[IORecorder] = None,
        **kwargs: Optional[dict],
    ) -> None:
        """
        Initialize UnitBaseClass.

        Parameters
        ----------
        component_system : ComponentSystem
            The component system to which this unit belongs.
        name : str
            Name identifier for the unit instance.
        *args : tuple
            Positional arguments passed to the superclass constructor.
        binding_model : BindingBaseClass, optional
            A model describing specific binding interactions within the unit.
            If None, a default `NoBinding` model is used.
        bulk_reaction_model : BulkReactionBase, optional
            A model for bulk phase reactions. Defaults to `NoReaction` if not provided.
        particle_reaction_model : ParticleReactionBase, optional
            A model for reactions within particles. Defaults to `NoReaction`.
        discretization : DiscretizationParametersBase, optional
            Parameters defining spatial/temporal discretization of the unit.
            Defaults to `NoDiscretization` if None.
        solution_recorder : IORecorder, optional
            Object responsible for recording the simulation results.
            Defaults to a new `IORecorder` instance.
        **kwargs : dict
            Additional keyword arguments passed to the superclass constructor.
        """
        self.name = name
        self.component_system = component_system

        if binding_model is None:
            binding_model = NoBinding()
        self.binding_model = binding_model

        if bulk_reaction_model is None:
            bulk_reaction_model = NoReaction()
        self.bulk_reaction_model = bulk_reaction_model

        if particle_reaction_model is None:
            particle_reaction_model = NoReaction()
        self.particle_reaction_model = particle_reaction_model

        if discretization is None:
            discretization = NoDiscretization()
        self.discretization = discretization

        if solution_recorder is None:
            solution_recorder = IORecorder()
        self.solution_recorder = solution_recorder

        super().__init__(*args, **kwargs)

    @property
    def model(self) -> str:
        """str: Return model name."""
        return self.__class__.__name__

    @property
    def component_system(self) -> ComponentSystem:
        """ComponentSystem: Component of the unit operation."""
        return self._component_system

    @component_system.setter
    def component_system(self, component_system: ComponentSystem) -> None:
        if not isinstance(component_system, ComponentSystem):
            raise TypeError("Expected ComponentSystem")
        self._component_system = component_system

    @property
    def discretization(self) -> DiscretizationParametersBase:
        """DiscretizationParametersBase: Discretization parameters of the unit operation."""
        return self._discretization

    @discretization.setter
    def discretization(self, discretization: DiscretizationParametersBase) -> None:
        if not isinstance(discretization, DiscretizationParametersBase):
            raise TypeError("Expected DiscretizationParametersBase")

        if not isinstance(discretization, NoDiscretization):
            if not isinstance(discretization, self.discretization_schemes):
                raise CADETProcessError(
                    f"Unit does not support {type(discretization)}."
                )

        self._discretization = discretization

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self.component_system.n_comp

    @property
    def ports(self) -> list:
        """list: Ports of the unit operation."""
        return [None]

    @property
    def n_ports(self) -> int:
        """int: Number of ports."""
        return 1

    @property
    def parameters(self) -> dict:
        """dict: Dictionary with parameter values."""
        parameters = super().parameters

        if not isinstance(self.binding_model, NoBinding):
            parameters["binding_model"] = self.binding_model.parameters
        if not isinstance(self.bulk_reaction_model, NoReaction):
            parameters["bulk_reaction_model"] = self.bulk_reaction_model.parameters
        if not isinstance(self.particle_reaction_model, NoReaction):
            parameters['particle_reaction_model'] = \
                self.particle_reaction_model.parameters
        if not isinstance(self.discretization, NoDiscretization):
            parameters["discretization"] = self.discretization.parameters

        return parameters

    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        try:
            self.binding_model.parameters = parameters.pop("binding_model")
        except KeyError:
            pass
        try:
            self.bulk_reaction_model.parameters = parameters.pop("bulk_reaction_model")
        except KeyError:
            pass
        try:
            self.particle_reaction_model.parameters = parameters.pop(
                "particle_reaction_model"
            )
        except KeyError:
            pass
        try:
            self.discretization.parameters = parameters.pop("discretization")
        except KeyError:
            pass

        super(UnitBaseClass, self.__class__).parameters.fset(self, parameters)

    @property
    def section_dependent_parameters(self) -> dict:
        """Return section dependent parameters."""
        parameters = {
            key: value
            for key, value in self.parameters.items()
            if key in self._section_dependent_parameters
        }
        return parameters

    @property
    def initial_state(self) -> dict:
        """dict: Dictionary with initial states."""
        initial_state = {st: getattr(self, st) for st in self._initial_state}

        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state: dict) -> None:
        for st, value in initial_state.items():
            if st not in self._initial_state:
                raise CADETProcessError("Not a valid parameter")
            if value is not None:
                setattr(self, st, value)

    @property
    def missing_parameters(self) -> list:
        """Return list of missing parameters."""
        missing_parameters = super().missing_parameters

        missing_parameters += [
            f"binding_model.{param}" for param in self.binding_model.missing_parameters
        ]

        return missing_parameters

    def check_required_parameters(self) -> bool:
        """Checkf if there are missing parameters left."""
        if len(self.missing_parameters) == 0:
            return True
        else:
            for param in self.missing_parameters:
                warnings.warn(f'Unit {self.name}: Missing parameter "{param}".')
            return False

    @property
    def binding_model(self) -> BindingBaseClass:
        """
        BindingBaseClass: binding model parameters of the unit operation.

        Raises
        ------
        TypeError
            If binding_model object is not an instance of BindingBaseClass.
        CADETProcessError
            If number of components do not match.

        """
        return self._binding_model

    @binding_model.setter
    def binding_model(self, binding_model: BindingBaseClass) -> None:
        if not isinstance(binding_model, BindingBaseClass):
            raise TypeError("Expected BindingBaseClass")

        if not isinstance(binding_model, NoBinding):
            if not self.supports_binding:
                raise CADETProcessError("Unit does not support binding models.")

            if (
                binding_model.component_system is not self.component_system
                and not isinstance(binding_model, NoBinding)
            ):
                raise CADETProcessError("Component systems do not match.")

        self._binding_model = binding_model

    @property
    def n_bound_states(self) -> int:
        """int: Return number of bound states."""
        if isinstance(self.binding_model, NoBinding):
            return 0
        return self.binding_model.n_bound_states

    @property
    def bulk_reaction_model(self) -> BulkReactionBase:
        """
        BulkReactionBase: Reaction in bulk phase.

        Raises
        ------
        TypeError
            If bulk_reaction_model is not an instance of ReactionBaseClass.
        CADETProcessError
            If unit does not support bulk reaction model.
            If number of components do not match.

        """
        return self._bulk_reaction_model

    @bulk_reaction_model.setter
    def bulk_reaction_model(self, bulk_reaction_model: BulkReactionBase) -> None:
        if not isinstance(bulk_reaction_model, NoReaction):
            if not isinstance(bulk_reaction_model, BulkReactionBase):
                raise TypeError("Expected BulkReactionBase")

            if not self.supports_bulk_reaction:
                raise CADETProcessError("Unit does not support bulk reactions.")
            if bulk_reaction_model.component_system is not self.component_system:
                raise CADETProcessError("Component systems do not match.")

        self._bulk_reaction_model = bulk_reaction_model

    @property
    def particle_reaction_model(self) -> ParticleReactionBase:
        """
        ParticleReactionBase: Reaction in particle liquid phase.

        Raises
        ------
        TypeError
            If particle_reaction_model is not an instance of ReactionBaseClass.
        CADETProcessError
            If unit does not support particle reaction model.
            If number of components do not match.

        """
        return self._particle_reaction_model

    @particle_reaction_model.setter
    def particle_reaction_model(
        self, particle_reaction_model: ParticleReactionBase | BulkReactionBase
    ) -> None:
        if isinstance(particle_reaction_model, BulkReactionBase):
            try:
                warnings.warn(
                    "Detected Bulk Reaction Model. "
                    "Attempt casting to Particle Reaction Model."
                )
                particle_reaction_model = particle_reaction_model.to_particle_model()
            except NotImplementedError:
                pass

        if not isinstance(particle_reaction_model, NoReaction):
            if not isinstance(particle_reaction_model, ParticleReactionBase):
                raise TypeError("Expected ReactionBaseClass")

            if not self.supports_particle_reaction:
                raise CADETProcessError("Unit does not support particle reactions.")
            if particle_reaction_model.component_system is not self.component_system:
                raise CADETProcessError("Component systems do not match.")

        self._particle_reaction_model = particle_reaction_model

    def __repr__(self) -> str:
        """str: String-representation of the object."""
        return f"{self.__class__.__name__}(n_comp={self.n_comp}, name={self.name})"

    def __str__(self) -> str:
        """str: String-representation of the object."""
        return self.name


class SourceMixin(Structure):
    """
    Mixin class for Units that have Source-like behavior.

    See Also
    --------
    SinkMixin
    Inlet
    Cstr
    """

    _n_poly_coeffs = 4
    flow_rate = Polynomial(size=("_n_poly_coeffs"))
    _parameters = ["flow_rate"]
    _section_dependent_parameters = ["flow_rate"]


class SinkMixin:
    """
    Mixin class for Units that have Sink-like behavior.

    See Also
    --------
    SourceMixin
    Cstr
    """

    pass


class Inlet(UnitBaseClass, SourceMixin):
    """
    Pseudo unit operation model for streams entering the system.

    Attributes
    ----------
    c : NdPolynomial
        Polynomial coefficients for component concentration.
    flow_rate : NdPolynomial
        Polynomial coefficients for volumetric flow rate.
    solution_recorder : IORecorder
        Solution recorder for the unit operation.
    """

    c = NdPolynomial(size=("n_comp", "_n_poly_coeffs"), default=0)
    flow_rate = Polynomial(size=("_n_poly_coeffs"), default=0)
    _n_poly_coeffs = 4
    _parameters = ["c"]
    _section_dependent_parameters = (
        UnitBaseClass._section_dependent_parameters
        + SourceMixin._section_dependent_parameters
        + ["c"]
    )


class Outlet(UnitBaseClass, SinkMixin):
    """
    Pseudo unit operation model for streams leaving the system.

    Attributes
    ----------
    solution_recorder : IORecorder
        Solution recorder for the unit operation.
    """

    pass


class MixerSplitter(UnitBaseClass):
    """Pseudo unit operation for mixing/splitting streams in the system."""

    pass


class TubularReactorBase(UnitBaseClass):
    """
    Base class for tubular reactors and chromatographic columns.

    Provides methods for calculating geometric properties such as the cross
    section area and volume, as well as methods for convective and dispersive
    properties like mean residence time or NTP.

    Notes
    -----
        For subclassing, check that the total porosity and interstitial cross
        section area are computed correctly depending on the model porosities!

    Attributes
    ----------
    length : UnsignedFloat
        Length of column.
    diameter : UnsignedFloat
        Diameter of column.
    axial_dispersion : List of unsigned floats. Length depends on n_comp.
        Axial dispersion coefficient for each component.
    flow_direction : Switch
        If 1: Forward flow.
        If -1: Backwards flow.
    discretization : DiscretizationParametersBase
        Discretization scheme of the unit.
    """

    length = UnsignedFloat()
    diameter = UnsignedFloat()
    axial_dispersion = SizedUnsignedList(size="n_comp")
    flow_direction = Switch(valid=[-1, 1], default=1)
    c = SizedFloatList(size="n_comp", default=0)

    _initial_state = UnitBaseClass._initial_state + ["c"]
    _parameters = [
        "length",
        "diameter",
        "axial_dispersion",
        "flow_direction",
    ] + _initial_state
    _section_dependent_parameters = UnitBaseClass._section_dependent_parameters + [
        "axial_dispersion",
        "flow_direction",
    ]

    @property
    @abstractmethod
    def total_porosity(self) -> None:
        """Float | None: Total porosity of the unit operation."""
        pass

    @property
    def cross_section_area(self) -> float:
        """float: Cross section area of a Column.

        See Also
        --------
        volume
        cross_section_area_interstitial

        """
        if self.diameter is not None:
            return math.pi / 4 * self.diameter**2

    @cross_section_area.setter
    def cross_section_area(self, cross_section_area: float) -> None:
        self.diameter = (4 * cross_section_area / math.pi) ** 0.5

    def set_diameter_from_interstitial_velocity(self, Q: float, u0: float) -> None:
        """
        Set diamter from flow rate and interstitial velocity.

        In literature, often only the interstitial velocity is given.
        This method, the diameter / cross section area can be inferred from
        the flow rate, velocity, and porosity.


        Parameters
        ----------
        Q : float
            Volumetric flow rate.
        u0 : float
            Interstitial velocity.

        Notes
        -----
            Needs to be overwritten depending on the model porosities!
        """
        self.cross_section_area = Q / (u0 * self.total_porosity)

    @property
    def cross_section_area_interstitial(self) -> float:
        """float: Interstitial area between particles.

        Notes
        -----
            Needs to be overwritten depending on the model porosities!

        See Also
        --------
        cross_section_area

        """
        return self.total_porosity * self.cross_section_area

    @property
    def volume(self) -> float:
        """
        float: Volume of the TubularReactor.

        See Also
        --------
        cross_section_area

        """
        return self.cross_section_area * self.length

    @property
    def volume_interstitial(self) -> float:
        """
        float: Interstitial volume between particles.

        See Also
        --------
        cross_section_area

        """
        return self.cross_section_area_interstitial * self.length

    @property
    def volume_liquid(self) -> float:
        """float: Volume of the liquid phase."""
        return self.total_porosity * self.cross_section_area * self.length

    @property
    def volume_solid(self) -> float:
        """float: Volume of the solid phase."""
        return (1 - self.total_porosity) * self.cross_section_area * self.length

    def calculate_interstitial_rt(self, flow_rate: float) -> float:
        """
        Calculate mean residence time of a (non adsorbing) volume element.

        Parameters
        ----------
        flow_rate : float
            Volumetric flow rate.

        Returns
        -------
        t0 : float
            Mean residence time of packed bed.

        See Also
        --------
        calculate_interstitial_velocity
        calculate_superficial_rt
        """
        return self.volume_interstitial / flow_rate

    def calculate_superficial_rt(self, flow_rate: float) -> float:
        """
        Calculate mean residence time of a volume element in an empty column.

        Parameters
        ----------
        flow_rate : float
            Volumetric flow rate.

        Returns
        -------
        t_s : float
            Mean residence time of empty column.

        See Also
        --------
        calculate_superficial_velocity
        calculate_interstitial_rt
        """
        return self.volume / flow_rate

    def calculate_interstitial_velocity(self, flow_rate: float) -> float:
        """
        Calculate flow velocity of a (non adsorbing) volume element.

        Parameters
        ----------
        flow_rate : float
            Volumetric flow rate.

        Returns
        -------
        interstitial_velocity : float
            Interstitial flow velocity.

        See Also
        --------
        calculate_interstitial_rt
        calculate_superficial_velocity
        """
        return self.length / self.calculate_interstitial_rt(flow_rate)

    def calculate_superficial_velocity(self, flow_rate: float) -> float:
        """
        Calculate superficial flow velocity of a volume element in an empty column.

        Parameters
        ----------
        flow_rate : float
            Volumetric flow rate.

        Returns
        -------
        u_s : float
            Superficial flow velocity.

        See Also
        --------
        calculate_superficial_rt
        calculate_interstitial_velocity
        NTP
        """
        return self.length / self.calculate_superficial_rt(flow_rate)

    def calculate_flow_rate_from_velocity(self, u0: float) -> float:
        """
        Calculate volumetric flow rate from interstitial velocity.

        Parameters
        ----------
        u0 : float
            Interstitial flow velocity.

        Returns
        -------
        Q : float
            Volumetric flow rate.

        See Also
        --------
        calculate_interstitial_velocity
        calculate_interstitial_rt
        """
        return u0 * self.cross_section_area_interstitial

    def NTP(self, flow_rate: float) -> float:
        r"""
        Calculate number of theoretical plates.

        Parameters
        ----------
        flow_rate : float
            volumetric flow rate

        Calculated using the axial dispersion coefficient:

        .. math::
            NTP = \frac{u \cdot L_{Column}}{2 \cdot D_a}

        Returns
        -------
        NTP : float
            Number of theretical plates
        """
        u0 = self.calculate_interstitial_velocity(flow_rate)
        return u0 * self.length / (2 * np.array(self.axial_dispersion))

    def set_axial_dispersion_from_NTP(self, NTP: float, flow_rate: float) -> float:
        r"""
        Set axial dispersion from number of theoretical plates (NTP).

        Parameters
        ----------
        NTP : float
            Number of theroetical plates
        flow_rate : float
            volumetric flow rate

        Calculated using the axial dispersion coefficient:

        .. math::
            NTP = \frac{u_{int} \cdot L_{Column}}{2 \cdot D_{ax}}

        Returns
        -------
        NTP : float
            Number of theretical plates

        See Also
        --------
        calculate_interstitial_velocity
        NTP
        """
        u0 = self.calculate_interstitial_velocity(flow_rate)
        self.axial_dispersion = u0 * self.length / (2 * NTP)

    def calculate_bodenstein_number(self, flow_rate: float) -> float:
        r"""
        Calculate the Bodenstein number for a given flow rate.

        Parameters
        ----------
        flow_rate : float
            volumetric flow rate

        The Bodenstein number is calculated as follows:

        .. math::
            Bo = \frac{u_{int} \cdot L_{Column}}{D_{ax}},

        where $u_int$ is the interstitial velocity


        Returns
        -------
        Bo : float
            Bodenstein number

        See Also
        --------
        calculate_interstitial_velocity
        NTP
        """
        u0 = self.calculate_interstitial_velocity(flow_rate)
        return u0 * self.length / np.array(self.axial_dispersion)


class TubularReactor(TubularReactorBase):
    """
    Class for tubular reactors and tubing.

    Class can be used for a regular tubular reactor.

    Attributes
    ----------
    c : List of unsigned floats. Length depends on n_comp
        Initial concentration of the reactor.
    solution_recorder : TubularReactorRecorder
        Solution recorder for the unit operation.
    """

    supports_bulk_reaction = True
    discretization_schemes = (LRMDiscretizationFV, LRMDiscretizationDG)

    total_porosity = Constant(1)

    def __init__(
        self, *args: Any, discretization_scheme: Optional[str] = "FV", **kwargs: Any
    ) -> None:
        if discretization_scheme == "FV":
            discretization = LRMDiscretizationFV()
        elif discretization_scheme == "DG":
            discretization = LRMDiscretizationDG()

        super().__init__(
            *args,
            discretization=discretization,
            solution_recorder=TubularReactorRecorder(),
            **kwargs,
        )


class ChromatographicColumnBase(TubularReactorBase):
    """Base class for chromatogrpahic columns."""

    pass


class LumpedRateModelWithoutPores(ChromatographicColumnBase):
    """
    Parameters for a lumped rate model without pores.

    Attributes
    ----------
    total_porosity : UnsignedFloat between 0 and 1.
        Total porosity of the column.
    c : List of unsigned floats. Length depends on n_comp
            Initial concentration of the reactor.
    q : List of unsigned floats. Length depends on n_comp
        Initial concentration of the bound phase.
    solution_recorder : LRMRecorder
        Solution recorder for the unit operation.

    Notes
    -----
    Although technically the LumpedRateModelWithoutPores does not have
    particles, the particle reactions interface is used to support
    reactions in the solid phase and cross-phase reactions.
    """

    supports_binding = True
    supports_bulk_reaction = False
    supports_particle_reaction = True
    discretization_schemes = (LRMDiscretizationFV, LRMDiscretizationDG)

    total_porosity = UnsignedFloat(ub=1)
    _parameters = ["total_porosity"]

    _q = SizedUnsignedList(size="n_bound_states", default=0)

    _initial_state = ChromatographicColumnBase._initial_state + ["q"]
    _parameters = _parameters + _initial_state

    def __init__(
        self, *args: Any, discretization_scheme: Optional[str] = "FV", **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        if discretization_scheme == "FV":
            self.discretization = LRMDiscretizationFV()
        elif discretization_scheme == "DG":
            self.discretization = LRMDiscretizationDG()

        self.solution_recorder = LRMRecorder()

    @property
    def q(self) -> list:
        """list[float]: Initial solid phase concentration."""
        return self._q

    @q.setter
    def q(self, q: list) -> None:
        if isinstance(self.binding_model, NoBinding) and q not in (None, []):
            raise CADETProcessError("Cannot set q without binding model.")
        self._q = q

        self.parameters["q"] = self._q


class LumpedRateModelWithPores(ChromatographicColumnBase):
    """
    Parameters for the lumped rate model with pores.

    Attributes
    ----------
    bed_porosity : UnsignedFloat between 0 and 1.
        Porosity of the bed
    particle_porosity : UnsignedFloat between 0 and 1.
        Porosity of particles.
    particle_radius : UnsignedFloat
        Radius of the particles.
    film_diffusion : List of unsigned floats. Length depends on n_comp.
        Film diffusion coefficients for each component.
    pore_accessibility : List of unsigned floats. Length depends on n_comp.
        Accessibility of pores for components.
    c : List of unsigned floats. Length depends on n_comp
        Initial concentration of the reactor.
    cp : List of unsigned floats. Length depends on n_comp
        Initial concentration of the pores
    q : List of unsigned floats. Length depends on n_comp
        Initial concntration of the bound phase.
    solution_recorder : LRMPRecorder
        Solution recorder for the unit operation.
    """

    supports_binding = True
    supports_bulk_reaction = True
    supports_particle_reaction = True
    discretization_schemes = (LRMPDiscretizationFV, LRMPDiscretizationDG)

    bed_porosity = UnsignedFloat(ub=1)
    particle_porosity = UnsignedFloat(ub=1)
    particle_radius = UnsignedFloat()
    film_diffusion = SizedUnsignedList(size="n_comp")
    pore_accessibility = SizedUnsignedList(ub=1, size="n_comp", default=1)
    _parameters = [
        "bed_porosity",
        "particle_porosity",
        "particle_radius",
        "film_diffusion",
        "pore_accessibility",
    ]

    _section_dependent_parameters = TubularReactorBase._section_dependent_parameters + [
        "film_diffusion",
        "pore_accessibility",
    ]

    _cp = SizedUnsignedList(size="n_comp")
    _q = SizedUnsignedList(size="n_bound_states", default=0)

    _initial_state = ChromatographicColumnBase._initial_state + ["cp", "q"]
    _parameters = _parameters + _initial_state

    def __init__(
        self, *args: Any, discretization_scheme: Optional[str] = "FV", **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        if discretization_scheme == "FV":
            self.discretization = LRMPDiscretizationFV()
        elif discretization_scheme == "DG":
            self.discretization = LRMPDiscretizationDG()

        self.solution_recorder = LRMPRecorder()

    @property
    def total_porosity(self) -> float:
        """float: Total porosity of the column."""
        return self.bed_porosity + (1 - self.bed_porosity) * self.particle_porosity

    @property
    def cross_section_area_interstitial(self) -> float:
        """float: Interstitial area between particles.

        See Also
        --------
        cross_section_area
        """
        return self.bed_porosity * self.cross_section_area

    def set_diameter_from_interstitial_velocity(self, Q: float, u0: float) -> None:
        """
        Set diamter from flow rate and interstitial velocity.

        In literature, often only the interstitial velocity is given.
        This method, the diameter / cross section area can be inferred from
        the flow rate, velocity, and bed porosity.


        Parameters
        ----------
        Q : float
            Volumetric flow rate.
        u0 : float
            Interstitial velocity.

        Notes
        -----
            Overwrites parent method.
        """
        self.cross_section_area = Q / (u0 * self.bed_porosity)

    @property
    def cp(self) -> list[float]:
        """list[float]: Initial particle liquid concentration."""
        if self._cp is None:
            return self.c
        else:
            return self._cp

    @cp.setter
    def cp(self, cp: list[float]) -> None:
        self._cp = cp

        self.parameters["cp"] = cp

    @property
    def q(self) -> list[float]:
        """list[float]: Initial solid phase concentration."""
        return self._q

    @q.setter
    def q(self, q: list) -> None:
        if isinstance(self.binding_model, NoBinding) and q not in (None, []):
            raise CADETProcessError("Cannot set q without binding model.")
        self._q = q

        self.parameters["q"] = q


class GeneralRateModel(ChromatographicColumnBase):
    """
    Parameters for the general rate model.

    Attributes
    ----------
    bed_porosity : UnsignedFloat between 0 and 1.
        Porosity of the bed
    particle_porosity : UnsignedFloat between 0 and 1.
        Porosity of particles.
    particle_radius : UnsignedFloat
        Radius of the particles.
    film_diffusion : List of unsigned floats. Length depends on n_comp.
        Film diffusion coefficients for each component.
    pore_accessibility : List of unsigned floats. Length depends on n_comp.
        Accessibility of pores for components.
    pore_diffusion : List of unsigned floats. Length depends on n_comp.
        Diffusion rate for components in pore volume.
    surface_diffusion : List of unsigned floats. Length depends on n_comp.
        Diffusion rate for components in adsorbed state.
    c : List of unsigned floats. Length depends on n_comp
        Initial concentration in the mobile phase.
    cp : List of unsigned floats. Length depends on n_comp
        Initial concentration in the stagnant mobile phase within the pores.
    q : List of unsigned floats. Length depends on n_comp
        Initial concentration of the bound phase.
    solution_recorder : GRMRecorder
        Solution recorder for the unit operation.
    """

    supports_binding = True
    supports_bulk_reaction = True
    supports_particle_reaction = True
    discretization_schemes = (GRMDiscretizationFV, GRMDiscretizationDG)

    bed_porosity = UnsignedFloat(ub=1)
    particle_porosity = UnsignedFloat(ub=1)
    particle_radius = UnsignedFloat()
    film_diffusion = SizedUnsignedList(size="n_comp")
    pore_accessibility = SizedUnsignedList(ub=1, size="n_comp", default=1)
    pore_diffusion = SizedUnsignedList(size="n_comp")
    _surface_diffusion = SizedUnsignedList(size="n_bound_states")
    _parameters = [
        "bed_porosity",
        "particle_porosity",
        "particle_radius",
        "film_diffusion",
        "pore_accessibility",
        "pore_diffusion",
        "surface_diffusion",
    ]
    _section_dependent_parameters = TubularReactorBase._section_dependent_parameters + [
        "film_diffusion",
        "pore_accessibility",
        "pore_diffusion",
        "surface_diffusion",
    ]

    _cp = SizedUnsignedList(size="n_comp")
    _q = SizedUnsignedList(size="n_bound_states", default=0)

    _initial_state = ChromatographicColumnBase._initial_state + ["cp", "q"]
    _parameters = _parameters + _initial_state

    def __init__(
        self, *args: Any, discretization_scheme: Optional[str] = "FV", **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        if discretization_scheme == "FV":
            self.discretization = GRMDiscretizationFV()
        elif discretization_scheme == "DG":
            self.discretization = GRMDiscretizationDG()

        self.solution_recorder = GRMRecorder()

    @property
    def total_porosity(self) -> float:
        """float: Total porosity of the column."""
        return self.bed_porosity + (1 - self.bed_porosity) * self.particle_porosity

    @property
    def cross_section_area_interstitial(self) -> float:
        """float: Interstitial area between particles.

        See Also
        --------
        cross_section_area

        """
        return self.bed_porosity * self.cross_section_area

    def set_diameter_from_interstitial_velocity(self, Q: float, u0: float) -> None:
        """
        Set diamter from flow rate and interstitial velocity.

        In literature, often only the interstitial velocity is given.
        This method, the diameter / cross section area can be inferred from
        the flow rate, velocity, and bed porosity.


        Parameters
        ----------
        Q : float
            Volumetric flow rate.
        u0 : float
            Interstitial velocity.

        Notes
        -----
            Overwrites parent method.
        """
        self.cross_section_area = Q / (u0 * self.bed_porosity)

    @property
    def cp(self) -> list[float]:
        """list[float]: Initial particle liquid concentration."""
        if self._cp is None:
            return self.c
        else:
            return self._cp

    @cp.setter
    def cp(self, cp: list[float]) -> None:
        self._cp = cp

        self.parameters["cp"] = cp

    @property
    def q(self) -> list[float]:
        """list[float]: Initial solid phase concentration."""
        return self._q

    @q.setter
    def q(self, q: list[float]) -> None:
        if isinstance(self.binding_model, NoBinding) and q not in (None, []):
            raise CADETProcessError("Cannot set q without binding model.")
        self._q = q

        self.parameters["q"] = q

    @property
    def surface_diffusion(self) -> list[float]:
        """list[float] | None: Surface diffusion coefficients."""
        return self._surface_diffusion

    @surface_diffusion.setter
    def surface_diffusion(self, surface_diffusion: list[float]) -> None:
        if isinstance(self.binding_model, NoBinding):
            raise CADETProcessError(
                "Cannot set surface diffusion without binding model."
            )
        self._surface_diffusion = surface_diffusion

        self.parameters["surface_diffusion"] = surface_diffusion


class Cstr(UnitBaseClass, SourceMixin, SinkMixin):
    """
    Parameters for an ideal mixer.

    Parameters
    ----------
    c : List of unsigned floats. Length depends on n_comp
        Initial concentration of the reactor.
    q : List of unsigned floats. Length depends on n_comp
        Initial concentration of the bound phase.
    V : unsigned float
        Initial volume of the reactor.
    porosity : UnsignedFloat between 0 and 1.
        Total porosity of the Cstr.
    initial_liquid_volume : UnsignedFloat above 0.
        Initial liquid volume of the reactor.
    const_solid_volume : UnsignedFloat above or equal 0.
        Initial and constant solid volume of the reactor.
    flow_rate_filter: float
        Flow rate of pure liquid without components to reduce volume.
    solution_recorder : CSTRRecorder
        Solution recorder for the unit operation.

    Notes
    -----
    CADET generally supports particle reactions for the CSTR, however, this is currently
    not exposed since there are some issues with the interface (.
    """

    supports_binding = True
    supports_bulk_reaction = True
    supports_particle_reaction = False

    flow_rate_filter = UnsignedFloat(default=0)
    _parameters = ["const_solid_volume", "flow_rate_filter"]

    _section_dependent_parameters = (
        UnitBaseClass._section_dependent_parameters
        + SourceMixin._section_dependent_parameters
        + ["flow_rate_filter"]
    )

    c = SizedFloatList(size="n_comp", default=0)
    _q = SizedUnsignedList(size="n_bound_states", default=0)
    init_liquid_volume = UnsignedFloat()
    const_solid_volume = UnsignedFloat(default=0)
    _V = UnsignedFloat()
    _initial_state = UnitBaseClass._initial_state + ["c", "q", "init_liquid_volume"]
    _parameters = _parameters + _initial_state

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.solution_recorder = CSTRRecorder()

    @property
    def required_parameters(self) -> list:
        """
        Remove 'flow_rate' from required parameters.

        If flow rate is None, Q_in == Q_out'.
        """
        required_parameters = super().required_parameters.copy()
        required_parameters.remove("flow_rate")
        return required_parameters

    @property
    def porosity(self) -> float:
        """float: Porosity of the unit operation."""
        if self.const_solid_volume is None or self.init_liquid_volume is None:
            return None
        return self.init_liquid_volume / (
            self.init_liquid_volume + self.const_solid_volume
        )

    @porosity.setter
    def porosity(self, porosity: float) -> None:
        warnings.warn(
            "Field POROSITY is only supported for backwards compatibility, but the implementation "
            "of the CSTR has changed, please refer to the documentation. The POROSITY will be "
            "used to compute the constant solid volume from the total volume V."
        )
        if self.V is None:
            raise RuntimeError("Please set the volume first before setting a porosity.")
        self.const_solid_volume = self.V * (1 - porosity)
        self.init_liquid_volume = self.V * porosity

    @property
    def V(self) -> float:
        """float: Total volume of the unit operation."""
        return self._V

    @V.setter
    def V(self, V: float) -> None:
        warnings.warn(
            "The field V is only supported for backwards compatibility. "
            "Please set initial_liquid_volume and const_solid_volume."
        )
        self.init_liquid_volume = V
        self._V = V

    @property
    def volume(self) -> float:
        """float: Alias for volume."""
        return self.const_solid_volume + self.init_liquid_volume

    @property
    def volume_liquid(self) -> float:
        """float: Volume of the liquid phase."""
        return self.init_liquid_volume

    @property
    def volume_solid(self) -> float:
        """float: Volume of the solid phase."""
        return self.const_solid_volume

    def calculate_interstitial_rt(self, flow_rate: float) -> float:
        """
        Calculate mean residence time of a (non adsorbing) volume element.

        Parameters
        ----------
        flow_rate : float
            Volumetric flow rate.

        Returns
        -------
        t0 : float
            Mean residence time

        See Also
        --------
        calculate_interstitial_velocity
        """
        return self.volume_liquid / flow_rate

    @property
    def q(self) -> list:
        """list[float]: Initial solid phase concentration."""
        return self._q

    @q.setter
    def q(self, q: list) -> None:
        if isinstance(self.binding_model, NoBinding) and q not in (None, []):
            raise CADETProcessError("Cannot set q without binding model.")
        self._q = q

        self.parameters["q"] = q


class MCT(UnitBaseClass):
    """
    Parameters for multi-channel transportmodel.

    Parameters
    ----------
    length : UnsignedFloat
        Length of column.
    channel_cross_section_areas : List of unsinged floats. Lenght depends on nchannel.
        Diameter of column.
    axial_dispersion : List of unsigned floats. Length depends on n_comp and nchannel.
        Axial dispersion coefficient for component and each component.
    flow_direction : Switch
        If 1: Forward flow.
        If -1: Backwards flow.
    c : List of unsigned floats. Length depends n_comp or n_comp*nchannel.
        Initial concentration for components.
    exchange_matrix : List of unsigned floats. Lenght depends on nchannel.
    solution_recorder : MCTRecorder
        Solution recorder for the unit operation.
    n_channel : int
        Number of channels
    """

    has_ports = True
    supports_bulk_reaction = True

    discretization_schemes = MCTDiscretizationFV

    length = UnsignedFloat()
    channel_cross_section_areas = SizedFloatList(size="nchannel")
    axial_dispersion = SizedUnsignedList(size=("n_comp", "nchannel"))
    flow_direction = Switch(valid=[-1, 1], default=1)

    exchange_matrix = SizedNdArray(size=("nchannel", "nchannel", "n_comp"))

    _parameters = [
        "length",
        "channel_cross_section_areas",
        "axial_dispersion",
        "flow_direction",
        "exchange_matrix",
        "nchannel",
    ]
    _section_dependent_parameters = UnitBaseClass._section_dependent_parameters + [
        "axial_dispersion",
        "flow_direction",
    ]

    _section_dependent_parameters = UnitBaseClass._section_dependent_parameters + []

    c = SizedNdArray(size=("n_comp", "nchannel"), default=0)
    _initial_state = UnitBaseClass._initial_state + ["c"]
    _parameters = _parameters + _initial_state

    def __init__(self, *args: Any, nchannel: int, **kwargs: Any) -> None:
        """Initialize MCT."""
        discretization = MCTDiscretizationFV()
        self._nchannel = nchannel
        super().__init__(
            *args,
            discretization=discretization,
            solution_recorder=MCTRecorder(),
            **kwargs,
        )

    @property
    def nchannel(self) -> int:
        """int: The number of channels in the MCT."""
        return self._nchannel

    @nchannel.setter
    def nchannel(self, nchannel: int) -> None:
        self._nchannel = nchannel

    @property
    def ports(self) -> list:
        """list[str]: Ports of the unit operation."""
        return [f"channel_{i}" for i in range(self.nchannel)]

    @property
    def n_ports(self) -> int:
        """int: Number of ports (here the number of channels)."""
        return self.nchannel

    @property
    def volume(self) -> float:
        """
        float: Combined Volumes of all channels.

        See Also
        --------
        channel_cross_section_areas

        """
        return sum(self.channel_cross_section_areas) * self.length

    @property
    def volume_liquid(self) -> float:
        """float: Volume of the liquid phase. Equals the volume, since there is no solid phase."""
        return self.volume

    @property
    def volume_solid(self) -> float:
        """float: Volume of the solid phase. Equals zero, since there is no solid phase."""
        return 0
