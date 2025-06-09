from typing import Any, Optional

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import (
    Bool,
    DependentlyModulatedUnsignedList,
    RangedInteger,
    SizedFloatList,
    SizedRangedIntegerList,
    SizedUnsignedIntegerList,
    SizedUnsignedList,
    String,
    Structure,
    UnsignedFloat,
    UnsignedInteger,
    frozen_attributes,
)

from .componentSystem import ComponentSystem

__all__ = [
    "BindingBaseClass",
    "NoBinding",
    "Linear",
    "Langmuir",
    "LangmuirLDF",
    "LangmuirLDFLiquidPhase",
    "BiLangmuir",
    "BiLangmuirLDF",
    "FreundlichLDF",
    "StericMassAction",
    "AntiLangmuir",
    "Spreading",
    "MobilePhaseModulator",
    "ExtendedMobilePhaseModulator",
    "SelfAssociation",
    "BiStericMassAction",
    "MultistateStericMassAction",
    "SimplifiedMultistateStericMassAction",
    "Saska",
    "GeneralizedIonExchange",
    "HICConstantWaterActivity",
    "HICWaterOnHydrophobicSurfaces",
    "MultiComponentColloidal",
]


@frozen_attributes
class BindingBaseClass(Structure):
    """
    Abstract base class for parameters of binding models.

    Attributes
    ----------
    name : str
        name of the binding model.
    component_system : ComponentSystem
        system of components.
    n_comp : int
        number of components.
    n_binding_sites : int
        Number of binding sites.
        Relevant for Multi-Site isotherms such as Bi-Langmuir.
        The default is 1.
    bound_states : list of unsigned integers.
        Number of binding sites per component.
    non_binding_component_indices : list
        (Hardcoded) list of non binding modifier components (e.g. pH).
    is_kinetic : bool
        If False, adsorption is assumed to be in rapid equilibriu.
        The default is True.
    parameters : dict
        dict with parameter values.
    """

    name = String()
    is_kinetic = Bool(default=True)

    n_binding_sites = RangedInteger(lb=1, ub=1, default=1)
    _bound_states = SizedRangedIntegerList(
        size=("n_binding_sites", "n_comp"), lb=0, ub=1, default=1
    )
    non_binding_component_indices = []

    _parameters = ["is_kinetic"]

    def __init__(
        self,
        component_system: ComponentSystem,
        name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize binding model.

        Parameters
        ----------
        component_system: ComponentSystem
            Component system of the binding model.
        name: str
            Name of the binding model.

        """
        self.component_system = component_system
        self.name = name

        super().__init__(*args, **kwargs)

    @property
    def model(self) -> str:
        """str: Name of the binding model."""
        return self.__class__.__name__

    @property
    def component_system(self) -> ComponentSystem:
        """ComponentSystem: Component system of the binding model."""
        return self._component_system

    @component_system.setter
    def component_system(self, component_system: ComponentSystem) -> None:
        if not isinstance(component_system, ComponentSystem):
            raise TypeError("Expected ComponentSystem")
        self._component_system = component_system

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self.component_system.n_comp

    @property
    def bound_states(self) -> list[int]:
        """list[int]: Number of bound states per component."""
        bound_states = self._bound_states
        for i in self.non_binding_component_indices:
            bound_states[i] = 0
        return bound_states

    @bound_states.setter
    def bound_states(self, bound_states: np.ndarray) -> None:
        indices = self.non_binding_component_indices
        if any(bound_states[i] > 0 for i in indices):
            raise CADETProcessError("Cannot set bound state for non-binding component.")

        self._bound_states = bound_states

    @property
    def n_bound_states(self) -> int:
        """int: Number of bound states."""
        return sum(self.bound_states)

    def __repr__(self) -> str:
        """str: String representation of the binding model."""
        return f"{self.__class__.__name__}(\
            component_system={self.component_system}, name={self.name})')"

    def __str__(self) -> str:
        """str: String representation of the binding model."""
        if self.name is None:
            return self.__class__.__name__
        return self.name


class NoBinding(BindingBaseClass):
    """
    Dummy class for units that do not experience binging behavior.

    The number of components is set to zero for this class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize NoBinding."""
        super().__init__(ComponentSystem(), name="NoBinding")


class Linear(BindingBaseClass):
    """
    Linear binding model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants. Length depends on `n_comp`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants. Length depends on `n_comp`.
    """

    adsorption_rate = SizedUnsignedList(size="n_comp")
    desorption_rate = SizedUnsignedList(size="n_comp")

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
    ]


class Langmuir(BindingBaseClass):
    """
    Multi Component Langmuir binding model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants. Length depends on `n_comp`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants. Length depends on `n_comp`.
    capacity : list of unsigned floats.
        Maximum adsorption capacities. Length depends on `n_comp`.
    """

    adsorption_rate = SizedUnsignedList(size="n_comp")
    desorption_rate = SizedUnsignedList(size="n_comp")
    capacity = SizedUnsignedList(size="n_comp")

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
        "capacity",
    ]


class LangmuirLDF(BindingBaseClass):
    """
    Multi Component Langmuir binding model with linear driving force approximation.

    Note, this variant is based on the equilibrium concentration q* for given c.

    Attributes
    ----------
    equilibrium_constant : list of unsigned floats.
        Adsorption rate constants. Length depends on `n_comp`.
    driving_force_coefficient : list of unsigned floats.
        Desorption rate constants. Length depends on `n_comp`.
    capacity : list of unsigned floats.
        Maximum adsorption capacities. Length depends on `n_comp`.
    """

    equilibrium_constant = SizedUnsignedList(size="n_comp")
    driving_force_coefficient = SizedUnsignedList(size="n_comp")
    capacity = SizedUnsignedList(size="n_comp")

    _parameters = [
        "equilibrium_constant",
        "driving_force_coefficient",
        "capacity",
    ]


class LangmuirLDFLiquidPhase(BindingBaseClass):
    """
    Multi Component Langmuir binding model with linear driving force approximation.

    Note, this variant is based on the equilibrium concentration c* for given q.

    Attributes
    ----------
    equilibrium_constant : list of unsigned floats.
        Adsorption rate constants. Length depends on `n_comp`.
    driving_force_coefficient : list of unsigned floats.
        Desorption rate constants. Length depends on `n_comp`.
    capacity : list of unsigned floats.
        Maximum adsorption capacities. Length depends on `n_comp`.
    """

    equilibrium_constant = SizedUnsignedList(size="n_comp")
    driving_force_coefficient = SizedUnsignedList(size="n_comp")
    capacity = SizedUnsignedList(size="n_comp")

    _parameters = [
        "equilibrium_constant",
        "driving_force_coefficient",
        "capacity",
    ]


class BiLangmuir(BindingBaseClass):
    """
    Multi Component Bi-Langmuir binding model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants in state-major ordering.
        Length depends on `n_bound_states`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants in state-major ordering.
        Length depends on `n_bound_states`.
    capacity : list of unsigned floats.
        Maximum adsorption capacities in state-major ordering.
        Length depends on `n_bound_states`.
    """

    n_binding_sites = UnsignedInteger(default=2)

    adsorption_rate = SizedUnsignedList(size="n_bound_states")
    desorption_rate = SizedUnsignedList(size="n_bound_states")
    capacity = SizedUnsignedList(size="n_bound_states")

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
        "capacity",
    ]

    def __init__(self, *args: Any, n_binding_sites: int = 2, **kwargs: Any) -> None:
        """Initialize BiLangmuir."""
        self.n_binding_sites = n_binding_sites

        super().__init__(*args, **kwargs)


class BiLangmuirLDF(BindingBaseClass):
    """
    Multi Component Bi-Langmuir binding model.

    Attributes
    ----------
    equilibrium_constant : list of unsigned floats.
        Adsorption rate constants in state-major ordering.
        Length depends on `n_bound_states`.
    driving_force_coefficient : list of unsigned floats.
        Desorption rate constants in state-major ordering.
        Length depends on `n_bound_states`.
    capacity : list of unsigned floats.
        Maximum adsorption capacities in state-major ordering.
        Length depends on `n_bound_states`.
    """

    n_binding_sites = UnsignedInteger(default=2)

    equilibrium_constant = SizedUnsignedList(size="n_bound_states")
    driving_force_coefficient = SizedUnsignedList(size="n_bound_states")
    capacity = SizedUnsignedList(size="n_bound_states")

    _parameters = [
        "equilibrium_constant",
        "driving_force_coefficient",
        "capacity",
    ]

    def __init__(self, *args: Any, n_binding_sites: int = 2, **kwargs: Any) -> None:
        """Initialize BiLangmuirLDF."""
        self.n_binding_sites = n_binding_sites

        super().__init__(*args, **kwargs)


class FreundlichLDF(BindingBaseClass):
    """
    Freundlich isotherm model.

    Attributes
    ----------
    driving_force_coefficient : list of unsigned floats.
        Driving force coefficient for each component. Length depends on `n_comp`.
    freundlich_coefficient : list of unsigned floats.
        Freundlich coefficient for each component. Length depends on `n_comp`.
    exponent : list of unsigned floats.
        Freundlich exponent for each component. Length depends on `n_comp`.
    """

    driving_force_coefficient = SizedUnsignedList(size="n_comp")
    freundlich_coefficient = SizedUnsignedList(size="n_comp")
    exponent = SizedUnsignedList(size="n_comp")

    _parameters = [
        "driving_force_coefficient",
        "freundlich_coefficient",
        "exponent",
    ]


class StericMassAction(BindingBaseClass):
    r"""
    Steric Mass Action Law binding model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants. Length depends on `n_comp`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants. Length depends on `n_comp`.
    characteristic_charge : list of unsigned floats.
        Characteristic charges of the protein; The number of sites $\nu$ that the
        protein interacts with on the resin surface.
        Length depends on `n_comp`.
    steric_factor : list of unsigned floats.
        Steric factors of the protein: The number of sites $\sigma$ on the surface
        that are shielded by the protein and prevented from exchange with salt
        counterions in solution.
        Length depends on `n_comp`.
    capacity : unsigned float.
        Stationary phase capacity (monovalent salt counterions); The total
        number of binding sites available on the resin surface.
    reference_liquid_phase_conc : unsigned float.
        Reference liquid phase concentration.
        The default is 1.0
    reference_solid_phase_conc : unsigned float.
        Reference liquid phase concentration.
        The default is 1.0
    """

    adsorption_rate = SizedUnsignedList(size="n_comp")
    desorption_rate = SizedUnsignedList(size="n_comp")
    characteristic_charge = SizedUnsignedList(size="n_comp")
    steric_factor = SizedUnsignedList(size="n_comp")
    capacity = UnsignedFloat()
    reference_liquid_phase_conc = UnsignedFloat(default=1.0)
    reference_solid_phase_conc = UnsignedFloat(default=1.0)

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
        "characteristic_charge",
        "steric_factor",
        "capacity",
        "reference_liquid_phase_conc",
        "reference_solid_phase_conc",
    ]

    @property
    def adsorption_rate_untransformed(self) -> list[UnsignedFloat]:
        """list[float]: Untransformed adsorption rate."""
        if self.adsorption_rate is None:
            return None

        nu = np.array(self.characteristic_charge)
        return self.adsorption_rate * self.reference_solid_phase_conc ** (-nu)

    @adsorption_rate_untransformed.setter
    def adsorption_rate_untransformed(
        self,
        adsorption_rate_untransformed: list[float],
    ) -> None:
        if self.characteristic_charge is None:
            raise ValueError(
                "Please set nu before setting an untransformed rate constant."
            )

        nu = np.array(self.characteristic_charge)
        self.adsorption_rate = (
            adsorption_rate_untransformed / self.reference_solid_phase_conc ** (-nu)
        ).tolist()

    @property
    def desorption_rate_untransformed(self) -> list[UnsignedFloat]:
        """list[float]: Untransformed desorption rate."""
        if self.desorption_rate is None:
            return None

        nu = np.array(self.characteristic_charge)
        return self.desorption_rate * self.reference_liquid_phase_conc ** (-nu)

    @desorption_rate_untransformed.setter
    def desorption_rate_untransformed(
        self,
        desorption_rate_untransformed: list[float],
    ) -> None:
        if self.characteristic_charge is None:
            raise ValueError(
                "Please set nu before setting a transformed rate constant."
            )

        nu = np.array(self.characteristic_charge)
        self.desorption_rate = (
            desorption_rate_untransformed / self.reference_liquid_phase_conc ** (-nu)
        ).tolist()


class AntiLangmuir(BindingBaseClass):
    """
    Multi Component Anti-Langmuir adsorption isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants. Length depends on `n_comp`.
    desorption_rate : list of unsigned floats
        Desorption rate constants. Length depends on `n_comp`.
    capacity : list of unsigned floats.
        Maximum adsorption capacities. Length depends on `n_comp`.
    antilangmuir : list of {-1, 1}.
        Anti-Langmuir coefficients. Length depends on `n_comp`.
    """

    adsorption_rate = SizedUnsignedList(size="n_comp")
    desorption_rate = SizedUnsignedList(size="n_comp")
    capacity = SizedUnsignedList(size="n_comp")
    antilangmuir = SizedFloatList(size="n_comp")

    _parameters = ["adsorption_rate", "desorption_rate", "capacity", "antilangmuir"]


class Spreading(BindingBaseClass):
    """
    Multi Component Spreading adsorption isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants in state-major ordering.
        Length depends on `n_total_bound`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants in state-major ordering.
        Length depends on `n_total_bound`.
    capacity : list of unsigned floats.
        Maximum adsorption capacities in state-major ordering.
        Length depends on `n_total_bound`.
    exchange_from_1_2 : list of unsigned floats.
        Exchange rates from the first to the second bound state.
        Length depends on `n_comp`.
    exchange_from_2_1 : list of unsigned floats.
        Exchange rates from the second to the first bound state.
        Length depends on `n_comp`.
    """

    n_binding_sites = RangedInteger(lb=2, ub=2, default=2)

    adsorption_rate = SizedUnsignedList(size="n_total_bound")
    desorption_rate = SizedUnsignedList(size="n_total_bound")
    capacity = SizedUnsignedList(size="n_total_bound")
    exchange_from_1_2 = SizedUnsignedList(size="n_comp")
    exchange_from_2_1 = SizedUnsignedList(size="n_comp")

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
        "capacity",
        "exchange_from_1_2",
        "exchange_from_2_1",
    ]


class MobilePhaseModulator(BindingBaseClass):
    """
    Mobile Phase Modulator adsorption isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants. Length depends on `n_comp`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants. Length depends on `n_comp`.
    capacity : list of unsigned floats.
        Maximum adsorption capacities. Length depends on `n_comp`.
    ion_exchange_characteristic : list of unsigned floats.
        Parameters describing the ion-exchange characteristics (IEX).
        Length depends on `n_comp`.
    hydrophobicity : list of floats.
        Parameters describing the hydrophobicity (HIC).
        Length depends on `n_comp`.
    linear_threshold : UnsignedFloat
        Concentration of c0 at which to switch to linear model approximation.
    """

    adsorption_rate = SizedUnsignedList(size="n_comp")
    desorption_rate = SizedUnsignedList(size="n_comp")
    capacity = SizedUnsignedList(size="n_comp")
    ion_exchange_characteristic = SizedUnsignedList(size="n_comp")
    beta = ion_exchange_characteristic
    hydrophobicity = SizedFloatList(size="n_comp")
    gamma = hydrophobicity
    linear_threshold = UnsignedFloat(default=1e-8)

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
        "capacity",
        "ion_exchange_characteristic",
        "hydrophobicity",
        "linear_threshold",
    ]


class ExtendedMobilePhaseModulator(BindingBaseClass):
    """
    Mobile Phase Modulator adsorption isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants. Length depends on `n_comp`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants. Length depends on `n_comp`.
    capacity : list of unsigned floats.
        Maximum adsorption capacities. Length depends on `n_comp`.
    ion_exchange_characteristic : list of unsigned floats.
        Parameters describing the ion-exchange characteristics (IEX).
        Length depends on `n_comp`.
    hydrophobicity : list of floats.
        Parameters describing the hydrophobicity (HIC).
        Length depends on `n_comp`.
    component_mode : list of unsigned integers.
        Mode of each component;
        0 denotes the modifier component,
        1 is linear binding,
        2 is modified Langmuir binding.
        Length depends on `n_comp`.
    """

    adsorption_rate = SizedUnsignedList(size="n_comp")
    desorption_rate = SizedUnsignedList(size="n_comp")
    capacity = SizedUnsignedList(size="n_comp")
    ion_exchange_characteristic = SizedUnsignedList(size="n_comp")
    beta = ion_exchange_characteristic
    hydrophobicity = SizedFloatList(size="n_comp")
    gamma = hydrophobicity
    component_mode = SizedUnsignedIntegerList(size="n_comp", ub=2)

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
        "capacity",
        "ion_exchange_characteristic",
        "hydrophobicity",
        "component_mode",
    ]


class SelfAssociation(BindingBaseClass):
    r"""
    Self Association adsorption isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants. Length depends on `n_comp`.
    adsorption_rate_dimerization : list of unsigned floats.
        Adsorption rate constants of dimerization. Length depends on `n_comp`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants. Length depends on `n_comp`.
    characteristic_charge : list of unsigned floats.
        The characteristic charge $\nu$ of the protein. Length depends on `n_comp`.
    steric_factor : list of unsigned floats.
        Steric factor of of the protein. Length depends on `n_comp`.
    capacity : unsigned float.
        Stationary phase capacity (monovalent salt counterions); The total number of
        binding sites available on the resin surface.
    reference_liquid_phase_conc : unsigned float.
        Reference liquid phase concentration (optional, default value = 1.0).
        The default = 1.0
    reference_solid_phase_conc : unsigned float.
        Reference liquid phase concentration (optional)
        The default = 1.0
    """

    adsorption_rate = SizedUnsignedList(size="n_comp")
    adsorption_rate_dimerization = SizedUnsignedList(size="n_comp")
    desorption_rate = SizedUnsignedList(size="n_comp")
    characteristic_charge = SizedUnsignedList(size="n_comp")
    steric_factor = SizedUnsignedList(size="n_comp")
    capacity = UnsignedFloat()
    reference_liquid_phase_conc = UnsignedFloat(default=1.0)
    reference_solid_phase_conc = UnsignedFloat(default=1.0)

    _parameters = [
        "adsorption_rate",
        "adsorption_rate_dimerization",
        "desorption_rate",
        "characteristic_charge",
        "steric_factor",
        "capacity",
        "reference_liquid_phase_conc",
        "reference_solid_phase_conc",
    ]


class BiStericMassAction(BindingBaseClass):
    """
    Bi Steric Mass Action adsorption isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants in state-major ordering.
        Length depends on `n_bound_states`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants in state-major ordering.
        Length depends on `n_bound_states`.
    characteristic_charge : list of unsigned floats.
        Characteristic charges v(i,j) of the it-h protein with respect to the
        j-th binding site type in state-major ordering.
        Length depends on `n_bound_states`.
    steric_factor : list of unsigned floats.
        Steric factor o (i,j) of the it-h protein with respect to the j-th
        binding site type in state-major ordering.
        Length depends on `n_bound_states`.
    capacity : unsigned float.
        Stationary phase capacity (monovalent salt counterions); The total
        number of binding sites available on the resin surface.
        Length depends on `n_binding_sites`.
    reference_liquid_phase_conc : list of unsigned floats.
        Reference liquid phase concentration for each binding site type or one
        value for all types.
        The default is 1.0
    reference_solid_phase_conc : list of unsigned floats.
        Reference solid phase concentration for each binding site type or one
        value for all types.
        The default is 1.0
    """

    n_binding_sites = UnsignedInteger(default=2)

    adsorption_rate = SizedUnsignedList(size="n_bound_states")
    adsorption_rate_dimerization = SizedUnsignedList(size="n_bound_states")
    desorption_rate = SizedUnsignedList(size="n_bound_states")
    characteristic_charge = SizedUnsignedList(size="n_bound_states")
    steric_factor = SizedUnsignedList(size="n_bound_states")
    capacity = SizedUnsignedList(size="n_binding_sites")
    reference_liquid_phase_conc = SizedUnsignedList(size="n_binding_sites", default=1)
    reference_solid_phase_conc = SizedUnsignedList(size="n_binding_sites", default=1)

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
        "characteristic_charge",
        "steric_factor",
        "capacity",
        "reference_liquid_phase_conc",
        "reference_solid_phase_conc",
    ]

    def __init__(self, *args: Any, n_states: int = 2, **kwargs: Any) -> None:
        """Initialize BiStericMassAction class."""
        self.n_states = n_states
        super().__init__(*args, **kwargs)


class MultistateStericMassAction(BindingBaseClass):
    r"""
    Multistate Steric Mass Action adsorption isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants of the components to different bound states in
        component-major ordering.
        Length depends on `n_bound_states`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants of the components to different bound states in
        component-major ordering.
        Length depends on `n_bound_states`.
    characteristic_charge : list of unsigned floats.
        Characteristic charges of the components to different bound states in
        component-major ordering.
        Length depends on `n_bound_states`.
    steric_factor : list of unsigned floats.
        Steric factor of the components to different bound states in component-major
        ordering.
        Length depends on `n_bound_states`.
    conversion_rate : list of unsigned floats.
        Conversion rates between different bound states in
        component-major ordering.
        Length: $sum_{i=1}^{n_{comp}} n_{bound, i}$
    capacity : unsigned float.
        Stationary phase capacity (monovalent salt counterions); The total number of
        binding sites available on the resin surface.
    reference_liquid_phase_conc : unsigned float.
        Reference liquid phase concentration.
        The default = 1.0
    reference_solid_phase_conc : unsigned float, optional
        Reference solid phase concentration.
        The default = 1.0
    """

    bound_states = SizedUnsignedIntegerList(
        size=("n_binding_sites", "n_comp"), default=1
    )

    adsorption_rate = SizedUnsignedList(size="n_bound_states")
    desorption_rate = SizedUnsignedList(size="n_bound_states")
    characteristic_charge = SizedUnsignedList(size="n_bound_states")
    steric_factor = SizedUnsignedList(size="n_bound_states")
    conversion_rate = SizedUnsignedList(size="_conversion_entries")
    capacity = UnsignedFloat()
    reference_liquid_phase_conc = UnsignedFloat(default=1)
    reference_solid_phase_conc = UnsignedFloat(default=1)

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
        "characteristic_charge",
        "steric_factor",
        "conversion_rate",
        "capacity",
        "reference_liquid_phase_conc",
        "reference_solid_phase_conc",
    ]

    @property
    def _conversion_entries(self) -> int:
        n = 0
        for state in self.bound_states:
            n += state**2

        return n


class SimplifiedMultistateStericMassAction(BindingBaseClass):
    """
    Simplified multistate Steric Mass Action adsorption isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants of the components to different bound states
        in component-major ordering.
        Length depends on `n_bound_states`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants of the components to different bound states
        in component-major ordering.
        Length depends on `n_bound_states`.
    characteristic_charge_first : list of unsigned floats.
        Characteristic charges of the components in the first (weakest) bound state.
        Length depends on `n_comp`.
    characteristic_charge_last : list of unsigned floats.
        Characteristic charges of the components in the last (strongest) bound state.
        Length depends on `n_comp`.
    quadratic_modifiers_charge : list of unsigned floats.
        Quadratic modifiers of the characteristic charges of the different components
        depending on the index of the bound state.
        Length depends on `n_comp`.
    steric_factor_first : list of unsigned floats.
        Steric factor of the components in the first (weakest) bound state.
        Length depends on `n_comp`.
    steric_factor_last : list of unsigned floats.
        Steric factor of the components in the last (strongest) bound state.
        Length depends on `n_comp`.
    quadratic_modifiers_steric : list of unsigned floats.
        Quadratic modifiers of the sterif factors of the different components depending
        on the index of the bound state.
        Length depends on `n_comp`.
    capacity : unsigned float.
        Stationary phase capacity (monovalent salt counterions): The total number of
        binding sites available on the resin surface.
    exchange_from_weak_stronger : list of unsigned floats.
        Exchangde rated from a weakly bound state to the next stronger bound state.
    linear_exchange_ws : list of unsigned floats.
        Linear exchange rate coefficients from a weakly bound state to the next stronger
        bound state.
        Length depends on `n_comp`.
    quadratic_exchange_ws : list of unsigned floats.
        Quadratic exchange rate coefficients from a weakly bound state to the next
        stronger bound state.
        Length depends on `n_comp`.
    exchange_from_stronger_weak : list of unsigned floats.
        Exchange rate coefficients from a strongly bound state to the next weaker bound
        state.
        Length depends on `n_comp`.
    linear_exchange_sw : list of unsigned floats.
        Linear exchange rate coefficients from a strongly bound state to the next weaker
        bound state.
        Length depends on `n_comp`.
    quadratic_exchange_sw : list of unsigned floats.
        Quadratic exchange rate coefficients from a strongly bound state to the next
        weaker bound state.
        Length depends on `n_comp`.
    reference_liquid_phase_conc : list of unsigned floats.
        Reference liquid phase concentration (optional, default value = 1.0).
    reference_solid_phase_conc : list of unsigned floats.
        Reference solid phase concentration (optional, default value = 1.0).
    """

    bound_states = SizedUnsignedIntegerList(
        size=("n_binding_sites", "n_comp"), default=1
    )

    adsorption_rate = SizedUnsignedList(size="n_bound_states")
    desorption_rate = SizedUnsignedList(size="n_bound_states")
    characteristic_charge_first = SizedUnsignedList(size="n_comp")
    characteristic_charge_last = SizedUnsignedList(size="n_comp")
    quadratic_modifiers_charge = SizedUnsignedList(size="n_comp")
    steric_factor_first = SizedUnsignedList(size="n_comp")
    steric_factor_last = SizedUnsignedList(size="n_comp")
    quadratic_modifiers_steric = SizedUnsignedList(size="n_comp")
    capacity = UnsignedFloat()
    exchange_from_weak_stronger = SizedUnsignedList(size="n_comp")
    linear_exchange_ws = SizedUnsignedList(size="n_comp")
    quadratic_exchange_ws = SizedUnsignedList(size="n_comp")
    exchange_from_stronger_weak = SizedUnsignedList(size="n_comp")
    linear_exchange_sw = SizedUnsignedList(size="n_comp")
    quadratic_exchange_sw = SizedUnsignedList(size="n_comp")
    reference_liquid_phase_conc = UnsignedFloat(default=1)
    reference_solid_phase_conc = UnsignedFloat(default=1)

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
        "characteristic_charge_first",
        "characteristic_charge_last",
        "quadratic_modifiers_charge",
        "steric_factor_first",
        "steric_factor_last",
        "quadratic_modifiers_steric",
        "capacity",
        "exchange_from_weak_stronger",
        "linear_exchange_ws",
        "quadratic_exchange_ws",
        "exchange_from_stronger_weak",
        "linear_exchange_sw",
        "quadratic_exchange_sw",
        "reference_liquid_phase_conc",
        "reference_solid_phase_conc",
    ]


class Saska(BindingBaseClass):
    """
    Quadratic Isotherm.

    Attributes
    ----------
    henry_const : list of unsigned floats.
        The Henry coefficient. Length depends on `n_comp`.
    quadratic_factor : list of unsigned floats.
        Quadratic factors. Length depends on `n_comp`.
    """

    henry_const = SizedUnsignedList(size="n_comp")
    quadratic_factor = SizedUnsignedList(size=("n_comp", "n_comp"))

    _parameters = [
        "henry_const",
        "quadratic_factor",
    ]


class GeneralizedIonExchange(BindingBaseClass):
    r"""
    Generalized Ion Exchange isotherm model.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants. Length depends on `n_comp`.
    adsorption_rate_linear : list of unsigned floats.
        Linear dependence coefficient of adsorption rate on modifier component
        Length depends on `n_comp`.
    adsorption_rate_quadratic : list of unsigned floats.
        Quadratic dependence coefficient of adsorption rate on modifier component.
        Length depends on `n_comp`.
    adsorption_rate_cubic : list of unsigned floats.
        Cubic dependence coefficient of adsorption rate on modifier component.
        Length depends on `n_comp`.
    adsorption_rate_salt : list of unsigned floats.
        Salt coefficient of adsorption rate;
        difference of water-protein and salt-protein interactions.
        Length depends on `n_comp`.
    adsorption_rate_protein : list of unsigned floats.
        Protein coefficient of adsorption rate;
        difference of water-protein and protein-protein interactions.
        Length depends on `n_comp`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants. Length depends on `n_comp`.
    desorption_rate_linear : list of unsigned floats.
        Linear dependence coefficient of desorption rate on modifier component.
        Length depends on `n_comp`.
    desorption_rate_quadratic : list of unsigned floats.
        Quadratic dependence coefficient of desorption rate on modifier component.
        Length depends on `n_comp`.
    desorption_rate_cubic : list of unsigned floats.
        Cubic dependence coefficient of desorption rate on modifier component.
        Length depends on `n_comp`.
    desorption_rate_salt : list of unsigned floats.
        Salt coefficient of desorption rate;
        difference of water-protein and salt-protein interactions.
        Length depends on `n_comp`.
    desorption_rate_protein : list of unsigned floats.
        Protein coefficient of desorption rate;
        difference of water-protein and protein-protein interactions
        Length depends on `n_comp`.
    characteristic_charge_breaks : list of unsigned floats, optional
        Breaks of the characteristic charge pieces in component-major ordering.
        Optional, only required if a piecewise cubic polynomial is used for $\nu$.
        Length must be a multiple of `n_comp`.
    characteristic_charge : list of unsigned floats.
        Base value for characteristic charges of the protein; The number of sites $\nu$
        that the protein interacts with on the resin surface.
        Length depends on `n_comp` * `n_pieces`.
    characteristic_charge_linear : list of unsigned floats.
        Linear dependence coefficient of characteristic charge on modifier component.
        Length depends on `n_comp` * `n_pieces`.
    characteristic_charge_quadratic : list of unsigned floats.
        Quadratic dependence coefficient of characteristic charge on modifier component.
        Length depends on `n_comp` * `n_pieces`.
    characteristic_charge_cubic : list of unsigned floats.
        Cubic dependence coefficient of characteristic charge on modifier component.
        Length depends on `n_comp` * `n_pieces`.
    steric_factor : list of unsigned floats.
        Steric factors of the protein: The number of sites $\sigma$ on the surface that
        are shielded by the protein and prevented from exchange with salt counterions in
        solution.
        Length depends on `n_comp`.
    capacity : unsigned float.
        Stationary phase capacity (monovalent salt counterions); The total number of
        binding sites available on the resin surface.
    reference_liquid_phase_conc : unsigned float.
        Reference liquid phase concentration (optional, default value = 1.0).
    reference_solid_phase_conc : unsigned float.
        Reference liquid phase concentration (optional, default value = 1.0).
    """

    non_binding_component_indices = [1]

    adsorption_rate = SizedFloatList(size="n_comp")
    adsorption_rate_linear = SizedFloatList(size="n_comp")
    adsorption_rate_quadratic = SizedFloatList(size="n_comp", default=0)
    adsorption_rate_cubic = SizedFloatList(size="n_comp", default=0)
    adsorption_rate_salt = SizedFloatList(size="n_comp", default=0)
    adsorption_rate_protein = SizedFloatList(size="n_comp", default=0)
    desorption_rate = SizedFloatList(size="n_comp")
    desorption_rate_linear = SizedFloatList(size="n_comp", default=0)
    desorption_rate_quadratic = SizedFloatList(size="n_comp", default=0)
    desorption_rate_cubic = SizedFloatList(size="n_comp", default=0)
    desorption_rate_salt = SizedFloatList(size="n_comp", default=0)
    desorption_rate_protein = SizedFloatList(size="n_comp", default=0)
    characteristic_charge_breaks = DependentlyModulatedUnsignedList(
        size="n_comp", is_optional=True
    )
    characteristic_charge = SizedFloatList(
        size=("n_pieces", "n_comp"),
    )
    characteristic_charge_linear = SizedFloatList(
        size=("n_pieces", "n_comp"), default=0
    )
    characteristic_charge_quadratic = SizedFloatList(
        size=("n_pieces", "n_comp"), default=0
    )
    characteristic_charge_cubic = SizedFloatList(size=("n_pieces", "n_comp"), default=0)
    steric_factor = SizedUnsignedList(size="n_comp")
    capacity = UnsignedFloat()
    reference_liquid_phase_conc = UnsignedFloat(default=1)
    reference_solid_phase_conc = UnsignedFloat(default=1)

    _parameters = [
        "adsorption_rate",
        "adsorption_rate_linear",
        "adsorption_rate_quadratic",
        "adsorption_rate_cubic",
        "adsorption_rate_salt",
        "adsorption_rate_protein",
        "desorption_rate",
        "desorption_rate_linear",
        "desorption_rate_quadratic",
        "desorption_rate_cubic",
        "desorption_rate_salt",
        "desorption_rate_protein",
        "characteristic_charge_breaks",
        "characteristic_charge",
        "characteristic_charge_linear",
        "characteristic_charge_quadratic",
        "characteristic_charge_cubic",
        "steric_factor",
        "capacity",
        "reference_liquid_phase_conc",
        "reference_solid_phase_conc",
    ]

    @property
    def n_pieces(self) -> int:
        """int: Number of pieces for cubic polynomial description of nu."""
        if self.characteristic_charge_breaks is None:
            return 1

        n_pieces_all = len(self.characteristic_charge_breaks) - self.n_comp
        n_pieces_comp = int(n_pieces_all / self.n_comp)

        return n_pieces_comp


class HICConstantWaterActivity(BindingBaseClass):
    """
    HIC based on Constant Water Activity adsorption isotherm.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants. Size depends on `n_comp`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants. Size depends on `n_comp`.
    capacity : list of unsigned floats.
        Maximum adsorption capacities. Size depends on `n_comp`.
    hic_characteristic : list of unsigned floats.
        Parameters describing the number of ligands per ligand-protein interaction.
        Size depends on `n_comp`.
    beta_0 : unsigned float.
        Parameter describing the number of highly ordered water molecules that stabilize
        the hydrophobic surfaces at infinitely diluted salt concentration.
    beta_1 : unsigned float.
        Parameter describing the change in the number of highly ordered water molecules that
        stabilize the hydrophobic surfaces with respect to changes in the salt concentration.
    """

    adsorption_rate = SizedFloatList(size="n_comp")
    desorption_rate = SizedFloatList(size="n_comp")
    capacity = SizedFloatList(size="n_comp")
    hic_characteristic = SizedFloatList(size="n_comp")

    beta_0 = UnsignedFloat()
    beta_1 = UnsignedFloat()

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
        "hic_characteristic",
        "capacity",
        "beta_0",
        "beta_1",
    ]


class HICWaterOnHydrophobicSurfaces(BindingBaseClass):
    """
    HIC isotherm by Wang et al. based on their 2016 paper.

    Attributes
    ----------
    adsorption_rate : list of unsigned floats.
        Adsorption rate constants. Size depends on `n_comp`.
    desorption_rate : list of unsigned floats.
        Desorption rate constants. Size depends on `n_comp`.
    capacity : list of unsigned floats.
        Maximum adsorption capacities. Size depends on `n_comp`.
    hic_characteristic : list of unsigned floats.
        Parameters describing the number of ligands per ligand-protein interaction.
        Size depends on `n_comp`.
    beta_0 : unsigned float.
        Parameter describing the number of highly ordered water molecules that stabilize
        the hydrophobic surfaces at infinitely diluted salt concentration.
    beta_1 : unsigned float.
        Parameter describing the change in the number of highly ordered water molecules that
        stabilize the hydrophobic surfaces with respect to changes in the salt concentration.
    """

    adsorption_rate = SizedFloatList(size="n_comp")
    desorption_rate = SizedFloatList(size="n_comp")
    capacity = SizedFloatList(size="n_comp")
    hic_characteristic = SizedFloatList(size="n_comp")

    beta_0 = UnsignedFloat()
    beta_1 = UnsignedFloat()

    _parameters = [
        "adsorption_rate",
        "desorption_rate",
        "hic_characteristic",
        "capacity",
        "beta_0",
        "beta_1",
    ]


class MultiComponentColloidal(BindingBaseClass):
    """
    Colloidal isotherm from Xu and Lenhoff 2009.

    Attributes
    ----------
    phase_ratio : unsigned float.
        Phase ratio.
    kappa_exponential : unsigned float.
        Screening term exponential factor.
    kappa_factor : unsigned float.
        Screening term factor.
    kappa_constant : unsigned float.
        Screening term constant.
    coordination_number : unsigned integer.
        Coordination number.
    logkeq_ph_exponent : list of unsigned floats.
        Equilibrium constant factor exponent term for pH. Size depends on `n_comp`.
    logkeq_power_exponent : list of unsigned floats.
        Equilibrium constant power exponent term for salt. Size depends on `n_comp`.
    logkeq_power_factor : list of unsigned floats.
        Equilibrium constant power factor term for salt. Size depends on `n_comp`.
    logkeq_exponent_factor : list of unsigned floats.
        Equilibrium constant exponent factor term for salt. Size depends on `n_comp`.
    logkeq_exponent_multiplier : list of unsigned floats.
        Equilibrium constant exponent multiplier term for salt. Size depends on `n_comp`.
    bpp_ph_exponent : list of unsigned floats..
        BPP constant exponent factor term for pH. Size depends on `n_comp`.
    bpp_power_exponent : list of unsigned floats.
        Bpp constant power exponent term for salt. Size depends on `n_comp`.
    bpp_power_factor : list of unsigned floats.
        Bpp constant power factor term for salt. Size depends on `n_comp`.
    bpp_exponent_factor  : list of unsigned floats.
        Bpp constant exponent factor term for salt. Size depends on `n_comp`.
    bpp_exponent_multiplier : list of unsigned floats.
        Bpp constant exponent multiplier term for salt. Size depends on `n_comp`.
    protein_radius : list of unsigned floats.
        Protein radius. Size depends on `n_comp`.
    kinetic_rate_constant : list of unsigned floats.
        Adsorption kinetics. Size depends on `n_comp`.
    linear_threshold : unsigned float.
        Linear threshold.
    use_ph : Boolean.
        Include pH or not.
    """

    bound_states = SizedUnsignedIntegerList(
        size=("n_binding_sites", "n_comp"), default=1
    )

    phase_ratio = UnsignedFloat()
    kappa_exponential = UnsignedFloat()
    kappa_factor = UnsignedFloat()
    kappa_constant = UnsignedFloat()
    coordination_number = UnsignedInteger()
    logkeq_ph_exponent = SizedFloatList(size="n_comp")
    logkeq_power_exponent = SizedFloatList(size="n_comp")
    logkeq_power_factor = SizedFloatList(size="n_comp")
    logkeq_exponent_factor = SizedFloatList(size="n_comp")
    logkeq_exponent_multiplier = SizedFloatList(size="n_comp")
    bpp_ph_exponent = SizedFloatList(size="n_comp")
    bpp_power_exponent = SizedFloatList(size="n_comp")
    bpp_power_factor = SizedFloatList(size="n_comp")
    bpp_exponent_factor = SizedFloatList(size="n_comp")
    bpp_exponent_multiplier = SizedFloatList(size="n_comp")
    protein_radius = SizedFloatList(size="n_comp")
    kinetic_rate_constant = SizedFloatList(size="n_comp")
    linear_threshold = UnsignedFloat(default=1e-8)
    use_ph = Bool(default=False)

    _parameters = [
        "phase_ratio",
        "kappa_exponential",
        "kappa_factor",
        "kappa_constant",
        "coordination_number",
        "logkeq_ph_exponent",
        "logkeq_power_exponent",
        "logkeq_power_factor",
        "logkeq_exponent_factor",
        "logkeq_exponent_multiplier",
        "bpp_ph_exponent",
        "bpp_power_exponent",
        "bpp_power_factor",
        "bpp_exponent_factor",
        "bpp_exponent_multiplier",
        "protein_radius",
        "kinetic_rate_constant",
        "linear_threshold",
        "use_ph",
    ]
