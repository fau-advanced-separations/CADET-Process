from __future__ import annotations

import warnings
from functools import wraps
from typing import Any, Optional

import numpy as np

from CADETProcess.dataStructure import (
    Aggregator,
    Bool,
    SizedAggregator,
    SizedClassDependentAggregator,
    SizedNdArray,
    String,
    Structure,
    UnsignedFloat,
    deprecated_alias,
)

from .componentSystem import ComponentSystem


class Reaction(Structure):
    """
    Helper class to store information about individual Mass Action Law Reactions.

    This class represents an individual reaction within a MAL model, and
    stores information about the reaction's stoichiometry, rate constants,
    and concentration exponents. It is used to create a ReactionScheme object,
    which contains information about all of the reactions in the MAL model.

    Attributes
    ----------
    component_system : ComponentSystem
        Component system of the reaction.
    k_fwd : float
        Forward reaction rate.
    k_bwd : float
        Backward reaction rate.
    is_kinetic : bool
        Flag indicating whether the reaction is kinetic (i.e., whether the
        reaction rates are explicitly defined) or whether the reaction is
        assumed to be at rapid equilibrium.
    k_fwd_min : float
        Minimum value of the forward reaction rate in case of rapid equilibrium.
    exponents_fwd : list of float
        Concentration exponents of the components in order of indices for
        forward reaction.
    exponents_bwd : list of float
        Concentration exponents of the components in order of indices for
        backward reaction.
    stoich : np.ndarray
        Stoichiometric coefficients of the components in the reaction.
    n_comp : int
        The number of components in the reaction.
    k_eq : float
        The equilibrium constant for the reaction.
    """

    is_kinetic = Bool(default=True)
    stoich = SizedNdArray(size="n_comp")
    k_fwd = UnsignedFloat()
    k_bwd = UnsignedFloat()
    k_fwd_min = UnsignedFloat(default=100)
    exponents_fwd = SizedNdArray(size="n_comp", default=0)
    exponents_bwd = SizedNdArray(size="n_comp", default=0)

    _parameters = [
        "is_kinetic",
        "stoich",
        "k_fwd",
        "k_bwd",
        "k_fwd_min",
        "exponents_fwd",
        "exponents_bwd",
    ]

    @deprecated_alias(indices="components")
    def __init__(
        self,
        component_system: ComponentSystem,
        components: list[int | str],
        coefficients: np.ndarray,
        k_fwd: float,
        k_bwd: float = 1,
        is_kinetic: bool = True,
        k_fwd_min: float = 100,
        exponents_fwd: Optional[list[float]] = None,
        exponents_bwd: Optional[list[float]] = None,
    ) -> None:
        """
        Initialize individual Mass Action Law Reaction.

        Parameters
        ----------
        component_system : ComponentSystem
            Component system of the reaction.
        components : list of int or strings
            Component names of the components involved in the reaction.
        coefficients : np.ndarray
            Stoichiometric coefficients in the same order of components .
        k_fwd : float
            Forward reaction rate.
        k_bwd : float, optional
            Backward reaction rate. The default is 1.
        is_kinetic : bool, optional
            If False, reaction rates are scaled up to approximate rapid
            equilibriums. The default is True.
        k_fwd_min : float, optional
            Minimum value of foward reaction rate in case of rapid equilbrium.
            The default is 100.
        exponents_fwd : list of float, optional
            Concentration exponents of the components in order of components for
            forward reaction. If None is given, values are inferred from the
            stoichiometric coefficients. The default is None.
        exponents_bwd : list of float, optional
            Concentration exponents of the components in order of components for
            backward reaction. If None is given, values are inferred from the
            stoichiometric coefficients. The default is None.
        """
        self.component_system = component_system
        super().__init__()

        self.is_kinetic = is_kinetic
        if not is_kinetic:
            k_fwd, k_bwd = scale_to_rapid_equilibrium(k_fwd, k_fwd_min)

        self.k_fwd = k_fwd
        self.k_bwd = k_bwd

        if isinstance(components[0], str):
            indices = [component_system.species.index(i) for i in components]
        else:
            warnings.warn(
                "Component are expected to be specified by name. "
                "This will be deprecated in future versions.",
                DeprecationWarning,
                stacklevel=2,
            )
            indices = components

        self.stoich = np.zeros((self.n_comp,))
        for i, c in zip(indices, coefficients):
            self.stoich[i] = c

        if exponents_fwd is None:
            e_fwd = np.maximum(np.zeros((self.n_comp,)), -self.stoich)
        else:
            e_fwd = np.zeros((self.n_comp,))
            for i, e in zip(indices, exponents_fwd):
                e_fwd[i] = e
        self.exponents_fwd = e_fwd

        if exponents_bwd is None:
            e_bwd = np.maximum(np.zeros((self.n_comp,)), self.stoich)
        else:
            e_bwd = np.zeros((self.n_comp,))
            for i, e in zip(indices, exponents_bwd):
                e_bwd[i] = e
        self.exponents_bwd = e_bwd

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self.component_system.n_comp

    @property
    def k_eq(self) -> float:
        """float: Equilibrium constant (Ratio of forward and backward reaction)."""
        return self.k_fwd / self.k_bwd

    def __str__(self) -> str:
        """str: String representation of the Reaction."""
        educts = []
        products = []
        for i, nu in enumerate(self.stoich):
            if nu < 0:
                if nu == -1:
                    educts.append(f"{self.component_system.species[i]}")
                else:
                    educts.append(f"{abs(nu)} {self.component_system.species[i]}")
            elif nu > 0:
                if nu == 1:
                    products.append(f"{self.component_system.species[i]}")
                else:
                    products.append(f"{nu} {self.component_system.species[i]}")

        if self.is_kinetic:
            reaction_operator = f" <=>[{self.k_fwd:.2E}][{self.k_bwd:.2E}] "
        else:
            reaction_operator = f" <=>[{self.k_eq:.2E}] "

        return " + ".join(educts) + reaction_operator + " + ".join(products)


class CrossPhaseReaction(Structure):
    """
    Helper class to store information about cross-phase Mass Action Law reactions.

    Attributes
    ----------
    component_system : ComponentSystem
        The component system of the reaction.
    stoich_liquid : ndarray
        An array of stoichiometric coefficients of the components in the liquid phase.
    stoich_solid : ndarray
        An array of stoichiometric coefficients of the components in the solid phase.
    exponents_fwd_solid_modliquid : ndarray
        An array of concentration exponents of the components in the solid phase for the
        forward reaction in the liquid phase.
    exponents_bwd_solid_modliquid : ndarray
        An array of concentration exponents of the components in the solid phase for the
        backward reaction in the liquid phase.
    exponents_fwd_liquid_modsolid : ndarray
        An array of concentration exponents of the components in the liquid phase for
        the forward reaction in the solid phase.
    exponents_bwd_liquid_modsolid : ndarray
        An array of concentration exponents of the components in the liquid phase for
        the backward reaction in the solid phase.
    is_kinetic : bool
        A boolean flag indicating whether the reaction is kinetic or not.
    k_fwd : float
        The forward reaction rate.
    k_bwd : float
        The backward reaction rate.
    exponents_fwd_liquid : ndarray
        An array of concentration exponents of the components in the liquid phase for
        the forward reaction.
    exponents_bwd_liquid : ndarray
        An array of concentration exponents of the components in the liquid phase for
        the backward reaction.
    exponents_fwd_solid : ndarray
        An array of concentration exponents of the components in the solid phase for the
        forward reaction.
    exponents_bwd_solid : ndarray
        An array of concentration exponents of the components in the solid phase for the
        backward reaction.
    """

    is_kinetic = Bool(default=True)

    stoich_liquid = SizedNdArray(size="n_comp", default=0)
    stoich_solid = SizedNdArray(size="n_comp", default=0)
    k_fwd = UnsignedFloat()
    k_bwd = UnsignedFloat()
    k_fwd_min = UnsignedFloat(default=100)

    exponents_fwd_liquid = SizedNdArray(size="n_comp", default=0)
    exponents_fwd_solid = SizedNdArray(size="n_comp", default=0)

    exponents_bwd_liquid = SizedNdArray(size="n_comp", default=0)
    exponents_bwd_solid = SizedNdArray(size="n_comp", default=0)

    exponents_fwd_liquid_modsolid = SizedNdArray(size="n_comp", default=0)
    exponents_fwd_solid_modliquid = SizedNdArray(size="n_comp", default=0)

    exponents_bwd_liquid_modsolid = SizedNdArray(size="n_comp", default=0)
    exponents_bwd_solid_modliquid = SizedNdArray(size="n_comp", default=0)

    _parameters = [
        "stoich_liquid",
        "stoich_solid",
        "k_fwd",
        "k_bwd",
        "k_fwd_min",
        "exponents_fwd_liquid",
        "exponents_fwd_solid",
        "exponents_bwd_liquid",
        "exponents_bwd_solid",
        "exponents_fwd_liquid_modsolid",
        "exponents_fwd_solid_modliquid",
        "exponents_bwd_liquid_modsolid",
        "exponents_bwd_solid_modliquid",
    ]

    @deprecated_alias(indices="components")
    def __init__(
        self,
        component_system: ComponentSystem,
        components: list[list | str],
        coefficients: list[float],
        phases: list[int],
        k_fwd: float,
        k_bwd: float = 1,
        is_kinetic: bool = True,
        k_fwd_min: float = 100,
        exponents_fwd_liquid: Optional[list[float]] = None,
        exponents_bwd_liquid: Optional[list[float]] = None,
        exponents_fwd_solid: Optional[list[float]] = None,
        exponents_bwd_solid: Optional[list[float]] = None,
    ) -> None:
        """
        Initialize individual cross-phase MAL reaction.

        Parameters
        ----------
        component_system : ComponentSystem
            Component system of the reaction.
        components : list of int or strings
            Component names of the components involved in the reaction.
        coefficients : list
            Stoichiometric coefficients in the same order of component indices.
        phases : list
            phase indices of the component.
            0: liquid phase
            1: solid phase
        k_fwd : float
            Forward reaction rate.
        k_bwd : float, optional
            Backward reaction rate. The default is 1.
        is_kinetic : Bool, optional
            If False, reaction rates are scaled up to approximate rapid
            equilibriums. The default is True.
        k_fwd_min : float, optional
            Minimum value of foward reaction rate in case of rapid equilbrium.
            The default is 100.
        exponents_fwd_liquid : list, optional
            Concentration exponents of the components in order of indices for
            forward reaction in liquid phase. If None is given, values are
            inferred from the stoichiometric coefficients. The default is None.
        exponents_bwd_liquid : list, optional
            Concentration exponents of the components in order of indices for
            backward reaction in liquid phase. If None is given, values are
            inferred from the stoichiometric coefficients. The default is None.
        exponents_fwd_solid : list, optional
            Concentration exponents of the components in order of indices for
            forward reaction in solid phase. If None is given, values are
            inferred from the stoichiometric coefficients. The default is None.
        exponents_bwd_solid : list, optional
            Concentration exponents of the components in order of indices for
            backward reaction in solid phase. If None is given, values are
            inferred from the stoichiometric coefficients. The default is None.
        """
        self.component_system = component_system
        super().__init__()

        self.stoich_liquid = np.zeros((self.n_comp,))
        self.stoich_solid = np.zeros((self.n_comp,))
        self.exponents_fwd_solid_modliquid = np.zeros((self.n_comp,))
        self.exponents_bwd_solid_modliquid = np.zeros((self.n_comp,))
        self.exponents_fwd_liquid_modsolid = np.zeros((self.n_comp,))
        self.exponents_bwd_liquid_modsolid = np.zeros((self.n_comp,))

        if isinstance(components[0], str):
            indices = [component_system.species.index(i) for i in components]
        else:
            warnings.warn(
                "Component are expected to be specified by name. "
                "This will be deprecated in future versions.",
                DeprecationWarning,
                stacklevel=2,
            )
            indices = components

        if phases is None:
            phases = [0 for n in indices]

        for i, c, p in zip(indices, coefficients, phases):
            if p == 0:
                self.stoich_liquid[i] = c
                if c < 0:
                    self.exponents_fwd_solid_modliquid[i] = abs(c)
                elif c > 0:
                    self.exponents_bwd_solid_modliquid[i] = c
            elif p == 1:
                self.stoich_solid[i] = c
                if c < 0:
                    self.exponents_fwd_liquid_modsolid[i] = abs(c)
                elif c > 0:
                    self.exponents_bwd_liquid_modsolid[i] = c

        self.is_kinetic = is_kinetic
        if not is_kinetic:
            k_fwd, k_bwd = scale_to_rapid_equilibrium(k_fwd, k_fwd_min)

        self.k_fwd = k_fwd
        self.k_bwd = k_bwd

        if exponents_fwd_liquid is None:
            e_fwd = np.maximum(np.zeros((self.n_comp,)), -self.stoich_liquid)
        else:
            e_fwd = np.zeros((self.n_comp,))
            for i, e in zip(indices, exponents_fwd_liquid):
                e_fwd[i] = e
        self.exponents_fwd_liquid = e_fwd

        if exponents_bwd_liquid is None:
            e_bwd = np.maximum(np.zeros((self.n_comp,)), self.stoich_liquid)
        else:
            e_bwd = np.zeros((self.n_comp,))
            for i, e in zip(indices, exponents_bwd_liquid):
                e_bwd[i] = e
        self.exponents_bwd_liquid = e_bwd

        if exponents_fwd_solid is None:
            e_fwd = np.maximum(np.zeros((self.n_comp,)), -self.stoich_solid)
        else:
            e_fwd = np.zeros((self.n_comp,))
            for i, e in zip(indices, exponents_fwd_solid):
                e_fwd[i] = e
        self.exponents_fwd_solid = e_fwd

        if exponents_bwd_solid is None:
            e_bwd = np.maximum(np.zeros((self.n_comp,)), self.stoich_solid)
        else:
            e_bwd = np.zeros((self.n_comp,))
            for i, e in zip(indices, exponents_bwd_solid):
                e_bwd[i] = e
        self.exponents_bwd_solid = e_bwd

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self.component_system.n_comp

    @property
    def k_eq(self) -> float:
        """float: Equilibrium constant (Ratio of forward and backward reaction)."""
        return self.k_fwd / self.k_bwd

    def __str__(self) -> str:
        """str: String representation of the Reaction."""
        educts = []
        products = []
        for i, nu in enumerate(self.stoich_liquid):
            if nu < 0:
                if nu == -1:
                    educts.append(f"{self.component_system.species[i]}(l)")
                else:
                    educts.append(f"{abs(nu)} {self.component_system.species[i]}(l)")
            elif nu > 0:
                if nu == 1:
                    products.append(f"{self.component_system.species[i]}(l)")
                else:
                    products.append(f"{nu} {self.component_system.species[i]}(l)")
        for i, nu in enumerate(self.stoich_solid):
            if nu < 0:
                if nu == -1:
                    educts.append(f"{self.component_system.species[i]}(s)")
                else:
                    educts.append(f"{abs(nu)} {self.component_system.species[i]}(s)")
            elif nu > 0:
                if nu == 1:
                    products.append(f"{self.component_system.species[i]}(s)")
                else:
                    products.append(f"{nu} {self.component_system.species[i]}(s)")

        if self.is_kinetic:
            reaction_operator = f" <=>[{self.k_fwd:.2E}][{self.k_bwd:.2E}] "
        else:
            reaction_operator = f" <=>[{self.k_eq:.2E}] "

        return " + ".join(educts) + reaction_operator + " + ".join(products)


class ReactionBaseClass(Structure):
    """
    Abstract base class for parameters of reaction models.

    Attributes
    ----------
    parameters : dict
        dict with parameter values.
    name : String
        name of the reaction model.
    """

    name = String()

    _parameters = []

    def __init__(
        self,
        component_system: ComponentSystem,
        name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize reaction base."""
        self.component_system = component_system
        self.name = name

        super().__init__(*args, **kwargs)

    @property
    def model(self) -> str:
        """str: Name of the reaction model."""
        return self.__class__.__name__

    @property
    def component_system(self) -> ComponentSystem:
        """ComponentSystem: Component System."""
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

    def __repr__(self) -> str:
        """str: String representation of the Reaction."""
        return f"{self.__class__.__name__}(n_comp={self.n_comp}, name={self.name})"

    def __str__(self) -> str:
        """str: Name of the Reaction."""
        if self.name is None:
            return self.__class__.__name__
        return self.name


class NoReaction(ReactionBaseClass):
    """
    Dummy class for units that do not experience reaction behavior.

    The number of components is set to zero for this class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize NoReaction."""
        super().__init__(ComponentSystem(), name="NoReaction")


class BulkReactionBase(ReactionBaseClass):
    """Base class for bulk reaction systems."""

    @classmethod
    def to_particle_model() -> ParticleReactionBase:
        """Convert bulk reaction model to particle reaction model."""
        raise NotImplementedError


class MassActionLaw(BulkReactionBase):
    """Parameters for Reaction in Bulk Phase."""

    k_fwd = Aggregator("k_fwd", "reactions")
    k_bwd = Aggregator("k_bwd", "reactions")
    stoich = SizedAggregator("stoich", "reactions", transpose=True)
    exponents_fwd = SizedAggregator("exponents_fwd", "reactions", transpose=True)
    exponents_bwd = SizedAggregator("exponents_bwd", "reactions", transpose=True)

    _parameters = ["stoich", "exponents_fwd", "exponents_bwd", "k_fwd", "k_bwd"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize MassActionLaw."""
        self._reactions = []
        super().__init__(*args, **kwargs)

    @wraps(Reaction.__init__)
    def add_reaction(self, *args: Any, **kwargs: Any) -> Reaction:
        """Add reaction to ReactionSystem."""
        r = Reaction(self.component_system, *args, **kwargs)
        self._reactions.append(r)

        return r

    @property
    def reactions(self) -> int:
        """list: Reactions in ReactionSystem."""
        return self._reactions

    @property
    def n_reactions(self) -> int:
        """int: Number of Reactions."""
        return len(self.reactions)

    @property
    def k_eq(self) -> list:
        """list: Equilibrium constants of liquid phase Reactions."""
        return [r.k_eq for r in self.reactions]

    def to_particle_model(self) -> "MassActionLawParticle":
        """Convert Bulk Reaction Model to Particle Reaction Model."""
        particle_model = MassActionLawParticle(self.component_system, self.name)
        particle_model._liquid_reactions = self.reactions

        return particle_model


class ParticleReactionBase(ReactionBaseClass):
    """Base class for bulk reaction systems."""


class MassActionLawParticle(ParticleReactionBase):
    """Parameters for Reaction in Particle Phase."""

    stoich_liquid = SizedClassDependentAggregator(
        "stoich_liquid",
        "liquid_reactions",
        mapping={CrossPhaseReaction: "stoich_liquid", None: "stoich"},
        transpose=True,
    )
    k_fwd_liquid = Aggregator("k_fwd", "liquid_reactions")
    k_bwd_liquid = Aggregator("k_bwd", "liquid_reactions")
    exponents_fwd_liquid = SizedAggregator(
        "exponents_fwd", "liquid_reactions", transpose=True
    )
    exponents_bwd_liquid = SizedAggregator(
        "exponents_bwd", "liquid_reactions", transpose=True
    )

    stoich_solid = SizedClassDependentAggregator(
        "stoich_solid",
        "solid_reactions",
        mapping={CrossPhaseReaction: "stoich_solid", None: "stoich"},
        transpose=True,
    )
    k_fwd_solid = Aggregator("k_fwd", "solid_reactions")
    k_bwd_solid = Aggregator("k_bwd", "solid_reactions")
    exponents_fwd_solid = SizedAggregator(
        "exponents_fwd", "solid_reactions", transpose=True
    )
    exponents_bwd_solid = SizedAggregator(
        "exponents_bwd", "solid_reactions", transpose=True
    )

    exponents_fwd_liquid_modsolid = SizedClassDependentAggregator(
        "exponents_fwd_liquid_modsolid",
        "liquid_reactions",
        mapping={CrossPhaseReaction: "exponents_fwd_liquid_modsolid", None: None},
        transpose=True,
    )
    exponents_bwd_liquid_modsolid = SizedClassDependentAggregator(
        "exponents_bwd_liquid_modsolid",
        "liquid_reactions",
        mapping={CrossPhaseReaction: "exponents_bwd_liquid_modsolid", None: None},
        transpose=True,
    )

    exponents_fwd_solid_modliquid = SizedClassDependentAggregator(
        "exponents_fwd_solid_modliquid",
        "solid_reactions",
        mapping={CrossPhaseReaction: "exponents_fwd_solid_modliquid", None: None},
        transpose=True,
    )
    exponents_bwd_solid_modliquid = SizedClassDependentAggregator(
        "exponents_bwd_solid_modliquid",
        "solid_reactions",
        mapping={CrossPhaseReaction: "exponents_bwd_solid_modliquid", None: None},
        transpose=True,
    )

    _parameters = [
        "stoich_liquid",
        "exponents_fwd_liquid",
        "exponents_bwd_liquid",
        "k_fwd_liquid",
        "k_bwd_liquid",
        "exponents_fwd_liquid_modsolid",
        "exponents_bwd_liquid_modsolid",
        "stoich_solid",
        "exponents_fwd_solid",
        "exponents_bwd_solid",
        "k_fwd_solid",
        "k_bwd_solid",
        "exponents_fwd_solid_modliquid",
        "exponents_bwd_solid_modliquid",
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize MassActionLawParticle."""
        self._liquid_reactions = []
        self._solid_reactions = []
        self._cross_phase_reactions = []

        super().__init__(*args, **kwargs)

    @wraps(Reaction.__init__)
    def add_liquid_reaction(self, *args: Any, **kwargs: Any) -> None:
        """Add liquid phase to ReactionSystem."""
        r = Reaction(self.component_system, *args, **kwargs)
        self._liquid_reactions.append(r)

    @wraps(Reaction.__init__)
    def add_solid_reaction(self, *args: Any, **kwargs: Any) -> None:
        """Add solid phase to ReactionSystem."""
        r = Reaction(self.component_system, *args, **kwargs)
        self._solid_reactions.append(r)

    @wraps(CrossPhaseReaction.__init__)
    def add_cross_phase_reaction(self, *args: Any, **kwargs: Any) -> None:
        """Add cross phase to ReactionSystem."""
        r = CrossPhaseReaction(self.component_system, *args, **kwargs)
        self._cross_phase_reactions.append(r)

    # Pore Liquid
    @property
    def liquid_reactions(self) -> list:
        """list: Liquid phase Reactions."""
        return self._liquid_reactions + self.cross_phase_reactions

    @property
    def n_liquid_reactions(self) -> int:
        """int: Number of liquid phase Reactions."""
        return len(self.liquid_reactions)

    @property
    def k_eq_liquid(self) -> list:
        """list: Equilibrium constants of liquid phase Reactions."""
        return [r.k_eq for r in self.liquid_reactions]

    # Solid
    @property
    def solid_reactions(self) -> list:
        """list: Solid phase Reactions."""
        return self._solid_reactions + self.cross_phase_reactions

    @property
    def n_solid_reactions(self) -> int:
        """int: Number of solid phase Reactions."""
        return len(self.solid_reactions)

    @property
    def k_eq_solid(self) -> list:
        """list: Equilibrium constants of solid phase Reactions."""
        return [r.k_eq for r in self.solid_reactions]

    # Cross Phase
    @property
    def cross_phase_reactions(self) -> list:
        """list: Cross phase reactions."""
        return self._cross_phase_reactions

    @property
    def n_cross_phase_reactions(self) -> int:
        """int: Number of cross phase Reactions."""
        return len(self.cross_phase_reactions)


def scale_to_rapid_equilibrium(
    k_eq: float,
    k_fwd_min: float = 10,
) -> tuple[float, float]:
    """
    Scale forward and backward reaction rates if only k_eq is known.

    Parameters
    ----------
    k_eq : float
       Equilibrium constant.
    k_fwd_min : float, optional
        Minimum value for forwards reaction. The default is 10.

    Returns
    -------
    k_fwd : float
        Forward reaction rate.
    k_bwd : float
        Backward reaction rate.
    """
    k_fwd = k_fwd_min
    k_bwd = k_fwd_min / k_eq

    return k_fwd, k_bwd
