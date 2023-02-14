from addict import Dict
from functools import wraps
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import String, UnsignedInteger

from .componentSystem import ComponentSystem


class Reaction():
    """Helper class to store information about individual Mass Action Law Reactions.

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

    def __init__(
            self, component_system, indices, coefficients,
            k_fwd, k_bwd=1, is_kinetic=True, k_fwd_min=100,
            exponents_fwd=None, exponents_bwd=None):
        """Initialize individual Mass Action Law Reaction.

        Parameters
        ----------
        component_system : ComponentSystem
            Component system of the reaction.
        indices : list of int
            Component indices.
        coefficients : list of float
            Stoichiometric coefficients in the same order of component indices.
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
            Concentration exponents of the components in order of indices for
            forward reaction. If None is given, values are inferred from the
            stoichiometric coefficients. The default is None.
        exponents_bwd : list of float, optional
            Concentration exponents of the components in order of indices for
            backward reaction. If None is given, values are inferred from the
            stoichiometric coefficients. The default is None.

        """
        self.component_system = component_system

        self.stoich = np.zeros((self.n_comp,))
        for i, c in zip(indices, coefficients):
            self.stoich[i] = c

        self.is_kinetic = is_kinetic
        if not is_kinetic:
            k_fwd, k_bwd = scale_to_rapid_equilibrium(k_fwd, k_fwd_min)

        self.k_fwd = k_fwd
        self.k_bwd = k_bwd

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
    def n_comp(self):
        """int: Number of components."""
        return self.component_system.n_comp

    @property
    def k_eq(self):
        """float: Equilibrium constant (Ratio of forward and backward reaction)."""
        return self.k_fwd/self.k_bwd

    def __str__(self):
        educts = []
        products = []
        for i, nu in enumerate(self.stoich):
            if nu < 0:
                if nu == - 1:
                    educts.append(f"{self.component_system.labels[i]}")
                else:
                    educts.append(
                        f"{abs(nu)} {self.component_system.labels[i]}"
                    )
            elif nu > 0:
                if nu == 1:
                    products.append(f"{self.component_system.labels[i]}")
                else:
                    products.append(f"{nu} {self.component_system.labels[i]}")

        if self.is_kinetic:
            reaction_operator = f' <=>[{self.k_fwd:.2E}][{self.k_bwd:.2E}] '
        else:
            reaction_operator = f' <=>[{self.k_eq:.2E}] '

        return " + ".join(educts) + reaction_operator + " + ".join(products)


class CrossPhaseReaction():
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

    def __init__(
            self, component_system, indices, coefficients, phases,
            k_fwd, k_bwd=1, is_kinetic=True, k_fwd_min=100,
            exponents_fwd_liquid=None, exponents_bwd_liquid=None,
            exponents_fwd_solid=None, exponents_bwd_solid=None):
        """Initialize individual cross-phase MAL reaction.

        Parameters
        ----------
        component_system : ComponentSystem
            Component system of the reaction.
        indices : list
            Component indices.
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

        self.stoich_liquid = np.zeros((self.n_comp,))
        self.stoich_solid = np.zeros((self.n_comp,))
        self.exponents_fwd_solid_modliquid = np.zeros((self.n_comp,))
        self.exponents_bwd_solid_modliquid = np.zeros((self.n_comp,))
        self.exponents_fwd_liquid_modsolid = np.zeros((self.n_comp,))
        self.exponents_bwd_liquid_modsolid = np.zeros((self.n_comp,))

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
    def n_comp(self):
        """ind: number of components."""
        return self.component_system.n_comp

    @property
    def k_eq(self):
        """float: Equilibrium constant (Ratio of forward and backward reaction)."""
        return self.k_fwd/self.k_bwd

    def __str__(self):
        educts = []
        products = []
        for i, nu in enumerate(self.stoich_liquid):
            if nu < 0:
                if nu == - 1:
                    educts.append(f"{self.component_system.labels[i]}(l)")
                else:
                    educts.append(
                        f"{abs(nu)} {self.component_system.labels[i]}(l)"
                    )
            elif nu > 0:
                if nu == 1:
                    products.append(f"{self.component_system.labels[i]}(l)")
                else:
                    products.append(
                        f"{nu} {self.component_system.labels[i]}(l)"
                    )
        for i, nu in enumerate(self.stoich_solid):
            if nu < 0:
                if nu == - 1:
                    educts.append(f"{self.component_system.labels[i]}(s)")
                else:
                    educts.append(
                        f"{abs(nu)} {self.component_system.labels[i]}(s)"
                    )
            elif nu > 0:
                if nu == 1:
                    products.append(f"{self.component_system.labels[i]}(s)")
                else:
                    products.append(
                        f"{nu} {self.component_system.labels[i]}(s)"
                    )

        if self.is_kinetic:
            reaction_operator = f' <=>[{self.k_fwd:.2E}][{self.k_bwd:.2E}] '
        else:
            reaction_operator = f' <=>[{self.k_eq:.2E}] '

        return " + ".join(educts) + reaction_operator + " + ".join(products)


class ReactionBaseClass(metaclass=StructMeta):
    """Abstract base class for parameters of reaction models.

    Attributes
    ----------
    n_comp : UnsignedInteger
        number of components.
    parameters : dict
        dict with parameter values.
    name : String
        name of the reaction model.

    """
    name = String()
    n_comp = UnsignedInteger()

    _parameter_names = []

    def __init__(self, component_system, name=None):
        self.component_system = component_system
        self.name = name

        self._parameters = {
            param: getattr(self, param)
            for param in self._parameter_names
        }

    @property
    def model(self):
        return self.__class__.__name__

    @property
    def component_system(self):
        """ComponentSystem: Component System"""
        return self._component_system

    @component_system.setter
    def component_system(self, component_system):
        if not isinstance(component_system, ComponentSystem):
            raise TypeError('Expected ComponentSystem')
        self._component_system = component_system

    @property
    def n_comp(self):
        """int: Number of components."""
        return self.component_system.n_comp

    @property
    def parameters(self):
        """dict: Dictionary with parameter values."""
        return {param: getattr(self, param) for param in self._parameter_names}

    @parameters.setter
    def parameters(self, parameters):
        for param, value in parameters.items():
            if param not in self._parameter_names:
                raise CADETProcessError('Not a valid parameter')
            setattr(self, param, value)

    def __repr__(self):
        return \
            f'{self.__class__.__name__}(' \
            f'n_comp={self.n_comp}, name={self.name}' \
            f')'

    def __str__(self):
        if self.name is None:
            return self.__class__.__name__
        return self.name


class NoReaction(ReactionBaseClass):
    """Dummy class for units that do not experience reaction behavior.

    The number of components is set to zero for this class.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(ComponentSystem(), name='NoReaction')


class MassActionLaw(ReactionBaseClass):
    """Parameters for Reaction in Bulk Phase."""

    _parameter_names = ReactionBaseClass._parameter_names + [
        'stoich', 'exponents_fwd', 'exponents_bwd', 'k_fwd', 'k_bwd'
    ]

    def __init__(self, *args, **kwargs):
        self._reactions = []

        super().__init__(*args, **kwargs)

    @wraps(Reaction.__init__)
    def add_reaction(self, *args, **kwargs):
        """Add reaction to ReactionSystem."""
        r = Reaction(self.component_system, *args, **kwargs)
        self._reactions.append(r)

    @property
    def reactions(self):
        """list: Reactions in ReactionSystem."""
        return self._reactions

    @property
    def n_reactions(self):
        """int: Number of Reactions."""
        return len(self.reactions)

    @property
    def stoich(self):
        """np.array: Stoichiometric coefficients of reactions."""
        stoich = np.zeros((self.n_comp, self.n_reactions))

        for i, r in enumerate(self.reactions):
            stoich[:, i] = r.stoich

        return stoich

    @property
    def exponents_fwd(self):
        """np.array: Exponents of forward reactions."""
        exponents = np.zeros((self.n_comp, self.n_reactions))

        for i, r in enumerate(self.reactions):
            exponents[:, i] = r.exponents_fwd

        return exponents

    @property
    def exponents_bwd(self):
        """np.array: Exponents of backward reactions."""
        exponents = np.zeros((self.n_comp, self.n_reactions))

        for i, r in enumerate(self.reactions):
            exponents[:, i] = r.exponents_bwd

        return exponents

    @property
    def k_fwd(self):
        """list: Forward reaction rates."""
        return [r.k_fwd for r in self.reactions]

    @property
    def k_bwd(self):
        """list: Backward reaction rates."""
        return [r.k_bwd for r in self.reactions]

    @property
    def k_eq(self):
        """list: Equilibrium constants."""
        return [r.k_eq for r in self.reactions]


class MassActionLawParticle(ReactionBaseClass):
    """Parameters for Reaction in Particle Phase."""

    _parameter_names = ReactionBaseClass._parameter_names + [
        'stoich_liquid', 'exponents_fwd_liquid', 'exponents_bwd_liquid',
        'k_fwd_liquid', 'k_bwd_liquid',
        'exponents_fwd_liquid_modsolid', 'exponents_bwd_liquid_modsolid',
        'stoich_solid', 'exponents_fwd_solid', 'exponents_bwd_solid',
        'k_fwd_solid', 'k_bwd_solid',
        'exponents_fwd_solid_modliquid', 'exponents_bwd_solid_modliquid'
    ]

    def __init__(self, *args, **kwargs):
        self._liquid_reactions = []
        self._solid_reactions = []
        self._cross_phase_reactions = []

        super().__init__(*args, **kwargs)

    @wraps(Reaction.__init__)
    def add_liquid_reaction(self, *args, **kwargs):
        """Add liquid phase to ReactionSystem."""
        r = Reaction(self.component_system, *args, **kwargs)
        self._liquid_reactions.append(r)

    @wraps(Reaction.__init__)
    def add_solid_reaction(self, *args, **kwargs):
        """Add solid phase to ReactionSystem."""
        r = Reaction(self.component_system, *args, **kwargs)
        self._solid_reactions.append(r)

    @wraps(CrossPhaseReaction.__init__)
    def add_cross_phase_reaction(self, *args, **kwargs):
        """Add cross phase to ReactionSystem."""
        r = CrossPhaseReaction(self.component_system, *args, **kwargs)
        self._cross_phase_reactions.append(r)

    # Pore Liquid
    @property
    def liquid_reactions(self):
        """list: Liquid phase Reactions."""
        return self._liquid_reactions + self.cross_phase_reactions

    @property
    def n_liquid_reactions(self):
        """int: Number of liquid phase Reactions."""
        return len(self.liquid_reactions)

    @property
    def stoich_liquid(self):
        """int: Number of liquid phase Reactions."""
        stoich_liquid = np.zeros((self.n_comp, self.n_liquid_reactions))

        for i, r in enumerate(self.liquid_reactions):
            if isinstance(r, CrossPhaseReaction):
                stoich_liquid[:, i] = r.stoich_liquid
            else:
                stoich_liquid[:, i] = r.stoich

        return stoich_liquid

    @property
    def exponents_fwd_liquid(self):
        """np.array: Forward reaction exponents of liquid phase Reactions."""
        exponents = np.zeros((self.n_comp, self.n_liquid_reactions))

        for i, r in enumerate(self.liquid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:, i] = r.exponents_fwd_liquid
            else:
                exponents[:, i] = r.exponents_fwd

        return exponents

    @property
    def exponents_bwd_liquid(self):
        """np.array: Backward reaction exponents of liquid phase Reactions."""
        exponents = np.zeros((self.n_comp, self.n_liquid_reactions))

        for i, r in enumerate(self.liquid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:, i] = r.exponents_bwd_liquid
            else:
                exponents[:, i] = r.exponents_bwd

        return exponents

    @property
    def k_fwd_liquid(self):
        """list: Forward reaction rates of liquid phase Reactions."""
        return [r.k_fwd for r in self.liquid_reactions]

    @property
    def k_bwd_liquid(self):
        """list: Backward reaction rates of liquid phase Reactions."""
        return [r.k_bwd for r in self.liquid_reactions]

    @property
    def k_eq_liquid(self):
        """list: Equilibrium constants of liquid phase Reactions."""
        return [r.k_eq for r in self.liquid_reactions]

    # Solid
    @property
    def solid_reactions(self):
        """list: Solid phase Reactions."""
        return self._solid_reactions + self.cross_phase_reactions

    @property
    def n_solid_reactions(self):
        """int: Number of solid phase Reactions."""
        return len(self.solid_reactions)

    @property
    def stoich_solid(self):
        stoich_solid = np.zeros((self.n_comp, self.n_solid_reactions))

        for i, r in enumerate(self.solid_reactions):
            if isinstance(r, CrossPhaseReaction):
                stoich_solid[:, i] = r.stoich_solid
            else:
                stoich_solid[:, i] = r.stoich

        return stoich_solid

    @property
    def exponents_fwd_solid(self):
        """np.array: Forward reaction exponents of solid phase Reactions."""
        exponents = np.zeros((self.n_comp, self.n_solid_reactions))

        for i, r in enumerate(self.solid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:, i] = r.exponents_fwd_solid
            else:
                exponents[:, i] = r.exponents_fwd

        return exponents

    @property
    def exponents_bwd_solid(self):
        """np.array: Backward reaction exponents of solid phase Reactions."""
        exponents = np.zeros((self.n_comp, self.n_solid_reactions))

        for i, r in enumerate(self.solid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:, i] = r.exponents_bwd_solid
            else:
                exponents[:, i] = r.exponents_bwd

        return exponents

    @property
    def k_fwd_solid(self):
        """list: Forward reaction rates of solid phase Reactions."""
        return [r.k_fwd for r in self.solid_reactions]

    @property
    def k_bwd_solid(self):
        """list: Backward reaction rates of solid phase Reactions."""
        return [r.k_bwd for r in self.solid_reactions]

    @property
    def k_eq_solid(self):
        """list: Equilibrium constants of solid phase Reactions."""
        return [r.k_eq for r in self.solid_reactions]

    # Cross Phase
    @property
    def cross_phase_reactions(self):
        """list: Cross phase reactions."""
        return self._cross_phase_reactions

    @property
    def n_cross_phase_reactions(self):
        """int: Number of cross phase Reactions."""
        return len(self.cross_phase_reactions)

    @property
    def exponents_fwd_liquid_modsolid(self):
        """
        np.array: Forward reaction solid phase modifiers of liquid phase Reactions.
        """
        if self.n_cross_phase_reactions == 0:
            return Dict()

        liquid_fwd_modsolid = np.zeros((self.n_comp, self.n_liquid_reactions))

        for i, r in enumerate(self.liquid_reactions):
            if isinstance(r, CrossPhaseReaction):
                liquid_fwd_modsolid[:, i] = r.exponents_fwd_liquid_modsolid

        return liquid_fwd_modsolid

    @property
    def exponents_bwd_liquid_modsolid(self):
        """
        np.array: Backward reaction solid phase modifiers of liquid phase Reactions.
        """
        if self.n_cross_phase_reactions == 0:
            return Dict()

        liquid_bwd_modsolid = np.zeros((self.n_comp, self.n_liquid_reactions))

        for i, r in enumerate(self.liquid_reactions):
            if isinstance(r, CrossPhaseReaction):
                liquid_bwd_modsolid[:, i] = r.exponents_bwd_liquid_modsolid

        return liquid_bwd_modsolid

    @property
    def exponents_fwd_solid_modliquid(self):
        """
        np.array: Forward reaction liquid phase modifiers of solid phase Reactions.
        """
        if self.n_cross_phase_reactions == 0:
            return Dict()

        exponents = np.zeros((self.n_comp, self.n_solid_reactions))

        for i, r in enumerate(self.solid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:, i] = r.exponents_fwd_solid_modliquid

        return exponents

    @property
    def exponents_bwd_solid_modliquid(self):
        """
        np.array: Backward reaction liquid phase modifiers of solid phase Reactions.
        """
        if self.n_cross_phase_reactions == 0:
            return Dict()

        exponents = np.zeros((self.n_comp, self.n_solid_reactions))

        for i, r in enumerate(self.solid_reactions):
            if isinstance(r, CrossPhaseReaction):
                exponents[:, i] = r.exponents_bwd_solid_modliquid

        return exponents


def scale_to_rapid_equilibrium(k_eq, k_fwd_min=10):
    """Scale forward and backward reaction rates if only k_eq is known.

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
    k_bwd = k_fwd_min/k_eq

    return k_fwd, k_bwd
