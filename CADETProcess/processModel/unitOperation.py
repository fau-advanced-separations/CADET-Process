from abc import abstractproperty
import math
from warnings import warn

from CADETProcess import CADETProcessError

from CADETProcess.dataStructure import frozen_attributes
from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import (
    Constant, UnsignedFloat,
    String, Switch,
    DependentlySizedUnsignedList,
    Polynomial, NdPolynomial, DependentlySizedList
)

from .componentSystem import ComponentSystem
from .binding import BindingBaseClass, NoBinding
from .reaction import ReactionBaseClass, NoReaction
from .discretization import (
    DiscretizationParametersBase, NoDiscretization,
    LRMDiscretizationFV, LRMDiscretizationDG,
    LRMPDiscretizationFV, LRMPDiscretizationDG,
    GRMDiscretizationFV, GRMDiscretizationDG
)

from .solutionRecorder import (
    IORecorder,
    TubularReactorRecorder, LRMRecorder, LRMPRecorder, GRMRecorder, CSTRRecorder
)


__all__ = [
    'UnitBaseClass',
    'SourceMixin',
    'SinkMixin',
    'Inlet',
    'Outlet',
    'TubularReactorBase',
    'TubularReactor',
    'LumpedRateModelWithoutPores',
    'LumpedRateModelWithPores',
    'GeneralRateModel',
]


@frozen_attributes
class UnitBaseClass(metaclass=StructMeta):
    """Base class for all UnitOperation classes.

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

    _parameter_names = []
    _section_dependent_parameters = []
    _polynomial_parameters = []
    _initial_state = []
    _required_parameters = []

    supports_bulk_reaction = False
    supports_particle_reaction = False
    discretization_schemes = ()

    def __init__(self, component_system, name):
        self.name = name
        self.component_system = component_system

        self.binding_model = NoBinding()

        self.bulk_reaction_model = NoReaction()
        self.particle_reaction_model = NoReaction()

        self.discretization = NoDiscretization()

        self.solution_recorder = IORecorder()

        self._parameters = {
            param: getattr(self, param)
            for param in self._parameter_names
        }

    @property
    def model(self):
        return self.__class__.__name__

    @property
    def component_system(self):
        return self._component_system

    @component_system.setter
    def component_system(self, component_system):
        if not isinstance(component_system, ComponentSystem):
            raise TypeError('Expected ComponentSystem')
        self._component_system = component_system

    @property
    def discretization(self):
        return self._discretization

    @discretization.setter
    def discretization(self, discretization):
        if not isinstance(discretization, DiscretizationParametersBase):
            raise TypeError('Expected DiscretizationParametersBase')

        if not isinstance(discretization, NoDiscretization):
            if not isinstance(discretization, self.discretization_schemes):
                raise CADETProcessError(
                    f'Unit does not support {type(discretization)}.'
                )

        self._discretization = discretization

    @property
    def n_comp(self):
        return self.component_system.n_comp

    @property
    def parameters(self):
        """dict: Dictionary with parameter values.
        """
        parameters = self._parameters

        if not isinstance(self.binding_model, NoBinding):
            parameters['binding_model'] = self.binding_model.parameters
        if not isinstance(self.bulk_reaction_model, NoReaction):
            parameters['bulk_reaction_model'] = self.bulk_reaction_model.parameters
        if not isinstance(self.particle_reaction_model, NoReaction):
            parameters['particle_reaction_model'] = \
                self.particle_reaction_model.parameters
        if not isinstance(self.discretization, NoDiscretization):
            parameters['discretization'] = \
                self.discretization.parameters

        return parameters

    @parameters.setter
    def parameters(self, parameters):
        try:
            self.binding_model.parameters = parameters.pop('binding_model')
        except KeyError:
            pass
        try:
            self.bulk_reaction_model.parameters = parameters.pop(
                'bulk_reaction_model'
            )
        except KeyError:
            pass
        try:
            self.particle_reaction_model.parameters = parameters.pop(
                'particle_reaction_model'
            )
        except KeyError:
            pass
        try:
            self.discretization.parameters = parameters.pop('discretization')
        except KeyError:
            pass

        for param, value in parameters.items():
            if param not in self._parameters:
                raise CADETProcessError('Not a valid parameter')
            if value is not None:
                setattr(self, param, value)

    @property
    def section_dependent_parameters(self):
        parameters = {
            key: value for key, value in self.parameters.items()
            if key in self._section_dependent_parameters
        }

        return parameters

    @property
    def polynomial_parameters(self):
        parameters = {
            key: value for key, value in self.parameters.items()
            if key in self._polynomial_parameters
        }
        return parameters

    @property
    def required_parameters(self):
        return self._required_parameters

    @property
    def initial_state(self):
        """dict: Dictionary with initial states.
        """
        initial_state = {st: getattr(self, st) for st in self._initial_state}

        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        for st, value in initial_state.items():
            if st not in self._initial_state:
                raise CADETProcessError('Not a valid parameter')
            if value is not None:
                setattr(self, st, value)

    @property
    def missing_parameters(self):
        missing_parameters = []
        for param in self._required_parameters:
            if getattr(self, param) is None:
                missing_parameters.append(param)

        missing_parameters += [
            f'binding_model.{param}' for param in self.binding_model.missing_parameters
        ]

        return missing_parameters

    def check_required_parameters(self):
        if len(self.missing_parameters) == 0:
            return True
        else:
            for param in self.missing_parameters:
                warn(f'Unit {self.name}: Missing parameter "{param}".')
            return False

    @property
    def binding_model(self):
        """binding_model: BindingModel of the unit operation.

        Raises
        ------
        TypeError
            If binding_model object is not an instance of BindingBaseClass.
        CADETProcessError
            If number of components do not match.

        """
        return self._binding_model

    @binding_model.setter
    def binding_model(self, binding_model):
        if not isinstance(binding_model, BindingBaseClass):
            raise TypeError('Expected BindingBaseClass')

        if binding_model.component_system is not self.component_system \
                and not isinstance(binding_model, NoBinding):
            raise CADETProcessError('Component systems do not match.')

        self._binding_model = binding_model

    @property
    def n_bound_states(self):
        return self.binding_model.n_bound_states

    @property
    def bulk_reaction_model(self):
        """bulk_reaction_model: Reaction in bulk phase.

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
    def bulk_reaction_model(self, bulk_reaction_model):
        if not isinstance(bulk_reaction_model, ReactionBaseClass):
            raise TypeError('Expected ReactionBaseClass')

        if not isinstance(bulk_reaction_model, NoReaction):
            if not self.supports_bulk_reaction:
                raise CADETProcessError(
                    'Unit does not support bulk reactions.'
                )
            if bulk_reaction_model.component_system \
                    is not self.component_system:
                raise CADETProcessError('Component systems do not match.')

        self._bulk_reaction_model = bulk_reaction_model

    @property
    def particle_reaction_model(self):
        """particle_liquid_reaction_model: Reaction in particle liquid phase.

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
    def particle_reaction_model(self, particle_reaction_model):
        if not isinstance(particle_reaction_model, ReactionBaseClass):
            raise TypeError('Expected ReactionBaseClass')

        if not isinstance(particle_reaction_model, NoReaction):
            if not self.supports_particle_reaction:
                raise CADETProcessError(
                    'Unit does not support particle reactions.'
                )
            if particle_reaction_model.component_system \
                    is not self.component_system:
                raise CADETProcessError('Component systems do not match.')

        self._particle_reaction_model = particle_reaction_model

    def __repr__(self):
        """str: String-representation of the object."""
        return \
            f'{self.__class__.__name__}' \
            f'(n_comp={self.n_comp}, name={self.name})'

    def __str__(self):
        """str: String-representation of the object."""
        return self.name


class SourceMixin(metaclass=StructMeta):
    """Mixin class for Units that have Source-like behavior

    See Also
    --------
    SinkMixin
    Inlet
    Cstr

    """
    _n_poly_coeffs = 4
    flow_rate = Polynomial(dep=('_n_poly_coeffs'))
    _parameter_names = ['flow_rate']
    _section_dependent_parameters = ['flow_rate']
    _polynomial_parameters = ['flow_rate']


class SinkMixin():
    """Mixin class for Units that have Sink-like behavior

    See Also
    --------
    SourceMixin
    Cstr

    """
    pass


class Inlet(UnitBaseClass, SourceMixin):
    """Pseudo unit operation model for streams entering the system.

    Attributes
    ----------
    c : NdPolynomial
        Polynomial coefficients for component concentration.
    flow_rate : NdPolynomial
        Polynomial coefficients for volumetric flow rate.
    solution_recorder : IORecorder
        Solution recorder for the unit operation.

    """

    c = NdPolynomial(dep=('n_comp', '_n_poly_coeffs'), default=0)
    flow_rate = Polynomial(dep=('_n_poly_coeffs'), default=0)
    _n_poly_coeffs = 4
    _parameter_names = \
        UnitBaseClass._parameter_names + \
        SourceMixin._parameter_names + \
        ['c']
    _section_dependent_parameters = \
        UnitBaseClass._section_dependent_parameters + \
        SourceMixin._section_dependent_parameters + \
        ['c']
    _polynomial_parameters = \
        UnitBaseClass._polynomial_parameters + \
        SourceMixin._polynomial_parameters + \
        ['c']
    _required_parameters = ['flow_rate']


Source = Inlet


class Outlet(UnitBaseClass, SinkMixin):
    """Pseudo unit operation model for streams leaving the system.

    Attributes
    ----------
    solution_recorder : IORecorder
        Solution recorder for the unit operation.
    """

    pass


Sink = Outlet


class MixerSplitter(UnitBaseClass):
    """Pseudo unit operation for mixing/splitting streams in the system."""
    pass


class TubularReactorBase(UnitBaseClass):
    """Base class for tubular reactors and chromatographic columns.

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
    axial_dispersion : UnsignedFloat
        Dispersion rate of compnents in axial direction.
    flow_direction : Switch
        If 1: Forward flow.
        If -1: Backwards flow.
    discretization : DiscretizationParametersBase
        Discretization scheme of the unit.

    """

    length = UnsignedFloat()
    diameter = UnsignedFloat()
    axial_dispersion = UnsignedFloat()
    flow_direction = Switch(valid=[-1, 1], default=1)
    _initial_state = UnitBaseClass._initial_state + ['c']
    _parameter_names = UnitBaseClass._parameter_names + [
        'length', 'diameter',
        'axial_dispersion', 'flow_direction'
    ]
    _section_dependent_parameters = \
        UnitBaseClass._section_dependent_parameters + \
        ['axial_dispersion', 'flow_direction']
    _required_parameters = ['length', 'axial_dispersion', 'diameter']

    @abstractproperty
    def total_porosity(self):
        pass

    @property
    def cross_section_area(self):
        """float: Cross section area of a Column.

        See Also
        --------
        volume
        cross_section_area_interstitial
        cross_section_area_liquid
        cross_section_area_solid

        """
        if self.diameter is not None:
            return math.pi/4 * self.diameter**2

    @cross_section_area.setter
    def cross_section_area(self, cross_section_area):
        self.diameter = (4*cross_section_area/math.pi)**0.5

    def set_diameter_from_interstitial_velicity(self, Q, u0):
        """Set diamter from flow rate and interstitial velocity.

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
        self.cross_section_area = Q/(u0*self.total_porosity)

    @property
    def cross_section_area_interstitial(self):
        """float: Interstitial area between particles.

        Notes
        -----
            Needs to be overwritten depending on the model porosities!

        See Also
        --------
        cross_section_area
        cross_section_area_liquid
        cross_section_area_solid

        """
        return self.total_porosity * self.cross_section_area

    @property
    def cross_section_area_liquid(self):
        """float: Liquid fraction of column cross section area.

        See Also
        --------
        cross_section_area
        cross_section_area_interstitial
        cross_section_area_solid
        volume

        """
        return self.total_porosity * self.cross_section_area

    @property
    def cross_section_area_solid(self):
        """float: Liquid fraction of column cross section area.

        See Also
        --------
        cross_section_area
        cross_section_area_interstitial
        cross_section_area_liquid

        """
        return (1 - self.total_porosity) * self.cross_section_area

    @property
    def volume(self):
        """float: Volume of the TubularReactor.

        See Also
        --------
        cross_section_area

        """
        return self.cross_section_area * self.length

    @property
    def volume_interstitial(self):
        """float: Interstitial volume between particles.

        See Also
        --------
        cross_section_area

        """
        return self.cross_section_area_interstitial * self.length

    @property
    def volume_liquid(self):
        """float: Volume of the liquid phase."""
        return self.cross_section_area_liquid * self.length

    @property
    def volume_solid(self):
        """float: Volume of the solid phase."""
        return self.cross_section_area_solid * self.length

    def t0(self, flow_rate):
        """Mean residence time of a (non adsorbing) volume element.

        Parameters
        ----------
        flow_rate : float
            volumetric flow rate

        Returns
        -------
        t0 : float
            Mean residence time

        See Also
        --------
        u0

        """
        return self.volume_interstitial / flow_rate

    def u0(self, flow_rate):
        """Flow velocity of a (non adsorbing) volume element.

        Parameters
        ----------
        flow_rate : float
            volumetric flow rate

        Returns
        -------
        u0 : float
            interstitial flow velocity

        See Also
        --------
        t0
        NTP

        """
        return self.length/self.t0(flow_rate)

    def NTP(self, flow_rate):
        r"""Number of theoretical plates.

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
        return self.u0(flow_rate) * self.length / (2 * self.axial_dispersion)

    def set_axial_dispersion_from_NTP(self, NTP, flow_rate):
        r"""Set axial dispersion from number of theoretical plates (NTP).

        Parameters
        ----------
        NTP : float
            Number of theroetical plates
        flow_rate : float
            volumetric flow rate

        Calculated using the axial dispersion coefficient:

        .. math::
            NTP = \frac{u \cdot L_{Column}}{2 \cdot D_a}

        Returns
        -------
        NTP : float
            Number of theretical plates

        See Also
        --------
        u0
        NTP

        """
        self.axial_dispersion = self.u0(flow_rate) * self.length / (2 * NTP)


class TubularReactor(TubularReactorBase):
    """Class for tubular reactors and tubing.

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

    c = DependentlySizedList(dep='n_comp', default=0)
    _initial_state = UnitBaseClass._initial_state + ['c']
    _parameter_names = TubularReactorBase._parameter_names + _initial_state

    def __init__(self, *args, discretization_scheme='FV', **kwargs):
        super().__init__(*args, **kwargs)

        if discretization_scheme == 'FV':
            self.discretization = LRMDiscretizationFV()
        elif discretization_scheme == 'DG':
            self.discretization = LRMDiscretizationDG()

        self.solution_recorder = TubularReactorRecorder()


class LumpedRateModelWithoutPores(TubularReactorBase):
    """Parameters for a lumped rate model without pores.

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
    supports_bulk_reaction = False
    supports_particle_reaction = True
    discretization_schemes = (LRMDiscretizationFV, LRMDiscretizationDG)

    total_porosity = UnsignedFloat(ub=1)
    _parameter_names = TubularReactorBase._parameter_names + [
        'total_porosity'
    ]
    _required_parameters = TubularReactorBase._required_parameters + ['total_porosity']

    c = DependentlySizedList(dep='n_comp', default=0)
    _q = DependentlySizedUnsignedList(dep='n_bound_states', default=0)
    _initial_state = TubularReactorBase._initial_state + ['q']
    _parameter_names = _parameter_names + _initial_state

    def __init__(self, *args, discretization_scheme='FV', **kwargs):
        super().__init__(*args, **kwargs)

        if discretization_scheme == 'FV':
            self.discretization = LRMDiscretizationFV()
        elif discretization_scheme == 'DG':
            self.discretization = LRMDiscretizationDG()

        self.solution_recorder = LRMRecorder()

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        if isinstance(self.binding_model, NoBinding):
            raise CADETProcessError("Cannot set q without binding model.")
        self._q = q

        self._parameters['q'] = q


class LumpedRateModelWithPores(TubularReactorBase):
    """Parameters for the lumped rate model with pores.

    Attributes
    ----------
    bed_porosity : UnsignedFloat between 0 and 1.
        Porosity of the bed
    particle_porosity : UnsignedFloat between 0 and 1.
        Porosity of particles.
    particle_radius : UnsignedFloat
        Radius of the particles.
    film_diffusion : List of unsigned floats. Length depends on n_comp.
        Diffusion rate for components in pore volume.
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
    supports_bulk_reaction = True
    supports_particle_reaction = True
    discretization_schemes = (LRMPDiscretizationFV, LRMPDiscretizationDG)

    bed_porosity = UnsignedFloat(ub=1)
    particle_porosity = UnsignedFloat(ub=1)
    particle_radius = UnsignedFloat()
    film_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    pore_accessibility = DependentlySizedUnsignedList(dep='n_comp')
    _parameter_names = TubularReactorBase._parameter_names + [
            'bed_porosity', 'particle_porosity', 'particle_radius',
            'film_diffusion'
            ]
    _section_dependent_parameters = \
        TubularReactorBase._section_dependent_parameters + ['film_diffusion']
    _required_parameters = TubularReactorBase._required_parameters + [
        'bed_porosity', 'particle_porosity', 'particle_radius', 'film_diffusion'
    ]

    c = DependentlySizedList(dep='n_comp', default=0)
    _cp = DependentlySizedUnsignedList(dep='n_comp')
    _q = DependentlySizedUnsignedList(dep='n_bound_states', default=0)

    _initial_state = TubularReactorBase._initial_state + ['cp', 'q']
    _parameter_names = _parameter_names + _initial_state

    def __init__(self, *args, discretization_scheme='FV', **kwargs):
        super().__init__(*args, **kwargs)

        if discretization_scheme == 'FV':
            self.discretization = LRMPDiscretizationFV()
        elif discretization_scheme == 'DG':
            self.discretization = LRMPDiscretizationDG()

        self.solution_recorder = LRMPRecorder()

    @property
    def total_porosity(self):
        """float: Total porosity of the column
        """
        return self.bed_porosity + \
            (1 - self.bed_porosity) * self.particle_porosity

    @property
    def cross_section_area_interstitial(self):
        """float: Interstitial area between particles.

        See Also
        --------
        cross_section_area
        cross_section_area_liquid
        cross_section_area_solid

        """
        return self.bed_porosity * self.cross_section_area

    def set_diameter_from_interstitial_velicity(self, Q, u0):
        """Set diamter from flow rate and interstitial velocity.

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
        self.cross_section_area = Q/(u0*self.bed_porosity)

    @property
    def cp(self):
        if self._cp is None:
            return self.c
        else:
            return self._cp

    @cp.setter
    def cp(self, cp):
        self._cp = cp

        self._parameters['cp'] = cp

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        if isinstance(self.binding_model, NoBinding):
            raise CADETProcessError("Cannot set q without binding model.")
        self._q = q

        self._parameters['q'] = q


class GeneralRateModel(TubularReactorBase):
    """Parameters for the general rate model.

    Attributes
    ----------
    bed_porosity : UnsignedFloat between 0 and 1.
        Porosity of the bed
    particle_porosity : UnsignedFloat between 0 and 1.
        Porosity of particles.
    particle_radius : UnsignedFloat
        Radius of the particles.
    pore_diffusion : List of unsigned floats. Length depends on n_comp.
        Diffusion rate for components in pore volume.
    surface_diffusion : List of unsigned floats. Length depends on n_comp.
        Diffusion rate for components in adsrobed state.
    pore_accessibility : List of unsigned floats. Length depends on n_comp.
        Accessibility of pores for components.
    c : List of unsigned floats. Length depends on n_comp
        Initial concentration of the reactor.
    cp : List of unsigned floats. Length depends on n_comp
        Initial concentration of the pores
    q : List of unsigned floats. Length depends on n_comp
        Initial concntration of the bound phase.
    solution_recorder : GRMRecorder
        Solution recorder for the unit operation.

    """
    supports_bulk_reaction = True
    supports_particle_reaction = True
    discretization_schemes = (GRMDiscretizationFV, GRMDiscretizationDG)

    bed_porosity = UnsignedFloat(ub=1)
    particle_porosity = UnsignedFloat(ub=1)
    particle_radius = UnsignedFloat()
    film_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    pore_diffusion = DependentlySizedUnsignedList(dep='n_comp')
    _surface_diffusion = DependentlySizedUnsignedList(dep='n_bound_states')
    pore_accessibility = DependentlySizedUnsignedList(dep='n_comp')
    _parameter_names = \
        TubularReactorBase._parameter_names + \
        [
            'bed_porosity', 'particle_porosity', 'particle_radius',
            'film_diffusion', 'pore_diffusion', 'surface_diffusion'
        ]
    _section_dependent_parameters = \
        TubularReactorBase._section_dependent_parameters + \
        ['film_diffusion', 'pore_diffusion', 'surface_diffusion']
    _required_parameters = TubularReactorBase._required_parameters + [
        'bed_porosity', 'particle_porosity', 'particle_radius', 'film_diffusion',
        'pore_diffusion'
    ]

    c = DependentlySizedList(dep='n_comp', default=0)
    _cp = DependentlySizedUnsignedList(dep='n_comp')
    _q = DependentlySizedUnsignedList(dep='n_bound_states', default=0)

    _initial_state = TubularReactorBase._initial_state + ['cp', 'q']
    _parameter_names = _parameter_names + _initial_state

    def __init__(self, *args, discretization_scheme='FV', **kwargs):
        super().__init__(*args, **kwargs)

        if discretization_scheme == 'FV':
            self.discretization = GRMDiscretizationFV()
        elif discretization_scheme == 'DG':
            self.discretization = GRMDiscretizationDG()

        self.solution_recorder = GRMRecorder()

    @property
    def total_porosity(self):
        """float: Total porosity of the column
        """
        return self.bed_porosity + \
            (1 - self.bed_porosity) * self.particle_porosity

    @property
    def cross_section_area_interstitial(self):
        """float: Interstitial area between particles.

        See Also
        --------
        cross_section_area
        cross_section_area_liquid
        cross_section_area_solid

        """
        return self.bed_porosity * self.cross_section_area

    def set_diameter_from_interstitial_velicity(self, Q, u0):
        """Set diamter from flow rate and interstitial velocity.

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
        self.cross_section_area = Q/(u0*self.bed_porosity)

    @property
    def cp(self):
        if self._cp is None:
            return self.c
        else:
            return self._cp

    @cp.setter
    def cp(self, cp):
        self._cp = cp

        self._parameters['cp'] = cp

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        if isinstance(self.binding_model, NoBinding):
            raise CADETProcessError("Cannot set q without binding model.")
        self._q = q

        self._parameters['q'] = q

    @property
    def surface_diffusion(self):
        return self._surface_diffusion

    @surface_diffusion.setter
    def surface_diffusion(self, surface_diffusion):
        if isinstance(self.binding_model, NoBinding):
            raise CADETProcessError(
                "Cannot set surface diffusion without binding model."
            )
        self._surface_diffusion = surface_diffusion

        self._parameters['_surface_diffusion'] = surface_diffusion


class Cstr(UnitBaseClass, SourceMixin, SinkMixin):
    """Parameters for an ideal mixer.

    Parameters
    ----------
    c : List of unsigned floats. Length depends on n_comp
        Initial concentration of the reactor.
    q : List of unsigned floats. Length depends on n_comp
        Initial concentration of the bound phase.
    V : unsigned float
        Initial volume of the reactor.
    total_porosity : UnsignedFloat between 0 and 1.
        Total porosity of the column.
    flow_rate_filter: float:
        Flow rate of pure liquid without components to reduce volume.
    solution_recorder : CSTRRecorder
        Solution recorder for the unit operation.

    """
    supports_bulk_reaction = True
    supports_particle_reaction = True

    porosity = UnsignedFloat(ub=1, default=1)
    flow_rate_filter = UnsignedFloat(default=0)
    _parameter_names = \
        UnitBaseClass._parameter_names + \
        SourceMixin._parameter_names + \
        ['porosity', 'flow_rate_filter']
    _section_dependent_parameters = \
        UnitBaseClass._section_dependent_parameters + \
        SourceMixin._section_dependent_parameters + \
        ['flow_rate_filter']
    _polynomial_parameters = \
        UnitBaseClass._polynomial_parameters + \
        SourceMixin._polynomial_parameters + \
        ['flow_rate_filter']
    _required_parameters = UnitBaseClass._required_parameters + ['V']

    c = DependentlySizedList(dep='n_comp', default=0)
    _q = DependentlySizedUnsignedList(dep='n_bound_states', default=0)
    V = UnsignedFloat()
    _initial_state = \
        UnitBaseClass._initial_state + \
        ['c', 'q', 'V']
    _parameter_names = _parameter_names + _initial_state

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solution_recorder = CSTRRecorder()

    @property
    def volume(self):
        """float: Alias for volume."""
        return self.V

    @property
    def volume_liquid(self):
        """float: Volume of the liquid phase."""
        return self.porosity * self.V

    @property
    def volume_solid(self):
        """float: Volume of the solid phase."""
        return (1 - self.porosity) * self.V

    def t0(self, flow_rate):
        """Mean residence time of a (non adsorbing) volume element.

        Parameters
        ----------
        flow_rate : float
            volumetric flow rate

        Returns
        -------
        t0 : float
            Mean residence time

        See Also
        --------
        u0

        """
        return self.volume_liquid / flow_rate

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        if isinstance(self.binding_model, NoBinding):
            raise CADETProcessError("Cannot set q without binding model.")
        self._q = q

        self._parameters['q'] = q
