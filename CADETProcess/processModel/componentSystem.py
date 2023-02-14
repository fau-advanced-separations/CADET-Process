from collections import defaultdict
from functools import wraps

from addict import Dict

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Structure, StructMeta
from CADETProcess.dataStructure import String, Integer, UnsignedFloat


__all__ = ['ComponentSystem', 'Component', 'Species']


class Species(Structure):
    """Species class.

    Represent a species in a chemical system.

    Attributes
    ----------
    name : str
        The name of the species.
    charge : int, optional
        The charge of the species. Default is 0.
    molecular_weight : float
        The molecular weight of the species.

    """
    name = String()
    charge = Integer(default=0)
    molecular_weight = UnsignedFloat()


class Component(metaclass=StructMeta):
    """Information about single component.

    A component can contain subspecies (e.g. differently charged variants).

    Attributes
    ----------
    name : String
        Name of the component.
    species : list
        List of Subspecies.
    n_species : int
        Number of Subspecies.
    label : list
        Name of component (including species).
    charge : list
        Charge of component (including species).
    molecular_weight : list
        Molecular weight of component (including species).

    See Also
    --------
    Species
    ComponentSystem

    """
    name = String()

    def __init__(
            self, name=None, species=None, charge=None, molecular_weight=None):
        """
        Parameters
        ----------
        name : str, optional
            Name of the component.
        species : str or list of str, optional
            Names of the subspecies.
        charge : int or list of int or None, optional
            Charges of the subspecies.
        molecular_weight : float or list of float or None, optional
            Molecular weights of the subspecies.
        """
        self.name = name
        self._species = []

        if species is None:
            self.add_species(name, charge, molecular_weight)
        elif isinstance(species, str):
            self.add_species(species, charge, molecular_weight)
        elif isinstance(species, list):
            if charge is None:
                charge = len(species) * [None]
            if molecular_weight is None:
                molecular_weight = len(species) * [None]
            for i, spec in enumerate(species):
                self.add_species(spec, charge[i], molecular_weight[i])
        else:
            raise CADETProcessError("Could not determine number of species")

    @property
    def species(self):
        """list: The subspecies of the component."""
        return self._species

    @wraps(Species.__init__)
    def add_species(self, species, *args, **kwargs):
        """
        Add a subspecies to the component.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        Species
            The subspecies that was added.
        """
        if not isinstance(species, Species):
            species = Species(species, *args, **kwargs)
        self._species.append(species)

    @property
    def n_species(self):
        """int: The number of subspecies in the component."""
        return len(self.species)

    @property
    def label(self):
        """list of str: The names of the subspecies."""
        return [spec.name for spec in self.species]

    @property
    def charge(self):
        """list of int or None: The charges of the subspecies."""
        return [spec.charge for spec in self.species]

    @property
    def molecular_weight(self):
        """list of float or None: The molecular weights of the subspecies."""
        return [spec.molecular_weight for spec in self.molecular_weight]

    def __str__(self):
        """String representation of the component."""
        return self.name

    def __iter__(self):
        """Iterate over the subspecies of the component."""
        yield from self.species


class ComponentSystem(metaclass=StructMeta):
    """Information about components in system.

    A component can contain subspecies (e.g. differently charged variants).

    Attributes
    ----------
    name : String
        Name of the component system.
    components : list
        List of individual components.
    n_species : int
        Number of Subspecies.
    n_comp : int
        Number of all components including species.
    n_components : int
        Number of components.
    indices : dict
        Component indices.
    names : list
        Names of all components.
    labels : list
        Labels of all components (including species).
    charge : list
        Charges of all components (including species).
    molecular_weight : list
        Molecular weights of all components (including species).

    See Also
    --------
    Species
    Component

    """
    name = String()

    def __init__(
            self, components=None, name=None, charges=None, molecular_weights=None):
        """Initialize the ComponentSystem object.

        Parameters
        ----------
        components : int, list, None
            The number of components or the list of components to be added.
            If None, no components are added.
        name : str, None
            The name of the ComponentSystem.
        charges : list, None
            The charges of each component.
        molecular_weights : list, None
            The molecular weights of each component.

        Raises
        ------
        CADETProcessError
            If the `components` argument is neither an int nor a list.

        """
        self.name = name

        self._components = []

        if components is None:
            return

        if isinstance(components, int):
            n_comp = components
            components = [str(i) for i in range(n_comp)]
        elif isinstance(components, list):
            n_comp = len(components)
        else:
            raise CADETProcessError("Could not determine number of components")

        if charges is None:
            charges = n_comp * [None]
        if molecular_weights is None:
            molecular_weights = n_comp * [None]

        for i, comp in enumerate(components):
            self.add_component(
                comp,
                charge=charges[i],
                molecular_weight=molecular_weights[i],
            )

    @property
    def components(self):
        """list: List of components in the system."""
        return self._components

    @property
    def components_dict(self):
        """dict: Components indexed by name."""
        return {
            name: comp
            for name, comp in zip(self.names, self.components)
        }

    @property
    def n_components(self):
        """int: Number of components."""
        return len(self.components)

    @property
    def n_comp(self):
        """int: Number of species."""
        return self.n_species

    @property
    def n_species(self):
        """int: Number of species."""
        return sum([comp.n_species for comp in self.components])

    @wraps(Component.__init__)
    def add_component(self, component, *args, **kwargs):
        """
        Add a component to the system.

        Parameters
        ----------
        component : {str, Component}
            The class of the component to be added.
        *args : list
            The positional arguments to be passed to the component class's constructor.
        **kwargs : dict
            The keyword arguments to be passed to the component class's constructor.

        """
        if not isinstance(component, Component):
            component = Component(component, *args, **kwargs)

        if component.name in self.names:
            raise CADETProcessError(
                f"Component '{component.name}' "
                "already exists in ComponentSystem."
            )

        self._components.append(component)

    def remove_component(self, component):
        """Remove a component from the system.

        Parameters
        ----------
        component : {str, Component}
            The name of the component or the component instance to be removed.

        Raises
        ------
        CADETProcessError
            If the component is unknown or not present in the system.

        """
        if isinstance(component, str):
            try:
                component = self.components_dict[component]
            except KeyError:
                raise CADETProcessError("Unknown Component.")

        if component not in self.components:
            raise CADETProcessError("Unknown Component.")

        self._components.remove(component)

    @property
    def indices(self):
        indices = defaultdict(list)

        index = 0
        for comp in self.components:
            for spec in comp.species:
                indices[comp.name].append(index)
                index += 1

        return Dict(indices)

    @property
    def names(self):
        """list: List of component names."""
        names = [
            comp.name if comp.name is not None else str(i)
            for i, comp in enumerate(self.components)
        ]

        return names

    @property
    def species(self):
        """list: List of species names."""
        return self.labels

    @property
    def labels(self):
        """list: List of species names."""
        labels = []
        index = 0
        for comp in self.components:
            for label in comp.label:
                if label is None:
                    labels.append(str(index))
                else:
                    labels.append(label)

                index += 1

        return labels

    @property
    def charges(self):
        """list: List of species charges."""
        charges = []
        for comp in self.components:
            charges += comp.charge

        return charges

    @property
    def molecular_weights(self):
        """list: List of species molecular weights."""
        molecular_weights = []
        for comp in self.components:
            molecular_weights += comp.molecular_weight

        return molecular_weights

    def __repr__(self):
        return f'{self.__class__.__name__}({self.names})'

    def __iter__(self):
        yield from self.components

    def __getitem__(self, item):
        return self._components[item]
