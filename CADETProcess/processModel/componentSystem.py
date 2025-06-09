from collections import defaultdict
from functools import wraps
from typing import Any, Iterator

from addict import Dict

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Integer, String, Structure, UnsignedFloat

__all__ = ["ComponentSystem", "Component", "Species"]


class Species(Structure):
    """
    Species class.

    Represent a species in a chemical system.

    Attributes
    ----------
    name : str
        The name of the species.
    charge : int, optional
        The charge of the species. Default is 0.
    molecular_weight : float
        The molecular weight of the species.
    density : float
        Density of the species.
    """

    name: String = String()
    charge: Integer = Integer(default=0)
    molecular_weight: UnsignedFloat = UnsignedFloat()
    density: UnsignedFloat = UnsignedFloat()

    def __str__(self) -> str:
        """str: String representation of the component."""
        return self.name


class Component(Structure):
    """
    Information about single component.

    A component can contain subspecies (e.g. differently charged variants).

    Attributes
    ----------
    name : str | None
        Name of the component.
    species : list[Species]
        List of Subspecies.
    n_species : int
        Number of Subspecies.
    label : list[str]
        Name of component (including species).
    charge : int | list[int | None]
        Charge of component (including species).
    molecular_weight : float | list[float | None]
        Molecular weight of component (including species).
    density : float | list[float | None]
        Density of component (including species).

    See Also
    --------
    Species
    ComponentSystem
    """

    name: String = String()

    def __init__(
        self,
        name: str | None = None,
        species: str | list[str | None] = None,
        charge: int | list[int | None] = None,
        molecular_weight: float | list[float | None] = None,
        density: float | list[float | None] = None,
    ) -> None:
        """
        Initialize Component.

        Parameters
        ----------
        name : str | None
            Name of the component.
        species : str | list[str | None]
            Name(s) of the subspecies to initialize. If None, the component's name is used.
        charge : int | list [int | None]
            Charges of the subspecies. Defaults to None for each species.
        molecular_weight : float | list[float | None]
            Molecular weights of the subspecies. Defaults to None for each species.
        density : float | list[float | None]
            Density of component (including species). Defaults to None for each species.
        """
        self.name: str | None = name
        self._species: list[Species] = []

        if species is None:
            self.add_species(name, charge, molecular_weight, density)
        elif isinstance(species, str):
            self.add_species(species, charge, molecular_weight, density)
        elif isinstance(species, list):
            if charge is None:
                charge = len(species) * [None]
            if molecular_weight is None:
                molecular_weight = len(species) * [None]
            if density is None:
                density = len(species) * [None]
            for i, spec in enumerate(species):
                self.add_species(spec, charge[i], molecular_weight[i], density[i])
        else:
            raise CADETProcessError("Could not determine number of species")

    @property
    def species(self) -> list[Species]:
        """list[Species]: The subspecies of the component."""
        return self._species

    @wraps(Species.__init__)
    def add_species(
        self,
        species: str | Species,
        *args: Any,
        **kwargs: Any,
    ) -> Species:
        """
        Add a subspecies to the component.

        Parameters
        ----------
        species: string | Species
            Species to add
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
        return species

    @property
    def n_species(self) -> int:
        """int: The number of subspecies in the component."""
        return len(self.species)

    @property
    def label(self) -> list[str]:
        """list[str]: The names of the subspecies."""
        return [spec.name for spec in self.species]

    @property
    def charge(self) -> list[int | None]:
        """list[int | None]: The charges of the subspecies."""
        return [spec.charge for spec in self.species]

    @property
    def molecular_weight(self) -> list[float | None]:
        """list[float | None]: The molecular weights of the subspecies."""
        return [spec.molecular_weight for spec in self.species]

    @property
    def density(self) -> list[float | None]:
        """list[float | None]: The density of the subspecies."""
        return [spec.density for spec in self.species]

    def __str__(self) -> str:
        """str: String representation of the component."""
        return self.name

    def __iter__(self) -> Iterator[Species]:
        """Iterate over the subspecies of the component."""
        yield from self.species


class ComponentSystem(Structure):
    """
    Information about components in system.

    A component can contain subspecies (e.g. differently charged variants).

    Attributes
    ----------
    name : String
        Name of the component system.
    components : list[Component]
        List of individual components.
    n_species : int
        Number of Subspecies.
    n_comp : int
        Number of all component species.
    n_components : int
        Number of components.
    indices : dict[str, list[int]]
        Component indices.
    names : list[str]
        Names of all components.
    species : list[str]
        Names of all component species.
    charges : list[int | None]
        Charges of all components species.
    molecular_weights : list[float | None]
        Molecular weights of all component species.
    densities : list[float | None]
        Densities of all component species.

    See Also
    --------
    Species
    Component
    """

    name: String = String()

    def __init__(
        self,
        components: int | list[str | Component | None] = None,
        name: str | None = None,
        charges: list[int | None] = None,
        molecular_weights: list[float | None] = None,
        densities: list[float | None] = None,
    ) -> None:
        """
        Initialize the ComponentSystem object.

        Parameters
        ----------
        components : int | list[str | Component | None]
            The number of components or the list of components to be added.
            If None, no components are added.
        name : str | None
            The name of the ComponentSystem.
        charges : list[int | None]
            The charges of each component.
        molecular_weights : list[float | None]
            The molecular weights of each component.
        densities : list[float | None]
            The densities of each component.

        Raises
        ------
        CADETProcessError
            If the `components` argument is neither an int nor a list.
        """
        self.name: str | None = name
        self._components: list[Component] = []

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
        if densities is None:
            densities = n_comp * [None]

        for i, comp in enumerate(components):
            self.add_component(
                comp,
                charge=charges[i],
                molecular_weight=molecular_weights[i],
                density=densities[i],
            )

    @property
    def components(self) -> list[Component]:
        """list[Component]: List of components in the system."""
        return self._components

    @property
    def components_dict(self) -> dict[str, Component]:
        """dict[str, Component]: Components indexed by name."""
        return {name: comp for name, comp in zip(self.names, self.components)}

    @property
    def n_components(self) -> int:
        """int: Number of components."""
        return len(self.components)

    @property
    def n_comp(self) -> int:
        """int: Number of species."""
        return self.n_species

    @property
    def n_species(self) -> int:
        """int: Number of species."""
        return sum([comp.n_species for comp in self.components])

    @wraps(Component.__init__)
    def add_component(
        self,
        component: str | Component,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Add a component to the system.

        Parameters
        ----------
        component : str | Component
            The component instance or name of the component to be added.
        *args : Any
            The positional arguments to be passed to the component class's constructor.
        **kwargs : Any
            The keyword arguments to be passed to the component class's constructor.
        """
        if not isinstance(component, Component):
            component = Component(component, *args, **kwargs)

        if component.name in self.names:
            raise CADETProcessError(
                f"Component '{component.name}' already exists in ComponentSystem."
            )

        self._components.append(component)

    def remove_component(self, component: str | Component) -> None:
        """
        Remove a component from the system.

        Parameters
        ----------
        component : str | Component
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
    def indices(self) -> dict[str, list[int]]:
        """dict[str, list[int]]: List of species indices for each component name."""
        indices = defaultdict(list)

        index = 0
        for comp in self.components:
            for spec in comp.species:
                indices[comp.name].append(index)
                index += 1

        return Dict(indices)

    @property
    def species_indices(self) -> dict[str, int]:
        """dict[str, int]: Indices for each species."""
        indices = Dict()

        index = 0
        for comp in self.components:
            for spec in comp.species:
                indices[spec.name] = index
                index += 1

        return indices

    @property
    def names(self) -> list[str]:
        """list[str]: List of component names."""
        names = [
            comp.name if comp.name is not None else str(i)
            for i, comp in enumerate(self.components)
        ]

        return names

    @property
    def species(self) -> list[str]:
        """list[str]: List of species names."""
        species = []
        index = 0
        for comp in self.components:
            for label in comp.label:
                if label is None:
                    species.append(str(index))
                else:
                    species.append(label)

                index += 1

        return species

    @property
    def charges(self) -> list[int | None]:
        """list[int | None]: List of species charges."""
        charges = []
        for comp in self.components:
            charges += comp.charge

        return charges

    @property
    def molecular_weights(self) -> list[float | None]:
        """list[float | None]: List of species molecular weights."""
        molecular_weights = []
        for comp in self.components:
            molecular_weights += comp.molecular_weight

        return molecular_weights

    @property
    def densities(self) -> list[float | None]:
        """list[float | None]: List of species densities."""
        densities = []
        for comp in self.components:
            densities += comp.density

        return densities

    def __repr__(self) -> str:
        """str: Return the string representation of the object."""
        return f"{self.__class__.__name__}({self.names})"

    def __len__(self) -> int:
        """int: Return the number of components in the system."""
        return self.n_comp

    def __iter__(self) -> Iterator[Component]:
        """Iterate over components in the system."""
        yield from self.components

    def __getitem__(self, item: int) -> Component:
        """Component: Retrieve a component by its index."""
        return self._components[item]
