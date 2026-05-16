from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from addict import Dict

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import deprecated_alias

__all__ = ["ComponentSystem", "Component", "Species"]


@dataclass
class Species:
    """
    A specific chemical form of a component.

    Attributes
    ----------
    name : str
        Name of the species.
    charge : int
        Charge of the species. Default is 0.
    molar_mass : float, optional
        Molar mass in kg/mol.
    density : float, optional
        Pure-component density in kg/m³.
    """

    name: str
    charge: int = 0
    molar_mass: Optional[float] = None
    density: Optional[float] = None

    @property
    def molecular_weight(self) -> Optional[float]:
        """Deprecated. Use molar_mass."""
        warnings.warn(
            "`molecular_weight` is deprecated; use `molar_mass` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.molar_mass

    @molecular_weight.setter
    def molecular_weight(self, value: Optional[float]) -> None:
        warnings.warn(
            "`molecular_weight` is deprecated; use `molar_mass` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.molar_mass = value

    def __str__(self) -> str:  # noqa: D105
        return self.name


class Component:
    """
    A conserved chemical entity that may exist as multiple species.

    Parameters
    ----------
    name : str, optional
        Name of the component.
    species : str | list[str], optional
        Name(s) of the subspecies. If None the component name is used.
    charge : int | list[int | None], optional
        Charge(s) of the subspecies.
    molar_mass : float | list[float | None], optional
        Molar mass(es) of the subspecies in kg/mol.
    density : float | list[float | None], optional
        Density(ies) of the subspecies in kg/m³.

    See Also
    --------
    Species
    ComponentSystem
    """

    @deprecated_alias(molecular_weight="molar_mass")
    def __init__(
        self,
        name: str | None = None,
        species: str | list[str | None] = None,
        charge: int | list[int | None] = None,
        molar_mass: float | list[float | None] = None,
        density: float | list[float | None] = None,
    ) -> None:
        self.name = name
        self._species: list[Species] = []

        if species is None:
            self._add_species(name, charge, molar_mass, density)
        elif isinstance(species, str):
            self._add_species(species, charge, molar_mass, density)
        elif isinstance(species, list):
            if charge is None:
                charge = len(species) * [None]
            if molar_mass is None:
                molar_mass = len(species) * [None]
            if density is None:
                density = len(species) * [None]
            for i, spec in enumerate(species):
                self._add_species(spec, charge[i], molar_mass[i], density[i])
        else:
            raise CADETProcessError("Could not determine number of species")

    def _add_species(
        self,
        name: str | None,
        charge: int | None,
        molar_mass: float | None,
        density: float | None,
    ) -> Species:
        kwargs: dict[str, Any] = {}
        if charge is not None:
            kwargs["charge"] = charge
        if molar_mass is not None:
            kwargs["molar_mass"] = molar_mass
        if density is not None:
            kwargs["density"] = density
        species = Species(name, **kwargs)
        self._species.append(species)
        return species

    def add_species(
        self,
        species: str | Species,
        charge: int | None = None,
        molar_mass: float | None = None,
        density: float | None = None,
    ) -> Species:
        """
        Add a subspecies to the component.

        Parameters
        ----------
        species : str | Species
            Species or name of the species to add.
        charge : int, optional
            Charge of the species.
        molar_mass : float, optional
            Molar mass in kg/mol.
        density : float, optional
            Pure-component density in kg/m³.

        Returns
        -------
        Species
            The added species.
        """
        if isinstance(species, Species):
            self._species.append(species)
            return species
        return self._add_species(species, charge, molar_mass, density)

    @property
    def species(self) -> list[Species]:
        """list[Species]: The subspecies of the component."""
        return self._species

    @property
    def n_species(self) -> int:
        """int: Number of subspecies."""
        return len(self._species)

    @property
    def label(self) -> list[str]:
        """list[str]: Names of the subspecies."""
        return [s.name for s in self._species]

    @property
    def charge(self) -> list[int | None]:
        """list[int | None]: Charges of the subspecies."""
        return [s.charge for s in self._species]

    @property
    def molar_mass(self) -> list[float | None]:
        """list[float | None]: Molar masses of the subspecies in kg/mol."""
        return [s.molar_mass for s in self._species]

    @property
    def molecular_weight(self) -> list[float | None]:
        """Deprecated. Use molar_mass."""
        warnings.warn(
            "`molecular_weight` is deprecated; use `molar_mass` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.molar_mass

    @property
    def density(self) -> list[float | None]:
        """list[float | None]: Densities of the subspecies in kg/m³."""
        return [s.density for s in self._species]

    def __str__(self) -> str:  # noqa: D105
        return self.name if self.name is not None else ""

    def __iter__(self) -> Iterator[Species]:  # noqa: D105
        yield from self._species


class ComponentSystem:
    """
    An ordered collection of components defining the chemical system.

    Parameters
    ----------
    components : int | list[str | Component], optional
        Number of anonymous components, or an explicit list of names or
        Component instances.
    name : str, optional
        Name of the system.
    charges : list[int | None], optional
        Charges per component.
    molar_masses : list[float | None], optional
        Molar masses per component in kg/mol.
    densities : list[float | None], optional
        Densities per component in kg/m³.

    See Also
    --------
    Species
    Component
    """

    @deprecated_alias(molecular_weights="molar_masses")
    def __init__(
        self,
        components: int | list[str | Component | None] = None,
        name: str | None = None,
        charges: list[int | None] = None,
        molar_masses: list[float | None] = None,
        densities: list[float | None] = None,
    ) -> None:
        self.name = name
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
        if molar_masses is None:
            molar_masses = n_comp * [None]
        if densities is None:
            densities = n_comp * [None]

        for i, comp in enumerate(components):
            self.add_component(
                comp,
                charge=charges[i],
                molar_mass=molar_masses[i],
                density=densities[i],
            )

    @property
    def components(self) -> list[Component]:
        """list[Component]: Components in the system."""
        return self._components

    @property
    def components_dict(self) -> dict[str, Component]:
        """dict[str, Component]: Components indexed by name."""
        return {name: comp for name, comp in zip(self.names, self._components)}

    @property
    def n_components(self) -> int:
        """int: Number of components."""
        return len(self._components)

    @property
    def n_comp(self) -> int:
        """int: Total number of species."""
        return self.n_species

    @property
    def n_species(self) -> int:
        """int: Total number of species."""
        return sum(comp.n_species for comp in self._components)

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
            Component instance or name of the component to add.
        *args, **kwargs
            Passed to Component constructor when component is a string.
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
            Name or instance of the component to remove.
        """
        if isinstance(component, str):
            try:
                component = self.components_dict[component]
            except KeyError:
                raise CADETProcessError("Unknown Component.")

        if component not in self._components:
            raise CADETProcessError("Unknown Component.")

        self._components.remove(component)

    @property
    def indices(self) -> dict[str, list[int]]:
        """dict[str, list[int]]: Species indices per component name."""
        indices = defaultdict(list)
        index = 0
        for comp in self._components:
            for _ in comp.species:
                indices[comp.name].append(index)
                index += 1
        return Dict(indices)

    @property
    def species_indices(self) -> dict[str, int]:
        """dict[str, int]: Index per species name."""
        indices = Dict()
        index = 0
        for comp in self._components:
            for spec in comp.species:
                indices[spec.name] = index
                index += 1
        return indices

    @property
    def names(self) -> list[str]:
        """list[str]: Component names."""
        return [
            comp.name if comp.name is not None else str(i)
            for i, comp in enumerate(self._components)
        ]

    @property
    def species(self) -> list[str]:
        """list[str]: Species names in order."""
        result = []
        index = 0
        for comp in self._components:
            for label in comp.label:
                result.append(label if label is not None else str(index))
                index += 1
        return result

    @property
    def charges(self) -> list[int | None]:
        """list[int | None]: Charges per species."""
        return [charge for comp in self._components for charge in comp.charge]

    @property
    def molar_masses(self) -> list[float | None]:
        """list[float | None]: Molar masses per species in kg/mol."""
        return [mm for comp in self._components for mm in comp.molar_mass]

    @property
    def molecular_weights(self) -> list[float | None]:
        """Deprecated. Use molar_masses."""
        warnings.warn(
            "`molecular_weights` is deprecated; use `molar_masses` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.molar_masses

    @property
    def densities(self) -> list[float | None]:
        """list[float | None]: Densities per species in kg/m³."""
        return [density for comp in self._components for density in comp.density]

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}({self.names!r})"

    def __len__(self) -> int:  # noqa: D105
        return self.n_comp

    def __iter__(self) -> Iterator[Component]:  # noqa: D105
        yield from self._components

    def __getitem__(self, item: int) -> Component:  # noqa: D105
        return self._components[item]
