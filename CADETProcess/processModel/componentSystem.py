from collections import defaultdict

from addict import Dict
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Structure, StructMeta
from CADETProcess.dataStructure import String, Integer, UnsignedFloat

class ComponentSystem(metaclass=StructMeta):
    name = String()

    def __init__(
            self, components=None, name=None,
            charges=None, molecular_weights=None
            ):
        self.name = name

        self._components = []

        if components is None:
            return
        elif isinstance(components, int):
            n_comp = components
            components = n_comp*[None]
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
                species=comp,
                charge=charges[i],
                molecular_weight=molecular_weights[i]
            )

    @property
    def components(self):
        return self._components

    @property
    def n_components(self):
        return len(self.components)

    @property
    def n_comp(self):
        return sum([comp.n_species for comp in self.components])

    def add_component(self, *args, **kwargs):
        """Todo: check duplicates"""
        component = Component(*args, **kwargs)
        self._components.append(component)

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
        return [comp.name for comp in self.components]

    @property
    def labels(self):
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
        charges = []
        for comp in self.components:
            charges += comp.charge

        return charges

    @property
    def molecular_weights(self):
        molecular_weights = []
        for comp in self.components:
            molecular_weights += comp.molecular_weight

        return molecular_weights

    def total_concentration(self, solution):
        """Compute total concentration of components by summing up species.

        Parameters
        ----------
        solution : np.array
            Solution array.

        Returns
        -------
        total_concentration : np.array
            Total concentration of components.

        """
        total_concentration = np.zeros(
            solution.shape[0:-1] + (self.n_components,)
        )

        counter = 0
        for index, comp in enumerate(self.components):
            comp_indices = slice(counter, counter+comp.n_species)
            total_concentration[:,index] = np.sum(solution[...,comp_indices], axis=1)
            counter += comp.n_species

        return total_concentration
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.names})'


class Component(metaclass=StructMeta):
    name = String()

    def __init__(
            self, name=None, species=None, charge=None, molecular_weight=None
            ):
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
        return self._species

    def add_species(self, name, charge, molecular_weight):
        species = Species(name, charge, molecular_weight)
        self._species.append(species)

    @property
    def n_species(self):
        return len(self.species)

    @property
    def label(self):
        return [spec.name for spec in self.species]

    @property
    def charge(self):
        return [spec.charge for spec in self.species]

    @property
    def molecular_weight(self):
        return [spec.molecular_weight for spec in self.molecular_weight]

class Species(Structure):
    name = String()
    charge = Integer(default=0)
    molecular_weight = UnsignedFloat()
