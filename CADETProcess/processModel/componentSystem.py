from CADETProcess.dataStructure import Structure, StructMeta
from CADETProcess.dataStructure import List, String, DependentlySizedList

class ComponentSystem(metaclass=StructMeta):
    name = String()

    def __init__(self, name=None, n_comp=None):
        self.name = name
        self._components = []

        if n_comp is not None:
            for comp in range(n_comp):
                self.add_component()

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
        component = Component(*args, **kwargs)
        self._components.append(component)

    @property
    def labels(self):
        labels = []
        index = 0
        for comp in self.components:
            for label in comp.labels:
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
            charges += comp.charges

        return charges


class Component(Structure):
    name = String()
    species = List()
    charges = DependentlySizedList(dep='n_species')
    molecular_weight = DependentlySizedList(dep='n_species')

    @property
    def n_species(self):
        if self.species is None:
            return 1
        else:
            return len(self.species)

    @property
    def labels(self):
        if self.species is None:
            return [self.name]
        else:
            return self.species
