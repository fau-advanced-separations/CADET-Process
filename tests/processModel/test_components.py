import numpy as np
import pytest
from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem


@pytest.fixture
def anonymous_components():
    return ComponentSystem(2)


@pytest.fixture
def named_components():
    return ComponentSystem(["A", "B"])


@pytest.fixture
def multispecies_component():
    cs = ComponentSystem()
    cs.add_component("A")
    cs.add_component("B", species=["B+", "B-"])
    return cs


@pytest.fixture
def ionic_system():
    cs = ComponentSystem()
    cs.add_component("Ammonia", species=["NH4+", "NH3"], charge=[1, 0])
    cs.add_component("Lysine", ["Lys2+", "Lys+", "Lys", "Lys"], [2, 1, 0, -1])
    cs.add_component("H+", charge=1)
    return cs


@pytest.fixture
def mixed_components():
    cs = ComponentSystem(2)
    cs.add_component("manual_label")
    return cs


@pytest.fixture
def components_with_physical_properties():
    cs = ComponentSystem()
    cs.add_component("A", species=["A+", "A-"], molar_mass=[1, 0], density=[1, 0])
    return cs


def test_names(
    anonymous_components,
    named_components,
    multispecies_component,
    ionic_system,
    mixed_components,
):
    np.testing.assert_equal(anonymous_components.names, ["0", "1"])
    np.testing.assert_equal(named_components.names, ["A", "B"])
    np.testing.assert_equal(multispecies_component.names, ["A", "B"])
    np.testing.assert_equal(ionic_system.names, ["Ammonia", "Lysine", "H+"])
    np.testing.assert_equal(mixed_components.names, ["0", "1", "manual_label"])


def test_duplicate_name(named_components):
    with pytest.raises(CADETProcessError):
        named_components.add_component("A")


def test_species(
    anonymous_components,
    named_components,
    multispecies_component,
    ionic_system,
    mixed_components,
):
    np.testing.assert_equal(anonymous_components.species, ["0", "1"])
    np.testing.assert_equal(named_components.species, ["A", "B"])
    np.testing.assert_equal(multispecies_component.species, ["A", "B+", "B-"])
    np.testing.assert_equal(
        ionic_system.species, ["NH4+", "NH3", "Lys2+", "Lys+", "Lys", "Lys", "H+"]
    )
    np.testing.assert_equal(mixed_components.species, ["0", "1", "manual_label"])


def test_indices(ionic_system):
    expected = {"Ammonia": [0, 1], "Lysine": [2, 3, 4, 5], "H+": [6]}
    np.testing.assert_equal(ionic_system.indices, expected)


def test_n_comp(multispecies_component, ionic_system):
    assert multispecies_component.n_components == 2
    assert multispecies_component.n_comp == 3
    assert ionic_system.n_components == 3
    assert ionic_system.n_comp == 7


def test_charge(ionic_system):
    np.testing.assert_equal(ionic_system.charges, [1, 0, 2, 1, 0, -1, 1])


def test_molar_masses(components_with_physical_properties):
    np.testing.assert_equal(components_with_physical_properties.molar_masses, [1, 0])


def test_molecular_weights_deprecated(components_with_physical_properties):
    with pytest.warns(DeprecationWarning):
        ComponentSystem(["A"], molecular_weights=[1.0])
    with pytest.warns(DeprecationWarning):
        _ = components_with_physical_properties.molecular_weights


def test_densities(components_with_physical_properties):
    np.testing.assert_equal(components_with_physical_properties.densities, [1, 0])
