from dataclasses import dataclass, field

import numpy as np
import pytest
from CADETProcess.dataStructure import NdPolynomial, Structure
from CADETProcess.parameter_space import (
    CallableMapper,
    DotPathMapper,
    IndexedMapper,
    parse_path,
)

# ── parse_path ──────────────────────────────────────────────────────────────


def test_parse_path_single_segment():
    assert parse_path("length") == ("length",)


def test_parse_path_multiple_segments():
    assert parse_path("column.length") == ("column", "length")


def test_parse_path_three_segments():
    assert parse_path("unit_operations.feed.duration") == (
        "unit_operations",
        "feed",
        "duration",
    )


def test_parse_path_empty_raises():
    with pytest.raises(ValueError):
        parse_path("")


def test_parse_path_empty_segment_raises():
    with pytest.raises(ValueError):
        parse_path("a..b")


def test_parse_path_trailing_dot_raises():
    with pytest.raises(ValueError):
        parse_path("a.b.")


def test_parse_path_integer_index():
    assert parse_path("film_diffusion[2]") == ("film_diffusion", 2)


def test_parse_path_index_with_dot_prefix():
    assert parse_path("column.film_diffusion[2]") == ("column", "film_diffusion", 2)


def test_parse_path_slice():
    assert parse_path("film_diffusion[1:3]") == ("film_diffusion", slice(1, 3, None))


def test_parse_path_open_slice():
    assert parse_path("film_diffusion[:]") == ("film_diffusion", slice(None, None, None))


def test_parse_path_slice_with_step():
    assert parse_path("film_diffusion[::2]") == ("film_diffusion", slice(None, None, 2))


def test_parse_path_empty_bracket_name_raises():
    with pytest.raises(ValueError):
        parse_path("[2]")


def test_parse_path_malformed_bracket_raises():
    with pytest.raises(ValueError):
        parse_path("film_diffusion[2]extra")


# ── fixtures ─────────────────────────────────────────────────────────────────


@dataclass
class Leaf:
    """Simple object with a scalar attribute."""

    value: float = 0.0


@dataclass
class Container:
    """Object whose attribute is a dict."""

    store: dict


@pytest.fixture
def obj_with_attr():
    """Root object → nested object → attribute."""
    return Leaf(value=1.0)


@pytest.fixture
def obj_to_dict():
    """Root object → dict key."""
    return Container(store={"ka": 2e-5})


@pytest.fixture
def dict_to_obj():
    """Root dict → object attribute."""
    return {"column": Leaf(value=1e-5)}


@pytest.fixture
def nested_dict():
    """Deeply nested pure-dict structure."""
    return {
        "unit_1": {"params": {"k_a": 1e-5}},
        "unit_2": {"params": {"k_a": 2e-5}},
    }


# ── DotPathMapper ─────────────────────────────────────────────────────────────


def test_dotpath_sets_attribute(obj_with_attr):
    mapper = DotPathMapper([obj_with_attr], "value")
    mapper.set_value(42.0)
    assert obj_with_attr.value == 42.0


def test_dotpath_obj_to_dict(obj_to_dict):
    mapper = DotPathMapper([obj_to_dict], "store.ka")
    mapper.set_value(9.9e-5)
    assert obj_to_dict.store["ka"] == pytest.approx(9.9e-5)


def test_dotpath_dict_to_obj(dict_to_obj):
    mapper = DotPathMapper([dict_to_obj], "column.value")
    mapper.set_value(3.3e-4)
    assert dict_to_obj["column"].value == pytest.approx(3.3e-4)


def test_dotpath_nested_dict(nested_dict):
    mapper = DotPathMapper([nested_dict], "unit_1.params.k_a")
    mapper.set_value(5e-4)
    assert nested_dict["unit_1"]["params"]["k_a"] == pytest.approx(5e-4)
    assert nested_dict["unit_2"]["params"]["k_a"] == pytest.approx(2e-5)


def test_dotpath_broadcasts_to_multiple_objects():
    objs = [Leaf(1.0), Leaf(2.0), Leaf(3.0)]
    mapper = DotPathMapper(objs, "value")
    mapper.set_value(99.0)
    assert all(o.value == 99.0 for o in objs)


def test_dotpath_missing_intermediate_raises(dict_to_obj):
    mapper = DotPathMapper([dict_to_obj], "column.missing.leaf")
    with pytest.raises(AttributeError):
        mapper.set_value(0.0)


def test_dotpath_missing_dict_key_raises():
    d = {"a": {"b": 1}}
    mapper = DotPathMapper([d], "a.c.d")
    with pytest.raises(KeyError):
        mapper.set_value(0.0)


def test_dotpath_empty_path_raises():
    with pytest.raises(ValueError):
        DotPathMapper([object()], "")


# ── CallableMapper ────────────────────────────────────────────────────────────


def test_callable_mapper_invokes_fn():
    calls = []
    obj = object()
    mapper = CallableMapper([obj], fn=lambda o, v: calls.append((o, v)))
    mapper.set_value(7)
    assert calls == [(obj, 7)]


def test_callable_mapper_broadcasts():
    results = {}
    objs = [Leaf(0.0), Leaf(0.0)]
    mapper = CallableMapper(objs, fn=lambda o, v: results.__setitem__(id(o), v))
    mapper.set_value(5.5)
    assert all(v == 5.5 for v in results.values())
    assert len(results) == 2


# ── IndexedMapper ─────────────────────────────────────────────────────────────


@dataclass
class ArrayHolder:
    """Object with array-valued attributes."""

    data: list = field(default_factory=lambda: [1.0, 2.0, 3.0])
    matrix: list = field(default_factory=lambda: [[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def arr_obj():
    return ArrayHolder()


def test_indexed_explicit_index_sets_element(arr_obj):
    mapper = IndexedMapper([arr_obj], path="data", index=1)
    mapper.set_value(99.0)
    assert arr_obj.data[1] == pytest.approx(99.0)
    assert arr_obj.data[0] == pytest.approx(1.0)
    assert arr_obj.data[2] == pytest.approx(3.0)


def test_indexed_embedded_index_sets_element(arr_obj):
    mapper = IndexedMapper([arr_obj], path="data[2]")
    mapper.set_value(42.0)
    assert arr_obj.data[2] == pytest.approx(42.0)
    assert arr_obj.data[0] == pytest.approx(1.0)


def test_indexed_embedded_and_explicit_raises(arr_obj):
    with pytest.raises(ValueError, match="Cannot specify both"):
        IndexedMapper([arr_obj], path="data[1]", index=0)


def test_indexed_no_index_raises(arr_obj):
    with pytest.raises(ValueError, match="requires an index"):
        IndexedMapper([arr_obj], path="data")


def test_indexed_empty_path_raises(arr_obj):
    with pytest.raises(ValueError):
        IndexedMapper([arr_obj], path="[1]")


def test_indexed_slice_sets_range(arr_obj):
    mapper = IndexedMapper([arr_obj], path="data[1:3]")
    mapper.set_value([10.0, 20.0])
    assert arr_obj.data[1] == pytest.approx(10.0)
    assert arr_obj.data[2] == pytest.approx(20.0)


def test_indexed_explicit_slice(arr_obj):
    mapper = IndexedMapper([arr_obj], path="data", index=slice(0, 2))
    mapper.set_value([7.0, 8.0])
    assert arr_obj.data[0] == pytest.approx(7.0)
    assert arr_obj.data[1] == pytest.approx(8.0)


def test_indexed_list_written_back_as_list(arr_obj):
    mapper = IndexedMapper([arr_obj], path="data", index=0)
    mapper.set_value(5.0)
    assert isinstance(arr_obj.data, list)


def test_indexed_numpy_array_written_back_as_array():
    @dataclass
    class NpHolder:
        arr: np.ndarray = field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))

    obj = NpHolder()
    mapper = IndexedMapper([obj], path="arr", index=0)
    mapper.set_value(9.0)
    assert isinstance(obj.arr, np.ndarray)
    assert obj.arr[0] == pytest.approx(9.0)


def test_indexed_broadcasts_multiple_objects():
    objs = [ArrayHolder(), ArrayHolder()]
    mapper = IndexedMapper(objs, path="data", index=0)
    mapper.set_value(55.0)
    assert all(o.data[0] == pytest.approx(55.0) for o in objs)


def test_indexed_dotpath_to_array(arr_obj):
    """Traverse a dot path to reach the array, then index into it."""

    @dataclass
    class Nested:
        holder: ArrayHolder = field(default_factory=ArrayHolder)

    obj = Nested()
    mapper = IndexedMapper([obj], path="holder.data[0]")
    mapper.set_value(77.0)
    assert obj.holder.data[0] == pytest.approx(77.0)


def test_indexed_inhomogeneous_raises(arr_obj):
    @dataclass
    class IrregHolder:
        data: list = field(default_factory=lambda: [[1.0, 2.0], [3.0]])

    obj = IrregHolder()
    mapper = IndexedMapper([obj], path="data", index=0)
    with pytest.raises(NotImplementedError):
        mapper.set_value(0.0)


def test_dotpath_rejects_indexed_path(arr_obj):
    with pytest.raises(ValueError, match="ends with an array index"):
        DotPathMapper([arr_obj], "data[1]")


# ── IndexedMapper: multidimensional / polynomial ──────────────────────────────


@dataclass
class TwoDHolder:
    """Object with a uniform 2-D array attribute (e.g. reaction exponents)."""

    exponents: list = field(
        default_factory=lambda: [[1.0, 1.0], [1.0, 1.0]]
    )


@dataclass
class PolyHolder:
    """Object with a polynomial (1-D) parameter stored as a flat array."""

    flow_rate: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


@dataclass
class PolyConcHolder:
    """Object with a 2-D polynomial parameter (n_comp × n_coeff)."""

    c: list = field(
        default_factory=lambda: [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )


def test_indexed_2d_integer_index_sets_row():
    """Integer index on a 2-D array sets the whole row."""
    obj = TwoDHolder()
    mapper = IndexedMapper([obj], path="exponents", index=0)
    mapper.set_value([9.0, 9.0])
    assert obj.exponents[0] == pytest.approx([9.0, 9.0])
    assert obj.exponents[1] == pytest.approx([1.0, 1.0])


def test_indexed_2d_tuple_index_sets_element():
    """Tuple index (row, col) sets a single element of a 2-D array."""
    obj = TwoDHolder()
    mapper = IndexedMapper([obj], path="exponents", index=(0, 1))
    mapper.set_value(5.0)
    assert obj.exponents[0][1] == pytest.approx(5.0)
    assert obj.exponents[0][0] == pytest.approx(1.0)
    assert obj.exponents[1] == pytest.approx([1.0, 1.0])


def test_indexed_2d_numpy_slice_sets_row():
    """np.s_ slice sets an entire row of a 2-D array."""
    obj = TwoDHolder()
    mapper = IndexedMapper([obj], path="exponents", index=np.s_[0, :])
    mapper.set_value(3.0)
    assert obj.exponents[0] == pytest.approx([3.0, 3.0])
    assert obj.exponents[1] == pytest.approx([1.0, 1.0])


def test_indexed_polynomial_constant_coefficient():
    """Index 0 into a polynomial array sets the constant term."""
    obj = PolyHolder()
    mapper = IndexedMapper([obj], path="flow_rate", index=0)
    mapper.set_value(1.0)
    assert obj.flow_rate[0] == pytest.approx(1.0)
    assert obj.flow_rate[1:] == pytest.approx([0.0, 0.0, 0.0])


def test_indexed_polynomial_linear_coefficient():
    """Index 1 into a polynomial array sets the linear term."""
    obj = PolyHolder()
    mapper = IndexedMapper([obj], path="flow_rate", index=1)
    mapper.set_value(2.0)
    assert obj.flow_rate[0] == pytest.approx(0.0)
    assert obj.flow_rate[1] == pytest.approx(2.0)
    assert obj.flow_rate[2:] == pytest.approx([0.0, 0.0])


def test_indexed_2d_polynomial_tuple_index():
    """Tuple index into a 2-D polynomial array sets a single coefficient."""
    obj = PolyConcHolder()
    mapper = IndexedMapper([obj], path="c", index=(0, 1))
    mapper.set_value(3.0)
    assert obj.c[0][1] == pytest.approx(3.0)
    assert obj.c[0][0] == pytest.approx(0.0)
    assert obj.c[1] == pytest.approx([0.0, 0.0, 0.0, 0.0])


class NdPolynomialHolder(Structure):
    """Object with a genuine ``NdPolynomial`` descriptor (n_comp x n_coeff).

    Unlike ``PolyHolder``/``PolyConcHolder`` above (plain lists that only look
    polynomial by name), this fixture exercises the real descriptor and its
    ``fill_values`` method, which is what ``IndexedMapper`` must detect and
    delegate to for a bare index.
    """

    c = NdPolynomial(size=(2, 4), default=0)

    _parameters = ["c"]


def test_indexed_bare_index_into_real_polynomial_descriptor_fills_row():
    """A bare int index selecting a whole row delegates to the descriptor's
    ``fill_values``: constant coefficient set, rest of the row zeroed."""
    obj = NdPolynomialHolder()
    mapper = IndexedMapper([obj], path="c", index=0)
    mapper.set_value(2.0)
    assert obj.c[0] == pytest.approx([2.0, 0.0, 0.0, 0.0])
    assert obj.c[1] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_indexed_tuple_index_into_real_polynomial_descriptor_sets_single_cell():
    """A tuple index fully specifies one cell, so it is a plain scalar write
    regardless of the descriptor being polynomial."""
    obj = NdPolynomialHolder()
    mapper = IndexedMapper([obj], path="c", index=(0, 1))
    mapper.set_value(3.0)
    assert obj.c[0] == pytest.approx([0.0, 3.0, 0.0, 0.0])
    assert obj.c[1] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_indexed_2d_list_written_back_as_list():
    """A list-backed 2-D array is written back as a list, not ndarray."""
    obj = TwoDHolder()
    mapper = IndexedMapper([obj], path="exponents", index=(0, 0))
    mapper.set_value(7.0)
    assert isinstance(obj.exponents, list)
    assert isinstance(obj.exponents[0], list)


# ── get_value round-trips ─────────────────────────────────────────────────────


def test_dotpath_get_value_after_set(obj_with_attr):
    mapper = DotPathMapper([obj_with_attr], "value")
    mapper.set_value(42.0)
    assert mapper.get_value() == pytest.approx(42.0)


def test_dotpath_get_value_dict_key(obj_to_dict):
    mapper = DotPathMapper([obj_to_dict], "store.ka")
    mapper.set_value(1.23e-4)
    assert mapper.get_value() == pytest.approx(1.23e-4)


def test_dotpath_get_value_reads_first_object_only():
    objs = [Leaf(1.0), Leaf(2.0)]
    mapper = DotPathMapper(objs, "value")
    mapper.set_value(99.0)
    # get_value reads from the first object only
    assert mapper.get_value() == pytest.approx(99.0)
    assert objs[0].value == pytest.approx(99.0)
    assert objs[1].value == pytest.approx(99.0)


def test_indexed_get_value_scalar_index(arr_obj):
    mapper = IndexedMapper([arr_obj], path="data", index=1)
    mapper.set_value(77.0)
    assert mapper.get_value() == pytest.approx(77.0)


def test_indexed_get_value_slice(arr_obj):
    mapper = IndexedMapper([arr_obj], path="data[0:2]")
    mapper.set_value([5.0, 6.0])
    result = mapper.get_value()
    assert np.allclose(result, [5.0, 6.0])


def test_indexed_get_value_tuple_index():
    obj = TwoDHolder()
    mapper = IndexedMapper([obj], path="exponents", index=(1, 0))
    mapper.set_value(3.5)
    assert mapper.get_value() == pytest.approx(3.5)


def test_callable_mapper_get_value_returns_none():
    obj = object()
    mapper = CallableMapper([obj], fn=lambda o, v: None)
    assert mapper.get_value() is None
